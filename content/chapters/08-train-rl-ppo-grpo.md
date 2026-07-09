# 第 8 章 训练算法 - PPO & GRPO 强化学习

本章介绍 MiniMind 中两种主流的 RLHF（Reinforcement Learning from Human Feedback）训练算法：PPO（Proximal Policy Optimization）和 GRPO（Group Relative Policy Optimization）。两者都以 SFT 模型为起点，通过 rollout 采样、奖励打分、优势估计、策略更新这一闭环不断提升模型回答质量，但在「价值函数」和「优势估计」上采用了完全不同的方案。

---

## 8.1 RLHF 整体流程

### 原理说明

无论是 PPO 还是 GRPO，MiniMind 的强化学习训练都遵循同一条主循环：

1. **Rollout 采样**：用当前 actor（策略模型）对一批 prompt 采样生成回答，并记录每个生成 token 在采样时刻的对数概率 `old_logp`。
2. **Reward 打分**：用规则奖励（长度/thinking 格式/重复惩罚）+ 外部 reward model 给每条回答打分。
3. **优势估计**：PPO 用 Critic + GAE 把标量奖励分配到每个 token；GRPO 用同一 prompt 的多条采样做组内标准化。
4. **策略更新**：构造 clipped surrogate loss（PPO）或 CISPO/GRPO loss，再用 reference model 的 KL 散度约束 actor 不要偏离 SFT 太远。

四个角色协同工作：

| 角色 | PPO 中存在 | GRPO 中存在 | 作用 |
|------|-----------|------------|------|
| Actor / Policy | ✓ | ✓ | 待训练的生成模型 |
| Reference Model | ✓ | ✓ | 冻结的 SFT 模型，提供 KL 约束 |
| Critic (Value Model) | ✓ | ✗ | 估计每个状态的价值，PPO 专用 |
| Reward Model | ✓ | ✓ | 外部偏好打分模型 |

PPO 主循环代码：[trainer/train_ppo.py:82](file:///home/zhangxm/model_minimind/trainer/train_ppo.py#L82-L308)  
GRPO 主循环代码：[trainer/train_grpo.py:73](file:///home/zhangxm/model_minimind/trainer/train_grpo.py#L73-L210)

---

## 8.2 Rollout 引擎：采样与 per-token logprob

### 原理说明

Rollout 阶段要完成两件事：用当前 actor 生成回答，并同时记录每个生成 token 在「采样时刻」策略下的对数概率 `old_logp`。`old_logp` 是后续策略梯度的基准——更新后的新策略在同一批 token 上的 logp 与它做比值得到 importance ratio。

MiniMind 把 rollout 抽象成一个可插拔引擎，提供两种实现：

- **TorchRolloutEngine**：直接用 PyTorch 模型的 `.generate()` 采样，再调用 `compute_per_token_logps` 重算 logp。
- **SGLangRolloutEngine**：通过 HTTP 调用外部 SGLang 服务，由服务端返回 token ids 和 logprob，适合大规模加速。

### compute_per_token_logps

每个 token 的 logprob 计算：对完整序列做一次前向得到 logits，取 `[:, :-1, :]`（每个位置预测下一个 token），再用 `log_softmax + gather` 取出「真实下一个 token」的对数概率。

- 代码链接：[trainer/rollout_engine.py:24](file:///home/zhangxm/model_minimind/trainer/rollout_engine.py#L24-L36)

```python
def compute_per_token_logps(model, input_ids, n_keep, attention_mask=None):
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    # 只保留最后 n_keep 个生成 token 的概率；RL loss 只关心 completion，不训练 prompt。
    logits = unwrapped(input_ids, attention_mask=attention_mask,
                       logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)
```

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| input_ids | [B, P+R] | long | prompt + completion 拼接序列 |
| logits (切片后) | [B, R, vocab] | float | 每个位置预测下一个 token 的 logits |
| per_token_logps | [B, R] | float32 | 每个生成 token 在采样策略下的 logπ_old |

### TorchRolloutEngine vs SGLangRolloutEngine

- 代码链接：[trainer/rollout_engine.py:66](file:///home/zhangxm/model_minimind/trainer/rollout_engine.py#L66-L95)（Torch）  
- 代码链接：[trainer/rollout_engine.py:103](file:///home/zhangxm/model_minimind/trainer/rollout_engine.py#L103-L175)（SGLang）

**TorchRolloutEngine.rollout 关键逻辑**：

```python
output_ids = model.generate(
    input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
    attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
    max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, ...
)  # [B*num_gen, P+R]
completion_ids = output_ids[:, prompt_len:]  # [B*num_gen, R]
per_token_logps = compute_per_token_logps(self.policy_model, output_ids,
                                          completion_ids.size(1), attention_mask=full_mask)
```

`repeat_interleave(num_generations, dim=0)` 把每个 prompt 复制 `num_gen` 份，让同一 prompt 采样出多条不同回答——这是 GRPO 组内比较的来源；PPO 中 `num_generations=1`。

**SGLangRolloutEngine** 通过 HTTP `/generate` 接口提交 `input_ids`，服务端用 vLLM/SGLang 的高性能推理返回 `output_ids` 和 `output_token_logprobs`。`update_policy` 会把当前 actor 权重写到共享目录，再调用 `/update_weights_from_disk` 让服务端热加载新权重。

---

## 8.3 calculate_rewards：奖励模型打分 + 规则奖励

### 原理说明

`calculate_rewards` 把外部 reward model 的语义偏好分数和人工规则奖励拼到一起，输出一个**回答级别**的标量奖励 `[B]`（GRPO 中为 `[B*num_gen]`）。规则部分约束长度、thinking 格式和重复；reward model 部分约束语义质量。

- PPO 版本：[trainer/train_ppo.py:54](file:///home/zhangxm/model_minimind/trainer/train_ppo.py#L54-L80)
- GRPO 版本：[trainer/train_grpo.py:38](file:///home/zhangxm/model_minimind/trainer/train_grpo.py#L38-L71)

两者的差异只在「外层循环结构」（GRPO 多一层 `for j in range(num_generations)`），核心打分逻辑一致：

```python
rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5      # 长度规则
if '</think>' in response:                                              # thinking 格式
    thinking_content, answer_content = response.split('</think>', 1)
    rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
    rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25
rewards[i] -= rep_penalty(answer)                                       # n-gram 重复惩罚
score = reward_model.get_score(messages, answer)                        # 外部 RM 打分
rewards += reward_model_scores
```

### Reward Model 包装

外部 reward model 通过 `LMForRewardModel` 封装，调用其 `get_score(messages, response)` 返回标量分数并裁剪到 `[-3, 3]`：

- 代码链接：[trainer/trainer_utils.py:170](file:///home/zhangxm/model_minimind/trainer/trainer_utils.py#L170-L190)

```python
score = self.model.get_score(self.tokenizer, eval_messages)
return max(min(score, 3.0), -3.0)
```

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| prompts | list[str], len=B | - | 输入对话上下文 |
| responses | list[str], len=B (或 B*num_gen) | - | actor 生成的回答 |
| rewards | [B]（或 [B*num_gen]） | float | 每条回答的标量奖励 |

---

## 8.4 PPO：CriticModel + GAE + Clipped Surrogate

PPO 是 RLHF 的「经典款」，需要训练一个 Critic 估计每个状态的价值，再用 GAE（Generalized Advantage Estimation）把回答级奖励分配到每个 token。

### 8.4.1 CriticModel 价值函数

### 原理说明

Critic 复用 MiniMind 主干提取隐藏状态，只把 `lm_head` 换成 `value_head: Linear(hidden_size, 1)`，输出每个 token 位置的价值估计 V(s_t)——即从当前位置到回答结束的期望累计奖励。

- 代码链接：[trainer/train_ppo.py:38](file:///home/zhangxm/model_minimind/trainer/train_ppo.py#L38-L51)

```python
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        # 替换 lm_head 为输出单一价值的线性层
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        values = self.value_head(hidden_states).squeeze(-1)
        return values
```

Critic 从与 actor 相同的 SFT 权重初始化（`load_state_dict(state_dict, strict=False)`），只训练 `value_head` 和主干中的差异部分。

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| input_ids | [B, P+R] | long | prompt + completion |
| values | [B, P+R] | float | 每个 token 位置的价值估计 V(s_t) |

### 8.4.2 奖励分配 + GAE 优势估计

### 原理说明

外部 reward 是「回答级别」标量，PPO 把它放到回答最后一个 token 上，再用 GAE 从后往前反向递推，得到每个 token 的优势 A_t 和回报 R_t。

GAE 公式：

```
δ_t   = r_t + γ * V(s_{t+1}) - V(s_t)              # TD 误差
A_t   = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}               # 广义优势
R_t   = A_t + V(s_t)                                # Critic 的回归目标
```

`γ` 是折扣因子（MiniMind 默认 1.0，因为回答长度有限），`λ` 是 GAE 平滑参数（默认 0.95）。

- 代码链接：[trainer/train_ppo.py:144](file:///home/zhangxm/model_minimind/trainer/train_ppo.py#L144-L161)

### 关键计算逻辑逐行解释

```python
# 1. 把外部标量奖励放到回答最后一个有效 token 上
token_rewards = torch.zeros_like(old_resp_logp)              # [B, R]
last_idx = resp_lengths - 1                                  # [B]
token_rewards[torch.arange(B)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]

# 2. GAE 反向递推
gen_len = old_resp_values.size(1)                            # R
lastgaelam = torch.zeros(B, device=args.device)
advs_rev = []
for t in reversed(range(gen_len)):
    nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0     # V(s_{t+1})，末尾为 0
    delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]   # δ_t
    lastgaelam = delta + args.gamma * args.lam * lastgaelam               # A_t 累加
    advs_rev.append(lastgaelam)
advantages = torch.stack(advs_rev[::-1], dim=1)              # [B, R]
returns = advantages + old_resp_values                       # [B, R]，Critic 回归目标

# 3. 优势标准化（用 mask 排除 padding/EOS 后的 token）
adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
adv_var  = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask
```

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| token_rewards | [B, R] | float | 每个 token 的即时奖励（仅末位非 0） |
| old_resp_values | [B, R] | float | Critic 在 rollout 时刻的 V(s_t) |
| delta | [B, R] | float | TD 误差 δ_t |
| advantages | [B, R] | float | GAE 优势 A_t（标准化后） |
| returns | [B, R] | float | Critic 回归目标 R_t = A_t + V(s_t) |

### 8.4.3 PPO 策略损失（Clipped Surrogate）

### 原理说明

PPO 用 importance ratio `r_t = exp(logπ_new - logπ_old)` 衡量新旧策略差异，并把它裁剪到 `[1-ε, 1+ε]` 防止单步更新过猛：

```
L_policy = -E[ min(r_t * A_t,  clip(r_t, 1-ε, 1+ε) * A_t) ]
```

MiniMind 用 `torch.max(-A*r, -A*clip(r))` 取负号实现最小化，并叠加 reference KL 惩罚防止偏离 SFT。

- 代码链接：[trainer/train_ppo.py:190](file:///home/zhangxm/model_minimind/trainer/train_ppo.py#L190-L213)

```python
log_ratio = mb_resp_logp - old_resp_logp[inds]               # log r_t
approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
# approx_kl 过大时提前结束本轮 PPO 更新
if approx_kl_val > args.early_stop_kl:  stop_ppo = True

ratio = torch.exp(log_ratio)                                  # r_t
# reference KL 惩罚（k3 估计器，见 8.6）
kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_resp_logp)
                   - (ref_resp_logp[inds] - mb_resp_logp) - 1.0)
                  * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)

# Actor loss: clipped surrogate + reference KL 惩罚
policy_loss = ((torch.max(-advantages[inds] * ratio,
                          -advantages[inds] * torch.clamp(ratio, 1.0 - args.clip_epsilon,
                                                          1.0 + args.clip_epsilon))
                * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
               + args.kl_coef * kl_ref_penalty)
```

关键超参（默认值）：`clip_epsilon=0.2`、`kl_coef=0.02`、`early_stop_kl=0.25`、`ppo_update_iters=2`。

### 8.4.4 PPO 价值损失（Clipped Value Loss）

Critic 也做 value clipping，避免 value head 一次更新过大：

- 代码链接：[trainer/train_ppo.py:213](file:///home/zhangxm/model_minimind/trainer/train_ppo.py#L213-L217)

```python
value_loss = 0.5 * (torch.max((mb_resp_values - returns[inds]) ** 2,
                              (torch.clamp(mb_resp_values,
                                           old_resp_values[inds] - args.cliprange_value,
                                           old_resp_values[inds] + args.cliprange_value) - returns[inds]) ** 2)
                    * resp_value_mask[inds]).sum() / resp_value_mask[inds].sum().clamp(min=1)
```

总 loss：`loss = policy_loss + vf_coef * value_loss + aux_loss`（`vf_coef=0.5`，`aux_loss` 仅 MoE 时存在）。

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| mb_resp_logp | [mb, R] | float | 新策略下 token 的 logπ_new |
| old_resp_logp | [mb, R] | float | rollout 时旧策略的 logπ_old |
| ratio | [mb, R] | float | importance ratio r_t |
| policy_loss | 标量 | float | Actor 损失 |
| value_loss | 标量 | float | Critic 损失 |

---

## 8.5 GRPO：组内基线 + CISPO

### 原理说明

GRPO 的核心洞察：**不需要 Critic**。对同一 prompt 采样 `num_generations` 条回答，组内奖励的均值和方差就是天然的基线。优势标准化公式：

```
A_i = (r_i - mean(r_group)) / (std(r_group) + ε)
```

这等价于把 Critic 替换成「组内均值基线」，省去了一整个价值网络的训练开销。MiniMind 默认 `num_generations=6`。

### 8.5.1 组内优势标准化

- 代码链接：[trainer/train_grpo.py:127](file:///home/zhangxm/model_minimind/trainer/train_grpo.py#L127-L131)

```python
grouped_rewards = rewards.view(-1, args.num_generations)                    # [B, num_gen]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
std_r  = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
# GRPO 不训练 critic，而是在同一 prompt 的多条回答内部做相对优势标准化。
advantages = (rewards - mean_r) / (std_r + 1e-4)                            # [B*num_gen]
```

注意 GRPO 的优势是**回答级别**的标量 `[B*num_gen]`，会通过 `advantages.unsqueeze(1)` 广播到每个 token；而 PPO 的优势是 token 级别 `[B, R]`（由 GAE 展开）。

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| rewards | [B*num_gen] | float | 每条回答的标量奖励 |
| grouped_rewards | [B, num_gen] | float | 按 prompt 分组的奖励 |
| mean_r / std_r | [B*num_gen] | float | 组内均值/标准差（广播回原 shape） |
| advantages | [B*num_gen] | float | 组内标准化优势 |

### 8.5.2 CISPO 损失（默认）

### 原理说明

GRPO 中每条样本只采样一次，没有 PPO 那种「同批数据多 epoch 更新」的需求，但仍然需要约束新策略不要偏离旧策略太远。MiniMind 提供两种 loss：

- **CISPO（默认）**：不裁剪 ratio，而是把 ratio 上界裁剪后作为**加权系数**，对 `logπ_new` 直接做策略梯度。其优势是即便某 token 的 ratio 很大，CISPO 也不会像 PPO 那样把它「截断到 0」，而是限制它的贡献度，训练更稳定。
- **GRPO（标准 PPO 风格）**：用 `min(r*A, clip(r)*A)` 做 clipped surrogate，和 PPO 完全一致。

- 代码链接：[trainer/train_grpo.py:139](file:///home/zhangxm/model_minimind/trainer/train_grpo.py#L139-L153)

```python
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1                       # k3 KL 估计器，见 8.6
ratio = torch.exp(per_token_logps - old_per_token_logps)            # [B*num_gen, R]

if args.loss_type == "cispo":
    # CISPO 用上界裁剪 ratio，降低极端概率比对训练稳定性的影响。
    clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
    per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps
                       - args.beta * per_token_kl)
else:
    # 标准 GRPO/PPO 风格裁剪：限制新旧策略概率比，避免一步更新过猛。
    clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
    per_token_loss1 = ratio * advantages.unsqueeze(1)
    per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
    per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)

policy_loss = ((per_token_loss * completion_mask).sum(dim=1)
               / completion_mask.sum(dim=1).clamp(min=1)).mean()
loss = (policy_loss + aux_loss) / args.accumulation_steps
```

CISPO 与标准 GRPO 的关键差异：

| 方面 | CISPO | 标准 GRPO |
|------|-------|----------|
| ratio 处理 | `clamp(ratio, max=ε_high).detach()` 作为系数 | `min(ratio*A, clip(ratio)*A)` |
| 梯度来源 | `per_token_logps`（直接对 logπ 求导） | `ratio`（对 logπ_new - logπ_old 求导） |
| 截断行为 | 极端 ratio 仍贡献梯度，只是权重受限 | 极端 ratio 被完全截断 |
| 默认参数 | `epsilon_high=5.0`, `beta=0.1` | `epsilon=0.2`, `beta=0.1` |

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| per_token_logps | [B*num_gen, R] | float | 新策略下每个 token 的 logπ_new |
| old_per_token_logps | [B*num_gen, R] | float | rollout 时旧策略的 logπ_old |
| ref_per_token_logps | [B*num_gen, R] | float | reference model 的 logπ_ref |
| ratio | [B*num_gen, R] | float | importance ratio |
| per_token_kl | [B*num_gen, R] | float | k3 KL 估计（每 token） |
| per_token_loss | [B*num_gen, R] | float | 每 token 损失 |
| policy_loss | 标量 | float | 平均后的策略损失 |

---

## 8.6 k3 KL 估计器（Schulman 近似）

### 原理说明

直接用 `KL(p‖q) = E_p[log p - log q]` 作为损失项会导致梯度在 q 概率为 0 时不稳定。Schulman 提出的 **k3 估计器**是一个无偏的低方差近似：

```
k3 = exp(log p - log q) - (log p - log q) - 1
```

其中 `log p` 是 reference（固定），`log q` 是当前 actor。当 p ≈ q 时 k3 ≈ 0；当 actor 偏离 reference 时 k3 > 0 且光滑可导。

### 代码位置

PPO 中的 reference KL 惩罚（注意符号约定：ref - new）：

- 代码链接：[trainer/train_ppo.py:206](file:///home/zhangxm/model_minimind/trainer/train_ppo.py#L206-L207)

```python
kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_resp_logp)
                   - (ref_resp_logp[inds] - mb_resp_logp) - 1.0)
                  * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
```

GRPO 中的 per-token KL（同样的 k3 公式）：

- 代码链接：[trainer/train_grpo.py:139](file:///home/zhangxm/model_minimind/trainer/train_grpo.py#L139-L140)

```python
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1   # [B*num_gen, R]
```

PPO 中 k3 用于**回答级** KL 惩罚（`kl_coef=0.02`），GRPO 中用于**每 token** KL 惩罚（`beta=0.1`）。

---

## 8.7 PPO vs GRPO 对比

| 维度 | PPO | GRPO |
|------|-----|------|
| 价值函数 | CriticModel（trainable） | 无 |
| 优势估计 | GAE：δ_t + γλ 递推 | 组内均值/方差标准化 |
| 优势 shape | [B, R]（token 级） | [B*num_gen]（回答级，广播） |
| 采样数 | num_generations=1 | num_generations=6 |
| 损失 | clipped surrogate + value loss | CISPO 或 clipped surrogate |
| KL 惩罚 | k3，回答级，kl_coef=0.02 | k3，每 token，beta=0.1 |
| 显存开销 | 高（多一个 Critic） | 低 |
| 训练稳定性 | Critic 提供平滑基线 | 依赖组内方差，组内 reward 全相同时退化 |

---

## 小结

本章拆解了 MiniMind 的两种 RLHF 算法：

1. **Rollout 引擎**抽象了采样与 logprob 计算，支持 PyTorch 原生与 SGLang 两种后端，是 PPO/GRPO 共享的基础设施。
2. **calculate_rewards** 把规则奖励（长度/thinking/重复）与外部 reward model 分数拼接为回答级标量奖励。
3. **PPO** 用 CriticModel 估计价值，GAE 反向递推得到 token 级优势，再用 clipped surrogate + clipped value loss 更新。
4. **GRPO** 省去 Critic，用同一 prompt 的多条采样做组内标准化得到回答级优势，默认使用 CISPO 损失（裁剪 ratio 上界作为加权系数而非截断梯度）。
5. **k3 KL 估计器** `exp(Δ) - Δ - 1` 提供光滑无偏的 reference KL 约束，防止 actor 偏离 SFT 模型。

理解 PPO 与 GRPO 的取舍——稳定性 vs 显存效率——是选择 RLHF 算法的关键。下一章我们将介绍 LoRA 低秩适配，它在参数高效微调场景下能让 PPO/GRPO 的训练成本进一步大幅降低。
