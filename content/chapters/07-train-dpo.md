# 第 7 章 训练算法 - DPO 偏好优化

本章介绍 MiniMind 的偏好对齐阶段：Direct Preference Optimization（DPO，直接偏好优化）。SFT 让模型学会聊天格式，但模型还不知道“哪种回答更受人喜欢”。DPO 利用人类标注的 chosen/rejected 偏好对，无需训练独立的奖励模型，直接用偏好数据微调策略模型，使其对 chosen 回复的概率高于 rejected。

---

## 7.1 DPO 原理：从 RLHF 到直接偏好优化

### 原理说明

经典的 RLHF（Reinforcement Learning from Human Feedback）分三步：训练奖励模型 → 用 PPO 等强化学习算法优化策略 → 加 KL 约束防止偏离参考模型。流程复杂、训练不稳定、需要在线采样。

DPO 的核心洞察是：**最优策略可以从偏好数据中直接推导出来，无需显式训练奖励模型或在线 RL**。

#### 推导：从偏好到隐式奖励

RLHF 假设偏好服从 Bradley-Terry 模型：人类偏好 chosen（y_w）胜过 rejected（y_l）的概率为：

```
P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))
```

其中 `r(x, y)` 是奖励函数，`σ` 是 sigmoid。DPO 进一步利用 RLHF 的最优解形式——在 KL 约束下最优策略对应的奖励可写为：

```
r(x, y) = β · (log π_θ(y|x) - log π_ref(y|x)) + β·log Z(x)
```

其中 `π_θ` 是策略模型，`π_ref` 是冻结的参考模型，`β` 是 KL 强度，`Z(x)` 是只与 prompt 有关的配分项。把它代入 Bradley-Terry 模型，`Z(x)` 在 chosen/rejected 相减时抵消，得到：

```
P(y_w ≻ y_l | x) = σ( β·[ (log π_θ(y_w|x) - log π_ref(y_w|x))
                       - (log π_θ(y_l|x) - log π_ref(y_l|x)) ] )
```

这就是 DPO 的核心公式。**隐式奖励** `r̂(x,y) = β·(log π_θ(y|x) - log π_ref(y|x))` 直接由两个对数概率之差表示，无需单独训练奖励模型。

#### DPO 损失

最大化偏好对数似然，等价于最小化：

```
L_DPO = -E[ log σ( β · [ (log π_θ(y_w|x) - log π_ref(y_w|x))
                       - (log π_θ(y_l|x) - log π_ref(y_l|x)) ] ) ]
```

直觉：策略模型相对参考模型，应该在 chosen 上概率上升、在 rejected 上概率下降。`β` 越大，约束越强（越贴近参考模型）；`β` 越小，越激进地调整偏好。MiniMind 默认 `beta=0.15`。

---

## 7.2 DPODataset：chosen/rejected 偏好对

### 原理说明

DPO 数据每条是一个偏好对：同一个 prompt 下，`chosen` 是更好的回答，`rejected` 是更差的回答。两者都是完整的多轮对话（含 prompt + response），分别用 chat 模板拼成文本后 tokenize。

与 SFT 类似，DPO 只比较 **assistant 回复部分** 的概率，prompt/user/system 部分不参与偏好比较——因为 chosen 和 rejected 的 prompt 完全相同，比较它们的概率没有意义。因此需要 `generate_loss_mask` 标记哪些 token 属于 assistant 回复。

### 代码位置引用

- 类定义：[dataset/lm_dataset.py:132-187](file:///home/zhangxm/model_minimind/dataset/lm_dataset.py#L132-L187)
- `generate_loss_mask`：[dataset/lm_dataset.py:189-208](file:///home/zhangxm/model_minimind/dataset/lm_dataset.py#L189-L208)

```python
class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        ...
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']
        rejected = sample['rejected']
        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        chosen_prompt = post_processing_chat(chosen_prompt)
        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
        rejected_prompt = post_processing_chat(rejected_prompt)

        chosen_encoding = self.tokenizer(chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length')
        rejected_encoding = self.tokenizer(rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length')

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
        ...
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        ...
        return {'x_chosen': ..., 'y_chosen': ..., 'mask_chosen': ...,
                'x_rejected': ..., 'y_rejected': ..., 'mask_rejected': ...}
```

### 输入/输出张量说明

每条样本返回 6 个张量（chosen 和 rejected 各 3 个）：

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| x_chosen / x_rejected | [seq_len-1] | long | 输入序列 = input_ids[:-1] |
| y_chosen / y_rejected | [seq_len-1] | long | 预测目标 = input_ids[1:]（右移一位） |
| mask_chosen / mask_rejected | [seq_len-1] | long | 1 表示该位置属于 assistant 回复（参与概率求和），0 表示 prompt 部分 |

经 DataLoader 堆叠后 batch 维变为 `[batch_size, seq_len-1]`。

### 关键计算逻辑解释

**步骤 1：分别构造 chosen/rejected 文本**

两者都用 `apply_chat_template` 拼成完整对话文本。它们的 prompt 部分完全相同，只有 assistant 最后一段回复不同（chosen 质量高于 rejected）。

**步骤 2：右移一位构造 X/Y**

注意这里与 Pretrain/SFT 不同——DPO 在 **Dataset 内部** 就做了位移：

```python
x_chosen = chosen_input_ids[:-1]   # 输入
y_chosen = chosen_input_ids[1:]    # 目标（next token）
mask_chosen = chosen_loss_mask[1:] # mask 也同步右移，对齐到预测位置
```

因为 DPO 训练时不再调用 `model(input_ids, labels=labels)`（那会触发内部 loss 计算），而是直接取 `model(x).logits` 自行处理，所以位移必须显式完成。mask 右移是为了与 `y` 对齐：位置 t 的 logits 预测的是 t+1 位置的 token，mask 应标记被预测的 token（即 t+1）是否属于 assistant。

**步骤 3：generate_loss_mask 标记 assistant 区间**

- 代码链接：[dataset/lm_dataset.py:189-208](file:///home/zhangxm/model_minimind/dataset/lm_dataset.py#L189-L208)

逻辑与 SFT 的 `generate_labels` 几乎一致：滑动匹配 `bos_id` 定位 assistant 起点，扫描到 `eos_id`，把这一段（含 EOS）的 mask 置 1，其余为 0。区别只是把“写入 token id”改成“写入 1”。

效果：chosen 和 rejected 的 prompt 部分概率不被计入偏好比较，只有 assistant 回复部分的 log 概率之差参与 DPO loss。

---

## 7.3 logits_to_log_probs：从 logits 到序列对数概率

### 原理说明

DPO 公式需要 `log π_θ(y|x)`，即策略模型对完整 response 序列的对数概率。它等于 response 中每个 token 对数概率之和：

```
log π(y|x) = Σ_t log π(y_t | x, y_<t)
```

实现上：模型前向输出每个位置的 logits，经 log_softmax 得到该位置对全词表的 log 概率分布，再用 `gather` 按“真实下一个 token”的 id 取出对应的 log 概率，最后按 mask 求和（只累加 assistant 部分）。

### 代码位置引用

- 代码链接：[trainer/train_dpo.py:25-32](file:///home/zhangxm/model_minimind/trainer/train_dpo.py#L25-L32)

```python
def logits_to_log_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # log_probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token
```

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| logits | [batch, seq_len, vocab_size] | bf16/fp16 | 模型输出 logits（注意 x 已右移，seq_len = max_length-1） |
| labels (y) | [batch, seq_len] | long | 真实下一个 token 的 id（即 input_ids[1:]） |
| log_probs_per_token | [batch, seq_len] | float32 | 每个位置真实 token 的 log 概率 |

### 关键计算逻辑解释

**步骤 1：log_softmax**

```python
log_probs = F.log_softmax(logits, dim=2)   # [batch, seq, vocab]
```

对最后一维（vocab）做 log_softmax，得到每个位置对全词表的 log 概率分布。

**步骤 2：gather 取目标 token 的 log 概率**

```python
log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
```

- `labels.unsqueeze(2)`：`[batch, seq] → [batch, seq, 1]`，作为 gather 的索引。
- `torch.gather(..., dim=2, index=...)`：在 vocab 维上按 labels 取出对应位置的值，结果 `[batch, seq, 1]`。
- `.squeeze(-1)`：去掉最后一维，得到 `[batch, seq]`，即每个位置真实 token 的 log 概率。

**步骤 3：按 mask 求和（在 dpo_loss 中完成）**

`log_probs_per_token` 是逐 token 的 log 概率，还需乘以 mask 后在 seq 维求和才得到 response 的总 log 概率（见 7.4）。

---

## 7.4 dpo_loss：偏好损失计算

### 原理说明

`dpo_loss` 实现 7.1 推导的 DPO 损失公式。它接收 ref 模型和 policy 模型各自的逐 token log 概率，按 mask 求和得到 response 总 log 概率，再分别取 chosen/rejected 部分构造 `logits = (Δlogπ_chosen - Δlogπ_rejected)`，最后套 `-log σ(β·logits)`。

为节省算力，chosen 和 rejected 在 batch 维拼接后**一次前向**完成，前半 batch 是 chosen、后半是 rejected。

### 代码位置引用

- 代码链接：[trainer/train_dpo.py:35-53](file:///home/zhangxm/model_minimind/trainer/train_dpo.py#L35-L53)

```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

### 输入/输出张量说明

| 张量 | Shape | 含义 |
|------|-------|------|
| ref_log_probs | [batch, seq] | ref 模型逐 token log 概率（batch = 2N，前 N 为 chosen，后 N 为 rejected） |
| policy_log_probs | [batch, seq] | policy 模型逐 token log 概率 |
| mask | [batch, seq] | assistant 区间为 1，其余为 0 |
| ref_log_probs（求和后） | [batch] | 每个 sample 的 response 总 log 概率（ref） |
| logits | [N] | chosen 与 rejected 的隐式奖励差 |
| loss | 标量 | DPO 损失 |

### 关键计算逻辑逐行解释

**步骤 1：按 mask 求和，得到每个 sample 的 response 总 log 概率**

```python
ref_log_probs = (ref_log_probs * mask).sum(dim=1)        # [batch, seq] -> [batch]
policy_log_probs = (policy_log_probs * mask).sum(dim=1) # [batch, seq] -> [batch]
```

mask 为 0 的 prompt 位置被清零，只累加 assistant 回复部分。这一步实现了 `log π(y|x) = Σ_{t∈response} log π(y_t|...)`。

**步骤 2：切分 chosen / rejected**

```python
batch_size = ref_log_probs.shape[0]      # 2N
chosen_ref_log_probs = ref_log_probs[:batch_size // 2]       # 前 N 条 chosen
reject_ref_log_probs = ref_log_probs[batch_size // 2:]       # 后 N 条 rejected
chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
reject_policy_log_probs = policy_log_probs[batch_size // 2:]
```

因为 chosen/rejected 在 train_epoch 中沿 batch 维拼接（见 7.5），这里用切片分开。

**步骤 3：构造 DPO logits**

```python
pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
logits = pi_logratios - ref_logratios
```

展开看：

```
logits = [log π_θ(y_w) - log π_θ(y_l)] - [log π_ref(y_w) - log π_ref(y_l)]
       = [log π_θ(y_w) - log π_ref(y_w)] - [log π_θ(y_l) - log π_ref(y_l)]
       = r̂(x, y_w) - r̂(x, y_l)
```

这正是隐式奖励在 chosen 与 rejected 上的差值。当策略模型相对参考模型更偏好 chosen 时，`logits > 0`。

**步骤 4：DPO 损失**

```python
loss = -F.logsigmoid(beta * logits)
return loss.mean()
```

即 `-log σ(β · logits)`。

- 当 `logits > 0`（策略已正确偏好 chosen），`σ(β·logits) → 1`，`-log σ → 0`，loss 小。
- 当 `logits < 0`（策略错误偏好 rejected），`σ(β·logits) → 0`，`-log σ → 大`，loss 大。

梯度会推动策略模型提高 chosen 的概率、降低 rejected 的概率，同时 `β` 通过 KL 约束防止偏离参考模型过远。

---

## 7.5 train_epoch：双模型前向与拼接

### 原理说明

DPO 训练每步需要两个模型：**policy 模型**（可训练，会更新参数）和 **ref 模型**（冻结，仅提供基线）。两者对同一条数据各做一次前向，得到各自的 logits，再转成 log 概率送入 `dpo_loss`。

为节省算力，chosen 和 rejected 在 batch 维拼接成 `[2N, seq]` 一次前向，比分别前向少一半 kernel launch 开销。ref 模型用 `torch.no_grad()` 包裹，不计算梯度。

### 代码位置引用

- 代码链接：[trainer/train_dpo.py:56-122](file:///home/zhangxm/model_minimind/trainer/train_dpo.py#L56-L122)

### 输入/输出张量说明（单步）

| 张量 | Shape | 含义 |
|------|-------|------|
| x | [2N, seq] | chosen + rejected 拼接的输入 |
| y | [2N, seq] | 拼接的预测目标 |
| mask | [2N, seq] | 拼接的 loss mask |
| ref_logits | [2N, seq, vocab] | ref 模型输出（no_grad） |
| policy_logits | [2N, seq, vocab] | policy 模型输出 |
| ref_log_probs | [2N, seq] → [2N] | ref 逐 token log 概率 → response 总和 |
| policy_log_probs | [2N, seq] → [2N] | policy 逐 token log 概率 → response 总和 |
| dpo_loss_val | 标量 | DPO 损失 |
| loss | 标量 | dpo_loss + aux_loss（MoE 时） |

其中 N = batch_size，2N 是 chosen 和 rejected 拼接后的总 batch。

### 关键计算逻辑逐行解释

```python
for step, batch in enumerate(loader, start=start_step + 1):
    x_chosen = batch['x_chosen'].to(args.device)
    x_rejected = batch['x_rejected'].to(args.device)
    y_chosen = batch['y_chosen'].to(args.device)
    y_rejected = batch['y_rejected'].to(args.device)
    mask_chosen = batch['mask_chosen'].to(args.device)
    mask_rejected = batch['mask_rejected'].to(args.device)
    # 把 chosen 和 rejected 拼到 batch 维度，前半段 chosen、后半段 rejected
    x = torch.cat([x_chosen, x_rejected], dim=0)
    y = torch.cat([y_chosen, y_rejected], dim=0)
    mask = torch.cat([mask_chosen, mask_rejected], dim=0)
```

**步骤 1：拼接 chosen/rejected**

三对张量沿 batch 维 cat，得到 `[2N, seq]`。前 N 行 chosen、后 N 行 rejected——这个顺序与 `dpo_loss` 中的切片 `[:batch_size//2]` / `[batch_size//2:]` 严格对应。

```python
    lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    with autocast_ctx:
        with torch.no_grad():
            # reference model 冻结不训练，只提供"原模型偏好"作为对照基线
            ref_outputs = ref_model(x)
            ref_logits = ref_outputs.logits
        ref_log_probs = logits_to_log_probs(ref_logits, y)

        outputs = model(x)
        logits = outputs.logits
        policy_log_probs = logits_to_log_probs(logits, y)

        dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
        loss = dpo_loss_val + outputs.aux_loss
        loss = loss / args.accumulation_steps

    scaler.scale(loss).backward()
```

**步骤 2：ref 模型前向（no_grad）**

ref 模型在 `__main__` 中已 `eval()` + `requires_grad_(False)`，这里再用 `torch.no_grad()` 包裹确保不建图、不占额外显存。ref 的 logits 转 log 概率的过程与 policy 完全相同（同一个 `logits_to_log_probs` 函数）。

**步骤 3：policy 模型前向**

policy 模型（`model`）正常前向，构建计算图。注意 DPO 不传 `labels` 给模型——直接取 `outputs.logits` 自行处理，避免触发模型内部的 CE loss。

**步骤 4：算 DPO loss**

调用 `dpo_loss(ref_log_probs, policy_log_probs, mask, beta)`。若为 MoE 模型，额外加上 `outputs.aux_loss`（路由均衡损失，非 MoE 为 0）。再除以 `accumulation_steps` 做梯度累积（与 Pretrain/SFT 一致）。

**步骤 5：反向 + 更新**

```python
    if step % args.accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

梯度累积、混合精度、梯度裁剪的逻辑与 Pretrain/SFT 完全一致（见第 6 章），此处不重复。**关键区别**：只有 policy 模型的参数会被更新，ref 模型始终冻结。

### ref 模型的初始化

[trainer/train_dpo.py:175-181](file:///home/zhangxm/model_minimind/trainer/train_dpo.py#L175-L181)：

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)       # policy
ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)           # ref
ref_model.eval()
ref_model.requires_grad_(False)
```

policy 与 ref 从**同一个 SFT 权重**（`from_weight='full_sft'`，默认）初始化，保证训练起点一致。之后 policy 随训练更新，ref 始终保持初始 SFT 权重作为 KL 参考点。

### DPO 超参特点

| 参数 | 默认值 | 说明 |
|------|--------|------|
| learning_rate | 4e-8 | 比 SFT（1e-5）小 3 个数量级，避免大幅遗忘 SFT 能力 |
| beta | 0.15 | KL 约束强度，越大越保守 |
| batch_size | 4 | 因一次前向要算 2N 条 + 双模型，显存压力大 |
| accumulation_steps | 1 | DPO 数据量小，通常不累积 |
| from_weight | full_sft | 必须基于 SFT 权重起训 |

学习率极小（4e-8）是 DPO 的典型设置——偏好对齐是对已学能力的**微调**，大学习率会迅速破坏 SFT 学到的语言能力（catastrophic forgetting）。

---

## 7.6 张量 shape 全流程汇总

以 `batch_size=N=4`、`seq_len=L=1023`（max_length-1）、`vocab_size=6400` 为例，追踪一次前向的形状变化：

| 阶段 | 张量 | Shape | 说明 |
|------|------|-------|------|
| Dataset 输出 | x_chosen | [L] | 单条 chosen 输入 |
| DataLoader 堆叠 | x_chosen | [N, L] | batch 维 |
| 拼接 chosen+rejected | x | [2N, L] = [8, 1023] | 一次前向 |
| policy 前向 | logits | [2N, L, vocab] = [8, 1023, 6400] | 模型输出 |
| log_softmax | log_probs | [2N, L, vocab] | log 概率分布 |
| gather | log_probs_per_token | [2N, L] = [8, 1023] | 真实 token 的 log 概率 |
| mask 求和 | policy_log_probs | [2N] = [8] | 每个 sample 的 response 总 log 概率 |
| 切分 chosen/rejected | chosen_policy_log_probs | [N] = [4] | 前 N 条 |
| | reject_policy_log_probs | [N] = [4] | 后 N 条 |
| DPO logits | logits | [N] = [4] | 隐式奖励差 |
| loss | 标量 | [] | -log σ(β·logits) 的均值 |

ref 模型走完全相同的形状链路，只是不建图。最终 `dpo_loss` 把两条链路的 `[N]` 张量按公式组合成标量损失。

---

## 小结

本章拆解了 MiniMind 的 DPO 偏好优化流程：

| 组件 | 作用 | 所在位置 |
|------|------|---------|
| DPODataset | 构造 chosen/rejected 对 + loss mask | lm_dataset.py:132 |
| generate_loss_mask | 标记 assistant 回复区间 | lm_dataset.py:189 |
| logits_to_log_probs | logits → 逐 token log 概率 | train_dpo.py:25 |
| dpo_loss | 偏好损失 -log σ(β·Δlogπ) | train_dpo.py:35 |
| train_epoch | 双模型前向 + 拼接 + 更新 | train_dpo.py:56 |

核心要点：

1. **DPO 的本质**是用 `β·(log π_θ - log π_ref)` 作为隐式奖励，把 RLHF 的“奖励模型 + RL”简化为“直接在偏好对上做监督学习”，无需在线采样、训练更稳定。
2. **双模型设计**：policy 可训练、ref 冻结，两者从同一 SFT 权重起步。ref 提供 KL 基线，防止 policy 为迎合偏好而跑偏。
3. **拼接前向**：chosen/rejected 在 batch 维拼接一次前向，配合 `dpo_loss` 中的切片，节省一半 kernel 开销。
4. **mask 求和**：只对 assistant 回复部分累加 log 概率，prompt 部分被 mask 过滤——因为 chosen/rejected 的 prompt 相同，比较其概率无意义。
5. **极小学习率**（4e-8）是 DPO 的典型特征，偏好对齐是对 SFT 能力的精修，而非重学。

至此 MiniMind 的三阶段训练流程（Pretrain → SFT → DPO）介绍完毕：预训练赋予语言能力，SFT 赋予对话格式，DPO 对齐人类偏好。
