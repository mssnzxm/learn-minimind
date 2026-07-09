# 第 4 章 前馈网络 MLP & MoE

本章讲解 Transformer Decoder Block 中的另一半核心组件——前馈神经网络（Feed-Forward Network, FFN）。我们将从传统 FFN 的不足出发，介绍 MiniMind 采用的 SwiGLU 门控前馈结构，并进一步扩展到 Mixture-of-Experts（MoE，混合专家）层，理解其路由机制与负载均衡辅助损失。结合 MiniMind 实际代码，每一步的张量变换和计算逻辑都会被详细拆解。

---

## 4.1 前馈网络概述

### FFN 在 Transformer 中的作用

注意力机制（第 3 章）负责 token 之间的信息混合，而 FFN 负责对**每个位置独立地**做非线性变换。可以理解为：

- **Attention**：token 之间"交流"，聚合上下文信息
- **FFN**：每个 token"思考"，把聚合到的信息通过非线性层进行加工和变换

FFN 对序列中的每个 token 独立施加相同的变换（参数共享），因此也叫 position-wise feed-forward network。

### 传统 FFN 的结构

传统 Transformer（如原始 Attention is All You Need 论文）使用两层线性层加 ReLU 激活：

```
FFN(x) = W2 · ReLU(W1 · x + b1) + b2
```

- 输入 x: [hidden_size]
- 中间层：[intermediate_size]，通常 intermediate_size = 4 × hidden_size
- 输出：[hidden_size]

**传统 FFN 的不足**：

1. **表达能力有限**：ReLU 是硬门控（负数直接置零），无法精细控制信息流
2. **参数效率低**：要达到同样的效果，往往需要更多参数
3. **扩展性差**：当模型规模增大时，传统 FFN 的收益递减

### 现代大模型的改进：SwiGLU

LLaMA 系列模型采用 SwiGLU（Swish-Gated Linear Unit）替代传统 FFN。SwiGLU 引入门控机制，让模型自己学习"哪些信息应该通过、哪些应该被抑制"，在同等参数量下效果更好。

MiniMind 默认配置：
- `hidden_size = 768`
- `intermediate_size = 2432`（由 `math.ceil(768 × π / 64) × 64` 计算得到，约为 hidden_size 的 π 倍）
- `hidden_act = 'silu'`（SwiGLU 使用的激活函数）

---

## 4.2 SwiGLU 前馈网络

### 原理说明

SwiGLU 的核心思想是引入一个**门控分支**（gate）和一个**值分支**（up），两者通过逐元素相乘实现门控：

```
SwiGLU(x) = down_proj( silu(gate_proj(x)) * up_proj(x) )
```

其中：
- `gate_proj`：将 hidden_size 映射到 intermediate_size，输出经过 SiLU 激活，作为"门"
- `up_proj`：将 hidden_size 映射到 intermediate_size，不激活，作为"值"
- `down_proj`：将 intermediate_size 映射回 hidden_size，得到最终输出
- `silu(x) = x * sigmoid(x)`：Swish 激活函数，平滑且处处可导

### 门控机制的优势

对比传统 ReLU FFN：

| 方面 | 传统 FFN | SwiGLU |
|------|---------|--------|
| 激活函数 | ReLU（硬门控） | SiLU（软门控，平滑） |
| 信息流控制 | 负数直接置零 | 门控分支学习动态权重 |
| 参数量 | 2 个线性层 | 3 个线性层（gate/up/down） |
| 表达能力 | 较弱 | 更强（门控让模型自适应） |
| 同效果参数效率 | 基准 | 更高 |

**SiLU 与 ReLU 的对比**：

- ReLU(x) = max(0, x)：负数全部置零，梯度在负区间为 0
- SiLU(x) = x · σ(x)：负数小幅保留，过渡平滑，梯度处处非零

SwiGLU 中 `silu(gate_proj(x))` 输出的是 [0, 1] 附近的软掩码（因为 SiLU 在正区间接近 x，在负区间接近 0），再乘以 `up_proj(x)` 实现"按比例放行"。

### 代码位置引用

FeedForward 类定义在模型文件中：

- 代码链接：[model/model_minimind.py:149-160](file:///home/zhangxm/model_minimind/model/model_minimind.py#L149-L160)

```python
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # SwiGLU: act(gate_proj(x)) * up_proj(x)，再投影回 hidden_size。
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| x（输入） | [batch, seq_len, hidden_size] | float32/bf16 | 来自 Attention 残差后的隐藏状态 |
| gate_proj(x) | [batch, seq_len, intermediate_size] | float32/bf16 | 门控分支（激活前） |
| silu(gate_proj(x)) | [batch, seq_len, intermediate_size] | float32/bf16 | 门控分支（激活后，软掩码） |
| up_proj(x) | [batch, seq_len, intermediate_size] | float32/bf16 | 值分支 |
| act(gate) * up | [batch, seq_len, intermediate_size] | float32/bf16 | 门控后的中间表示 |
| output | [batch, seq_len, hidden_size] | float32/bf16 | down_proj 投影回 hidden_size |

**以 MiniMind 默认配置为例**（batch=1, seq_len=100）：

| 阶段 | 张量 | Shape |
|------|------|-------|
| 输入 | x | [1, 100, 768] |
| gate_proj | gate | [1, 100, 2432] |
| up_proj | up | [1, 100, 2432] |
| SiLU(gate) * up | mid | [1, 100, 2432] |
| down_proj | output | [1, 100, 768] |

### 关键计算逻辑解释

SwiGLU 的前向计算分三步：

**步骤 1：双路投影（gate 与 up）**

```python
gate = self.gate_proj(x)   # [batch, seq, 768] -> [batch, seq, 2432]
up   = self.up_proj(x)     # [batch, seq, 768] -> [batch, seq, 2432]
```

两个线性层都是 hidden_size → intermediate_size，但参数独立。gate 分支负责"决定放多少"，up 分支负责"提供什么内容"。

**步骤 2：门控相乘**

```python
mid = self.act_fn(gate) * up   # SiLU(gate) 逐元素乘 up
```

- `act_fn = ACT2FN['silu']`，即 `SiLU(x) = x * sigmoid(x)`
- 逐元素相乘：门控分支的每个维度独立调制 up 分支对应维度
- SiLU 让 gate 在正区间接近线性放行，在负区间小幅抑制，形成平滑的软门控

**步骤 3：降维投影**

```python
output = self.down_proj(mid)   # [batch, seq, 2432] -> [batch, seq, 768]
```

down_proj 将 intermediate_size 投影回 hidden_size，使得 FFN 的输出可以与输入做残差相加（详见第 5 章 MiniMindBlock）。

### 参数量分析

以默认配置（hidden_size=768, intermediate_size=2432）为例：

| 子层 | 权重 Shape | 参数量 |
|------|-----------|--------|
| gate_proj | [768, 2432] | 1,871,616 |
| up_proj | [768, 2432] | 1,871,616 |
| down_proj | [2432, 768] | 1,871,616 |
| **合计** | - | **5,614,848** |

注意：SwiGLU 有 3 个线性层，而传统 FFN 只有 2 个。为公平对比，SwiGLU 通常把 intermediate_size 缩小（如 LLaMA 用 2/3 × 4 × hidden_size），MiniMind 这里用 π × hidden_size 也是类似考量。

---

## 4.3 Mixture-of-Experts（MoE）混合专家

### 原理说明

MoE（Mixture-of-Experts，混合专家）是一种**稀疏激活**的前馈网络结构。它的核心思想是：用多个独立的 FFN（称为"专家"）替代单个 FFN，对每个 token 动态路由到最合适的若干个专家，只激活被选中的专家参与计算。

```
MoE(x) = Σ_{i ∈ top-k}  gate_score_i · Expert_i(x)
```

- **专家（Expert）**：每个 Expert 是一个独立的 SwiGLU FFN
- **门控（Gate）/路由器（Router）**：一个线性层，输出每个 token 对每个专家的偏好分数
- **Top-K 选择**：每个 token 只激活得分最高的 K 个专家

### MoE 的优势

1. **参数容量大，计算量小**：模型总参数量随专家数线性增长，但每个 token 只激活 K 个专家，FLOPs 与 dense 模型相当
2. **专业化分工**：不同专家可以学习不同模式（如代码、数学、语言），提升模型整体能力
3. **扩展性好**：增加专家数即可扩大模型容量，而不显著增加单 token 计算成本

### MiniMind MoE 默认配置

- `num_experts = 4`：专家数量
- `num_experts_per_tok = 1`：每个 token 激活的专家数（top-1 路由）
- `moe_intermediate_size = 2432`：每个专家的中间层宽度（与 dense 的 intermediate_size 相同）
- `norm_topk_prob = True`：对 top-k 权重做归一化
- `router_aux_loss_coef = 5e-4`：辅助损失系数

### 代码位置引用

MOEFeedForward 类定义在模型文件中：

- 代码链接：[model/model_minimind.py:162-192](file:///home/zhangxm/model_minimind/model/model_minimind.py#L162-L192)

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        # gate 为每个 token 选择最合适的专家；topk_idx 是专家编号，topk_weight 是路由权重。
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            # 辅助损失鼓励 token 更均匀地分配到各专家，避免某个专家被过度使用。
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)
```

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| x（输入） | [batch, seq_len, hidden_size] | float32/bf16 | 隐藏状态 |
| x_flat | [num_tokens, hidden_size] | float32/bf16 | 展平后的 token（num_tokens = batch × seq） |
| gate(x_flat) | [num_tokens, num_experts] | float32/bf16 | 每个 token 对每个专家的 logits |
| scores | [num_tokens, num_experts] | float32/bf16 | softmax 后的路由概率 |
| topk_weight | [num_tokens, num_experts_per_tok] | float32/bf16 | 被选中专家的权重 |
| topk_idx | [num_tokens, num_experts_per_tok] | long | 被选中专家的编号 |
| expert(x) | [n_routed, hidden_size] | float32/bf16 | 单个专家的输出 |
| y | [num_tokens, hidden_size] | float32/bf16 | 加权聚合后的输出 |
| output | [batch, seq_len, hidden_size] | float32/bf16 | reshape 回原始形状 |

**以 MiniMind MoE 默认配置为例**（batch=1, seq_len=100, num_experts=4, top-1）：

| 阶段 | 张量 | Shape |
|------|------|-------|
| 输入 | x | [1, 100, 768] |
| 展平 | x_flat | [100, 768] |
| gate 投影 | logits | [100, 4] |
| softmax | scores | [100, 4] |
| top-1 选择 | topk_weight | [100, 1] |
| top-1 编号 | topk_idx | [100, 1] |
| 各专家输出加权聚合 | y | [100, 768] |
| reshape | output | [1, 100, 768] |

### 关键计算逻辑解释

MoE 的前向计算分为路由、专家计算、聚合三步：

**步骤 1：门控路由（Gate Routing）**

```python
x_flat = x.view(-1, hidden_dim)                       # [batch*seq, hidden]
scores = F.softmax(self.gate(x_flat), dim=-1)          # [batch*seq, num_experts]
topk_weight, topk_idx = torch.topk(scores, k=..., dim=-1, sorted=False)
```

- 先把 [batch, seq, hidden] 展平成 [num_tokens, hidden]，便于按 token 处理
- `gate` 线性层将 hidden_size 映射到 num_experts 维，得到每个 token 对每个专家的偏好 logits
- `softmax` 归一化为概率分布
- `torch.topk` 选出得分最高的 K 个专家，返回权重和编号

**步骤 2：权重归一化**

```python
if self.config.norm_topk_prob:
    topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
```

- top-k 选出的权重之和通常不为 1（因为只是从全量 softmax 中取出最大的 K 个）
- `norm_topk_prob=True` 时，对选中的 K 个权重重新归一化，使它们之和为 1
- `+ 1e-20` 防止除零
- 这样最终输出是各专家输出的加权平均，权重之和为 1

**步骤 3：逐专家计算并聚合**

```python
y = torch.zeros_like(x_flat)
for i, expert in enumerate(self.experts):
    mask = (topk_idx == i)               # [num_tokens, K]，标记哪些 token 选了专家 i
    if mask.any():
        token_idx = mask.any(dim=-1).nonzero().flatten()    # 选了专家 i 的 token 编号
        weight = topk_weight[mask].view(-1, 1)              # 对应权重
        y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
```

- 遍历每个专家 i
- `mask = (topk_idx == i)`：找出哪些 token 在 top-k 选择中选了专家 i（mask 形状 [num_tokens, K]）
- `mask.any(dim=-1)`：只要 token 在 K 个选择中有任意一个等于 i，就标记为 True
- `index_add_`：把"专家 i 的输出 × 权重"累加到 y 的对应 token 位置
- 一个 token 如果选了多个专家（K>1），多个专家的加权结果会累加，最终得到加权和

**步骤 4：空专家的处理（保持计算图完整）**

```python
elif self.training:
    y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
```

- 如果某个专家在当前 batch 中**没有任何 token 选中**（mask.any() 为 False），它的参数不会出现在前向计算中
- 但反向传播时，DDP（分布式数据并行）要求所有参数参与梯度同步，否则会死锁
- 这里通过 `0 * sum(p.sum())` 把该专家的参数加入计算图，梯度为 0，保证 DDP 同步正常

### 完整数据流示意

以 batch=1, seq=100, num_experts=4, top-1 为例：

```
x [1, 100, 768]
  │ view
  ▼
x_flat [100, 768]
  │ gate 线性层
  ▼
logits [100, 4]  ── softmax ──▶ scores [100, 4]
                                   │ topk(k=1)
                                   ▼
                          topk_weight [100, 1], topk_idx [100, 1]
                                   │ 归一化 + 遍历专家
                                   ▼
                              y [100, 768]
                                   │ view
                                   ▼
                            output [1, 100, 768]
```

---

## 4.4 负载均衡辅助损失

### 原理说明

MoE 训练中常见的问题是**路由崩塌（Routing Collapse）**：所有 token 都倾向于选择同少数几个专家，导致其他专家得不到训练，模型容量被浪费。

产生原因：
- 门控路由器是端到端学习的，初始阶段某些专家"碰巧"表现好，就会吸引更多 token
- 更多 token → 更多梯度 → 该专家更强 → 吸引更多 token，形成正反馈循环
- 最终可能所有 token 都涌向同一个专家，MoE 退化为 dense FFN

### 辅助损失的设计

MiniMind 采用经典的负载均衡损失（Load Balancing Loss），鼓励 token 均匀分布到各专家：

```
aux_loss = num_experts × Σ_i ( f_i × P_i )
```

其中：
- `f_i`（专家频率）：实际被路由到专家 i 的 token 占比（基于 top-k 选择统计）
- `P_i`（专家概率）：所有 token 对专家 i 的平均路由概率（基于 softmax 输出）
- `num_experts`：归一化系数

**两个统计量的区别**：
- `f_i` 是离散的（基于 hard 的 top-k 选择），不可导
- `P_i` 是连续的（基于 softmax 概率），可导，用于反传梯度
- 两者相乘构成可导的损失，间接优化路由分布

### 代码位置引用

辅助损失计算在 MOEFeedForward.forward 末尾：

- 代码链接：[model/model_minimind.py:188-191](file:///home/zhangxm/model_minimind/model/model_minimind.py#L188-L191)

```python
if self.training and self.config.router_aux_loss_coef > 0:
    # 辅助损失鼓励 token 更均匀地分配到各专家，避免某个专家被过度使用。
    load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
    self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
else:
    self.aux_loss = scores.new_zeros(1).squeeze()
```

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| topk_idx | [num_tokens, num_experts_per_tok] | long | 每个 token 选中的专家编号 |
| scores | [num_tokens, num_experts] | float32 | softmax 后的路由概率 |
| load (f_i) | [num_experts] | float32 | 每个专家被选中的频率 |
| scores.mean(0) (P_i) | [num_experts] | float32 | 每个专家的平均路由概率 |
| aux_loss | 标量 [] | float32 | 负载均衡辅助损失 |

### 关键计算逻辑解释

**步骤 1：计算专家频率 f_i（load）**

```python
load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
```

- `topk_idx` 形状 [num_tokens, K]（每个 token 选了 K 个专家）
- `F.one_hot(topk_idx, num_experts)`：把专家编号转成 one-hot，形状 [num_tokens, K, num_experts]
- `.float().mean(0)`：对 num_tokens 维度求平均，得到每个专家被选中的频率
- 例如 num_experts=4, top-1, 100 个 token，如果专家 0 被选了 30 次，则 `load[0] = 0.3`

**步骤 2：计算专家概率 P_i（scores.mean(0)）**

```python
scores.mean(0)   # [num_experts]
```

- `scores` 是 softmax 后的路由概率 [num_tokens, num_experts]
- 对 num_tokens 维度求平均，得到每个专家的平均路由概率
- 注意：这里用的是**全量 softmax 概率**，不是 top-k 后的，所以即使某专家没被选中，它的 P_i 也不为 0

**步骤 3：计算辅助损失**

```python
self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
```

- `(load * scores.mean(0)).sum()`：f_i × P_i 对所有专家求和
- `× num_experts`：乘以专家数做归一化，使得均匀分布时损失最小
- `× router_aux_loss_coef`（默认 5e-4）：缩放系数，控制辅助损失相对主损失的权重

**为什么这个损失能促进均衡？**

考虑两种极端情况：

| 情况 | load (f_i) | P_i | f_i × P_i 之和 | 说明 |
|------|-----------|-----|---------------|------|
| 完全均衡（每个专家 25%） | [0.25, 0.25, 0.25, 0.25] | [0.25, 0.25, 0.25, 0.25] | 4 × 0.0625 = 0.25 | 最小值 |
| 完全崩塌（全选专家 0） | [1.0, 0, 0, 0] | [0.7, 0.1, 0.1, 0.1] | 0.7 + 0 = 0.7 | 较大值 |

均匀分布时损失最小（0.25 × 4 = 1.0，乘 num_experts 后为 1.0），崩塌时损失大（0.7 × 4 = 2.8）。最小化 aux_loss 会推动路由器向均衡分布靠拢。

**辅助损失的累加**：

在 MiniMindModel 中，所有 MoE 层的 aux_loss 会累加：

```python
aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], ...)
```

最终在 MiniMindForCausalLM 的输出中作为 `aux_loss` 返回，训练时与主交叉熵损失相加（详见第 5 章）。

---

## 小结

本章介绍了 MiniMind 的两种前馈网络结构：

| 结构 | 适用场景 | 关键机制 | 代码位置 |
|------|---------|---------|---------|
| FeedForward (SwiGLU) | Dense 模型（use_moe=False） | gate/up/down 三路，SiLU 门控 | L149-160 |
| MOEFeedForward | MoE 模型（use_moe=True） | 多专家 + 门控路由 + 负载均衡损失 | L162-192 |

核心要点：

1. **SwiGLU** 通过 `silu(gate_proj(x)) * up_proj(x)` 的门控机制，让模型自适应控制信息流，比传统 ReLU FFN 表达能力更强
2. **MoE** 通过多个稀疏激活的专家，在不增加单 token 计算量的前提下大幅扩展模型参数容量
3. **负载均衡辅助损失** 通过 `f_i × P_i` 的形式鼓励 token 均匀分布到各专家，防止路由崩塌
4. Dense 与 MoE 在 MiniMindBlock 中通过 `config.use_moe` 切换，接口完全一致（输入输出都是 [batch, seq, hidden_size]）

下一章我们将把 Attention 和 FFN 组装成完整的 Transformer Block，并讲解 MiniMindModel 与 MiniMindForCausalLM 的整体前向传播。
