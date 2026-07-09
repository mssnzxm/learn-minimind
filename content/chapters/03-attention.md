# 第 3 章 注意力机制 (Attention)

本章详细讲解 Transformer 最核心的组件——注意力机制（Attention）。我们将从自注意力的基本概念出发，逐步拆解 Q/K/V 投影、RoPE 位置编码、GQA 分组查询、因果掩码、Flash Attention 等关键技术，并结合 MiniMind 的实际代码，理解每一步的张量变换和计算逻辑。

---

## 3.1 注意力机制概述

### 什么是自注意力（Self-Attention）

自注意力（Self-Attention）是一种让序列中每个 token 都能关注到其他所有 token 的机制。通过计算 token 之间的相关性权重，模型可以动态地聚合上下文信息，从而更好地理解语义。

与传统的 RNN/LSTM 相比，自注意力的优势在于：
- **并行计算**：所有位置的计算可以同时进行，训练效率高
- **长距离依赖**：任意两个 token 之间的路径长度都是 O(1)，不受距离影响
- **动态权重**：根据内容动态调整注意力权重，而不是固定的卷积核

### Query、Key、Value 的含义

自注意力机制通过三个线性投影将输入向量映射为三个不同的向量：

| 向量 | 全称 | 作用 | 类比 |
|------|------|------|------|
| Q | Query（查询） | 表示当前 token "想要找什么" | 搜索框里的关键词 |
| K | Key（键） | 表示每个 token "包含什么" | 文档的标题/标签 |
| V | Value（值） | 表示每个 token "实际内容是什么" | 文档的正文内容 |

计算过程：
1. 用 Q 与所有 K 做点积，得到注意力分数（相关性）
2. 对分数做 softmax 归一化，得到注意力权重
3. 用权重对 V 加权求和，得到最终输出

### 为什么需要多头注意力（Multi-Head Attention）

多头注意力（Multi-Head Attention）将 Q、K、V 分成多个"头"，每个头学习不同类型的注意力模式：

- **不同头关注不同关系**：有的头关注语法依赖，有的头关注指代关系，有的头关注局部相邻词
- **增加表达能力**：多个头的结果拼接后，模型可以同时捕捉多种类型的语义关系
- **类比**：就像读书时从不同角度理解文章——有的关注人物关系，有的关注时间线，有的关注因果逻辑

### 代码位置：Attention 类

MiniMind 的 Attention 类定义在模型文件中：

- 代码链接：[model/model_minimind.py:100-150](file:///home/zhangxm/model_minimind/model/model_minimind.py#L100-L150)

 `python
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        # ...
 `

**MiniMind 默认配置**：
- 
um_attention_heads = 8：Q 的头数
- 
um_key_value_heads = 4：KV 的头数（GQA，为 Q 头数的一半）
- head_dim = 96：每个头的维度

---

## 3.2 分步详解

### 步骤 1：Q/K/V 线性投影

#### 原理说明

输入的 hidden_states 分别通过三个独立的线性层（q_proj、k_proj、v_proj），投影得到 Q、K、V 三个向量。然后将它们 reshape 成多头格式。

#### 代码位置引用

- q_proj / k_proj / v_proj 定义：[model/model_minimind.py:108-113](file:///home/zhangxm/model_minimind/model/model_minimind.py#L108-L113)
- 前向传播投影部分：[model/model_minimind.py:119-123](file:///home/zhangxm/model_minimind/model/model_minimind.py#L119-L123)

 `python
self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
 `

#### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| x（输入） | [batch, seq_len, hidden_size] | float32/bf16 | 输入隐藏状态 |
| xq（Q 投影后） | [batch, seq_len, n_heads * head_dim] | float32/bf16 | Q 向量（未 reshape） |
| xk（K 投影后） | [batch, seq_len, n_kv_heads * head_dim] | float32/bf16 | K 向量（未 reshape） |
| xv（V 投影后） | [batch, seq_len, n_kv_heads * head_dim] | float32/bf16 | V 向量（未 reshape） |
| xq（reshape 后） | [batch, seq_len, n_heads, head_dim] | float32/bf16 | Q 多头格式 |
| xk（reshape 后） | [batch, seq_len, n_kv_heads, head_dim] | float32/bf16 | K 多头格式 |
| xv（reshape 后） | [batch, seq_len, n_kv_heads, head_dim] | float32/bf16 | V 多头格式 |

**以 MiniMind 默认配置为例**：

| 张量 | Shape |
|------|-------|
| x | [1, 100, 768] |
| xq 投影后 | [1, 100, 768]（8 x 96） |
| xk 投影后 | [1, 100, 384]（4 x 96） |
| xv 投影后 | [1, 100, 384]（4 x 96） |
| xq reshape 后 | [1, 100, 8, 96] |
| xk reshape 后 | [1, 100, 4, 96] |
| xv reshape 后 | [1, 100, 4, 96] |

#### 为什么 KV 头数可以比 Q 少（GQA）

GQA（Grouped Query Attention，分组查询注意力）是一种优化技术：

- **MHA（多头注意力）**：n_heads = n_kv_heads，每个 Q 头对应一个独立的 K/V 头
- **GQA（分组查询注意力）**：n_kv_heads < n_heads，多个 Q 头共享同一组 K/V 头
- **MQA（多查询注意力）**：n_kv_heads = 1，所有 Q 头共享同一组 K/V（GQA 的特例）

**GQA 的优势**：
- 减少 KV cache 的显存占用（KV cache 只有 n_kv_heads 份）
- 推理速度更快（需要加载的 KV 数据更少）
- 效果接近 MHA，远好于 MQA（是效果和效率的良好折中）

**分组方式**：将 n_heads 个 Q 头分成 n_kv_heads 组，每组共享一对 K/V 头。
例如 n_heads=8, n_kv_heads=4 时，Q 头 0-1 用 KV 头 0，Q 头 2-3 用 KV 头 1，以此类推。

---

### 步骤 2：QK-Norm

#### 原理说明

QK-Norm 是指在计算注意力之前，先对 Q 和 K 分别做 RMSNorm 归一化。归一化作用在每个 head 的 head_dim 维度上。

#### 作用

1. **稳定训练**：防止 Q、K 的模长过大导致注意力分数爆炸
2. **提升注意力质量**：归一化后的点积更稳定，softmax 分布更合理
3. **配合 RoPE**：旋转位置编码后向量模长不变，但数值分布可能变化，Norm 可以进一步稳定

#### 代码位置引用

- q_norm / k_norm 定义：[model/model_minimind.py:114-115](file:///home/zhangxm/model_minimind/model/model_minimind.py#L114-L115)
- 前向传播使用：[model/model_minimind.py:124](file:///home/zhangxm/model_minimind/model/model_minimind.py#L124-L124)

 `python
self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
 `

#### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| xq（输入） | [batch, seq_len, n_heads, head_dim] | float32/bf16 | Q 多头格式 |
| xk（输入） | [batch, seq_len, n_kv_heads, head_dim] | float32/bf16 | K 多头格式 |
| xq（输出） | [batch, seq_len, n_heads, head_dim] | float32/bf16 | 归一化后的 Q |
| xk（输出） | [batch, seq_len, n_kv_heads, head_dim] | float32/bf16 | 归一化后的 K |

**注意**：QK-Norm 只对 Q 和 K 做归一化，不对 V 做。因为注意力权重是由 QK 点积计算的，V 只是被加权求和的对象。

---

### 步骤 3：RoPE 旋转位置编码

#### 原理说明

对 Q 和 K 施加旋转位置编码（RoPE），为注意力机制注入位置信息。RoPE 通过旋转向量的方式，让点积天然具有相对位置感知能力。

详细原理请参考第 2 章的 RoPE 章节。

#### 代码位置引用

- 函数名：pply_rotary_pos_emb
- 代码链接：[model/model_minimind.py:87-93](file:///home/zhangxm/model_minimind/model/model_minimind.py#L87-L93)

 `python
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed
 `

#### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| q | [batch, seq_len, n_heads, head_dim] | float32/bf16 | Q 向量 |
| k | [batch, seq_len, n_kv_heads, head_dim] | float32/bf16 | K 向量 |
| cos | [seq_len, head_dim] | float32 | 预计算的 cos 值 |
| sin | [seq_len, head_dim] | float32 | 预计算的 sin 值 |
| q_embed | [batch, seq_len, n_heads, head_dim] | float32/bf16 | 旋转后的 Q |
| k_embed | [batch, seq_len, n_kv_heads, head_dim] | float32/bf16 | 旋转后的 K |

**注意**：RoPE 只作用于 Q 和 K，不作用于 V。因为位置信息是通过 QK 点积引入注意力权重的，V 不需要位置信息。

---

### 步骤 4：GQA 分组查询注意力

#### 原理说明

由于 KV 的头数（n_kv_heads）少于 Q 的头数（n_heads），在计算注意力之前，需要将 KV 的头数扩展（重复）到与 Q 相同的数量，这样才能进行矩阵乘法。

这个重复操作由 
epeat_kv 函数完成。

#### 代码位置引用

- 函数名：
epeat_kv
- 代码链接：[model/model_minimind.py:94-99](file:///home/zhangxm/model_minimind/model/model_minimind.py#L94-L99)

 `python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))
 `

#### 为什么能省显存

KV Cache 只需要保存 n_kv_heads 份 K 和 V，而不是 n_heads 份。

以 MiniMind 默认配置为例：
- n_heads = 8, n_kv_heads = 4
- 显存节省比例：(8 - 4) / 8 = **50%**
- 对于更长的上下文（如 32k、128k），KV cache 的显存节省非常可观

#### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| x（输入 K/V） | [batch, seq_len, n_kv_heads, head_dim] | float32/bf16 | KV 头数较少 |
| n_rep | 标量（int） | - | 重复次数 = n_heads / n_kv_heads |
| x（输出 K/V） | [batch, seq_len, n_heads, head_dim] | float32/bf16 | KV 头数与 Q 一致 |

**示例**（n_heads=8, n_kv_heads=4, n_rep=2）：

| 阶段 | K 的 Shape |
|------|-----------|
| repeat 前 | [1, 100, 4, 96] |
| repeat 后 | [1, 100, 8, 96] |

**repeat_kv 的实现方式**：
1. 在 n_kv_heads 维度后插入一个新维度 -> [batch, seq, n_kv_heads, 1, head_dim]
2. expand 扩展 n_rep 倍 -> [batch, seq, n_kv_heads, n_rep, head_dim]
3. reshape 合并 -> [batch, seq, n_kv_heads * n_rep, head_dim] = [batch, seq, n_heads, head_dim]

---

### 步骤 5：Scaled Dot-Product Attention

#### 原理说明

缩放点积注意力（Scaled Dot-Product Attention）是自注意力的核心计算。公式如下：

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

#### 为什么要除以 sqrt(d_k)

当 d_k（head_dim）很大时，Q 和 K 的点积可能会非常大，导致 softmax 函数进入饱和区（梯度消失）。

举例说明：
- 如果 head_dim = 96，向量每个元素的方差约为 1，那么点积的方差约为 96
- 点积值可能在 [-20, 20] 甚至更大的范围
- softmax 后会变成"one-hot"分布（最大的位置接近 1，其余接近 0），梯度几乎为 0

除以 sqrt(d_k) 后，点积的方差被缩放回约 1，softmax 的分布更平缓，梯度更健康。

#### 张量形状变化

在计算之前，需要先调整维度顺序，将 head 维度移到 seq 维度前面，便于矩阵乘法：

| 阶段 | 张量 | Shape |
|------|------|-------|
| 转置前 | Q | [batch, seq_q, n_heads, head_dim] |
| 转置后 | Q | [batch, n_heads, seq_q, head_dim] |
| 转置前 | K | [batch, seq_k, n_heads, head_dim] |
| 转置后（K^T） | K | [batch, n_heads, head_dim, seq_k] |
| 点积结果 | scores | [batch, n_heads, seq_q, seq_k] |
| softmax 后 | attn_weights | [batch, n_heads, seq_q, seq_k] |
| 乘 V 后 | output | [batch, n_heads, seq_q, head_dim] |

**矩阵乘法维度分析**：
- Q: [batch, n_heads, seq_q, head_dim]
- K^T: [batch, n_heads, head_dim, seq_k]
- scores = Q @ K^T: [batch, n_heads, seq_q, seq_k]
- V: [batch, n_heads, seq_k, head_dim]
- output = scores @ V: [batch, n_heads, seq_q, head_dim]

#### 代码位置引用

- 手写注意力实现：[model/model_minimind.py:140-144](file:///home/zhangxm/model_minimind/model/model_minimind.py#L140-L144)

 `python
scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
# ...
output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
 `

---

### 步骤 6：Causal Mask（因果掩码）

#### 原理说明

因果掩码（Causal Mask）用于确保自回归模型在预测第 t 个 token 时，只能看到第 t 个及之前的 token，不能看到未来的 token。

#### 为什么需要

语言模型是自回归的：生成第 t 个 token 时，只能依赖已经生成的前 t-1 个 token。如果不加掩码，训练时模型就能"偷看"后面的答案，这在推理时是做不到的，会导致训练和推理不一致。

#### 实现方式

构造一个上三角矩阵（对角线以上为 1），将其乘以一个很大的负数（如 -inf），加到注意力分数上。这样 softmax 后，未来位置的权重就变成 0。

掩码矩阵形状：[seq_q, seq_k]（通常 seq_q = seq_k = seq_len）

 `
位置 j ->  0    1    2    3
位置 i v
    0     0  -inf -inf -inf   # 第 0 个 token 只能看自己
    1     0    0  -inf -inf   # 第 1 个 token 能看 0,1
    2     0    0    0  -inf   # 第 2 个 token 能看 0,1,2
    3     0    0    0    0    # 第 3 个 token 能看全部
 `

**ASCII 可视化**（# 表示可见，. 表示不可见）：

 `
        K 的位置
      0  1  2  3
    0 #  .  .  .
Q   1 #  #  .  .
的  2 #  #  #  .
位  3 #  #  #  #
置
 `

下三角（含对角线）是可见的，上三角被 mask 掉。

#### 代码位置引用

- 代码位置：[model/model_minimind.py:142](file:///home/zhangxm/model_minimind/model/model_minimind.py#L142-L142)

 `python
if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
 `

#### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| scores | [batch, n_heads, seq_q, seq_k] | float32/bf16 | 原始注意力分数 |
| causal mask | [seq_q, seq_k] | float32 | 因果掩码（上三角为 -inf） |
| scores（masked） | [batch, n_heads, seq_q, seq_k] | float32/bf16 | 掩码后的分数 |
| attn_weights | [batch, n_heads, seq_q, seq_k] | float32/bf16 | softmax 后的注意力权重 |

---

### 步骤 7：Flash Attention

#### 原理说明

Flash Attention 是一种优化的注意力计算算法，通过**分块计算**和**重计算**技术，在保证计算结果正确的前提下，大幅提升速度并减少显存占用。

MiniMind 使用 PyTorch 内置的 F.scaled_dot_product_attention，它会自动调用 Flash Attention 后端（如果可用）。

#### 优势

| 指标 | 普通注意力 | Flash Attention |
|------|-----------|----------------|
| 速度 | 基准 | 快 2-4 倍 |
| 显存占用 | O(seq^2)（保存完整 scores） | O(seq x block_size)（分块计算） |
| 精度 | - | 相当（甚至更稳定，因为分块计算数值更稳） |

#### 什么时候回退到手写

Flash Attention 虽然快，但不是所有场景都支持。以下情况 MiniMind 会回退到手写实现：

1. **KV Cache 场景**：推理时逐 token 生成（past_key_value 不为 None）
2. **复杂 Mask**：自定义的 attention_mask 不全为 1
3. **单 token 推理**：seq_len == 1 时手写更高效
4. **设备不支持**：没有 Flash Attention 内核时

#### 代码位置引用

- Flash Attention 调用：[model/model_minimind.py:138-139](file:///home/zhangxm/model_minimind/model/model_minimind.py#L138-L139)

 `python
output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
 `

---

### 步骤 8：输出投影 o_proj

#### 原理说明

多头注意力的计算结果是按头分开的，需要先将各个头的结果拼接（concatenate）起来，再通过一个线性层投影回 hidden_size 维度。

#### 计算过程

1. **转置回来**：从 [batch, n_heads, seq, head_dim] 转回 [batch, seq, n_heads, head_dim]
2. **拼接多头**：reshape 成 [batch, seq, n_heads * head_dim] = [batch, seq, hidden_size]
3. **线性投影**：通过 o_proj 线性层输出

#### 代码位置引用

- o_proj 定义：[model/model_minimind.py:113](file:///home/zhangxm/model_minimind/model/model_minimind.py#L113-L113)
- 前向传播输出：[model/model_minimind.py:145-147](file:///home/zhangxm/model_minimind/model/model_minimind.py#L145-L147)

 `python
self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
# ...
output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
output = self.resid_dropout(self.o_proj(output))
 `

#### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| attn_output | [batch, n_heads, seq_len, head_dim] | float32/bf16 | 多头注意力输出 |
| attn_output（转置后） | [batch, seq_len, n_heads, head_dim] | float32/bf16 | 调整维度顺序 |
| attn_output（拼接后） | [batch, seq_len, n_heads * head_dim] | float32/bf16 | 多头拼接 |
| output | [batch, seq_len, hidden_size] | float32/bf16 | o_proj 投影后的最终输出 |

**以 MiniMind 默认配置为例**：

| 阶段 | Shape |
|------|-------|
| 注意力输出 | [1, 8, 100, 96] |
| 转置后 | [1, 100, 8, 96] |
| 拼接后 | [1, 100, 768]（8 x 96 = 768） |
| o_proj 输出 | [1, 100, 768] |

---

## 3.3 KV Cache 原理

### 什么是 KV Cache

KV Cache 是自回归生成中的一种优化技术：保存之前所有 token 的 K 和 V，生成下一个 token 时直接复用，不需要重新计算。

### 为什么需要

自回归生成是逐 token 进行的：
- 第 1 步：输入 [token_0] -> 生成 token_1
- 第 2 步：输入 [token_0, token_1] -> 生成 token_2
- 第 3 步：输入 [token_0, token_1, token_2] -> 生成 token_3
- ...

如果不使用 KV Cache，每一步都要重新计算所有历史 token 的 K 和 V，这是非常浪费的。使用 KV Cache 后，每一步只需要计算新 token 的 K 和 V，然后拼接到历史缓存上。

### past_key_value 的数据结构

KV Cache 通常是一个元组列表，每层对应一个 (k_cache, v_cache) 元组：

 `python
past_key_values = [
    (k_layer_0, v_layer_0),  # 第 0 层的 KV cache
    (k_layer_1, v_layer_1),  # 第 1 层的 KV cache
    ...
    (k_layer_N, v_layer_N),  # 第 N 层的 KV cache
]
 `

每层 KV cache 的形状：
- k_cache: [batch, seq_len, n_kv_heads, head_dim]（MiniMind 格式）
- v_cache: [batch, seq_len, n_kv_heads, head_dim]

### 代码位置引用

- KV Cache 拼接逻辑：[model/model_minimind.py:127-131](file:///home/zhangxm/model_minimind/model/model_minimind.py#L127-L131)

 `python
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)
    xv = torch.cat([past_key_value[1], xv], dim=1)
past_kv = (xk, xv) if use_cache else None
 `

### 为什么能加速

| 指标 | 无 KV Cache | 有 KV Cache |
|------|------------|------------|
| 每步计算量 | O(t^2 x d)（t 是当前序列长度） | O(t x d)（只算新 token） |
| 每步时间 | 随生成线性增加 | 基本恒定 |
| 第 100 步相对速度 | 1x | ~100x（估算） |

### 输入/输出张量说明

**prefill 阶段（第 1 步，输入完整 prompt）**：

| 张量 | Shape | 含义 |
|------|-------|------|
| input_ids | [batch, seq_len] | 完整 prompt |
| past_key_values | None | 没有历史缓存 |
| output_k | [batch, seq_len, n_kv_heads, head_dim] | 本层 K（用于缓存） |
| output_v | [batch, seq_len, n_kv_heads, head_dim] | 本层 V（用于缓存） |

**decode 阶段（后续步，输入 1 个新 token）**：

| 张量 | Shape | 含义 |
|------|-------|------|
| input_ids | [batch, 1] | 新生成的 1 个 token |
| past_k | [batch, past_len, n_kv_heads, head_dim] | 历史 K cache |
| past_v | [batch, past_len, n_kv_heads, head_dim] | 历史 V cache |
| new_k | [batch, 1, n_kv_heads, head_dim] | 新 token 的 K |
| new_v | [batch, 1, n_kv_heads, head_dim] | 新 token 的 V |
| full_k | [batch, past_len+1, n_kv_heads, head_dim] | 拼接后的完整 K |
| full_v | [batch, past_len+1, n_kv_heads, head_dim] | 拼接后的完整 V |

**拼接方式**：在 seq_len 维度上拼接（dim=1）
 `python
xk = torch.cat([past_key_value[0], xk], dim=1)
xv = torch.cat([past_key_value[1], xv], dim=1)
 `

---

## 3.4 完整前向传播数据流

下面以 MiniMind 默认配置（batch=1, seq_len=100, hidden_size=768, n_heads=8, n_kv_heads=4, head_dim=96）为例，展示从输入 x 到输出 output 的完整 shape 变化路径：

| 步骤 | 操作 | 张量 | Shape |
|------|------|------|-------|
| 0 | 输入 | x | [1, 100, 768] |
| 1 | Q 线性投影 | xq | [1, 100, 768] |
| 1 | K 线性投影 | xk | [1, 100, 384] |
| 1 | V 线性投影 | xv | [1, 100, 384] |
| 1 | Q reshape 成多头 | xq | [1, 100, 8, 96] |
| 1 | K reshape 成多头 | xk | [1, 100, 4, 96] |
| 1 | V reshape 成多头 | xv | [1, 100, 4, 96] |
| 2 | Q RMSNorm | xq | [1, 100, 8, 96] |
| 2 | K RMSNorm | xk | [1, 100, 4, 96] |
| 3 | RoPE 旋转 | xq | [1, 100, 8, 96] |
| 3 | RoPE 旋转 | xk | [1, 100, 4, 96] |
| 4 | repeat_kv 扩展 K | xk | [1, 100, 8, 96] |
| 4 | repeat_kv 扩展 V | xv | [1, 100, 8, 96] |
| 5 | 转置 Q | q | [1, 8, 100, 96] |
| 5 | 转置 K | k | [1, 8, 100, 96] |
| 5 | 转置 V | v | [1, 8, 100, 96] |
| 5 | Q @ K^T（点积） | scores | [1, 8, 100, 100] |
| 5 | 除以 sqrt(d_k) | scores | [1, 8, 100, 100] |
| 6 | 加 causal mask | scores | [1, 8, 100, 100] |
| 5 | softmax | attn_weights | [1, 8, 100, 100] |
| 5 | @ V | output | [1, 8, 100, 96] |
| 8 | 转置回来 | output | [1, 100, 8, 96] |
| 8 | 拼接多头 | output | [1, 100, 768] |
| 8 | o_proj 线性投影 | output | [1, 100, 768] |

**最终输出 shape 与输入 x 相同**，都是 [batch, seq_len, hidden_size]。这是残差连接（Residual Connection）的要求——Attention 输出要和输入相加，所以维度必须一致。

---

## 小结

本章详细拆解了注意力机制的 8 个关键步骤：

| 步骤 | 名称 | 作用 |
|------|------|------|
| 1 | Q/K/V 线性投影 | 将输入映射为查询、键、值三个向量 |
| 2 | QK-Norm | 归一化 Q 和 K，稳定训练 |
| 3 | RoPE 旋转位置编码 | 注入位置信息 |
| 4 | GQA repeat_kv | 扩展 KV 头数以匹配 Q |
| 5 | Scaled Dot-Product | 计算注意力权重并加权求和 |
| 6 | Causal Mask | 防止看到未来 token |
| 7 | Flash Attention | 加速计算、节省显存 |
| 8 | o_proj 输出投影 | 拼接多头并投影回 hidden_size |

注意力机制是 Transformer 的核心，理解它的每一步对于深入掌握大语言模型至关重要。下一章我们将介绍前馈神经网络（MLP）和 Transformer Block 的整体结构。
