# 第 2 章 基础组件

本章介绍 MiniMind 模型的三个核心基础组件：Token Embedding（词嵌入层）、RMSNorm（均方根层归一化）和 RoPE（旋转位置编码）。这些组件是 Transformer 架构的基石，理解它们的工作原理对于深入掌握大语言模型至关重要。

---

## 2.1 Token Embedding（词嵌入层）

### 原理说明

Token Embedding（词嵌入层）是大语言模型的入口，负责将离散的 token id（整数）映射为连续的高维向量表示。

在自然语言处理中，文本首先被分词器（Tokenizer）转换为一个个 token，每个 token 对应一个整数 id。但这些整数 id 本身没有语义信息，只是离散的标识符。词嵌入层通过一张可学习的查找表（Lookup Table），将每个 token id 映射为一个固定维度的稠密向量，使得语义相近的 token 在向量空间中距离也更近。

### 代码位置引用

词嵌入层定义在 `MiniMindModel` 类的 `__init__` 方法中：

- 代码链接：[model/model_minimind.py:218](file:///home/zhangxm/model_minimind/model/model_minimind.py#L218-L218)
- 前向传播使用位置：[model/model_minimind.py:232](file:///home/zhangxm/model_minimind/model/model_minimind.py#L232-L232)

```python
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
```

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| input_ids | [batch_size, seq_len] | long | 输入 token 的整数 id |
| hidden_states | [batch_size, seq_len, hidden_size] | float32/bf16 | 输出词向量 |

### 关键计算逻辑解释

词嵌入层的核心操作是**查表（Embedding Lookup）**：

1. **初始化**：创建一个形状为 `[vocab_size, hidden_size]` 的可学习权重矩阵
2. **前向传播**：根据输入的 `input_ids`，从权重矩阵中取出对应行的向量
3. **输出**：每个 token id 被替换为对应的 `hidden_size` 维向量

**权重绑定（Weight Tying）**：

MiniMind 默认启用 `tie_word_embeddings=True`，即将语言模型输出层（lm_head）的权重与词嵌入层的权重共享：

- 代码位置：[model/model_minimind.py:262](file:///home/zhangxm/model_minimind/model/model_minimind.py#L262-L262)
- 绑定关系：`lm_head.weight = embed_tokens.weight`

权重绑定的优势：
- **减少参数量**：省去了 lm_head 的独立权重，约减少 vocab_size × hidden_size 个参数
- **正则化效果**：共享约束可以防止过拟合
- **对称性**：输入和输出使用同一语义空间

**MiniMind 默认配置**：
- `vocab_size = 6400`：词表大小
- `hidden_size = 768`：词向量维度

---

## 2.2 RMSNorm（均方根层归一化）

### 原理说明

RMSNorm（Root Mean Square Layer Normalization，均方根层归一化）是一种轻量级的归一化方法。它只根据张量的均方根进行缩放，不减去均值，因此比标准的 LayerNorm 计算更快、参数更少。

### 与 LayerNorm 的区别

| 方法 | 公式 | 可学习参数 | 计算量 |
|------|------|-----------|--------|
| LayerNorm | `(x - mean) / std * gamma + beta` | gamma, beta | 较大（需算均值） |
| RMSNorm | `x / rms * gamma` | gamma | 较小（只需算均方根） |

**为什么 LLaMA 系列用 RMSNorm？**
- **计算更快**：省去了均值计算和偏置项
- **效果相当**：在 Transformer 架构中，RMSNorm 与 LayerNorm 性能接近
- **更稳定**：只关注向量的模长归一化，简化了归一化逻辑

### 代码位置引用

RMSNorm 类定义在模型文件开头：

- 代码链接：[model/model_minimind.py:55-66](file:///home/zhangxm/model_minimind/model/model_minimind.py#L55-L66)

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)
```

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| x | [batch_size, seq_len, hidden_size] | float32/bf16 | 输入张量 |
| output | [batch_size, seq_len, hidden_size] | float32/bf16 | 归一化后的张量 |

### 关键计算逻辑解释

RMSNorm 的计算分为两步：

**步骤 1：计算均方根（RMS）**

rms = sqrt(mean(x^2) + eps)

代码实现：
```python
x.pow(2).mean(-1, keepdim=True)  # 计算最后一维的平方均值
torch.rsqrt(...)                  # 计算倒数平方根（rsqrt = 1/sqrt）
```

**步骤 2：归一化并缩放**

output = weight * x / rms

代码实现：
```python
self.weight * self.norm(x.float())
```

**关键细节**：
- `eps=1e-6`：防止除零的极小值（MiniMind 配置中为 1e-6）
- `.float()`：先转 float32 计算保证精度，再转回原 dtype
- `keepdim=True`：保持维度以便广播乘法

RMSNorm 不改变数据的均值，只对模长进行归一化，这是它与 LayerNorm 的核心区别。

---

## 2.3 RoPE 旋转位置编码

### 原理说明

RoPE（Rotary Position Embedding，旋转位置编码）是一种通过**旋转**方式给 Query 和 Key 向量注入位置信息的位置编码方法。它的核心思想是：将每个位置的 Q/K 向量按照特定的频率旋转不同的角度，使得注意力机制中的点积天然具有相对位置感知能力。

### RoPE 的优势

1. **相对位置不变性**：位置 i 和位置 j 的 Q、K 向量点积只与相对位置 i-j 有关，与绝对位置无关
2. **长上下文外推能力强**：通过频率缩放技术（如 YaRN）可以支持比训练时更长的上下文
3. **实现简洁**：只需要对 Q、K 施加旋转变换，不改变模型主体结构

### 代码位置引用

RoPE 的实现分为两个函数：

**1. 预计算 cos/sin 表**
- 函数名：`precompute_freqs_cis`
- 代码链接：[model/model_minimind.py:68-85](file:///home/zhangxm/model_minimind/model/model_minimind.py#L68-L85)

**2. 应用旋转位置编码**
- 函数名：`apply_rotary_pos_emb`
- 代码链接：[model/model_minimind.py:87-93](file:///home/zhangxm/model_minimind/model/model_minimind.py#L87-L93)

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| q / k | [batch, seq, num_heads, head_dim] | float32/bf16 | 查询/键向量 |
| cos / sin | [seq_len, head_dim] | float32 | 预计算的旋转矩阵元素 |
| q_embed / k_embed | [batch, seq, num_heads, head_dim] | float32/bf16 | 旋转后的 Q/K |

### 关键计算逻辑解释

#### 核心思想

将每个位置的向量看作复数平面上的向量，通过旋转不同的角度来注入位置信息。

对于位置 m 处的向量 q，其二元组 (q0, q1) 旋转后的结果为：

q0' = q0 * cos(m*theta) - q1 * sin(m*theta)
q1' = q0 * sin(m*theta) + q1 * cos(m*theta)

#### 实现步骤

**步骤 1：预计算频率（precompute_freqs_cis）**

1. 生成频率序列：theta_i = rope_base^(-2i/d)，其中 i = 0, 1, ..., d/2-1
2. 对每个位置 m，计算 m * theta_i
3. 预先计算好 cos 和 sin 值，存成表格供 forward 时直接查表

代码中的关键操作：
```python
freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
t = torch.arange(end)
freqs = torch.outer(t, freqs).float()  # [seq_len, dim//2]
freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
```

**步骤 2：应用旋转（apply_rotary_pos_emb）**

```python
def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
```

这个实现巧妙地利用了 `rotate_half` 函数，将向量的后半部分取负并与前半部分拼接，等效于复数乘法的虚部。

#### 相对位置不变性验证

对于位置 m 的 Q 和位置 n 的 K，经过 RoPE 旋转后的点积：

<R_m q, R_n k> = q^T R_m^T R_n k = q^T R_{n-m} k

点积结果只依赖于相对位置 n-m，与绝对位置 m、n 无关。这就是 RoPE 相对位置不变性的数学基础。

### YaRN 长上下文外推（可选）

MiniMind 支持通过 YaRN（Yet another RoPE extensioN）方法进行长上下文外推：

- 配置项：`inference_rope_scaling = True` 时启用
- 代码位置：`precompute_freqs_cis` 函数中的 `rope_scaling` 分支
- 原理：对低频部分进行频率缩放，高频部分保持不变，通过平滑过渡实现长上下文扩展
- 参数：`factor`（缩放因子）、`beta_fast`/`beta_slow`（过渡带控制）

启用后，模型可以在推理时处理远超训练长度的上下文。

---

## 小结

本章介绍的三个基础组件各司其职：

| 组件 | 作用 | 所在位置 |
|------|------|---------|
| Token Embedding | 离散 token id -> 连续向量 | 模型输入层 |
| RMSNorm | 归一化，稳定训练 | 每层 Attention/MLP 前后 |
| RoPE | 注入位置信息 | Attention 中的 Q/K 上 |

理解这三个组件是深入学习 Transformer 和大语言模型的基础。下一章我们将介绍注意力机制（Attention）的实现细节。
