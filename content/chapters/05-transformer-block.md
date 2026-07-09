# 第 5 章 Transformer Block 与整体前向

本章把前面几章介绍的组件（Token Embedding、RMSNorm、RoPE、Attention、FFN/MoE）组装成完整的 Transformer 模型。我们将从单个 Decoder Block 的 Pre-Norm 结构出发，讲解 MiniMindModel 如何堆叠多层 Block，再到 MiniMindForCausalLM 如何输出 logits 并计算交叉熵损失。最后给出从 input_ids 到 loss 的完整数据流。

---

## 5.1 Pre-Norm 结构

### 原理说明

Transformer 有两种归一化放置方式：

- **Post-Norm**（原始 Transformer）：先做子层计算，再做归一化。`output = Norm(x + Sublayer(x))`
- **Pre-Norm**（GPT、LLaMA 系）：先做归一化，再做子层计算。`output = x + Sublayer(Norm(x))`

MiniMind 采用 **Pre-Norm** 结构，这是现代大语言模型的通用做法。

### Pre-Norm 的优势

| 方面 | Post-Norm | Pre-Norm |
|------|-----------|----------|
| 归一化时机 | 子层之后 | 子层之前 |
| 残差路径 | 经过 Norm | 不经过 Norm（直连） |
| 训练稳定性 | 较差（深层需 warmup） | 较好（残差路径干净） |
| 深层模型 | 难训练 | 易训练 |

**Pre-Norm 训练更稳定的原因**：残差主路径上没有归一化操作，梯度可以无衰减地直通到模型底部，深层模型也能有效训练。Post-Norm 中残差路径经过 Norm，梯度会被缩放，深层容易出现梯度消失。

### Pre-Norm Block 的计算公式

一个 Decoder Block 包含两个子层（Attention 和 FFN），每个子层都套用 Pre-Norm 残差：

```
# 子层 1：Attention
h = x + Attention(Norm1(x))

# 子层 2：FFN
out = h + FFN(Norm2(h))
```

其中 Norm1 = input_layernorm，Norm2 = post_attention_layernorm，是两个独立的 RMSNorm。

---

## 5.2 MiniMindBlock：单个 Decoder 层

### 原理说明

MiniMindBlock 是 Transformer 的基本构建单元。每个 Block 包含：
1. 一个 **Self-Attention** 子层（第 3 章）
2. 一个 **FFN/MoE** 子层（第 4 章）
3. 两个 **RMSNorm**（分别用于两个子层的 Pre-Norm）
4. 两个 **残差连接**（分别绕过两个子层）

数据流：
```
输入 hidden_states
    │
    ├──▶ input_layernorm ──▶ Attention ──┐
    │                                     ▼
    └─────────────── 残差相加 (+) ◀────────┘
                    │
                    ├──▶ post_attention_layernorm ──▶ MLP ──┐
                    │                                       ▼
                    └─────────────── 残差相加 (+) ◀──────────┘
                                    │
                                    ▼
                                输出 hidden_states
```

### 代码位置引用

MiniMindBlock 类定义在模型文件中：

- 代码链接：[model/model_minimind.py:194-211](file:///home/zhangxm/model_minimind/model/model_minimind.py#L194-L211)

```python
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # Pre-Norm Transformer：先归一化再进入子层，残差连接负责保留原始信息。
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value
```

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| hidden_states（输入） | [batch, seq_len, hidden_size] | float32/bf16 | 来自上一层（或 embedding）的隐藏状态 |
| position_embeddings | (cos, sin) 各 [seq_len, head_dim] | float32 | RoPE 预计算的 cos/sin |
| past_key_value | (k_cache, v_cache) 各 [batch, past_len, n_kv_heads, head_dim] | float32/bf16 | 历史 KV cache（推理时） |
| present_key_value | (k_full, v_full) 各 [batch, past_len+seq, n_kv_heads, head_dim] | float32/bf16 | 更新后的 KV cache |
| hidden_states（输出） | [batch, seq_len, hidden_size] | float32/bf16 | 经过 Block 处理后的隐藏状态 |

**注意**：Block 的输入和输出 shape 完全一致，都是 [batch, seq_len, hidden_size]，这是残差连接的要求。

### 关键计算逻辑解释

MiniMindBlock 的前向分为两段残差：

**第 1 段：Attention 残差**

```python
residual = hidden_states                              # 保存原始输入
hidden_states, present_key_value = self.self_attn(
    self.input_layernorm(hidden_states),              # Pre-Norm：先归一化
    position_embeddings,
    past_key_value, use_cache, attention_mask
)
hidden_states += residual                             # 残差相加
```

- `input_layernorm` 对输入做 RMSNorm 归一化（不改变 shape）
- 归一化后的向量送入 Attention 子层
- Attention 输出与原始输入 `residual` 相加（残差连接）
- 残差连接让梯度可以直通，同时保留原始信息

**第 2 段：FFN 残差**

```python
hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
```

- `post_attention_layernorm` 对 Attention 残差后的结果做归一化
- 归一化后的向量送入 FFN（或 MoE）子层
- FFN 输出与输入相加（残差连接）
- 这一行等价于：
  ```python
  residual2 = hidden_states
  h_norm = self.post_attention_layernorm(hidden_states)
  h_mlp = self.mlp(h_norm)
  hidden_states = residual2 + h_mlp
  ```

**为什么需要两个独立的 Norm？**

input_layernorm 和 post_attention_layernorm 是两个独立的 RMSNorm，各有自己的 weight 参数。它们分别归一化 Attention 和 FFN 的输入，因为两个子层对输入分布的要求不同，独立归一化更灵活。

---

## 5.3 MiniMindModel：模型主体

### 原理说明

MiniMindModel 是 Transformer 的主体，负责把 token id 序列转换为隐藏状态序列。它包含：

1. **embed_tokens**：词嵌入层，token id → 向量
2. **layers**：N 层 MiniMindBlock 堆叠
3. **norm**：最终归一化层（Final Norm）
4. **freqs_cos / freqs_sin**：预计算的 RoPE cos/sin 表（register_buffer，不参与训练）

### 代码位置引用

MiniMindModel 类定义在模型文件中：

- 代码链接：[model/model_minimind.py:213-252](file:///home/zhangxm/model_minimind/model/model_minimind.py#L213-L252)

```python
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        # token id 先查 embedding 表变成连续向量，再送入多层 Decoder Block。
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        # 如果有 KV cache，当前位置从历史长度 start_pos 开始，而不是从 0 重新编号。
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        # Dense 模型 aux_loss 为 0；MoE 模型会累加每层路由均衡损失。
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss
```

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| input_ids | [batch, seq_len] | long | 输入 token id |
| past_key_values | List[(k, v)] 或 None | - | 每层的历史 KV cache |
| hidden_states | [batch, seq_len, hidden_size] | float32/bf16 | 最终隐藏状态 |
| presents | List[(k, v)] | - | 每层更新后的 KV cache |
| aux_loss | 标量 [] | float32 | MoE 辅助损失（dense 模型为 0） |

### 关键计算逻辑解释

**步骤 1：Embedding 查表**

```python
hidden_states = self.dropout(self.embed_tokens(input_ids))
```

- `embed_tokens` 把 [batch, seq_len] 的 token id 映射为 [batch, seq_len, hidden_size] 的向量
- `dropout` 在训练时随机丢弃部分元素（推理时关闭）

**步骤 2：确定起始位置（KV Cache 场景）**

```python
past_key_values = past_key_values or [None] * len(self.layers)
start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
```

- 如果没有 KV cache（首次 prefill 或训练），`start_pos = 0`，位置编号从 0 开始
- 如果有 KV cache（decode 阶段），`start_pos = 历史 KV cache 的长度`，新 token 的位置从 start_pos 开始编号
- 这保证 RoPE 位置编码连续：prefill 时位置 0..L-1，decode 时位置 L, L+1, ...

**步骤 3：RoPE 位置编码切片**

```python
position_embeddings = (
    self.freqs_cos[start_pos:start_pos + seq_length],
    self.freqs_sin[start_pos:start_pos + seq_length]
)
```

- `freqs_cos` / `freqs_sin` 是预计算的 [max_position_embeddings, head_dim] 表
- 根据当前 start_pos 和 seq_length 切片出对应位置的 cos/sin
- 这些 cos/sin 会传给每层的 Attention，用于 RoPE 旋转

**步骤 4：逐层前向**

```python
presents = []
for layer, past_key_value in zip(self.layers, past_key_values):
    hidden_states, present = layer(
        hidden_states, position_embeddings,
        past_key_value=past_key_value, use_cache=use_cache, attention_mask=attention_mask
    )
    presents.append(present)
```

- 遍历 num_hidden_layers 层 MiniMindBlock
- 每层的输出 hidden_states 作为下一层的输入
- 每层的 KV cache (`present`) 收集到 `presents` 列表，作为整体的 past_key_values 返回
- `position_embeddings` 对每层是相同的（因为每层用同样的 RoPE 位置编码）

**步骤 5：最终归一化**

```python
hidden_states = self.norm(hidden_states)
```

- 经过所有 Block 后，再做一次 RMSNorm 归一化
- 这是 Pre-Norm 结构的标准做法：因为残差路径没有归一化，最后一层的输出需要补一个 Norm

**步骤 6：累加 MoE 辅助损失**

```python
aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
```

- 遍历所有层，如果是 MoE 层（`isinstance(l.mlp, MOEFeedForward)`），收集其 aux_loss
- 求和得到所有 MoE 层的辅助损失总和
- Dense 模型（use_moe=False）时列表为空，返回 0

### position_embeddings 缓存机制

MiniMind 把 RoPE 的 cos/sin 表预计算并缓存为 buffer：

```python
freqs_cos, freqs_sin = precompute_freqs_cis(
    dim=config.head_dim,
    end=config.max_position_embeddings,    # 默认 32768
    rope_base=config.rope_theta,           # 默认 1e6
    rope_scaling=config.rope_scaling       # YaRN 配置（推理外推时启用）
)
self.register_buffer("freqs_cos", freqs_cos, persistent=False)
self.register_buffer("freqs_sin", freqs_sin, persistent=False)
```

- `persistent=False`：不保存到 state_dict（因为可以重新计算）
- 预计算一次，forward 时只需切片，避免每次重复计算
- 如果 `freqs_cos[0,0] == 0`（meta device 初始化丢失 buffer），会重新计算

---

## 5.4 MiniMindForCausalLM：语言模型头

### 原理说明

MiniMindForCausalLM 在 MiniMindModel 之上添加：
1. **lm_head**：线性层，把 hidden_size 投影到 vocab_size，得到每个 token 在词表上的 logits
2. **权重绑定**（可选）：lm_head 的权重与 embed_tokens 共享
3. **交叉熵损失**：训练时根据 labels 计算损失

### 代码位置引用

MiniMindForCausalLM 类定义在模型文件中：

- 类定义：[model/model_minimind.py:254-264](file:///home/zhangxm/model_minimind/model/model_minimind.py#L254-L264)
- forward 方法：[model/model_minimind.py:265-278](file:///home/zhangxm/model_minimind/model/model_minimind.py#L265-L278)

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        # 推理时只需要最后几个 token 的 logits，logits_to_keep 可以减少无用计算和显存占用。
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            # 因果语言模型训练目标：第 t 个位置预测第 t+1 个 token；-100 会被 CE 忽略。
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
```

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| input_ids | [batch, seq_len] | long | 输入 token id |
| labels | [batch, seq_len] | long | 目标 token id（训练时） |
| hidden_states | [batch, seq_len, hidden_size] | float32/bf16 | 模型主体的隐藏状态 |
| logits | [batch, keep_len, vocab_size] | float32/bf16 | 每个 token 在词表上的预测分数 |
| loss | 标量 [] | float32 | 交叉熵损失 |
| aux_loss | 标量 [] | float32 | MoE 辅助损失 |

### 关键计算逻辑解释

**步骤 1：调用模型主体**

```python
hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
```

- 调用 MiniMindModel 得到隐藏状态、KV cache、辅助损失

**步骤 2：logits_to_keep 机制**

```python
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
logits = self.lm_head(hidden_states[:, slice_indices, :])
```

- `logits_to_keep` 控制只保留最后若干位置的 logits
- 训练时 `logits_to_keep=0`，保留全部位置（slice(-0, None) 即 slice(0, None)，全部保留）
- 推理生成时只关心最后一个 token 的 logits，可设 `logits_to_keep=1`，大幅减少 lm_head 计算量和显存

**注意**：`slice(-0, None)` 在 Python 中等价于 `slice(0, None)`（因为 -0 == 0），所以 logits_to_keep=0 时保留全部。这是 MiniMind 的一个巧妙实现。

**步骤 3：lm_head 投影**

```python
self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
```

- 把 hidden_size（768）投影到 vocab_size（6400）
- 输出 logits 形状 [batch, seq_len, vocab_size]，表示每个位置对词表中每个 token 的预测分数

**步骤 4：权重绑定（Weight Tying）**

```python
if self.config.tie_word_embeddings:
    self.model.embed_tokens.weight = self.lm_head.weight
```

- 默认 `tie_word_embeddings=True`
- lm_head 的权重与 embed_tokens 共享同一份参数
- `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}` 告诉 HuggingFace 框架这两个权重是绑定的，加载/保存时正确处理

权重绑定的好处：
- **省参数**：减少 vocab_size × hidden_size = 6400 × 768 ≈ 490 万参数
- **语义一致**：输入和输出在同一向量空间，token 的"编码"和"解码"用同一张表
- **正则化**：共享约束防止过拟合

### 交叉熵损失：标签错位对齐

```python
if labels is not None:
    # 因果语言模型训练目标：第 t 个位置预测第 t+1 个 token；-100 会被 CE 忽略。
    x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
    loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

**为什么需要错位？**

因果语言模型的训练目标是：根据前 t 个 token 预测第 t+1 个 token。

- 模型在第 t 个位置（输入 token t）输出的 logits，应该预测第 t+1 个 token
- 所以用 `logits[..., :-1, :]`（去掉最后一个位置）与 `labels[..., 1:]`（去掉第一个位置）对齐

**错位示意**（seq_len=4）：

```
位置:      0      1      2      3
输入:     tok0   tok1   tok2   tok3
logits:   L0     L1     L2     L3      # L0 预测 tok1, L1 预测 tok2, ...

logits[..., :-1, :]  = [L0, L1, L2]    # 去掉最后
labels[..., 1:]      = [t1, t2, t3]    # 去掉最前

对齐: L0 → t1, L1 → t2, L2 → t3
```

- L0（输入 tok0 时的输出）应该预测 tok1
- L1（输入 tok1 时的输出）应该预测 tok2
- L2（输入 tok2 时的输出）应该预测 tok3
- L3（输入 tok3 时的输出）预测的是未来的 tok4，训练时没有标签，丢弃

**ignore_index=-100**：

- labels 中值为 -100 的位置不参与损失计算
- 常用于屏蔽 padding token 或仅对部分位置计算损失（如指令微调时只对回答部分算损失）

**Cross Entropy 计算**：

```python
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

- `x.view(-1, vocab_size)`：把 [batch, seq-1, vocab] 展平为 [batch × (seq-1), vocab]
- `y.view(-1)`：把 [batch, seq-1] 展平为 [batch × (seq-1)]
- 对每个位置计算 softmax + NLL loss，再求平均

---

## 5.5 完整前向数据流

下面以 MiniMind 默认配置（batch=2, seq_len=100, hidden_size=768, num_hidden_layers=8, vocab_size=6400）为例，展示从 input_ids 到 loss 的完整 shape 变化路径：

| 步骤 | 操作 | 张量 | Shape |
|------|------|------|-------|
| 0 | 输入 | input_ids | [2, 100] |
| 0 | 输入 | labels | [2, 100] |
| 1 | embed_tokens 查表 | hidden_states | [2, 100, 768] |
| 2 | RoPE 切片 | cos / sin | [100, 96] |
| 3 | Block 0 ~ Block 7（8 层） | hidden_states | [2, 100, 768] |
| 3 | 每层 KV cache | present_k/v | [2, 100, 4, 96] |
| 4 | 最终 norm | hidden_states | [2, 100, 768] |
| 5 | logits_to_keep=0 切片 | hidden_states | [2, 100, 768] |
| 6 | lm_head 投影 | logits | [2, 100, 6400] |
| 7 | logits 错位 | logits[..., :-1, :] | [2, 99, 6400] |
| 7 | labels 错位 | labels[..., 1:] | [2, 99] |
| 8 | cross_entropy | loss | 标量 [] |

**单层 Block 内部数据流**（以 Block 0 为例）：

| 步骤 | 操作 | 张量 | Shape |
|------|------|------|-------|
| 0 | 输入 | hidden_states | [2, 100, 768] |
| 1 | input_layernorm | h_norm1 | [2, 100, 768] |
| 2 | Attention | attn_out | [2, 100, 768] |
| 3 | 残差相加 | hidden_states | [2, 100, 768] |
| 4 | post_attention_layernorm | h_norm2 | [2, 100, 768] |
| 5 | MLP (SwiGLU) | mlp_out | [2, 100, 768] |
| 6 | 残差相加 | hidden_states | [2, 100, 768] |

**训练与推理的差异**：

| 方面 | 训练 | 推理（生成） |
|------|------|------------|
| use_cache | 通常 False | 通常 True |
| past_key_values | None | 非空（每步更新） |
| seq_len | 完整序列 | 1（decode 阶段） |
| labels | 提供 | 不提供 |
| loss | 计算并反传 | 不计算 |
| logits_to_keep | 0（全部） | 1（仅最后） |

### MoE 模型的损失总和

对于 MoE 模型（use_moe=True），总损失 = 交叉熵损失 + 辅助损失：

```python
# 训练时（外部）
total_loss = loss + aux_loss
# loss: 主交叉熵损失
# aux_loss: 所有 MoE 层的负载均衡损失之和（乘以 router_aux_loss_coef=5e-4）
```

Dense 模型（use_moe=False）的 aux_loss 恒为 0，总损失即交叉熵损失。

---

## 小结

本章把所有组件组装成完整的 Transformer 模型：

| 组件 | 作用 | 代码位置 |
|------|------|---------|
| MiniMindBlock | 单个 Decoder 层（Pre-Norm + 双残差） | L194-211 |
| MiniMindModel | 模型主体（embed + N 层 Block + norm） | L213-252 |
| MiniMindForCausalLM | 语言模型头（lm_head + 损失计算） | L254-278 |

核心要点：

1. **Pre-Norm 结构**：`h = x + Sublayer(Norm(x))`，残差路径不经过 Norm，训练更稳定
2. **MiniMindBlock**：两个子层（Attention + FFN）各自配一个 RMSNorm 和残差连接
3. **MiniMindModel**：embed → 8 层 Block 堆叠 → final norm，预计算 RoPE cos/sin 缓存，KV cache 逐层传递
4. **权重绑定**：lm_head 与 embed_tokens 共享权重，省参数且语义一致
5. **logits_to_keep**：推理时只保留最后若干 logits，省显存
6. **标签错位**：`logits[:, :-1]` 对 `labels[:, 1:]`，实现"用第 t 个位置预测第 t+1 个 token"的因果训练目标

下一章我们将介绍模型推理生成算法，包括自回归生成、温度采样、Top-K/Top-P 采样等。
