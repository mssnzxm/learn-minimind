# 第 10 章 推理生成算法

本章讲解 MiniMind 的推理生成算法，即模型如何从一段 prompt 自回归地生成后续文本。我们将拆解 `generate` 方法中的每一步：自回归循环、KV Cache 复用、温度采样、Top-K 与 Top-P（nucleus）采样、重复惩罚、EOS 批量处理，以及贪心解码。每一步都结合实际代码与张量 shape 进行说明。

---

## 10.1 自回归生成概述

### 原理说明

自回归（Autoregressive）生成是大语言模型推理的核心范式：每一步根据已有 token 序列预测下一个 token，把新 token 追加到序列末尾，再预测下一个，如此循环直到满足停止条件。

生成过程分为两个阶段：

1. **Prefill（预填充）阶段**：输入完整 prompt，一次性计算所有 prompt token 的 KV 并缓存。这一步相当于一次完整的 forward（seq_len = prompt 长度）。
2. **Decode（解码）阶段**：每次只输入 1 个新 token（上一步生成的），利用 KV Cache 避免重算历史，逐步生成后续 token。每步 seq_len = 1。

### 停止条件

生成的停止条件通常有：
- 生成了 EOS（End of Sequence）token
- 达到最大生成长度 `max_new_tokens`
- 遇到特定停止词（stop words）

### 代码位置引用

generate 方法定义在 MiniMindForCausalLM 中：

- 代码链接：[model/model_minimind.py:279-313](file:///home/zhangxm/model_minimind/model/model_minimind.py#L279-L313)

```python
@torch.inference_mode()
def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85,
             top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True,
             num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
    input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
    attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
    past_key_values = kwargs.pop("past_key_values", None)
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    if streamer: streamer.put(input_ids.cpu())
    for _ in range(max_new_tokens):
        past_len = past_key_values[0][0].shape[1] if past_key_values else 0
        # 有 cache 时只喂新增 token；无 cache 时才会反复喂完整序列。
        outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
        logits = outputs.logits[:, -1, :] / temperature
        if repetition_penalty != 1.0:
            for i in range(input_ids.shape[0]):
                seen = torch.unique(input_ids[i]); score = logits[i, seen]
                logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
        if top_k > 0:
            logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
            mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
            logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
        if eos_token_id is not None:
            next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        past_key_values = outputs.past_key_values if use_cache else None
        if streamer: streamer.put(next_token.cpu())
        if eos_token_id is not None:
            finished |= next_token.squeeze(-1).eq(eos_token_id)
            if finished.all(): break
    if streamer: streamer.end()
    if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
    return input_ids
```

### 默认参数说明

| 参数 | 默认值 | 含义 |
|------|--------|------|
| max_new_tokens | 8192 | 最大生成 token 数 |
| temperature | 0.85 | 温度（越高分布越平坦） |
| top_p | 0.85 | nucleus 采样的累计概率阈值 |
| top_k | 50 | top-k 采样的候选数 |
| eos_token_id | 2 | 结束 token 的 id |
| do_sample | True | True 采样，False 贪心 |
| repetition_penalty | 1.0 | 重复惩罚系数（1.0 表示不惩罚） |
| num_return_sequences | 1 | 每条 prompt 生成的序列数 |
| use_cache | True | 是否使用 KV Cache |

---

## 10.2 自回归生成循环

### 原理说明

`generate` 的主体是一个 `for _ in range(max_new_tokens)` 循环，每轮：
1. 用当前 input_ids 调用 forward 得到 logits
2. 从 logits 采样出 next_token
3. 把 next_token 追加到 input_ids
4. 更新 KV cache
5. 检查是否全部序列都遇到 EOS

### 代码位置引用

循环主体：[model/model_minimind.py:286-310](file:///home/zhangxm/model_minimind/model/model_minimind.py#L286-L310)

### 关键计算逻辑解释

**步骤 1：确定本轮输入的 token 范围**

```python
past_len = past_key_values[0][0].shape[1] if past_key_values else 0
outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
```

- `past_len`：KV cache 中已保存的 token 数（取第 0 层 K cache 的 seq 维度长度）
- 首次循环（prefill）：`past_key_values=None`，`past_len=0`，输入完整 prompt `input_ids[:, 0:]`
- 后续循环（decode）：`past_len` 为已生成长度，只输入新增部分 `input_ids[:, past_len:]`（通常只有 1 个 token）
- 这是 KV Cache 的核心：避免重复 forward 历史 token

**步骤 2：更新 attention_mask**

```python
attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
```

- 每生成 1 个新 token，attention_mask 在末尾追加一个 1（表示该位置有效）
- 保持 attention_mask 长度与 input_ids 一致

**步骤 3：取最后一位 logits**

```python
logits = outputs.logits[:, -1, :] / temperature
```

- `outputs.logits` 形状 [batch, seq_len, vocab_size]
- 只取最后一个位置 `[:, -1, :]`，因为只有最后一个位置的 logits 是用来预测下一个 token 的
- 除以 temperature 做温度缩放（详见 10.4）

**步骤 4：采样下一 token**

经过 temperature、repetition_penalty、top_k、top_p 处理后：

```python
next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
```

- `do_sample=True`：从 softmax 概率分布中采样
- `do_sample=False`：取 argmax（贪心解码）

**步骤 5：追加新 token 并更新 cache**

```python
input_ids = torch.cat([input_ids, next_token], dim=-1)
past_key_values = outputs.past_key_values if use_cache else None
```

- 把 next_token 追加到 input_ids 末尾
- 保存 forward 返回的 past_key_values 供下一轮使用

### 输入/输出张量说明

**Prefill 阶段**（首轮）：

| 张量 | Shape | 含义 |
|------|-------|------|
| input_ids | [batch, prompt_len] | 完整 prompt |
| past_key_values | None | 无历史 cache |
| outputs.logits | [batch, prompt_len, vocab] | 所有位置 logits |
| 取最后一位 | [batch, vocab] | 用于预测下一个 token |
| past_key_values（返回） | List[(k, v)] 各 [batch, prompt_len, n_kv_heads, head_dim] | 缓存的 KV |

**Decode 阶段**（后续轮）：

| 张量 | Shape | 含义 |
|------|-------|------|
| input_ids[:, past_len:] | [batch, 1] | 仅新 token |
| past_key_values | List[(k, v)] 各 [batch, past_len, n_kv_heads, head_dim] | 历史 cache |
| outputs.logits | [batch, 1, vocab] | 当前位置 logits |
| 取最后一位 | [batch, vocab] | 用于预测下一个 token |
| past_key_values（返回） | List[(k, v)] 各 [batch, past_len+1, n_kv_heads, head_dim] | 更新后的 cache |

---

## 10.3 KV Cache 原理

### 原理说明

KV Cache 是自回归生成的关键优化。回顾第 3 章：注意力计算需要所有历史 token 的 K 和 V。如果不缓存，每生成一个新 token 都要重新计算所有历史 token 的 K/V，计算量随序列长度平方增长。

**KV Cache 的做法**：把每层 Attention 计算出的 K、V 缓存下来，下一步只计算新 token 的 K/V，拼接到缓存末尾。这样每步计算量从 O(L²) 降为 O(L)。

### 代码位置引用

- generate 中复用 cache：[model/model_minimind.py:287-288](file:///home/zhangxm/model_minimind/model/model_minimind.py#L287-L288)
- Attention 中拼接 cache：[model/model_minimind.py:127-131](file:///home/zhangxm/model_minimind/model/model_minimind.py#L127-L131)

### KV Cache 数据结构

```python
past_key_values = [
    (k_layer_0, v_layer_0),   # 第 0 层
    (k_layer_1, v_layer_1),   # 第 1 层
    ...
    (k_layer_7, v_layer_7),   # 第 7 层
]
```

每层 K/V cache 的 shape：

| 张量 | Shape | dtype |
|------|-------|-------|
| k_cache | [batch, cached_len, n_kv_heads, head_dim] | float32/bf16 |
| v_cache | [batch, cached_len, n_kv_heads, head_dim] | float32/bf16 |

以 MiniMind 默认配置（batch=1, n_kv_heads=4, head_dim=96）为例，缓存 100 个 token 时：

- 每层 K cache: [1, 100, 4, 96]，V cache 同形
- 8 层总共缓存: 8 × 2 × 1 × 100 × 4 × 96 = 614,400 个元素
- bf16 下约 1.2 MB

### KV Cache 的加速效果

| 指标 | 无 KV Cache | 有 KV Cache |
|------|------------|------------|
| 每步计算量 | O(L²·d)（重算所有历史） | O(L·d)（只算新 token） |
| 第 100 步相对速度 | 1x | ~100x |
| 显存占用 | 低（不存 cache） | 随序列长度线性增长 |

### 代码中的 cache 逻辑

在 Attention.forward 中：

```python
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)   # 拼接历史 K
    xv = torch.cat([past_key_value[1], xv], dim=1)   # 拼接历史 V
past_kv = (xk, xv) if use_cache else None
```

- 新 token 的 K/V 形状 [batch, 1, n_kv_heads, head_dim]
- 与历史 cache 在 seq 维度（dim=1）拼接，得到 [batch, past_len+1, n_kv_heads, head_dim]
- 拼接后的完整 K/V 用于注意力计算，Attention 输出的新 KV cache 通过 `past_kv` 返回

### start_pos 与位置编码

在 MiniMindModel.forward 中：

```python
start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
```

- 有 cache 时，新 token 的绝对位置从 `start_pos`（=历史长度）开始
- 例如已缓存 100 个 token，新 token 的位置是 100，不是 0
- 保证 RoPE 位置编码的连续性

---

## 10.4 温度采样（Temperature Scaling）

### 原理说明

温度（Temperature）控制采样分布的"陡峭程度"。在 softmax 之前对 logits 除以温度：

```
logits_scaled = logits / temperature
probs = softmax(logits_scaled)
```

- **temperature < 1.0**：logits 被放大，softmax 分布更"尖锐"（趋向 argmax），生成更确定、更保守
- **temperature = 1.0**：不缩放，原始分布
- **temperature > 1.0**：logits 被缩小，softmax 分布更"平坦"，生成更多样、更随机
- **temperature → 0**：等价于贪心解码（argmax）
- **temperature → ∞**：等价于均匀采样

### 代码位置引用

- 温度缩放：[model/model_minimind.py:294](file:///home/zhangxm/model_minimind/model/model_minimind.py#L294-L294)

```python
logits = outputs.logits[:, -1, :] / temperature
```

### 温度对分布的影响示意

假设原始 logits = [2.0, 1.0, 0.5]（3 个 token）：

| temperature | scaled logits | softmax probs | 效果 |
|-------------|--------------|---------------|------|
| 0.5 | [4.0, 2.0, 1.0] | [0.86, 0.12, 0.07] | 更尖锐（偏向高概率） |
| 1.0 | [2.0, 1.0, 0.5] | [0.59, 0.22, 0.13] | 原始 |
| 2.0 | [1.0, 0.5, 0.25] | [0.42, 0.25, 0.20] | 更平坦（趋向均匀） |

MiniMind 默认 `temperature=0.85`，略低于 1，使分布稍微更尖锐，生成质量与多样性的折中。

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| logits（输入） | [batch, vocab_size] | float32 | 原始 logits |
| temperature | 标量 | float | 温度值 |
| logits（输出） | [batch, vocab_size] | float32 | 缩放后的 logits |

---

## 10.5 Top-K 采样

### 原理说明

Top-K 采样只保留概率最高的 K 个 token，把其余 token 的概率置为 -inf（softmax 后为 0），然后在 K 个候选中重新归一化采样。这样可以避免采样到概率极低的"噪声 token"。

### 代码位置引用

- Top-K 实现：[model/model_minimind.py:297-298](file:///home/zhangxm/model_minimind/model/model_minimind.py#L297-L298)

```python
if top_k > 0:
    logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
```

### 关键计算逻辑解释

```python
logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
```

逐行拆解：

1. `torch.topk(logits, top_k)`：返回 logits 中最大的 K 个值及其索引，形状 [batch, K]
2. `[0]`：取 values（不是 indices），形状 [batch, K]
3. `[..., -1, None]`：取第 K 大的那个值（即 top-K 中的最小值），形状 [batch, 1]
   - 这是"门槛值"：只有大于等于这个值的 token 才保留
4. `logits < 门槛`：布尔掩码，True 表示该 token 不在 top-K 中
5. `logits[mask] = -inf`：把不在 top-K 的 token logits 置为 -inf

**示例**（top_k=3，vocab=6）：

```
logits = [2.0, 5.0, 1.0, 4.0, 0.5, 3.0]
topk values = [5.0, 4.0, 3.0]   # 最大的 3 个
门槛 = 3.0  (topk values 的最后一个)

掩码: logits < 3.0 → [True, False, True, False, True, False]
结果: [-inf, 5.0, -inf, 4.0, -inf, 3.0]
```

后续 `torch.softmax` 时，-inf 位置的 prob 为 0，只在保留的 K 个 token 中归一化。

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| logits（输入） | [batch, vocab_size] | float32 | 温度缩放后的 logits |
| top_k | 标量 | int | 保留的候选数 |
| logits（输出） | [batch, vocab_size] | float32 | 非 top-K 位置为 -inf |

---

## 10.6 Top-P（Nucleus）采样

### 原理说明

Top-P 采样（也称 nucleus sampling）是 Top-K 的动态版本：不固定保留 K 个 token，而是按概率从大到小排序，累计概率达到 P 的最小 token 集合作为候选。

- **Top-K**：固定候选数量，但概率分布尖锐时可能保留过多噪声，平坦时可能保留过少
- **Top-P**：固定累计概率阈值，分布尖锐时候选少，平坦时候选多，自适应性强

### 代码位置引用

- Top-P 实现：[model/model_minimind.py:299-303](file:///home/zhangxm/model_minimind/model/model_minimind.py#L299-L303)

```python
if top_p < 1.0:
    # nucleus sampling：只保留累计概率达到 top_p 的高概率候选 token。
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
    mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
    logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
```

### 关键计算逻辑解释

**步骤 1：按 logits 降序排序**

```python
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
```

- `sorted_logits`：从大到小排列的 logits
- `sorted_indices`：原始索引（用于后续还原顺序）

**步骤 2：计算累计概率并生成掩码**

```python
mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
```

- 对排序后的 logits 做 softmax 得到概率（从大到小）
- `cumsum` 计算累计概率
- 累计概率 > top_p 的位置标记为 True（这些是要被移除的）

**步骤 3：掩码右移一位**

```python
mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
```

这一步很巧妙：把掩码向右移一位（保留刚好使累计概率达到 top_p 的那个 token）。

- 原始 mask：累计概率**超过** top_p 的位置为 True
- 但"刚好达到 top_p"的那个 token 应该保留（否则累计概率不足 top_p）
- 右移一位后，第 0 个位置固定为 0（保留），其余位置用前一位的值

**示例**（top_p=0.85，排序后概率 = [0.5, 0.3, 0.15, 0.04, 0.01]）：

```
累计概率: [0.5, 0.8, 0.95, 0.99, 1.0]
原始 mask (> 0.85): [F, F, T, T, T]
右移后 mask:         [F, F, F, T, T]   # 第 2 位（0.15）保留，因为加上它才达到 0.95 > 0.85
```

保留前 3 个 token（累计概率 0.95），移除后 2 个。

**步骤 4：把掩码还原到原始顺序并置 -inf**

```python
logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
```

- `mask` 是按排序后顺序的掩码
- `scatter(1, sorted_indices, mask)`：把掩码按 sorted_indices 散回原始顺序
- 在原始 logits 上把对应位置置为 -inf

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| logits（输入） | [batch, vocab_size] | float32 | top-k 后的 logits |
| sorted_logits | [batch, vocab_size] | float32 | 降序排列的 logits |
| sorted_indices | [batch, vocab_size] | long | 排序后的原始索引 |
| mask | [batch, vocab_size] | bool | True 表示要移除 |
| logits（输出） | [batch, vocab_size] | float32 | nucleus 外位置为 -inf |

---

## 10.7 重复惩罚（Repetition Penalty）

### 原理说明

重复惩罚（Repetition Penalty）用于减少模型生成重复内容。对已经出现在生成序列中的 token，降低其 logits，使模型更倾向于生成新 token。

### 代码位置引用

- 重复惩罚实现：[model/model_minimind.py:295-296](file:///home/zhangxm/model_minimind/model/model_minimind.py#L295-L296)

```python
if repetition_penalty != 1.0:
    # 对已经出现过的 token 降权，减少模型重复同一句话的概率。
    for i in range(input_ids.shape[0]):
        seen = torch.unique(input_ids[i]); score = logits[i, seen]
        logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
```

### 关键计算逻辑解释

```python
for i in range(input_ids.shape[0]):
    seen = torch.unique(input_ids[i])                          # 该序列已出现过的 token
    score = logits[i, seen]                                    # 这些 token 的当前 logits
    logits[i, seen] = torch.where(score > 0,
                                   score / repetition_penalty, # 正 logits：除以惩罚系数（降低）
                                   score * repetition_penalty) # 负 logits：乘以惩罚系数（更负）
```

**非对称惩罚的设计**：

- `repetition_penalty > 1.0`（如 1.1）时：
  - 正 logits（score > 0）→ `score / 1.1`：值变小（但仍为正）
  - 负 logits（score < 0）→ `score * 1.1`：值更负
  - 两种情况都使该 token 的概率降低
- 用 `torch.where` 区分正负是为了保证惩罚的单调性：无论 logits 正负，惩罚后该 token 的相对概率都下降

**为什么不能简单除以惩罚系数？**

如果 score < 0（如 -2.0），除以 1.1 得到 -1.818，绝对值反而变小（更接近 0），该 token 概率反而上升。所以负 logits 要用乘法。

**示例**：

| score | repetition_penalty | score > 0? | 结果 | 效果 |
|-------|-------------------|-----------|------|------|
| 2.0 | 1.1 | True | 2.0 / 1.1 = 1.818 | 降低（仍正） |
| -2.0 | 1.1 | False | -2.0 × 1.1 = -2.2 | 降低（更负） |

两种情况都让该 token 概率下降。

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| input_ids | [batch, total_len] | long | 当前所有 token（含 prompt + 已生成） |
| seen | [num_unique] | long | 该序列出现过的去重 token |
| logits（输入） | [batch, vocab_size] | float32 | 温度缩放后的 logits |
| logits（输出） | [batch, vocab_size] | float32 | 已出现 token 的 logits 被惩罚 |

---

## 10.8 采样与贪心解码

### 原理说明

经过 temperature、repetition_penalty、top_k、top_p 处理后，得到最终的 logits。下一步是从中选出 next_token，有两种方式：

- **采样（do_sample=True）**：先 softmax 得到概率分布，再按概率随机采样
- **贪心（do_sample=False）**：直接取 argmax，每步选概率最高的 token

### 代码位置引用

- token 选择：[model/model_minimind.py:304](file:///home/zhangxm/model_minimind/model/model_minimind.py#L304-L304)

```python
next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
```

### 关键计算逻辑解释

**采样模式**：

```python
next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
```

- `torch.softmax(logits, dim=-1)`：把 logits 转为概率分布 [batch, vocab_size]
  - 被 top_k/top_p 置为 -inf 的位置 prob=0，不会被采样到
- `torch.multinomial`：按概率分布随机采样 1 个 token，返回索引
- 每次运行结果可能不同（随机性）

**贪心模式**：

```python
next_token = torch.argmax(logits, dim=-1, keepdim=True)
```

- 直接取 logits 最大的 token
- 每次运行结果确定（无随机性）
- 等价于 temperature→0 的极限情况

**两种模式对比**：

| 方面 | 采样（do_sample=True） | 贪心（do_sample=False） |
|------|----------------------|----------------------|
| 选择方式 | 按概率随机 | argmax |
| 多样性 | 高（同样 prompt 多次结果不同） | 无（结果确定） |
| 质量 | 可能偶尔出错 | 稳定但易重复 |
| 适用场景 | 创意写作、对话 | 翻译、摘要、代码 |

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| logits | [batch, vocab_size] | float32 | 处理后的 logits |
| probs | [batch, vocab_size] | float32 | softmax 概率分布 |
| next_token | [batch, 1] | long | 采样/贪心得到的下一个 token |

---

## 10.9 EOS 批量处理

### 原理说明

批量生成时，不同序列可能在不同的步数遇到 EOS。例如 batch=3，序列 A 在第 10 步结束，序列 B 在第 15 步，序列 C 在第 20 步。需要保证：
- 已结束的序列不再生成新内容（用 EOS 填充）
- 所有序列都结束后才停止整个生成循环

### 代码位置引用

- finished 标记初始化：[model/model_minimind.py:284](file:///home/zhangxm/model_minimind/model/model_minimind.py#L284-L284)
- EOS 替换：[model/model_minimind.py:305-306](file:///home/zhangxm/model_minimind/model/model_minimind.py#L305-L306)
- finished 更新与停止判断：[model/model_minimind.py:309-310](file:///home/zhangxm/model_minimind/model/model_minimind.py#L309-L310)

### 关键计算逻辑解释

**步骤 1：初始化 finished 标记**

```python
finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
```

- `finished` 形状 [batch]，bool 类型
- 初始全 False，表示所有序列都未结束

**步骤 2：对已结束序列强制输出 EOS**

```python
if eos_token_id is not None:
    next_token = torch.where(
        finished.unsqueeze(-1),                                    # [batch, 1] bool
        next_token.new_full((next_token.shape[0], 1), eos_token_id),  # 全 EOS
        next_token                                                 # 原始采样结果
    )
```

- `finished.unsqueeze(-1)`：[batch, 1] 的布尔掩码
- 对于已结束的序列（finished=True），next_token 被替换为 EOS
- 对于未结束的序列（finished=False），保留原始采样的 next_token
- 这样已结束的序列后续全是 EOS，不会"乱说话"

**步骤 3：更新 finished 标记**

```python
finished |= next_token.squeeze(-1).eq(eos_token_id)
```

- `next_token.squeeze(-1)`：[batch] 的 token id
- `.eq(eos_token_id)`：判断是否等于 EOS，得到 [batch] 的 bool
- `|=`：或赋值，只要之前结束过或当前生成了 EOS，就标记为结束

**步骤 4：全部结束则停止循环**

```python
if finished.all(): break
```

- `finished.all()`：所有序列都结束时返回 True
- 提前 break，避免无意义的 EOS 填充循环

### 批量处理示意

batch=3，假设各序列在第 2、4、6 步生成 EOS：

| 步骤 | seq0 | seq1 | seq2 | finished |
|------|------|------|------|----------|
| 1 | tok_a | tok_b | tok_c | [F, F, F] |
| 2 | EOS | tok_d | tok_e | [T, F, F] |
| 3 | EOS（强制） | tok_f | tok_g | [T, F, F] |
| 4 | EOS（强制） | EOS | tok_h | [T, T, F] |
| 5 | EOS（强制） | EOS（强制） | tok_i | [T, T, F] |
| 6 | EOS（强制） | EOS（强制） | EOS | [T, T, T] → break |

最终输出各序列长度相同（都是 6 步），但 seq0 的第 3-6 位、seq1 的第 5-6 位都是 EOS 填充，使用时可以截断到第一个 EOS。

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| finished | [batch] | bool | 每个序列是否已结束 |
| next_token | [batch, 1] | long | 采样结果（已结束序列被替换为 EOS） |

---

## 10.10 完整生成流程总结

以 batch=2, prompt_len=10, max_new_tokens=20 为例，完整流程：

| 步骤 | 操作 | input_ids Shape | past_key_values |
|------|------|----------------|-----------------|
| 初始化 | repeat + 初始化 finished | [2, 10] | None |
| 循环 0（prefill） | forward 完整 prompt | [2, 10] | 各层 [2, 10, 4, 96] |
| 循环 0 | 采样 next_token | [2, 11] | 各层 [2, 11, 4, 96] |
| 循环 1（decode） | forward 新 1 token | [2, 11] | 各层 [2, 11, 4, 96] |
| 循环 1 | 采样 next_token | [2, 12] | 各层 [2, 12, 4, 96] |
| ... | ... | ... | ... |
| 循环 19 | 采样 next_token | [2, 30] | 各层 [2, 30, 4, 96] |
| 结束 | 返回 input_ids | [2, 30] | - |

### 各步骤在 logits 上的处理顺序

```
原始 logits [batch, vocab]
    │ / temperature
    ▼
温度缩放 logits
    │ repetition_penalty（对已出现 token 降权）
    ▼
惩罚后 logits
    │ top_k（保留 K 个，其余 -inf）
    ▼
top-k logits
    │ top_p（nucleus，累计概率外的 -inf）
    ▼
最终 logits
    │ softmax + multinomial  或  argmax
    ▼
next_token [batch, 1]
    │ EOS 替换（已结束序列）
    ▼
final next_token
```

---

## 小结

本章详细拆解了 MiniMind 的推理生成算法：

| 步骤 | 机制 | 作用 | 代码位置 |
|------|------|------|---------|
| 自回归循环 | prefill + decode | 逐步生成 token | L286-310 |
| KV Cache | 缓存历史 K/V | 避免重算，加速推理 | L287-288 |
| 温度采样 | logits / temperature | 控制分布尖锐度 | L294 |
| 重复惩罚 | 已出现 token 降权 | 减少重复 | L295-296 |
| Top-K | 保留最高 K 个 | 避免噪声 token | L297-298 |
| Top-P | 累计概率阈值 | 动态候选集 | L299-303 |
| 采样/贪心 | multinomial / argmax | 多样性 vs 确定性 | L304 |
| EOS 批处理 | finished 标记 | 不同序列异步结束 | L284, L305-310 |

核心要点：

1. **自回归生成** = prefill（一次处理完整 prompt）+ decode（逐 token 生成）
2. **KV Cache** 把每步计算量从 O(L²) 降为 O(L)，是推理加速的关键
3. **温度**控制分布形状，**Top-K/Top-P**控制候选范围，**重复惩罚**防止循环重复
4. **采样**（do_sample=True）产生多样性，**贪心**（do_sample=False）保证确定性
5. **EOS 批处理**用 finished 标记优雅处理不同序列异步结束，已结束序列填充 EOS
