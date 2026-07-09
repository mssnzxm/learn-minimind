"""
MiniMind 注意力机制示例代码
==========================

本脚本演示注意力机制（Attention）的完整计算流程，对应教程第 3 章：
1. Q/K/V 线性投影与多头拆分
2. GQA 分组注意力的 repeat_kv
3. RoPE 旋转位置编码
4. 因果掩码（causal mask）
5. 完整注意力前向传播
6. prefill + decode 两阶段生成
7. KV Cache 正确性验证

运行方式：
    python 03-attention-demo.py

依赖：
    - PyTorch（CPU 版本即可）

运行环境：CPU 即可运行，无需 GPU。
本示例为自包含简化实现，不依赖 MiniMind 源码。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# 简化版 RMSNorm（用于 QK-Norm）
# ---------------------------------------------------------------------------
class SimpleRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x / rms


# ---------------------------------------------------------------------------
# RoPE 预计算与应用（简化版，对应 model_minimind.py 的 precompute_freqs_cis / apply_rotary_pos_emb）
# ---------------------------------------------------------------------------
def precompute_freqs_cis(head_dim, end, rope_base=10000.0):
    freqs = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(end).float()
    freqs = torch.outer(t, freqs)                       # [end, head_dim//2]
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # [end, head_dim]
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)  # [end, head_dim]
    return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [batch, seq, n_heads, head_dim] ；cos/sin: [seq, head_dim]
    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(2)   # [1, seq, 1, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# repeat_kv：把 KV 头复制到与 Q 头数一致（GQA 的关键步骤）
# ---------------------------------------------------------------------------
def repeat_kv(x, n_rep):
    # x: [batch, seq, n_kv_heads, head_dim]
    bs, slen, n_kv, hd = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, n_kv, n_rep, hd).reshape(bs, slen, n_kv * n_rep, hd)


# ---------------------------------------------------------------------------
# SimpleAttention：含 GQA、RoPE、QK-Norm、因果掩码、KV Cache
# ---------------------------------------------------------------------------
class SimpleAttention(nn.Module):
    def __init__(self, hidden_size=16, n_heads=4, n_kv_heads=2, head_dim=8):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_size, bias=False)
        self.q_norm = SimpleRMSNorm(head_dim)
        self.k_norm = SimpleRMSNorm(head_dim)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        # QK-Norm：在 RoPE 之前对每个 head 做 RMSNorm，稳定训练
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        # RoPE：只作用于 Q/K
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        # KV Cache：把历史 K/V 拼接回来
        if past_kv is not None:
            xk = torch.cat([past_kv[0], xk], dim=1)
            xv = torch.cat([past_kv[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        # 拆头：[batch, heads, seq, head_dim]
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)
        # 注意力分数
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 因果掩码：当前位置不能看到未来 token（只对"新输入"这一段做掩码）
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf")).triu(1)
            scores[:, :, :, -seq_len:] += mask
        attn = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = attn @ xv                                   # [batch, heads, seq, head_dim]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.o_proj(output)
        return output, past_kv


torch.manual_seed(42)


# =============================================================================
# 示例 1：Q/K/V 线性投影与多头拆分
# =============================================================================
print("=" * 70)
print("示例 1：Q/K/V 线性投影与多头拆分")
print("=" * 70)

hidden_size, n_heads, n_kv_heads, head_dim = 16, 4, 2, 8
attn = SimpleAttention(hidden_size, n_heads, n_kv_heads, head_dim)
batch, seq = 2, 5
x = torch.randn(batch, seq, hidden_size)

q = attn.q_proj(x).view(batch, seq, n_heads, head_dim)
k = attn.k_proj(x).view(batch, seq, n_kv_heads, head_dim)
v = attn.v_proj(x).view(batch, seq, n_kv_heads, head_dim)

print(f"\n输入 x shape: {x.shape}  # [batch, seq, hidden_size]")
print(f"Q shape: {q.shape}  # [batch, seq, n_heads={n_heads}, head_dim={head_dim}]")
print(f"K shape: {k.shape}  # [batch, seq, n_kv_heads={n_kv_heads}, head_dim={head_dim}]")
print(f"V shape: {v.shape}  # [batch, seq, n_kv_heads={n_kv_heads}, head_dim={head_dim}]")
print(f"\nGQA 配置：Q 头数 {n_heads} 是 KV 头数 {n_kv_heads} 的 {n_heads // n_kv_heads} 倍")
print(f"  → 每个 KV 头会被复制 {n_heads // n_kv_heads} 次供 {n_heads // n_kv_heads} 个 Q 头共享")

assert q.shape == (batch, seq, n_heads, head_dim)
assert k.shape == (batch, seq, n_kv_heads, head_dim)
print("\n验证通过：多头拆分 shape 正确")
print()


# =============================================================================
# 示例 2：repeat_kv（GQA 的关键步骤）
# =============================================================================
print("=" * 70)
print("示例 2：repeat_kv（把 KV 头复制到与 Q 头数一致）")
print("=" * 70)

n_rep = n_heads // n_kv_heads
k_expanded = repeat_kv(k, n_rep)
v_expanded = repeat_kv(v, n_rep)

print(f"\n原始 K shape: {k.shape}  # [batch, seq, n_kv_heads={n_kv_heads}, head_dim]")
print(f"repeat_kv 后 K shape: {k_expanded.shape}  # [batch, seq, n_heads={n_heads}, head_dim]")

# 验证：复制后的第 0、1 个头应该相同（都来自 KV 头 0）
same = torch.allclose(k_expanded[:, :, 0, :], k_expanded[:, :, 1, :])
same2 = torch.allclose(k_expanded[:, :, 2, :], k_expanded[:, :, 3, :])
print(f"\n验证：KV 头 0 被复制给 Q 头 0 和 1 → 两者相同？{same}")
print(f"验证：KV 头 1 被复制给 Q 头 2 和 3 → 两者相同？{same2}")
print("  （GQA 通过共享 KV 头显著减少 KV Cache 显存占用）")
print()


# =============================================================================
# 示例 3：RoPE 旋转位置编码
# =============================================================================
print("=" * 70)
print("示例 3：RoPE 旋转位置编码")
print("=" * 70)

max_seq = 16
cos, sin = precompute_freqs_cis(head_dim, max_seq, rope_base=10000.0)
print(f"\n预计算 cos/sin shape: {cos.shape}  # [max_seq, head_dim]")

# 取两个位置的 K 向量，观察 RoPE 后点积只依赖相对距离
k_raw = torch.randn(1, 2, n_kv_heads, head_dim)
k_pos0 = k_raw[:, 0:1].clone()
k_pos1 = k_raw[:, 1:2].clone()
k0_rot, _ = apply_rotary_pos_emb(k_pos0, k_pos0, cos[0:1], sin[0:1])
k1_rot, _ = apply_rotary_pos_emb(k_pos1, k_pos1, cos[1:2], sin[1:2])

print(f"\n位置 0 旋转前 K[0,0,0]: {k_pos0[0,0,0]}")
print(f"位置 0 旋转后 K[0,0,0]: {k0_rot[0,0,0]}")
print(f"  位置 0 旋转角度为 0，所以旋转前后模长不变")
norm_before = k_pos0[0,0,0].norm().item()
norm_after = k0_rot[0,0,0].norm().item()
print(f"  旋转前模长: {norm_before:.6f}")
print(f"  旋转后模长: {norm_after:.6f}")
print(f"  模长不变？{abs(norm_before - norm_after) < 1e-5}")

# 相对位置不变性：同一向量在位置 (0,1) 和 (1,2) 旋转后点积应相等
v_test = torch.randn(1, 1, n_kv_heads, head_dim)
v0, _ = apply_rotary_pos_emb(v_test, v_test, cos[0:1], sin[0:1])
v1, _ = apply_rotary_pos_emb(v_test, v_test, cos[1:2], sin[1:2])
v2, _ = apply_rotary_pos_emb(v_test, v_test, cos[2:2+1] if 2 < max_seq else cos[1:2], sin[2:2+1] if 2 < max_seq else sin[1:2])
dot_01 = (v0 * v1).sum().item()
dot_12 = (v1 * v2).sum().item()
print(f"\n相对位置不变性验证（同一向量，相邻位置点积）：")
print(f"  <v(0), v(1)> = {dot_01:.6f}")
print(f"  <v(1), v(2)> = {dot_12:.6f}")
print(f"  差异: {abs(dot_01 - dot_12):.2e}  → {'通过' if abs(dot_01 - dot_12) < 1e-5 else '失败'}")
print()


# =============================================================================
# 示例 4：因果掩码（causal mask）
# =============================================================================
print("=" * 70)
print("示例 4：因果掩码（causal mask）")
print("=" * 70)

seq_len = 5
# 手动构造一个 scores 矩阵并施加因果掩码
scores = torch.zeros(seq_len, seq_len)
causal_mask = torch.full((seq_len, seq_len), float("-inf")).triu(1)
masked_scores = scores + causal_mask
attn_weights = F.softmax(masked_scores, dim=-1)

print(f"\n原始 scores（全 0，假设未归一化）:")
print(scores)
print(f"\n因果掩码（上三角为 -inf）:")
print(causal_mask)
print(f"\n施加掩码后 softmax 的注意力权重:")
print(attn_weights)
print(f"\n观察：下三角（含对角线）权重均匀，上三角为 0")
# 验证上三角全为 0
upper_zero = torch.all(attn_weights.triu(1) == 0)
print(f"验证：上三角（未来 token）权重全为 0？{upper_zero}")
print(f"验证：每行权重和为 1？{torch.allclose(attn_weights.sum(-1), torch.ones(seq_len))}")
print()


# =============================================================================
# 示例 5：完整注意力前向传播
# =============================================================================
print("=" * 70)
print("示例 5：完整注意力前向传播（GQA + RoPE + QK-Norm + 因果掩码）")
print("=" * 70)

batch, seq = 2, 6
x = torch.randn(batch, seq, hidden_size)
cos_s, sin_s = precompute_freqs_cis(head_dim, seq, rope_base=10000.0)

output, past_kv = attn(x, cos_s, sin_s, use_cache=True)
print(f"\n输入 x shape: {x.shape}  # [batch={batch}, seq={seq}, hidden_size={hidden_size}]")
print(f"输出 shape: {output.shape}  # [batch, seq, hidden_size]（与输入相同）")
print(f"KV Cache K shape: {past_kv[0].shape}  # [batch, seq, n_kv_heads={n_kv_heads}, head_dim={head_dim}]")
print(f"KV Cache V shape: {past_kv[1].shape}  # [batch, seq, n_kv_heads={n_kv_heads}, head_dim={head_dim}]")

assert output.shape == x.shape
assert past_kv[0].shape == (batch, seq, n_kv_heads, head_dim)
print("\n验证通过：前向输出 shape 与输入一致，KV Cache 已保存")
print()


# =============================================================================
# 示例 6：prefill + decode 两阶段生成
# =============================================================================
print("=" * 70)
print("示例 6：prefill + decode 两阶段生成（使用 KV Cache）")
print("=" * 70)

prompt_len = 4
prompt = torch.randn(1, prompt_len, hidden_size)
cos_p, sin_p = precompute_freqs_cis(head_dim, prompt_len, rope_base=10000.0)

# 阶段 1：prefill —— 一次性处理整个 prompt
out_prefill, past_kv = attn(prompt, cos_p, sin_p, use_cache=True)
print(f"\n[Prefill] 输入 prompt shape: {prompt.shape}")
print(f"[Prefill] 输出 shape: {out_prefill.shape}")
print(f"[Prefill] KV Cache 长度: {past_kv[0].shape[1]}  # = prompt_len={prompt_len}")

# 阶段 2：decode —— 逐 token 解码，每次只输入 1 个新 token
decode_steps = 3
print(f"\n[Decode] 逐 token 解码 {decode_steps} 步：")
for step in range(decode_steps):
    new_token = torch.randn(1, 1, hidden_size)           # 模拟上一步生成的新 token
    total_len = past_kv[0].shape[1] + 1
    cos_d, sin_d = precompute_freqs_cis(head_dim, total_len, rope_base=10000.0)
    # 只取最新位置的 cos/sin（decode 时 seq_len=1）
    out_decode, past_kv = attn(new_token, cos_d[total_len-1:total_len], sin_d[total_len-1:total_len],
                               past_kv=past_kv, use_cache=True)
    print(f"  步骤 {step+1}: 输入 1 token，KV Cache 长度 {past_kv[0].shape[1]}，输出 shape {out_decode.shape}")

print(f"\n最终 KV Cache 长度: {past_kv[0].shape[1]}  # = prompt_len + decode_steps = {prompt_len + decode_steps}")
print(f"  → decode 阶段每步只需计算 1 个 token 的 Q，K/V 从缓存读取，复杂度 O(1) 而非 O(n)")
print()


# =============================================================================
# 示例 7：KV Cache 正确性验证
# =============================================================================
print("=" * 70)
print("示例 7：KV Cache 正确性验证（有/无 cache 结果应一致）")
print("=" * 70)

total_len = prompt_len + decode_steps
all_tokens = torch.randn(1, total_len, hidden_size)
cos_all, sin_all = precompute_freqs_cis(head_dim, total_len, rope_base=10000.0)

# 方式 A：无 cache，一次性前向整段序列
out_no_cache, _ = attn(all_tokens, cos_all, sin_all, use_cache=False)

# 方式 B：有 cache，prefill + 逐 token decode（用相同的输入 token）
out_prefill2, past_kv2 = attn(all_tokens[:, :prompt_len], cos_all[:prompt_len], sin_all[:prompt_len], use_cache=True)
cached_outputs = [out_prefill2[:, -1:]]  # 保存每个位置的输出（最后一个）
for i in range(prompt_len, total_len):
    tok = all_tokens[:, i:i+1]
    out_d, past_kv2 = attn(tok, cos_all[i:i+1], sin_all[i:i+1], past_kv=past_kv2, use_cache=True)
    cached_outputs.append(out_d)
out_with_cache = torch.cat(cached_outputs, dim=1)

# cache 路径覆盖位置 prompt_len-1 ~ total_len-1，与无 cache 的对应位置比较
overlap_no_cache = out_no_cache[:, prompt_len-1:]
max_diff = (overlap_no_cache - out_with_cache).abs().max().item()
print(f"\n无 cache 一次性前向（位置 {prompt_len-1}~{total_len-1}）shape: {overlap_no_cache.shape}")
print(f"有 cache 分步前向输出 shape: {out_with_cache.shape}")
print(f"两者最大绝对差异: {max_diff:.2e}")
print(f"验证通过（差异 < 1e-5）？{max_diff < 1e-5}")
print("  → KV Cache 是纯加速优化，不改变计算结果")
print()

print("=" * 70)
print("所有示例运行完毕！")
print("=" * 70)
