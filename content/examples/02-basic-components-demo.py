"""
MiniMind 基础组件示例代码
========================

本脚本演示三个核心基础组件的工作原理：
1. Token Embedding（词嵌入层）
2. RMSNorm（均方根层归一化）
3. RoPE（旋转位置编码）

运行方式：
    python 02-basic-components-demo.py

依赖：
    - PyTorch（CPU 版本即可）

运行环境：CPU 即可运行，无需 GPU。
"""

import torch
import torch.nn as nn
import math


# =============================================================================
# 示例 1：Token Embedding（词嵌入层）
# =============================================================================
print("=" * 70)
print("示例 1：Token Embedding（词嵌入层）")
print("=" * 70)

vocab_size = 10  # 词表大小
hidden_size = 4   # 词向量维度

# 创建一个简单的 Embedding 层
embedding = nn.Embedding(vocab_size, hidden_size)

# 用固定的随机种子初始化，便于复现
torch.manual_seed(42)
nn.init.xavier_uniform_(embedding.weight)

# 构造输入：batch_size=2, seq_len=3
input_ids = torch.tensor([
    [1, 3, 5],   # 第 1 个句子的 token ids
    [2, 4, 6]    # 第 2 个句子的 token ids
], dtype=torch.long)

print("\n输入 input_ids:")
print(f"  shape: {input_ids.shape}")
print(f"  dtype: {input_ids.dtype}")
print(f"  内容:\n{input_ids}")

# 前向传播：token ids -> 词向量
hidden_states = embedding(input_ids)

print("\n输出 hidden_states:")
print(f"  shape: {hidden_states.shape}  # [batch_size, seq_len, hidden_size]")
print(f"  dtype: {hidden_states.dtype}")
print(f"  内容:\n{hidden_states}")

# 打印某个 token 的词向量
token_id = 3
token_embedding = embedding.weight[token_id]
print(f"\nToken {token_id} 的词向量:")
print(f"  shape: {token_embedding.shape}")
print(f"  值: {token_embedding}")

# 验证：直接从 embedding.weight 中查找，与前向传播结果一致
print(f"\n验证：从 input_ids[0, 1]（token {input_ids[0, 1].item()}）查表结果与前向传播一致？")
print(f"  查表: {embedding.weight[input_ids[0, 1]]}")
print(f"  前向: {hidden_states[0, 1]}")
print(f"  一致: {torch.allclose(embedding.weight[input_ids[0, 1]], hidden_states[0, 1])}")

print()


# =============================================================================
# 示例 2：RMSNorm（均方根层归一化）
# =============================================================================
print("=" * 70)
print("示例 2：RMSNorm（均方根层归一化）")
print("=" * 70)


# 手写一个简化的 RMSNorm 类
class SimpleRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 计算均方根 rms = sqrt(mean(x^2) + eps)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # 归一化并乘以可学习权重
        return self.weight * x / rms


# 测试输入
batch_size, seq_len, hidden_size = 2, 3, 4
torch.manual_seed(123)
x = torch.randn(batch_size, seq_len, hidden_size)

print("\n输入 x:")
print(f"  shape: {x.shape}")
print(f"  第 0 个样本第 0 个位置的值: {x[0, 0]}")
print(f"  原始均值: {x[0, 0].mean().item():.6f}")
print(f"  原始均方根: {torch.sqrt(x[0, 0].pow(2).mean()).item():.6f}")

# 创建 RMSNorm 层
rms_norm = SimpleRMSNorm(hidden_size, eps=1e-6)
rms_output = rms_norm(x)

print("\nRMSNorm 输出:")
print(f"  shape: {rms_output.shape}")
print(f"  第 0 个样本第 0 个位置的值: {rms_output[0, 0]}")
print(f"  输出均值: {rms_output[0, 0].mean().item():.6f}  # 注意：RMSNorm 不主动减均值")
print(f"  输出均方根: {torch.sqrt(rms_output[0, 0].pow(2).mean()).item():.6f}  # 约等于 1（weight 全 1 时）")

# 对比 PyTorch 内置的 LayerNorm
layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
layer_norm.weight.data.fill_(1.0)
layer_norm.bias.data.fill_(0.0)
ln_output = layer_norm(x)

print("\nLayerNorm 输出:")
print(f"  第 0 个样本第 0 个位置的值: {ln_output[0, 0]}")
print(f"  输出均值: {ln_output[0, 0].mean().item():.6f}  # LayerNorm 均值为 0")
print(f"  输出均方根: {torch.sqrt(ln_output[0, 0].pow(2).mean()).item():.6f}")

# 对比两者差异
diff = (rms_output - ln_output).abs().mean()
print(f"\nRMSNorm 与 LayerNorm 的平均绝对差异: {diff.item():.6f}")
print("  差异的原因：LayerNorm 会减去均值，而 RMSNorm 不会")

# 验证 RMSNorm 的缩放特性
x_mean = x.mean(-1, keepdim=True)
rms_mean = rms_output.mean(-1, keepdim=True)
ratio = rms_mean / x_mean  # 均值的缩放比例
rms_value = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
print(f"\n验证：均值缩放比例 ≈ 1/rms 吗？")
print(f"  实际比例: {ratio[0, 0].item():.6f}")
print(f"  1 / rms: {(1.0 / rms_value[0, 0]).item():.6f}")
print(f"  一致: {torch.allclose(ratio, 1.0 / rms_value, atol=1e-5)}")

print()


# =============================================================================
# 示例 3：RoPE（旋转位置编码）
# =============================================================================
print("=" * 70)
print("示例 3：RoPE（旋转位置编码）")
print("=" * 70)


# 手写简化版的 precompute_freqs_cis
def precompute_freqs_cis(dim: int, end: int, rope_base: float = 10000.0):
    """
    预计算 RoPE 需要的 cos/sin 表
    输入:
        dim: head_dim 的大小（必须是偶数）
        end: 最大序列长度
        rope_base: 频率基值
    输出:
        freqs_cos: [end, dim]
        freqs_sin: [end, dim]
    """
    # 生成频率序列: theta_i = rope_base ^ (-2i / dim)
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
    # 生成位置序列: 0, 1, 2, ..., end-1
    t = torch.arange(end, device=freqs.device)
    # 外积得到每个位置的旋转角度: [end, dim//2]
    freqs = torch.outer(t, freqs).float()
    # 拼接成完整的 cos/sin: [end, dim]
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(x, cos, sin):
    """
    对输入张量 x 应用旋转位置编码
    输入:
        x: [seq_len, head_dim] 或 [batch, seq, num_heads, head_dim]
        cos, sin: [seq_len, head_dim]
    输出:
        旋转后的 x，形状与输入相同
    """
    def rotate_half(x_in):
        # 将后半部分取负，与前半部分拼接，等效于复数乘法的虚部
        half = x_in.shape[-1] // 2
        return torch.cat((-x_in[..., half:], x_in[..., :half]), dim=-1)

    # 确保 cos/sin 的形状匹配（支持广播）
    if x.dim() == 4:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

    return (x * cos) + (rotate_half(x) * sin)


print("\n3.1 预计算 cos/sin 表")
head_dim = 8   # 每个注意力头的维度
seq_len = 4    # 序列长度
freqs_cos, freqs_sin = precompute_freqs_cis(dim=head_dim, end=seq_len, rope_base=10000.0)

print(f"  head_dim: {head_dim}, seq_len: {seq_len}")
print(f"  freqs_cos shape: {freqs_cos.shape}")
print(f"  freqs_sin shape: {freqs_sin.shape}")
print(f"  位置 0 的 cos 值: {freqs_cos[0]}")
print(f"  位置 1 的 cos 值: {freqs_cos[1]}")
print(f"  位置 0 的 sin 值: {freqs_sin[0]}  # 全 0，因为 sin(0) = 0")

print("\n3.2 用一个简单的 Q 向量测试旋转效果")
# 创建一个 Q 向量：所有位置的 Q 都相同，便于观察旋转效果
q = torch.zeros(seq_len, head_dim)
q[:, 0] = 1.0  # 第 0 维设为 1，其他为 0
print(f"  原始 Q (所有位置相同): {q[0]}")

# 应用 RoPE
q_rotated = apply_rotary_pos_emb(q, freqs_cos, freqs_sin)

print(f"\n  旋转后的 Q:")
for i in range(seq_len):
    print(f"    位置 {i}: {q_rotated[i]}")
print(f"\n  观察：位置 0 的向量不变（旋转角度为 0）")
print(f"  观察：位置越远，旋转角度越大")

print("\n3.3 验证相对位置不变性")
print("  验证：位置 0 和 1 的点积 ≈ 位置 1 和 2 的点积 ≈ 位置 2 和 3 的点积")

# 创建两个完全相同的向量作为 Q 和 K（简化测试）
v = torch.randn(head_dim)
print(f"  测试向量 v: {v}")

# 为各个位置生成旋转后的向量
v_at = []
for pos in range(seq_len):
    cos_p = freqs_cos[pos:pos+1, :]
    sin_p = freqs_sin[pos:pos+1, :]
    v_rot = apply_rotary_pos_emb(v.unsqueeze(0), cos_p, sin_p).squeeze(0)
    v_at.append(v_rot)

# 计算相邻位置的点积
dot_0_1 = torch.dot(v_at[0], v_at[1]).item()
dot_1_2 = torch.dot(v_at[1], v_at[2]).item()
dot_2_3 = torch.dot(v_at[2], v_at[3]).item()

print(f"\n  点积结果:")
print(f"    <v(0), v(1)> = {dot_0_1:.6f}")
print(f"    <v(1), v(2)> = {dot_1_2:.6f}")
print(f"    <v(2), v(3)> = {dot_2_3:.6f}")

# 验证相对位置不变性
diff_01_12 = abs(dot_0_1 - dot_1_2)
diff_12_23 = abs(dot_1_2 - dot_2_3)

print(f"\n  差异:")
print(f"    |<0,1> - <1,2>| = {diff_01_12:.2e}")
print(f"    |<1,2> - <2,3>| = {diff_12_23:.2e}")

print(f"\n  相对位置不变性验证: {'通过' if diff_01_12 < 1e-5 and diff_12_23 < 1e-5 else '失败'}")
print(f"  （差异在 1e-5 以内视为通过，浮点精度误差范围内）")

# 额外验证：距离为 2 的点积也应该相等
print(f"\n  额外验证：距离为 2 的点积也相等")
dot_0_2 = torch.dot(v_at[0], v_at[2]).item()
dot_1_3 = torch.dot(v_at[1], v_at[3]).item()
print(f"    <v(0), v(2)> = {dot_0_2:.6f}")
print(f"    <v(1), v(3)> = {dot_1_3:.6f}")
print(f"    差异: {abs(dot_0_2 - dot_1_3):.2e}")
print(f"    验证: {'通过' if abs(dot_0_2 - dot_1_3) < 1e-5 else '失败'}")

print()
print("=" * 70)
print("所有示例运行完毕！")
print("=" * 70)
