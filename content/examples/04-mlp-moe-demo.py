"""
MiniMind 前馈网络 MLP & MoE 示例代码
====================================

本脚本演示前馈网络（FeedForward）和混合专家模型（MoE）的工作原理，对应教程第 4 章：
1. SwiGLU FFN 前向传播
2. SiLU 门控与 ReLU 对比
3. MoE 路由（gate + top-k 选择）
4. MoE 前向传播（条件计算）
5. 参数量对比（Dense vs MoE 总参数 vs MoE 激活参数）
6. top-2 路由权重归一化
7. 负载均衡辅助损失

运行方式：
    python 04-mlp-moe-demo.py

依赖：
    - PyTorch（CPU 版本即可）

运行环境：CPU 即可运行，无需 GPU。
本示例为自包含简化实现，不依赖 MiniMind 源码。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SwiGLU FFN：gate_proj + up_proj + down_proj，act(gate(x)) * up(x)
# 对应 model_minimind.py 的 FeedForward 类
# ---------------------------------------------------------------------------
class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        # SwiGLU: silu(gate_proj(x)) * up_proj(x)，再投影回 hidden_size
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# MoE FFN：gate 路由 + top-k 专家 + 辅助负载均衡损失
# 对应 model_minimind.py 的 MOEFeedForward 类
# ---------------------------------------------------------------------------
class MoEFFN(nn.Module):
    def __init__(self, hidden_size, moe_intermediate_size, num_experts, num_experts_per_tok, aux_loss_coef=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.aux_loss_coef = aux_loss_coef
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUFFN(hidden_size, moe_intermediate_size) for _ in range(num_experts)
        ])
        self.aux_loss = None

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)                       # [n_tokens, hidden]
        # gate 为每个 token 打分，选择 top-k 个专家
        scores = F.softmax(self.gate(x_flat), dim=-1)          # [n_tokens, num_experts]
        topk_weight, topk_idx = torch.topk(scores, k=self.num_experts_per_tok, dim=-1, sorted=False)
        # 归一化 top-k 权重，使其和为 1
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)                              # [n_tokens, k] 哪些 token 选了专家 i
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, expert(x_flat[token_idx]) * weight)
        # 辅助负载均衡损失
        load = F.one_hot(topk_idx, self.num_experts).float().mean(0)   # 每个专家被选中的比例
        self.aux_loss = (load * scores.mean(0)).sum() * self.num_experts * self.aux_loss_coef
        return y.view(batch_size, seq_len, hidden_dim)


torch.manual_seed(42)


# =============================================================================
# 示例 1：SwiGLU FFN 前向传播
# =============================================================================
print("=" * 70)
print("示例 1：SwiGLU FFN 前向传播")
print("=" * 70)

hidden_size, intermediate_size = 16, 32
ffn = SwiGLUFFN(hidden_size, intermediate_size)
batch, seq = 2, 5
x = torch.randn(batch, seq, hidden_size)

# 分步展示中间张量
gate = ffn.gate_proj(x)
up = ffn.up_proj(x)
act = F.silu(gate)
gated = act * up
out = ffn.down_proj(gated)

print(f"\n输入 x shape: {x.shape}  # [batch, seq, hidden_size={hidden_size}]")
print(f"gate_proj(x) shape: {gate.shape}  # [batch, seq, intermediate={intermediate_size}]")
print(f"up_proj(x) shape: {up.shape}  # [batch, seq, intermediate={intermediate_size}]")
print(f"silu(gate) shape: {act.shape}")
print(f"silu(gate) * up shape: {gated.shape}  # 门控后逐元素相乘")
print(f"down_proj(...) shape: {out.shape}  # [batch, seq, hidden_size={hidden_size}]（回到原维度）")

assert out.shape == x.shape
print("\n验证通过：SwiGLU 输出 shape 与输入相同")
print()


# =============================================================================
# 示例 2：SiLU 门控与 ReLU 对比
# =============================================================================
print("=" * 70)
print("示例 2：SiLU 门控与 ReLU 对比")
print("=" * 70)

vals = torch.linspace(-3, 3, 7)
silu_vals = F.silu(vals)
relu_vals = F.relu(vals)
print(f"\n输入值:    {vals.tolist()}")
print(f"SiLU(x):   {[round(v, 4) for v in silu_vals.tolist()]}")
print(f"ReLU(x):   {[round(v, 4) for v in relu_vals.tolist()]}")
print(f"\n观察：SiLU 在负值区域平滑过渡（有微小负值），ReLU 直接截断为 0")
print(f"  SiLU(0) = {F.silu(torch.tensor(0.0)).item()}  （过原点）")
print(f"  SiLU 最小值出现在 x≈-1.278 附近，SiLU(-1.278)≈-0.278")
print(f"  门控机制 silu(gate) 允许少量负信号通过，比 ReLU 更平滑、梯度更稳定")
print()


# =============================================================================
# 示例 3：MoE 路由（gate + top-k 选择）
# =============================================================================
print("=" * 70)
print("示例 3：MoE 路由（gate 打分 + top-k 选择）")
print("=" * 70)

num_experts, k = 4, 2
moe = MoEFFN(hidden_size, intermediate_size, num_experts, k)
n_tokens = 6
x_flat = torch.randn(n_tokens, hidden_size)

scores = F.softmax(moe.gate(x_flat), dim=-1)
topk_weight, topk_idx = torch.topk(scores, k=k, dim=-1, sorted=False)

print(f"\n配置：num_experts={num_experts}, top-k={k}")
print(f"输入 token 数: {n_tokens}")
print(f"gate 输出 scores shape: {scores.shape}  # [n_tokens, num_experts]")
print(f"\n每个 token 对各专家的路由概率（softmax 后）:")
for t in range(n_tokens):
    probs = [f"{v:.3f}" for v in scores[t].tolist()]
    chosen = sorted(topk_idx[t].tolist())
    print(f"  token {t}: [{', '.join(probs)}]  → 选中专家 {chosen}")
print(f"\n观察：每个 token 选 {k} 个概率最高的专家，其余专家不参与该 token 计算")
print()


# =============================================================================
# 示例 4：MoE 前向传播（条件计算）
# =============================================================================
print("=" * 70)
print("示例 4：MoE 前向传播（条件计算）")
print("=" * 70)

batch, seq = 2, 5
x = torch.randn(batch, seq, hidden_size)
out = moe(x)

print(f"\n输入 x shape: {x.shape}  # [batch={batch}, seq={seq}, hidden_size={hidden_size}]")
print(f"MoE 输出 shape: {out.shape}  # [batch, seq, hidden_size]（与输入相同）")
assert out.shape == x.shape
print(f"\n条件计算说明：每个 token 只激活 {k}/{num_experts} 个专家")
print(f"  总 token 数 = {batch * seq}，每个 token 仅前向 {k} 个专家的 FFN")
print(f"  相比 Dense FFN（所有 token 过同一个 FFN），MoE 用更多参数但每个 token 计算量更小")
print()


# =============================================================================
# 示例 5：参数量对比（Dense vs MoE 总参数 vs MoE 激活参数）
# =============================================================================
print("=" * 70)
print("示例 5：参数量对比")
print("=" * 70)

def count_params(m):
    return sum(p.numel() for p in m.parameters())

# 用一个等价中间维度的 Dense FFN 做对比
# MoE 激活参数 = 1 个专家的参数（每个 token 只用 k 个专家，但这里用单专家作对比基准）
dense_ffn = SwiGLUFFN(hidden_size, intermediate_size)
dense_params = count_params(dense_ffn)
moe_total_params = count_params(moe)
moe_active_per_token = count_params(moe.experts[0]) + count_params(moe.gate)  # 每 token 激活 k 个专家

print(f"\nDense FFN 参数量: {dense_params}")
print(f"  = 3 个 Linear: gate({hidden_size}x{intermediate_size}) + up({hidden_size}x{intermediate_size}) + down({intermediate_size}x{hidden_size})")
print(f"  = 3 x {hidden_size} x {intermediate_size} = {3 * hidden_size * intermediate_size}")
print(f"\nMoE 总参数量（{num_experts} 个专家 + gate）: {moe_total_params}")
print(f"  = {num_experts} x 专家参数 + gate({hidden_size}x{num_experts})")
print(f"  = {num_experts} x {count_params(moe.experts[0])} + {hidden_size * num_experts}")
print(f"\nMoE 每 token 激活参数量（{k} 个专家 + gate）: {count_params(moe.experts[0]) * k + hidden_size * num_experts}")
print(f"\n对比：")
print(f"  MoE 总参数 / Dense 参数 = {moe_total_params / dense_params:.2f} 倍  （容量更大）")
print(f"  MoE 激活参数 / Dense 参数 = {(count_params(moe.experts[0]) * k + hidden_size * num_experts) / dense_params:.2f} 倍  （单 token 计算量更小）")
print(f"  → MoE 用 {moe_total_params / dense_params:.1f}x 参数换 {k}/{num_experts} = {k/num_experts:.0%} 的计算量")
print()


# =============================================================================
# 示例 6：top-2 路由权重归一化
# =============================================================================
print("=" * 70)
print("示例 6：top-2 路由权重归一化")
print("=" * 70)

topk_weight_raw, topk_idx_demo = torch.topk(scores, k=k, dim=-1, sorted=True)
topk_weight_normed = topk_weight_raw / (topk_weight_raw.sum(dim=-1, keepdim=True) + 1e-20)

print(f"\n归一化前（softmax 后的 top-{k} 原始权重）:")
for t in range(min(3, n_tokens)):
    print(f"  token {t}: 选中专家 {topk_idx_demo[t].tolist()}, 权重 {[round(v,4) for v in topk_weight_raw[t].tolist()]}, 和={topk_weight_raw[t].sum().item():.4f}")
print(f"\n归一化后（除以权重和）:")
for t in range(min(3, n_tokens)):
    print(f"  token {t}: 选中专家 {topk_idx_demo[t].tolist()}, 权重 {[round(v,4) for v in topk_weight_normed[t].tolist()]}, 和={topk_weight_normed[t].sum().item():.4f}")

# 验证归一化后每行和为 1
all_sum_one = torch.allclose(topk_weight_normed.sum(-1), torch.ones(n_tokens), atol=1e-5)
print(f"\n验证：归一化后每个 token 的 top-{k} 权重和均为 1？{all_sum_one}")
print("  → 归一化保证被选中专家的加权输出不改变整体量级")
print()


# =============================================================================
# 示例 7：负载均衡辅助损失
# =============================================================================
print("=" * 70)
print("示例 7：负载均衡辅助损失")
print("=" * 70)

out = moe(x)
aux_loss = moe.aux_loss

# 重新计算中间量以便展示
scores_full = F.softmax(moe.gate(x.view(-1, hidden_size)), dim=-1)
topk_idx_full = torch.topk(scores_full, k=k, dim=-1, sorted=False).indices
load = F.one_hot(topk_idx_full, num_experts).float().reshape(-1, num_experts).mean(0)   # 每个专家被选中的频率 [num_experts]
prob_mean = scores_full.mean(0)                                 # 每个专家的平均路由概率

print(f"\n每个专家被选中的频率 f_i（load）: {[round(v,4) for v in load.tolist()]}")
print(f"  （理想情况下每个专家 = {1/num_experts:.2f} = 1/num_experts）")
print(f"\n每个专家的平均路由概率 P_i: {[round(v,4) for v in prob_mean.tolist()]}")
print(f"  （理想情况下每个专家 = {1/num_experts:.2f} = 1/num_experts）")
print(f"\n辅助损失 = (f_i * P_i).sum() * num_experts * aux_loss_coef")
print(f"  = ({(load * prob_mean).sum().item():.4f}) * {num_experts} * {moe.aux_loss_coef}")
print(f"  = {aux_loss.item():.6f}")
print(f"\n作用：当某个专家被过度使用时 f_i 和 P_i 都变大，损失增大 → 反向传播惩罚 → 鼓励均衡")
print(f"  理想均衡时损失 = (1/{num_experts})^2 * {num_experts}^2 * {k} * {moe.aux_loss_coef} ... 的下界")
print(f"  当前损失 {aux_loss.item():.6f}（越接近均衡越小）")

assert aux_loss.item() >= 0
print(f"\n验证：辅助损失非负？{aux_loss.item() >= 0}")
print()

print("=" * 70)
print("所有示例运行完毕！")
print("=" * 70)
