"""
MiniMind LoRA 低秩适配原理示例
============================

本脚本演示 LoRA（Low-Rank Adaptation）低秩适配的工作原理：
1. LoRA 模块结构与初始化（B 零初始化、A 高斯初始化，初始 BA=0）
2. 参数量对比（原 Linear vs LoRA 增量参数）
3. apply_lora 挂载（forward = Wx + BAx）
4. 训练前后对比（W 不变、BA 变化）
5. 权重合并验证（merge: W' = W + BA，合并前后 forward 一致）
6. 低秩直觉（低秩近似捕获主要方差）

运行方式：
    python 09-lora-demo.py

依赖：
    - PyTorch（CPU 版本即可）
    - numpy

运行环境：CPU 即可运行，无需 GPU。

说明：本脚本使用自包含的简化 LoRA 类，仅依赖 torch(+numpy)。
对应 MiniMind 源码：model/model_lora.py。
"""

import torch
import torch.nn as nn
import numpy as np
import unicodedata


def _disp_width(s):
    """计算字符串的显示宽度（CJK/全角字符计 2，其余计 1）。"""
    w = 0
    for ch in str(s):
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


def _rpad(s, width):
    """右对齐，左侧补空格至指定显示宽度（兼容 CJK 全角字符）。"""
    s = str(s)
    return " " * max(0, width - _disp_width(s)) + s


# =============================================================================
# 自包含简化 LoRA 类（对应 model/model_lora.py 的 LoRA）
# =============================================================================
class LoRA(nn.Module):
    """
    低秩适配模块：用 B @ A 近似 ΔW，其中 A: [r, in]，B: [out, r]
    初始化：A 高斯，B 置零 → 初始 BA = 0，训练开始时不改变原模型输出。
    """
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        # A: [rank, in_features]，高斯初始化
        self.A = nn.Linear(in_features, rank, bias=False)
        # B: [rank, out_features]（用 Linear 表示，weight 形状 [out, rank]）
        self.B = nn.Linear(rank, out_features, bias=False)
        # A 高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # B 全零初始化（关键：保证训练开始时 BA=0，不破坏原模型）
        self.B.weight.data.zero_()

    def forward(self, x):
        # 增量 Δy = B(A(x)) = (B @ A) @ x
        return self.B(self.A(x))

    def delta_weight(self):
        # 返回 BA，形状 [out_features, in_features]
        return self.B.weight @ self.A.weight   # [out, rank] @ [rank, in] = [out, in]


def apply_lora(model, rank=8):
    """
    给模型中所有方阵 Linear 挂载 LoRA（对应源码只对方阵挂载）。
    挂载后 forward = Wx + BAx。
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank)
            setattr(module, "lora", lora)
            # 保存原始 forward（Linear.forward 的 bound method），便于合并时恢复
            original_forward = module.forward
            module._lora_original_forward = original_forward

            # 闭包绑定，避免循环引用问题
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


def merge_lora_weights(model):
    """
    把 LoRA 的 BA 合并回原权重：W' = W + B @ A。
    合并后 forward 恢复为原始 W'x（与合并前 Wx + BAx 数值一致）。
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "lora"):
            # W' = W + BA，注意 BA 形状 [out, in] 与 W 一致
            module.weight.data += module.lora.delta_weight()
            # 恢复原始 forward（仅 W'x + b），避免合并后重复计算 BAx
            if hasattr(module, "_lora_original_forward"):
                module.forward = module._lora_original_forward
                delattr(module, "_lora_original_forward")
            # 合并后移除 lora 分支
            delattr(module, "lora")


# =============================================================================
# 示例 1：LoRA 模块结构与初始化
# =============================================================================
print("=" * 70)
print("示例 1：LoRA 模块结构与初始化")
print("=" * 70)

in_features = 8
out_features = 8
rank = 4

torch.manual_seed(42)
lora = LoRA(in_features, out_features, rank)

print(f"\n配置: in_features={in_features}, out_features={out_features}, rank={rank}")
print(f"\nLoRA 参数形状:")
print(f"  A.weight (Linear in->r): {lora.A.weight.shape}  # [rank={rank}, in={in_features}]")
print(f"  B.weight (Linear r->out): {lora.B.weight.shape}  # [out={out_features}, rank={rank}]")

print(f"\n初始化统计:")
print(f"  A.weight 均值: {lora.A.weight.mean().item():.4f}  std: {lora.A.weight.std().item():.4f}  # 高斯初始化")
print(f"  B.weight 均值: {lora.B.weight.mean().item():.4f}  std: {lora.B.weight.std().item():.4f}  # 全零初始化")

# 验证初始 BA = 0
BA = lora.delta_weight()
print(f"\nBA (B @ A) 形状: {BA.shape}  # [out, in]")
print(f"  BA 的最大绝对值: {BA.abs().max().item():.2e}")
print(f"  BA 是否全零: {'是' if BA.abs().max().item() < 1e-8 else '否'}")
print(f"  验证初始 BA == 0: {'通过' if BA.abs().max().item() < 1e-8 else '失败'}")

# 验证初始时 LoRA 增量为 0（不改变原模型输出）
x = torch.randn(2, in_features)
delta_y = lora(x)
print(f"\n初始 LoRA 前向输出 Δy = BAx:")
print(f"  shape: {delta_y.shape}")
print(f"  最大绝对值: {delta_y.abs().max().item():.2e}")
print(f"  验证 Δy ≈ 0（训练开始时不破坏原模型）: "
      f"{'通过' if delta_y.abs().max().item() < 1e-7 else '失败'}")

print("\n  说明：B 零初始化是 LoRA 的关键设计——训练开始时 ΔW=BA=0，")
print("  模型行为与微调前完全一致，随后逐步学习低秩增量。")

print()


# =============================================================================
# 示例 2：参数量对比
# =============================================================================
print("=" * 70)
print("示例 2：参数量对比（原 Linear vs LoRA 增量）")
print("=" * 70)

# 原 Linear [in, out] 参数量 = in * out
# LoRA 参数量 = r * (in + out)（A: r*in, B: out*r）
# 比例 = r*(in+out) / (in*out) = r * (1/in + 1/out)

cases = [
    (768, 768, 8),     # 典型注意力投影
    (768, 768, 16),
    (1024, 1024, 8),
    (3072, 768, 8),    # 非 MoE MLP（注：MiniMind 只对方阵挂载，这里仅作对比）
]

_h2 = ["in", "out", "rank", "原参数(in*out)", "LoRA参数(r*(in+out))", "比例", "节省"]
_w2 = [6, 6, 4, 16, 22, 8, 8]
print("\n  " + " | ".join(_rpad(h, _w2[i]) for i, h in enumerate(_h2)))
print("  " + "-" * (sum(_w2) + 3 * (len(_w2) - 1)))
for in_f, out_f, r in cases:
    orig = in_f * out_f
    lora_params = r * (in_f + out_f)
    ratio = lora_params / orig
    saving = (1 - ratio) * 100
    cells = [str(in_f), str(out_f), str(r), str(orig), str(lora_params),
             f"{ratio*100:.2f}%", f"{saving:.2f}%"]
    print("  " + " | ".join(_rpad(c, _w2[k]) for k, c in enumerate(cells)))

# 重点：in=out=768, r=8
in_f, out_f, r = 768, 768, 8
orig = in_f * out_f
lora_params = r * (in_f + out_f)
ratio = lora_params / orig
print(f"\n  重点示例（in=out={in_f}, r={r}）:")
print(f"    原参数: {in_f} * {out_f} = {orig}")
print(f"    LoRA参数: {r} * ({in_f} + {out_f}) = {lora_params}")
print(f"    比例: {ratio*100:.2f}%（仅训练原参数量的 {ratio*100:.2f}%）")
print(f"    验证比例 = r*(1/in + 1/out) = {r}*(1/{in_f}+1/{out_f}) = {r*(1/in_f+1/out_f)*100:.2f}%")
print(f"    一致: {'通过' if abs(ratio - r*(1/in_f+1/out_f)) < 1e-9 else '失败'}")

print("\n  说明：rank 越小，可训练参数越少。r=8 时仅训练约 2% 的参数，")
print("  显存与训练开销大幅下降，且原权重 W 可冻结不存梯度。")

print()


# =============================================================================
# 示例 3：apply_lora 挂载
# =============================================================================
print("=" * 70)
print("示例 3：apply_lora 挂载（forward = Wx + BAx）")
print("=" * 70)

# 构造一个简单 MLP 模型
class SimpleMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


dim = 8
rank = 4
torch.manual_seed(0)
model = SimpleMLP(dim)

# 挂载前记录 fc1 的原始输出
x = torch.randn(2, dim)
fc1 = model.fc1
W = fc1.weight.data.clone()   # 原权重 W [out, in]
b = fc1.bias.data.clone()     # 原偏置
y_before = fc1(x).clone()     # 挂载前输出

print(f"\n模型: SimpleMLP(dim={dim})，含 fc1, fc2 两个方阵 Linear")
print(f"  fc1.weight 形状: {fc1.weight.shape}")
print(f"  挂载前 fc1(x) 输出形状: {y_before.shape}")
print(f"  挂载前 fc1 是否有 lora 属性: {hasattr(fc1, 'lora')}")

# 应用 apply_lora
apply_lora(model, rank=rank)
fc1 = model.fc1   # 重新取引用（属性已变化）
y_after = fc1(x)

print(f"\n挂载 LoRA (rank={rank}) 后:")
print(f"  fc1 是否有 lora 属性: {hasattr(fc1, 'lora')}")
print(f"  fc1.lora.A.weight 形状: {fc1.lora.A.weight.shape}  # [rank, in]")
print(f"  fc1.lora.B.weight 形状: {fc1.lora.B.weight.shape}  # [out, rank]")
print(f"  挂载后 fc1(x) 输出形状: {y_after.shape}")

# 验证：挂载后初始 BA=0，所以输出应与挂载前一致
print(f"\n验证：挂载后初始输出 == 挂载前输出（因 BA=0）")
print(f"  最大差异: {(y_after - y_before).abs().max().item():.2e}")
print(f"  一致: {'通过' if torch.allclose(y_after, y_before, atol=1e-6) else '失败'}")

# 手动验证 forward = Wx + b + BAx
BA = fc1.lora.delta_weight()
manual = x @ W.t() + b + x @ BA.t()
print(f"\n验证：forward == Wx + b + BAx")
print(f"  最大差异: {(y_after - manual).abs().max().item():.2e}")
print(f"  一致: {'通过' if torch.allclose(y_after, manual, atol=1e-6) else '失败'}")

print()


# =============================================================================
# 示例 4：训练前后对比（W 不变、BA 变化）
# =============================================================================
print("=" * 70)
print("示例 4：训练前后对比（W 不变、BA 变化）")
print("=" * 70)

# 重新构造模型，冻结 W，只训练 LoRA 参数
dim = 8
rank = 4
torch.manual_seed(1)
model = SimpleMLP(dim)
apply_lora(model, rank=rank)

# 冻结所有原始 Linear 权重（W、bias 不计算梯度），仅训练 LoRA 参数
for m in model.modules():
    if isinstance(m, nn.Linear):
        m.weight.requires_grad_(False)
        if m.bias is not None:
            m.bias.requires_grad_(False)
# 收集所有 LoRA 参数并设为可训练
lora_params = []
for m in model.modules():
    if hasattr(m, "lora"):
        for p in m.lora.parameters():
            p.requires_grad_(True)
            lora_params.append(p)

fc1 = model.fc1
print(f"\n可训练参数数量: {len(lora_params)}  (fc1 与 fc2 各自的 A、B)")
print(f"  fc1.weight.requires_grad: {fc1.weight.requires_grad}  # 冻结")
print(f"  fc1.lora.A.weight.requires_grad: {fc1.lora.A.weight.requires_grad}  # 可训练")
print(f"  fc1.lora.B.weight.requires_grad: {fc1.lora.B.weight.requires_grad}  # 可训练")

# 记录训练前的 W 和 BA
W_before = fc1.weight.data.clone()
BA_before = fc1.lora.delta_weight().clone()

# 构造一个简单目标：让 fc1(x) 接近全 1 向量
x = torch.randn(2, dim)
target = torch.ones(2, dim)
optimizer = torch.optim.SGD(lora_params, lr=0.1)

print(f"\n训练前:")
print(f"  W 范数: {W_before.norm().item():.4f}")
print(f"  BA 范数: {BA_before.norm().item():.4f}  # 初始为 0")
print(f"  BA 最大绝对值: {BA_before.abs().max().item():.2e}")

# 一步梯度下降
y = fc1(x)
loss = ((y - target) ** 2).mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 训练后
W_after = fc1.weight.data.clone()
BA_after = fc1.lora.delta_weight().clone()
y_after_train = fc1(x)

print(f"\n一步训练后:")
print(f"  W 范数: {W_after.norm().item():.4f}")
print(f"  BA 范数: {BA_after.norm().item():.4f}  # 已变化")
print(f"  BA 最大绝对值: {BA_after.abs().max().item():.4f}")
print(f"  loss: {loss.item():.4f} -> {((fc1(x)-target)**2).mean().item():.4f}")

# 验证
print(f"\n验证：")
print(f"  W 不变: {'通过' if torch.allclose(W_before, W_after) else '失败'}"
      f"  (差异 = {(W_before-W_after).abs().max().item():.2e})")
print(f"  BA 变化: {'通过' if not torch.allclose(BA_before, BA_after) else '失败'}"
      f"  (差异 = {(BA_before-BA_after).abs().max().item():.4f})")
print(f"  输出变化: {'通过' if not torch.allclose(y, y_after_train) else '失败'}")

print("\n  说明：LoRA 微调时原权重 W 冻结，只更新 A、B，训练完成后只保存")
print("  小体量的 LoRA 参数，原模型权重可复用。")

print()


# =============================================================================
# 示例 5：权重合并验证（merge: W' = W + BA）
# =============================================================================
print("=" * 70)
print("示例 5：权重合并验证（merge: W' = W + BA）")
print("=" * 70)

# 用示例 4 训练后的模型，先记录合并前的输出，再合并验证
dim = 8
rank = 4
torch.manual_seed(2)
model = SimpleMLP(dim)
apply_lora(model, rank=rank)

# 给 LoRA 一点非零增量（模拟训练后状态）
fc1 = model.fc1
with torch.no_grad():
    fc1.lora.B.weight.data.normal_(0, 0.1)  # 制造非零 BA

x = torch.randn(3, dim)
y_before_merge = fc1(x).clone()   # 合并前输出 = Wx + b + BAx

W_orig = fc1.weight.data.clone()
BA = fc1.lora.delta_weight().clone()
print(f"\n合并前:")
print(f"  W 范数: {W_orig.norm().item():.4f}")
print(f"  BA 范数: {BA.norm().item():.4f}  # 非零增量")
print(f"  fc1(x) 输出形状: {y_before_merge.shape}")

# 合并前手动验证：Wx + b + BAx
manual_before = x @ W_orig.t() + fc1.bias + x @ BA.t()
print(f"  验证 forward == Wx+b+BAx: "
      f"{'通过' if torch.allclose(y_before_merge, manual_before, atol=1e-6) else '失败'}")

# 执行合并
merge_lora_weights(fc1)   # 仅合并 fc1（演示用，简化处理）
print(f"\n合并后:")
print(f"  fc1 是否还有 lora 属性: {hasattr(fc1, 'lora')}  # 应被移除")
print(f"  W' 范数: {fc1.weight.norm().item():.4f}  # W + BA")

# 合并后输出
y_after_merge = fc1(x)

# 验证：合并后 forward = W'x + b == 合并前 Wx + b + BAx
print(f"\n验证：合并后 forward == 合并前 forward")
print(f"  最大差异: {(y_after_merge - y_before_merge).abs().max().item():.2e}")
print(f"  一致: {'通过' if torch.allclose(y_after_merge, y_before_merge, atol=1e-6) else '失败'}")

# 验证：W' = W + BA
W_merged = fc1.weight.data
print(f"\n验证：W' == W + BA")
print(f"  最大差异: {(W_merged - W_orig - BA).abs().max().item():.2e}")
print(f"  一致: {'通过' if torch.allclose(W_merged, W_orig + BA, atol=1e-6) else '失败'}")

print("\n  说明：合并后推理不再需要 LoRA 分支，前向路径变短，部署更高效；")
print("  但合并后 W 不再可逆地分离出 BA，无法继续训练 LoRA。")
print("  对应源码 model_lora.py 的 merge_lora: W' = W + B @ A。")

print()


# =============================================================================
# 示例 6：低秩直觉（低秩近似捕获主要方差）
# =============================================================================
print("=" * 70)
print("示例 6：低秩直觉（低秩近似捕获主要方差）")
print("=" * 70)

# 构造一个秩为 r 的矩阵 M = U @ V（U: [m, r], V: [r, n]）
# LoRA 的假设：微调产生的 ΔW 是低秩的，可用 BA 近似

m, n, true_rank = 64, 48, 4
torch.manual_seed(3)
# 用 float64 构造，保证 M 是数值上严格的秩 4 矩阵（float32 会有 ~1e-6 舍入扰动）
U = torch.randn(m, true_rank, dtype=torch.float64)
V = torch.randn(true_rank, n, dtype=torch.float64)
M = U @ V   # 真实秩 = true_rank = 4

print(f"\n构造低秩矩阵 M = U @ V，形状 [{m}, {n}]，真实秩 = {true_rank}")
print(f"  M 的 Frobenius 范数: {M.norm().item():.4f}")

# SVD 分解（转 float64 提升数值精度，便于验证无损重建）
M_np = M.numpy().astype(np.float64)
U_svd, S_svd, Vt_svd = np.linalg.svd(M_np, full_matrices=False)
print(f"\nSVD 分解后的奇异值（前 10 个）:")
for i in range(min(10, len(S_svd))):
    print(f"  σ_{i} = {S_svd[i]:.4f}")
print(f"  前 {true_rank} 个奇异值能量占比: "
      f"{(S_svd[:true_rank]**2).sum()/(S_svd**2).sum()*100:.2f}%  （能量 = σ^2 之和）")

# 用不同秩 k 做低秩近似，比较误差
print(f"\n不同秩 k 的低秩近似误差（截断 SVD）:")
_h6 = ["k", "捕获能量", "相对误差", "参数量(k*(m+n))", "原参数(m*n)"]
_w6 = [4, 10, 12, 18, 14]
print("  " + " | ".join(_rpad(h, _w6[i]) for i, h in enumerate(_h6)))
print("  " + "-" * (sum(_w6) + 3 * (len(_w6) - 1)))
orig_params = m * n
for k in [1, 2, 4, 8, 16, min(m, n)]:
    # 截断 SVD: M_k = U[:,:k] @ diag(S[:k]) @ Vt[:k,:]
    M_k = U_svd[:, :k] @ np.diag(S_svd[:k]) @ Vt_svd[:k, :]
    rel_err = np.linalg.norm(M_np - M_k) / np.linalg.norm(M_np)
    energy = (S_svd[:k]**2).sum() / (S_svd**2).sum() * 100
    k_params = k * (m + n)
    cells = [str(k), f"{energy:.2f}%", f"{rel_err:.2e}", str(k_params), str(orig_params)]
    print("  " + " | ".join(_rpad(c, _w6[j]) for j, c in enumerate(cells)))

print(f"\n  观察：")
print(f"    - k >= 真实秩({true_rank}) 时，相对误差为 0，完全无损")
print(f"    - k=4 时参数量 {4*(m+n)} = {4*(m+n)}，仅为原参数 {m*n} 的 {4*(m+n)/(m*n)*100:.2f}%")
print(f"    - 这正是 LoRA 的理论基础：若 ΔW 本身低秩，用 BA 近似几乎无损")

# 验证：k >= true_rank 时近似无损
M_k4 = U_svd[:, :4] @ np.diag(S_svd[:4]) @ Vt_svd[:4, :]
err_k4 = np.linalg.norm(M_np - M_k4)
print(f"\n验证：k=4 (= true_rank) 时近似无损")
print(f"  ||M - M_4|| = {err_k4:.2e}")
print(f"  无损: {'通过' if err_k4 < 1e-8 else '失败'}")

# 模拟 LoRA：用 B @ A 拟合 M（梯度下降）
# 先把 M 归一化到单位标准差，使优化条件数更好（M 原始尺度较大）
print(f"\n进一步：用 LoRA 形式 BA 拟合 M（梯度下降）")
r_fit = 6
print(f"  r_fit={r_fit} > true_rank={true_rank}，理论上可无损拟合")
M_target = M.detach() / M.std()   # 归一化目标（float64）
torch.manual_seed(0)
A_fit = torch.randn(r_fit, n, requires_grad=True, dtype=torch.float64)    # [r, n]，高斯初始化
B_fit = torch.zeros(m, r_fit, requires_grad=True, dtype=torch.float64)    # [m, r]，零初始化（LoRA 风格）
# Adam + 较小学习率，保证 B=0 初始化下也能稳定收敛
opt = torch.optim.Adam([A_fit, B_fit], lr=0.005)
for step in range(5001):
    M_hat = B_fit @ A_fit   # [m, r] @ [r, n] = [m, n]
    loss = ((M_hat - M_target) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 1000 == 0 or step == 5000:
        rel = (M_hat - M_target).norm().item() / M_target.norm().item()
        print(f"  step {step:>4}: loss={loss.item():.6e}, 相对误差={rel:.4e}")

rel_final = (B_fit @ A_fit - M_target).detach().norm().item() / M_target.norm().item()
print(f"\n  最终相对误差: {rel_final:.4e}  (r_fit={r_fit} > true_rank={true_rank})")
print(f"  LoRA 能拟合低秩矩阵: {'通过' if rel_final < 1e-3 else '失败'}")

print("\n  说明：LoRA 假设'微调增量 ΔW 是低秩的'。当 ΔW 确实低秩时，")
print("  用 rank=r 的 BA 可几乎无损拟合，参数量却远小于全量微调。")
print("  实践中 r=8~64 通常足够，是大模型微调的高效方案。")

print()
print("=" * 70)
print("所有示例运行完毕！")
print("=" * 70)
