"""
MiniMind 强化学习算法数值示例
============================

本脚本演示 MiniMind 中强化学习（RLHF/GRPO）相关算法的数值原理：
1. GAE 优势估计（Generalized Advantage Estimation）
2. 优势标准化（Advantage Normalization）
3. PPO clip 策略损失（Clipped Surrogate Objective）
4. GRPO 组内标准化（Group-normalized Advantage）
5. CISPO vs PPO clip 对比
6. k3 KL 估计器（Schulman 无偏低方差 KL 估计）
7. PPO vs GRPO 对比表

运行方式：
    python 08-rl-demo.py

依赖：
    - PyTorch（CPU 版本即可）
    - numpy

运行环境：CPU 即可运行，无需 GPU。

说明：本脚本不依赖完整模型，全部用 torch 张量手动构造数值演示算法。
对应 MiniMind 源码：trainer/train_ppo.py、trainer/train_grpo.py。
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


def _pad(s, width):
    """左对齐，右侧补空格至指定显示宽度（兼容 CJK 全角字符）。"""
    s = str(s)
    return s + " " * max(0, width - _disp_width(s))


def _rpad(s, width):
    """右对齐，左侧补空格至指定显示宽度（兼容 CJK 全角字符）。"""
    s = str(s)
    return " " * max(0, width - _disp_width(s)) + s


# =============================================================================
# 示例 1：GAE 优势估计（Generalized Advantage Estimation）
# =============================================================================
print("=" * 70)
print("示例 1：GAE 优势估计（Generalized Advantage Estimation）")
print("=" * 70)

# 手动构造一条轨迹：T=5 步，每步有 reward、value、done
# 对应源码 train_ppo.py 中：
#   delta = token_rewards[:, t] + gamma * nv - old_resp_values[:, t]
#   lastgaelam = delta + gamma * lam * lastgaelam
T = 5
gamma = 0.99   # 折扣因子
lam = 0.95     # GAE 偏差参数：lam=0 退化为单步 TD，lam=1 退化为 Monte-Carlo

rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.5])   # 仅在 t=0 和末尾有奖励
values  = torch.tensor([0.5, 0.6, 0.4, 0.5, 0.2])   # Critic 对 V(s_t) 的估计
dones   = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])   # 末尾结束
# 结束后 V(s_{T}) = 0（episode 终止）
next_values = torch.cat([values[1:], torch.tensor([0.0])])

print(f"\n轨迹长度 T={T}, gamma={gamma}, lam={lam}")
print(f"  rewards: {rewards.tolist()}")
print(f"  values : {values.tolist()}")
print(f"  dones  : {dones.tolist()}")
print(f"  next_values (V_{{t+1}}, 末尾为 0): {next_values.tolist()}")

# 第一步：计算 TD 误差 δ_t = r_t + γ * V_{t+1} * (1 - done_t) - V_t
deltas = rewards + gamma * next_values * (1.0 - dones) - values
print(f"\n第一步：TD 误差 δ_t = r_t + γ·V_{{t+1}}·(1-done) - V_t")
for t in range(T):
    print(f"  δ_{t} = {rewards[t].item():.3f} + {gamma}·{next_values[t].item():.3f}·{1-dones[t].item():.1f}"
          f" - {values[t].item():.3f} = {deltas[t].item():.4f}")

# 第二步：从后往前递推 A_t = Σ_{l=0}^{T-t-1} (γλ)^l · δ_{t+l}
# 递推式：A_t = δ_t + γλ · A_{t+1}
advantages = torch.zeros(T)
advantages[T - 1] = deltas[T - 1]
for t in reversed(range(T - 1)):
    advantages[t] = deltas[t] + gamma * lam * advantages[t + 1]

print(f"\n第二步：GAE 递推 A_t = δ_t + γλ·A_{{t+1}}（从后往前）")
for t in range(T):
    # 同时打印展开式的各项，便于理解
    coeffs = [(gamma * lam) ** l for l in range(T - t)]
    terms = [coeffs[l] * deltas[t + l].item() for l in range(T - t)]
    terms_str = " + ".join([f"{coeffs[l]:.4f}·{deltas[t+l].item():.4f}" for l in range(T - t)])
    print(f"  A_{t} = {terms_str} = {advantages[t].item():.4f}")

# 验证：A_T-1 = δ_{T-1}
print(f"\n验证：A_{{T-1}} = δ_{{T-1}} = {deltas[T-1].item():.4f}？"
      f" {'通过' if torch.allclose(advantages[T-1], deltas[T-1]) else '失败'}")

# returns = advantages + values（Critic 的回归目标）
returns = advantages + values
print(f"\nreturns = A + V（Critic 回归目标）: {returns.tolist()}")

print()


# =============================================================================
# 示例 2：优势标准化（Advantage Normalization）
# =============================================================================
print("=" * 70)
print("示例 2：优势标准化（Advantage Normalization）")
print("=" * 70)

# 对应源码 train_ppo.py：
#   adv_mean = (advantages * mask).sum() / mask.sum()
#   adv_var  = ((advantages - adv_mean)^2 * mask).sum() / mask.sum()
#   advantages = (advantages - adv_mean) * rsqrt(adv_var + 1e-8)

# 用上一步的 advantages 做标准化（此处不带 mask，简化演示）
adv = advantages.clone()
adv_mean = adv.mean()
adv_var = ((adv - adv_mean) ** 2).mean()
adv_std = torch.sqrt(adv_var + 1e-8)
adv_norm = (adv - adv_mean) * torch.rsqrt(adv_var + 1e-8)

print(f"\n标准化前 advantages: {[f'{x:.4f}' for x in adv.tolist()]}")
print(f"  mean = {adv_mean.item():.4f}")
print(f"  var  = {adv_var.item():.4f}")
print(f"  std  = {adv_std.item():.4f}")
print(f"\n标准化公式: A_norm = (A - mean) / sqrt(var + 1e-8)")
print(f"标准化后 advantages: {[f'{x:.4f}' for x in adv_norm.tolist()]}")

# 验证：标准化后均值 ≈ 0，方差 ≈ 1
norm_mean = adv_norm.mean().item()
norm_var = ((adv_norm - norm_mean) ** 2).mean().item()
print(f"\n验证：标准化后 mean ≈ 0 ？ mean = {norm_mean:.2e}")
print(f"验证：标准化后 var  ≈ 1 ？ var  = {norm_var:.6f}")
print(f"  均值接近 0: {'通过' if abs(norm_mean) < 1e-5 else '失败'}")
print(f"  方差接近 1: {'通过' if abs(norm_var - 1.0) < 1e-4 else '失败'}")
print("  说明：标准化使不同 batch 的优势尺度统一，训练更稳定。")

print()


# =============================================================================
# 示例 3：PPO clip 策略损失（Clipped Surrogate Objective）
# =============================================================================
print("=" * 70)
print("示例 3：PPO clip 策略损失（Clipped Surrogate Objective）")
print("=" * 70)

# 对应源码 train_ppo.py：
#   ratio = exp(log_ratio) = exp(logπ_new - logπ_old)
#   policy_loss = max(-A*ratio, -A*clamp(ratio, 1-eps, 1+eps))
#               = -min(A*ratio, A*clamp(ratio, 1-eps, 1+eps))

clip_epsilon = 0.2  # PPO 裁剪范围

# 构造 4 个 token 的新旧 logπ，覆盖 ratio 的不同区间
old_logp = torch.tensor([-1.0, -1.0, -1.0, -1.0])
new_logp = torch.tensor([-0.5, -0.9, -1.3, -2.5])  # 对应 ratio 升高/略升/略降/大降
A = torch.tensor([1.0, 1.0, 1.0, 1.0])            # 优势为正（鼓励增大概率）

ratio = torch.exp(new_logp - old_logp)
clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

# unclipped 项与 clipped 项
unclipped_obj = ratio * A
clipped_obj = clipped_ratio * A
# PPO 取两者较小值（保守更新），loss = -min(...)
surrogate = torch.min(unclipped_obj, clipped_obj)
policy_loss = -surrogate

print(f"\nclip_epsilon = {clip_epsilon}, 裁剪范围 [{1-clip_epsilon}, {1+clip_epsilon}]")
print(f"  优势 A = {A.tolist()}（正优势，鼓励提升概率）")
# 表头与每列显示宽度（兼容 CJK 全角字符）
_headers = ["token", "logπ_old", "logπ_new", "ratio", "clip(ratio)", "是否裁剪", "min目标", "loss"]
_widths = [5, 8, 8, 7, 11, 8, 8, 7]
print("\n  " + " | ".join(_rpad(h, _widths[i]) for i, h in enumerate(_headers)))
print("  " + "-" * (sum(_widths) + 3 * (len(_widths) - 1)))
for i in range(len(ratio)):
    is_clipped = (ratio[i] - clipped_ratio[i]).abs() > 1e-8
    cells = [
        str(i),
        f"{old_logp[i].item():.3f}",
        f"{new_logp[i].item():.3f}",
        f"{ratio[i].item():.3f}",
        f"{clipped_ratio[i].item():.3f}",
        "是" if is_clipped else "否",
        f"{surrogate[i].item():.4f}",
        f"{policy_loss[i].item():.4f}",
    ]
    print("  " + " | ".join(_rpad(c, _widths[k]) for k, c in enumerate(cells)))

print(f"\n  总 policy_loss（均值）= {policy_loss.mean().item():.4f}")
print("\n  观察：")
print("    - token 0: ratio=1.65 > 1.2，被裁剪到 1.2，防止正优势时概率增长过快")
print("    - token 1,2: ratio 在 [0.8,1.2] 内，未裁剪")
print("    - token 3: ratio=0.37 < 0.8，被裁剪到 0.8，但 A>0 时 min 选 0.37（更小，不裁剪）")
print("    即 PPO 只裁剪'让目标变大'的方向，保留'让目标变小'的方向（悲观更新）")

# 验证：当 ratio 在裁剪范围内，loss = -ratio*A
print(f"\n验证：未裁剪 token 的 loss == -ratio*A ？")
for i in [1, 2]:
    expected = -(ratio[i] * A[i]).item()
    ok = torch.allclose(policy_loss[i], torch.tensor(expected))
    print(f"  token {i}: loss={policy_loss[i].item():.4f}, -ratio*A={expected:.4f}, 一致={ok}")

print()


# =============================================================================
# 示例 4：GRPO 组内标准化（Group-normalized Advantage）
# =============================================================================
print("=" * 70)
print("示例 4：GRPO 组内标准化（Group-normalized Advantage）")
print("=" * 70)

# 对应源码 train_grpo.py：
#   grouped_rewards = rewards.view(-1, num_generations)
#   mean_r = grouped_rewards.mean(dim=1).repeat_interleave(num_generations)
#   std_r  = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(num_generations)
#   advantages = (rewards - mean_r) / (std_r + 1e-4)

num_generations = 4  # 每个 prompt 采样 4 条回答

# 构造 2 个 prompt，每个 4 条回答的奖励
grouped_rewards = torch.tensor([
    [0.8, 0.4, 0.6, 0.2],   # prompt 1 的 4 条回答奖励
    [0.1, 0.3, 0.9, 0.5],   # prompt 2 的 4 条回答奖励
])
num_prompts = grouped_rewards.shape[0]
rewards = grouped_rewards.view(-1)  # 展平为 [8]

print(f"\nnum_generations = {num_generations}, num_prompts = {num_prompts}")
print(f"  分组奖励 grouped_rewards:\n{grouped_rewards}")
print(f"  展平 rewards: {rewards.tolist()}")

# 组内均值与标准差
mean_r = grouped_rewards.mean(dim=1)
std_r = grouped_rewards.std(dim=1, unbiased=False)
print(f"\n  组内 mean_r: {mean_r.tolist()}")
print(f"  组内 std_r : {std_r.tolist()}")

# 展开到每条回答
mean_r_expanded = mean_r.repeat_interleave(num_generations)
std_r_expanded = std_r.repeat_interleave(num_generations)
advantages = (rewards - mean_r_expanded) / (std_r_expanded + 1e-4)

print(f"\n  GRPO 优势 A = (r - mean) / (std + 1e-4):")
print(f"  {advantages.tolist()}")

# 验证：每组内优势之和 ≈ 0（组内基线）
adv_grouped = advantages.view(num_prompts, num_generations)
group_sums = adv_grouped.sum(dim=1)
group_means = adv_grouped.mean(dim=1)
print(f"\n验证：每组内优势之和 ≈ 0（组内基线的作用）")
print(f"  组 1 优势和: {group_sums[0].item():.2e}")
print(f"  组 2 优势和: {group_sums[1].item():.2e}")
print(f"  组 1 优势均值: {group_means[0].item():.2e}")
print(f"  组 2 优势均值: {group_means[1].item():.2e}")
print(f"  组内基线验证: {'通过' if all(abs(s) < 1e-3 for s in group_sums.tolist()) else '失败'}")
print("\n  说明：GRPO 不训练 Critic，而是用同一 prompt 的多条回答互相比")
print("  较作为基线，奖励高于均值的回答得正优势，低于均值得负优势。")

print()


# =============================================================================
# 示例 5：CISPO vs PPO clip 对比
# =============================================================================
print("=" * 70)
print("示例 5：CISPO vs PPO clip 对比")
print("=" * 70)

# 对应源码 train_grpo.py：
#   PPO/GRPO clip:  clipped_ratio = clamp(ratio, 1-eps, 1+eps)
#                   loss = -min(ratio*A, clipped_ratio*A)
#   CISPO:          clamped_ratio = clamp(ratio, max=eps_high).detach()
#                   loss = -(clamped_ratio * A * logπ - beta * kl)

# 关键差异：
# 1. PPO 对 ratio 双侧裁剪 [1-eps, 1+eps]；CISPO 只对上界裁剪 (max=eps_high)
# 2. CISPO 把 ratio detach（梯度只过 logπ，不过 ratio），更稳定
# 3. PPO 用 surrogate (ratio*A)；CISPO 用策略梯度形式 (ratio * A * logπ)

eps = 0.2          # PPO 裁剪范围
eps_high = 2.0     # CISPO 上界

# 构造一组 ratio，包含范围外的大值和小值
ratios = torch.tensor([0.3, 0.9, 1.5, 3.0, 5.0])
A_pos = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])  # 正优势

print(f"\n参数: PPO eps={eps} (双侧裁剪 [{1-eps}, {1+eps}])")
print(f"      CISPO eps_high={eps_high} (仅上界裁剪)")

# PPO clip
ppo_clipped = torch.clamp(ratios, 1 - eps, 1 + eps)
ppo_obj = torch.min(ratios * A_pos, ppo_clipped * A_pos)

# CISPO（此处演示有效 ratio，即用于加权的 ratio）
cispo_clamped = torch.clamp(ratios, max=eps_high)
# CISPO 的有效更新方向由 clamp(ratio, max=eps_high) * A 决定（detach 只影响梯度，不影响前向数值）

_h5 = ["ratio", "PPO clip", "PPO有效ratio", "CISPO clamp", "CISPO有效ratio", "差异"]
_w5 = [7, 10, 14, 12, 14, 8]
print("\n  " + " | ".join(_rpad(h, _w5[i]) for i, h in enumerate(_h5)))
print("  " + "-" * (sum(_w5) + 3 * (len(_w5) - 1)))
for i in range(len(ratios)):
    ppo_eff = min(ratios[i].item(), ppo_clipped[i].item()) if A_pos[i] > 0 else ratios[i].item()
    diff = (cispo_clamped[i] - ppo_clipped[i]).item()
    cells = [
        f"{ratios[i].item():.3f}",
        f"{ppo_clipped[i].item():.3f}",
        f"{ppo_eff:.3f}",
        f"{cispo_clamped[i].item():.3f}",
        f"{cispo_clamped[i].item():.3f}",
        f"{diff:.3f}",
    ]
    print("  " + " | ".join(_rpad(c, _w5[k]) for k, c in enumerate(cells)))

print(f"\n  观察（A > 0 时）：")
print(f"    - ratio=0.3: PPO 裁剪到 0.8（限制下降），CISPO 保留 0.3（允许自由下降）")
print(f"    - ratio=3.0: PPO 裁剪到 1.2，CISPO 裁剪到 2.0（上界更宽松）")
print(f"    - ratio=5.0: PPO 裁剪到 1.2，CISPO 裁剪到 2.0")
print(f"\n  核心差异：")
print(f"    PPO clip:  双侧裁剪 ratio ∈ [0.8, 1.2]，对称保守")
print(f"    CISPO:     仅上界裁剪 ratio ≤ 2.0，允许概率自由下降，只限制上升幅度")
print(f"    CISPO 还会 detach ratio，梯度仅通过 logπ 流动，避免 ratio 梯度带来的不稳定")

# 数值验证：PPO 在 ratio<1-eps 时仍会裁剪，CISPO 不会
ratio_low = torch.tensor([0.3])
ppo_low = torch.clamp(ratio_low, 1 - eps, 1 + eps)
cispo_low = torch.clamp(ratio_low, max=eps_high)
print(f"\n验证：ratio=0.3 时")
print(f"  PPO  clip(ratio) = {ppo_low.item():.3f}（被抬高到下界 0.8）")
print(f"  CISPO clamp(ratio) = {cispo_low.item():.3f}（保持 0.3 不变）")
print(f"  CISPO 允许概率下降得更彻底: {'通过' if cispo_low.item() < ppo_low.item() else '失败'}")

print()


# =============================================================================
# 示例 6：k3 KL 估计器（Schulman 无偏低方差 KL 估计）
# =============================================================================
print("=" * 70)
print("示例 6：k3 KL 估计器（Schulman 无偏低方差 KL 估计）")
print("=" * 70)

# 对应源码 train_grpo.py：
#   kl_div = ref_per_token_logps - per_token_logps  # = log(p_ref / p_policy)
#   per_token_kl = torch.exp(kl_div) - kl_div - 1   # k3 形式
# 对应源码 train_ppo.py：
#   kl_ref_penalty = exp(ref - mb) - (ref - mb) - 1 # 同样是 k3 形式

# k3 估计器（Schulman 2015）：用从 p 采样的 x 估计 KL(p||q)
#   令 kl = log(p(x)/q(x))（单点估计，E_p[kl] = KL(p||q)，但高方差）
#   k3 = exp(-kl) - 1 + kl = q(x)/p(x) - 1 + log(p(x)/q(x))
#   E_p[k3] = KL(p||q)，且 k3 >= 0 恒成立（低方差，无负值）

# 构造两个离散分布 p（策略）和 q（参考模型）
p = torch.tensor([0.5, 0.3, 0.15, 0.05])     # 策略分布
q = torch.tensor([0.4, 0.4, 0.15, 0.05])     # 参考分布
print(f"\n分布 p (策略):    {p.tolist()}")
print(f"分布 q (参考):    {q.tolist()}")

# 真实 KL(p||q) = Σ p·log(p/q)
true_kl = (p * (torch.log(p) - torch.log(q))).sum()
print(f"\n真实 KL(p||q) = Σ p·log(p/q) = {true_kl.item():.6f}")

# k1 估计器：单点 kl = log(p/q)，期望 = KL，但单点可负（高方差）
k1 = torch.log(p) - torch.log(q)              # 每个样本的 k1
k1_mean = (p * k1).sum()                       # 期望（精确计算）
print(f"\nk1 估计器: kl = log(p/q)")
print(f"  各点 k1 = log(p/q): {k1.tolist()}")
print(f"  E_p[k1] = {k1_mean.item():.6f}  （= 真实 KL，无偏）")
print(f"  k1 最小值 = {k1.min().item():.4f}  （可为负，高方差）")

# k3 估计器：k3 = exp(-kl) - 1 + kl = q/p - 1 + log(p/q)
k3 = torch.exp(-k1) - 1 + k1                   # 各点 k3 = q/p - 1 + log(p/q)
k3_mean = (p * k3).sum()                        # 期望
print(f"\nk3 估计器: k3 = exp(-kl) - 1 + kl = q/p - 1 + log(p/q)")
print(f"  各点 k3: {[f'{x:.4f}' for x in k3.tolist()]}")
print(f"  E_p[k3] = {k3_mean.item():.6f}  （应 = 真实 KL，无偏）")
print(f"  k3 最小值 = {k3.min().item():.4f}  （恒 >= 0，低方差）")

# 对比
print(f"\n  数值对比:")
print(f"    真实 KL(p||q) = {true_kl.item():.6f}")
print(f"    E_p[k1]       = {k1_mean.item():.6f}  偏差 = {abs(k1_mean-true_kl).item():.2e}")
print(f"    E_p[k3]       = {k3_mean.item():.6f}  偏差 = {abs(k3_mean-true_kl).item():.2e}")
print(f"    k1 方差 = {((k1 - k1_mean)**2 * p).sum().item():.6f}")
print(f"    k3 方差 = {((k3 - k3_mean)**2 * p).sum().item():.6f}  （更小）")

# 验证
print(f"\n验证：")
print(f"  E_p[k1] == KL(p||q) ？ {'通过' if torch.allclose(k1_mean, true_kl, atol=1e-5) else '失败'}")
print(f"  E_p[k3] == KL(p||q) ？ {'通过' if torch.allclose(k3_mean, true_kl, atol=1e-5) else '失败'}")
print(f"  k3 恒非负 ？ {'通过' if k3.min().item() >= -1e-6 else '失败'}")
print(f"  k3 方差 < k1 方差 ？ {'通过' if ((k3-k3_mean)**2*p).sum() < ((k1-k1_mean)**2*p).sum() else '失败'}")

print("\n  说明：MiniMind 的 PPO/GRPO 都用 k3 形式计算 KL 惩罚，因为 k3 恒非负、")
print("  方差更低，训练更稳定。源码中 exp(ref-mb)-(ref-mb)-1 即 k3（kl=log(p/q)）。")

print()


# =============================================================================
# 示例 7：PPO vs GRPO 对比表
# =============================================================================
print("=" * 70)
print("示例 7：PPO vs GRPO 对比表")
print("=" * 70)

# 总结 MiniMind 中 PPO（train_ppo.py）与 GRPO（train_grpo.py）的核心差异

headers = ["对比维度", "PPO", "GRPO"]
rows = [
    ("是否需要 Critic",     "需要（Value 网络 V(s)）",      "不需要"),
    ("基线来源",            "Critic V(s) 估计的状态价值",   "同一 prompt 的多条回答奖励均值"),
    ("优势计算",            "GAE: A_t = Σ(γλ)^l δ_{t+l}",  "(r - 组内mean) / (组内std)"),
    ("TD 误差 δ_t",         "r_t + γV_{t+1} - V_t",        "不使用 TD 误差"),
    ("样本利用",            "每条轨迹单次，mini-batch 多轮复用", "每个 prompt 采样 num_generations 条"),
    ("裁剪方式",            "ratio 双侧裁剪 [1-ε, 1+ε]",    "GRPO clip 或 CISPO（仅上界）"),
    ("KL 约束",             "k3 KL ref 惩罚 (kl_coef)",     "k3 KL ref 惩罚 (beta)"),
    ("早停机制",            "approx_kl > 阈值时提前停止",    "无早停"),
    ("Value 损失",          "有（value clipping）",          "无"),
    ("额外显存",            "Actor + Critic + Ref",          "Actor + Ref（省 Critic）"),
    ("适用场景",            "奖励密集、需细粒度信用分配",    "奖励稀疏、可多次采样、省显存"),
]

# 打印表格（按显示宽度自动对齐，兼容中文全角字符）
all_rows = [headers] + [list(r) for r in rows]
col_widths = [max(_disp_width(str(r[i])) for r in all_rows) for i in range(len(headers))]
sep_line = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"

print()
print(sep_line)
print("| " + " | ".join(_pad(h, col_widths[i]) for i, h in enumerate(headers)) + " |")
print(sep_line)
for row in rows:
    print("| " + " | ".join(_pad(c, col_widths[i]) for i, c in enumerate(row)) + " |")
print(sep_line)

print("\n  总结：")
print("    PPO 依赖 Critic 提供逐 token 的价值基线，适合长序列、奖励密集的场景；")
print("    GRPO 用'同一 prompt 多次采样'的组内统计代替 Critic，省去价值网络，")
print("    显存更省、实现更简，是当前开源 RLHF 的主流方案（如 DeepSeek-R1）。")
print("    MiniMind 同时提供两种实现，便于对比学习。")

print()
print("=" * 70)
print("所有示例运行完毕！")
print("=" * 70)
