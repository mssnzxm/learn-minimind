"""
MiniMind 训练算法 - DPO 偏好优化示例代码
========================================

本脚本演示直接偏好优化（DPO）的核心计算，对应教程第 7 章：
1. logits_to_log_probs：gather 取目标 token 的 log 概率
2. DPO 损失公式：-log σ(β(Δlogπ_chosen - Δlogπ_rejected))
3. 隐式奖励 = β(logπ_θ - logπ_ref)
4. chosen/rejected 批量拼接（单次前向）
5. 损失掩码（仅 assistant 回复区域）
6. 完整 DPO 训练步骤（policy + ref 双模型）
7. β 参数对优化的影响

运行方式：
    python 07-dpo-demo.py

依赖：
    - PyTorch（CPU 版本即可）

运行环境：CPU 即可运行，无需 GPU。
本示例用极简模型模拟，不依赖 MiniMind 源码。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# logits_to_log_probs（对应 train_dpo.py L25-L31）
# ---------------------------------------------------------------------------
def logits_to_log_probs(logits, labels):
    """
    logits: [batch, seq, vocab]
    labels: [batch, seq]  （每个位置要预测的目标 token id）
    return: [batch, seq]  每个位置目标 token 的 log 概率
    """
    log_probs = F.log_softmax(logits, dim=2)                              # [batch, seq, vocab]
    # gather 按 labels 取出"真实下一个 token"的 logprob
    log_probs_per_token = torch.gather(log_probs, dim=2,
                                       index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


# ---------------------------------------------------------------------------
# dpo_loss（对应 train_dpo.py L35-L50）
# ---------------------------------------------------------------------------
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    ref_log_probs / policy_log_probs: [batch, seq]
    mask: [batch, seq]  仅 assistant 区域为 1
    batch 维前半是 chosen，后半是 rejected（已拼接）
    """
    # 只统计 assistant 回复区域的 logprob（按 mask 求和得序列总 log 概率）
    ref_sum = (ref_log_probs * mask).sum(dim=1)             # [batch]
    policy_sum = (policy_log_probs * mask).sum(dim=1)       # [batch]
    # 切分 chosen / rejected（前半 chosen，后半 rejected）
    n = ref_sum.shape[0] // 2
    chosen_ref, reject_ref = ref_sum[:n], ref_sum[n:]
    chosen_pol, reject_pol = policy_sum[:n], policy_sum[n:]
    # logits = (π_chosen - π_rejected) - (ref_chosen - ref_rejected)
    pi_logratios = chosen_pol - reject_pol
    ref_logratios = chosen_ref - reject_ref
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean(), logits


# ---------------------------------------------------------------------------
# 极简模型（policy / ref 共用结构）
# ---------------------------------------------------------------------------
class TinyLM(nn.Module):
    def __init__(self, vocab_size=20, hidden=16, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab_size, hidden)
        self.linear = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids):
        return self.linear(self.embed(input_ids))   # [batch, seq, vocab]


torch.manual_seed(42)


# =============================================================================
# 示例 1：logits_to_log_probs
# =============================================================================
print("=" * 70)
print("示例 1：logits_to_log_probs（gather 取目标 token 的 log 概率）")
print("=" * 70)

vocab = 8
logits = torch.randn(1, 4, vocab)
labels = torch.tensor([[2, 5, 1, 4]])
log_probs = logits_to_log_probs(logits, labels)

print(f"\nlogits shape: {logits.shape}  # [batch=1, seq=4, vocab={vocab}]")
print(f"labels: {labels.tolist()}")
print(f"log_probs shape: {log_probs.shape}  # [batch, seq]")
print(f"log_probs: {[round(v,4) for v in log_probs[0].tolist()]}")

# 手动验证：log_probs[t] 应等于 log_softmax(logits[t])[labels[t]]
manual = []
for t in range(4):
    lp = F.log_softmax(logits[0, t], dim=-1)[labels[0, t]].item()
    manual.append(round(lp, 4))
print(f"手动计算: {manual}")
print(f"\n验证一致？{torch.allclose(log_probs[0], torch.tensor(manual), atol=1e-5)}")
print(f"  → gather 高效地从 vocab 维「挑出」目标 token 的 log 概率，避免全量计算")
print()


# =============================================================================
# 示例 2：DPO 损失公式
# =============================================================================
print("=" * 70)
print("示例 2：DPO 损失公式 -log σ(β·logits)")
print("=" * 70)

# 模拟：policy 更偏好 chosen，ref 中性
# Δπ = logπ_chosen - logπ_rejected；Δref 类似
beta = 0.1
cases = [
    ("policy 偏好 chosen（理想）", 2.0, 0.0),
    ("policy 中性（无改进）",       0.0, 0.0),
    ("policy 偏好 rejected（恶化）", -2.0, 0.0),
]
print(f"\nβ = {beta}")
print(f"  {'场景':<28} {'Δπ':>6} {'Δref':>6} {'logits':>8} {'loss':>8}")
for desc, dpi, dref in cases:
    logits_val = dpi - dref
    loss_val = (-F.logsigmoid(torch.tensor(beta * logits_val))).item()
    print(f"  {desc:<26} {dpi:6.1f} {dref:6.1f} {logits_val:8.2f} {loss_val:8.4f}")

print(f"\n观察：")
print(f"  policy 越「偏好 chosen」（logits 越大）→ loss 越小（接近 0）")
print(f"  policy 「偏好 rejected」（logits 负）→ loss 越大（惩罚）")
print(f"  β 控制偏离 ref 的强度，β 大则对偏差更敏感")
print()


# =============================================================================
# 示例 3：隐式奖励 = β(logπ_θ - logπ_ref)
# =============================================================================
print("=" * 70)
print("示例 3：隐式奖励 r̂ = β(logπ_θ - logπ_ref)")
print("=" * 70)

# DPO 的核心洞察：无需显式奖励模型，logπ_θ - logπ_ref 即隐式奖励
logp_chosen_ref = torch.tensor([-12.0, -12.0])
logp_chosen_pol = torch.tensor([-10.0, -14.0])   # case0: policy 提高；case1: policy 降低
beta = 0.1

print(f"\n  {'case':>6} {'logπ_ref':>10} {'logπ_θ':>10} {'隐式奖励r̂':>12} {'解读':<16}")
for i in range(2):
    r = beta * (logp_chosen_pol[i] - logp_chosen_ref[i]).item()
    judge = "policy 喜爱↑" if r > 0 else "policy 厌恶↓"
    print(f"  {i:6d} {logp_chosen_ref[i]:10.2f} {logp_chosen_pol[i]:10.2f} {r:12.4f} {judge}")
print(f"\n  → DPO 通过约束 logπ_θ 不偏离 logπ_ref 太多（KL 约束），")
print(f"    把「偏好学习」转化为「提高 chosen 相对 rejected 的对数概率差」")
print()


# =============================================================================
# 示例 4：chosen/rejected 批量拼接（单次前向）
# =============================================================================
print("=" * 70)
print("示例 4：chosen/rejected 批量拼接（单次前向省算力）")
print("=" * 70)

chosen_ids = torch.tensor([[1, 2, 3, 4, 5]])      # [1, seq]
rejected_ids = torch.tensor([[1, 2, 3, 6, 7]])
# 拼接：前半 chosen，后半 rejected
batch_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
print(f"\nchosen_ids shape: {chosen_ids.shape}")
print(f"rejected_ids shape: {rejected_ids.shape}")
print(f"拼接后 batch_ids shape: {batch_ids.shape}  # [2, seq]（前半 chosen 后半 rejected）")

model = TinyLM(vocab_size=20, hidden=16, seed=1)
logits = model(batch_ids)
print(f"\n单次前向 logits shape: {logits.shape}  # [2, seq, vocab]")
print(f"  → 一次前向同时算 chosen 和 rejected，避免两次独立前向的开销")
print(f"  dpo_loss 内部用 batch_size//2 切分回 chosen/rejected")
print()


# =============================================================================
# 示例 5：损失掩码（仅 assistant 回复区域）
# =============================================================================
print("=" * 70)
print("示例 5：损失掩码（mask 仅 assistant 区域为 1）")
print("=" * 70)

# 模拟：prompt 区 mask=0，answer 区 mask=1
seq_len = 6
mask = torch.tensor([[0, 0, 1, 1, 1, 0]])   # 位置 2,3,4 是 answer，0,1 是 prompt，5 是 pad
logp = torch.tensor([[-3.0, -2.5, -1.0, -1.2, -0.8, -5.0]])
print(f"\nlog_probs: {[round(v,2) for v in logp[0].tolist()]}")
print(f"mask:      {mask[0].tolist()}  （1=answer 区域参与求和）")

sum_masked = (logp * mask).sum(dim=1).item()
sum_all = logp.sum(dim=1).item()
print(f"\n带 mask 求和（仅 answer）: {sum_masked:.2f}")
print(f"不带 mask 求和（全部）: {sum_all:.2f}")
print(f"  → mask 确保只对 assistant 回复 token 累加 log 概率，prompt 不影响偏好比较")
print()


# =============================================================================
# 示例 6：完整 DPO 训练步骤（policy + ref 双模型）
# =============================================================================
print("=" * 70)
print("示例 6：完整 DPO 训练步骤（policy 可训练 + ref 冻结）")
print("=" * 70)

vocab_size = 20
policy = TinyLM(vocab_size, 16, seed=10)
ref = TinyLM(vocab_size, 16, seed=10)          # ref 与 policy 同起点（from SFT）
for p in ref.parameters():
    p.requires_grad_(False)                     # ref 冻结

beta = 0.1
# 构造一对 chosen/rejected（相同 prompt，不同回答）
chosen = torch.tensor([[1, 2, 3, 4, 5]])
rejected = torch.tensor([[1, 2, 3, 6, 7]])
labels = torch.cat([chosen, rejected], dim=0)   # 用 input_ids 自身作 labels
mask = torch.ones_like(labels, dtype=torch.float)
mask[:, :2] = 0                                  # 前 2 位 prompt 不算

optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2)
print(f"\n初始：policy 与 ref 同权重（from SFT 起点）")
print(f"β={beta}, prompt 长度=2, answer 长度=3\n")

print(f"  {'step':>4} {'loss':>8} {'logits':>8} {'π_chosen':>10} {'π_rejected':>10}")
for step in range(1, 9):
    batch_ids = torch.cat([chosen, rejected], dim=0)
    # policy 前向（可训练）
    pol_logits = policy(batch_ids)
    pol_logp = logits_to_log_probs(pol_logits, labels)
    # ref 前向（不计算梯度）
    with torch.no_grad():
        ref_logits = ref(batch_ids)
        ref_logp = logits_to_log_probs(ref_logits, labels)
    loss, logits_val = dpo_loss(ref_logp, pol_logp, mask, beta)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 监控 chosen/rejected 的序列 log 概率
    pol_sum = (pol_logp * mask).sum(dim=1)
    if step % 2 == 0 or step == 1:
        print(f"  {step:4d} {loss.item():8.4f} {logits_val.item():8.3f} "
              f"{pol_sum[0].item():10.3f} {pol_sum[1].item():10.3f}")

print(f"\n观察：训练后 logits 增大（policy 相对 ref 更偏好 chosen），")
print(f"  π_chosen 上升、π_rejected 下降 → policy 学会「喜欢 chosen 回答」")
print()


# =============================================================================
# 示例 7：β 参数对优化的影响
# =============================================================================
print("=" * 70)
print("示例 7：β 参数对优化的影响")
print("=" * 70)

print(f"\nβ 控制 policy 偏离 ref 的强度：")
print(f"  {'β':>6} {'logits=2 时 loss':>16} {'logits=-2 时 loss':>18} {'解读':<20}")
for beta in [0.01, 0.1, 0.5, 1.0]:
    loss_pos = (-F.logsigmoid(torch.tensor(beta * 2.0))).item()
    loss_neg = (-F.logsigmoid(torch.tensor(beta * -2.0))).item()
    desc = "β大→约束强" if beta >= 0.5 else "β小→约束弱"
    print(f"  {beta:6.2f} {loss_pos:16.4f} {loss_neg:18.4f} {desc}")

print(f"\n  β 小：policy 可自由偏离 ref（学得快但可能跑偏）")
print(f"  β 大：强约束 logπ_θ≈logπ_ref（稳定但学得慢）")
print(f"  MiniMind 默认 β=0.1，平衡学习效率与 KL 约束")
print()

print("=" * 70)
print("所有示例运行完毕！")
print("=" * 70)
