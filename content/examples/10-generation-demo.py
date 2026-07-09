"""
MiniMind 推理生成算法示例代码
============================

本脚本演示大语言模型推理生成阶段的各种采样策略，对应教程第 10 章：
1. 贪心解码（argmax）
2. 温度采样（temperature）
3. Top-K 采样
4. Top-P（nucleus）采样
5. 重复惩罚（repetition penalty）
6. 组合策略（temperature + top-k + top-p + repetition penalty）
7. EOS 批量处理

运行方式：
    python 10-generation-demo.py

依赖：
    - PyTorch（CPU 版本即可）

运行环境：CPU 即可运行，无需 GPU。
本示例用极小的随机权重 Linear 作为 logit 生成器，不依赖完整模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 极简 logit 生成器：用一个线性层模拟"模型对下一个 token 的打分"
# 真实模型中 logits = lm_head(hidden_states)，这里用随机权重替代
# ---------------------------------------------------------------------------
class ToyLogitModel(nn.Module):
    def __init__(self, vocab_size=20, hidden_size=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        # input_ids: [batch, seq] → logits: [batch, seq, vocab]
        h = self.embed(input_ids)
        return self.linear(h)


torch.manual_seed(42)
vocab_size = 12
model = ToyLogitModel(vocab_size=vocab_size, hidden_size=16)


# =============================================================================
# 示例 1：贪心解码（argmax）
# =============================================================================
print("=" * 70)
print("示例 1：贪心解码（argmax）")
print("=" * 70)

input_ids = torch.tensor([[1, 3, 5]])  # [batch=1, seq=3]
logits = model(input_ids)[:, -1, :]    # 取最后一个位置的 logits
print(f"\n输入 input_ids: {input_ids.tolist()}")
print(f"最后位置 logits shape: {logits.shape}  # [batch=1, vocab_size={vocab_size}]")
print(f"logits: {[round(v,3) for v in logits[0].tolist()]}")

probs = F.softmax(logits, dim=-1)
greedy_token = torch.argmax(logits, dim=-1)
print(f"\nsoftmax 后概率: {[round(v,4) for v in probs[0].tolist()]}")
print(f"贪心解码选择的 token: {greedy_token.item()}（概率最高的）")
print(f"该 token 概率: {probs[0, greedy_token].item():.4f}")
print(f"\n特点：贪心解码每步取概率最大的 token，输出确定但容易重复、缺乏多样性")
print()


# =============================================================================
# 示例 2：温度采样（temperature）
# =============================================================================
print("=" * 70)
print("示例 2：温度采样（temperature）")
print("=" * 70)

print(f"\n原始 logits: {[round(v,3) for v in logits[0].tolist()]}")
for temp in [0.5, 1.0, 2.0]:
    scaled = logits / temp
    probs_t = F.softmax(scaled, dim=-1)
    top1 = probs_t.max().item()
    entropy = -(probs_t * (probs_t + 1e-20).log()).sum().item()
    print(f"\n  temperature={temp}:")
    print(f"    概率: {[round(v,4) for v in probs_t[0].tolist()]}")
    print(f"    最高概率: {top1:.4f}，熵: {entropy:.4f}")

print(f"\n观察：")
print(f"  temperature < 1 → 分布更尖锐（倾向高概率 token），确定性增强")
print(f"  temperature > 1 → 分布更平坦（低概率 token 机会增加），多样性增强")
print(f"  temperature → 0  → 退化为贪心解码")
print()


# =============================================================================
# 示例 3：Top-K 采样
# =============================================================================
print("=" * 70)
print("示例 3：Top-K 采样")
print("=" * 70)

top_k = 5
probs = F.softmax(logits, dim=-1)
topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)

# 把非 top-k 的 logits 设为 -inf
filtered = logits.clone()
filtered[filtered < topk_vals[..., -1, None]] = float("-inf")
probs_topk = F.softmax(filtered, dim=-1)

print(f"\n原始概率（{vocab_size} 个 token）: {[round(v,4) for v in probs[0].tolist()]}")
print(f"Top-{top_k} 的 token 索引: {topk_idx[0].tolist()}")
print(f"Top-{top_k} 概率（未归一化）: {[round(v,4) for v in topk_vals[0].tolist()] if False else [round(probs[0,i].item(),4) for i in topk_idx[0].tolist()]}")
print(f"Top-{top_k} 重新归一化后概率: {[round(v,4) for v in probs_topk[0].tolist() if v > 0]}")
print(f"\n被屏蔽的 token 概率变为 0，保留的 {top_k} 个 token 概率重新归一化（和为 1）")
assert torch.isinf(filtered[0, filtered[0] == float("-inf")]).any()
print(f"验证：非 top-{top_k} 的 logits 已设为 -inf")
print()


# =============================================================================
# 示例 4：Top-P（nucleus）采样
# =============================================================================
print("=" * 70)
print("示例 4：Top-P（nucleus）采样")
print("=" * 70)

top_p = 0.6
sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
sorted_probs = F.softmax(sorted_logits, dim=-1)
cumulative = torch.cumsum(sorted_probs, dim=-1)

print(f"\n按概率降序排列:")
print(f"  排序后 token 索引: {sorted_idx[0].tolist()}")
print(f"  排序后概率: {[round(v,4) for v in sorted_probs[0].tolist()]}")
print(f"  累计概率:   {[round(v,4) for v in cumulative[0].tolist()]}")

# 找到累计概率超过 top_p 的位置，把它们屏蔽
# 注意 MiniMind 的技巧：mask 右移一位，保留"刚好达到 top_p"的那个 token
mask = cumulative > top_p
mask[..., 1:] = mask[..., :-1].clone()
mask[..., 0] = False
print(f"\n  屏蔽掩码（True=移除）: {mask[0].tolist()}")
print(f"  Top-P={top_p} 保留的 token 索引: {sorted_idx[0][~mask[0]].tolist()}")
print(f"  保留 token 概率: {[round(v,4) for v in sorted_probs[0][~mask[0]].tolist()]}")

# 应用掩码
sorted_logits[mask] = float("-inf")
# 恢复原始顺序
final_logits = torch.full_like(logits, float("-inf"))
final_logits.scatter_(1, sorted_idx, sorted_logits)
probs_topp = F.softmax(final_logits, dim=-1)
print(f"\n  Top-P 采样后概率（原始顺序）: {[round(v,4) for v in probs_topp[0].tolist()]}")
print(f"\n观察：Top-P 保留的高概率 token 数量是动态的，概率分布越集中保留越少")
print()


# =============================================================================
# 示例 5：重复惩罚（repetition penalty）
# =============================================================================
print("=" * 70)
print("示例 5：重复惩罚（repetition penalty）")
print("=" * 70)

# 模拟已生成的 token 序列
generated = torch.tensor([[1, 3, 5, 1, 3]])  # 1 和 3 已重复出现
rep_penalty = 1.3

logits_raw = model(generated)[:, -1, :].clone()
print(f"\n已生成 token: {generated[0].tolist()}")
print(f"原始 logits: {[round(v,3) for v in logits_raw[0].tolist()]}")

# 对已出现过的 token 施加惩罚：正 logit 除以惩罚系数，负 logit 乘以惩罚系数
seen = torch.unique(generated[0])
penalized = logits_raw.clone()
score = penalized[0, seen]
penalized[0, seen] = torch.where(score > 0, score / rep_penalty, score * rep_penalty)

print(f"\n已出现过的 token: {seen.tolist()}")
print(f"惩罚后 logits: {[round(v,3) for v in penalized[0].tolist()]}")
print(f"\n对比（已出现 token 的 logits 变化）:")
for t in seen.tolist():
    before = logits_raw[0, t].item()
    after = penalized[0, t].item()
    print(f"  token {t}: {before:.3f} → {after:.3f}（{'降低' if after < before else '降低'}）")

probs_before = F.softmax(logits_raw, dim=-1)
probs_after = F.softmax(penalized, dim=-1)
print(f"\n惩罚前后已出现 token 的概率:")
for t in seen.tolist():
    print(f"  token {t}: {probs_before[0,t].item():.4f} → {probs_after[0,t].item():.4f}")
print(f"\n观察：重复惩罚降低已出现 token 的概率，减少模型重复同一内容的倾向")
print(f"  非对称处理：正 logit 除以系数（缩小），负 logit 乘以系数（更负），保证单调性")
print()


# =============================================================================
# 示例 6：组合策略（temperature + top-k + top-p + repetition penalty）
# =============================================================================
print("=" * 70)
print("示例 6：组合策略（完整采样流程）")
print("=" * 70)


def sample_token(model, input_ids, temperature=0.8, top_k=5, top_p=0.8,
                 repetition_penalty=1.2, do_sample=True):
    """模拟 MiniMind generate 的单步采样逻辑"""
    logits = model(input_ids)[:, -1, :]
    # 1. 温度
    logits = logits / temperature
    # 2. 重复惩罚
    if repetition_penalty != 1.0:
        for i in range(input_ids.shape[0]):
            seen = torch.unique(input_ids[i])
            score = logits[i, seen]
            logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
    # 3. Top-K
    if top_k > 0:
        logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = float("-inf")
    # 4. Top-P
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = 0
        logits[mask.scatter(1, sorted_indices, mask)] = float("-inf")
    # 5. 采样或贪心
    probs = F.softmax(logits, dim=-1)
    if do_sample:
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
    return next_token, probs


input_ids = torch.tensor([[1, 3, 5]])
print(f"\n初始 input_ids: {input_ids[0].tolist()}")
print(f"采样参数: temperature=0.8, top_k=5, top_p=0.8, repetition_penalty=1.2\n")

# 生成 8 个 token
print("生成过程:")
for step in range(8):
    next_token, probs = sample_token(model, input_ids, temperature=0.8, top_k=5,
                                     top_p=0.8, repetition_penalty=1.2, do_sample=True)
    # 统计有效候选数（概率>0 的 token）
    n_candidates = (probs > 0).sum().item()
    print(f"  步骤 {step+1}: 候选数={n_candidates}, 选中 token={next_token.item()}, "
          f"其概率={probs[0, next_token].item():.4f}")
    input_ids = torch.cat([input_ids, next_token], dim=-1)

print(f"\n最终生成序列: {input_ids[0].tolist()}")
print(f"\n完整流程顺序：logits → /temperature → 重复惩罚 → top-k → top-p → softmax → 多项分布采样")
print()


# =============================================================================
# 示例 7：EOS 批量处理
# =============================================================================
print("=" * 70)
print("示例 7：EOS 批量处理（不同序列异步结束）")
print("=" * 70)

batch = 3
input_ids = torch.tensor([[1, 2], [3, 4], [5, 6]])
finished = torch.zeros(batch, dtype=torch.bool)
eos_token_id = 11
max_new_tokens = 8

print(f"\n批次大小: {batch}, EOS token id: {eos_token_id}, 最大生成长度: {max_new_tokens}")
print(f"初始 finished: {finished.tolist()}\n")

# 模拟生成（用随机 logit，但强制让某些序列在某步生成 EOS）
torch.manual_seed(7)
for step in range(max_new_tokens):
    logits = torch.randn(batch, vocab_size)
    # 人为设定：batch 0 在第 2 步、batch 1 在第 4 步生成 EOS
    if step == 1:
        logits[0, eos_token_id] = 100.0
    if step == 3:
        logits[1, eos_token_id] = 100.0
    next_token = torch.argmax(logits, dim=-1)
    # 已结束的序列强制输出 EOS（占位）
    next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
    # 更新 finished 标记
    finished |= next_token.eq(eos_token_id)
    input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    print(f"  步骤 {step+1}: next_tokens={next_token.tolist()}, finished={finished.tolist()}")
    if finished.all():
        print(f"  → 所有序列已结束，提前停止")
        break

print(f"\n最终序列:")
for i in range(batch):
    seq = input_ids[i].tolist()
    eos_pos = seq.index(eos_token_id) if eos_token_id in seq[2:] else len(seq)
    print(f"  序列 {i}: {seq}  （实际内容到位置 {eos_pos}，之后为 EOS 占位）")
print(f"\n观察：不同序列在不同步生成 EOS，已结束序列用 finished 标记跳过，避免影响后续计算")
print(f"  → 全部序列结束后提前 break，节省计算")
print()

print("=" * 70)
print("所有示例运行完毕！")
print("=" * 70)
