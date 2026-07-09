"""
MiniMind 训练算法 - Pretrain & SFT 示例代码
==========================================

本脚本演示预训练和监督微调的数据构造与训练流程，对应教程第 6 章：
1. PretrainDataset：BOS/EOS、padding、标签掩码
2. SFTDataset：generate_labels 仅对 assistant 回复计算 loss
3. 余弦学习率调度（get_lr）
4. 梯度累积（gradient accumulation）
5. 混合精度训练（autocast）
6. 标签错位与 loss 掩码可视化
7. 完整训练步骤模拟

运行方式：
    python 06-train-pretrain-sft-demo.py

依赖：
    - PyTorch（CPU 版本即可）

运行环境：CPU 即可运行，无需 GPU。
本示例用整数 token id 模拟 tokenizer 输出，不依赖 HuggingFace datasets/tokenizers。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 模拟常量（代替真实 tokenizer 的特殊 token）
# ---------------------------------------------------------------------------
BOS_ID, EOS_ID, PAD_ID = 1, 2, 0
# 模拟 SFT 的 assistant 段起止标记（多 token 序列）
ASSISTANT_BOS = [BOS_ID, 10]   # <bos>assistant
ASSISTANT_EOS = [EOS_ID]       # <eos>


# ---------------------------------------------------------------------------
# PretrainDataset（简化版，对应 lm_dataset.py 的 PretrainDataset）
# ---------------------------------------------------------------------------
class SimplePretrainDataset:
    """预训练数据：纯文本加 BOS/EOS，padding 到 max_length，pad 位 label=-100"""
    def __init__(self, text_token_lists, max_length=16):
        self.max_length = max_length
        self.samples = text_token_lists

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx][:self.max_length - 2]
        tokens = [BOS_ID] + tokens + [EOS_ID]                       # 加 BOS/EOS
        input_ids = tokens + [PAD_ID] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == PAD_ID] = -100                          # pad 不参与 loss
        return input_ids, labels


# ---------------------------------------------------------------------------
# SFTDataset（简化版，对应 lm_dataset.py 的 SFTDataset + generate_labels）
# ---------------------------------------------------------------------------
class SimpleSFTDataset:
    """SFT 数据：完整对话作为 input_ids，labels 只在 assistant 回复区间保留"""
    def __init__(self, conversation_token_lists, max_length=20):
        self.max_length = max_length
        self.samples = conversation_token_lists  # 每个是已 tokenize 的完整对话 token 列表

    def __len__(self):
        return len(self.samples)

    def generate_labels(self, input_ids):
        """只把 assistant 段落设为可学习标签，其余 -100"""
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(ASSISTANT_BOS)] == ASSISTANT_BOS:
                start = i + len(ASSISTANT_BOS)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(ASSISTANT_EOS)] == ASSISTANT_EOS:
                        break
                    end += 1
                # assistant 回复区间（含 EOS）设为可学习
                for j in range(start, min(end + len(ASSISTANT_EOS), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(ASSISTANT_EOS) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, idx):
        input_ids = self.samples[idx][:self.max_length]
        input_ids = input_ids + [PAD_ID] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# 余弦学习率（对应 trainer_utils.py 的 get_lr）
# ---------------------------------------------------------------------------
def get_lr(current_step, total_steps, lr):
    # 余弦退火：从 lr 平滑降到约 0.1*lr
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


# ---------------------------------------------------------------------------
# 极简模型（用于训练步骤演示）
# ---------------------------------------------------------------------------
class TinyLM(nn.Module):
    def __init__(self, vocab_size=30, hidden=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.linear = nn.Linear(hidden, vocab_size, bias=False)
        self.embed.weight = self.linear.weight   # 权重绑定

    def forward(self, input_ids, labels=None):
        logits = self.linear(self.embed(input_ids))
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), ignore_index=-100)
        return logits, loss


torch.manual_seed(42)


# =============================================================================
# 示例 1：PretrainDataset 数据构造
# =============================================================================
print("=" * 70)
print("示例 1：PretrainDataset 数据构造")
print("=" * 70)

texts = [[5, 6, 7, 8, 9], [11, 12, 13], [20, 21, 22, 23, 24, 25, 26]]
ds = SimplePretrainDataset(texts, max_length=10)
input_ids, labels = ds[0]
print(f"\n原始文本 tokens: {texts[0]}")
print(f"加 BOS/EOS 后: {[BOS_ID] + texts[0] + [EOS_ID]}")
print(f"input_ids (padding 到 10): {input_ids.tolist()}")
print(f"labels:                    {labels.tolist()}  （PAD 位=-100）")
print(f"input_ids shape: {input_ids.shape}")

# 验证：只有真实 token 的位置 label != -100
real_positions = (labels != -100).sum().item()
expected = len(texts[0]) + 2  # tokens + BOS + EOS
print(f"\n验证：非 -100 的 label 数 = {real_positions}，期望 = {expected}（tokens+BOS+EOS）")
assert real_positions == expected
print(f"  → 预训练中所有真实 token 都参与 next-token 预测，PAD 被忽略")
print()


# =============================================================================
# 示例 2：SFTDataset 标签掩码（仅 assistant 回复计算 loss）
# =============================================================================
print("=" * 70)
print("示例 2：SFTDataset 标签掩码（generate_labels）")
print("=" * 70)

# 模拟一段对话：user 问 + assistant 答
# 结构：[BOS,user] [user tokens...] [BOS,assistant] [answer tokens...] [EOS]
# 注意：user 与 assistant 用不同 role token（11 vs 10），generate_labels 只匹配
# ASSISTANT_BOS=[BOS,10] 来定位 assistant 段（对应真实实现中的 "<bos>assistant\n"）
conv = [BOS_ID, 11,  100, 101, 102,  # user 部分（role=11，不学）
        BOS_ID, 10,  200, 201, 202,  # assistant 回复（role=10，要学）
        EOS_ID]
sft_ds = SimpleSFTDataset([conv], max_length=15)
input_ids, labels = sft_ds[0]
print(f"\ninput_ids: {input_ids.tolist()}")
print(f"labels:    {labels.tolist()}")
print(f"\n解读：")
for i, (x, y) in enumerate(zip(input_ids.tolist(), labels.tolist())):
    role = "学习" if y != -100 else "忽略"
    print(f"  位置 {i}: token={x:4d}  label={y:4d}  [{role}]")
learned = (labels != -100).sum().item()
print(f"\n验证：可学习 label 数 = {learned}（仅 assistant 回复 + EOS = {len([200,201,202])+1}）")
assert learned == 4
print(f"  → SFT 只对 assistant 回复计算 loss，user/system 文本作为条件输入但不学习")
print()


# =============================================================================
# 示例 3：余弦学习率调度
# =============================================================================
print("=" * 70)
print("示例 3：余弦学习率调度（get_lr）")
print("=" * 70)

total_steps, base_lr = 100, 1e-3
print(f"\nbase_lr={base_lr}, total_steps={total_steps}")
print(f"公式: lr = base_lr * (0.1 + 0.45 * (1 + cos(π * step / total)))")
print(f"\n  step    lr         相对base")
for step in [0, 10, 25, 50, 75, 90, 100]:
    lr = get_lr(step, total_steps, base_lr)
    print(f"  {step:4d}    {lr:.6f}   {lr/base_lr:.3f}")

# 验证端点
lr_start = get_lr(0, total_steps, base_lr)
lr_end = get_lr(total_steps, total_steps, base_lr)
print(f"\n验证：step=0 时 lr={lr_start:.6f} ≈ base_lr={base_lr}？{abs(lr_start - base_lr) < 1e-9}")
print(f"验证：step=total 时 lr={lr_end:.6f} ≈ 0.1*base_lr={0.1*base_lr}？{abs(lr_end - 0.1*base_lr) < 1e-9}")
print(f"  → 从 base_lr 余弦衰减到 0.1*base_lr（不到 0，保留最小学习率防停滞）")
print()


# =============================================================================
# 示例 4：梯度累积
# =============================================================================
print("=" * 70)
print("示例 4：梯度累积（gradient accumulation）")
print("=" * 70)

model = TinyLM(vocab_size=30, hidden=16)
accum_steps = 4
real_batch = 8            # 真实想用的 batch size
micro_batch = real_batch // accum_steps  # 每次实际跑的 batch

print(f"\n目标 batch_size={real_batch}, accum_steps={accum_steps}, micro_batch={micro_batch}")
print(f"→ 显存不够时，用 {accum_steps} 次小 batch 累积梯度模拟 1 次大 batch")

# 模拟一次完整累积
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
optimizer.zero_grad()
total_loss = 0
for micro in range(accum_steps):
    input_ids = torch.randint(0, 30, (micro_batch, 6))
    labels = input_ids.clone()
    _, loss = model(input_ids, labels=labels)
    loss = loss / accum_steps          # 关键：loss 除以累积步数
    loss.backward()                    # 梯度累积
    total_loss += loss.item()
    print(f"  micro step {micro+1}: loss={loss.item()*accum_steps:.4f}（已除以{accum_steps}）, 累积梯度")

optimizer.step()                       # 累积满后才更新参数
print(f"\n累积 {accum_steps} 步后总 loss={total_loss*accum_steps:.4f}（等价于 batch={real_batch} 一次前向）")
print(f"  → 梯度累积用 {accum_steps} 次 micro_batch 的梯度之和，等价于 1 次 real_batch 的梯度")
print()


# =============================================================================
# 示例 5：混合精度训练（autocast）
# =============================================================================
print("=" * 70)
print("示例 5：混合精度训练（autocast）")
print("=" * 70)

model = TinyLM(vocab_size=30, hidden=16)
input_ids = torch.randint(0, 30, (2, 6))
labels = input_ids.clone()

# FP32 前向
model.float()
_, loss_fp32 = model(input_ids, labels=labels)
print(f"\nFP32 前向 loss: {loss_fp32.item():.6f}, logits dtype: {model.embed(input_ids).dtype}")

# 混合精度前向（bf16 在 CPU 上可用）
try:
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _, loss_amp = model(input_ids, labels=labels)
    print(f"BF16 autocast loss: {loss_amp.item():.6f}")
    print(f"  → autocast 自动把部分算子降到 bf16，省显存、加速；数值范围略变但 loss 接近")
    print(f"  loss 差异: {abs(loss_fp32.item() - loss_amp.item()):.2e}（混合精度带来的精度损失，可接受）")
except Exception as e:
    print(f"BF16 autocast 不可用: {e}")
    print(f"  → CPU 上 bf16 支持取决于硬件；GPU 用 fp16 时需配合 GradScaler 防下溢")
print()


# =============================================================================
# 示例 6：标签错位与 loss 掩码可视化
# =============================================================================
print("=" * 70)
print("示例 6：标签错位与 loss 掩码可视化")
print("=" * 70)

# 构造一个 SFT 样本：prompt(-100) + answer(真实)
input_ids = torch.tensor([[BOS_ID, 100, 101, BOS_ID, 10, 200, 201, EOS_ID]])
labels = torch.tensor([[-100, -100, -100, -100, -100, 200, 201, EOS_ID]])
print(f"\ninput_ids: {input_ids[0].tolist()}")
print(f"labels:    {labels[0].tolist()}")

# 错位：logits[:, :-1] 预测 labels[:, 1:]
print(f"\n错位对齐（位置 t 预测 t+1）：")
print(f"  {'位置':>4}  {'输入token':>8}  {'预测目标':>8}  {'是否计算loss':>12}")
for t in range(input_ids.size(1) - 1):
    in_tok = input_ids[0, t].item()
    target = labels[0, t + 1].item()
    learn = "是" if target != -100 else "否（-100）"
    print(f"  {t:4d}  {in_tok:8d}  {target:8d}  {learn:>12}")

model = TinyLM(vocab_size=300, hidden=16)
logits, loss = model(input_ids, labels=labels)
print(f"\nloss = {loss.item():.4f}（只对 answer 部分的 3 个位置计算交叉熵）")
print(f"  → prompt 区间 label=-100 被 ignore_index 跳过，模型只学「回答」不学「提问」")
print()


# =============================================================================
# 示例 7：完整训练步骤模拟
# =============================================================================
print("=" * 70)
print("示例 7：完整训练步骤模拟（forward → backward → clip → step → lr 更新）")
print("=" * 70)

model = TinyLM(vocab_size=300, hidden=16)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
total_steps = 20

# 准备 SFT 批数据
batch_data = [
    ([BOS_ID, 100, BOS_ID, 10, 200, 201, EOS_ID], [-100, -100, -100, -100, 200, 201, EOS_ID])
    for _ in range(8)
]
input_ids = torch.tensor([b[0] for b in batch_data])
labels = torch.tensor([b[1] for b in batch_data])

print(f"\n批数据: batch=8, seq=7, vocab=300")
print(f"训练 {total_steps} 步，余弦学习率：\n")
print(f"  {'step':>4}  {'lr':>10}  {'loss':>8}  {'grad_norm':>10}")
losses = []
for step in range(1, total_steps + 1):
    lr = get_lr(step - 1, total_steps, 1e-3)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    optimizer.zero_grad()
    _, loss = model(input_ids, labels=labels)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    losses.append(loss.item())
    if step % 4 == 0 or step == 1:
        print(f"  {step:4d}  {lr:10.6f}  {loss.item():8.4f}  {grad_norm.item():10.4f}")

print(f"\nloss 变化：{losses[0]:.4f} → {losses[-1]:.4f}（下降 {losses[0]-losses[-1]:.4f}）")
print(f"\n完整训练循环：")
print(f"  1. get_lr 计算当前步学习率 → 更新 optimizer")
print(f"  2. forward(input_ids, labels) → loss")
print(f"  3. loss.backward() 反向传播")
print(f"  4. clip_grad_norm_ 梯度裁剪（防爆炸）")
print(f"  5. optimizer.step() 更新参数")
print(f"  6. zero_grad 清空梯度")
print()

print("=" * 70)
print("所有示例运行完毕！")
print("=" * 70)
