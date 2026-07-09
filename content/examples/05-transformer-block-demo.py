"""
MiniMind Transformer Block 与整体前向示例代码
============================================

本脚本演示 Transformer Block 的组装和整体前向传播，对应教程第 5 章：
1. 单个 Transformer Block 前向（Pre-Norm + 双残差）
2. Pre-Norm vs Post-Norm 对比
3. 多层 Block 堆叠
4. LM Head 与权重绑定
5. logits_to_keep 机制
6. 交叉熵损失与标签错位对齐
7. 完整前向：input_ids → loss

运行方式：
    python 05-transformer-block-demo.py

依赖：
    - PyTorch（CPU 版本即可）

运行环境：CPU 即可运行，无需 GPU。
本示例为自包含简化实现，不依赖 MiniMind 源码。
RoPE/GQA 等注意力细节见 03-attention-demo.py，本示例聚焦 Block/Model 组装。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# 简化组件
# ---------------------------------------------------------------------------
class SimpleRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SimpleAttention(nn.Module):
    """极简多头因果注意力，用于 Block 演示。RoPE/GQA 细节见 03 示例。"""
    def __init__(self, hidden_size, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        bsz, seq, _ = x.shape
        qkv = self.qkv(x).view(bsz, seq, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                       # each [bsz, seq, n_heads, head_dim]
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # [bsz, n_heads, seq, head_dim]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.full((seq, seq), float("-inf")).triu(1)
        scores += mask
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(bsz, seq, -1)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SimpleBlock(nn.Module):
    """对应 MiniMindBlock：Pre-Norm + Attention + 残差 + Pre-Norm + FFN + 残差"""
    def __init__(self, hidden_size, n_heads, intermediate_size):
        super().__init__()
        self.input_layernorm = SimpleRMSNorm(hidden_size)
        self.self_attn = SimpleAttention(hidden_size, n_heads)
        self.post_attention_layernorm = SimpleRMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, intermediate_size)

    def forward(self, x):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = residual + x                                   # 第一个残差连接
        x = x + self.mlp(self.post_attention_layernorm(x)) # 第二个残差连接
        return x


class SimpleModel(nn.Module):
    """对应 MiniMindModel：embed + N 层 Block + final norm"""
    def __init__(self, vocab_size, hidden_size, n_layers, n_heads, intermediate_size):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            SimpleBlock(hidden_size, n_heads, intermediate_size) for _ in range(n_layers)
        ])
        self.norm = SimpleRMSNorm(hidden_size)

    def forward(self, input_ids):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


class SimpleForCausalLM(nn.Module):
    """对应 MiniMindForCausalLM：model + lm_head + 权重绑定 + 交叉熵"""
    def __init__(self, vocab_size, hidden_size, n_layers, n_heads, intermediate_size, tie_weights=True):
        super().__init__()
        self.model = SimpleModel(vocab_size, hidden_size, n_layers, n_heads, intermediate_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.model.embed_tokens.weight   # 权重绑定

    def forward(self, input_ids, labels=None, logits_to_keep=0):
        hidden = self.model(input_ids)
        if logits_to_keep > 0:
            logits = self.lm_head(hidden[:, -logits_to_keep:, :])
        else:
            logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            # 标签错位：位置 t 的 logits 预测位置 t+1 的 token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), ignore_index=-100)
        return logits, loss


torch.manual_seed(42)

# 全局配置（缩小版，便于 CPU 运行）
VOCAB, HIDDEN, N_LAYERS, N_HEADS, INTER = 30, 16, 3, 4, 32


# =============================================================================
# 示例 1：单个 Transformer Block 前向（Pre-Norm + 双残差）
# =============================================================================
print("=" * 70)
print("示例 1：单个 Transformer Block 前向（Pre-Norm + 双残差）")
print("=" * 70)

block = SimpleBlock(HIDDEN, N_HEADS, INTER)
x = torch.randn(2, 5, HIDDEN)
out = block(x)
print(f"\n输入 x shape: {x.shape}  # [batch=2, seq=5, hidden={HIDDEN}]")
print(f"Block 输出 shape: {out.shape}  # 与输入相同（残差保持维度）")

# 拆解中间过程
residual = x
h1 = block.input_layernorm(x)
h2 = block.self_attn(h1)
h3 = residual + h2                  # 第一残差后
h4 = block.post_attention_layernorm(h3)
h5 = block.mlp(h4)
h6 = h3 + h5                        # 第二残差后 = 最终输出
print(f"\n前向拆解：")
print(f"  norm1(attn) 输出 shape: {h2.shape}")
print(f"  + 残差 后 shape: {h3.shape}")
print(f"  norm2(ffn) 输出 shape: {h5.shape}")
print(f"  + 残差 后 shape: {h6.shape}")
print(f"\n验证：拆解结果与直接前向一致？{torch.allclose(out, h6, atol=1e-6)}")
print(f"  → Pre-Norm：子层输入先归一化，残差路径不经过 norm，梯度可直接回流")
print()


# =============================================================================
# 示例 2：Pre-Norm vs Post-Norm 对比
# =============================================================================
print("=" * 70)
print("示例 2：Pre-Norm vs Post-Norm 对比")
print("=" * 70)


class PostNormBlock(nn.Module):
    """Post-Norm：先子层再归一化，残差先相加再 norm"""
    def __init__(self, hidden_size, n_heads, intermediate_size):
        super().__init__()
        self.attn = SimpleAttention(hidden_size, n_heads)
        self.norm1 = SimpleRMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, intermediate_size)
        self.norm2 = SimpleRMSNorm(hidden_size)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))     # 残差先加，再 norm
        x = self.norm2(x + self.mlp(x))
        return x


prenorm = SimpleBlock(HIDDEN, N_HEADS, INTER)
postnorm = PostNormBlock(HIDDEN, N_HEADS, INTER)
x = torch.randn(2, 5, HIDDEN)

# 深层堆叠时观察输出范数变化
def stack_norm_norm(module, n, x):
    for _ in range(n):
        x = module(x)
    return x

pre_out = stack_norm_norm(prenorm, 6, x)
post_out = stack_norm_norm(postnorm, 6, x)
print(f"\n输入 x 范数: {x.norm().item():.4f}")
print(f"Pre-Norm 堆叠 6 层后输出范数: {pre_out.norm().item():.4f}")
print(f"Post-Norm 堆叠 6 层后输出范数: {post_out.norm().item():.4f}")
print(f"\n观察：Post-Norm 每层都归一化残差和，深层更易「压平」输出；")
print(f"  Pre-Norm 残差路径不归一化，深层梯度更通畅，因此大模型普遍用 Pre-Norm")
print()


# =============================================================================
# 示例 3：多层 Block 堆叠
# =============================================================================
print("=" * 70)
print("示例 3：多层 Block 堆叠（MiniMindModel）")
print("=" * 70)

model = SimpleModel(VOCAB, HIDDEN, N_LAYERS, N_HEADS, INTER)
input_ids = torch.randint(0, VOCAB, (2, 6))
print(f"\n输入 input_ids shape: {input_ids.shape}  # [batch=2, seq=6]")
print(f"配置：vocab={VOCAB}, hidden={HIDDEN}, n_layers={N_LAYERS}, n_heads={N_HEADS}")

h = model.embed_tokens(input_ids)
print(f"\n逐层追踪 hidden_states shape：")
print(f"  embed 后: {h.shape}")
for i, layer in enumerate(model.layers):
    h = layer(h)
    print(f"  block {i} 后: {h.shape}  范数={h.norm().item():.4f}")
h_final = model.norm(h)
print(f"  final norm 后: {h_final.shape}")
print(f"\n模型输出 hidden_states shape: {h_final.shape}  # [batch, seq, hidden]")
print(f"  → 各层 shape 不变，只有范数/分布随深度变化")
print()


# =============================================================================
# 示例 4：LM Head 与权重绑定
# =============================================================================
print("=" * 70)
print("示例 4：LM Head 与权重绑定")
print("=" * 70)

lm = SimpleForCausalLM(VOCAB, HIDDEN, N_LAYERS, N_HEADS, INTER, tie_weights=True)
hidden = torch.randn(2, 6, HIDDEN)
logits = lm.lm_head(hidden)

print(f"\nhidden_states shape: {hidden.shape}  # [batch, seq, hidden={HIDDEN}]")
print(f"lm_head 输出 logits shape: {logits.shape}  # [batch, seq, vocab={VOCAB}]")
print(f"\n权重绑定验证：")
print(f"  embed_tokens.weight shape: {lm.model.embed_tokens.weight.shape}")
print(f"  lm_head.weight shape: {lm.lm_head.weight.shape}")
print(f"  两者是同一内存对象？{lm.model.embed_tokens.weight.data_ptr() == lm.lm_head.weight.data_ptr()}")
print(f"  修改 embed 是否影响 lm_head？", end="")
lm.model.embed_tokens.weight.data.zero_()
print(f" {lm.lm_head.weight.abs().sum().item() == 0}（lm_head 也变 0）")

# 恢复
torch.manual_seed(42)
lm = SimpleForCausalLM(VOCAB, HIDDEN, N_LAYERS, N_HEADS, INTER, tie_weights=True)
print(f"\n  → 权重绑定节省 {VOCAB * HIDDEN} 个参数（embed 与 lm_head 共享）")
print()


# =============================================================================
# 示例 5：logits_to_keep 机制
# =============================================================================
print("=" * 70)
print("示例 5：logits_to_keep 机制")
print("=" * 70)

input_ids = torch.randint(0, VOCAB, (2, 8))
# 完整 logits
full_logits, _ = lm(input_ids)
# 只保留最后 3 个位置的 logits
keep = 3
kept_logits, _ = lm(input_ids, logits_to_keep=keep)

print(f"\ninput_ids shape: {input_ids.shape}  # [batch, seq=8]")
print(f"完整 logits shape: {full_logits.shape}  # [batch, 8, vocab]")
print(f"logits_to_keep={keep} 时 logits shape: {kept_logits.shape}  # [batch, 3, vocab]")
print(f"\n验证：保留的 logits 与完整 logits 末尾 {keep} 位一致？"
      f"{torch.allclose(kept_logits, full_logits[:, -keep:, :], atol=1e-6)}")
print(f"  → 训练/推理时只需最后几位 logits，省显存与计算（ lm_head 只算尾部）")
print()


# =============================================================================
# 示例 6：交叉熵损失与标签错位对齐
# =============================================================================
print("=" * 70)
print("示例 6：交叉熵损失与标签错位对齐（ignore_index=-100）")
print("=" * 70)

input_ids = torch.randint(0, VOCAB, (1, 5))
# 构造 labels：前 2 位是 prompt（-100 不参与 loss），后 3 位是答案
labels = torch.tensor([[-100, -100, input_ids[0, 2], input_ids[0, 3], input_ids[0, 4]]])
print(f"\ninput_ids: {input_ids.tolist()}")
print(f"labels:    {labels.tolist()}  （前 2 位 -100 表示 prompt 不计算 loss）")

logits, loss = lm(input_ids, labels=labels)
print(f"\nlogits shape: {logits.shape}  # [batch, seq=5, vocab]")
print(f"loss: {loss.item():.4f}")

# 手动验证错位：logits[:, :-1] 预测 labels[:, 1:]
shift_logits = logits[..., :-1, :].contiguous()   # 位置 0~3 的预测
shift_labels = labels[..., 1:].contiguous()        # 位置 1~4 的目标
print(f"\n错位对齐：")
print(f"  shift_logits shape: {shift_logits.shape}  # 用位置 0~3 预测")
print(f"  shift_labels shape: {shift_labels.shape}  # 目标是位置 1~4")
print(f"  对应关系：位置 0→预测 token1, 位置 1→预测 token2(prompt,-100), 位置 2→预测 token3, ...")
manual_loss = F.cross_entropy(shift_logits.view(-1, VOCAB), shift_labels.view(-1), ignore_index=-100)
print(f"\n手动计算 loss: {manual_loss.item():.4f}")
print(f"模型返回 loss: {loss.item():.4f}")
print(f"验证一致？{torch.allclose(manual_loss, loss)}")
print(f"  → -100 的位置被 cross_entropy 忽略，只对答案 token 计算 loss")
print()


# =============================================================================
# 示例 7：完整前向 input_ids → loss
# =============================================================================
print("=" * 70)
print("示例 7：完整前向（input_ids → hidden → logits → loss）")
print("=" * 70)

batch, seq = 2, 8
input_ids = torch.randint(0, VOCAB, (batch, seq))
labels = input_ids.clone()                          # 预训练：labels = input_ids
labels[:, :2] = -100                                # 模拟前 2 位不计算 loss

print(f"\n输入 input_ids shape: {input_ids.shape}  # [batch={batch}, seq={seq}]")
print(f"labels shape: {labels.shape}  （前 2 位 -100）")

# 逐步追踪 shape
embed = lm.model.embed_tokens(input_ids)
print(f"\n数据流：")
print(f"  1. embed_tokens: {input_ids.shape} → {embed.shape}")
h = embed
for i, layer in enumerate(lm.model.layers):
    h = layer(h)
print(f"  2. {N_LAYERS} 层 Block: → {h.shape}")
h = lm.model.norm(h)
print(f"  3. final norm: → {h.shape}")
logits = lm.lm_head(h)
print(f"  4. lm_head: → {logits.shape}  # [batch, seq, vocab={VOCAB}]")
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
loss = F.cross_entropy(shift_logits.view(-1, VOCAB), shift_labels.view(-1), ignore_index=-100)
print(f"  5. 错位 + 交叉熵: logits {shift_logits.shape} vs labels {shift_labels.shape}")
print(f"     → loss = {loss.item():.4f}")

# 反向传播验证
loss.backward()
grad_norm = sum(p.grad.norm().item() ** 2 for p in lm.parameters() if p.grad is not None) ** 0.5
print(f"\n反向传播后参数梯度总范数: {grad_norm:.4f}  （验证梯度可正常回流）")
print(f"\n完整流程：input_ids → embed → blocks → norm → lm_head → shift → cross_entropy → loss")
print()

print("=" * 70)
print("所有示例运行完毕！")
print("=" * 70)
