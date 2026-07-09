# MiniMind 大模型神经网络结构与算法原理教程

## 教程简介

本教程基于 MiniMind 项目，系统讲解 Transformer Decoder 架构的各个组件。通过结合实际代码，我们将标注每个网络结构的输入输出张量形状、数据类型和计算逻辑，帮助你从代码层面深入理解大语言模型的工作原理。

教程的核心特点：
- **代码驱动**：每一个知识点都对应 MiniMind 项目中的真实代码
- **张量可视化**：每个网络组件都有详细的输入/输出张量 shape 说明
- **循序渐进**：从基础组件到完整模型，从预训练到强化学习
- **动手实践**：每个核心组件都附带可运行的最小示例代码

## 学习路径建议

建议按照以下顺序学习，由浅入深、循序渐进：

```
基础组件 → 注意力机制 → 前馈网络 → Transformer Block
     ↓
  训练算法（Pretrain → SFT → DPO → PPO/GRPO）
     ↓
  高效微调（LoRA）→ 推理生成
```

如果你是初学者：
1. 先完整阅读第 1 章，建立整体认知
2. 按顺序学习第 2-5 章，掌握模型结构
3. 根据兴趣选择训练算法或推理生成继续深入

如果你有一定基础：
- 可以直接跳转到感兴趣的章节
- 每章都有独立的代码引用，可以按需查阅

## 前置知识

学习本教程需要具备以下基础知识：

- **Python 基础**：熟悉 Python 语法、类与对象、列表推导式等
- **PyTorch 基础**：了解张量（Tensor）、`nn.Module`、前向传播、反向传播等概念
- **线性代数基础**：矩阵乘法、转置等基本运算

> 如果你对 PyTorch 还不熟悉，建议先学习 PyTorch 官方入门教程，再开始本教程的学习。

## 环境配置说明

本教程的示例代码只需要 **PyTorch** 即可运行：

```bash
pip install torch
```

> 所有示例代码都可以在 CPU 上运行，无需 GPU。如果你有 GPU 并安装了 CUDA 版本的 PyTorch，示例代码会自动使用 GPU。

查看 MiniMind 完整项目代码：
- 模型定义：[`model/model_minimind.py`](../../model/model_minimind.py)
- 训练脚本：[`trainer/`](../../trainer/)

## 如何使用示例代码

示例代码位于 `examples/` 目录下，每个示例都是独立可运行的 Python 脚本：

```bash
# 进入示例代码目录
cd learn/tutorial/examples

# 运行某个示例
python 03-attention-demo.py
```

每个示例脚本：
- 只依赖 PyTorch，不需要训练完整模型
- 包含详细注释，解释每一步在做什么
- 运行后会打印张量形状和关键结果，便于理解

---

## 章节目录索引

| 章节 | 标题 | 文件 |
|------|------|------|
| 第 1 章 | 模型架构总览 | （本章，见下文） |
| 第 2 章 | 基础组件 - Embedding & RMSNorm & RoPE | [chapters/02-basic-components.md](chapters/02-basic-components.md) |
| 第 3 章 | 注意力机制 Attention | [chapters/03-attention.md](chapters/03-attention.md) |
| 第 4 章 | 前馈网络 MLP & MoE | [chapters/04-mlp-moe.md](chapters/04-mlp-moe.md) |
| 第 5 章 | Transformer Block 与整体前向 | [chapters/05-transformer-block.md](chapters/05-transformer-block.md) |
| 第 6 章 | 训练算法 - Pretrain & SFT | [chapters/06-train-pretrain-sft.md](chapters/06-train-pretrain-sft.md) |
| 第 7 章 | 训练算法 - DPO 偏好优化 | [chapters/07-train-dpo.md](chapters/07-train-dpo.md) |
| 第 8 章 | 训练算法 - PPO & GRPO 强化学习 | [chapters/08-train-rl-ppo-grpo.md](chapters/08-train-rl-ppo-grpo.md) |
| 第 9 章 | LoRA 低秩适配 | [chapters/09-lora.md](chapters/09-lora.md) |
| 第 10 章 | 推理生成算法 | [chapters/10-inference-generation.md](chapters/10-inference-generation.md) |

## 示例代码索引

| 示例文件 | 对应章节 | 说明 |
|----------|----------|------|
| [`02-basic-components-demo.py`](examples/02-basic-components-demo.py) | 第 2 章 | Embedding、RMSNorm、RoPE 的最小可运行示例 |
| [`03-attention-demo.py`](examples/03-attention-demo.py) | 第 3 章 | 注意力机制的完整计算流程演示 |
| [`04-mlp-moe-demo.py`](examples/04-mlp-moe-demo.py) | 第 4 章 | SwiGLU MLP 和 MoE 混合专家模型演示 |
| [`05-transformer-block-demo.py`](examples/05-transformer-block-demo.py) | 第 5 章 | Transformer Block 前向、Pre-Norm、权重绑定与标签错位演示 |
| [`06-train-pretrain-sft-demo.py`](examples/06-train-pretrain-sft-demo.py) | 第 6 章 | Pretrain/SFT 数据构造、余弦学习率、梯度累积与混合精度演示 |
| [`07-dpo-demo.py`](examples/07-dpo-demo.py) | 第 7 章 | DPO logits_to_log_probs、dpo_loss 与隐式奖励演示 |
| [`08-rl-demo.py`](examples/08-rl-demo.py) | 第 8 章 | GAE 优势估计、PPO clip、GRPO 数值示例 |
| [`09-lora-demo.py`](examples/09-lora-demo.py) | 第 9 章 | LoRA 低秩适配原理和权重合并演示 |
| [`10-generation-demo.py`](examples/10-generation-demo.py) | 第 10 章 | 温度采样、Top-K、Top-P 等生成策略演示 |

---

## 第 1 章：模型架构总览

### 1.1 MiniMindForCausalLM 整体结构

MiniMind 是一个 **Decoder-only** 的 Transformer 语言模型，采用 Pre-Norm 结构。整体可以分为三层：

1. **MiniMindForCausalLM**：最外层封装，包含模型主体 + LM Head + 损失计算
2. **MiniMindModel**：模型主体，Embedding + 多层 Transformer Block + 最终 Norm
3. **MiniMindBlock**：单个 Transformer 层，Attention + MLP + 残差连接

参考代码位置：[`model/model_minimind.py`](../../model/model_minimind.py) 的 `MiniMindForCausalLM` 类

### 1.2 数据流描述：从 input_ids 到 logits/loss

```
输入: input_ids [batch, seq_len] (long)
  │
  ▼
┌─────────────────────────────┐
│   Token Embedding           │  → 词向量嵌入
│   [batch, seq_len, hidden_size]
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│   RMSNorm                   │  ← 第 1 层 Pre-Norm
│   Attention (GQA + RoPE)    │  ← 注意力机制
│   残差连接                   │
│   RMSNorm                   │  ← 第 2 层 Pre-Norm
│   FeedForward (SwiGLU/MoE)  │  ← 前馈网络
│   残差连接                   │
└─────────────────────────────┘
  │  × N 层 (num_hidden_layers)
  ▼
┌─────────────────────────────┐
│   最终 RMSNorm              │
│   [batch, seq_len, hidden_size]
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│   LM Head (线性投影)         │
│   [batch, seq_len, vocab_size]
│   → logits
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│   CrossEntropyLoss          │  ← 训练时计算损失
│   loss (标量)
└─────────────────────────────┘
```

### 1.3 核心组件及其作用

| 组件 | 作用 | 所在章节 |
|------|------|----------|
| **Token Embedding** | 将 token id 转换为词向量 | 第 2 章 |
| **RMSNorm** | 层归一化，稳定训练 | 第 2 章 |
| **RoPE** | 旋转位置编码，注入位置信息 | 第 2 章 |
| **Attention** | 多头注意力机制（GQA），捕捉 token 间依赖 | 第 3 章 |
| **FeedForward (SwiGLU)** | 前馈网络，非线性变换 | 第 4 章 |
| **MoE** | 混合专家模型，条件计算 | 第 4 章 |
| **Transformer Block** | Attention + MLP 的基本单元，多层堆叠 | 第 5 章 |
| **LM Head** | 将隐藏状态投影到词表空间 | 第 5 章 |
| **Pretrain / SFT** | 预训练和监督微调 | 第 6 章 |
| **DPO** | 直接偏好优化 | 第 7 章 |
| **PPO / GRPO** | 强化学习算法 | 第 8 章 |
| **LoRA** | 低秩适配，高效微调 | 第 9 章 |
| **生成算法** | 温度采样、Top-K、Top-P、KV Cache | 第 10 章 |

### 1.4 架构 ASCII 图

```
                          ┌──────────────────────┐
                          │    input_ids         │
                          │  [batch, seq_len]    │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │   embed_tokens       │
                          │   nn.Embedding       │
                          │  [batch, seq_len, hidden_size]
                          └──────────┬───────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
            │  ┌─────────────────────▼─────────────────────┐  │
            │  │           MiniMindBlock (× N)             │  │
            │  │                                           │  │
            │  │  ┌─────────┐   ┌──────────────────────┐  │  │
            │  │  │ RMSNorm │ → │ Attention (GQA+RoPE) │  │  │
            │  │  └─────────┘   └──────────┬───────────┘  │  │
            │  │       │                   │              │  │
            │  │       └─────── + ────────┘              │  │
            │  │                │                         │  │
            │  │  ┌─────────────▼─────────────┐          │  │
            │  │  │         RMSNorm           │          │  │
            │  │  └─────────────┬─────────────┘          │  │
            │  │                │                        │  │
            │  │  ┌─────────────▼─────────────┐          │  │
            │  │  │  FeedForward (SwiGLU/MoE) │          │  │
            │  │  └─────────────┬─────────────┘          │  │
            │  │                │                        │  │
            │  │  ┌─────────────▼─────────────┐          │  │
            │  │  │       残差连接 (+)         │          │  │
            │  │  └─────────────┬─────────────┘          │  │
            │  └────────────────│────────────────────────┘  │
            └───────────────────│───────────────────────────┘
                                │
                     ┌──────────▼───────────┐
                     │      final RMSNorm    │
                     │  [batch, seq_len, hidden_size]
                     └──────────┬───────────┘
                                │
                     ┌──────────▼───────────┐
                     │       lm_head        │
                     │    nn.Linear         │
                     │  [batch, seq_len, vocab_size]
                     │      → logits        │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼───────────┐
                     │  CrossEntropyLoss    │  ← 训练时
                     │      loss            │
                     └──────────────────────┘
```

### 1.5 默认配置参数

MiniMind 默认配置（小型模型，适合学习和实验）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_size` | 768 | 隐藏层维度（词向量维度） |
| `num_hidden_layers` | 8 | Transformer Block 层数 |
| `num_attention_heads` | 8 | 注意力头数（Query 头数） |
| `num_key_value_heads` | 4 | KV 头数（GQA，为 Query 头数的 1/2） |
| `vocab_size` | 6400 | 词表大小 |
| `intermediate_size` | 2048 | 前馈网络中间层维度 |
| `max_position_embeddings` | 512 | 最大序列长度 |
| `rms_norm_eps` | 1e-6 | RMSNorm 的 epsilon |
| `rope_theta` | 10000.0 | RoPE 的基础频率 |

> 这些参数定义在 `MiniMindConfig` 类中，你可以根据需要调整。

### 1.6 下一步

现在你对 MiniMind 的整体架构有了初步认识，接下来让我们深入学习第一个基础组件：

➡️ [**第 2 章：基础组件 - Embedding & RMSNorm & RoPE**](chapters/02-basic-components.md)
