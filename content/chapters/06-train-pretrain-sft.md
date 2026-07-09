# 第 6 章 训练算法 - Pretrain & SFT

本章介绍 MiniMind 的两阶段训练流程：预训练（Pretrain）和有监督微调（Supervised Fine-Tuning, SFT）。预训练让模型从海量纯文本中学习语言的通用规律（next-token prediction），SFT 则在预训练权重基础上学习聊天格式与指令跟随能力。两者共享同一套训练循环骨架，区别在于数据集的构造方式与标签掩码策略。

---

## 6.1 PretrainDataset：纯文本拼接与 next-token 标签

### 原理说明

预训练的目标是**语言建模**：给定前文的 token 序列，预测下一个 token。模型看到的是一整段文本，但训练时通过将输入序列右移一位构造出“输入-目标”对——位置 t 的输入预测位置 t+1 的 token。

MiniMind 的预训练数据是单字段纯文本（`{"text": "..."}`），每条样本独立成句。为了避免过短的样本浪费算力，每条文本被截断到 `max_length - 2`（预留 BOS/EOS 两个特殊 token），再 padding 到固定长度便于批量堆叠。

### 代码位置引用

`PretrainDataset` 定义在数据集模块中：

- 代码链接：[dataset/lm_dataset.py:38-60](file:///home/zhangxm/model_minimind/dataset/lm_dataset.py#L38-L60)

```python
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False,
                                max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels
```

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| input_ids | [batch_size, max_length] | long | 完整 token 序列（含 BOS/EOS/PAD） |
| labels | [batch_size, max_length] | long | 与 input_ids 相同，但 PAD 位置置为 -100 |

### 关键计算逻辑解释

1. **分词**：`add_special_tokens=False` 表示不自动加 BOS/EOS，由代码手动添加，便于精确控制序列结构。
2. **首尾加特殊 token**：`[BOS] + tokens + [EOS]`，让模型学会“一段文本的开始与结束”。
3. **Padding 对齐**：用 `pad_token_id` 填充到 `max_length`，使整个 batch 能堆叠成 `[batch, seq]` 的矩形张量。
4. **标签掩码**：`labels[input_ids == pad_token_id] = -100`，PyTorch 的 `cross_entropy` 会忽略标签为 `-100` 的位置，因此 PAD 不参与 loss。

**关于 X/Y 的位移**：数据集返回的 `input_ids` 和 `labels` 在位置上完全对齐，序列内的“右移一位”预测是在模型前向内部完成的——`MiniMindForCausalLM` 的 loss 计算会自动取 `logits[:, :-1]` 与 `labels[:, 1:]` 配对。因此从概念上等价于：

```
X = input_ids[:, :-1]   # [batch, seq-1]  模型输入
Y = input_ids[:, 1:]    # [batch, seq-1]  预测目标（next token）
```

预训练阶段对整段文本（包括 BOS、正文、EOS）都计算 loss，模型因此学到“如何续写任意文本”。

---

## 6.2 SFTDataset：chat 模板与 assistant 标签掩码

### 原理说明

SFT 的目标是让模型学会**多轮对话格式**与**指令跟随**。数据是结构化的 `conversations` 列表（system/user/assistant 多轮），需要先用 tokenizer 的 `apply_chat_template` 拼成模型实际看到的文本。

关键区别在于**只对 assistant 的回复计算 loss**：user/system 部分作为条件输入提供上下文，但不参与梯度更新——否则模型会学着“生成用户的提问”，这显然不是我们想要的。

### 代码位置引用

`SFTDataset` 及其标签生成函数：

- 类定义：[dataset/lm_dataset.py:62-131](file:///home/zhangxm/model_minimind/dataset/lm_dataset.py#L62-L131)
- `create_chat_prompt`：[dataset/lm_dataset.py:77-95](file:///home/zhangxm/model_minimind/dataset/lm_dataset.py#L77-L95)
- `generate_labels`：[dataset/lm_dataset.py:96-115](file:///home/zhangxm/model_minimind/dataset/lm_dataset.py#L96-L115)

```python
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        ...
        # assistant 段落的起止标记
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def create_chat_prompt(self, conversations):
        ...
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools)

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels
```

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| input_ids | [batch_size, max_length] | long | 完整对话文本 token（含 system/user/assistant 多轮） |
| labels | [batch_size, max_length] | long | 仅 assistant 回复区间保留 token id，其余为 -100 |

### 关键计算逻辑解释

**步骤 1：构造对话文本（create_chat_prompt）**

`apply_chat_template` 把多轮 messages 拼成模型实际看到的字符串，形如：

```
<|im_start|>system
你是minimind...<|im_end|>
<|im_start|>user
今天天气怎么样？<|im_end|>
<|im_start|>assistant
今天晴朗。<|im_end|>
```

工具调用数据（`tools`/`tool_calls`）会被反序列化为结构化对象再交给模板，保证 function calling 格式正确。

**步骤 2：定位 assistant 区间（generate_labels）**

- 初始 `labels` 全部为 `-100`（不学习）。
- 用 `bos_id`（即 `<|im_start|>assistant\n` 的 token 序列）做滑动匹配，找到每个 assistant 段落的起点。
- 从该起点向后扫描到 `eos_id`（`<|im_end|>\n`），把这一段（含 EOS）的 `labels` 还原为 `input_ids`，其余位置保持 `-100`。

效果上：模型只对“assistant 说出的每个 token”计算 loss，user/system 部分仅作为上文条件输入。这就是 SFT 区别于预训练的核心——**让模型学说话，而不是学提问**。

---

## 6.3 get_lr 余弦学习率调度

### 原理说明

学习率调度（Learning Rate Scheduling）控制训练过程中学习率的变化曲线。余弦退火（Cosine Annealing）让学习率按余弦曲线从初始值平滑下降，训练后期步长更小、更新更稳，有助于模型在最优解附近收敛。

### 代码位置引用

- 代码链接：[trainer/trainer_utils.py:42-44](file:///home/zhangxm/model_minimind/trainer/trainer_utils.py#L42-L44)

```python
def get_lr(current_step, total_steps, lr):
    # 余弦退火学习率：从初始 lr 平滑降到约 0.1*lr，训练后期更新更稳。
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

### 关键计算逻辑解释

公式为：

```
lr(step) = lr_base * (0.1 + 0.45 * (1 + cos(π * step / total_steps)))
```

分析两个端点：

| step | cos(π·step/total) | 括号内 | 实际 lr |
|------|-------------------|--------|---------|
| 0（开头） | cos(0) = 1 | 0.1 + 0.45 × 2 = 1.0 | `lr_base × 1.0` |
| total/2（中期） | cos(π/2) = 0 | 0.1 + 0.45 × 1 = 0.55 | `lr_base × 0.55` |
| total（结尾） | cos(π) = -1 | 0.1 + 0.45 × 0 = 0.1 | `lr_base × 0.1` |

因此学习率从 `lr_base` 沿余弦曲线衰减到 `0.1 × lr_base`，最低不低于初始值的 10%，避免后期学习率过小导致停滞。

**与经典余弦调度的差异**：标准余弦退火通常衰减到 0，并常配合 warmup（线性预热）。MiniMind 这里做了简化——不衰减到 0（保留 10% 下限），也不设显式 warmup，直接从初始 lr 起步按余弦下降。这种简化对小模型训练足够稳定，且实现极简。

### 调用方式

每个 step 动态计算并写入优化器：

```python
lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

`epoch * iters + step` 是全局训练步，保证跨 epoch 学习率连续下降而非每个 epoch 重置。代码位置：[trainer/train_pretrain.py:30-32](file:///home/zhangxm/model_minimind/trainer/train_pretrain.py#L30-L32)。

---

## 6.4 梯度累积（Gradient Accumulation）

### 原理说明

大模型训练需要较大的有效 batch size 以稳定梯度估计，但显存容量限制了单步能放下的样本数。**梯度累积**把一个大 batch 拆成若干小步：每步前向 + 反向计算梯度并累加，但不更新参数；累积到指定步数后再统一 `optimizer.step()` 更新一次。

数学上等价于：`grad_total = (1/N) * Σ grad_i`，与一次性用 N 倍 batch 算出的梯度均值一致。

### 代码位置引用

梯度累积逻辑位于训练循环内：

- 代码链接：[trainer/train_pretrain.py:40-50](file:///home/zhangxm/model_minimind/trainer/train_pretrain.py#L40-L50)

```python
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    # 梯度累积：把一个大 batch 拆成多个小 step，loss 先除以累积步数保持梯度尺度一致。
    loss = loss / args.accumulation_steps

scaler.scale(loss).backward()

if step % args.accumulation_steps == 0:
    # unscale 后才能做梯度裁剪；否则裁剪到的是放大后的梯度。
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

### 关键计算逻辑解释

1. **loss 缩放**：`loss = loss / accumulation_steps`。因为 `backward()` 是把梯度**累加**到 `.grad`，连续累积 N 次会变成 N 倍梯度。提前除以 N 让累加结果等于平均梯度，与大 batch 等价。
2. **每 N 步才 step**：`if step % args.accumulation_steps == 0` 触发一次完整的参数更新流程（unscale → clip → step → update → zero_grad）。
3. **末尾补齐**：epoch 结束时若剩余步数不足 `accumulation_steps`，`train_epoch` 末尾的 `if last_step % args.accumulation_steps != 0` 分支会补做一次更新，避免最后几个 batch 的梯度被丢弃。代码位置：[trainer/train_pretrain.py:67-72](file:///home/zhangxm/model_minimind/trainer/train_pretrain.py#L67-L72)。

MiniMind 预训练默认 `accumulation_steps=8`、`batch_size=32`，等价于有效 batch size = 256；SFT 默认 `accumulation_steps=1`（SFT 数据较少且学习率小，无需大 batch）。

---

## 6.5 混合精度训练（autocast + GradScaler）

### 原理说明

混合精度训练（Automatic Mixed Precision, AMP）同时使用 float32 与低精度浮点（float16 或 bfloat16）：

- **前向计算用低精度**：矩阵乘法、激活函数等用 bf16/fp16 存储，显存减半、Tensor Core 加速。
- **主权重保持 float32**：优化器维护的参数副本仍是 fp32，保证更新精度。
- **GradScaler（仅 fp16 需要）**：fp16 的表示范围小，小梯度会下溢为 0。Scaler 在反向时把 loss 放大一个因子，使梯度不致下溢；更新前再 unscale 还原。bfloat16 与 float32 指数位相同、动态范围一致，**不需要** scaling。

### 代码位置引用

- 上下文与 scaler 定义：[trainer/train_pretrain.py:130-133](file:///home/zhangxm/model_minimind/trainer/train_pretrain.py#L130-L133)

```python
device_type = "cuda" if "cuda" in args.device else "cpu"
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
```

### 关键计算逻辑解释

| dtype | autocast | GradScaler | 说明 |
|-------|----------|------------|------|
| bfloat16（默认） | 启用，前向用 bf16 | `enabled=False`（空操作） | bf16 动态范围足够，无需 scale |
| float16 | 启用，前向用 fp16 | `enabled=True` | 反向时放大 loss 防梯度下溢 |

**使用流程**（结合 6.4 的累积逻辑）：

```python
with autocast_ctx:                  # 前向 + loss 计算在低精度下
    res = model(input_ids, labels=labels)
    loss = res.loss / args.accumulation_steps
scaler.scale(loss).backward()       # 放大 loss 后反向（fp16 才生效）
scaler.unscale_(optimizer)          # 还原梯度到真实尺度，才能正确裁剪
torch.nn.utils.clip_grad_norm_(...) # 梯度裁剪
scaler.step(optimizer)              # 若无 inf/nan，执行 optimizer.step()
scaler.update()                     # 动态调整下一轮的 scale 因子
```

**为什么 unscale 必须在 clip 之前**：`scaler.scale(loss).backward()` 把梯度放大了 `scale` 倍。如果直接对放大后的梯度做 `clip_grad_norm_`，裁剪阈值相当于被放大了，失去意义。`unscale_` 先还原真实梯度，再裁剪才正确。代码注释也明确点出这一点。

---

## 6.6 权重加载、检查点与断点续训

### 原理说明

训练过程可能因中断需要恢复。完整的断点续训不仅需要恢复模型权重，还要恢复优化器状态（AdamW 的一阶/二阶动量）、GradScaler 状态、当前 epoch/step，以及已训练 batch 的位置——否则会重复消费样本，破坏学习率曲线的连续性。

MiniMind 把这些功能拆成三个组件：`init_model`（加载预训练权重起训）、`lm_checkpoint`（保存/恢复完整训练状态）、`SkipBatchSampler`（跳过已训练 batch）。

### 6.6.1 init_model：加载预训练权重

- 代码链接：[trainer/trainer_utils.py:126-138](file:///home/zhangxm/model_minimind/trainer/trainer_utils.py#L126-L138)

```python
def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)
    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
    ...
    return model.to(device), tokenizer
```

**关键点**：

- `from_weight='none'`：随机初始化，用于从头预训练。
- `from_weight='pretrain'`：加载预训练权重做 SFT（`train_full_sft.py` 默认值）。
- `from_weight='full_sft'`：加载 SFT 权重做 DPO。
- `strict=False`：允许权重与模型结构有少量不匹配（如 MoE 路由器新增参数），便于灵活迁移。

### 6.6.2 lm_checkpoint：保存与恢复完整状态

- 代码链接：[trainer/trainer_utils.py:68-124](file:///home/zhangxm/model_minimind/trainer/trainer_utils.py#L68-L124)

```python
def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, ...):
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:   # 保存模式
        ...
        torch.save(state_dict, ckp_tmp); os.replace(ckp_tmp, ckp_path)       # 推理权重
        resume_data = {'model': ..., 'optimizer': ..., 'epoch': ..., 'step': ..., 'world_size': ...}
        torch.save(resume_data, resume_tmp); os.replace(resume_tmp, resume_path)  # 续训状态
    else:                   # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            # GPU数量变化时按 world_size 比例换算已跳过的 step
            ...
            return ckp_data
        return None
```

**双文件设计**：

| 文件 | 内容 | 用途 |
|------|------|------|
| `{weight}_{hidden}.pth` | 仅模型权重（half） | 推理 / 作为下一阶段 `from_weight` |
| `{weight}_{hidden}_resume.pth` | 模型 + 优化器 + scaler + epoch + step + world_size | 断点续训恢复 |

**原子写入**：先写 `.tmp` 再 `os.replace`，避免训练中途崩溃留下半截损坏文件。模型权重统一转 `half()` 存盘以节省空间。

**world_size 自适应**：续训时若 GPU 数量变化（如从 4 卡恢复到 2 卡），每个 step 处理的样本数变了，`step` 会按 `saved_ws / current_ws` 比例换算，保证跳过的样本数与原训练一致。

### 6.6.3 SkipBatchSampler：跳过已训练 batch

- 代码链接：[trainer/trainer_utils.py:143-167](file:///home/zhangxm/model_minimind/trainer/trainer_utils.py#L143-L167)

```python
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        ...
```

**作用**：续训只在恢复的第一个 epoch 跳过已训练 batch（`skip = start_step`），后续 epoch 从头正常训练。配合 `setup_seed(42 + epoch)` 固定每个 epoch 的打乱顺序，保证恢复后看到的是与原训练**完全一致**的 batch 序列——这是断点续训正确性的关键：若 batch 顺序变了，跳过 N 个 batch 就不再对应原训练的前 N 步。

### 6.6.4 训练入口的串联

[trainer/train_pretrain.py:175-188](file:///home/zhangxm/model_minimind/trainer/train_pretrain.py#L175-L188) 把上述组件串起来：

```python
ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
...
if ckp_data:
    model.load_state_dict(ckp_data['model'])
    optimizer.load_state_dict(ckp_data['optimizer'])
    scaler.load_state_dict(ckp_data['scaler'])
    start_epoch = ckp_data['epoch']
    start_step = ckp_data.get('step', 0)
...
for epoch in range(start_epoch, args.epochs):
    setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
    skip = start_step if (epoch == start_epoch and start_step > 0) else 0
    batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
```

---

## 6.7 训练循环详解

### 原理说明

预训练与 SFT 共享同一套训练循环骨架，区别仅在数据集类型（`PretrainDataset` vs `SFTDataset`）和默认超参。每一步的流程：前向算 loss → 反向算梯度 →（累积满后）裁剪 + 更新 + 清零 → 动态调学习率 → 周期性保存。

### 代码位置引用

- Pretrain 训练循环：[trainer/train_pretrain.py:24-72](file:///home/zhangxm/model_minimind/trainer/train_pretrain.py#L24-L72)
- SFT 训练循环：[trainer/train_full_sft.py:24-72](file:///home/zhangxm/model_minimind/trainer/train_full_sft.py#L24-L72)

### 输入/输出张量说明（单步前向）

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| input_ids | [batch, seq_len] | long | 输入 token（pretrain 含 PAD；sft 含完整对话） |
| labels | [batch, seq_len] | long | 目标 token，不可学习位置为 -100 |
| logits | [batch, seq_len, vocab_size] | bf16/fp16 | 模型输出 logits |
| loss | 标量 | float32 | CE loss + aux_loss（MoE 路由均衡损失） |

### 关键计算逻辑逐行解释

```python
for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
    input_ids = input_ids.to(args.device)
    labels = labels.to(args.device)
    last_step = step
    # 1. 动态学习率
    lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 2. 前向 + loss（混合精度上下文）
    with autocast_ctx:
        res = model(input_ids, labels=labels)
        loss = res.loss + res.aux_loss          # CE loss + MoE 路由均衡 aux_loss
        loss = loss / args.accumulation_steps    # 梯度累积：缩放 loss

    # 3. 反向（GradScaler 放大，仅 fp16 生效）
    scaler.scale(loss).backward()

    # 4. 累积满 N 步才更新参数
    if step % args.accumulation_steps == 0:
        scaler.unscale_(optimizer)              # 还原梯度真实尺度
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪
        scaler.step(optimizer)                   # 参数更新
        scaler.update()                          # 调整 scale 因子
        optimizer.zero_grad(set_to_none=True)    # 清零梯度（set_to_none 省内存）

    # 5. 周期性保存
    if (step % args.save_interval == 0 or step == iters) and is_main_process():
        model.eval()
        ...
        torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)   # 推理权重
        lm_checkpoint(..., model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, ...)  # 续训状态
        model.train()
```

**几个细节**：

1. **`res.loss + res.aux_loss`**：`res.loss` 是语言模型 CE loss；`res.aux_loss` 是 MoE 架构的路由均衡辅助损失（非 MoE 模型为 0），用于鼓励专家负载均匀。两者相加作为总 loss。
2. **`clip_grad_norm_`**：对全部参数的梯度做 L2 范数裁剪，阈值 `grad_clip=1.0`，抑制偶发大梯度导致训练发散，对小模型微调尤其重要。
3. **`zero_grad(set_to_none=True)`**：把 `.grad` 直接置为 `None` 而非填 0，省去一次内存写入，且利于下次 `backward` 时自动按需分配。
4. **`is_main_process()`**：DDP 下只有 rank 0 保存权重和打印日志，避免多进程重复写文件冲突。
5. **DDP/compile 外壳剥离**：保存前通过 `model.module`（DDP）和 `_orig_mod`（torch.compile）层层取回原始模型，保证 state_dict 的 key 与裸模型一致。

### Pretrain 与 SFT 的超参对比

| 参数 | Pretrain 默认 | SFT 默认 | 说明 |
|------|---------------|----------|------|
| learning_rate | 5e-4 | 1e-5 | SFT 学习率小一个数量级，避免冲掉预训练通用能力 |
| batch_size | 32 | 16 | SFT 序列更长（768 vs 340），单条显存更大 |
| accumulation_steps | 8 | 1 | 预训练需大有效 batch 稳定梯度；SFT 数据少无需累积 |
| max_seq_len | 340 | 768 | SFT 多轮对话更长 |
| from_weight | none | pretrain | SFT 必须基于预训练权重 |

---

## 小结

本章拆解了 MiniMind 的 Pretrain + SFT 两阶段训练流程：

| 组件 | 作用 | 所在位置 |
|------|------|---------|
| PretrainDataset | 纯文本 + BOS/EOS，全段计算 loss | lm_dataset.py:38 |
| SFTDataset | chat 模板 + assistant 标签掩码 | lm_dataset.py:62 |
| get_lr | 余弦学习率衰减（lr → 0.1·lr） | trainer_utils.py:42 |
| 梯度累积 | 小 batch 模拟大 batch | train_pretrain.py:40 |
| autocast + GradScaler | 混合精度训练加速 | train_pretrain.py:130 |
| init_model | 加载预训练权重起训 | trainer_utils.py:126 |
| lm_checkpoint | 保存推理权重 + 续训状态 | trainer_utils.py:68 |
| SkipBatchSampler | 续训跳过已训练 batch | trainer_utils.py:143 |

核心要点：

1. **预训练学语言，SFT 学对话**——前者对全文本算 loss，后者只对 assistant 回复算 loss（`labels=-100` 掩码）。
2. **梯度累积 + 混合精度**是有限显存下训练大模型的两项基础设施，让有效 batch size 与计算速度都可调。
3. **断点续训的正确性**依赖三件事协同：恢复优化器状态、固定 epoch 打乱种子、`SkipBatchSampler` 跳过对应 batch，缺一不可。
4. Pretrain 与 SFT 共享训练循环骨架，仅数据集与超参不同，体现了框架的统一性。

下一章将介绍在 SFT 基础上进一步对齐人类偏好的 DPO 算法。
