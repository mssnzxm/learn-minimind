# 第 9 章 LoRA 低秩适配

本章介绍 MiniMind 中的 LoRA（Low-Rank Adaptation）实现。LoRA 是目前大模型参数高效微调（PEFT）的事实标准，它通过在冻结的原始权重旁挂一个低秩可训练增量，把可训练参数量从亿级压到百万级，同时保持微调效果接近全量。MiniMind 的 LoRA 实现非常精简（不到 60 行），是理解 LoRA 原理的最佳起点。

---

## 9.1 LoRA 原理：W' = W + BA

### 原理说明

LoRA 的核心假设：大模型在下游任务上的微调权重更新 ΔW 具有**低秩结构**——即可以用两个小矩阵的乘积来近似。

对于原始线性层 `W: [out, in]`，LoRA 把权重更新参数化为：

```
W' = W + BA
```

其中：
- `W`：原始权重，**冻结不训练**，shape `[out_features, in_features]`
- `A`：降秩矩阵，shape `[rank, in_features]`，把输入从 `in_features` 维压到 `rank` 维
- `B`：升秩矩阵，shape `[out_features, rank]`，把 `rank` 维还原回 `out_features` 维
- `rank`：低秩维度，远小于 `hidden_size`

前向传播变为 `y = W'x = Wx + BAx`。由于 `rank << min(in, out)`，可训练参数从 `in × out` 降到 `rank × (in + out)`。

### 参数量对比（以 MiniMind 默认配置为例）

MiniMind 默认配置：`hidden_size=768`、`num_hidden_layers=8`、`num_attention_heads=8`、`num_key_value_heads=4`、`head_dim=96`、`intermediate_size=2432`、`vocab_size=6400`、`tie_word_embeddings=True`。

以一个 `768 × 768` 的注意力投影层为例（q_proj 或 o_proj）：

| 方式 | 可训练参数量 | 计算式 |
|------|------------|--------|
| 全量微调 | 589,824 | 768 × 768 |
| LoRA (rank=16) | 24,576 | 16 × (768 + 768) |
| 压缩比 | 24× | 589,824 / 24,576 |

**全模型可训练参数对比**（rank=16）：

| 方式 | 可训练参数量 | 说明 |
|------|------------|------|
| 全量微调 | ~63.9M | 含 embedding（tied）、8 层 Transformer、final norm |
| LoRA (rank=16) | ~0.39M | 仅 q_proj + o_proj 的 A/B 矩阵（每层 2 个，共 16 个模块） |
| 比例 | **~0.6%** | 393,216 / 63,910,656 |

> 注：MiniMind 的 `apply_lora` 只给**方阵 Linear**（`in_features == out_features`）挂 LoRA。在默认 GQA 配置下（`num_key_value_heads=4`），`k_proj`/`v_proj` 是 `768→384` 非方阵会被跳过，只有 `q_proj`（768→768）和 `o_proj`（768→768）符合条件。若改成标准 MHA（`num_key_value_heads=num_attention_heads`），则四个注意力投影都是方阵都会被挂上 LoRA。

---

## 9.2 LoRA 模块定义

### 代码位置引用

LoRA 模块定义在 `LoRA` 类中：

- 代码链接：[model/model_lora.py:6](file:///home/zhangxm/model_minimind/model/model_lora.py#L6-L17)

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA 的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)   # 低秩矩阵 A: [rank, in]
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵 B: [out, rank]
        # 矩阵 A 高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵 B 全 0 初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))
```

### 初始化策略（关键）

LoRA 的初始化有一个**精心设计的非对称结构**：

| 矩阵 | 初始化方式 | 初始值 |
|------|-----------|--------|
| A | 高斯随机 `N(0, 0.02)` | 非零 |
| B | 全零 `zero_()` | 0 |

这保证了训练初始时 `BA = 0`，即 `W' = W + 0 = W`，**LoRA 增量在训练开始时对原模型毫无影响**。模型从 SFT 权重出发平滑地学习 ΔW，避免了从头训练的不稳定性。这是 LoRA 能稳定微调大模型的关键设计之一。

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| x | [batch, seq, in_features] | float32/bf16 | 输入特征 |
| A(x) | [batch, seq, rank] | float32/bf16 | 降维后的中间表示 |
| B(A(x)) | [batch, seq, out_features] | float32/bf16 | 升维还原后的增量 |

### 关键计算逻辑

前向 `y = B(A(x))` 等价于 `y = (B.weight @ A.weight) @ x`，其中：
- `A.weight` shape: `[rank, in_features]`
- `B.weight` shape: `[out_features, rank]`
- `B.weight @ A.weight` shape: `[out_features, in_features]`（与 W 同 shape，可相加）

---

## 9.3 apply_lora：Monkey-Patch 挂载

### 原理说明

`apply_lora` 不修改模型源代码，而是用 **monkey-patch** 的方式：遍历模型所有子模块，找到符合条件的 `nn.Linear`，给它挂一个 `lora` 子模块，并把它的 `forward` 替换为 `Wx + BAx`。这种实现与具体模型架构解耦，对 MiniMind、LLaMA、Qwen 等任何用 `nn.Linear` 的模型都通用。

- 代码链接：[model/model_lora.py:21](file:///home/zhangxm/model_minimind/model/model_lora.py#L21-L34)

```python
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            # 这里只给方阵 Linear 挂 LoRA，覆盖注意力投影等主要可调层，参数量很小。
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                # 推理/训练时原层输出保持不变，再叠加一个可训练的低秩增量。
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora
```

### 关键细节逐行解释

1. **筛选条件** `module.in_features == module.out_features`：只对方阵线性层挂载。这覆盖了注意力的 `q_proj`/`o_proj`（在标准 MHA 下还有 `k_proj`/`v_proj`），跳过了非方阵的 MLP 投影（`gate_proj` 768→2432 等）。

2. **挂载方式** `setattr(module, "lora", lora)`：把 LoRA 模块作为子模块挂上去，PyTorch 会自动注册其参数到 `model.parameters()` 中，优化器会自动更新它们。

3. **闭包绑定** `def forward_with_lora(x, layer1=original_forward, layer2=lora)`：用默认参数显式捕获当前循环的 `original_forward` 和 `lora`，避免 Python 闭包延迟绑定导致的「所有层都引用最后一个 lora」的经典 bug。

4. **替换 forward** `module.forward = forward_with_lora`：直接替换实例方法，原 `Wx` 通过 `layer1(x)` 保留，叠加 `layer2(x) = BAx`。

5. **冻结原权重**：`apply_lora` 本身不冻结原权重，需要调用方在创建优化器前手动设置 `requires_grad_(False)`，只把 LoRA 参数加入优化器（典型用法见 MiniMind 的 LoRA 训练脚本）。

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| x | [batch, seq, hidden] | float32/bf16 | 输入特征 |
| original_forward(x) | [batch, seq, hidden] | float32/bf16 | 原始 Wx |
| lora(x) | [batch, seq, hidden] | float32/bf16 | 低秩增量 BAx |
| output | [batch, seq, hidden] | float32/bf16 | Wx + BAx |

---

## 9.4 save_lora / load_lora：只保存 LoRA 参数

### 原理说明

LoRA 微调的最大优势之一：**检查点极小**。全量微调要保存整个模型权重（MiniMind ~64MB，大模型可达几十 GB），而 LoRA 只需保存 A/B 矩阵（MiniMind rank=16 时仅 ~0.8MB）。这让 LoRA 权重可以方便地分发、版本管理、按任务热切换。

### save_lora

- 代码链接：[model/model_lora.py:48](file:///home/zhangxm/model_minimind/model/model_lora.py#L48-L57)

```python
def save_lora(model, path):
    # LoRA 微调只保存新增的低秩矩阵，不保存原始大模型权重。
    raw_model = getattr(model, '_orig_mod', model)   # 兼容 torch.compile
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name  # 兼容 DDP 前缀
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half()
                          for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
```

关键点：
- `getattr(model, '_orig_mod', model)`：兼容 `torch.compile` 包装（原始模型存在 `_orig_mod` 属性下）。
- `name[7:] if name.startswith("module.")`：兼容 DDP（DistributedDataParallel）保存出来的 `module.` 前缀。
- `v.cpu().half()`：转 CPU + half 精度，进一步压缩体积。
- 只收集 `hasattr(module, 'lora')` 的模块，**不保存任何原始权重**。

### load_lora

- 代码链接：[model/model_lora.py:37](file:///home/zhangxm/model_minimind/model/model_lora.py#L37-L45)

```python
def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    # 兼容 DDP 保存出来的 module.xxx 前缀。
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v
                          for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)
```

加载逻辑：先去掉 `module.` 前缀做兼容，再遍历模型中所有挂了 `lora` 的模块，从 state_dict 中筛选出对应的 `*.lora.A.weight` / `*.lora.B.weight` 加载回去。

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| lora.A.weight | [rank, in_features] | float16 | 降秩矩阵（保存/加载） |
| lora.B.weight | [out_features, rank] | float16 | 升秩矩阵（保存/加载） |

---

## 9.5 merge_lora：合并回原始权重

### 原理说明

LoRA 训练完成后，可以把 `BA` 直接加到原始 `W` 上，得到一个**单一体**的权重 `W' = W + BA`。合并后模型结构与原始完全一致，**推理时无任何额外开销**——这是 LoRA 相比 Adapter Tuning（需要在推理时多跑一层）的重要优势。

合并公式：

```
W'.weight = W.weight + B.weight @ A.weight
```

- `A.weight` shape: `[rank, in_features]`
- `B.weight` shape: `[out_features, rank]`
- `B.weight @ A.weight` shape: `[out_features, in_features]`（与 `W.weight` 同 shape）

### 代码位置引用

- 代码链接：[model/model_lora.py:60](file:///home/zhangxm/model_minimind/model/model_lora.py#L60-L71)

```python
def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    # 1. 先收集所有非 lora 的原始权重
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    # 2. 再把 BA 合并回对应的 Linear 权重
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            # 合并公式：W' = W + B @ A，合并后推理不再需要额外 LoRA 分支。
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data
                                                 @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
```

### 关键计算逻辑逐行解释

1. **先加载 LoRA 权重** `load_lora(model, lora_path)`：把保存的 A/B 矩阵恢复到模型的 `lora` 子模块中。

2. **收集原始权重**：先复制一份所有非 `lora` 的权重作为基底。`module.weight.data.clone()` 是必要的——直接引用会被后续 `+=` 修改原模型权重。

3. **合并**：遍历所有 `nn.Linear`，对挂了 `lora` 的模块执行 `W += B @ A`。注意 `B.weight @ A.weight` 的矩阵乘法维度匹配：`[out, rank] @ [rank, in] = [out, in]`。

4. **保存合并后的完整权重** `torch.save(state_dict, save_path)`：输出的是一个标准模型权重文件，可以直接用 `init_model` 加载，**不再需要 apply_lora**。

### 输入/输出张量说明

| 张量 | Shape | dtype | 含义 |
|------|-------|-------|------|
| module.weight | [out_features, in_features] | float16 | 原始权重 W |
| B.weight @ A.weight | [out_features, in_features] | float16 | LoRA 增量 BA |
| state_dict[name].weight | [out_features, in_features] | float16 | 合并后 W' = W + BA |

---

## 9.6 LoRA 完整使用流程

把上述四个 API 串起来，LoRA 微调的典型生命周期如下：

```
# 1. 加载预训练模型
model, tokenizer = init_model(lm_config, base_weight)

# 2. 挂载 LoRA（冻结原权重）
apply_lora(model, rank=16)
for name, param in model.named_parameters():
    param.requires_grad = 'lora' in name   # 只训练 LoRA 参数

# 3. 训练（与全量微调代码完全一致，只是优化器只更新 LoRA 参数）
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=...)
# ... 标准 train loop ...

# 4. 保存（只存 LoRA，体积很小）
save_lora(model, "lora_weights.pth")

# === 推理阶段有两种选择 ===

# 选项 A：保持 LoRA 形式推理（可热切换多个 LoRA）
apply_lora(model, rank=16)
load_lora(model, "lora_weights.pth")
# 推理时 Wx + BAx，有微小额外开销

# 选项 B：合并后推理（无额外开销，适合生产部署）
merge_lora(model, "lora_weights.pth", "merged_weights.pth")
# 之后直接用 merged_weights.pth 加载原始模型结构即可
```

---

## 小结

本章拆解了 MiniMind 的 LoRA 实现：

1. **LoRA 原理**：`W' = W + BA`，用两个小矩阵 `A [rank, in]`、`B [out, rank]` 近似权重更新 ΔW，把可训练参数从 `in × out` 降到 `rank × (in + out)`。MiniMind 默认配置下仅约 0.6% 参数可训练。

2. **非对称初始化**：A 高斯随机、B 全零，保证训练初始 `BA = 0` 不破坏原模型，是 LoRA 稳定微调的关键。

3. **apply_lora** 用 monkey-patch 给符合条件的方阵 `nn.Linear` 挂载 LoRA 子模块并替换 forward 为 `Wx + BAx`，与具体模型架构解耦，闭包默认参数避免了延迟绑定 bug。

4. **save_lora / load_lora** 只保存/加载 A/B 矩阵，检查点极小，方便分发和热切换。

5. **merge_lora** 训练后把 `BA` 合并回 `W`，得到单一权重 `W' = W + BA`，推理时无任何额外开销，适合生产部署。

LoRA 的精妙之处在于：它不是对模型结构的改造，而是对「权重更新」本身做的低秩假设。这个假设在大多数下游任务上都成立，使得 LoRA 成为参数高效微调的事实标准。理解了 MiniMind 这 60 行实现，就掌握了 LoRA 的全部核心思想。
