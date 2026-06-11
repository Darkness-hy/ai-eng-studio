# 使用 LoRA 与 QLoRA 进行微调

> 全量微调一个 7B 模型需要 56GB 显存。你没有这么多，绝大多数公司也没有。LoRA 让你只训练不到 1% 的参数，就能在 6GB 显存里微调同一个模型。这不是妥协方案——在大多数任务上，它的效果与全量微调相当。整个开源微调生态系统都建立在这一个技巧之上。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 06 (Instruction Tuning / SFT)
**Time:** ~75 minutes
**Related:** Phase 10 covers the SFT/DPO loops from scratch. This lesson plugs those into the 2026 PEFT toolkits (PEFT, TRL, Unsloth, Axolotl, LLaMA-Factory).

## 学习目标

- 通过向预训练模型的注意力层注入低秩适配器矩阵（A 和 B）来实现 LoRA
- 计算 LoRA 相对于全量微调的参数节省量：秩为 r、维度为 d_model 时，只需训练 2*r*d 个参数，而不是 d^2 个
- 使用 QLoRA（4-bit 量化基座模型 + LoRA 适配器）微调模型，使其能装进消费级 GPU 的显存
- 将 LoRA 权重合并回基座模型以便部署，并比较带适配器与不带适配器的推理速度

## 问题背景

你有一个基座模型，比如 Llama 3 8B。你希望它用你公司的语气回复客服工单。SFT 是答案。但 SFT 有一个成本问题。

全量微调会更新模型中的每一个参数。Llama 3 8B 有 80 亿个参数。在 fp16 精度下，每个参数占 2 字节，光加载权重就要 16GB。训练时你还需要梯度（16GB）、Adam 优化器状态（动量 + 方差共 32GB），以及激活值。合计：训练一个 8B 模型大约需要 56GB 显存。

一块 A100 80GB 勉强装得下。云服务商上两块 A100 的价格是每小时 3-4 美元。在 50,000 条样本上训练 3 个 epoch 需要 6-10 小时。也就是每次实验花 30-40 美元。为了调好超参数跑 10 次实验，还没部署任何东西就已经花掉了 400 美元。

把规模扩大到 Llama 3 70B，数字就变得离谱了。光权重就要 140GB。你需要一个集群。每次实验 100 美元以上。

还有一个更深层的问题。全量微调会修改模型中的所有权重。如果你在客服数据上微调，模型的通用能力可能会退化。这叫灾难性遗忘（catastrophic forgetting）。模型在你的任务上变好了，在其他所有事情上变差了。

你需要一种方法：训练更少的参数、占用更少的内存，并且不会破坏模型已有的知识。

## 核心概念

### LoRA：低秩适配

Microsoft 的 Edward Hu 及其同事在 2021 年 6 月发表了 LoRA。这篇论文的洞见是：微调过程中的权重更新具有较低的内在秩。在一个 4096x4096 的权重矩阵中，你不需要更新全部 1670 万个参数。更新中的有效信息可以用一个秩为 16 或 32 的矩阵来捕获。

数学表达如下。标准的线性层计算的是：

```
y = Wx
```

其中 W 是一个 d_out x d_in 的矩阵。对于一个 4096x4096 的注意力投影层，这是 16,777,216 个参数。

LoRA 冻结 W，并添加一个低秩分解：

```
y = Wx + BAx
```

其中 B 是 (d_out x r)，A 是 (r x d_in)。秩 r 远小于 d——通常取 8、16 或 32。

对于一个 4096x4096 的层，取 r=16：
- 原始参数量：4096 x 4096 = 16,777,216
- LoRA 参数量：(4096 x 16) + (16 x 4096) = 65,536 + 65,536 = 131,072
- 缩减比例：131,072 / 16,777,216 = 0.78%

你只训练 0.78% 的参数，却能获得 95-100% 的质量。

```mermaid
graph LR
    X["Input x"] --> W["Frozen W (d x d)"]
    X --> A["A (r x d)"]
    A --> B["B (d x r)"]
    W --> Plus["+ (merge)"]
    B --> Plus
    Plus --> Y["Output y"]

    style W fill:#1a1a2e,stroke:#e94560,color:#fff
    style A fill:#0f3460,stroke:#16213e,color:#fff
    style B fill:#0f3460,stroke:#16213e,color:#fff
```

A 用随机高斯分布初始化，B 初始化为零。这意味着 LoRA 的贡献从零开始——模型从其原始行为出发开始训练，逐步学习适配。

### 缩放因子：Alpha

LoRA 引入了一个缩放因子 alpha，用于控制低秩更新对输出的影响程度：

```
y = Wx + (alpha / r) * BAx
```

当 alpha = r 时，缩放为 1 倍。当 alpha = 2r（常见默认值）时，缩放为 2 倍。这个超参数让你可以独立于基础学习率，单独控制 LoRA 路径的学习速度。

实用建议：
- alpha = 2 * rank 是社区的常见约定（原论文在大多数实验中使用 alpha = rank）
- alpha = rank 给出 1 倍缩放，保守但稳定
- alpha 越高，每步的更新幅度越大，可能加速收敛，也可能导致不稳定

### LoRA 应该加在哪里

Transformer 有很多线性层，你不需要给所有层都加 LoRA。原论文测试了不同的组合：

| 目标层 | 可训练参数量（7B） | 质量 |
|--------------|----------------------|---------|
| 仅 q_proj | 4.7M | 好 |
| q_proj + v_proj | 9.4M | 更好 |
| q_proj + k_proj + v_proj + o_proj | 18.9M | 注意力层的最佳选择 |
| 所有线性层（注意力 + MLP） | 37.7M | 收益微小，参数翻倍 |

对大多数任务来说，最佳选择是 q_proj + v_proj。这针对的是自注意力中的查询投影和值投影，它们分别控制模型关注什么以及提取什么信息。加上 MLP 层对代码生成等复杂任务有帮助，但参数量翻倍，对简单任务来说收益递减。

### 秩的选择

秩 r 控制适配的表达能力：

| 秩 | 可训练参数量（每层） | 适用场景 |
|------|---------------------------|----------|
| 4 | 32,768 | 简单分类、情感分析 |
| 8 | 65,536 | 单领域问答、摘要 |
| 16 | 131,072 | 多领域任务、指令遵循 |
| 32 | 262,144 | 复杂推理、代码生成 |
| 64 | 524,288 | 对大多数任务收益递减 |
| 128 | 1,048,576 | 很少有充分理由使用 |

Hu 等人的研究表明，对于简单任务，r=4 就已经能捕获大部分适配信息。实践中最常见的选择是 r=8 和 r=16。超过 r=64 很少能提升质量，反而开始失去 LoRA 的内存优势。

### QLoRA：4-Bit 量化 + LoRA

华盛顿大学的 Tim Dettmers 及其同事在 2023 年 5 月发表了 QLoRA。其思路是：将冻结的基座模型量化到 4-bit 精度，再在其上挂载 fp16 精度的 LoRA 适配器。

这极大地改变了内存的计算公式：

| 方法 | 权重内存（7B） | 训练内存（7B） | 所需 GPU |
|--------|-------------------|---------------------|-------------|
| 全量微调（fp16） | 14GB | ~56GB | 1x A100 80GB |
| LoRA（fp16 基座） | 14GB | ~18GB | 1x A100 40GB |
| QLoRA（4-bit 基座） | 3.5GB | ~6GB | 1x RTX 3090 24GB |

QLoRA 有三项技术贡献：

**NF4（Normal Float 4-bit）**：一种专为神经网络权重设计的新数据类型。神经网络权重大致服从正态分布。NF4 把它的 16 个量化级别放在标准正态分布的分位数上。对于正态分布的数据，这在信息论上是最优的。相比均匀 4-bit 量化（INT4）或标准 Float4，它丢失的信息更少。

**双重量化（Double quantization）**：量化常数本身也占内存。每 64 个权重组成的块需要一个 fp32 缩放因子（4 字节）。对一个 7B 模型来说，这是额外的 0.4GB。双重量化将这些常数再量化到 fp8，把开销降到 0.1GB。虽小，但积少成多。

**分页优化器（Paged optimizers）**：训练过程中，优化器状态（Adam 的动量和方差）在长序列上可能超出 GPU 内存。分页优化器利用 NVIDIA 的统一内存，在 GPU 内存耗尽时自动把优化器状态换页到 CPU 内存，需要时再换回来。这以牺牲一些吞吐量为代价，避免了 OOM 崩溃。

### 质量问题

减少参数量或量化基座模型会损害质量吗？来自多篇论文的结果：

| 方法 | MMLU（5-shot） | MT-Bench | HumanEval |
|--------|--------------|----------|-----------|
| 全量微调（Llama 2 7B） | 48.3 | 6.72 | 14.6 |
| LoRA r=16 | 47.9 | 6.68 | 14.0 |
| QLoRA r=16（NF4） | 47.5 | 6.61 | 13.4 |
| QLoRA r=64（NF4） | 48.1 | 6.70 | 14.2 |

在大多数基准上，r=16 的 LoRA 与全量微调的差距在 1% 以内。r=16 的 QLoRA 再损失零点几个百分点。r=64 的 QLoRA 基本追平全量微调，同时少用 90% 的内存。

### 真实世界的成本

在 50,000 条样本上微调 Llama 3 8B（3 个 epoch）：

| 方法 | GPU | 时间 | 成本 |
|--------|-----|------|------|
| 全量微调 | 2x A100 80GB | 8 小时 | ~$32 |
| LoRA r=16 | 1x A100 40GB | 4 小时 | ~$8 |
| QLoRA r=16 | 1x RTX 4090 24GB | 6 小时 | ~$5 |
| QLoRA r=16（Unsloth） | 1x RTX 4090 24GB | 2.5 小时 | ~$2 |
| QLoRA r=16 | 1x T4 16GB | 12 小时 | ~$4 |

在单块消费级 GPU 上跑 QLoRA，花费还不到一顿午饭钱。这就是开放权重微调社区在 2023 年爆发式增长的原因，也是为什么到 2026 年，下面列出的每个训练框架都默认内置 QLoRA。

### 2026 年的 PEFT 技术栈

| 框架 | 是什么 | 何时选用 |
|-----------|-----------|-----------|
| **Hugging Face PEFT** | 官方标准的 LoRA/QLoRA/DoRA/IA3 库 | 你需要底层控制，且训练循环已经基于 `transformers.Trainer` |
| **TRL** | HF 的反馈强化训练器集合（SFT、DPO、GRPO、PPO、ORPO） | SFT 之后还需要 DPO/GRPO；构建在 PEFT 之上 |
| **Unsloth** | 用 Triton 内核重写的前向/反向传播 | 你想要 2-5 倍加速 + 显存减半且无精度损失；适用于 Llama/Mistral/Qwen 系列 |
| **Axolotl** | 基于 YAML 配置的 PEFT + TRL + DeepSpeed + Unsloth 封装 | 你想要可复现、可版本控制的训练运行 |
| **LLaMA-Factory** | 基于 PEFT + TRL 的 GUI/CLI/API | 你想要零代码微调；支持 100 多个模型系列 |
| **torchtune** | 原生 PyTorch 训练方案，不依赖 `transformers` | 你想要最少的依赖，且组织已统一使用 PyTorch |

经验法则：科研用途或一次性实验 → PEFT。可重复的生产流水线 → 启用 Unsloth 内核的 Axolotl。一次性快速原型 → LLaMA-Factory。

### 合并适配器

训练完成后，你手上有两样东西：冻结的基座模型和一个小的 LoRA 适配器（通常 10-100MB）。你可以二选一：

1. **保持分离**：先加载基座模型，再在其上加载适配器。不同任务换用不同的适配器。这就是从一个基座模型服务多个微调变体的方式。

2. **永久合并**：计算 W' = W + (alpha/r) * BA，并将结果保存为一个新的完整模型。合并后的模型与原模型大小相同。没有推理开销，也没有适配器需要管理。

如果要服务多个任务（客服适配器、代码适配器、翻译适配器），就保持分离。如果只部署一个专用模型，就合并。

用于组合多个适配器的高级合并技术：

- **TIES-Merging**（Yadav et al. 2023）：先修剪掉幅值较小的参数，解决符号冲突，再合并。可减少适配器之间的相互干扰。
- **DARE**（Yu et al. 2023）：合并前随机丢弃部分适配器参数，并对剩余部分重新缩放。在组合多种能力上出乎意料地有效。
- **任务算术（Task arithmetic）**：直接对适配器权重做加减。把一个"代码"适配器和一个"数学"适配器相加，往往能得到一个两者都擅长的模型。

### 什么时候不该微调

微调是第三选项，不是第一选项。

**第一步：提示工程。** 写一个更好的系统提示词。加入少样本（few-shot）示例。使用思维链（chain-of-thought）。这不花一分钱，几分钟就能完成。如果提示工程能帮你达到 80% 的目标，你大概率不需要微调。

**第二步：RAG。** 如果模型需要了解你的特定数据（文档、知识库、产品目录），检索比把知识烤进权重更便宜、更易维护。参见 Lesson 06。

**第三步：微调。** 当你需要模型采用某种无法通过提示实现的特定风格、格式或推理模式时使用。当你需要稳定一致的结构化输出时。当你需要把一个大模型蒸馏到一个小模型时。当延迟很关键、你负担不起少样本提示带来的额外 token 时。

```mermaid
graph TD
    Start["Need better model behavior?"] --> PE["Try prompt engineering"]
    PE -->|"Works"| Done["Ship it"]
    PE -->|"Not enough"| RAG["Need external knowledge?"]
    RAG -->|"Yes"| RAGBuild["Build RAG pipeline"]
    RAG -->|"No, need style/format change"| FT["Fine-tune with LoRA/QLoRA"]
    RAGBuild -->|"Works"| Done
    RAGBuild -->|"Also need style change"| FT
    FT --> Done

    style Start fill:#1a1a2e,stroke:#e94560,color:#fff
    style Done fill:#0f3460,stroke:#16213e,color:#fff
```

```figure
lora-params
```

## 从零实现

我们用纯 PyTorch 从零实现 LoRA。不用任何库，没有任何魔法。你将构建 LoRA 层，把它注入模型，训练它，再把权重合并回去。

### 第 1 步：LoRA 层

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A = nn.Parameter(torch.randn(in_features, rank) * (1 / math.sqrt(rank)))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return (x @ self.A @ self.B) * self.scaling
```

A 用缩放后的随机值初始化，B 初始化为零。乘积 BA 从零开始，因此模型从其原始行为出发。

### 第 2 步：带 LoRA 的线性层封装

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

原始线性层被冻结，只有 LoRA 参数（A 和 B）是可训练的。

### 第 3 步：把 LoRA 注入模型

```python
def inject_lora(model, target_modules, rank=8, alpha=16):
    for param in model.parameters():
        param.requires_grad = False

    lora_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = dict(model.named_modules())[parent_name]
                lora_linear = LinearWithLoRA(module, rank, alpha)
                setattr(parent, child_name, lora_linear)
                lora_layers[name] = lora_linear
    return lora_layers
```

首先冻结模型中的所有参数。然后遍历模型树，找到名称与目标匹配的线性层，将其替换为带 LoRA 封装的版本。整个模型中，LoRA 的 A 和 B 矩阵是唯一可训练的参数。

### 第 4 步：统计参数量

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100 * trainable / total if total > 0 else 0
    }
```

### 第 5 步：把权重合并回去

```python
def merge_lora_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            with torch.no_grad():
                merged = (
                    module.lora.A @ module.lora.B
                ) * module.lora.scaling
                module.linear.weight.data += merged.T
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            setattr(parent, child_name, module.linear)
```

合并之后，LoRA 层就消失了。模型与原模型大小相同，适配信息已烤进权重中。没有任何推理开销。

### 第 6 步：模拟 QLoRA 量化

```python
def quantize_to_nf4(tensor, block_size=64):
    blocks = tensor.reshape(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True).values / 7.0
    scales = torch.clamp(scales, min=1e-8)
    quantized = torch.round(blocks / scales).clamp(-8, 7).to(torch.int8)
    return quantized, scales

def dequantize_from_nf4(quantized, scales, original_shape):
    dequantized = quantized.float() * scales
    return dequantized.reshape(original_shape)
```

这段代码通过在每 64 个权重的块内将权重映射到 16 个离散级别来模拟 4-bit 量化。生产环境的 QLoRA 使用 bitsandbytes 库在 GPU 上实现真正的 NF4。

### 第 7 步：训练循环

```python
def train_lora(model, data, epochs=5, lr=1e-3, batch_size=4):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        indices = torch.randperm(len(data["inputs"]))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            x = data["inputs"][batch_idx]
            y = data["targets"][batch_idx]

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

    return losses
```

### 第 8 步：完整演示

```python
def demo():
    torch.manual_seed(42)
    d_model = 256
    n_classes = 10

    model = nn.Sequential(
        nn.Linear(d_model, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, n_classes),
    )

    n_samples = 500
    x = torch.randn(n_samples, d_model)
    y = torch.randint(0, n_classes, (n_samples,))
    y_onehot = torch.zeros(n_samples, n_classes).scatter_(1, y.unsqueeze(1), 1.0)

    data = {"inputs": x, "targets": y_onehot}

    params_before = count_parameters(model)

    lora_layers = inject_lora(
        model, target_modules=["0", "2"], rank=8, alpha=16
    )

    params_after = count_parameters(model)

    losses = train_lora(model, data, epochs=20, lr=1e-3)

    merge_lora_weights(model)
    params_merged = count_parameters(model)

    return {
        "params_before": params_before,
        "params_after": params_after,
        "params_merged": params_merged,
        "losses": losses,
    }
```

这个演示创建一个小模型，向两个层注入 LoRA，训练它，再把权重合并回去。在 LoRA 训练期间，可训练参数占比从全部可训练降到约 1%，合并后又恢复为原始架构。

## 生产实践

借助 Hugging Face 生态，在真实模型上跑 LoRA 大约只需 20 行代码：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

要使用 QLoRA，加上 bitsandbytes 量化即可：

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

model = get_peft_model(model, lora_config)
```

就这么简单。训练循环不变，数据流水线不变。基座模型现在以 4-bit 存放，LoRA 适配器以 fp16 训练，整套东西能装进 6GB 显存。

使用 Hugging Face Trainer 进行训练：

```python
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

training_args = TrainingArguments(
    output_dir="./lora-llama",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./lora-adapter")
```

保存下来的适配器只有 10-100MB。基座模型保持原样。你可以在 Hugging Face Hub 上分享适配器，而无需重新分发完整模型。

## 交付产物

本课产出：
- `outputs/prompt-lora-advisor.md` —— 一个提示词，帮助你针对特定任务确定 LoRA 的秩、目标模块和超参数
- `outputs/skill-fine-tuning-guide.md` —— 一个技能，教会智能体何时以及如何微调的决策树

## 练习

1. **秩消融实验。** 分别用秩 2、4、8、16、32、64 运行演示。绘制最终损失随秩变化的曲线。找到收益递减点，即秩翻倍不再使损失减半的位置。对于 256 维特征上的简单分类任务，这个点应该在 r=8-16 附近。

2. **目标模块对比。** 修改 inject_lora，分别只针对层 "0"、只针对层 "2"、只针对层 "4"，以及同时针对三者。每个变体训练 20 个 epoch，比较收敛速度和最终损失。这对应真实场景中针对 q_proj、v_proj 还是所有线性层的取舍。

3. **量化误差分析。** 取训练后模型的权重矩阵，对比 quantize_to_nf4 / dequantize_from_nf4 前后的差异。计算均方误差、最大绝对误差，以及原始权重与重建权重之间的相关系数。尝试 32、64、128、256 等不同的 block_size 取值。

4. **多适配器服务。** 在数据的不同子集上（偶数索引与奇数索引）训练两个 LoRA 适配器，分别保存。只加载一次基座模型，然后切换适配器，验证两个适配器对同一输入产生不同的输出。这就是生产系统从一个基座模型服务多个微调模型的方式。

5. **合并与未合并的推理对比。** 在相同的 100 个输入上，比较 merge_lora_weights 前后 LoRA 模型的输出。验证输出一致（在 1e-5 的浮点容差内）。然后对两者的推理速度做基准测试——合并后的版本应该略快，因为它只做一次矩阵乘法而不是两次。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|----------------------|
| LoRA | "高效微调" | 低秩适配（Low-Rank Adaptation）：冻结基座权重，训练两个小矩阵 A 和 B，其乘积近似完整的权重更新 |
| QLoRA | "在笔记本上微调" | 量化 LoRA：以 4-bit NF4 加载基座模型，在其上以 fp16 训练 LoRA 适配器，使 7B 模型可在 6GB 显存中微调 |
| 秩（r） | "模型能学多少" | A 和 B 矩阵的内部维度；在表达能力与参数量之间权衡 |
| Alpha | "LoRA 的学习率" | 施加在 LoRA 输出上的缩放因子；alpha/r 决定了适配部分对最终输出的贡献比例 |
| NF4 | "4-bit 量化" | Normal Float 4：一种 4-bit 数据类型，量化级别取在正态分布的分位数上，对神经网络权重最优 |
| 适配器（Adapter） | "训练出来的那个小东西" | 保存为独立文件（10-100MB）的 LoRA A 和 B 矩阵，可加载到任意一份基座模型副本之上 |
| 目标模块（Target modules） | "给哪些层加 LoRA" | 注入 LoRA 适配器的特定线性层（q_proj、v_proj 等） |
| 合并（Merging） | "烤进去" | 计算 W + (alpha/r) * BA 并替换原始权重，消除推理时的适配器开销 |
| 分页优化器（Paged optimizers） | "训练时别 OOM" | 在 GPU 内存耗尽时，将优化器状态（Adam 动量、方差）卸载到 CPU |
| 灾难性遗忘（Catastrophic forgetting） | "微调把别的能力搞坏了" | 更新所有权重导致模型丢失先前学到的能力 |

## 延伸阅读

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021) —— 提出低秩分解方法的原始论文，在 GPT-3 175B 上验证了低至 4 的秩
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized Language Models" (2023) —— 引入 NF4、双重量化和分页优化器，使 65B 模型可在单块 48GB GPU 上微调
- PEFT 库文档（huggingface.co/docs/peft）—— Hugging Face 生态中 LoRA、QLoRA 及其他参数高效方法的标准库
- Yadav et al., "TIES-Merging: Resolving Interference When Merging Models" (2023) —— 在不损失质量的前提下组合多个 LoRA 适配器的技术
- [Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (NeurIPS 2023)](https://arxiv.org/abs/2305.18290) —— DPO 的推导；SFT 之后的偏好调优阶段，无需奖励模型。
- [TRL 文档](https://huggingface.co/docs/trl/) —— `SFTTrainer`、`DPOTrainer`、`KTOTrainer` 的官方参考，以及与 PEFT/bitsandbytes/Unsloth 的集成接口。
- [Unsloth 文档](https://docs.unsloth.ai/) —— 融合内核，可将微调吞吐量翻倍、内存减半；TRL 之下的性能层。
- [Axolotl 文档](https://axolotl-ai-cloud.github.io/axolotl/) —— 基于 YAML 配置的多 GPU SFT/DPO/QLoRA 训练器；手写脚本的"配置即代码"替代方案。
