# OCR 与文档理解

> OCR 是一个三阶段流水线——检测文本框、识别字符、再恢复版面。所有现代 OCR 系统要么重新排列这些阶段，要么把它们合并起来。

**Type:** Learn + Use
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 06 (Detection), Phase 7 Lesson 02 (Self-Attention)
**Time:** ~45 minutes

## 学习目标

- 梳理经典 OCR 流水线（检测 -> 识别 -> 版面）以及现代端到端方案（Donut、Qwen-VL-OCR）
- 实现 CTC（Connectionist Temporal Classification）损失，用于序列到序列的 OCR 训练
- 使用 PaddleOCR 或 EasyOCR 完成生产级文档解析，无需训练
- 区分 OCR、版面解析（layout parsing）和文档理解（document understanding），并为每类任务选对工具

## 问题背景

充满文字的图像无处不在：小票、发票、证件、扫描书籍、表单、白板、路牌、截图。从中提取结构化数据——不只是字符本身，而是「这是总金额」——是应用视觉领域价值最高的问题之一。

这个领域分为三个能力层次：

1. **狭义的 OCR**：把像素变成文本。
2. **版面解析**：把 OCR 输出归组为区域（标题、正文、表格、页眉）。
3. **文档理解**：从版面中提取结构化字段（"invoice_total = $42.50"）。

每一层都有经典方法和现代方法，而「我想从图片里提取文字」与「我需要从这张小票上拿到总金额」之间的差距，比大多数团队意识到的要大得多。

## 核心概念

### 经典流水线

```mermaid
flowchart LR
    IMG["Image"] --> DET["Text detection<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["Word/line<br/>bounding boxes"]
    BOX --> CROP["Crop each region"]
    CROP --> REC["Recognition<br/>(CRNN + CTC)"]
    REC --> TXT["Text strings"]
    TXT --> LAY["Layout<br/>ordering"]
    LAY --> OUT["Reading-order text"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **文本检测**产出按行或按词的四边形框。
- **识别**把每个区域裁剪到固定高度，用 CNN + BiLSTM + CTC 输出字符序列。
- **版面**重建阅读顺序（拉丁文字是从上到下、从左到右；阿拉伯文、日文则不同）。

### 一段话讲清 CTC

OCR 识别要从固定长度的特征图产出可变长度的序列。CTC（Graves et al., 2006）让你无需字符级对齐就能训练这种模型。模型在每个时间步输出一个覆盖（词表 + blank）的分布；CTC 损失对所有「合并重复字符并去掉 blank 后能还原成目标文本」的对齐方式做边缘化求和。

```
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```

CTC 是 CRNN 在 2015 年得以成功的原因，到 2026 年仍是大多数生产级 OCR 模型的训练方式。

### 现代端到端模型

- **Donut**（Kim et al., 2022）——ViT 编码器 + 文本解码器；读入图像直接输出 JSON。没有文本检测器，也没有版面模块。
- **TrOCR**——ViT + Transformer 解码器，做行级 OCR。
- **Qwen-VL-OCR / InternVL**——为 OCR 任务微调的完整视觉语言模型；2026 年在复杂文档上精度最高。
- **PaddleOCR**——经典的 DB + CRNN 流水线，打包成成熟的生产级工具；至今仍是开源主力。

端到端模型需要更多数据和算力，但避开了多阶段流水线的误差累积。

### 版面解析

对结构化文档，运行一个版面检测器（LayoutLMv3、DocLayNet），给每个区域打标签：标题、段落、图、表、脚注。阅读顺序随之变成「按版面顺序遍历区域并拼接」。

对表单，使用**键值提取（Key-Value extraction）**模型（视觉信息丰富的文档用 Donut，普通扫描件用 LayoutLMv3）。它们接收图像 + 检测到的文本 + 位置信息，预测结构化的键值对。

### 评估指标

- **字符错误率（Character Error Rate, CER）**——Levenshtein 距离除以参考文本长度。越低越好。生产目标：干净扫描件上 < 2%。
- **词错误率（Word Error Rate, WER）**——同样的指标，按词计算。
- **结构化字段的 F1**——用于键值任务；衡量 `{invoice_total: 42.50}` 是否被正确抽取。
- **JSON 上的编辑距离**——用于端到端文档解析；Donut 论文引入了归一化的树编辑距离。

## 从零实现

### 第 1 步：CTC 损失 + 贪心解码器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax over vocab including blank at index 0
    targets:        (N, S) int targets (no blanks)
    input_lengths:  (N,) per-sample time steps used
    target_lengths: (N,) per-sample target length
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (blanks removed, repeats merged)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss` 在可用时会调用高效的 CuDNN 实现。贪心解码器比束搜索（beam search）简单得多，且 CER 通常只差 1% 以内。

### 第 2 步：迷你 CRNN 识别器

用于行级 OCR 的最小 CNN + BiLSTM。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

输入高度固定（CNN 通过最大池化把高度压到 1）。宽度就是 CTC 的时间维度。

### 第 3 步：合成 OCR 数据

生成白底黑字的数字串，做端到端冒烟测试。

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

真实的 OCR 数据集还要加上字体、噪声、旋转、模糊和颜色变化。流水线本身与上面完全一致。

### 第 4 步：训练框架

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

在这份简单的合成数据上，损失应在 200 步内从约 3 降到约 0.2。

## 生产实践

三条生产路径：

- **PaddleOCR**——成熟、快速、多语言。一行调用：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。
- **EasyOCR**——纯 Python、多语言、PyTorch 骨干网络。
- **Tesseract**——经典方案；当模型搞不定老旧扫描文档时仍然有用。

端到端文档解析则使用 Donut 或 VLM：

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

对结构可重复的小票、发票和表单，微调 Donut。对任意文档或需要推理的 OCR，目前的默认选择是 Qwen-VL-OCR 这类 VLM。

## 交付产物

本课产出：

- `outputs/prompt-ocr-stack-picker.md`——一个提示词，根据文档类型、语言和结构在 Tesseract / PaddleOCR / Donut / VLM-OCR 之间做选择。
- `outputs/skill-ctc-decoder.md`——一个技能，从零编写贪心和束搜索 CTC 解码器，包含长度归一化。

## 练习

1. **（简单）**在 5 位随机数字串上训练 TinyCRNN 500 步。报告留出集上的 CER。
2. **（中等）**用束搜索（beam_width=5）替换贪心解码。报告 CER 差值。束搜索在哪些输入上占优？
3. **（困难）**用 PaddleOCR 处理 20 张小票，提取明细条目，并针对手工标注的 {item_name, price} 真值计算 F1。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| OCR | 「从像素提取文字」 | 把图像区域转换为字符序列 |
| CTC | 「免对齐的损失」 | 无需逐时间步标签即可训练序列模型的损失；对所有对齐方式做边缘化 |
| CRNN | 「经典 OCR 模型」 | 卷积特征提取器 + BiLSTM + CTC；2015 年的基线，至今仍在生产环境使用 |
| Donut | 「端到端 OCR」 | ViT 编码器 + 文本解码器；从图像直接输出 JSON |
| 版面解析 | 「找区域」 | 检测并标注文档中的标题/表格/图/段落区域 |
| 阅读顺序 | 「文本序列」 | 把识别出的区域排列成连贯文本；拉丁文字很简单，混合版面则不简单 |
| CER / WER | 「错误率」 | Levenshtein 距离除以参考长度，分别按字符或词粒度计算 |
| VLM-OCR | 「会读字的 LLM」 | 为 OCR 任务训练或提示的视觉语言模型；复杂文档上的当前 SOTA |

## 延伸阅读

- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717) ——最初的 CNN+RNN+CTC 架构
- [CTC (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) ——CTC 原始论文；算法思想密度极高
- [Donut (Kim et al., 2022)](https://arxiv.org/abs/2111.15664) ——无需 OCR 的文档理解 Transformer
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) ——开源生产级 OCR 技术栈
