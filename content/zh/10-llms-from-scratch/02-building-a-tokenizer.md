# 从零构建一个分词器

> 第 01 课给了你一个玩具。这一课给你一件武器。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 01 (Tokenizers: BPE, WordPiece, SentencePiece)
**Time:** ~90 minutes

## 学习目标

- 构建一个生产级 BPE 分词器（tokenizer），能处理 Unicode、空白符归一化和特殊 token
- 实现字节级回退（byte-level fallback），让分词器可以编码任何输入（包括 emoji、中日韩文字和代码）而不产生未知 token
- 添加预分词正则模式，在应用 BPE 合并之前按词边界切分文本
- 在语料上训练自定义分词器，并在多语言文本上与 tiktoken 对比压缩率

## 问题背景

你在第 01 课写的 BPE 分词器在英文文本上能用。现在给它喂日语试试。或者 emoji。或者混用制表符和空格的 Python 代码。

它会崩。

不是因为 BPE 错了——而是因为实现不完整。生产级分词器要能处理任意编码的原始字节，在切分前做 Unicode 归一化，管理永远不参与合并的特殊 token，把预分词和子词切分串联起来，而且这一切都要足够快，才不会成为处理 15 万亿 token 的训练流水线的瓶颈。

GPT-2 的分词器有 50,257 个 token。Llama 3 有 128,256 个。GPT-4 大约 100,000 个。这些不是玩具量级的数字。这些词表背后的合并表是在数百 GB 文本上训练出来的，而围绕它的整套机制——归一化、预分词、特殊 token 注入、对话模板格式化——正是「能处理 hello world」的分词器与「能处理整个互联网」的分词器之间的差别。

你要构建的就是这套机制。

## 核心概念

### 完整流水线

生产级分词器不是单一算法，而是一条由五个阶段组成的流水线，每个阶段解决一个不同的问题。

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段都有明确的职责：

| 阶段 | 做什么 | 为什么重要 |
|-------|-------------|----------------|
| 归一化 | NFKC Unicode 归一化，可选小写化，可选去除重音符号 | "fi" 连字（U+FB01）会变成 "fi"（两个字符）。没有这一步，同一个词会得到不同的 token。 |
| 预分词 | 在 BPE 之前把文本切成块 | 防止 BPE 跨词边界合并。"the cat" 永远不应该产生 "e c" 这样的 token。 |
| BPE 合并 | 对字节序列应用学到的合并规则 | 核心压缩步骤。把原始字节变成子词 token。 |
| 特殊 token | 注入 [BOS]、[EOS]、[PAD] 以及对话模板标记 | 这些 token 有固定 ID，从不参与 BPE 合并。模型需要它们来表达结构。 |
| ID 映射 | 把 token 字符串转换为整数 ID | 模型看到的是整数，不是字符串。 |

### 字节级 BPE

第 01 课的分词器是在 UTF-8 字节上操作的。这个选择没错。但我们跳过了一个重要问题：如果那些字节不是合法的 UTF-8 怎么办？

字节级 BPE 的解法是把每个可能的字节值（0-255）都当作合法 token。基础词表恰好是 256 个条目。任何文件——文本、二进制、损坏的——都能被分词，且不会产生未知 token。

GPT-2 加了一个小技巧：把每个字节映射到一个可打印的 Unicode 字符，让词表保持人类可读。在他们的映射里，字节 0x20（空格）会变成字符 "G"。这纯粹是为了显示，算法本身并不在意。

真正的威力在于：字节级 BPE 能处理地球上所有语言。汉字每个占 3 个 UTF-8 字节。日文可能是 3-4 字节。阿拉伯文、天城文、emoji——全都只是字节序列。BPE 算法在这些字节序列中寻找模式的方式，和在英文 ASCII 字节中完全一样。

### 预分词

在 BPE 接触你的文本之前，需要先把它切成块。这能防止合并算法创造出跨越词边界的 token。

GPT-2 用一个正则模式来切分文本：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

这个模式会切分缩写（"don't" 变成 "don" + "'t"）、带可选前导空格的单词、数字、标点和空白符。前导空格保留在单词上——所以 "the cat" 变成 [" the", " cat"]，而不是 ["the", " ", "cat"]。

Llama 使用 SentencePiece，完全跳过正则。它把原始字节流当作一个长序列，让 BPE 算法自己找出边界。这更简单，但也给了 BPE 更多自由去创造跨词 token。

这个选择是有影响的。GPT-2 的正则阻止分词器学到「一个词结尾的 "the" 和下一个词开头的 "the" 应该合并」。SentencePiece 允许这种合并，有时压缩效率更高，但 token 的可解释性更差。

### 特殊 Token

每个生产级分词器都会为结构标记保留 token ID：

| Token | 用途 | 使用者 |
|-------|---------|---------|
| `[BOS]` / `<s>` | 序列开始 | Llama 3, GPT |
| `[EOS]` / `</s>` | 序列结束 | 所有模型 |
| `[PAD]` | 用于批次对齐的填充 | BERT, T5 |
| `[UNK]` | 未知 token（字节级 BPE 消除了它） | BERT, WordPiece |
| `<\|im_start\|>` | 对话消息边界开始 | ChatGPT, Qwen |
| `<\|im_end\|>` | 对话消息边界结束 | ChatGPT, Qwen |
| `<\|user\|>` | 用户轮次标记 | Llama 3 |
| `<\|assistant\|>` | 助手轮次标记 | Llama 3 |

特殊 token 永远不会被 BPE 拆分。它们在合并算法运行之前被精确匹配、替换为固定 ID，周围的文本则正常分词。

### 对话模板

这是最容易让人困惑、也最容易让实现出错的地方。

当你向对话模型发送消息时，API 接受的是一个消息列表：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型看不到 JSON。它看到的是一个扁平的 token 序列。对话模板（chat template）负责用特殊 token 把消息列表转换成那个扁平序列。每个模型的做法都不一样：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

模板写错，模型就会输出垃圾。它是在一种精确格式上训练出来的。任何偏差——少一个换行、换错一个 token、多一个空格——都会让输入落在训练分布之外。

### 速度

对生产环境的分词来说，Python 太慢了。

tiktoken（OpenAI）是用 Rust 写的，带 Python 绑定。HuggingFace tokenizers 也是 Rust。SentencePiece 是 C++。它们比纯 Python 快 10-100 倍。

给个直观感受：为 Llama 3 预训练对 15 万亿 token 分词，按每秒 100 万 token（很快的 Python）算需要 174 天；按每秒 1 亿 token（Rust）算只需 1.7 天。

你用 Python 实现是为了理解算法。在生产环境中，你会使用编译型实现，只接触它的 Python 封装层。

```figure
weight-tying
```

## 从零实现

### 第 1 步：字节级编码

地基。把任意字符串转换为字节序列，把每个字节映射为可打印字符以便显示，并能逆向还原。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

在多语言文本上测试，看看各自的字节数：

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" 是 5 个字节。"你好" 是 6 个字节（每个字符 3 字节）。火焰 emoji 是 4 个字节。字节级分词器不在乎是什么语言。字节就是字节。

### 第 2 步：基于正则的预分词器

用 GPT-2 的正则模式把文本切成块。每个块由 BPE 独立分词。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 模块支持 Unicode 属性转义（`\p{L}` 匹配字母，`\p{N}` 匹配数字）。标准库的 `re` 模块不支持，所以我们回退到 ASCII 字符类。要做生产环境的多语言分词器，请安装 `regex`。

试一下：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

前导空格保留在单词上。缩写在撇号处切开。标点自成一块。BPE 永远不会跨这些边界合并 token。

### 第 3 步：字节序列上的 BPE

第 01 课的核心算法，但现在是在预分词后的各个块上独立运行。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### 第 4 步：特殊 Token 处理

特殊 token 需要精确匹配和固定 ID。它们完全绕过 BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### 第 5 步：完整的分词器类

把所有环节串起来：归一化、按特殊 token 切分、预分词、BPE 合并、映射为 ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 第 6 步：多语言测试

真正的考验。把英文、中文、emoji 和代码一起扔给它。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

每个汉字产生 3 个字节。emoji 产生 4 个字节。这些都不会让分词器崩溃，也都不会产生未知 token。这就是字节级 BPE 的威力。

## 生产实践

### 对比真实分词器

加载 Llama 3、GPT-4 和 Mistral 的真实分词器，看看它们各自如何处理同一段多语言文本。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

你会看到同一段文本得到不同的 token 数。拥有 128K 词表的 Llama 3 在合并常见模式上更激进。100K 词表的 GPT-4 居中。32K 词表的 Mistral 产生更多 token，但嵌入层更小。

权衡始终是同一个：词表越大，序列越短，但参数越多。

## 交付产物

本课产出一个用于构建和调试生产级分词器的提示词。见 `outputs/prompt-tokenizer-builder.md`。

## 练习

1. **简单：** 添加一个 `get_token_bytes(id)` 方法，显示任意 token ID 对应的原始字节。用它检查你最常见的合并 token 实际代表什么。
2. **中等：** 实现 Llama 风格的预分词器，按空白符和数字切分但保留前导空格。在同一语料上，把它得到的词表与 GPT-2 正则方案做对比。
3. **困难：** 添加一个对话模板方法，接收 `{"role": ..., "content": ...}` 消息列表，产出符合 Llama 3 对话格式的正确 token 序列。与 HuggingFace 的实现对比验证。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 字节级 BPE | "在字节上工作的分词器" | 基础词表为 256 个字节值的 BPE——能处理任何输入，不产生未知 token |
| 预分词 | "BPE 之前的切分" | 基于正则或规则的切分，防止 BPE 跨词边界合并 |
| NFKC 归一化 | "Unicode 清理" | 先做规范分解，再做兼容性组合——"fi" 连字变成 "fi"，全角 "A" 变成 "A" |
| 对话模板 | "消息如何变成 token" | 把 role/content 消息列表转换为扁平 token 序列的精确格式——因模型而异，必须与训练格式一致 |
| 特殊 token | "控制 token" | 绕过 BPE 的保留 token ID——[BOS]、[EOS]、[PAD]、对话标记——在合并之前被精确匹配 |
| Fertility | "每个词的 token 数" | 输出 token 数与输入词数之比——GPT-4 上英文约 1.3，韩文 2-3，越高意味着浪费的上下文越多 |
| tiktoken | "OpenAI 的分词器" | 带 Python 绑定的 Rust BPE 实现——比纯 Python 快 10-100 倍 |
| 合并表 | "词表" | 训练中学到的字节对合并规则的有序列表——这就是分词器学到的全部知识 |

## 延伸阅读

- [OpenAI tiktoken source](https://github.com/openai/tiktoken) —— GPT-3.5/4 使用的 Rust BPE 实现
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) —— 支持 BPE、WordPiece、Unigram 的 Rust 分词器库
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783) —— 128K 词表与分词器训练的细节
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) —— 语言无关的分词方法
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py) —— 原版的字节到 Unicode 映射
