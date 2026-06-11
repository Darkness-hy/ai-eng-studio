# 说话人识别与验证

> ASR 问的是"他们说了什么？"，说话人识别问的是"这是谁说的？"。两者的数学形式看上去一样——嵌入加余弦相似度——但生产环境中的每个决策都取决于一个 EER 数值。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 22 (Embedding Models)
**Time:** ~45 minutes

## 问题背景

用户说出一句口令。你想知道：这个人是不是他声称的那个人（*验证*，1:1）？还是注册库里的某个人（*辨认*，1:N）？又或者都不是——这是一个未知说话人（*开集*）？

2018 年之前：GMM-UBM + i-vector。EER 还算可以，但对信道偏移（手机 vs 笔记本电脑）和情绪变化很脆弱。2018–2022：x-vector（用角度间隔损失训练的 TDNN 骨干网络）。2022 年之后：ECAPA-TDNN 和 WavLM-large 嵌入。到 2026 年，整个领域被三个模型和一个指标主导。

这个指标就是 **EER**——等错误率（Equal Error Rate）。把决策阈值调到误接受率（False Accept Rate）等于误拒绝率（False Reject Rate）的位置，交叉点就是 EER。每篇论文、每个排行榜、每次采购招标都用它。

## 核心概念

![Enrollment + verification pipeline with embedding + cosine + EER](../assets/speaker-verification.svg)

**整体流程。** 注册（enrollment）：录制目标说话人 5–30 秒的语音；计算固定维度的嵌入（ECAPA-TDNN 为 192 维，WavLM-large 为 256 维）。验证（verification）：取得测试语音的嵌入；计算余弦相似度；与阈值比较。

**ECAPA-TDNN（2020 年提出，2026 年仍是主流）。** 全称 Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network。带 squeeze-excitation 的一维卷积块，加多头注意力池化，再接一个线性层映射到 192 维。在 VoxCeleb 1+2 上训练（2,700 名说话人，110 万条语音），使用加性角度间隔损失（AAM-softmax）。

**WavLM-SV（2022 年之后）。** 用 AAM 损失微调预训练的 WavLM-large 自监督骨干网络。质量更高但更慢——300+ MB 对比 15 MB。

**x-vector（基线）。** TDNN + 统计池化。经典方案；在 CPU / 边缘设备上仍然有用。

**AAM-softmax。** 在标准 softmax 的角度空间中加入间隔 `m`：对正确类别使用 `cos(θ + m)`。强制拉开类间角度间隔。典型取值 `m=0.2`，缩放因子 `s=30`。

### 打分

- **余弦相似度**，在注册嵌入和测试嵌入之间计算。基于阈值做决策。
- **PLDA（概率线性判别分析，Probabilistic LDA）。** 把嵌入投影到一个潜空间，在其中"同一说话人 vs 不同说话人"有闭式解的似然比。叠加在余弦相似度之上，可让 EER 再降 10–20%。2020 年之前是标准做法；如今只在闭集场景中使用。
- **分数归一化。** `S-norm` 或 `AS-norm`：用一组冒充者（imposter）队列的均值和标准差对每个分数做归一化。跨域评估时必不可少。

### 你应该记住的数字（2026）

| 模型 | VoxCeleb1-O EER | 参数量 | 吞吐量（A100） |
|-------|-----------------|--------|-------------------|
| x-vector（经典） | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 分割 + 嵌入 | 0.65% | 6 M | 100× RT |
| ReDimNet（2024） | 0.39% | 24 M | 100× RT |

### 说话人日志（Diarization）

回答多说话人音频中"谁在什么时候说话"。流程：VAD → 切分 → 对每段计算嵌入 → 聚类（凝聚式或谱聚类）→ 平滑边界。现代技术栈：`pyannote.audio` 3.1，一次调用就打包了说话人分割 + 嵌入 + 聚类。2026 年 AMI 数据集上的 SOTA DER 约为 15%（2022 年还是 23%）。

## 从零实现

### 第 1 步：用 MFCC 统计量构造玩具嵌入

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

离 SOTA 差得很远——仅作教学用途。`code/main.py` 用它在合成说话人数据上做概念验证。

### 第 2 步：余弦相似度 + 阈值

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 第 3 步：从相似度配对计算 EER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

返回 (eer, threshold_at_eer)。两个都要报告。

### 第 4 步：用 SpeechBrain 上生产

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### 第 5 步：用 pyannote 做说话人日志

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 生产实践

2026 年的技术栈：

| 场景 | 选型 |
|-----------|------|
| 闭集 1:1 验证，边缘设备 | ECAPA-TDNN + 余弦阈值 |
| 开集验证，云端 | WavLM-SV + AS-norm |
| 说话人日志（会议、播客） | `pyannote/speaker-diarization-3.1` |
| 反欺骗（重放 / deepfake 检测） | AASIST 或 RawNet2 |
| 超小型嵌入式（KWS + 注册） | Titanet-Small（NeMo） |

## 常见陷阱

- **信道不匹配。** 在 VoxCeleb（网络视频）上训练的模型 ≠ 电话录音场景。务必在目标信道上评估。
- **短语音。** 测试音频低于 3 秒时 EER 会急剧恶化。
- **带噪注册。** 一条带噪的注册语音就会污染锚点嵌入。使用 ≥3 条干净样本并取平均。
- **跨条件使用固定阈值。** 务必在目标域的留出开发集上调阈值。
- **对未归一化的嵌入算余弦。** 先做 L2 归一化；否则向量模长会主导结果。

## 交付产物

保存为 `outputs/skill-speaker-verifier.md`。确定模型选型、注册协议、阈值调优方案和反欺诈防护措施。

## 练习

1. **简单。** 运行 `code/main.py`。构造合成"说话人"（不同的音色特征），完成注册，在 100 对试验列表上计算 EER。
2. **中等。** 用 SpeechBrain 的 ECAPA 处理 30 条 VoxCeleb1 语音（5 名说话人 × 每人 6 条）。分别用余弦和 PLDA 计算 EER。
3. **困难。** 用 `pyannote.audio` 搭建完整的注册 → 日志 → 验证流程。在 AMI 开发集上评估 DER。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| EER | 头条指标 | 误接受率 = 误拒绝率时对应的阈值点。 |
| 验证（Verification） | 1:1 | "这是 Alice 吗？" |
| 辨认（Identification） | 1:N | "现在是谁在说话？" |
| 开集（Open-set） | 可能有未知人 | 测试集中可能包含未注册的说话人。 |
| 注册（Enrollment） | 登记 | 计算说话人的参考嵌入。 |
| AAM-softmax | 那个损失函数 | 带加性角度间隔的 softmax；强制簇间分离。 |
| PLDA | 经典打分法 | 概率线性判别分析；在嵌入之上做似然比打分。 |
| DER | 说话人日志指标 | 日志错误率（Diarization Error Rate）——漏检 + 虚警 + 说话人混淆。 |

## 延伸阅读

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) —— 深度嵌入的经典论文。
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) —— 2020–2026 年的主流架构。
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) —— 用于说话人验证和日志的自监督骨干网络。
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) —— 生产级说话人日志 + 嵌入技术栈。
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) —— 各模型当前的 EER 排名。
