# Capstone 07 — 端到端微调流水线（从数据到 SFT、DPO 再到服务）

> 用你自己的数据训练一个 8B 模型，用你自己的偏好做 DPO 对齐，量化、加上投机解码，并以可度量的每百万 token 成本提供服务。2026 年的开源技术栈是：Axolotl v0.8、TRL 0.15、用 Unsloth 做快速迭代、GPTQ/AWQ/GGUF 做量化、vLLM 0.7 配 EAGLE-3 做服务。这个 Capstone 的目标是可复现地跑通整条流水线——输入 YAML，输出已上线的服务端点——并按 2026 年模型开放框架（Model Openness Framework）发布一份模型卡。

**Type:** Capstone
**Languages:** Python (pipeline), YAML (configs), Bash (scripts)
**Prerequisites:** Phase 2 (ML), Phase 3 (DL), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 11 (LLM engineering), Phase 17 (infrastructure), Phase 18 (safety)
**涉及阶段：** P2 · P3 · P7 · P10 · P11 · P17 · P18
**Time:** 35 hours

## 问题背景

2026 年，每个认真做 AI 的团队都会常备一条随时可用的微调流水线。不是因为他们要发布前沿基座模型，而是因为下游适配——领域 SFT、基于标注偏好的 DPO、为投机解码蒸馏草稿模型、用 EAGLE-3 提供服务——才是可度量收益真正所在的地方。Axolotl v0.8 负责多 GPU 的 SFT 配置，TRL 0.15 负责 DPO 和 GRPO，Unsloth 让你在单 GPU 上快速迭代，vLLM 0.7 配合 EAGLE-3 能在不损失质量的前提下把解码吞吐提升 2-3 倍。工具链已经成熟；真正的功夫在 YAML 配置、数据卫生和评测纪律上。

你将拿一个 8B 基座模型（Llama 3.3、Qwen3 或 Gemma 3），在任务相关数据上先做 SFT 再做 DPO，量化后用于推理服务，并用 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 衡量收益。你还要按 2026 年模型开放框架（Model Openness Framework）产出一份模型卡。重点是可复现性——一条命令就能端到端重跑整条流水线。

## 核心概念

流水线分五个阶段。**数据**：去重（MinHash / Datatrove）、质量过滤（Nemotron-CC 风格的分类器）、PII 清洗、针对公开基准污染的数据划分卫生检查。**SFT**：Axolotl YAML 配置，8xH100 上跑 ZeRO-3，余弦学习率调度，序列打包，训练 2-3 个 epoch。**DPO 或 GRPO**：TRL 配置，1 个 epoch，偏好对来自人工标注或模型评判，调优 beta。**量化**：GPTQ + AWQ + GGUF，保证部署灵活性。**服务**：vLLM 0.7 加 EAGLE-3 投机解码头（或 SGLang 配 SpecForge），K8s 部署，基于队列等待时间的 HPA 自动扩缩。

消融实验本身就是交付物：在三个任务相关基准上对比仅 SFT、SFT+DPO、SFT+GRPO。服务指标：批大小 1 / 8 / 32 下的 tokens/s、EAGLE-3 接受率、每百万 token 的美元成本。安全评测：Llama Guard 4 通过率。模型卡：偏见评估、可复现的随机种子、数据许可。

## 架构

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## 技术栈

- 数据：Datatrove 做去重，Nemotron-CC 分类器做质量过滤，Presidio 处理 PII
- 基座模型：Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B
- SFT：Axolotl v0.8，配 ZeRO-3、Flash Attention 3、序列打包
- 偏好调优：TRL 0.15 跑 DPO 或 GRPO；Unsloth 用于单 GPU 迭代
- 量化：GPTQ（Marlin）、AWQ、经 llama.cpp 产出的 GGUF
- 服务：vLLM 0.7 加 EAGLE-3 投机解码（或 SGLang 0.4 + SpecForge）
- 评测：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全评测：Llama Guard 4、ShieldGemma-2
- 基础设施：Kubernetes + NVIDIA device plugin，基于队列等待指标的 HPA
- 可观测性：训练用 W&B，推理用 Langfuse

## 从零实现

1. **数据流水线。** 对原始语料运行 Datatrove 去重，应用 Nemotron-CC 风格的质量分类器，用 Presidio 清洗 PII，使用显式随机种子写出训练/验证划分。

2. **污染检查。** 对每个验证划分，与 MMLU-Pro、MT-Bench-v2、RewardBench-2 的测试集计算 MinHash，拒绝任何重叠。

3. **Axolotl SFT。** YAML 配置 ZeRO-3、FA3、序列打包，在 8xH100 上训练 2-3 个 epoch，日志写入 W&B。

4. **TRL DPO / GRPO。** 取 SFT 检查点，在偏好对上跑一个 epoch 的 DPO（或在数学/代码任务上用可验证奖励跑 GRPO），对 beta 做参数扫描。

5. **量化。** 产出三种量化版本：GPTQ-INT4-Marlin、AWQ-INT4，以及供 llama.cpp 使用的 GGUF-Q4_K_M。记录模型体积和名义吞吐量。

6. **带投机解码的服务。** vLLM 0.7 配置 EAGLE-3 草稿头（用 Red Hat Speculators 训练）。在批大小 1 / 8 / 32 下测量接受率和尾延迟。在同一评测上报告相对 Anthropic / OpenAI 的每百万 token 成本。

7. **评测矩阵。** 在基座、仅 SFT、SFT+DPO、SFT+GRPO 四个版本上运行 lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro，产出对照表。

8. **安全评测。** 在开发集上统计 Llama Guard 4 通过率，并加上 ShieldGemma-2 输出过滤。

9. **模型卡。** 按 MOF 2026 模板编写：数据、训练、评测、安全、许可，以及包含 YAML 配置和 commit SHA 的可复现性章节。

## 生产实践

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## 交付产物

`outputs/skill-finetuning-pipeline.md` 描述了交付物。一条命令依次完成数据处理、SFT、DPO、量化、服务部署和评测，最终产出模型卡和已上线的服务端点。

| 权重 | 评分标准 | 度量方式 |
|:-:|---|---|
| 25 | 相对基座的评测增益 | 在目标任务上实测的提升（MMLU-Pro、MT-Bench-v2、任务相关基准） |
| 20 | 流水线可复现性 | 一条命令用相同随机种子端到端重跑 |
| 20 | 数据卫生 | 去重率、PII 清洗覆盖率、污染检查通过 |
| 20 | 服务效率 | bs=1/8/32 下的 tokens/s、EAGLE-3 接受率、每百万 token 成本 |
| 15 | 模型卡 + 安全评测 | 2026 MOF 完整度 + Llama Guard 4 通过率 |
| **100** | | |

## 练习

1. 在同一个任务相关基准上分别运行仅 SFT、SFT+DPO、SFT+GRPO。报告哪种偏好方法胜出，以及领先多少。

2. 把 Llama 3.3 8B 换成 Qwen3 14B。在质量持平的条件下测量每百万 token 成本。

3. 分别在领域数据和通用 ShareGPT 数据上测量 EAGLE-3 接受率。报告差值，以及它对延迟预算意味着什么。

4. 注入 1% 的污染（把 MMLU-Pro 答案泄漏进训练数据）并重跑评测。观察 MMLU-Pro 准确率不真实地飙升。构建一个能拦截这种问题的污染检查 CI 门禁。

5. 增加 LoRA SFT 作为全量微调的替代方案。在内存占用低 10 倍的条件下测量质量差距。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Axolotl | “SFT 训练器” | 由 YAML 驱动的统一训练器，支持 SFT、DPO 和蒸馏 |
| TRL | “偏好调优器” | Hugging Face 出品的库，用于在 LLM 上做 DPO、GRPO、PPO |
| GRPO | “组相对策略优化” | DeepSeek R1 的 RL 配方，基于可验证奖励 |
| EAGLE-3 | “投机解码草稿” | 提前预测 N 个 token 的草稿头；vLLM 用目标模型进行验证 |
| MOF | “模型开放框架” | 2026 年的标准，从数据、代码、许可三方面给模型发布评级 |
| 污染检查 | “数据划分卫生” | 基于 MinHash 检测测试集是否泄漏进训练数据 |
| 接受率 | “EAGLE / MTP 指标” | 草稿 token 中被目标模型接受的比例 |

## 延伸阅读

- [Axolotl 文档](https://axolotl-ai-cloud.github.io/axolotl/) — 参考级的 SFT / DPO 训练器
- [TRL 文档](https://huggingface.co/docs/trl) — DPO 和 GRPO 的参考实现
- [Unsloth](https://github.com/unslothai/unsloth) — 单 GPU 迭代的参考工具
- [DeepSeek R1 论文 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO 方法论
- [vLLM + EAGLE-3 文档](https://docs.vllm.ai) — 参考服务栈
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 另一个投机解码训练器
- [Model Openness Framework 2026](https://isocpp.org/) — 开放发布评级标准
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 标准评测运行器
