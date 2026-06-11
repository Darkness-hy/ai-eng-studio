# 自托管推理服务选型 — llama.cpp、Ollama、TGI、vLLM、SGLang

> 2026 年自托管推理由四大引擎主导，选型依据是硬件、规模和生态。**llama.cpp** 在 CPU 上最快——模型支持最广，对量化和线程调度有完全的控制权。**Ollama** 是开发笔记本上的一条命令即装方案，比 llama.cpp 慢约 15-30%（Go + CGo + HTTP 序列化），在类生产负载下吞吐量差距达 3 倍。**TGI 已于 2025 年 12 月 11 日进入维护模式**——此后只修 bug，原始吞吐量比 vLLM 慢约 10%，但其可观测性和 HF 生态集成历来是顶级水准。维护状态使它成为高风险的长期选择——新项目更稳妥的默认选项是 SGLang 或 vLLM。**vLLM** 是通用的生产默认引擎——v0.15.1（2026 年 2 月）新增 PyTorch 2.10、RTX Blackwell SM120、H200 优化。**SGLang** 是智能体多轮对话 / 前缀密集场景的专家——已在生产环境部署 400,000+ 块 GPU（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。硬件约束：纯 CPU → 只能选 llama.cpp。AMD / 非 NVIDIA → 只能选 vLLM（TRT-LLM 锁定 NVIDIA）。2026 年的流水线模式：开发 = Ollama，预发 = llama.cpp，生产 = vLLM 或 SGLang。全程使用同一份 GGUF/HF 权重。

**Type:** Learn
**Languages:** Python (stdlib, engine-decision tree walker)
**Prerequisites:** All Phase 17 lessons covering engines (04, 06, 07, 09, 18)
**Time:** ~45 minutes

## 学习目标

- 根据硬件（CPU / AMD / NVIDIA Hopper / Blackwell）、规模（1 用户 / 100 / 10,000）和工作负载（通用聊天 / 智能体 / 长上下文）选出合适的引擎。
- 说出 2026 年 TGI 的维护模式状态（2025 年 12 月 11 日），以及为什么这使新项目倾向于 vLLM 或 SGLang。
- 描述全程使用同一份 GGUF 或 HF 权重的开发/预发/生产流水线。
- 解释为什么"只有 CPU"意味着必须用 llama.cpp，以及"AMD"为什么排除了 TRT-LLM。

## 问题背景

你的团队启动一个新的自托管 LLM 项目。一位工程师说用 Ollama，另一位说用 vLLM，第三位说"TGI 不是开箱即用吗？"三个人在各自的语境下都对，但没有一个在所有场景下都对。

到了 2026 年，这棵决策树很重要：硬件第一，规模第二，工作负载第三。而 2025 年的一个具体事件——TGI 于 12 月 11 日进入维护模式——改变了新项目的默认选择。

## 核心概念

### 五大引擎

| 引擎 | 最适合 | 备注 |
|--------|----------|-------|
| **llama.cpp** | CPU / 边缘设备 / 最少依赖 / 最广模型支持 | CPU 上最快，控制权完整 |
| **Ollama** | 开发笔记本、单用户、一条命令即装 | 比 llama.cpp 慢 15-30%；生产吞吐量差 3 倍 |
| **TGI** | HF 生态、受监管行业 | **2025 年 12 月 11 日进入维护模式** |
| **vLLM** | 通用生产环境、100+ 用户 | 广泛的生产默认选项；v0.15.1 发布于 2026 年 2 月 |
| **SGLang** | 智能体多轮对话、前缀密集型工作负载 | 生产环境部署 400,000+ 块 GPU |

### 第一层决策：硬件

**纯 CPU** → llama.cpp。Ollama 也能用但更慢。其他引擎在 CPU 上都没有竞争力。

**AMD GPU** → vLLM（支持 AMD ROCm）。SGLang 也可以。TRT-LLM 锁定 NVIDIA，直接出局。

**NVIDIA Hopper（H100 / H200）** → vLLM、SGLang 或 TRT-LLM。三者都是第一梯队。

**NVIDIA Blackwell（B200 / GB200）** → TRT-LLM 是吞吐量领先者（Phase 17 · 07）。vLLM 和 SGLang 紧随其后。

**Apple Silicon（M 系列）** → llama.cpp（Metal）。Ollama 封装了它。

### 第二层决策：规模

**1 用户 / 本地开发** → Ollama。一条命令，几秒内出首个 token。

**10-100 用户 / 小团队** → vLLM 单 GPU。

**100-10k 用户 / 生产环境** → vLLM production-stack（Phase 17 · 18）或 SGLang。

**10k+ 用户 / 企业级** → vLLM production-stack + 分离式部署（Phase 17 · 17）+ LMCache（Phase 17 · 18）。

### 第三层决策：工作负载

**通用聊天 / 问答** → vLLM 凭借广泛的默认地位胜出。

**智能体多轮对话（工具、规划、记忆）** → SGLang 的 RadixAttention（Phase 17 · 06）占绝对优势。

**前缀大量复用的 RAG** → SGLang。

**代码生成** → vLLM 够用；SGLang 在缓存上略好。

**长上下文（128K+）** → vLLM + 分块预填充（chunked prefill）；SGLang + 分层 KV。

### TGI 维护模式陷阱

Hugging Face TGI 于 2025 年 12 月 11 日进入维护模式——此后只修 bug。历史上它具备：顶级的可观测性、同类最佳的 HF 生态集成（模型卡、安全工具），原始吞吐量略逊于 vLLM。

对 2026 年的新项目：默认避开 TGI。已有的 TGI 部署可以继续运行，但最终应当迁移。SGLang 和 vLLM 是更稳妥的默认选择。

### 流水线模式

开发（Ollama）→ 预发（llama.cpp）→ 生产（vLLM）。全程使用同一份 GGUF 或 HF 权重。工程师在笔记本上快速迭代；预发环境复刻生产环境的量化配置；生产环境是最终的服务目标。

### Ollama 的注意事项

Ollama 非常适合开发，但不适合共享的生产环境：Go 的 HTTP 序列化带来额外开销，并发管理比 vLLM 简陋，OpenTelemetry 支持滞后。在它擅长的地方用 Ollama——单用户、一条命令——共享场景切换到 vLLM。

### 自托管 vs 托管服务是另一个独立决策

Phase 17 · 01（托管超大规模云厂商）和 · 02（推理平台）覆盖了托管方案。本课假设你已经决定自托管。自托管的理由：数据驻留、自定义微调、规模化后的总拥有成本、托管服务上没有的领域模型。

### 应该记住的数字

- TGI 维护模式：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月；PyTorch 2.10；支持 Blackwell SM120。
- SGLang 生产部署规模：400,000+ 块 GPU。
- Ollama 相对 llama.cpp 的吞吐量差距：慢 15-30%；生产负载下差 3 倍。

```figure
data-parallel
```

## 生产实践

`code/main.py` 是一个决策树遍历器：给定硬件 + 规模 + 工作负载，它会选出一个引擎并解释原因。

## 交付产物

本课产出 `outputs/skill-engine-picker.md`。给定约束条件，它会选出一个引擎并写出迁移计划。

## 练习

1. 用你自己的硬件 / 规模 / 工作负载运行 `code/main.py`。输出和你的直觉一致吗？
2. 你的基础设施有 12 块 H100 和 8 块 MI300X AMD。选什么引擎？TRT-LLM 为什么不在考虑范围内？
3. 一个团队在 2026 年还想用 TGI，理由是"我们熟悉它"。请论证迁移的必要性。
4. 从 Ollama 开发环境迁到 vLLM 生产环境：量化、配置和可观测性各有什么变化？
5. 一个 RAG 产品，P99 前缀长度 8K，且租户间复用率很高。选一个引擎，并结合 Phase 17 · 11 + 18 搭建技术栈。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| llama.cpp | "那个跑 CPU 的" | 模型支持最广，CPU 上最快 |
| Ollama | "那个跑笔记本的" | 一条命令即装，开发级吞吐量 |
| TGI | "HF 家的推理服务" | 2025 年 12 月起进入维护模式 |
| vLLM | "默认选项" | 2026 年广泛的生产基线 |
| SGLang | "那个做智能体的" | 前缀密集型，RadixAttention |
| TRT-LLM | "NVIDIA 专属" | Blackwell 吞吐量领先者，仅限 NVIDIA |
| GGUF | "llama.cpp 格式" | 内置 K-quant 量化变体 |
| Production-stack | "vLLM 的 K8s 方案" | Phase 17 · 18 参考部署 |
| 流水线模式 | "开发→预发→生产" | Ollama → llama.cpp → vLLM，同一份权重 |

## 延伸阅读

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — 发布说明。
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)
