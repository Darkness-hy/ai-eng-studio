# 边缘推理 —— Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> 边缘侧的核心约束是内存带宽，而不是算力。移动端 DRAM 带宽在 50-90 GB/s，数据中心 HBM3 则超过 2-3 TB/s —— 差距达 30-50 倍。解码（decode）受内存带宽限制，因此这个差距起决定性作用。2026 年的格局分为四条路线。Apple M4/A18 Neural Engine 峰值 38 TOPS，配统一内存（无需 CPU↔NPU 拷贝）。Qualcomm Snapdragon X Elite / 8 Gen 4 的 Hexagon 达到 45 TOPS。WebGPU + WebLLM 在 M3 Max 上以约 41 tok/s 运行 Llama 3.1 8B（Q4），约为原生性能的 70-80%；GitHub 17.6k star，提供 OpenAI 兼容 API，移动端覆盖率约 70-75%。NVIDIA Jetson Orin Nano Super（8GB）能装下 Llama 3.2 3B / Phi-3；AGX Orin 通过 vLLM 以约 40 tok/s 运行 gpt-oss-20b；Jetson T4000（JetPack 7.1）性能是 AGX Orin 的 2 倍。TensorRT Edge-LLM 支持 EAGLE-3、NVFP4、分块预填充（chunked prefill）—— Bosch、ThunderSoft、MediaTek 已在 CES 2026 上展示。

**Type:** Learn
**Languages:** Python (stdlib, toy bandwidth-bound decode simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 09 (Production Quantization)
**Time:** ~60 minutes

## 学习目标

- 解释为什么移动端 LLM 推理受内存带宽限制，而算力是次要因素。
- 列举四个边缘目标平台（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson），并把每个平台对应到具体用例。
- 说出 2026 年 WebGPU 的覆盖缺口（Firefox Android 仍在追赶）以及 Safari iOS 26 的落地情况。
- 为每个目标平台选择量化格式（ANE 用 Core ML INT4 + FP16，Hexagon 用 QNN INT8/INT4，浏览器用 WebGPU Q4，Jetson Thor 用 NVFP4）。

## 问题背景

一位客户想要一个端侧聊天机器人：语音优先、默认保护隐私、离线可用。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 跑出约 55 tok/s —— 没问题。在 iPhone 16 Pro 上，同一个模型只有 3 tok/s —— 不可接受。在搭载 Snapdragon 8 Gen 3 的中端 Android 上是 7 tok/s。在 Chrome Android v121+ 的浏览器里走 WebGPU，则是 4-8 tok/s，取决于设备。

这种吞吐差异不是移植问题。它是带宽差距、量化格式、以及 NPU 能否从用户态访问这三个因素的乘积。2026 年的边缘推理是四个不同的问题，对应四套不同的解法。

## 核心概念

### 带宽才是真正的天花板

解码每生成一个 token 都要读取全部权重。一个 Q4 量化的 7B 模型是 3.5 GB。以 50 GB/s 读取 3.5 GB 需要 70 ms —— 理论上限约 14 tok/s。在 90 GB/s（高端移动 DRAM）下，上限提升到约 25 tok/s。低于这个数字时，再多的算力都无济于事。

数据中心 HBM3 以 3 TB/s 读完同样的 3.5 GB 只需 1.2 ms —— 上限是 830 tok/s。同一个模型、同一份权重，差别在于内存子系统。

### Apple Neural Engine（M4 / A18）

- 最高 38 TOPS。统一内存（CPU 与 ANE 共享同一内存池）—— 没有拷贝开销。
- 通过 Core ML + 编译后的 `.mlmodel` 模型访问，或经 PyTorch 走 Metal Performance Shaders（MPS）。
- Llama.cpp 的 Metal 后端用的是 MPS，并非直接调用 ANE；要原生使用 ANE 必须做 Core ML 转换。
- 2026 年 iOS 应用最实用的路径：Core ML，INT4 权重 + FP16 激活。

### Qualcomm Hexagon（Snapdragon X Elite / 8 Gen 4）

- 最高 45 TOPS。与 CPU 和 GPU 集成在同一 SoC 中，但内存域是独立的。
- QNN（Qualcomm Neural Network）SDK 和 AI Hub 提供从 PyTorch/ONNX 的转换。
- 聊天模板、Llama 3.2、Phi-3 都作为一等公民产物在 AI Hub 上发布。

### Intel / AMD NPU（Lunar Lake、Ryzen AI 300）

- 40-50 TOPS。软件生态落后于 Apple/Qualcomm；OpenVINO 在进步但仍属小众。
- 最适合 Windows ARM 上的 copilot 类应用；在 AMD/Intel 桌面端适合本地优先（local-first）场景。

### WebGPU + WebLLM

- 通过 WebGPU 计算着色器在浏览器中运行模型；无需安装。
- M3 Max 上 Llama 3.1 8B Q4 约 41 tok/s —— 走同一后端，约为原生性能的 70-80%。
- WebLLM 在 GitHub 上有 17.6k star；OpenAI 兼容的 JS API；Apache 2.0 协议。
- 2026 年覆盖情况：Chrome Android v121+、Safari iOS 26 正式版（GA），Firefox Android 仍在追赶。移动端总体覆盖率约 70-75%。

### NVIDIA Jetson 家族

- Orin Nano Super（8GB）：能装下 Llama 3.2 3B、Phi-3，tok/s 表现不错。
- AGX Orin：通过 vLLM 以约 40 tok/s 运行 gpt-oss-20b。
- Thor / T4000（JetPack 7.1）：性能是 AGX Orin 的 2 倍，支持 EAGLE-3 和 NVFP4。
- TensorRT Edge-LLM（2026）支持 EAGLE-3 投机解码、NVFP4 权重、分块预填充 —— 数据中心的优化手段被移植到了边缘侧。

### 各目标平台的量化选择

| 目标平台 | 格式 | 说明 |
|--------|--------|-------|
| Apple ANE | INT4 权重 + FP16 激活 | Core ML 转换路径 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub 转换器 |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | 使用 `mlc_llm convert_weight` + 编译后的 `.wasm`；不支持 GGUF |
| Jetson Orin Nano | Q4 GGUF 或 TRT-LLM INT4 | 受内存带宽限制 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 路径 |

### 边缘侧的长上下文陷阱

Llama 3.1 的 128K 上下文是数据中心特性。在一部 8 GB 内存的手机上，4 GB 模型 + 32K token 所需的 2 GB KV 缓存 + 操作系统开销 = OOM（内存溢出）。除非接受激进的 KV 量化（Q4 KV），边缘部署的上下文要控制在 4K-8K。

### 语音是杀手级应用

语音智能体对延迟敏感（首 token 需小于 500 ms）。本地推理彻底消除了网络延迟。再搭配语音转文字（Whisper Turbo 系列变体可在边缘运行），边缘推理就构成了生产级质量的语音闭环。

### 该记住的数字

- Apple M4 / A18 ANE：38 TOPS。
- Qualcomm Hexagon SD X Elite：45 TOPS。
- WebLLM 在 M3 Max 上：Llama 3.1 8B Q4 约 41 tok/s。
- AGX Orin：gpt-oss-20b 通过 vLLM 约 40 tok/s。
- 数据中心与边缘的带宽差距：30-50 倍。
- WebGPU 移动端覆盖率：约 70-75%（Firefox Android 落后）。

## 生产实践

`code/main.py` 用带宽受限的数学模型计算各边缘平台的理论解码吞吐上限。与实测基准对比，并指出瓶颈在带宽而非算力的位置。

## 交付产物

本课产出 `outputs/skill-edge-target-picker.md`。给定平台（iOS/Android/浏览器/Jetson）、模型以及延迟/内存预算，选出量化格式和转换流水线。

## 练习

1. 运行 `code/main.py`。对 Snapdragon 8 Gen 3（带宽约 77 GB/s）上的 Q4 量化 7B 模型，计算解码上限。与实测的 6-8 tok/s 对比 —— 这个运行时高效吗？
2. Android 上的 WebGPU 要求 Chrome v121+。为旧版浏览器设计一套回退方案 —— 通过同一个 OpenAI 兼容 API 走服务端。
3. 你的 iOS 应用需要 4K 上下文的流式输出。哪种模型/格式组合能让你在 iPhone 16 上把活跃内存控制在 4 GB 以内？
4. Jetson AGX Orin 以 40 tok/s 运行 gpt-oss-20b，Jetson Nano 只装得下 3B 模型。如果你的产品要同时支持两者，如何统一推理栈？
5. 论证「WebLLM 在 2026 年是否生产可用」。引用覆盖率、性能数据，以及 Firefox Android 的缺口。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| ANE | 「苹果神经引擎」 | M 系列和 A 系列中的端侧 NPU；统一内存 |
| Hexagon | 「高通 NPU」 | Snapdragon NPU；通过 QNN SDK 访问 |
| WebGPU | 「浏览器 GPU」 | W3C 标准化的浏览器 GPU API；2026 年 Chrome/Safari 支持 |
| WebLLM | 「浏览器 LLM 运行时」 | MLC-LLM 项目；Apache 2.0；OpenAI 兼容的 JS |
| Jetson | 「NVIDIA 边缘设备」 | Orin Nano / AGX / Thor / T4000 家族 |
| TRT Edge-LLM | 「边缘版 TensorRT」 | TensorRT-LLM 的 2026 边缘移植版；EAGLE-3 + NVFP4 |
| 统一内存 | 「共享内存池」 | CPU 和 NPU 看到同一块 RAM；无拷贝开销 |
| 带宽受限 | 「受内存限制」 | 解码受每秒读取权重的字节数制约 |
| Core ML | 「苹果转换工具」 | Apple 用于生成 ANE 原生模型的框架 |
| QNN | 「高通技术栈」 | Qualcomm Neural Network SDK |

## 延伸阅读

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) —— 全景与基准测试。
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) —— Orin / AGX / Thor。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) —— 2026 边缘移植版发布公告。
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) —— 设计与基准测试。
- [Apple Core ML](https://developer.apple.com/documentation/coreml) —— ANE 原生转换。
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) —— 面向 Hexagon 的预转换模型。
