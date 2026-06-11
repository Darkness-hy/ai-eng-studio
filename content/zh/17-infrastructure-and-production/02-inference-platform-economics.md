# 推理平台经济学 — Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026 年的推理（inference）市场早已不是单纯的 GPU 时长租赁，而是分化成三条赛道：定制芯片（Groq、Cerebras、SambaNova）、GPU 平台（Baseten、Together、Fireworks、Modal），以及 API 优先的市场型平台（Replicate、DeepInfra）。Fireworks 自 2026 年 5 月 1 日起把每块 GPU 的租赁价格上调了 $1/小时，其 $40 亿估值和每天 10 万亿以上 token 的处理量说明：靠规模驱动的商业模式行得通。Baseten 于 2026 年 1 月以 $50 亿估值完成 $3 亿 Series E 融资。各家的竞争定位规则很简单：Fireworks 主打延迟、Together 主打模型目录广度、Baseten 主打企业级成熟度、Modal 主打 Python 原生开发体验、Replicate 主打多模态覆盖、Anyscale 主打分布式 Python。这节课会给你一张可以直接递给创业者的对比矩阵。

**Type:** Learn
**Languages:** Python (stdlib, toy per-call economics comparator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~60 minutes

## 学习目标

- 说出三个市场细分（定制芯片、GPU 平台、API 优先）的名称，并把每家厂商归入对应细分。
- 解释为什么「按 token 计费」的 API 定价会向推理引擎的成本曲线收敛，而不是向硬件的成本曲线收敛。
- 至少在三家厂商之间计算单次请求的有效成本，并解释按分钟计费（Baseten、Modal）何时优于按 token 计费。
- 针对给定的工作负载（serverless 突发型、稳定高吞吐、微调变体、多模态），判断哪个平台是合适的默认选择。

## 问题背景

你已经评估过托管的超大规模云平台，并决定需要一家更专注、更快的服务商——要低延迟选 Fireworks，要模型广度选 Together，要部署微调后的自定义模型选 Baseten。现在摆在面前的是六个真实选项，而它们的定价页根本对不上号：Fireworks 标的是 $/百万 token；Baseten 标的是 $/分钟；Modal 标的是 $/秒；Replicate 标的是 $/次预测。不对工作负载建模，你根本没法把它们放在一起比较。

更麻烦的是，每张定价页背后的商业模式都不一样。Fireworks 在共享 GPU 上跑自家的定制引擎（FireAttention），按 token 的费率反映的是他们的利用率曲线；Baseten 给你的是 Truss 加专属 GPU，按分钟计费反映的是独占性；Modal 是真正的 Python 无服务器（serverless）——按秒计费，冷启动不到一秒。同样的产出（一条 LLM 响应），背后是三种不同的成本函数。

这节课会对这六家平台建模，并告诉你每一家在什么场景下胜出。

## 核心概念

### 三个市场细分

**定制芯片** — Groq（LPU）、Cerebras（WSE）、SambaNova（RDU）。在同一模型上，解码速度通常比基于 GPU 的集群快 5-10 倍。按 token 的单价更高（2025 年末 Groq 在 Llama-70B 上约为 $0.99/百万 token），但在延迟敏感的场景下无可匹敌。Groq 是语音智能体和实时翻译的生产首选。

**GPU 平台** — Baseten、Together、Fireworks、Modal、Anyscale。运行在 NVIDIA（2026 年的 H100、H200、B200）或有时是 AMD 的硬件上。它们是介于「裸 GPU 租赁」（RunPod、Lambda）和「超大规模云托管服务」（Bedrock）之间的经济层。

**API 优先的市场型平台** — Replicate、DeepInfra、OpenRouter、Fal。模型目录广泛，按次预测或按秒付费，强调「最快发出第一次调用」。

### Fireworks — 延迟优化型 GPU 平台

- FireAttention 引擎（自研）；宣传在同等配置下延迟比 vLLM 低 4 倍。
- 面向非交互式工作负载的批处理档位（batch tier），价格约为 serverless 费率的 50%。
- 微调模型按基础模型同等费率提供服务——相比那些对你的 LoRA 收取溢价的服务商，这是实打实的差异化优势。
- 2026 年年中：自 2026 年 5 月 1 日起，按需 GPU 租赁价格上调 $1/小时。规模化用量可议价。
- 财务信号：估值 $40 亿，每天处理 10 万亿以上 token。

### Together — 广度优化型

- 200 多个模型，开源新模型通常在上游发布后几天内上线。
- 在同等 LLM 模型上比 Replicate 便宜 50-70%——「AI Native Cloud」的定位靠的就是走量和目录广度。
- 推理 + 微调 + 训练统一在一个 API 里。

### Baseten — 企业级成熟度优化型

- Truss 框架：把依赖、密钥、服务配置打包进同一份模型清单（manifest）。
- GPU 覆盖从 T4 到 B200。按分钟计费，冷启动缓解措施做得不错。
- SOC 2 Type II 认证、HIPAA 就绪。金融科技和医疗行业的常见选择。
- 估值 $50 亿，2026 年 1 月 Series E（CapitalG、IVP、NVIDIA 投资 $3 亿）。

### Modal — Python 原生优化型

- 用纯 Python 写基础设施即代码。给函数加上 `@modal.function(gpu="A100")` 装饰器，一条命令即可部署。
- 按秒计费。配合预热（pre-warming）冷启动为 2-4 秒；小模型不到 1 秒。
- Series B 融资 $8700 万，估值 $11 亿（2025 年）。在独立调研中开发者体验得分最高。

### Replicate — 多模态广度

- 按次预测付费。图像、视频和音频模型的默认平台。
- 集成生态丰富（Zapier、Vercel、CMS 插件）。
- 在 LLM 的按 token 费率上竞争力较弱，但靠多模态品类多样性取胜。

### Anyscale — Ray 原生

- 构建在 Ray 之上；RayTurbo 是 Anyscale 的专有推理引擎（与 vLLM 竞争）。
- 最适合分布式 Python 工作负载，即推理步骤只是更大计算图中的一个节点的场景。
- 托管 Ray 集群；与 Ray AIR 和 Ray Serve 深度集成。

### 按 token 与按分钟计费——各自何时胜出

当工作负载对延迟不敏感且流量突发时，按 token 计费更划算——你只为实际用量付钱。当利用率高且可预测时，按分钟计费更划算——一旦你把 GPU 打满，它就能反超按 token 的价格。

经验法则：当工作负载对一块专属 GPU 的持续利用率超过约 30% 时，按分钟计费（Baseten、Modal）开始优于按 token 计费（Fireworks、Together）。低于这个水平则按 token 更划算，因为你不用为闲置时间买单。

### 真正的护城河是自研引擎

vLLM 和 SGLang 之上的每家平台都宣称有自研引擎：FireAttention、RayTurbo、Baseten 的推理栈。自研引擎的宣传带有营销色彩——诚实的说法是，vLLM 加 SGLang 大约占了生产环境开源推理的 80%，平台层真正的差异化在于开发体验（DX）、成本归因和 SLA。

### 你应该记住的几个数字

- Fireworks GPU 租赁：自 2026 年 5 月 1 日起涨价 $1/小时。
- Fireworks 宣称：同等配置下延迟比 vLLM 低 4 倍。
- Together：在 LLM 上比 Replicate 便宜 50-70%。
- Baseten 估值：$50 亿（Series E，2026 年 1 月，$3 亿融资）。
- Modal 估值：$11 亿（Series B，2025 年）。
- 持续利用率超过约 30% 时，按分钟计费优于按 token 计费。

```figure
cost-per-token
```

## 生产实践

`code/main.py` 在一个合成工作负载上跨定价模型比较这六家厂商，报告每日成本（$/天）和有效的每百万 token 成本（$/M tokens）。运行它，找出按 token 与按分钟计费之间的盈亏平衡点。

## 交付产物

这节课产出 `outputs/skill-inference-platform-picker.md`。给定工作负载画像、SLA 和预算，它会选出首选推理平台，并给出备选方案。

## 练习

1. 运行 `code/main.py`。对于在一块 H100 上跑 70B 模型的场景，持续利用率达到多少时 Baseten（按分钟）开始优于 Fireworks（按 token）？自己推导出交叉点，并与经验法则对比。
2. 你的产品同时提供图像生成、聊天和语音转文字。为每种模态选择平台，并说出能把它们统一起来的网关模式（gateway pattern）的名称。
3. Fireworks 把你主力模型的价格上调 $1/小时。假设 40% 的流量迁移到批处理档位（五折），对混合成本的影响进行建模。
4. 一个受监管的客户要求 SOC 2 Type II + HIPAA + 专属 GPU。哪三个平台可行？哪一个在 FinOps 上胜出？
5. 比较 Llama 3.1 70B 在 Fireworks serverless、Together 按需、Baseten 专属和 Replicate API 上每 1,000 次预测的成本。每天 10 次预测时哪家最便宜？每天 10,000 次时呢？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 定制芯片 | 「非 GPU 芯片」 | Groq LPU、Cerebras WSE、SambaNova RDU——针对解码优化 |
| FireAttention | 「Fireworks 引擎」 | 自研注意力内核；宣传延迟比 vLLM 低 4 倍 |
| Truss | 「Baseten 的格式」 | 模型打包清单；依赖 + 密钥 + 服务配置 |
| 按 token 计费 | 「API 定价」 | 按消耗的 token 收费；不为闲置付钱 |
| 按分钟计费 | 「专属定价」 | 按 GPU 实际占用时长收费；在高利用率下胜出 |
| 按次预测计费 | 「Replicate 定价」 | 按模型调用次数收费；图像/视频场景常见 |
| RayTurbo | 「Anyscale 引擎」 | 基于 Ray 的专有推理引擎；在 Ray 集群上与 vLLM 竞争 |
| 批处理档位 | 「五折」 | 费率打折的非交互式队列；Fireworks、OpenAI 上常见 |
| 微调按基础费率计费 | 「Fireworks LoRA」 | LoRA 服务的请求按基础模型费率收费（差异化优势） |

## 延伸阅读

- [Fireworks Pricing](https://fireworks.ai/pricing) — 按 token 费率、批处理档位、GPU 租赁。
- [Baseten Pricing](https://www.baseten.co/pricing/) — 按分钟费率、承诺容量、企业档位。
- [Modal Pricing](https://modal.com/pricing) — 按秒 GPU 费率和免费档位。
- [Together AI Pricing](https://www.together.ai/pricing) — 模型目录和按 token 费率。
- [Anyscale Pricing](https://www.anyscale.com/pricing) — RayTurbo 与托管 Ray 定价。
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 对比评估。
- [Infrabase — AI Inference API Providers 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — 厂商格局概览。
