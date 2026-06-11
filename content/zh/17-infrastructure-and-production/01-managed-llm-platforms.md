# 托管 LLM 平台 — Bedrock、Vertex AI、Azure OpenAI

> 三大云巨头，三种截然不同的战略。AWS Bedrock 是模型市场——Claude、Llama、Titan、Stability、Cohere 统一在一套 API 之后。Azure OpenAI 是与 OpenAI 的独家合作，外加预置吞吐单元（Provisioned Throughput Units，PTU）提供专用容量。Vertex AI 则以 Gemini 为先，拥有最强的长上下文与多模态能力。2026 年 Artificial Analysis 的测量显示，在 Llama 3.1 405B 同级模型上，Azure OpenAI 的中位延迟约 50 ms，Bedrock 约 75 ms——这个差距由 PTU 解释，因为专用容量天然优于共享按需容量。决策准则不是"哪家最快"，而是"哪家的模型目录和 FinOps 能力与我的产品匹配"。这节课教你把权衡明明白白写下来再做选择，而不是凭感觉。

**Type:** Learn
**Languages:** Python (stdlib, toy cost-and-latency comparator)
**Prerequisites:** Phase 11 (LLM Engineering), Phase 13 (Tools & Protocols)
**Time:** ~60 minutes

## 学习目标

- 说出三种平台战略（模型市场 vs 独家合作 vs Gemini 优先），并能将每种战略对应到合适的产品场景。
- 解释 Azure OpenAI 的预置吞吐单元（PTU）买到的是什么，以及为什么在 405B 规模上按需的 Bedrock 通常慢约 25 ms。
- 画出各平台的 FinOps 成本归因体系（Bedrock 的 Application Inference Profiles vs Vertex 的按团队分项目 vs Azure 的作用域 + PTU 预留）。
- 写出一条"至少双供应商"政策，并解释为什么在 2026 年单一供应商锁定是代价最高的错误。

## 问题背景

你为产品选定了 Claude 3.7 Sonnet。现在你需要把它跑起来。你可以直接调用 Anthropic API，可以通过 AWS Bedrock 调用，也可以走一层网关。直连 API 最简单；Bedrock 额外提供 BAA、VPC 端点、IAM 以及 CloudWatch 成本归因；网关则带来跨供应商的故障切换、统一计费和速率限制。

更深层的问题在于模型目录。如果同一个产品里既要 Claude，又要 Llama，还要 Gemini，那你无法从一个地方全部买到——除非这个"地方"是 Bedrock 加 Vertex 加 Azure OpenAI 同时使用。三大云巨头并不可互换——它们各自对"谁掌握模型层"下了不同的赌注。

这节课会梳理这三种赌注、延迟差距、FinOps 差距以及锁定风险。

## 核心概念

### 三种战略

**AWS Bedrock**——模型市场。Claude（Anthropic）、Llama（Meta）、Titan（AWS 自研）、Stability（图像）、Cohere（嵌入）、Mistral，外加图像与嵌入子目录。一套 API、一套 IAM 体系、一套 CloudWatch 导出。Bedrock 的赌注是：客户要的是可选性，而不是某一个特定模型。

**Azure OpenAI**——独家合作。你能用到 GPT-4 / 4o / 5 / o 系列、DALL·E、Whisper，以及在 Azure 数据中心内对 OpenAI 模型做微调。"Azure OpenAI Service" 目录里没有任何非 OpenAI 模型——那些归 Azure AI Foundry（一个独立产品）。Azure 的赌注是：OpenAI 会持续站在前沿，而客户希望在这层特定关系之上获得企业级管控。

**Vertex AI**——Gemini 优先，其他靠后。Gemini 1.5 / 2.0 / 2.5 的 Flash 与 Pro 系列，外加 Model Garden（第三方模型）。Vertex 的赌注是多模态长上下文——100 万 token 的 Gemini 上下文窗口就是差异化所在。

### 规模化下的延迟差距

Artificial Analysis 持续运行基准测试。在同等的 Llama 3.1 405B 部署（共享按需）上，Azure OpenAI 的首 token 延迟中位数约为 50 ms，Bedrock 约为 75 ms。这个差距不是 AWS 的失误——而是容量模型的差异。Azure 出售 PTU（Provisioned Throughput Units，预置吞吐单元），为你的租户预留 GPU 容量。Bedrock 也有对应产品（Provisioned Throughput），但起步价约为每单元每小时 21 美元，多数客户仍停留在共享按需模式。

按需共享容量要和所有其他客户的流量竞争，专用容量则不用。如果你的产品 SLA 要求 P99 的 TTFT 低于 100 ms，那你要么在 Azure 买 PTU，要么买 Bedrock Provisioned Throughput，要么接受默认的延迟波动。

### 预置吞吐的经济账

Azure PTU：一块预留的推理算力。对可预测的工作负载，相比按需最高可省约 70%。成本按小时固定计费，与流量无关——即使闲置也要为预留付费。盈亏平衡点通常在 40-60% 的持续利用率。

Bedrock Provisioned Throughput：每小时 21-50 美元，取决于模型和区域。算法类似——盈亏平衡点大约在峰值利用率的一半。需要按月承诺。

Vertex 的预置容量按 Gemini SKU 销售；价格因模型和区域而异，公开宣传较少。

### FinOps 体系——真正的差异化所在

**Bedrock 的 Application Inference Profiles** 是这个市场里最干净的成本归因方案。给一个 profile 打上 `team`、`product`、`feature` 标签，把所有模型调用都路由过去，CloudWatch 就能按 profile 拆分成本，无需任何后处理。该功能 2025 年推出，至今仍是云巨头原生方案中粒度最细的。

**Vertex** 的归因方式是按团队分项目，外加无处不在的标签。你把每个团队建成一个 GCP 项目，给每个资源都打上标签，然后用 BigQuery Billing Export + DataStudio 做汇总。工作量更大，但 BigQuery 让你可以对成本数据执行任意 SQL。

**Azure** 依赖订阅/资源组作用域加标签，并把 PTU 预留作为一等成本对象。标签从资源组继承，而非从请求继承，所以按请求级归因需要 Application Insights 自定义指标，或者一个会打标头的网关。

总体格局：Bedrock 原生方案最干净，Vertex 借 BigQuery 最灵活，Azure 在不加额外埋点的情况下最不透明。

### 锁定是 2026 年的核心风险

当一个模型一家独大时，押注单一云巨头没有问题。但 2026 年的前沿每月都在轮换——这个季度是 Claude 3.7，下个季度是 Gemini 2.5，再下个季度是 GPT-5。锁定一个平台，就等于把自己挡在三分之二的前沿模型之外。

实战团队普遍采用的模式：任何产品关键的 LLM 调用都执行"至少双供应商"。Bedrock 加 Azure OpenAI 是常见组合——一边拿 Claude，一边拿 GPT，两者之间互为故障切换，共用同一个网关。由于网关会选择最优路由，成本上浮可以忽略；而在故障期间（比如 2025 年 1 月的 Azure OpenAI 事故、AWS us-east-1 的宕机），可用性的提升是决定性的。

### 数据驻留、BAA 与受监管行业

Bedrock：多数区域可签 BAA；VPC 端点；护栏机制。金融科技的常见默认选择。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；欧盟数据驻留；受监管企业的默认选择。
Vertex：HIPAA、GDPR、按区域的数据驻留；背靠 Google Cloud 的合规体系。

三家都能满足基本的合规清单。差异在于数据保留策略、日志处理方式，以及滥用监控是否会读取你的流量（多数情况下默认开启；企业客户可申请退出）。

### 你应该记住的数字

- Azure OpenAI 在 Llama 3.1 405B 同级模型上的中位 TTFT：约 50 ms（使用 PTU）。
- Bedrock 按需模式的中位 TTFT：约 75 ms。
- Bedrock Provisioned Throughput：每单元每小时 21-50 美元。
- Azure PTU 盈亏平衡点：约 40-60% 的持续利用率。
- 高利用率下 PTU 相比按需的节省：最高 70%。

## 生产实践

`code/main.py` 在一个合成工作负载上比较三个平台——它对按需 vs PTU 的经济模型、TTFT 波动以及成本归因精度进行建模。跑一遍，看看 PTU 在什么场景下划算，以及模型市场的目录广度在什么场景下足以抵消 TTFT 差距。

## 交付产物

这节课产出 `outputs/skill-managed-platform-picker.md`。给定一个工作负载画像（所需模型、TTFT SLA、日调用量、合规要求），它会推荐主平台、备用平台，以及一套 FinOps 埋点方案。

## 练习

1. 运行 `code/main.py`。对一个 70B 级别的模型，Azure PTU 在多高的持续利用率下开始优于按需？计算盈亏平衡点，并与官方宣传的 40-60% 区间对比。
2. 你的产品同时需要 Claude 3.7 Sonnet 和 GPT-4o。设计一个双供应商部署方案——哪个模型放哪家云、前面放什么网关、故障切换策略是什么？
3. 一个受监管的医疗客户要求 BAA、US-East 数据驻留、P99 TTFT 低于 100 ms。选一个平台，并用三个具体特性来论证。
4. 你发现本月 Bedrock 账单涨了 4 倍，但流量没有变化。在没有 Application Inference Profiles 的情况下，你会怎么找出元凶？有了 profiles，又需要多久？
5. 阅读 Azure OpenAI 和 Bedrock 的定价页面。对一个每月 1 亿 token 的 Claude 工作负载，哪个更便宜——直连 Anthropic API、Bedrock 按需，还是 Bedrock Provisioned Throughput？

## 关键术语

| 术语 | 大家口中的说法 | 实际含义 |
|------|----------------|------------------------|
| Bedrock | "AWS 的 LLM 服务" | 覆盖 Claude、Llama、Titan、Mistral、Cohere 的模型市场 |
| Azure OpenAI | "Azure 版 ChatGPT" | 在 Azure 数据中心独家提供 OpenAI 模型，带企业级管控 |
| Vertex AI | "Google 的 LLM" | Gemini 优先的平台，附带容纳第三方模型的 Model Garden |
| PTU | "专用容量" | 预置吞吐单元（Provisioned Throughput Unit）——预留的推理 GPU，按小时计价 |
| Application Inference Profile | "Bedrock 打标签" | 按产品维度的成本/用量 profile，支持标签，CloudWatch 原生 |
| Model Garden | "Vertex 模型目录" | Vertex AI 的第三方模型板块，与 Gemini 分开 |
| 至少双供应商 | "LLM 冗余" | 所有关键 LLM 链路都跑在不少于 2 家云巨头上的政策 |
| BAA | "HIPAA 文书" | 商业伙伴协议（Business Associate Agreement）；处理 PHI 必需；三家均提供 |
| 滥用监控 | "盯日志的" | 供应商侧对提示词/输出的安全扫描；企业客户可申请退出 |

## 延伸阅读

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) —— 权威价目表与 Provisioned Throughput 定价。
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) —— PTU 经济模型与价目表。
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) —— Gemini 各档位与 Model Garden 附加费用。
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) —— 跨供应商的持续延迟与吞吐基准测试。
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) —— 企业级决策框架。
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) —— 成本归因机制的并排对比。
