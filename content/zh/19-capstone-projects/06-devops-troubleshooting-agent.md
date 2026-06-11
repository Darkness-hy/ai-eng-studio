# Capstone 06 — 面向 Kubernetes 的 DevOps 故障排查智能体

> AWS 的 DevOps Agent 已正式发布（GA），Resolve AI 公开了其 K8s 排障手册，NeuBird 演示了语义监控，Metoro 则把 AI SRE 与各服务的 SLO 绑定。生产形态已经尘埃落定：告警 webhook 触发，智能体读取遥测数据，遍历 K8s 对象构成的图，对根因假设排序，然后向 Slack 发送带审批按钮的简报。默认只读。每一次修复操作都由人工把关。本毕业设计要做的就是这样一个智能体，在 20 个合成事故上评估，并与 AWS 的 Agent 在三个共享案例上对比。

**Type:** Capstone
**Languages:** Python (agent), TypeScript (Slack integration)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools and MCP), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P11 · P13 · P14 · P15 · P17 · P18
**Time:** 30 hours

## 问题背景

2025-2026 年 SRE 圈的主流叙事变成了："AI 智能体负责事故分诊，人类负责审批修复。" AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 在生产环境中交付的全是这个形态。智能体读取 Prometheus 指标、Loki 日志、Tempo 链路追踪、kube-state-metrics，以及一张 K8s 对象的知识图谱。它在五分钟内给出一份附带遥测引用的根因假设排序。未经人工通过 Slack 明确批准，它绝不执行破坏性命令。

最难的部分是权限收敛和安全，而不是推理。智能体需要一个默认只读的 RBAC 面、一台经过加固的 MCP 工具服务器，以及记录每条"考虑过 vs 实际执行"命令的审计日志。它需要知道何时超出了自己的能力范围并升级处理。而且它的运行成本必须够低，不能让一次 OOM-kill 级联产生 5000 美元的智能体账单。

## 核心概念

智能体在一张知识图谱上工作。节点是 K8s 对象（Pod、Deployment、Service、Node、HPA、PVC）加上遥测源（Prometheus 序列、Loki 流、Tempo 链路）。边编码归属关系（Pod -> ReplicaSet -> Deployment）、调度关系（Pod -> Node）和观测关系（Pod -> Prometheus 序列）。这张图通过 kube-state-metrics 同步保持新鲜，并在每次告警时重新采样。

告警触发后，智能体从受影响的对象出发做根因分析。它沿边遍历，拉取相关的遥测切片（最近 15 分钟），起草一份假设。假设按证据排序：有多少条遥测引用支持它、有多新、有多具体。排名前 3 的假设连同图路径可视化和修复操作的审批按钮一起发送到 Slack。

修复操作受门控。默认允许的操作全部是只读的。破坏性操作（缩容、回滚、删除 Pod）需要 Slack 审批；ArgoCD 回滚钩子需要一个智能体永远不持有的认证 token。审计日志记录智能体*考虑过*的每条命令——而不仅是执行过的——这样复盘流程才能抓住险些发生的事故。

## 架构

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## 技术栈

- 可观测性数据源：Prometheus、Loki、Tempo、kube-state-metrics
- 知识图谱：用 Neo4j（托管版）或 kuzu（嵌入式）存储 K8s 对象 + 遥测边
- 智能体：LangGraph，按工具粒度配置允许列表，默认只读
- 工具传输层：FastMCP over StreamableHTTP；破坏性工具放在审批门控之后的独立服务器上
- 模型：Claude Sonnet 4.7 负责根因推理，Gemini 2.5 Flash 负责日志摘要
- 修复手段：ArgoCD 回滚 webhook、PagerDuty 升级、Slack 审批卡片
- 审计：仅追加的结构化日志（考虑过、已执行、已批准、结果）
- 部署：K8s deployment，配备专属的窄权限 RBAC 角色；独立命名空间

## 从零实现

1. **图谱摄取。** 每 30 秒将 kube-state-metrics 同步到 Neo4j/kuzu。节点：Pod、Deployment、Node、Service、PVC、HPA。边：OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。遥测覆盖层的边：OBSERVED_BY（一个 Pod 被某条 Prometheus 序列观测）。

2. **告警接收器。** 一个接受 PagerDuty 或 Alertmanager webhook 的 FastAPI 端点。从中提取受影响的对象和被突破的 SLO。

3. **只读工具面。** 通过 FastMCP 封装 kubectl、Prometheus 查询、Loki logql、Tempo traceql。每个工具只有窄范围的 RBAC 动词（"get"、"list"、"describe"）。默认服务器上不存在 "delete"、"exec"、"scale"。

4. **根因分析智能体。** LangGraph，三个节点：`sample` 拉取最近 15 分钟的遥测切片，`walk` 查询图谱获取相邻对象，`hypothesize` 起草带遥测引用的根因候选并排序。

5. **证据打分。** 每个假设的分数 = 新近度 * 具体度 * 图路径长度倒数 * 引用数量。返回前 3 名。

6. **Slack 简报。** 发送一条附件消息，包含假设、图路径可视化（服务端渲染的子图图片），以及最多对应一个修复操作的审批按钮。

7. **修复门控。** 破坏性工具（缩容、回滚、删除）放在第二台 MCP 服务器上，由审批 token 守护。智能体只有在 Slack 卡片被人工批准后才能调用它们。

8. **审计日志。** 仅追加的 JSONL：对每条候选命令，记录是否被考虑过、是否被执行、由谁批准。每天发送到 S3。

9. **合成事故套件。** 构建 20 个场景：OOMKill 级联、DNS 抖动、HPA 震荡、PVC 写满、吵闹邻居、有问题的 sidecar、错误的 ConfigMap 发布、证书轮换、镜像拉取退避等。按根因准确率和出假设耗时给智能体打分。

## 生产实践

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## 交付产物

交付物是 `outputs/skill-devops-agent.md`。给定一个 K8s 集群和告警源，智能体能产出排序后的根因假设，以及一条由 Slack 门控的修复流程。

| 权重 | 评估项 | 衡量方式 |
|:-:|---|---|
| 25 | 场景套件上的 RCA 准确率 | 在 20 个合成事故中根因判断正确率 ≥80% |
| 20 | 安全性 | 审计日志中破坏性操作防护从未在没有 Slack 审批的情况下放行 |
| 20 | 出假设耗时 | 从告警到 Slack 简报的 p50 低于 5 分钟 |
| 20 | 可解释性 | 每个假设都附有图路径和遥测引用 |
| 15 | 集成完整度 | PagerDuty、Slack、ArgoCD、Prometheus 端到端可用 |
| **100** | | |

## 练习

1. 把你的智能体跑在 AWS DevOps Agent 演示过的同样三个事故上。公布并排对比结果。报告智能体在哪些地方出现分歧。

2. 增加一项"险情（near-miss）"审计：标记出智能体*考虑过*的、若无审批就会造成破坏的任何命令。统计一周内的险情发生率。

3. 把假设生成模型从 Claude Sonnet 4.7 换成自托管的 Llama 3.3 70B。测量 RCA 准确率变化和单次事故的美元成本。

4. 构建一个因果过滤器：区分相关性遥测尖峰和真正的根因。用 20 个场景的标签训练一个小型分类器。

5. 增加回滚演练（dry-run）：在配置相同 manifest 的 staging 集群上执行 ArgoCD 回滚。在 Slack 审批按钮之前，先在真实集群中验证回滚计划。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| K8s 知识图谱 | "集群图" | 节点 = K8s 对象 + 遥测序列；边 = 归属、调度、观测 |
| 默认只读 | "收敛的 RBAC" | 智能体的 service account 只有 get/list/describe 动词；破坏性动词放在审批门控后的独立服务器上 |
| 审计日志 | "考虑过 vs 已执行" | 仅追加的记录：每条候选命令、是否运行、由谁批准 |
| 假设排序 | "证据分" | 新近度 × 具体度 × 图路径长度倒数 × 引用数量 |
| Slack 审批卡片 | "HITL 门控" | 带修复按钮的交互式 Slack 消息；人工点击之前智能体无法继续 |
| 遥测引用 | "证据指针" | 支撑某个论断的 Prometheus 查询、Loki 选择器或 Tempo 链路 URL |
| MTTR | "解决耗时" | 从告警触发到 SLO 恢复的真实时钟时间 |

## 延伸阅读

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026 年的权威参考
- [Resolve AI K8s troubleshooting](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 竞品参考
- [NeuBird semantic monitoring](https://www.neubird.ai) — 语义图谱路线
- [Metoro AI SRE](https://metoro.io) — SLO 优先的生产视角
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — 集群状态数据源
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 参考的智能体编排器
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP 服务器框架
- [ArgoCD rollback](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — 受门控的修复目标
