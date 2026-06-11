# 毕业项目 13 —— 带注册中心与治理的 MCP 服务器

> Model Context Protocol（MCP）在 2026 年不再是"未来"，而是成了工具调用的默认规范。Anthropic、OpenAI、Google 以及所有主流 IDE 都内置了 MCP 客户端。Pinterest 公开了其内部的 MCP 服务器生态。AAIF Registry 以 `.well-known` 的形式将能力元数据规范化。AWS ECS 发布了无状态部署的参考方案。Block 的 goose-agent 把同一套协议装进了托管助手。2026 年的生产形态是：StreamableHTTP 传输、OAuth 2.1 作用域、OPA 策略门控，再加上一个让平台团队能够发现、校验、启用服务器的注册中心。本项目要把这一切端到端地构建出来。

**Type:** Capstone
**Languages:** Python (server, via FastMCP) or TypeScript (@modelcontextprotocol/sdk), Go (registry service)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools and MCP), Phase 14 (agents), Phase 17 (infrastructure), Phase 18 (safety)
**涉及阶段：** P11 · P13 · P14 · P17 · P18
**Time:** 25 hours

## 问题背景

MCP 已经成为工具调用的通用语言。Claude Code、Cursor 3、Amp、OpenCode、Gemini CLI 以及所有托管智能体如今都在消费 MCP 服务器。生产环境的难点不在于编写服务器（FastMCP 让这件事变得很容易），而在于按企业级要求大规模部署：按租户划分的 OAuth 作用域、对破坏性工具施加 OPA 策略、StreamableHTTP 无状态扩展、用于发现的注册中心、每次工具调用的审计日志。Pinterest 的内部 MCP 生态和 AAIF Registry 规范定下了 2026 年的标杆。

你将构建一个暴露 10 个内部工具的 MCP 服务器（Postgres 只读查询、S3 列表、Jira、Linear、Datadog 等）、一个供平台团队发现工具的注册中心 UI，以及针对破坏性工具的人工审批关卡。负载测试要演示 StreamableHTTP 的水平扩展能力。审计链路要能通过企业安全评审。

## 核心概念

MCP 2026 修订版规定 StreamableHTTP 为默认传输方式。与早期 stdio 加 SSE 的形态不同，StreamableHTTP 默认是无状态的：单个 HTTP 端点接收 JSON-RPC 请求、流式返回响应，并支持用于通知的长连接。无状态意味着可以放在负载均衡器后面水平扩展。

授权采用带有按工具作用域（scope）的 OAuth 2.1。一个令牌携带诸如 `jira:read`、`s3:list`、`postgres:query:readonly` 这样的作用域。MCP 服务器在每次工具调用时检查作用域，而不是只在会话开始时检查。对于高风险工具，服务器会拒绝任何作用域未在最近 N 分钟内被提升为 `approved:by:human` 的调用——这一提升来自一张 Slack 审批卡片。

注册中心是一个独立服务。每个 MCP 服务器都暴露一份 `.well-known/mcp-capabilities` 文档，包含其工具清单、传输 URL 和认证要求。注册中心负责轮询、校验和建立索引。平台团队通过注册中心 UI 查看有哪些工具可用、它们需要哪些作用域、归哪个团队所有。

## 架构

```
MCP client (Claude Code, Cursor 3, ...)
          |
          v
StreamableHTTP over HTTPS (JSON-RPC + streaming)
          |
          v
MCP server (FastMCP) behind load balancer
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 listing  Jira       Linear     Datadog
(read-only) (paged)     (read)     (read)     (query)
          |
   +------+-------------+
   v                    v
 OPA policy gate   destructive tool MCP (separate server)
                        |
                        v
                   human approval via Slack
                        |
                        v
                   audit log (append-only, per-tenant)

  registry service
     |
     v  GET /.well-known/mcp-capabilities from each server
     v
     UI: search / validate / enable-disable / ownership
```

## 技术栈

- 服务器框架：FastMCP（Python）或 `@modelcontextprotocol/sdk`（TypeScript）
- 传输：StreamableHTTP over HTTPS（无状态）
- 认证：OAuth 2.1，通过 SPIFFE / SPIRE 提供工作负载身份（workload identity）
- 策略：每个工具一套 OPA / Rego 规则；每个请求都经过策略决策服务
- 注册中心：自托管，消费各服务器的 `.well-known/mcp-capabilities` 清单
- 人工审批：针对破坏性工具的 Slack 交互式消息
- 部署：AWS ECS Fargate 或 Fly.io，每租户一台服务器，或共享部署加租户级作用域隔离
- 审计：按租户分桶的结构化 JSONL，记录每次调用的完整链路

## 从零实现

1. **工具面。** 暴露 10 个内部工具：Postgres 只读查询、S3 对象列表、Jira 搜索/获取、Linear 搜索/获取、Datadog 指标查询、PagerDuty 值班查询、GitHub 只读、Notion 搜索、Slack 搜索、Salesforce 只读。每个工具都有类型化的 schema 和一个作用域标签。

2. **FastMCP 服务器。** 挂载这些工具。配置 StreamableHTTP 传输。添加一个用于 OAuth 令牌内省（introspection）和作用域强制检查的中间件。

3. **OPA 策略。** 为每个工具编写 Rego 策略：哪些作用域允许调用、应用何种 PII 脱敏、施加多大的载荷大小上限。每次工具调用都要经过策略决策服务。

4. **注册中心服务。** 独立的 Go 或 TS 服务，轮询已注册服务器的 `.well-known/mcp-capabilities`，用 JSON Schema 校验，并提供列表 / 搜索 / 校验 / 启用-禁用的 UI。

5. **能力清单。** 每个服务器都暴露 `.well-known/mcp-capabilities`，内容包括：工具列表、认证要求、传输 URL、所属团队、SLO。

6. **破坏性工具隔离。** 会修改状态的工具（Jira 创建、Linear 创建、Postgres 写入）部署在第二台 MCP 服务器上，采用更严格的认证流程：令牌必须带有 `approved:by:human` 作用域，且该作用域需在 15 分钟内通过 Slack 卡片完成提升。

7. **审计日志。** 按租户追加写入（append-only）的 JSONL：`{timestamp, user, tool, args_redacted, response_redacted, outcome}`。写入前用 Presidio 做 PII 脱敏。

8. **负载测试。** 100 个并发客户端打在 StreamableHTTP 上。通过增加第二个副本演示水平扩展；展示负载均衡器在无会话粘性（session stickiness）的情况下重新分配流量。

9. **一致性测试。** 对两台服务器运行官方 MCP 一致性测试套件。通过所有必选章节。

## 生产实践

```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## 交付产物

`outputs/skill-mcp-server.md` 描述了最终交付物：一套面向内部工具的生产级 MCP 服务器 + 注册中心 + 审计层，带有 OAuth 2.1 作用域和 OPA 门控。

| 权重 | 评分项 | 衡量方式 |
|:-:|---|---|
| 25 | 规范一致性 | StreamableHTTP + 能力清单通过 MCP 一致性测试 |
| 20 | 安全性 | 作用域强制检查、OPA 覆盖每一个工具、密钥管理规范 |
| 20 | 可观测性 | 带 PII 脱敏的逐次工具调用审计日志 |
| 20 | 扩展性 | 100 客户端负载测试下的水平扩展演示 |
| 15 | 注册中心体验 | 发现 / 校验 / 启用-禁用工作流 |
| **100** | | |

## 练习

1. 新增一个工具（Confluence 搜索）。在不动核心服务器的前提下，让它走完注册中心的校验流程上线。

2. 编写一条 OPA 策略，对包含名为 `email`、`ssn` 或 `phone` 列的 Postgres 查询结果进行脱敏。用一条探测查询验证它。

3. 在本地延迟上对比 StreamableHTTP 与 stdio 的基准性能。报告每次调用的 p50/p95。

4. 实现按租户配额：每个租户对每个工具每分钟最多 N 次调用。通过第二条 OPA 规则强制执行。

5. 运行 [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) 提供的 MCP 一致性测试套件，并修复所有失败项。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| StreamableHTTP | "2026 MCP 传输" | 无状态 HTTP + 流式传输；在网络化服务器场景中取代 SSE + stdio |
| 能力清单 | "Well-known 文档" | `.well-known/mcp-capabilities`，包含工具列表、认证要求、传输 URL |
| OPA / Rego | "策略引擎" | Open Policy Agent，依据外部规则对工具调用进行授权 |
| 作用域提升 | "人工批准" | 通过 Slack 审批授予的短时作用域，破坏性工具调用必须持有 |
| 注册中心 | "工具发现" | 根据能力清单为 MCP 服务器建立索引的服务 |
| 工作负载身份 | "SPIFFE / SPIRE" | 用于签发 OAuth 令牌的服务级密码学身份 |
| 一致性测试套件 | "规范测试" | 官方 MCP 测试集，验证 StreamableHTTP + 工具清单的正确性 |

## 延伸阅读

- [Model Context Protocol 2026 Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) —— StreamableHTTP、能力元数据、注册中心
- [AAIF MCP Registry spec](https://github.com/modelcontextprotocol/registry) —— 2026 年注册中心规范
- [AWS ECS reference deployment](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) —— 生产部署参考方案
- [Pinterest internal MCP ecosystem](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) —— 内部部署的参考案例
- [Block `goose` MCP usage](https://block.github.io/goose/) —— 智能体消费 MCP 的参考模式
- [FastMCP](https://github.com/jlowin/fastmcp) —— Python 服务器框架
- [Open Policy Agent](https://www.openpolicyagent.org/) —— 策略引擎参考
- [SPIFFE / SPIRE](https://spiffe.io) —— 工作负载身份参考
