# 毕业项目 — 构建完整的工具生态系统

> Phase 13 已经把每个组件都讲透了。本毕业项目（capstone）把它们串联成一个具备生产形态的系统：一个带有工具 + 资源 + 提示 + 任务 + UI 的 MCP 服务器、边缘的 OAuth 2.1、一个 RBAC 网关、一个多服务器客户端、一次 A2A 子代理调用、接入收集器的 OTel 链路追踪、CI 中的工具投毒检测，以及一份 AGENTS.md + SKILL.md 套件。学完之后，你能为每一个架构决策给出辩护。

**Type:** Build
**Languages:** Python (stdlib, end-to-end ecosystem harness)
**Prerequisites:** Phase 13 · 01 through 21
**Time:** ~120 minutes

## 学习目标

- 组装一个 MCP 服务器，对外暴露工具、资源、提示，以及一个带 `ui://` 应用的任务。
- 在服务器前面架设一个 OAuth 2.1 网关，强制执行 RBAC 和哈希固定（pinned hashes）。
- 编写一个多服务器客户端，用 OTel GenAI 属性做端到端链路追踪。
- 把一部分工作负载委派给 A2A 子代理；验证不透明性（opacity）得以保留。
- 用 AGENTS.md + SKILL.md 打包整套技术栈，让其他智能体也能驱动它。

## 问题背景

交付一个「研究并出报告」系统：

- 用户提问：「总结 2026 年 arXiv 上被引用最多的三篇关于智能体协议的论文。」
- 系统：通过 MCP 搜索 arXiv；通过 A2A 把论文摘要工作委派给一个专门的写作智能体；聚合结果；以 MCP Apps 的 `ui://` 资源形式渲染一份可交互报告；把每一步都记录到 OTel。

Phase 13 的所有原语都会登场。这不是玩具——Anthropic（Claude Research 产品）、OpenAI（搭载 Apps SDK 的 GPTs）以及第三方在 2026 年上线的生产级研究助理系统，正是这个形态。

## 核心概念

### 架构

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (long)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### 链路层级

```
agent.invoke_agent
 ├── llm.chat (kick off)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (opaque internals)
 ├── mcp.call -> tools/call generate_report (task-augmented)
 │    └── tasks/status polling
 │    └── tasks/result (completed, returns ui:// resource)
 └── llm.chat (final synthesis)
```

只有一个 trace id。每个 span 都带有正确的 `gen_ai.*` 属性。

### 安全态势

- OAuth 2.1 + PKCE，通过资源指示符（resource indicator）把受众（audience）固定到网关。
- 网关持有上游凭证；用户永远看不到它们。
- RBAC：`alice` 拥有 `research:read` 和 `research:write`，可以调用所有工具。`bob` 只有 `research:read`，不能调用 `generate_report`。
- 固定的描述清单（pinned description manifest）：任何工具哈希发生变化的服务器都会被剔除。
- Rule of Two 审计：没有任何工具同时组合不可信输入、敏感数据和高后果操作。

### 渲染

最终的 `generate_report` 任务返回内容块（content blocks）外加一个 `ui://report/current` 资源。客户端所在的宿主（Claude Desktop 等）在沙箱 iframe 中渲染这个交互式仪表盘。仪表盘包含排好序的论文列表、引用次数，以及一个按钮——用户点击任意论文时会触发 `host.callTool('summarize_paper', {arxiv_id})`。

### 打包

整套系统以如下形式交付：

```
research-system/
  AGENTS.md                     # project conventions
  skills/
    run-research/
      SKILL.md                  # the top-level workflow
  servers/
    research-mcp/               # the MCP server
      pyproject.toml
      src/
  agents/
    writer/                     # the A2A agent
  gateway/
    config.yaml                 # RBAC + pinned manifest
```

用户通过 `docker compose up` 部署。Claude Code、Cursor、Codex 和 opencode 的用户都可以通过调用 `run-research` 技能来驱动这个系统。

### Phase 13 各课的贡献

| 课程 | 毕业项目用到了什么 |
|--------|------------------------|
| 01-05 | 工具接口、跨供应商可移植性、并行调用、模式（schema）、静态检查 |
| 06-10 | MCP 原语、服务器、客户端、传输层、资源 + 提示 |
| 11-14 | 采样、roots + elicitation、异步任务、`ui://` 应用 |
| 15-17 | 工具投毒、OAuth 2.1、网关 + 注册表 |
| 18 | A2A 子代理委派 |
| 19 | OTel GenAI 链路追踪 |
| 20 | LLM 层的路由网关 |
| 21 | SKILL.md + AGENTS.md 打包 |

## 生产实践

`code/main.py` 把前面各课的模式缝合成一个可运行的演示。全部基于标准库、全部在进程内运行，因此你可以从头到尾通读。它跑通了「研究并出报告」场景的完整流程：与网关握手、模拟 OAuth 2.1、合并 tools/list、把 generate_report 作为任务执行、向 writer 发起 A2A 调用、返回 ui:// 资源、发出 OTel span。

值得关注的点：

- 每一跳共享同一个 trace id。
- 网关策略阻止了第二个用户执行写操作。
- 任务生命周期从 working → completed，同时返回文本和 ui:// 内容。
- A2A 调用的内部状态对编排器（orchestrator）不可见。
- AGENTS.md 和 SKILL.md 是另一个智能体复现这套工作流所需的全部文件。

## 交付产物

本课产出 `outputs/skill-ecosystem-blueprint.md`。给定一个产品需求（研究、摘要、自动化），该技能就能产出完整架构：用哪些 MCP 原语、哪些网关控制、哪些 A2A 调用、哪些遥测、哪种打包方式。

## 练习

1. 运行 `code/main.py`。留意那个唯一的 trace id 以及 span 的嵌套方式。数一数这个演示用到了 Phase 13 的多少个原语。

2. 扩展这个演示：添加第二个后端 MCP 服务器（例如 `bibliography`），并确认网关把它的工具合并进了同一个命名空间。

3. 把假的 A2A 写作智能体换成一个在子进程中运行的真实智能体。使用第 19 课的测试框架（harness）。

4. 在编排器和 LLM 之间的路由网关里加一个 PII 脱敏步骤。确认用户查询中的电子邮件地址被清洗掉了。

5. 为将要维护这个系统的队友写一份 AGENTS.md。它应该能在五分钟内读完，并且给到他们在 Cursor 或 Codex 中驱动这个毕业项目所需的全部信息。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 毕业项目（Capstone） | 「Phase 13 集成演示」 | 用到所有原语的端到端系统 |
| 研究并出报告 | 「那个场景」 | 搜索、摘要、渲染的模式 |
| 生态系统 | 「所有组件合在一起」 | 服务器 + 客户端 + 网关 + 子代理 + 遥测 + 打包 |
| 链路层级 | 「单一 trace id」 | 每一跳的 span 共享同一条链路；通过 span id 建立父子关系 |
| 网关签发的令牌 | 「传递式鉴权」 | 客户端只看到网关的令牌；网关持有上游凭证 |
| 合并命名空间 | 「所有工具放进一个扁平列表」 | 在网关处合并多服务器，冲突时加前缀 |
| 不透明边界 | 「A2A 调用隐藏内部细节」 | 子代理的推理过程对编排器不可见 |
| 三层技术栈 | 「AGENTS.md + SKILL.md + MCP」 | 项目上下文 + 工作流 + 工具 |
| 纵深防御 | 「多重安全层」 | 哈希固定、OAuth、RBAC、Rule of Two、审计日志 |
| 规范符合性矩阵 | 「我们交付的内容里哪些是规范要求的」 | 把交付物映射到 2025-11-25 规范要求的清单 |

## 延伸阅读

- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 整合后的规范参考
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 协议的发展方向
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 参考
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 链路追踪的权威约定
- [Anthropic — Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) — 生产级智能体运行时模式
