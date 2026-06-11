# MCP 网关与注册中心 —— 企业级控制平面

> 企业不可能允许每个开发者随意安装 MCP 服务器。网关（gateway）把认证、RBAC、审计、限流、缓存和工具投毒检测集中起来，再把合并后的工具面以单一 MCP 端点的形式对外暴露。Official MCP Registry（由 Anthropic、GitHub、PulseMCP、Microsoft 共同管理，经命名空间验证）是规范的上游来源。本课讲清网关在架构中的位置，逐步实现一个最小网关，并梳理 2026 年的厂商格局。

**Type:** Learn
**Languages:** Python (stdlib, minimal gateway)
**Prerequisites:** Phase 13 · 15 (tool poisoning), Phase 13 · 16 (OAuth 2.1)
**Time:** ~45 minutes

## 学习目标

- 说明 MCP 网关所处的位置（位于 MCP 客户端与多个后端 MCP 服务器之间）。
- 实现网关的五项职责：认证、RBAC、审计、限流、策略。
- 在网关层强制执行固定工具哈希（pinned-tool-hash）清单。
- 区分 Official MCP Registry 与元注册中心（Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## 问题背景

一家世界 500 强企业有 30 个经过审批的 MCP 服务器、5000 名开发者、合规与审计要求，以及一个想要集中管控策略的安全团队。让每个开发者在自己的 IDE 里随意安装服务器，根本行不通。

网关模式：

1. 网关以单个 Streamable HTTP 端点的形式运行，开发者统一连接它。
2. 网关持有每个后端 MCP 服务器的凭据。
3. 每个开发者请求都通过网关自身的 OAuth 完成认证和权限范围限定。
4. 网关将调用路由到后端服务器，并施加策略。
5. 所有调用都被记录以供审计。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway —— 这些厂商都在 2025-2026 年间推出了网关产品或网关功能。

与此同时，Official MCP Registry 作为规范上游正式上线：经过策展、命名空间验证、采用反向 DNS 命名的服务器，网关可以直接从中拉取。元注册中心（Glama、MCPMarket、MCP.so、Smithery、LobeHub）则聚合来自多个来源的服务器。

## 核心概念

### 网关的五项职责

1. **认证（Auth）。** 通过 OAuth 2.1 识别开发者身份；映射到用户角色。
2. **RBAC。** 按用户的策略：可访问哪些服务器、哪些工具、哪些权限范围。
3. **审计（Audit）。** 每次调用都记录谁、做了什么、何时、结果如何。
4. **限流（Rate limit）。** 按用户 / 按工具 / 按服务器设置上限，防止滥用。
5. **策略（Policy）。** 拒绝被投毒的描述、强制执行 Rule of Two、脱敏 PII。

### 网关作为单一端点

在开发者眼中，网关就是一个 MCP 服务器。其内部则路由到 N 个后端。会话 id（Phase 13 · 09）在边界处被重写。

### 凭据保管（Credential vaulting）

开发者永远看不到后端令牌。网关持有这些令牌（或代理到一个持有它们的身份提供方）。在网关上拥有 `notes:read` 权限的开发者，可以凭网关自身的后端凭据间接访问 notes MCP 服务器 —— 但仅限于约束这种间接访问的策略之内。

### 在网关层固定工具哈希

网关持有一份经过审批的工具描述清单（SHA256 哈希）。在发现阶段，它获取每个后端的 `tools/list`，将哈希与清单比对，并剔除任何描述发生变化的工具。这就是 Phase 13 · 15 中的「抽地毯」（rug-pull）防御在中心化场景下的应用。

### 策略即代码（Policy-as-code）

高级网关用 OPA/Rego、Kyverno 或 Styra 来表达策略。诸如「用户 `alice` 只能对 `acme` 组织内的仓库调用 `github.open_pr`」这样的规则以声明式方式编码。简单的网关则用手写 Python。两种形态都是合理的。

### 会话感知路由（Session-aware routing）

当用户的会话涉及多个服务器时，网关进行多路复用：开发者的单个 MCP 会话背后持有 N 个后端会话，每个服务器一个。来自任意后端的通知都经由网关路由到开发者的会话。

### 命名空间合并

网关合并所有后端的工具命名空间，通常采用冲突时加前缀的方式：`github.open_pr`、`notes.search`。这样路由就不会产生歧义。

### 注册中心

- **Official MCP Registry（`registry.modelcontextprotocol.io`）。** 在 Anthropic、GitHub、PulseMCP、Microsoft 的共同管理下上线。经命名空间验证（反向 DNS：`io.github.user/server`）。经过基础质量预筛。
- **Glama。** 以搜索为中心的元注册中心，聚合多个来源。
- **MCPMarket。** 偏商业化的目录，包含厂商列表。
- **MCP.so。** 社区目录；开放提交。
- **Smithery。** 包管理器风格的安装流程。
- **LobeHub。** 集成在其 LobeChat 应用中的 UI 化注册中心。

企业网关默认从 Official Registry 拉取，允许管理员从元注册中心策展式地补充条目，并拒绝任何未固定（unpinned）的内容。

### 反向 DNS 命名

Official Registry 强制要求公开服务器使用反向 DNS 命名：`io.github.alice/notes`。命名空间可以防止抢注，并让信任委托更加清晰。

### 厂商概览（2026 年 4 月）

| 厂商 | 优势 |
|--------|----------|
| Cloudflare MCP Portals | 边缘托管；集成 OAuth；有免费层 |
| Kong AI Gateway | K8s 原生；细粒度策略；日志输出到 OpenTelemetry |
| IBM ContextForge | 企业 IAM；合规；审计导出 |
| TrueFoundry | 偏 DevOps；以指标为先 |
| MintMCP | 面向开发者平台 |
| Envoy AI Gateway | 开源；可定制过滤器 |

Phase 17（生产基础设施）会更深入地讲解网关运维。

## 生产实践

`code/main.py` 提供了一个约 150 行的最小网关：通过伪造的 Bearer 令牌认证用户，持有按用户的 RBAC 策略，把请求路由到两个后端 MCP 服务器，把每次调用写入审计日志，强制执行限流，并拒绝任何描述哈希与固定清单不匹配的后端工具。

值得关注的点：

- `RBAC` 字典以 `user_id` 为键，值为允许的 `server_tool` 条目。
- `AUDIT_LOG` 是一个只追加（append-only）的事件列表。
- 限流采用按用户的令牌桶（token bucket）。
- 固定清单是一个 `server::tool -> hash` 的字典。

## 交付产物

本课产出 `outputs/skill-gateway-bootstrap.md`。给定一份企业 MCP 规划（用户、后端、合规要求），该 skill 会生成一份网关配置规格。

## 练习

1. 运行 `code/main.py`。先以被允许的用户身份发起调用；再以被禁止的用户身份发起调用；最后发起一轮超出限流的突发请求。验证这三种流程。

2. 添加一条策略：在结果返回客户端之前对 PII 进行脱敏。用一个简单的正则匹配 SSN 形态的字符串；并指出其遗漏之处（邮箱、电话号码）。

3. 扩展审计日志，使其输出 OpenTelemetry GenAI span。Phase 13 · 20 会讲解具体的属性。

4. 为一个 50 人开发团队设计 RBAC 策略，涉及五个后端（notes、github、postgres、jira、slack）。每个后端谁有只读权限？谁有写权限？

5. 从头到尾读完 Cloudflare 的企业级 MCP 博客文章。找出一个 Cloudflare 提供、而这个 stdlib 网关没有的功能。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| Gateway | 「MCP 代理」 | 位于客户端与后端之间的集中化服务器 |
| Credential vaulting | 「后端令牌留在服务端」 | 开发者永远看不到上游令牌 |
| Session-aware routing | 「多后端会话」 | 网关为每个开发者会话多路复用 N 个后端会话 |
| Tool-hash pinning | 「审批清单」 | 每个已审批工具描述的 SHA256；集中阻止「抽地毯」攻击 |
| RBAC | 「按用户的策略」 | 针对工具和服务器的基于角色的访问控制 |
| Policy-as-code | 「声明式规则」 | 在网关层执行的 OPA/Rego、Kyverno、Styra 策略 |
| Audit log | 「谁、做了什么、何时」 | 用于合规的只追加事件日志 |
| Rate limit | 「按用户的令牌桶」 | 按分钟的上限，防止滥用 |
| Official MCP Registry | 「规范上游」 | `registry.modelcontextprotocol.io`，经命名空间验证 |
| Reverse-DNS naming | 「注册中心命名空间」 | `io.github.user/server` 命名约定 |

## 延伸阅读

- [Official MCP Registry](https://registry.modelcontextprotocol.io/) —— 规范上游，经命名空间验证
- [Cloudflare — Enterprise MCP](https://blog.cloudflare.com/enterprise-mcp/) —— 带 OAuth 与策略的网关模式
- [agentic-community — MCP gateway registry](https://github.com/agentic-community/mcp-gateway-registry) —— 开源参考网关
- [TrueFoundry — What is an MCP gateway?](https://www.truefoundry.com/blog/what-is-mcp-gateway) —— 功能对比文章
- [IBM — MCP context forge](https://github.com/IBM/mcp-context-forge) —— IBM 出品的企业级网关
