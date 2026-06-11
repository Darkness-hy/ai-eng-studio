# MCP 安全（二）—— OAuth 2.1、Resource Indicators 与增量授权范围

> 远程 MCP 服务器需要的是授权（authorization），而不仅仅是认证（authentication）。2025-11-25 版规范对齐了 OAuth 2.1 + PKCE + resource indicators（RFC 8707）+ 受保护资源元数据（RFC 9728）。SEP-835 进一步加入了增量授权范围同意机制：在收到 403 WWW-Authenticate 时触发升级授权（step-up authorization）。本课将以状态机的形式实现这个升级授权流程，让你看清每一跳的细节。

**Type:** Build
**Languages:** Python (stdlib, OAuth state machine simulator)
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 15 (security I)
**Time:** ~75 minutes

## 学习目标

- 区分资源服务器（resource server）与授权服务器（authorization server）的职责。
- 走通由 PKCE 保护的 OAuth 2.1 授权码流程。
- 使用 `resource` 参数（RFC 8707）和受保护资源元数据（RFC 9728）防御混淆代理（confused-deputy）攻击。
- 实现升级授权：服务器返回 403 并在 WWW-Authenticate 中要求更高的授权范围；客户端重新征求用户同意后重试请求。

## 问题背景

早期的 MCP（2025 年之前）的远程服务器要么使用临时拼凑的 API key，甚至完全没有认证。2025-11-25 版规范用一套完整的 OAuth 2.1 配置（profile）补上了这个缺口。

三个真实场景的需求：

- **普通远程服务器。** 用户安装了一个访问其 Notion / GitHub / Gmail 的远程 MCP 服务器。带 PKCE 的 OAuth 2.1 正是适合这种场景的方案。
- **授权范围升级。** 一个笔记服务器最初只被授予 `notes:read`，之后某个操作可能需要 `notes:write`。与其重走整个流程，升级授权（SEP-835）只需请求额外的授权范围。
- **防御混淆代理。** 客户端持有一个受众（audience）限定为服务器 A 的令牌。如果服务器 A 是恶意的，它可能试图把这个令牌转交给服务器 B。Resource indicators（RFC 8707）把令牌钉死在其预期受众上。

OAuth 2.1 并不新鲜。新的是 MCP 的配置方式：明确规定的必选流程（只允许授权码 + PKCE；默认禁止隐式流程和客户端凭证流程）、每次令牌请求都必须携带 resource indicator，以及发布受保护资源元数据让客户端知道该去哪里授权。

## 核心概念

### 角色

- **客户端（Client）。** 即 MCP 客户端（Claude Desktop、Cursor 等）。
- **资源服务器（Resource server）。** 即 MCP 服务器（笔记、GitHub、Postgres 等任意服务）。
- **授权服务器（Authorization server）。** 负责签发令牌。可以与资源服务器是同一个服务，也可以是独立的身份提供方（IdP，如 Auth0、Keycloak、Cognito）。

在 MCP 的配置中，资源服务器和授权服务器可以（CAN）部署在同一主机上，但应当（SHOULD）通过 URL 加以区分。

### 授权码 + PKCE

流程如下：

1. 客户端生成 `code_verifier`（随机值）和 `code_challenge`（SHA256 哈希）。
2. 客户端将用户重定向到 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户同意授权。授权服务器重定向到 `redirect_uri?code=...`。
4. 客户端向 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...` 发送 POST 请求。
5. 授权服务器用存储的 challenge 校验 verifier 的哈希值，然后签发访问令牌。
6. 客户端使用该令牌：对资源服务器的每个请求都带上 `Authorization: Bearer ...`。

PKCE 防御授权码拦截攻击。Resource indicators 则保证令牌在其他地方无效。

### 受保护资源元数据（RFC 9728）

资源服务器发布一份 `.well-known/oauth-protected-resource` 文档：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

客户端通过资源服务器来发现授权服务器。这减少了配置负担——客户端只需要知道资源 URL。

### Resource indicators（RFC 8707）

令牌请求中的 `resource` 参数限定了令牌的预期受众。签发的令牌中包含 `aud: "https://notes.example.com"`。其他 MCP 服务器收到这个令牌时会检查 `aud` 并拒绝它。

### 授权范围模型

授权范围（scope）是以空格分隔的字符串。MCP 中常见的约定：

- `notes:read`、`notes:write`、`notes:delete`
- `admin:*` 用于管理能力（谨慎使用）
- `profile:read` 用于身份信息

授权范围的选择应遵循最小权限原则：现在需要什么就请求什么，需要更多时再升级。

### 升级授权（SEP-835）

用户授予了 `notes:read`。之后用户要求智能体删除一条笔记。服务器响应：

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

客户端看到 insufficient_scope 错误后，向用户弹出针对额外授权范围的同意对话框，为其执行一个迷你 OAuth 流程，然后用新令牌重试请求。

### 令牌受众校验

每个请求：服务器检查 `token.aud == self.resource_url`。不匹配则返回 401。这阻止了令牌的跨服务器复用。

### 短效令牌与轮换

访问令牌应当（SHOULD）是短效的（默认 1 小时）。刷新令牌在每次刷新时轮换。客户端在后台静默处理刷新。

### 禁止令牌透传

采样服务器（Phase 13 · 11）禁止（MUST NOT）把客户端的令牌透传给其他服务。采样请求就是边界。

### 防御混淆代理

令牌绑定到 `aud`，客户端绑定到 `client_id`。每个请求都同时校验这两者。规范明确禁止了旧式的「令牌转交」（pass-the-token）模式——这种模式在 MCP 之前的远程工具生态中很常见。

### 客户端 ID 发现

每个 MCP 客户端在一个固定 URL 上发布自己的元数据。授权服务器可以抓取客户端的元数据文档来发现其重定向 URI 和联系信息。这免去了手动注册客户端的步骤。

### 网关与 OAuth

Phase 13 · 17 会展示企业网关如何处理 OAuth：网关持有上游服务器的凭证，发给客户端的令牌由网关签发，上游令牌永远不离开网关。这颠倒了信任模型——用户只需向网关认证一次，由网关处理对 N 个服务器的授权。

## 生产实践

`code/main.py` 以状态机的形式模拟了完整的 OAuth 2.1 升级授权流程。它实现了：

- PKCE 的 code-verifier / challenge 生成。
- 带 resource indicator 的授权码流程。
- 受保护资源元数据端点。
- 带受众检查的令牌校验。
- 收到 `insufficient_scope` 时的升级授权。

本课没有 HTTP 服务器；状态机在内存中运行，方便你追踪每一跳。Phase 13 · 17 的网关课会把它接入真实的传输层。

## 交付产物

本课产出 `outputs/skill-oauth-scope-planner.md`。给定一个带工具的远程 MCP 服务器，该技能会设计授权范围集合、令牌绑定规则和升级授权策略。

## 练习

1. 运行 `code/main.py`。追踪双授权范围的升级流程。记下升级授权时哪些跳会重复执行。

2. 添加刷新令牌轮换：每次刷新签发新的刷新令牌并使旧令牌失效。模拟被盗刷新令牌在轮换后被使用的情况，确认它会失败。

3. 用标准库 http.server 把受保护资源元数据端点实现成真实的 HTTP 响应。参照第 09 课的 /mcp 端点写法。

4. 为一个 GitHub MCP 服务器设计授权范围层级：读仓库、写 PR、批准 PR、合并 PR、管理员。在每个层级之间使用升级授权。

5. 阅读 RFC 8707 和 RFC 9728。找出 RFC 9728 中 MCP 的用法与 RFC 示例不同的那一个字段。（提示：与 `scopes_supported` 有关。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| OAuth 2.1 | 「现代 OAuth」 | 整合后的 RFC，强制要求 PKCE 并禁止隐式流程 |
| PKCE | 「持有证明」 | code verifier + challenge，挫败授权码拦截攻击 |
| Resource indicator | 「令牌受众」 | RFC 8707 的 `resource` 参数，把令牌钉死在单个服务器上 |
| 受保护资源元数据 | 「发现文档」 | RFC 9728 的 `.well-known/oauth-protected-resource` |
| 升级授权 | 「增量同意」 | SEP-835 定义的按需追加授权范围的流程 |
| `insufficient_scope` | 「带 WWW-Authenticate 的 403」 | 服务器发出的信号，要求重新同意更大的授权范围 |
| 混淆代理 | 「跨服务令牌复用」 | 受信任的令牌持有者不当转发令牌的攻击 |
| 短效令牌 | 「访问令牌 TTL」 | 快速过期的 bearer 令牌；由刷新令牌续期 |
| 授权范围层级 | 「最小权限阶梯」 | 分级的授权范围集合，层级之间通过升级授权过渡 |
| 客户端 ID 元数据 | 「客户端发现文档」 | 客户端发布自身 OAuth 元数据的 URL |

## 延伸阅读

- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — MCP OAuth 配置的权威规范
- [den.dev — MCP November authorization spec](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 版变更的逐项解读
- [RFC 8707 — Resource indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — 受众绑定的 RFC
- [RFC 9728 — OAuth 2.0 protected resource metadata](https://datatracker.ietf.org/doc/html/rfc9728) — 发现文档的 RFC
- [Aembit — MCP OAuth 2.1, PKCE and the future of AI authorization](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — 升级授权流程的实战讲解
