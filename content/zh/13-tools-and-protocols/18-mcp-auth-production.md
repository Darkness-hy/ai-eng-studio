# 生产环境中的 MCP 鉴权 — 客户端注册、JWKS 刷新与受众绑定令牌

> 第 16 课在内存中搭建了 OAuth 2.1 状态机。到了 2026 年，你交付给真实组织的每一台 MCP 服务器都运行在生产级鉴权之后：能够支撑无上限客户端规模的注册机制（优先使用 Client ID Metadata Document，动态客户端注册作为向后兼容的回退方案）、授权服务器元数据发现（RFC 8414 *或* OpenID Connect Discovery）、不会在凌晨 3 点令牌校验时掉链子的 JWKS 缓存刷新，以及拒绝跨资源重放的受众绑定（audience-pinned）令牌。本课用三个角色——授权服务器、资源服务器（即 MCP 服务器）和客户端——对完整的鉴权面建模，让你能追踪从发现到一次通过校验的工具调用之间的每一跳。
>
> **规范说明（2025-11-25）：** 2025 年 11 月的 MCP 授权规范将动态客户端注册（Dynamic Client Registration）从 `SHOULD` 降级为 `MAY`，并将 **Client ID Metadata Document（CIMD）** 确立为推荐的默认注册机制。本课按规范的优先级顺序同时讲解两者；代码演示部分保留 DCR，因为它可以完全自包含地运行在单个进程内。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 16 (OAuth 2.1 state machine), Phase 13 · 17 (gateways)
**Time:** ~90 minutes

## 学习目标

- 通过 RFC 8414 元数据发现授权服务器，并验证其契约。
- 实现 RFC 7591 动态客户端注册，让 MCP 客户端无需管理员介入即可完成注册。
- 按计划缓存并刷新 JWKS 密钥，使签名校验在密钥轮换后依然存活。
- 使用 RFC 8707 资源指示符（resource indicator）将令牌绑定到单个 MCP 资源，并拒绝混淆代理人式的重用。
- 干净地分离三个角色——授权服务器、资源服务器、客户端——让每个角色只执行属于自己的检查。
- 读懂 IdP 能力矩阵，并在 IdP 无法满足 MCP 鉴权配置文件时拒绝部署。

## 问题背景

第 16 课的模拟器在内存中运行 OAuth 2.1。生产环境存在三个纯内存模拟器看不到的运维缺口。

第一个缺口是注册。真实的组织会运行数百台 MCP 服务器和数千个 MCP 客户端。运维人员不可能逐个把每位 Cursor 用户手工注册成 OAuth 客户端。2025-11-25 规范给客户端规定了解决该问题的优先级顺序：如果已经有预注册的 `client_id` 就用它；否则使用 **Client ID Metadata Document**（客户端用一个自己掌控的 HTTPS URL 来标识自己，授权服务器去*拉取*该元数据）；否则回退到 **RFC 7591 动态客户端注册**（客户端*推送*一个 `POST /register` 请求并当场获得 `client_id`）；最后才提示用户手动处理。CIMD 是推荐的默认方案，因为它彻底免去了按服务器逐个注册的环节，同时保留了以 DNS 为根的信任模型；DCR 则为向后兼容而保留。两者都从授权服务器的元数据中发现各自的入口：CIMD 看 `client_id_metadata_document_supported`，DCR 看 `registration_endpoint`。

第二个缺口是密钥轮换。JWT 校验依赖授权服务器的签名密钥，这些密钥以 JSON Web Key Set（JWKS）的形式发布。授权服务器会按计划轮换它们（通常每小时一次，事件响应时可能更频繁）。一台只在启动时拉取一次 JWKS 的 MCP 服务器，在轮换窗口到来之前一切正常——然后每个请求都会失败，直到重启。生产环境把 JWKS 做成带刷新任务的缓存值：刷新任务在旧密钥过期前覆盖缓存，并在缓存未命中时执行一次回退拉取，以应对令牌由比缓存更新的密钥签名的情况。

第三个缺口是受众绑定。第 16 课介绍了 RFC 8707 资源指示符。在生产环境中，该指示符成为每个请求上的硬性声明检查：MCP 服务器将 `token.aud` 与自己的规范资源 URL 比对，不匹配则以 HTTP 401 拒绝。这是抵御以下攻击的唯一防线——上游 MCP 服务器（或持有发给某台服务器的令牌的恶意客户端）将该令牌重放到同一信任网格中的另一台服务器。

本课把每个缺口都映射到鉴权面上的一个具体构件。元数据文档是一个 HTTP 端点；JWKS 缓存刷新是一个定时任务加一个键值缓存；JWT 校验是资源服务器在分发任何工具调用之前运行的例程。保持三个角色分离，每个角色只执行自己拥有的检查：授权服务器签发并轮换密钥，资源服务器缓存并校验，客户端发现并注册。

## 核心概念

### RFC 8414 — OAuth 授权服务器元数据

位于 `/.well-known/oauth-authorization-server` 的一份文档描述了客户端所需的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

拿到 MCP 资源 URL 的客户端会进行链式发现：先取 RFC 9728 的 `oauth-protected-resource`（资源服务器的文档）得到 issuer，再取 `oauth-authorization-server`（本 RFC）得到所有端点。客户端永远不硬编码授权 URL。

在信任一个 IdP 用于 MCP 之前，你要验证的契约是：

- `code_challenge_methods_supported` 包含 `S256`（即 RFC 7636 的 PKCE）。规范说得很明确：如果该字段**缺失**，则授权服务器不支持 PKCE，客户端 **MUST** 拒绝继续。
- `grant_types_supported` 包含 `authorization_code`，并排除 `password` 和 `implicit`。
- 至少声明了一条注册路径：`client_id_metadata_document_supported: true`（CIMD，首选）**或** `registration_endpoint`（RFC 7591 DCR，回退）。任一即可满足契约；不再硬性要求 DCR。
- 对于 OAuth 2.1，`response_types_supported` 必须恰好是 `["code"]`。

如果缺少 `S256`，MCP 服务器拒绝针对该 IdP 部署——PKCE 没有降级模式。如果*两条*注册路径都没有声明、且你也没有预注册的 `client_id`，那么同样无法注册；这时错的是部署清单，不是代码。

### RFC 9728（回顾）— 受保护资源元数据

第 16 课讲过 RFC 9728。生产环境的差异在于：这份文档是客户端查找*这台* MCP 服务器所信任的授权服务器的唯一来源。单台 MCP 服务器可能接受来自多个 IdP 的令牌（员工一个、合作伙伴一个）。RFC 9728 声明这个集合；RFC 8414 则记录每个 IdP 支持什么。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Client ID Metadata Document（推荐的默认方案）

CIMD 把注册从*推*反转为*拉*。客户端不再请求授权服务器铸造一个 `client_id`，而是直接用一个自己掌控的 HTTPS URL **作为** `client_id`。该 URL 解析为一份 JSON 元数据文档；授权服务器在 OAuth 流程中按需拉取它。信任以 DNS 为根：如果服务器运营方信任 `app.example.com`，它就信任由 `https://app.example.com/client.json` 提供的客户端。没有注册往返、没有会被耗尽的 `client_id` 命名空间、也没有需要逐服务器同步的状态。

客户端托管的元数据文档如下：

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

文档中的 `client_id` 值 **MUST** 与提供它的 URL 完全相等（授权服务器会验证这一点；不匹配即拒绝）。授权服务器通过在其 RFC 8414 元数据中声明 `client_id_metadata_document_supported: true` 来表明支持。

规范直言不讳的两个安全事实：

- **SSRF。** 授权服务器要拉取一个攻击者可控的 URL，因此必须防御服务端请求伪造（不得向内部/管理端点发起请求）。
- **localhost 冒充。** 仅靠 CIMD 无法阻止本地攻击者冒用合法客户端的元数据 URL 并绑定任意 `localhost` 重定向。授权服务器 **MUST** 在用户授权确认页面上清楚展示重定向 URI 的主机名，并且 **SHOULD** 对仅有 `localhost` 重定向的情况给出警告。

由于 CIMD 不需要任何服务端状态，也就不存在 DCR 那样需要搭建的注册器。客户端这边是只读的：把元数据文档放在一个静态 HTTPS 端点上，让授权服务器来拉取即可。

### RFC 7591 — 动态客户端注册（回退 / 向后兼容）

DCR 现在是 `MAY`，保留它是为了与 2025-11-25 之前的部署以及尚不支持 CIMD 的 IdP 向后兼容。没有它（且没有 CIMD 和预注册）时，每个 MCP 客户端（Cursor、Claude Desktop、自定义智能体）都需要与 IdP 管理员进行一次带外交换。有了 DCR，客户端只需提交：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

服务器返回 `client_id`，以及用于后续更新的 `registration_access_token`：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

对于运行在用户设备上的 MCP 客户端，`token_endpoint_auth_method: none` 是正确的默认值。它们只拿到 `client_id`——没有可被窃取的 `client_secret`。公共客户端所需的持有证明由 PKCE 提供。

三个生产环境陷阱：

- 注册端点必须按源 IP 限流。否则，恶意行为者可以脚本化提交数百万次伪造注册，耗尽 `client_id` 命名空间。应在注册器处理请求之前先执行限流检查。
- 部分企业级 IdP 要求 `software_statement`（一个为客户端背书的签名 JWT）。本课的模拟实现跳过了它；生产环境会接入一个验证步骤，对重定向 URI 不是 localhost 的未签名注册一律拒绝。
- `registration_access_token` 必须以哈希形式存储，而非明文。该令牌一旦被盗，攻击者就能改写客户端的重定向 URI。

### RFC 8707（回顾）— 资源指示符

第 16 课确立了基本形态。生产环境的规则是：每个令牌请求都携带 `resource=<canonical-mcp-url>`，MCP 服务器在每次调用时验证 `token.aud` 与自身的资源 URL 匹配。规范 URI 是该服务器*最具体*的标识符：scheme 与 host 使用小写、不带 fragment、按惯例不带尾部斜杠。路径部分**不会**被规则性剥离——当路径是区分单个 MCP 服务器所必需时，规范会保留它。`https://mcp.example.com`、`https://mcp.example.com/mcp`、`https://mcp.example.com:8443` 和 `https://mcp.example.com/server/mcp` 都是合法的规范 URI。为每台服务器选定一个，并把 `aud` 精确绑定到它。（为简洁起见，本课的模拟实现使用 `https://notes.example.com` 这种纯主机受众；在同一 origin 下托管多个 MCP 服务器的部署应通过路径区分它们。）

### RFC 7636（回顾）— PKCE

在 OAuth 2.1 中 PKCE 是强制的。本课的授权码流程始终携带 `code_challenge` 和 `code_verifier`。服务器会拒绝任何不带 verifier、或 verifier 哈希后与存储的 challenge 不匹配的令牌请求。

### MCP 规范 2025-11-25 鉴权配置文件

MCP 规范（2025-11-25）对 MCP 服务器的授权层必须做的事情有精确规定：

- 实现 RFC 9728 受保护资源元数据，并通过两种方式之一提供其位置：401 响应上的 `WWW-Authenticate: Bearer resource_metadata="..."` 头，**或**周知 URI `/.well-known/oauth-protected-resource`（SEP-985 将该头改为可选，以周知 URI 作为回退）。元数据中的 `authorization_servers` 字段 **MUST** 至少指明一个服务器。
- 仅通过 `Authorization: Bearer ...` 在**每个**请求上接受令牌——绝不放在查询字符串里，也绝不只在会话开始时校验一次。
- 每个请求都校验 `aud`、`iss`、`exp` 和所需作用域。服务器 **MUST** 验证令牌确实是为它专门签发的（受众检查）；`aud` 缺失或不匹配一律拒绝，绝不视为通配。
- 在 401/403 时返回携带以下内容的 `WWW-Authenticate: Bearer`：`error=...`、`resource_metadata="<PRM-URL>"` 参数（指向元数据文档的 URL，而*不是*裸资源地址），以及在 `insufficient_scope`（403）时附带 `scope="..."`。注意：该参数叫 `resource_metadata`，是一个发现指针——质询中不存在 `resource` 参数。
- 授权服务器发现接受 RFC 8414 OAuth 元数据**或** OpenID Connect Discovery 1.0 中的**任一种**；客户端必须按优先级顺序尝试两种周知后缀。
- 防御**混淆攻击（mix-up attack）** 的是客户端（而非服务器）：它在重定向前记录预期的 `issuer`，并在兑换授权码之前校验授权响应中的 `iss` 参数（RFC 9207）。仅靠 PKCE 挡不住 mix-up，因为客户端会把自己的 `code_verifier` 交给它被引导到的任何令牌端点。

OAuth 2.1 草案是基底；RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD 是表面；MCP 规范是配置文件（profile）。

### IdP 能力矩阵

并非所有 IdP 都支持完整的 MCP 配置文件。下表记录的是截至 2025-11-25 规范的事实性能力陈述。它是一道*部署门禁*，不是推荐。

CIMD 随 2025-11-25 规范发布，其底层 OAuth 草案直到 2025 年 10 月才被采纳，所以厂商支持仍在逐步到位——把下表中的 "CIMD" 一列理解为"当前现状，请在你的租户中自行验证"，而非永久性结论。

| IdP 类别 | AS 元数据（8414/OIDC） | CIMD | RFC 7591 DCR | RFC 8707 resource | RFC 7636 S256 PKCE | 备注 |
|---|---|---|---|---|---|---|
| 自托管（Keycloak） | 是 | 跟进中 | 是 | 是（自 24.x 起） | 是 | 本课 MCP 配置文件的参考 IdP；端到端完整的 DCR 路径，CIMD 正在跟进新规范。 |
| 企业 SSO（Microsoft Entra ID） | 是 | 跟进中 | 是（高级套餐） | 是 | 是 | DCR 可用性因租户套餐而异；部署前在目标租户中验证。 |
| 企业 SSO（Okta） | 是 | 跟进中 | 是（Okta CIC / Auth0） | 是 | 是 | DCR 在 Auth0（现 Okta CIC）上可用；经典 Okta 组织需要管理员预注册。 |
| 社交登录 IdP（泛指） | 不一 | 否 | 极少 | 极少 | 是 | 大多数社交 IdP 把客户端当作静态合作方；没有自助注册。只把它们用作身份来源，在其上叠加你自己的 MCP 感知授权服务器。 |
| 自定义 / 自研 | 视情况 | 视情况 | 视情况 | 视情况 | 视情况 | 如果自己实现，就实现完整配置文件并优先支持 CIMD。跳过 PKCE 或受众绑定会破坏 MCP 鉴权契约。 |

部署清单的拒绝规则：如果所选 IdP 的 `code_challenge_methods_supported` 中没有列出 `S256`，MCP 服务器拒绝启动——PKCE 没有降级模式。注册是较软的门禁：你需要*一条*可行的路径（预注册的 `client_id`、`client_id_metadata_document_supported: true`，或 `registration_endpoint`）。仅缺少 DCR 已不再触发拒绝，因为 CIMD 或预注册都可以覆盖它。

### JWKS 刷新模式（AS 负责轮换，资源服务器负责刷新）

把两个动词分开，因为混淆它们是真实存在的生产事故：

- **轮换（Rotate）** 是*授权服务器*做的事：铸造新签名密钥、发布到 JWKS、稍后下线旧密钥。资源服务器不参与也无法参与——它不持有 IdP 的私钥。
- **刷新（Refresh）** 是*资源服务器*做的事：重新 `GET` 已发布的 JWKS 到自己的缓存中。这是资源服务器唯一会执行的 JWKS 操作。

生产环境的故障模式是缓存过期。解法是定时刷新任务加键值缓存。资源服务器运行一个任务（cron、定时器，看你的运行时提供什么），按固定间隔拉取 `<issuer>/.well-known/jwks.json` 并覆写 `cache[issuer] = {keys, fetched_at}`。校验器从该缓存读取。如果令牌的 `kid` 在缓存中找不到，则触发**一次**同步刷新作为回退，然后重新查找。这一招同时覆盖两种情况：常规的定时刷新，以及密钥重叠窗口期——由全新密钥签名的令牌在下一次定时刷新之前就到达。

回退**必须是重新拉取，绝不能是轮换**。如果把缓存未命中路径接到"轮换并铸造新密钥"上，会坏两件事：（1）新铸的密钥产生的 `kid` *仍然*不会与令牌匹配，查找照样失败；（2）攻击者用随机 `kid` 喷洒令牌，就能迫使服务器无限制地创建密钥——一场自己造成的 DoS。重新拉取是幂等的，一个伪造的 `kid` 最多浪费一次拉取。

缓存的形态：

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同时存在两把密钥是稳态。授权服务器轮换时先引入下一把密钥（`k_2026_04`）再下线上一把（`k_2026_03`），因此旧密钥签发的令牌在过期前依然有效。缓存持有两者的并集；校验器按 `kid` 选取。

### 校验例程

MCP 服务器在分发任何工具调用之前先运行校验。`code/main.py` 使用的形态：

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` 解码 JWT，从 JWKS 缓存解析签名密钥（未命中时刷新一次），验证签名，然后依次检查 `iss` 是否在允许列表中、`aud` 是否匹配本服务器的规范资源、`exp` 以及所需作用域——在第一处失败时返回 `WWW-Authenticate` 质询。把它做成资源服务器上的单一例程，意味着每个入口（每次工具调用、每种传输方式）都走同一套检查；不存在任何未经校验就能触达工具的路径。

### 受众重放攻击演练（访问令牌权限限制）

服务器 A（`notes.example.com`）和服务器 B（`tasks.example.com`）都注册在同一个授权服务器上。服务器 A 被攻陷。攻击者拿走用户的 notes 令牌，重放到服务器 B。

服务器 B 的校验器：

1. 解码 JWT，按 `kid` 取 JWKS，验证签名。
2. 用其受保护资源元数据中的 `authorization_servers` 检查 `iss`。（通过——同一个 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败——令牌的 `aud` 是 `https://notes.example.com`。）
4. 返回 401，附带 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`。

在协议层面，受众声明是抵御此类攻击的唯一防线。为了性能而跳过它是最常见的生产错误；校验器必须在每个请求上运行，而不是只在会话开始时运行。规范称之为**访问令牌权限限制（access-token privilege restriction）**：MCP 服务器 `MUST` 拒绝任何未在受众中指明自己的令牌。

> **命名说明。** 规范将*混淆代理人（confused deputy）* 一词保留给一个相关但不同的问题：充当通往第三方 API 的 OAuth **代理**、使用静态 client ID 的 MCP 服务器，在未获得逐客户端用户同意的情况下转发令牌。受众绑定修复的是上面的重放攻击；混淆代理人的修复手段是逐客户端的用户同意，**外加**永不把入站令牌透传给上游 API（MCP 服务器 `MUST` 自行获取独立的上游令牌）。

### 混淆攻击（服务器无法代劳的客户端侧防御）

一个客户端在其生命周期中会与多个授权服务器打交道。恶意 AS 可以试图诱使客户端把诚实 AS 的授权码拿到攻击者的令牌端点去兑换。受众绑定在这里帮不上忙——攻击发生在任何令牌存在之前。防御在客户端一侧（RFC 9207）：

1. 重定向之前，客户端从经过验证的 AS 元数据中记录预期的 `issuer`。
2. 收到授权响应时，客户端先将返回的 `iss` 参数与记录的 issuer 比对（简单字符串比较，不做归一化），然后才把授权码发往任何地方。
3. 不匹配（或在 AS 声明了 `authorization_response_iss_parameter_supported` 的情况下 `iss` 缺失）→ 拒绝，连 `error` 字段都不要展示。

仅靠 PKCE 挡不住 mix-up，因为客户端会把自己的 `code_verifier` 交给它被引导到的任何令牌端点。这就是规范要求把 issuer 与 PKCE verifier、`state` 一起按请求逐个记录的原因。

### 故障模式

- **JWKS 过期。** AS 轮换密钥后，校验器拒绝合法令牌。解法是上文的定时刷新 + 缓存未命中重拉模式。绝不要在没有刷新任务的情况下缓存 JWKS。
- **把轮换当回退。** 将缓存未命中路径接到"轮换并铸造"而非重新拉取是一个真实存在的 bug：它永远造不出缺失的那个 `kid`，还会把攻击者可控的 `kid` 值变成一场密钥创建 DoS。回退必须是幂等的 `refresh-jwks`。
- **`aud` 声明缺失。** 一些 IdP 默认在令牌请求不带 `resource` 时省略 `aud`。校验器必须拒绝 `aud` 缺失的令牌，而不是把缺失当作通配。
- **缺失 `iss` 检查导致的 mix-up。** 如果客户端不把 RFC 9207 的 `iss` 授权响应参数与重定向前记录的 issuer 做校验，就可能被引导到攻击者的令牌端点去兑换诚实 AS 的授权码。这是客户端侧的失败；资源服务器无法弥补。
- **作用域升级竞态。** 同一用户的两个并发权限提升（step-up）流程可能都成功，产生两个作用域不同的访问令牌。校验器必须使用请求上呈递的那个令牌，而不是去查"该用户当前的作用域"——后者会制造 TOCTOU 窗口。
- **注册令牌被盗。** 泄露的 `registration_access_token` 让攻击者得以改写重定向 URI。静态存储时做哈希；要求客户端每次更新都呈递明文；一有怀疑立即轮换。
- **`iss` 未固定。** 接受任意 `iss` 的校验器会让攻击者自建授权服务器、为目标受众注册客户端并签发令牌。受保护资源元数据的 `authorization_servers` 列表就是允许列表；要强制执行它。

## 生产实践

`code/main.py` 用纯标准库 Python 和三个角色——`AuthorizationServer`、`ResourceServer`、`Client`——演练完整的生产流程。流程如下：

1. 授权服务器在 `/.well-known/oauth-authorization-server` 发布 RFC 8414 元数据。
2. MCP 客户端调用元数据端点，检查注册选项（CIMD 看 `client_id_metadata_document_supported`，DCR 看 `registration_endpoint`）以及 `S256` PKCE 支持。
3. 演练走的是 DCR 回退路径：客户端向 `/register`（RFC 7591）提交请求并获得 `client_id`。（CIMD 客户端则会直接出示自己的 HTTPS `client_id` URL，跳过此步。）
4. MCP 客户端运行带 PKCE 保护的授权码流程（RFC 7636），附带 `resource` 指示符（RFC 8707）。
5. MCP 客户端携带 `Authorization: Bearer ...` 调用 MCP 服务器上的工具。
6. MCP 服务器运行 `validate`，从 JWKS 缓存解析签名密钥。
7. IdP 轮换一把密钥；定时刷新把 JWKS 重新拉进缓存。
8. 下一次调用无需重启即可用刷新后的密钥通过校验，且在重叠窗口内之前的令牌依然有效。
9. 针对另一个 MCP 资源的受众重放尝试得到 401，附带 `audience mismatch` 和一个 `resource_metadata` 指针。

这里的 JWT 使用 HS256 共享密钥（这样本课只依赖标准库即可运行）。生产环境使用 RS256 或 EdDSA 配合上文的 JWKS 模式；校验逻辑除此之外完全相同。因为 IdP 和资源服务器运行在同一进程内，`refresh_jwks` 直接读取授权服务器的密钥列表；在真实网络上，它是对 `jwks_uri` 的一次 HTTP `GET`。

## 交付产物

本课产出 `outputs/skill-mcp-auth.md`。给定一份 MCP 服务器配置和一组 IdP 能力，该技能输出需要搭建的鉴权面——受保护资源元数据、应采用的注册路径（CIMD、预注册或 DCR 回退）、JWKS 刷新计划、作用域映射，以及当 IdP 不支持完整 RFC 配置文件时应执行的拒绝规则。

## 练习

1. 运行 `code/main.py`，追踪整个流程。注意 IdP 如何在第 6 步轮换密钥、定时的 `refresh_jwks` 如何重新拉取已发布的密钥集，以及旧令牌（重叠窗口内）和新令牌如何都无需重启即可通过校验。

2. 向受保护资源元数据的 `authorization_servers` 列表添加一个新 IdP。签发一个由新 IdP 签名的令牌，确认校验器接受它。再签发一个由未列入的 IdP 签名的令牌，确认校验器以 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"` 拒绝。

3. 为 `register_client` 添加一个在注册器接受请求之前运行的限流检查。按源 IP 实现令牌桶，存放在一个以 IP 为键的小字典中。

4. 阅读 RFC 7591，找出本课的 `/register` 处理器没有校验的两个字段，并补上校验。（提示：`software_statement` 和 `redirect_uris` 的 URI scheme。）

5. 添加一条 Client ID Metadata Document 路径。提供一份 `client.json`，其 `client_id` 等于其自身 URL，并让授权服务器拉取并验证它（`client_id` ≠ URL 即拒绝）。确认 CIMD 客户端无需调用 `register_client` 即可完成注册。

6. 证明 DoS 修复有效。向校验器发送一个带随机 `kid` 的令牌，确认 `refresh_jwks` 至多运行一次、且授权服务器的密钥数量没有增长。然后故意把回退改接成"轮换并铸造"，观察密钥数量随每个伪造令牌攀升——之后记得恢复成重新拉取。

7. 实现混淆攻击一节中客户端侧的 RFC 9207 `iss` 检查：在发起授权请求前记录预期 issuer，然后拒绝 `iss` 不匹配的授权响应。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| ASM | "OAuth 元数据文档" | RFC 8414 的 `/.well-known/oauth-authorization-server` JSON |
| CIMD | "客户端元数据 URL" | Client ID Metadata Document——用作 `client_id` 的 HTTPS URL；AS 拉取其 JSON。自 2025-11-25 起为推荐默认方案 |
| DCR | "自助式客户端注册" | RFC 7591 的 `POST /register` 流程；在 2025-11-25 中降级为 `MAY` 回退方案 |
| JWKS | "用于 JWT 校验的公钥" | JSON Web Key Set，从 `jwks_uri` 拉取，按 `kid` 索引 |
| 轮换 vs 刷新 | "更新密钥" | *轮换* = AS 铸造/下线签名密钥；*刷新* = 资源服务器重新拉取已发布的密钥集。资源服务器永远只做刷新 |
| 资源指示符 | "受众参数" | RFC 8707 的 `resource` 参数，把令牌绑定到单台服务器 |
| `aud` 声明 | "受众" | 校验器与规范资源 URL 比对的 JWT 声明 |
| 受众重放 | "令牌重放" | 为服务器 A 签发的令牌被呈递给服务器 B；由受众校验防御（规范术语：访问令牌权限限制） |
| 混淆代理人 | "代理令牌滥用" | 使用静态 client ID 的 MCP 代理在未经逐客户端同意的情况下转发令牌；与受众重放不同 |
| 混淆攻击（mix-up） | "错误的令牌端点" | 客户端被引导到攻击者的端点去兑换诚实 AS 的授权码；由客户端侧的 RFC 9207 `iss` 防御 |
| `iss` 允许列表 | "受信任的授权服务器" | 受保护资源元数据中 `authorization_servers` 指明的集合 |
| `resource_metadata` | "去哪里找 PRM 文档" | 401/403 时 `WWW-Authenticate` 中指明 RFC 9728 元数据 URL 的参数 |
| 公共客户端 | "原生或浏览器客户端" | 没有 `client_secret` 的 OAuth 客户端；由 PKCE 弥补 |
| `WWW-Authenticate` | "401/403 响应头" | 携带驱动客户端恢复流程的 `Bearer error=...` 指令 |

## 延伸阅读

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — 本课实现的 MCP 鉴权配置文件
- [MCP blog — One Year of MCP: November 2025 Spec Release](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 2025-11-25 的变化（CIMD、XAA、DCR 降级）
- [Aaron Parecki — Client Registration in the November 2025 MCP Authorization Spec](https://aaronparecki.com/2025/11/25/1/mcp-authorization-spec-update) — CIMD 优先于 DCR 的理由
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — 发现契约
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR（回退路径）
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — 公共客户端的持有证明
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — 受众绑定
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — 资源服务器发现
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) — 防御混淆攻击的 `iss` 参数
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — 整合后的 OAuth 基底
