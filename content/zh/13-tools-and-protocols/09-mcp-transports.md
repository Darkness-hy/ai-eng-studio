# MCP 传输层 — stdio、Streamable HTTP 与 SSE 迁移

> stdio 只在本地可用，其他场景一概不行。Streamable HTTP（2025-03-26）是远程传输的标准。旧的 HTTP+SSE 传输已被废弃，将在 2026 年年中移除。选错传输层意味着一次迁移的代价；选对了，你就拥有一个可远程托管、具备会话连续性和 DNS 重绑定防护的 MCP 服务器。

**Type:** Learn
**Languages:** Python (stdlib, Streamable HTTP endpoint skeleton)
**Prerequisites:** Phase 13 · 07, 08 (MCP server and client)
**Time:** ~45 minutes

## 学习目标

- 根据部署形态（本地 vs 远程、单进程 vs 集群）在 stdio 和 Streamable HTTP 之间做出选择。
- 实现 Streamable HTTP 的单端点模式：POST 处理请求，GET 建立会话流。
- 强制执行 `Origin` 校验和会话 ID 语义，抵御 DNS 重绑定（DNS-rebinding）攻击。
- 在 2026 年年中的移除期限前，将遗留的 HTTP+SSE 服务器迁移到 Streamable HTTP。

## 问题背景

MCP 最初的远程传输（2024-11）是 HTTP+SSE：两个端点，一个接收客户端的 POST 请求，另一个是服务器到客户端的 Server-Sent-Events 流式通道。它能用，但很笨拙：每个会话需要两个端点，部分 CDN 前置缓存会出问题，还硬性依赖长连接 SSE，而一些 WAF 会激进地切断这类连接。

2025-03-26 规范用 Streamable HTTP 取而代之：单一端点，POST 处理客户端请求，GET 建立会话流，两者共享一个 `Mcp-Session-Id` 请求头。此后构建或迁移的所有服务器都使用 Streamable HTTP。旧的 SSE 模式正在被废弃——Atlassian Rovo 于 2026 年 6 月 30 日移除；Keboola 于 2026 年 4 月 1 日移除；其余大多数企业级服务器将在 2026 年底前完成移除。

而 stdio 对本地服务器依然重要。Claude Desktop、VS Code 以及所有 IDE 形态的客户端都通过 stdio 启动服务器。正确的心智模型是：stdio 用于"本机"，Streamable HTTP 用于"跨网络"。两者互不越界。

## 核心概念

### stdio

- 子进程传输。客户端启动服务器进程，通过 stdin/stdout 通信。
- 每行一个 JSON 对象，以换行符分隔。
- 没有会话 ID；进程本身就是会话标识。
- 无需鉴权（子进程继承父进程的信任边界）。
- 绝不要用于远程服务器——否则你得用 SSH 或 socat 做隧道，到了那一步还不如直接用 Streamable HTTP。

### Streamable HTTP

单一端点 `/mcp`（或任意路径），支持三种 HTTP 方法：

- **POST /mcp。** 客户端发送一条 JSON-RPC 消息。服务器要么回复单个 JSON 响应，要么回复包含一个或多个响应的 SSE 流（适用于批量响应以及与该请求相关的通知）。
- **GET /mcp。** 客户端打开一条长连接 SSE 通道。服务器用它发送服务器到客户端的请求（采样、通知、信息征询）。
- **DELETE /mcp。** 客户端显式终止会话。

会话由 `Mcp-Session-Id` 请求头标识：服务器在首个响应中设置它，客户端在之后的每个请求中回传。会话 ID 必须是密码学随机的（128 位以上）；出于安全考虑，客户端自选的 ID 会被拒绝。

### 单端点 vs 双端点

旧规范的双端点模式在 2026 年仍可调用——规范将其声明为"遗留兼容"。但所有新服务器都应采用单端点。官方 SDK 默认生成单端点；只有在对接尚未迁移的远程服务器时才使用遗留模式。

### `Origin` 校验与 DNS 重绑定

浏览器（目前）不是 MCP 客户端，但攻击者可以构造一个网页，诱导浏览器向 `localhost:1234/mcp` 发送 POST 请求——而那里正运行着用户的本地 MCP 服务器。如果服务器不检查 `Origin`，浏览器的同源策略救不了它，因为 `Origin: http://evil.com` 是一个合法的跨域来源。

2025-11-25 规范要求服务器拒绝 `Origin` 不在允许列表（allowlist）中的请求。允许列表通常包含 MCP 客户端的主机（`https://claude.ai`、`vscode-webview://*`）以及供本地 UI 使用的各种 localhost 变体。

### 会话 ID 生命周期

1. 客户端发送首个请求时不带 `Mcp-Session-Id`。
2. 服务器分配一个随机 ID，在响应头中设置 `Mcp-Session-Id`。
3. 客户端在之后的所有请求以及用于建立流的 `GET /mcp` 中回传该请求头。
4. 服务器可以撤销会话；客户端在后续请求中收到 404，必须重新初始化。
5. 客户端可以显式 DELETE 会话以实现干净关闭。

### 保活与重连

SSE 连接会断。客户端用同一个 `Mcp-Session-Id` 重新发起 GET 即可恢复。服务器必须把断连期间错过的事件排入队列（在合理的时间窗口内），并依据客户端回传的 `last-event-id` 请求头进行重放。

Phase 13 · 13 会讲解 Tasks，它能让长时间运行的工作在整个会话级别的重连后依然存活。

### 向后兼容探测

一个想同时支持新旧服务器的客户端：

1. 向 `/mcp` 发送 POST。
2. 如果响应是带 JSON 或 SSE 的 `200 OK`，这是 Streamable HTTP。
3. 如果响应是带 `Content-Type: text/event-stream` 的 `200 OK`，并且有指向第二端点的 `Location` 请求头，这是遗留的 HTTP+SSE；跟随 `Location` 即可。

### Cloudflare、ngrok 与托管

2026 年的生产级远程 MCP 服务器运行在 Cloudflare Workers（配合其 MCP Agents SDK）、Vercel Functions 或容器化的 Node/Python 上。关键点：你的托管平台必须支持长连接 HTTP，以承载 SSE 的 GET 请求。Vercel 免费版上限 10 秒，并不合适；Cloudflare Workers 支持无限时长的流。

### 网关组合

当你用网关（Phase 13 · 17）统一接入多个 MCP 服务器时，网关对外就是一个 Streamable HTTP 单端点，它重写会话 ID 并对上游进行多路复用。工具在网关层合并；客户端看到的是一个逻辑上的单一服务器。

### 传输层故障模式

- **stdio SIGPIPE。** 子进程在写入过程中死亡会触发 SIGPIPE；服务器应当干净退出。客户端应检测 EOF 并将会话标记为失效。
- **HTTP 502 / 504。** Cloudflare、nginx 等代理在上游故障时会返回这些状态码。Streamable HTTP 客户端应在短暂退避后重试一次。
- **SSE 连接断开。** TCP RST、代理超时或客户端网络切换都会关闭流。客户端携带 `Mcp-Session-Id` 重连，并可附带 `last-event-id` 来恢复进度。
- **会话撤销。** 服务器使某个会话 ID 失效；客户端在下次请求时收到 404，必须重新握手。
- **时钟偏移。** 客户端的资源 TTL 计算与服务器产生偏差。客户端应以服务器时间戳为准。

### 何时绕开 Streamable HTTP

一些企业在自有网络内部把 MCP 服务器部署在 gRPC 或消息队列传输之上。这不是标准做法——MCP 规范没有正式定义这些传输。网关可以对 MCP 客户端暴露 Streamable HTTP 的外部接口，内部则使用 gRPC。保持对外接口符合规范；翻译工作由网关负责。

## 生产实践

`code/main.py` 用 `http.server`（标准库）实现了一个最小化的 Streamable HTTP 端点。它在 `/mcp` 上处理 POST、GET 和 DELETE，在首个响应中设置 `Mcp-Session-Id`，校验 `Origin`，并拒绝来自允许列表之外来源的请求。该处理器复用了第 07 课笔记服务器的分发逻辑。

值得关注的点：

- POST 处理器读取 JSON-RPC 请求体、分发并写回一个 JSON 响应（这是单响应变体；SSE 变体在结构上类似）。
- `Origin` 检查拒绝默认的 `http://evil.example` 探测请求，但接受 `http://localhost`。
- 会话 ID 是随机的 128 位十六进制字符串；服务器在内存中保存每个会话的状态。

## 交付产物

本课产出 `outputs/skill-mcp-transport-migrator.md`。给定一个 HTTP+SSE（遗留）MCP 服务器，该技能会生成一份迁移到 Streamable HTTP 的迁移方案，涵盖会话 ID 连续性、Origin 检查以及向后兼容的探测支持。

## 练习

1. 运行 `code/main.py`。用 `curl` POST 一个 `initialize` 请求，观察响应头中的 `Mcp-Session-Id`。再 POST 第二个回传该请求头的请求，验证会话连续性。

2. 添加一个打开 SSE 流的 GET 处理器。每五秒发送一条 `notifications/progress` 事件。用同一个会话 ID 重新发起 GET 进行重连，确认服务器接受它。

3. 实现 `last-event-id` 重放逻辑。重连时，重放自该 ID 之后产生的所有事件。

4. 扩展 `Origin` 校验以支持通配符模式（`https://*.example.com`），并确认它接受 `https://app.example.com`，但拒绝 `https://evil.example.com.attacker.net`。

5. 从官方注册中心选一个遗留的 HTTP+SSE 服务器（有不少），勾勒出迁移方案：端点处理、会话 ID 生成和请求头语义各需要做哪些改动。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| stdio 传输 | "本地子进程" | 基于 stdin/stdout 的 JSON-RPC，以换行符分隔 |
| Streamable HTTP | "远程传输" | 单端点的 POST + GET + 可选 SSE，2025-03-26 规范 |
| HTTP+SSE | "遗留传输" | 将在 2026 年年中移除的双端点模型 |
| `Mcp-Session-Id` | "会话请求头" | 服务器分配的随机 ID，在之后每个请求中回传 |
| `Origin` 允许列表 | "DNS 重绑定防御" | 拒绝 Origin 未获批准的请求 |
| 单端点 | "一个 URL" | `/mcp` 处理所有会话操作的 POST / GET / DELETE |
| `last-event-id` | "SSE 重放" | 用于恢复断开的流且不丢失事件的请求头 |
| 向后兼容探测 | "新旧检测" | 客户端通过响应形态检查自动选择传输方式 |
| 长连接 HTTP | "SSE 流式传输" | 服务器在一条 TCP 连接上持续推送事件数分钟乃至数小时 |
| 会话撤销 | "强制重新初始化" | 服务器使会话 ID 失效；客户端必须重新握手 |

## 延伸阅读

- [MCP — Basic transports spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — stdio 与 Streamable HTTP 的权威参考
- [MCP — Basic transports spec 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — 引入 Streamable HTTP 的那一版规范
- [Cloudflare — MCP transport](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — Workers 托管的 Streamable HTTP 模式
- [AWS — MCP transport mechanisms](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — 不同部署形态下的对比
- [Atlassian — HTTP+SSE deprecation notice](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — 一个具体的迁移期限实例
