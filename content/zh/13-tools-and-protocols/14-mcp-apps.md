# MCP Apps —— 通过 `ui://` 提供交互式 UI 资源

> 纯文本的工具输出限制了智能体能够展示的内容。MCP Apps（SEP-1724，于 2026 年 1 月 26 日正式发布）让工具可以返回沙箱化的交互式 HTML，并内嵌渲染在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中。仪表盘、表单、地图、3D 场景，全都通过一个扩展实现。本课将讲解 `ui://` 资源方案、`text/html;profile=mcp-app` MIME 类型、iframe 沙箱的 postMessage 协议，以及允许服务器渲染 HTML 所带来的安全面。

**Type:** Build
**Languages:** Python (stdlib, UI resource emitter), HTML (sample app)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources)
**Time:** ~75 minutes

## 学习目标

- 从工具调用返回 `ui://` 资源，并设置正确的 MIME 类型与元数据。
- 通过 `_meta.ui.resourceUri`、`_meta.ui.csp` 和 `_meta.ui.permissions` 声明工具关联的 UI。
- 实现 iframe 沙箱中用于 UI 与宿主通信的 postMessage JSON-RPC。
- 应用能够防御源自 UI 的攻击的 CSP 与权限策略（permissions-policy）默认值。

## 问题背景

一个 2025 年时代的 `visualize_timeline` 工具只能返回"以下是按时间顺序整理的 14 条笔记：……"。这只是一段文字。用户真正想要的是可交互的时间线。在 MCP Apps 出现之前，可选项只有：客户端专属的 widget API（Claude artifacts、OpenAI Custom GPT HTML），或者干脆没有 UI。

MCP Apps（SEP-1724，2026 年 1 月 26 日发布）把这一契约标准化了。工具结果中包含一个 `resource`，其 URI 为 `ui://...`，MIME 类型为 `text/html;profile=mcp-app`。宿主（host）将其渲染在一个受限 CSP 的沙箱 iframe 中，除非显式授权，否则没有网络访问权限。iframe 内的 UI 通过一种轻量的 postMessage JSON-RPC 方言向宿主发送消息。

每个兼容的客户端（Claude Desktop、ChatGPT、Goose、VS Code）都以相同的方式渲染同一个 `ui://` 资源。一个服务器、一个 HTML 包，UI 通用于所有客户端。

## 核心概念

### `ui://` 资源方案

工具返回：

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

随后宿主对 `ui://notes/timeline` 这个 URI 调用 `resources/read`，得到：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### iframe 沙箱

宿主在一个沙箱化的 `<iframe>` 中渲染该 HTML，并施加：

- `sandbox="allow-scripts allow-same-origin"`（或按服务器声明施加更严格的限制）
- 通过响应头应用服务器声明的 CSP。
- 无法访问宿主源（origin）的 cookie 和 localStorage。
- 网络访问受限于 CSP 中的 `connectSrc`。

### postMessage 协议

iframe 通过 `window.postMessage` 与宿主通信，使用一种轻量的 JSON-RPC 2.0 方言：

始终将 `targetOrigin` 固定为对端的精确源，并在接收端先用允许列表校验 `event.origin`，再处理任何载荷。这条通道的两端都绝不能使用 `"*"` —— 消息体承载着工具调用和资源读取。

```js
// iframe to host  (pin to host origin)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host to iframe  (pin to iframe origin)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// receiver on both sides
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});
```

UI 可以调用的宿主侧方法包括：

- `host.callTool(name, arguments)` —— 调用服务器端的工具。
- `host.readResource(uri)` —— 读取一个 MCP 资源。
- `host.getPrompt(name, arguments)` —— 获取提示词模板。
- `host.close()` —— 关闭 UI。

每次调用仍然走 MCP 协议，并继承服务器的权限。

### 权限

`_meta.ui.permissions` 列表用于申请额外能力：

- `camera` —— 访问用户的摄像头（用于"扫描文档"类 UI）。
- `microphone` —— 语音输入。
- `geolocation` —— 地理位置。
- `network:*` —— 比单独的 `connectSrc` 更宽的网络访问。

每项权限都会在 UI 渲染前向用户弹出确认提示。

### 安全风险

iframe 里的 HTML 终究还是 HTML。新增的攻击面包括：

- **通过 UI 进行提示注入（prompt injection）。** 恶意服务器的 UI 可以展示看起来像系统消息的文字来欺骗用户。宿主在渲染时应当让服务器 UI 与宿主 UI 在视觉上明显区分。
- **通过 `connectSrc` 外泄数据。** 如果 CSP 允许 `connect-src: *`，UI 就能把数据发往任何地方。默认值应当从严。
- **点击劫持（clickjacking）。** UI 覆盖在宿主界面之上。宿主必须阻止 z-index 操纵并强制执行不透明度规则。
- **窃取焦点。** UI 抢占键盘焦点并截获用户的下一条消息。宿主必须拦截这种行为。

Phase 13 · 15 会在 MCP 安全的部分深入讲解这些内容；本课先做引入。

### `ui/initialize` 握手

iframe 加载完成后，会通过 postMessage 发送 `ui/initialize`：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

宿主返回能力声明和一个会话令牌（session token）。UI 在之后每次调用宿主时都要带上这个会话令牌。

### AppRenderer / AppFrame SDK 原语

ext-apps SDK 提供了两个便捷原语：

- `AppRenderer`（服务器侧）—— 包装一个 React / Vue / Solid 组件，并以正确的 MIME 类型和元数据发出 `ui://` 资源。
- `AppFrame`（客户端侧）—— 接收该资源、挂载 iframe，并居中转发 postMessage。

你可以使用它们，也可以手写 HTML 和 JSON-RPC。

### 生态现状

MCP Apps 于 2026 年 1 月 26 日发布。截至 2026 年 4 月的客户端支持情况：

- **Claude Desktop。** 自 2026 年 1 月起完整支持。
- **ChatGPT。** 通过 Apps SDK 完整支持（底层是同一套 MCP Apps 协议）。
- **Cursor。** Beta 阶段；需在设置中开启。
- **VS Code。** 仅限 Insider 构建版本。
- **Goose。** 完整支持。
- **Zed、Windsurf。** 已列入路线图。

已投入生产的服务器：仪表盘、地图可视化、数据表格、图表构建器、沙箱 IDE 预览。

## 生产实践

`code/main.py` 在笔记服务器的基础上扩展出一个 `visualize_timeline` 工具，返回一个 `ui://notes/timeline` 资源；同时为该 URI 实现了 `resources/read` 处理器，返回一个小而完整的带 SVG 时间线的 HTML 包。HTML 用标准库模板生成 —— 没有构建系统。由于标准库无法驱动浏览器，postMessage 部分以 JS 注释的形式给出示意。

值得关注的点：

- 工具响应上的 `_meta.ui` 携带 resourceUri、CSP 和权限。
- HTML 在无网络访问的情况下完成渲染；所有数据都内联其中。
- JS 通过 `window.parent.postMessage` 调用 `host.callTool`（在这个标准库 demo 中有文档说明但不会真正执行）。

## 交付产物

本课产出 `outputs/skill-mcp-apps-spec.md`。给定一个适合配上交互式 UI 的工具，该技能会产出完整的 MCP Apps 契约：`ui://` URI、CSP、权限、postMessage 入口点，以及一份安全检查清单。

## 练习

1. 运行 `code/main.py` 并检查其输出的 HTML。直接在浏览器中打开该 HTML，确认 SVG 能正常渲染。然后草拟该 UI 调用 `host.callTool("notes_update", ...)` 时会使用的 postMessage 契约。

2. 收紧 CSP：移除 `'unsafe-inline'`，改用基于 nonce 的脚本策略。HTML 生成代码需要做哪些改动？

3. 添加第二个 UI 资源 `ui://notes/editor`，提供一个就地编辑笔记的表单。用户提交时，iframe 调用 `host.callTool("notes_update", ...)`。

4. 审计该 UI 的攻击面。恶意服务器可以在哪些位置注入内容？iframe 沙箱能防御什么，不能防御什么？

5. 阅读 SEP-1724 规范，找出 MCP Apps SDK 中一项本课玩具实现没有用到的能力。（提示：组件级状态同步。）

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| MCP Apps | "交互式 UI 资源" | 于 2026-01-26 发布的 SEP-1724 扩展 |
| `ui://` | "App URI 方案" | 用于 UI 包的资源方案 |
| `text/html;profile=mcp-app` | "那个 MIME" | MCP App HTML 的内容类型 |
| iframe 沙箱 | "渲染容器" | 浏览器对 UI 的沙箱化，附带 CSP 和权限 |
| postMessage JSON-RPC | "UI 到宿主的传输线" | 用于调用宿主的轻量 JSON-RPC-over-postMessage 方言 |
| `_meta.ui` | "工具-UI 绑定" | 将工具结果关联到 UI 资源的元数据 |
| CSP | "Content-Security-Policy" | 声明脚本、网络、样式的允许来源 |
| AppRenderer | "服务器侧 SDK 原语" | 把框架组件转换为 `ui://` 资源 |
| AppFrame | "客户端侧 SDK 原语" | 挂载 iframe 并居中转发 postMessage 的辅助组件 |
| `ui/initialize` | "握手" | UI 发往宿主的第一条 postMessage |

## 延伸阅读

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) —— 参考实现与 SDK
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) —— 正式规范文档
- [MCP — Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview) —— 高层文档
- [MCP blog — MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) —— 2026 年 1 月的发布博文
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/) —— JSDoc 风格的 SDK 参考
