# Roots 与 Elicitation —— 作用域限定与执行中途的用户输入

> 硬编码路径在用户打开另一个项目的那一刻就会失效。预填好的工具参数在用户描述不充分时也会失效。Roots（根目录）把服务器的作用域限定在一组由用户控制的 URI 内；elicitation（征询）则在工具调用执行中途暂停，通过表单或 URL 向用户请求结构化输入。两个客户端原语，解决 MCP 的两类常见故障模式。SEP-1036（URL 模式 elicitation，2025-11-25）在 2026 年上半年仍处于实验阶段——依赖它之前请先检查 SDK 版本。

**Type:** Build
**Languages:** Python (stdlib, roots + elicitation demo)
**Prerequisites:** Phase 13 · 07 (MCP server)
**Time:** ~45 minutes

## 学习目标

- 声明 `roots` 并响应 `notifications/roots/list_changed`。
- 将服务器的文件操作限制在已声明的根集合内的 URI。
- 使用 `elicitation/create` 在工具调用执行中途向用户请求确认或结构化输入。
- 在表单模式与 URL 模式 elicitation 之间做出选择（后者尚属实验性，存在已注明的漂移风险）。

## 问题背景

一个笔记类 MCP 服务器在生产环境中会遇到的两个具体故障。

**路径假设失效。** 服务器是针对 `~/notes` 编写的。换一台机器、笔记存放在 `~/Documents/Notes` 的用户，工具调用会静默失败（找不到文件），更糟的情况是写到了错误的位置。

**缺少一个用户本来知道的参数。** 用户说"删除那个旧的 TPS 报告笔记"。模型调用了 `notes_delete(title: "TPS report")`，但有三条匹配的笔记，分别来自 2023、2024 和 2025 年。工具没法猜。报"歧义"错误失败令人烦躁；三条全删则是灾难。

Roots 解决第一个问题：客户端在 `initialize` 时声明服务器允许触碰的 URI 集合。Elicitation 解决第二个问题：服务器暂停工具调用，发送 `elicitation/create` 请用户选出是哪一条。

## 核心概念

### Roots

客户端在 `initialize` 时声明根列表：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

之后服务器可以调用 `roots/list`：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

服务器必须（MUST）把 roots 当作边界：根集合之外的任何文件读写都应被拒绝。这一点并非由客户端强制执行（服务器仍然是用户选择信任的代码），但符合规范的服务器会遵守它。

当用户添加或移除一个根时，客户端会发送 `notifications/roots/list_changed`。服务器重新调用 `roots/list` 并更新自己的边界。

### 为什么 roots 是客户端原语

Roots 由客户端声明，因为它们代表的是用户的授权模型。是用户告诉 Claude Desktop"让这个笔记服务器访问这两个目录"。服务器无权擅自扩大这个范围。

### Elicitation：默认的表单模式

`elicitation/create` 接收一个表单 schema 加一段自然语言提示：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

客户端渲染表单，收集用户的回答，然后返回：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

三种可能的动作：`accept`（用户填写了表单）、`decline`（用户关闭了表单）、`cancel`（用户中止了整个工具调用）。

表单 schema 是扁平的——v1 不支持嵌套对象。SDK 通常会拒绝任何比单层结构更复杂的 schema。

### Elicitation：URL 模式（SEP-1036，实验性）

2025-11-25 新增。服务器不发送 schema，改为发送一个 URL：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

客户端在浏览器中打开该 URL，等待流程完成，待用户返回后再继续。适用于 OAuth 流程、支付授权、文档签署等表单力所不及的场景。

漂移风险提示：SEP-1036 的响应格式仍在变动；有的 SDK 返回回调 URL，有的返回完成令牌。在生产环境使用 URL 模式之前，请先阅读你所用 SDK 的发布说明。

### 何时该用 elicitation

- 破坏性操作前的用户确认（destructive 提示 + elicitation）。
- 消歧（从 N 个匹配项中选一个）。
- 首次运行的初始化设置（API key、目录、偏好设置）。
- OAuth 类流程（URL 模式）。

### 何时不该用 elicitation

- 填充模型本可以在对话中直接询问的工具必填参数。用普通的重新提问，而不是 elicitation 对话框。
- 高频调用。Elicitation 会打断对话；不要在循环里触发它。
- 任何服务器可以事后校验的内容。先校验，返回错误，让模型以文字形式去问用户。

### 人在回路的桥梁

Elicitation 与采样（sampling）结合，共同支撑起 MCP 的"人在回路"（human-in-the-loop）模型。服务器的智能体循环可以暂停下来，要么等待用户输入（elicitation），要么等待模型推理（sampling）。Phase 13 · 11 讲过采样；本课讲 elicitation。把两者结合起来，就能完整掌控循环中途的流程。

## 生产实践

`code/main.py` 在笔记服务器的基础上扩展了：

- 收到 root-list-changed 通知后，服务器会重新查询 `roots/list` 的响应。
- 一个 `notes_delete` 工具，在多条笔记匹配时使用 `elicitation/create` 进行消歧。
- 一个 `notes_setup` 工具，使用 URL 模式 elicitation 打开首次运行的配置页面（模拟）。
- 一个边界检查，拒绝对已声明 roots 之外的 URI 进行操作。

演示运行三个场景：正常路径（单条匹配）、消歧（三条匹配，触发 elicitation）、根外写入（被拒绝）。

## 交付产物

本课产出 `outputs/skill-elicitation-form-designer.md`。给定一个可能需要用户确认或消歧的工具，该 skill 会设计相应的 elicitation 表单 schema 和消息模板。

## 练习

1. 运行 `code/main.py`。触发消歧路径；确认模拟的用户回答被正确路由回工具。

2. 新增一个 `notes_archive` 工具，要求每次都通过 elicitation 确认（destructive 提示）。检查用户体验：这与模型用文字重新追问相比如何？

3. 为首次运行的 OAuth 流程实现 URL 模式 elicitation。注意漂移风险，并加上 SDK 版本守卫。

4. 扩展 `roots/list` 的处理逻辑：通知到达时，服务器应原子化地重新读取并重新扫描那些可能已超出作用域的已打开文件句柄。

5. 阅读 GitHub 上 SEP-1036 的 issue 讨论串。找出一个会影响服务器处理 URL 模式回调方式的未决问题。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Root | "授权边界" | 客户端允许服务器触碰的 URI |
| `roots/list` | "服务器询问作用域" | 客户端返回当前的根集合 |
| `notifications/roots/list_changed` | "用户改变了作用域" | 客户端发出信号：根集合已发生变更 |
| Elicitation | "调用中途问用户" | 由服务器发起的结构化用户输入请求 |
| `elicitation/create` | "那个方法" | 用于 elicitation 请求的 JSON-RPC 方法 |
| 表单模式 | "schema 驱动的表单" | 扁平的 JSON Schema，在客户端 UI 中渲染为表单 |
| URL 模式 | "浏览器跳转" | SEP-1036 实验性特性；打开一个 URL 并等待 |
| `accept` / `decline` / `cancel` | "用户响应的几种结果" | 服务器需要处理的三个分支 |
| 消歧 | "选一个" | 工具有 N 个候选项时常见的 elicitation 用例 |
| 扁平表单 | "只有顶层属性" | Elicitation 的 schema 不能嵌套 |

## 延伸阅读

- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) —— roots 的权威参考
- [MCP — Client elicitation spec](https://modelcontextprotocol.io/specification/draft/client/elicitation) —— elicitation 的权威参考
- [Cisco — What's new in MCP elicitation, structured content, OAuth enhancements](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) —— 2025-11-25 新增内容讲解
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) —— URL 模式 elicitation 提案（实验性，有漂移风险）
- [The New Stack — How elicitation brings human-in-the-loop to AI tools](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) —— 用户体验讲解
