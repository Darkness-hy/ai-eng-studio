# A2A —— 智能体间协议（Agent-to-Agent Protocol）

> MCP 解决的是智能体到工具的连接，而 A2A（Agent2Agent）解决的是智能体到智能体的连接——这是一个开放协议，让基于不同框架构建的不透明智能体能够相互协作。该协议由 Google 于 2025 年 4 月发布，2025 年 6 月捐赠给 Linux Foundation，2026 年 4 月发布 v1.0，拥有 150 多家支持方，包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow。它吸收了 IBM 的 ACP，并新增了 AP2 支付扩展。本课将逐一讲解 Agent Card、Task 生命周期以及两种传输绑定。

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## 学习目标

- 区分智能体到工具（MCP）与智能体到智能体（A2A）的使用场景。
- 在 `/.well-known/agent.json` 发布带有技能和端点元数据的 Agent Card。
- 走通 Task 生命周期（submitted → working → input-required → completed / failed / canceled / rejected）。
- 使用包含 Parts（text、file、data）的 Message，并以 Artifact 作为输出。

## 问题背景

一个客服智能体需要把报告撰写工作委派给一个专门的写作智能体。在 A2A 出现之前，可选方案有：

- 自定义 REST API。能用，但每一对智能体的对接都是一次性的定制工作。
- 共享代码库。要求两个智能体运行在同一个框架上。
- MCP。并不合适：MCP 是用来调用工具的，而不是让两个智能体在各自保持内部推理不透明的前提下协作。

A2A 填补了这一空白。它把交互建模为一个智能体向另一个智能体发送 Task，并配有生命周期、消息和产物（artifact）。被调用智能体的内部状态保持不透明——调用方只能看到任务的状态变迁和最终输出。

A2A 是那个「让跨框架的智能体彼此对话」的协议。它并不取代 MCP；两者是互补关系。

## 核心概念

### Agent Card

每个符合 A2A 规范的智能体都会在 `/.well-known/agent.json` 发布一张卡片：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现机制基于 URL：抓取卡片，得知 A2A 端点的 URL，再枚举技能。

### 签名 Agent Card（AP2）

AP2 扩展（2025 年 9 月）为 Agent Card 增加了密码学签名。发布方用 JWT 给自己的卡片签名；消费方进行验证。这可以防止冒充。

### Task 生命周期

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

客户端通过 `tasks/send` 发起任务。被调用智能体在各状态间流转；客户端通过 SSE 订阅状态更新，或采用轮询。

### Message 与 Part

一条消息携带一个或多个 Part：

- `text` —— 纯文本内容。
- `file` —— 带 mimeType 的 base64 二进制数据。
- `data` —— 有类型的 JSON 载荷（提供给被调用智能体的结构化输入）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifact

输出是 Artifact，而不是裸字符串。Artifact 是一个有名称、有类型的输出：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifact 可以按分块（chunk）流式传输，由调用方累积拼接。

### 两种传输绑定

1. **JSON-RPC over HTTP。** `/a2a` 端点，用 POST 发送请求，可选 SSE 实现流式传输。这是默认绑定。
2. **gRPC。** 面向 gRPC 已是原生选择的企业环境。

两种绑定承载的逻辑消息结构完全相同。

### 不透明性保障

一个关键设计原则：被调用智能体的内部状态是不透明的。调用方只能看到任务状态和产物。被调用智能体的思维链、工具调用、子智能体委派——全部不可见。这与 MCP 不同，MCP 中的工具调用是透明的。

设计依据：A2A 让竞争对手之间也能协作，而无需暴露内部实现。A2A 可以表达「调用这个客服智能体」，调用方却无从得知该智能体如何实现这项服务。

### 时间线

- **2025-04-09。** Google 宣布 A2A。
- **2025-06-23。** 捐赠给 Linux Foundation。
- **2025-08。** 吸收 IBM 的 ACP。
- **2025-09。** AP2 扩展（Agent Payments）发布。
- **2026-04。** v1.0 发布，获得 150 多家组织支持。

### 与 MCP 的关系

| 维度 | MCP | A2A |
|-----------|-----|-----|
| 使用场景 | 智能体到工具 | 智能体到智能体 |
| 不透明性 | 工具调用透明 | 内部推理不透明 |
| 典型调用方 | 智能体运行时 | 另一个智能体 |
| 状态 | 工具调用结果 | 带生命周期的 Task |
| 授权 | OAuth 2.1（Phase 13 · 16） | JWT 签名的 Agent Card（AP2） |
| 传输 | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

当你想调用某个具体工具时，用 MCP。当你想把一整项任务委派给另一个智能体时，用 A2A。许多生产系统两者并用：智能体用 MCP 构建工具层，用 A2A 构建协作层。

## 生产实践

`code/main.py` 实现了一个最小化的 A2A 测试框架：一个研究智能体发布其卡片，一个写作智能体接收包含 PDF 和文本指令两类 part 的 `tasks/send`，依次经历 working → input_required → working → completed 状态变迁，并返回一个文本 Artifact。全部使用标准库；采用内存传输，以聚焦消息结构本身。

值得关注的点：

- Agent Card 的 JSON 结构。
- Task id 的分配与状态变迁。
- 携带混合类型 part 的消息。
- 任务中途出现的 input-required 分支。
- 完成时返回的 Artifact。

## 交付产物

本课产出 `outputs/skill-a2a-agent-spec.md`。给定一个需要被其他智能体调用的新智能体，该 skill 会生成 Agent Card JSON、技能 schema 和端点蓝图。

## 练习

1. 运行 `code/main.py`。追踪完整的 Task 生命周期，包括被调用智能体请求澄清时的 input-required 暂停。

2. 添加一张签名 Agent Card。对卡片的规范化 JSON 做 HMAC 签名。编写一个验证器，并确认它在卡片被篡改时验证失败。

3. 实现任务流式传输：写作智能体通过 SSE 发出三个增量 Artifact 分块，由调用方累积拼接。

4. 设计一个包装 MCP 服务器的 A2A 智能体。把每个 MCP 工具映射为一个 A2A 技能。记录其中的权衡——损失了哪些不透明性？

5. 阅读 A2A v1.0 发布公告，找出截至 2026 年 4 月尚无任何框架实现的那个特性。（提示：它与多跳任务委派有关。）

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| A2A | 「智能体间协议」 | 用于不透明智能体协作的开放协议 |
| Agent Card | 「`.well-known/agent.json`」 | 发布的元数据，描述智能体的技能和端点 |
| Skill | 「一个可调用单元」 | 智能体支持的具名操作（类比 MCP 工具） |
| Task | 「委派的基本单位」 | 带有生命周期和最终产物的工作项 |
| Message | 「Task 输入」 | 携带 Parts（text、file、data） |
| Part | 「有类型的分块」 | 消息中的 `text` / `file` / `data` 元素 |
| Artifact | 「Task 输出」 | 任务完成时返回的具名、有类型的输出 |
| AP2 | 「Agent Payments Protocol」 | 用于信任与支付的签名 Agent Card 扩展 |
| Opacity | 「黑盒协作」 | 被调用智能体的内部对调用方隐藏 |
| Input-required | 「任务暂停」 | 智能体需要更多信息时所处的生命周期状态 |

## 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) —— A2A 官方规范
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) —— 参考实现与 SDK
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) —— 2025 年 6 月治理权移交
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) —— 路线图与合作伙伴进展
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) —— v1.0 发布说明与向后兼容指南
