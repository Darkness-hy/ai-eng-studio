# A2A —— 智能体间通信协议（Agent-to-Agent Protocol）

> Google 于 2025 年 4 月发布 A2A；到 2026 年 4 月，规范已发布于 https://a2a-protocol.org/latest/specification/，并获得 150 多家组织的支持。A2A 是 MCP（第 13 课）的横向补充：MCP 是纵向的（智能体 ↔ 工具），而 A2A 是点对点的（智能体 ↔ 智能体）。它定义了 Agent Card（用于发现）、带产物（artifact）的任务（文本、结构化数据、视频）、不透明的任务生命周期，以及鉴权机制。生产系统越来越多地将 MCP 与 A2A 搭配使用。Google Cloud 已在 2025-2026 年间将 A2A 支持纳入 Vertex AI Agent Builder。

**Type:** Learn + Build
**Languages:** Python (stdlib, `http.server`, `json`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## 问题背景

你的智能体需要调用另一个系统上的另一个智能体。怎么做？你可以暴露一个 HTTP 端点，定义一套专用的 JSON schema，然后指望对方能看懂。这样一来，每一对智能体之间都变成了一次定制集成。

A2A 就是这类调用的通用线上协议（wire protocol）。标准化的发现机制、标准化的任务模型、标准化的传输方式、标准化的产物格式。就像 HTTP+REST，但把智能体当作一等公民。

## 核心概念

### 四个核心要素

**Agent Card。** 位于 `/.well-known/agent.json` 的一份 JSON 文档，用于描述智能体：名称、技能、端点、支持的模态、鉴权要求。读取这张卡片即可完成发现。

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**任务（Task）。** 工作的基本单元。一个异步、有状态的对象，带有生命周期：`submitted → working → completed / failed / canceled`。客户端发送任务后，通过轮询或订阅获取更新。

**产物（Artifact）。** 任务产出的结果类型。文本、结构化 JSON、图像、视频、音频。产物带有类型，因此不同模态都是一等公民。

**不透明生命周期（Opaque lifecycle）。** A2A 不规定远端智能体*如何*完成任务。客户端只能看到状态转换和产物；具体实现可以自由选用任何框架。

### MCP 与 A2A 的分工

- **MCP**（第 13 课）：智能体 ↔ 工具。智能体通过 JSON-RPC 对工具服务器进行读写。默认无状态。
- **A2A**：智能体 ↔ 智能体。对等协议；两端都是拥有自身推理能力的智能体。

生产环境的多智能体系统两者都用。A2A 的对等方在自己这一侧调用 MCP 工具。这种分工让两类关注点保持清晰。

### 发现流程

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

或者使用流式方式：通过 SSE 订阅 `/tasks/{id}/events` 获取推送更新。

### 鉴权

A2A 支持三种常见模式：

- **Bearer token** —— OAuth2 或不透明令牌。
- **mTLS** —— 双向 TLS；组织之间互相证明身份。
- **签名请求** —— 对载荷做 HMAC 签名。

鉴权方式在 Agent Card 中声明；客户端发现后照此执行。

### 到 2026 年 4 月已有 150 多家组织支持

企业级采用推动了 A2A 的规模化。核心事实是：A2A 成为企业智能体系统跨越信任边界的标准方式。Google Cloud 在 Vertex AI Agent Builder 中提供了 A2A 支持；Microsoft Agent Framework 也支持它；多数主流框架（LangGraph、CrewAI、AutoGen）都提供了 A2A 适配器。

### A2A 的优势场景

- **跨组织调用。** A 公司的智能体调用 B 公司的智能体。没有 A2A，每一对组合都得签一份定制契约。
- **异构框架。** LangGraph 智能体调用 CrewAI 智能体，再调用自研 Python 智能体。A2A 把它们统一起来。
- **带类型的产物。** 视频结果、结构化 JSON、音频——全都是一等公民。
- **长时运行的任务。** 不透明生命周期加轮询，让运行数小时的任务变得简单直接。

### A2A 的不足场景

- **对延迟敏感的微型调用。** A2A 的生命周期是异步的。亚毫秒级的智能体间调用不适合它；请改用直接 RPC。
- **紧耦合的进程内智能体。** 如果两个智能体跑在同一个 Python 进程里，A2A 的 HTTP 往返就是杀鸡用牛刀。
- **小团队。** 规范带来的开销是实实在在的；仅供内部使用的智能体可能不需要这种正式程度。

### A2A 对比 ACP、ANP、NLIP

2024-2026 年间出现了几个相关规范：

- **ACP**（IBM/Linux Foundation）—— A2A 的前身，范围更窄。
- **ANP**（Agent Network Protocol）—— 侧重点对点发现，去中心化优先。
- **NLIP**（Ecma Natural Language Interaction Protocol，2025 年 12 月完成标准化）—— 以自然语言为内容类型。

截至 2026 年 4 月，A2A 是被采用最广泛的对等协议。对比分析参见 arXiv:2505.02279（Liu et al., "A Survey of Agent Interoperability Protocols"）。

## 从零实现

`code/main.py` 用 `http.server` 和 JSON 实现了一个最小化的 A2A 服务器和客户端。服务器：

- 暴露 `/.well-known/agent.json`，
- 接受 `POST /tasks`，
- 管理任务状态，
- 在 `GET /tasks/{id}` 时返回产物。

客户端：

- 获取 Agent Card，
- 提交任务，
- 轮询直至完成，
- 读取产物。

运行：

```
python3 code/main.py
```

脚本在后台线程中启动服务器，然后用客户端对其发起请求。你将看到完整流程：发现、提交、轮询、产物。

## 生产实践

`outputs/skill-a2a-integrator.md` 设计了一套 A2A 集成方案：Agent Card 内容、任务 schema、鉴权选型、流式还是轮询。

## 交付产物

检查清单：

- **锁定规范版本。** A2A 仍在演进；Agent Card 应声明协议版本。
- **任务创建幂等化。** 重复提交（网络重试）应当只产生一个任务。
- **产物 schema。** 声明智能体会返回哪些数据形状；消费方应做校验。
- **限流 + 鉴权。** A2A 面向公网；应用标准 Web 安全措施。
- **失败任务的死信队列。** 长期观察失败模式，识别反复出现的故障类型。

## 练习

1. 运行 `code/main.py`。确认客户端能发现服务器并收到正确的产物。
2. 给服务器添加第二个技能（例如 "summarize"）。更新 Agent Card。编写一个根据任务类型选择技能的客户端。
3. 实现一个 SSE 流式端点：`/tasks/{id}/events`，用于发送状态变更。客户端需要做哪些不同的处理？
4. 阅读 A2A 规范（https://a2a-protocol.org/latest/specification/）。找出三处规范强制要求、但本示例没有实现的内容。
5. 比较 A2A（通过 Agent Card 发现）与 MCP（通过 `listTools` 在服务端列举能力）。自描述智能体与能力探测之间的权衡是什么？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| A2A | "Agent-to-agent" | 让智能体跨系统调用其他智能体的对等协议。Google 2025 年发布。 |
| Agent Card | "智能体的名片" | 位于 `/.well-known/agent.json` 的 JSON，描述技能、端点、鉴权。 |
| 任务（Task） | "工作的基本单元" | 带生命周期的异步有状态对象；完成时产出产物。 |
| 产物（Artifact） | "结果" | 带类型的输出：文本、结构化 JSON、图像、视频、音频。多媒体是一等公民。 |
| 不透明生命周期 | "怎么解决是智能体自己的事" | 客户端只看到状态转换；服务端可自由选择框架和工具。 |
| 发现（Discovery） | "找到智能体" | `GET /.well-known/agent.json` 返回卡片。 |
| MCP vs A2A | "工具 vs 对等方" | MCP：纵向，智能体 ↔ 工具。A2A：横向，智能体 ↔ 智能体。 |
| ACP / ANP / NLIP | "兄弟协议" | 相邻规范；A2A 是 2026 年采用最广的一个。 |

## 延伸阅读

- [A2A specification](https://a2a-protocol.org/latest/specification/) —— 官方规范
- [Google Developers Blog — A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) —— 2025 年 4 月的发布文章
- [A2A GitHub repo](https://github.com/a2aproject/A2A) —— 参考实现与 SDK
- [Liu et al. — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1) —— MCP、ACP、A2A、ANP 对比
