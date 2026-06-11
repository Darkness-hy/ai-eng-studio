# 为什么需要多智能体？

> 单个智能体会撞上天花板。聪明的做法不是造一个更大的智能体，而是用更多的智能体。

**Type:** Learn
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## 学习目标

- 识别单智能体的能力天花板（上下文溢出、专长混杂、串行瓶颈），并解释什么时候应该拆分成多个智能体
- 比较各种编排模式（流水线、并行扇出、监督者、层级式），并根据任务结构选出合适的那一个
- 设计一个具备清晰角色边界、共享状态和通信契约的多智能体系统
- 分析多智能体复杂性（延迟、成本、调试难度）与单智能体简洁性之间的权衡

## 问题背景

你在 Phase 14 构建了一个单智能体。它能工作：读文件、执行命令、调用 API、对结果进行推理。然后你让它去处理一个真实代码库：200 个文件、三种编程语言、依赖基础设施的测试，还要求在写代码之前先调研外部 API。

智能体卡住了。不是因为 LLM 不够聪明，而是因为任务超出了单个智能体循环的处理能力。上下文窗口被文件内容塞满。智能体忘记了 40 次工具调用之前读到的内容。它试图同时扮演研究员、程序员和审查员，结果三个角色都做得很糟。

这就是单智能体天花板（single-agent ceiling）。每当任务有以下需求时，你就会撞上它：

- **所需上下文超出一个窗口的容量** - 读 50 个文件会轻松突破 200k token
- **不同阶段需要不同的专长** - 调研所需的提示方式与代码生成完全不同
- **工作本可以并行进行** - 既然可以同时读三个文件，为什么要一个接一个地读？

## 核心概念

### 单智能体天花板

单智能体就是一个循环、一个上下文窗口、一个系统提示词。把它画出来：

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

三个地方会出问题：

1. **上下文饱和** - 工具结果不断堆积。到第 30 轮时，智能体已经消耗了 150k token 的文件内容、命令输出和先前的推理。第 5 轮的关键细节就此丢失。

2. **角色混乱** - 一个写着「你是研究员、程序员、审查员兼测试员」的系统提示词，产出的智能体只会研究做一半、代码写一半，审查则永远做不完。

3. **串行瓶颈** - 智能体先读文件 A，再读文件 B，再读文件 C。三次串行的 LLM 调用，三次串行的工具执行。毫无并行可言。

### 多智能体方案

把工作拆开。给每个智能体一项职责、一个上下文窗口，以及一个为该职责量身定制的系统提示词：

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

每个智能体都有：
- 一个聚焦的系统提示词（「你是代码审查员。你唯一的工作就是找 bug。」）
- 自己的上下文窗口（不会被其他智能体的工作污染）
- 一个清晰的输入/输出契约（接收调研笔记，输出代码）

### 真实系统就是这么做的

**Claude Code 子智能体** - 当 Claude Code 通过 `Task` 派生子智能体时，它会创建一个任务范围明确的子智能体。父智能体保持上下文整洁。子智能体专注完成工作并返回摘要。

**Devin** - 运行一个规划智能体、一个编码智能体和一个浏览器智能体。规划者把工作拆解成步骤，编码者写代码，浏览器智能体查阅文档。各自拥有独立的上下文。

**多智能体编程团队（SWE-bench）** - 在 SWE-bench 上表现最好的系统使用一个阅读代码库的研究员、一个设计修复方案的规划者和一个负责实现的程序员。单智能体系统得分更低。

**ChatGPT Deep Research** - 并行派生多个搜索智能体，每个探索一个不同角度，最后综合所有结果。

### 复杂度光谱

多智能体不是非黑即白的选择，而是一条光谱：

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**单智能体（Single agent）** - 一个循环、一个提示词。适合简单任务。

**子智能体（Subagents）** - 父智能体为聚焦的子任务派生子智能体。父智能体掌握整体计划，子智能体完成后汇报。Claude Code 就是这种模式。

**流水线（Pipeline）** - 智能体按顺序运行。智能体 A 的输出作为智能体 B 的输入。适合分阶段的工作流：调研 -> 编码 -> 审查 -> 测试。

**团队（Team）** - 智能体借助共享消息总线并行运行。每个智能体有自己的角色，由一个编排器协调。适合需要同时调用不同技能的场景。

**蜂群（Swarm）** - 许多完全相同或近乎相同的智能体共享状态。没有固定的编排器，智能体从队列里领取任务。适合高吞吐量的并行任务。

### 四种多智能体模式

#### 模式 1：流水线

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

每个智能体对数据做一次变换后传给下一个。逻辑简单、易于推理。但任何一个阶段失败都会阻塞后续阶段。

#### 模式 2：扇出 / 扇入

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

把工作拆分给多个并行智能体，再合并结果。适合能分解成独立子任务的任务。

#### 模式 3：编排器-工作者

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

一个聪明的编排器（orchestrator）决定要做什么，把任务委派给工作者，再综合各方结果。编排器本身也是一个智能体，它的工具就包括派生工作者。

#### 模式 4：对等蜂群

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

没有中心编排器。智能体之间点对点通信，决策从交互中涌现。调试更难，但能扩展到大量智能体。

### 什么时候不该用多智能体

多智能体会增加复杂度。智能体之间的每一条消息都是一个潜在故障点。调试也从「读一段对话」变成「跨五个智能体追踪消息」。

**以下情况请坚持单智能体：**
- 任务能装进一个上下文窗口（工作数据不超过约 100k token）
- 不同阶段不需要不同的系统提示词
- 串行执行的速度已经够用
- 任务足够简单，拆分带来的开销大于收益

**复杂度的代价：**
- 每条智能体边界都是一次有损压缩：智能体 A 的完整上下文会被压缩成一条发给智能体 B 的消息
- 协调逻辑（谁做什么、什么时候做、按什么顺序做）本身就是 bug 的来源
- 延迟上升：N 个智能体意味着至少 N 次串行 LLM 调用，如果它们还需要来回沟通则更多
- 成本成倍增加：每个智能体都独立消耗 token

经验法则：如果一个任务的工具调用少于 20 次、能装进 100k token，就保持单智能体。

```figure
swarm-messages
```

## 从零实现

### 第 1 步：不堪重负的单智能体

下面是一个试图包揽一切的单智能体。它有一个庞大的系统提示词，以及一个同时装着调研、代码和审查的上下文窗口：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种做法的问题：
- 上下文窗口在每个阶段都在膨胀。到审查这一步，它已经同时装着调研笔记、代码和先前的推理。
- 系统提示词是通用的，无法针对每个阶段做调优。
- 没有任何环节是并行的。

### 第 2 步：专家智能体

现在拆分它。每个智能体只负责一件事：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每个专家都有一个聚焦的提示词，并且拿到的是一个只包含所需输入的干净上下文窗口。

### 第 3 步：用消息协调

用显式的消息传递把这些专家串起来：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个智能体只接收发给自己的消息，没有上下文污染。研究员读文档产生的那 50k token 永远不会进入审查员的上下文。

### 第 4 步：对比

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

多智能体版本消耗的总 token 更多（三个智能体，三次独立的 LLM 调用），但每个智能体的上下文都保持干净。由于系统提示词是专门化的，每个阶段的质量都得到提升。

## 生产实践

本课会产出一个可复用的提示词，用于判断何时该转向多智能体。参见 `outputs/prompt-multi-agent-decision.md`。

## 练习

1. 增加第四个专家：一个「测试员」智能体，它接收来自程序员的代码和来自审查员的反馈，然后编写测试
2. 修改流水线，让审查员能把反馈回传给程序员形成修订循环（最多 2 轮）
3. 把串行流水线改造成扇出模式：让研究员和一个「需求分析师」智能体并行运行，合并两者的输出后再传给程序员

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 蜂群（Swarm） | 「AI 智能体组成的蜂巢思维」 | 一组共享状态、没有固定领导者的对等智能体。行为从局部交互中涌现。 |
| 编排器（Orchestrator） | 「老板智能体」 | 一个工具集中包含派生和管理其他智能体能力的智能体。它负责规划和委派，但未必亲自干活。 |
| 协调器（Coordinator） | 「交通警察」 | 一个非智能体组件（通常只是代码，不是 LLM），按规则在智能体之间路由消息。 |
| 共识（Consensus） | 「智能体们达成一致」 | 一种要求多个智能体在继续之前必须达成一致的协议。用于需要化解输出冲突的场景。 |
| 涌现行为（Emergent behavior） | 「智能体自己想明白了」 | 由智能体交互产生、但并未被显式编程的系统级模式。可能有益，也可能有害。 |
| 扇出 / 扇入（Fan-out / fan-in） | 「智能体版的 map-reduce」 | 把任务拆分给并行智能体（扇出），再合并它们的结果（扇入）。 |
| 消息传递（Message passing） | 「智能体互相对话」 | 智能体之间的通信机制：从一个智能体发送给另一个智能体的结构化数据，用来取代共享上下文窗口。 |

## 延伸阅读

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - 多智能体模式综述
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - Microsoft 的多智能体对话框架
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - Claude Code 如何通过 Task 进行委派
- [CrewAI documentation](https://docs.crewai.com/) - 基于角色的多智能体框架
