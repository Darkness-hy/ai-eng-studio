# 人在回路：先提议后提交

> 2026 年关于 HITL 的共识是非常具体的。它不是"代理发问、用户点一下批准"，而是先提议后提交（propose-then-commit）：提议的操作连同幂等键（idempotency key）一起持久化到耐久存储；连同意图、数据来源链路、涉及的权限、影响范围和回滚方案一并呈现给审查者；只有在得到明确确认后才提交执行；执行之后还要验证，确认副作用确实发生了。LangGraph 的 `interrupt()` 加 PostgreSQL 检查点、Microsoft Agent Framework 的 `RequestInfoEvent`、Cloudflare 的 `waitForApproval()`，实现的都是同一套形态。最典型的失败模式是橡皮图章式批准：用户没有真正审查就点了"批准"。有文档支持的缓解手段是带明确清单的质询应答（challenge-and-response）。

**Type:** Learn
**Languages:** Python (stdlib, propose-then-commit state machine with idempotency)
**Prerequisites:** Phase 15 · 12 (Durable execution), Phase 15 · 14 (Tripwires)
**Time:** ~60 minutes

## 问题背景

代理要执行一个操作，用户必须做出决定：批准还是不批准。如果这个决定是瞬间做出的，那多半不算审查；如果决定是结构化的，则虽然慢，但值得信赖。工程上的问题是：如何让结构化审查成为阻力最小的路径。

2023 年的 HITL 模式是一个同步提示："代理想向 X 发送正文为 Y 的邮件——批准吗？"用户点击批准，所有人都觉得系统很安全。但实践中这个界面被大量橡皮图章式地通过：用户批得很快，批准本身几乎不能预测任何东西，而当代理出问题时，审计记录里是一长串用户根本想不起来的批准。

2026 年的模式——先提议后提交——把 HITL 搬到耐久基础设施上，附带结构化元数据，并要求明确的提交动作。每个托管代理 SDK 都提供了自己的版本：LangGraph 的 `interrupt()`、Microsoft Agent Framework 的 `RequestInfoEvent`、Cloudflare 的 `waitForApproval()`。API 名字各不相同，形态却完全一致。

## 核心概念

### 先提议后提交的状态机

1. **提议（Propose）。** 代理生成一个提议的操作，持久化到耐久存储（PostgreSQL、Redis、Durable Object）。其中包含：
   - 意图（代理为什么要做这件事）
   - 数据来源链路（是什么来源导致了这个提议）
   - 涉及的权限（哪些作用域 / 文件 / 端点）
   - 影响范围（最坏情况是什么）
   - 回滚方案（一旦提交，如何撤销）
   - 幂等键（每个提议唯一；重复提交返回同一条记录）
2. **呈现（Surface）。** 审查者看到附带全部元数据的提议。审查者必须是人（不能是代理审查自己）。
3. **提交（Commit）。** 明确确认。操作执行。
4. **验证（Verify）。** 执行之后，回读副作用并加以确认。如果验证步骤失败，系统进入一个已知的异常状态，并触发告警。

### 幂等键

没有幂等键，瞬时故障后的重试可能让已批准的操作执行两次。举个具体例子：用户批准了"从 A 向 B 转账 100 美元"。网络抖动了一下，工作流重试。用户只批准过一次，转账却执行了两次。幂等键把这次批准绑定到唯一的一个副作用上；第二次执行就是空操作（no-op）。

这与 Stripe 和 AWS API 使用的幂等模式完全相同。把它复用到代理审批上，在 Microsoft Agent Framework 的文档里有明确说明。

### 耐久性：为什么审批要比进程活得久

审批等待区是一块代理并不拥有的状态。工作流被暂停（第 12 课）。当批准到达时，工作流恰好从那个点恢复。这就是 LangGraph 把 `interrupt()` 与 PostgreSQL 检查点配对、而不是只用内存状态的原因——两天后才到来的批准，仍然能找到完好无损的工作流。

### 橡皮图章式批准与质询应答缓解手段

HITL 的默认界面（"批准" / "拒绝"按钮）带来的是快速批准，却没有真正的审查。有文档支持的缓解手段是：一份质询应答清单，要求审查者对特定问题给出肯定回答之后，批准按钮才会启用。具体形式如下：

- "你是否理解这个操作会触碰哪个资源？[ ]"
- "你是否已确认影响范围可以接受？[ ]"
- "如果失败了，你是否有回滚方案？[ ]"

这不是为了官僚主义而官僚主义，而是一种强制机制。勾不上这些选项的审查者，要么去寻求澄清（升级处理），要么直接拒绝（安全的默认选项）。Anthropic 的代理安全研究明确指出，清单驱动的 HITL 是针对橡皮图章式批准模式的缓解手段。

### 什么才算"有后果的"操作

并非每个操作都需要先提议后提交。2026 年的指导原则是：

- **有后果的操作**（始终需要 HITL）：不可逆的写入、金融交易、对外通信、生产数据库变更、破坏性的文件系统操作。
- **可逆操作**（有时需要 HITL）：对本地文件的编辑、staging 环境的变更、有清晰回滚路径的可逆写入。
- **读取与检查**（从不需要 HITL）：读取文件、列出资源、调用只读 API。

### 操作后验证

"提交跑完了"不等于"副作用发生了"。网络分区和竞态条件可能造成工作流自认为成功、而后端并未持久化的局面。验证步骤在提交之后重新读取目标资源以确认结果。这与数据库事务里的 `RETURNING` 子句、或者 AWS 在 `PutObject` 之后调用 `GetObject` 是同一种模式。

### 欧盟 AI 法案第 14 条

第 14 条要求欧盟境内的高风险 AI 系统具备有效的人类监督。"有效"不是装饰性的。监管措辞明确排除了橡皮图章式的模式。在 Microsoft Agent Governance Toolkit 的合规文档中，带质询应答的先提议后提交正是能经受第 14 条审查的那种形态。

## 生产实践

`code/main.py` 用纯标准库 Python 实现了一个先提议后提交的状态机。耐久存储是一个 JSON 文件。幂等键是 (thread_id, action_signature) 的哈希。驱动程序模拟了三种情形：一次干净的审批流程、一次瞬时故障后的重试（不得重复执行）、以及橡皮图章式默认流程与质询应答流程的对比。

## 交付产物

`outputs/skill-hitl-design.md` 审查一个被提议的 HITL 工作流是否符合先提议后提交的形态，并标记出缺失的元数据、幂等性、验证步骤或质询应答层。

## 练习

1. 运行 `code/main.py`。确认对已批准提议的重试会使用耐久记录而不会重新执行。然后修改幂等键，把时间戳也包含进去，演示重试导致重复执行。

2. 给提议记录扩展一个 `rollback` 字段。模拟一次验证步骤失败的执行，演示回滚自动触发。

3. 阅读 Microsoft Agent Framework 的 `RequestInfoEvent` 文档。找出该 API 包含、而这个玩具引擎缺失的一个元数据字段。把它加上，并解释它防范的是什么。

4. 为一个具体操作（例如"向公开 Twitter 账号发帖"）设计一份质询应答清单。审查者必须回答哪三个问题？为什么是这三个？

5. 找出一个用同步"批准吗？"提示就足够（无需耐久存储）的场景。解释原因，并说出你由此接受的是哪一类风险。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|---|---|---|
| 先提议后提交（Propose-then-commit） | "两阶段审批" | 持久化的提议 + 明确提交 + 验证 |
| 幂等键（Idempotency key） | "重试安全令牌" | 每个提议唯一；第二次执行为空操作 |
| 数据来源链路（Data lineage） | "它从哪来" | 导致该提议产生的具体来源内容 |
| 影响范围（Blast radius） | "最坏情况" | 操作出错时的波及范围 |
| 橡皮图章（Rubber-stamp） | "快速批准" | 没有真正审查就点了"批准" |
| 质询应答（Challenge-and-response） | "强制清单" | 审查者必须对特定问题给出明确确认 |
| RequestInfoEvent | "MS Agent Framework 原语" | 带结构化元数据的耐久 HITL 请求 |
| `interrupt()` / `waitForApproval()` | "框架原语" | LangGraph / Cloudflare 中同一形态的等价物 |

## 延伸阅读

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`、耐久审批。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` 与 Durable Objects。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将 HITL 作为长程风险的缓解手段。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — 高风险系统的监管基线。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 围绕监督的宪法式框架。
