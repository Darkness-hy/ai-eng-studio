# Claude Code 作为自主智能体：权限模式与 Auto Mode

> Claude Code 提供七种权限模式。"plan" 在每个动作执行前都会询问，"default" 只对有风险的动作询问，"acceptEdits" 自动批准文件写入但仍会在执行 shell 命令前确认，"bypassPermissions" 则批准一切操作。Auto Mode（2026 年 3 月 24 日发布）用一个两阶段并行安全分类器取代了逐动作审批：单 token 快速检查覆盖每个动作；被标记的动作触发链式思考（chain-of-thought）深度审查。动作预算通过 `max_turns` 和 `max_budget_usd` 强制执行。Auto Mode 以研究预览（research preview）形式发布——Anthropic 明确表示仅靠分类器并不足够。

**Type:** Learn
**Languages:** Python (stdlib, two-stage classifier simulator)
**Prerequisites:** Phase 15 · 01 (Long-horizon agents), Phase 15 · 09 (Coding-agent landscape)
**Time:** ~45 minutes

## 问题背景

在你机器上运行的自主编码智能体是一个独立的安全类别。攻击面是智能体能触及的一切——文件系统、网络、凭证、剪贴板、任何浏览器标签页、任何打开的终端。Bruce Schneier 等人已经公开指出这一点：computer-use 智能体不是聊天机器人的"功能升级"，而是一种带有全新风险特征的新型工具。

Claude Code 的权限系统是 Anthropic 给出的答案。它不是一个"自主 / 非自主"的单一开关，而是横跨能力阶梯的七种模式：plan → default → acceptEdits → … → bypassPermissions。每种模式都是速度与逐动作审查之间的不同权衡。Auto Mode（2026 年 3 月）增加了一个两阶段分类器：对于分类器判定安全的动作，审批不再占用用户的关键路径；对于分类器标记的动作，则保留一层审查。

工程问题在于：这套系统能拦截什么、会漏掉什么，以及给定任务到底该用哪种模式？

## 核心概念

### 七种权限模式

| 模式 | 行为 | 适用场景 |
|---|---|---|
| `plan` | 智能体提出计划；用户批准整个计划；每个动作在执行前都要经过审查 | 不熟悉的任务；贴近生产环境的代码；首次在某个仓库上使用该智能体 |
| `default` | 智能体执行动作；遇到任何"有风险"的动作（shell 执行、破坏性操作、网络调用）时向用户询问 | 大多数交互式编码会话 |
| `acceptEdits` | 文件写入自动批准；shell 执行和网络调用仍会询问 | 跨多个文件的重构工作 |
| `acceptExec` | 在精选的允许列表范围内，shell 命令自动批准；写入自动批准 | 每条 shell 命令都是 `npm test` 之类的紧凑内循环 |
| `autoMode` | 两阶段安全分类器；被标记的动作升级到审查 | 在受限工作区中进行长程无人值守运行 |
| `yolo` | 跳过大多数询问；仍执行工具允许列表 / 拒绝列表 | 临时沙箱、CI 任务、研究脚本 |
| `bypassPermissions` | 批准一切 | 文档中的说法是"仅限你愿意随时丢弃的临时容器内使用" |

（上述名称与 Claude Code 公开文档一致；"yolo" 是对宽松中间地带的非正式简称。）

### 一页讲清 Auto Mode

Auto Mode（2026 年 3 月 24 日上线）是首个将逐动作审批委托给模型的权限模式。其结构如下：

1. **阶段一——单 token 分类器。** 对每个提议的动作运行。开销低。返回"安全"或"标记"。设计为与主智能体循环并行运行，使获批的动作无需排队等待。
2. **阶段二——二次策略 / 安全审查。** 仅对被标记的动作运行。针对该动作、会话当前状态以及声明的任务，进行聚焦的策略与安全评估。置信度低时升级到用户人工介入（HITL）。

预算控制与分类器并行存在：

- `max_turns`——单次会话的总迭代数。
- `max_budget_usd`——触发会话中止的美元上限。
- 按工具的动作计数限制（`WebFetch` 调用不超过 N 次，等等）。

### 系统能拦截什么

- 工具输入中直白的提示注入（prompt injection），即注入的指令对应已知的高风险动作形态。
- 重复的工具循环——分类器能看出第 N+1 个动作与第 N 个动作几乎相同，且连续出现了五次。
- 在本应只做文件编辑的会话中，明显超出范围的 shell 命令。

### 系统可能漏掉什么

- **隐蔽的提示注入**：它在不产生任何单个被标记动作的前提下改变智能体行为。间接提示注入不是一个可以彻底修补的漏洞（OpenAI 准备度负责人，2025 年，针对浏览器智能体的发言——见第 11 课）。
- **语义层面的不当行为。** 每个单独的动作看起来都安全，但组合起来的轨迹是有害的。分类器只判断动作本身；它不会重新推导用户的意图。
- **通过合法渠道外泄数据。** 把数据写入你自己的文件，再 `git push` 到公开仓库，是一串各自被允许的动作，问题出在它们的组合上。

### 研究预览的定位

Anthropic 以研究预览形式发布 Auto Mode。文档明确指出分类器只是一层防护，而非完整解决方案：用户应将 Auto Mode 与预算、允许列表、隔离工作区以及轨迹审计（第 12–16 课）结合使用。研究预览的定位也反映了文档中提到的"评估与部署差距"（第 1 课）——一个通过离线评估的分类器，在用户上下文模糊的真实会话中可能表现迥异。

### 这条阶梯在你工作流中的位置

- 不熟悉的任务：从 `plan` 开始。读一份计划比回滚一次糟糕的运行便宜得多。
- 熟悉的重构：`acceptEdits` 能省下大量确认点击。
- 无人值守的后台运行：只在你已测量过爆炸半径的工作区内使用 `autoMode`（没有凭证、没有生产环境挂载、没有你未主动选择开启的出站流量）。
- 临时容器：当且仅当容器及其凭证可随时丢弃时，`yolo` / `bypassPermissions` 才是可接受的。

```figure
autonomy-oversight
```

## 生产实践

`code/main.py` 模拟了这个两阶段分类器。阶段一是对提议动作的廉价关键词规则；阶段二是较慢的多规则审查器。驱动程序输入一段简短的合成轨迹（安全动作、一次提示注入尝试、一个重复循环），并展示分类器在哪里拦截成功、在哪里漏判。

## 交付产物

`outputs/skill-permission-mode-picker.md` 将任务描述匹配到合适的权限模式、预算上限和所需的隔离措施。

## 练习

1. 运行 `code/main.py`。哪种合成动作类型从不被阶段一标记，却总是被阶段二捕获？哪种两个阶段都抓不到？

2. 扩展阶段一的规则集，捕获一种特定的已知恶意形态（例如 `curl $ATTACKER/exfil`）。在良性动作样本上测量误报率。

3. 阅读 Anthropic 的 "How the agent loop works" 文档。列出在 `default` 模式下智能体默认接触的每一项外部状态。在无人值守运行 `autoMode` 之前，哪些需要单独加以管控？

4. 设计一份 24 小时无人值守运行的预算：`max_turns`、`max_budget_usd`、按工具的上限、允许列表。为每个数字给出理由。

5. 描述一条轨迹：其中每个单独动作都被阶段一和阶段二批准，但组合后的行为却偏离了目标。（第 14 课讲解断路开关和金丝雀 token 如何应对这一问题。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| 权限模式 | "智能体能做多少事" | 七种具名策略之一，控制逐动作审批 |
| plan 模式 | "做任何事之前先问" | 智能体先写计划；用户批准后才执行 |
| acceptEdits | "让它写文件" | 文件写入自动批准；shell 执行仍会询问 |
| autoMode | "自动审批" | 两阶段安全分类器；被标记的动作升级处理 |
| bypassPermissions | "完全 YOLO" | 批准一切；面向临时容器设计 |
| 阶段一分类器 | "快速 token 检查" | 对提议动作的单 token 规则；并行运行 |
| 阶段二分类器 | "深度审查" | 对被标记动作进行链式思考推理 |
| 研究预览 | "尚未正式发布" | Anthropic 对失败模式仍在摸索中的功能的定位说法 |

## 延伸阅读

- [Anthropic — How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop)——权限模式、预算、动作格式。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview)——托管服务执行模型。
- [Anthropic — Claude Code product page](https://www.anthropic.com/product/claude-code)——功能面与 Auto Mode 发布公告。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution)——塑造分类器判断的基于理由的层。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy)——关于长程权限设计的内部视角。
