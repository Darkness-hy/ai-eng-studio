# 毕业项目 01 —— 终端原生编程智能体

> 到 2026 年，编程智能体（coding agent）的形态已经定型：一个 TUI harness、一份有状态的计划、一个沙箱化的工具面，以及一个规划、行动、观察、恢复的循环。从远处看，Claude Code、Cursor 3 和 OpenCode 长得一模一样。这个毕业项目要求你端到端构建一个——CLI 进、pull request 出——并在 SWE-bench Pro 上与 mini-swe-agent 和 Live-SWE-agent 对比测量。你会明白：难点不在模型调用，而在工具循环、沙箱，以及一次 50 轮运行的成本上限。

**Type:** Capstone
**Languages:** TypeScript / Bun (harness), Python (eval scripts)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools and protocols), Phase 14 (agents), Phase 15 (autonomous systems), Phase 17 (infrastructure)
**Phases exercised:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**Time:** 35 hours

## 问题背景

2026 年，编程智能体成为最主流的 AI 应用类别。Claude Code（Anthropic）、带 Composer 2 和 Agent Tabs 的 Cursor 3（Cursor）、Amp（Sourcegraph）、OpenCode（112k stars）、Factory Droids，以及 Google Jules，发布的都是同一架构的变体：终端 harness（执行框架）、带权限控制的工具面、沙箱，以及围绕前沿模型构建的规划-行动-观察循环。前沿很窄——Live-SWE-agent 用 Opus 4.5 在 SWE-bench Verified 上达到 79.2%——但工程功夫很宽。大多数失败模式并非模型出错，而是工具循环不稳定、上下文污染、token 成本失控，以及破坏性的文件系统操作。

你无法站在外面推理这些智能体的行为。你必须亲手构建一个，看着循环在第 47 轮因为 ripgrep 返回 8MB 匹配结果而崩溃，然后重建截断层。这正是这个毕业项目的意义所在。

## 核心概念

harness 有四个面。**规划（Plan）** 维护一个 TodoWrite 风格的状态对象，模型每轮都会重写它。**行动（Act）** 分发工具调用（读取、编辑、运行、搜索、git）。**观察（Observe）** 捕获 stdout / stderr / 退出码，做截断，再把摘要喂回去。**恢复（Recover）** 处理工具错误，既不撑爆上下文窗口，也不陷入无限循环。2026 年的形态还多了一样东西：**钩子（hooks）**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact`——这些可配置的扩展点让操作者注入策略、遥测和护栏。

沙箱用 E2B 或 Daytona。每个任务在一个全新的 devcontainer 里运行，挂载一个可读写的 git worktree。harness 永远不碰宿主机文件系统。无论成功还是失败，worktree 最后都会被销毁。成本控制在三层强制执行：每轮 token 上限、每会话美元预算，以及硬性轮数限制（通常是 50）。可观测层是带 GenAI 语义约定的 OpenTelemetry span，发送到自托管的 Langfuse。

## 架构

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## 技术栈

- Harness 运行时：Bun 1.2 + Ink 5（终端里的 React）
- 模型访问：OpenRouter 统一 API，接入 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（留给最难的任务）
- 工具传输层：Model Context Protocol StreamableHTTP（MCP 2026 修订版）
- 沙箱：E2B 沙箱（JS SDK）或 Daytona devcontainer
- 代码搜索：ripgrep 子进程，外加覆盖 17 种语言的 tree-sitter 解析器（预编译）
- 隔离：每个任务一次 `git worktree add`，成功 / 失败都清理
- 评测 harness：SWE-bench Pro（verified 子集）+ Terminal-Bench 2.0 + 你自己的 30 任务保留集
- 可观测性：OpenTelemetry SDK，带 `gen_ai.*` 语义约定 → 自托管 Langfuse
- PR 提交：GitHub App，细粒度 token，权限范围仅限目标仓库

## 从零实现

1. **TUI 与命令循环。** 用 Ink 搭建一个 Bun 项目脚手架。接受 `agent run <repo> "<task>"`。打印分栏视图：计划面板（顶部）、工具调用流（中部）、token 预算（底部）。加上 Ctrl-C 取消，退出前先触发 `SessionEnd` 钩子。

2. **计划状态。** 定义带类型的 TodoWrite schema（pending / in_progress / done 条目，附备注）。模型每轮通过一次工具调用重写完整状态——不要让它做增量修改。把计划持久化到 `.agent/state.json`，崩溃后即可恢复。

3. **工具面。** 定义六个工具：`read_file`、`edit_file`（带 diff 预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带超时）、`git`（status / diff / commit / push）。通过 MCP StreamableHTTP 暴露，让 harness 与传输层解耦。每个工具都返回截断后的输出（每次调用上限 4k token）。

4. **沙箱封装。** 每个任务启动一个 E2B 沙箱。用 `git worktree add -b agent/$TASK_ID` 创建全新分支。所有工具调用都在沙箱内执行。宿主机文件系统不可触达。

5. **钩子。** 实现全部八种 2026 钩子类型。至少接入四个用户编写的钩子：(a) `PreToolUse` 破坏性命令守卫，拦截 worktree 之外的 `rm -rf`；(b) `PostToolUse` token 记账；(c) `SessionStart` 预算初始化；(d) `Stop` 写出最终的 trace 包。

6. **评测循环。** 克隆 SWE-bench Pro Python 的一个 30 个 issue 的子集。用你的 harness 逐个跑。在 pass@1、每任务轮数和每任务美元成本上与 mini-swe-agent（最小基线）对比。把结果写入 `eval/results.jsonl`。

7. **成本控制。** 硬性截断：50 轮、200k 上下文、每任务 5 美元。`PreCompact` 钩子在 150k 处把较早的轮次总结成一个先验状态块，为新的观察腾出空间，同时不丢失计划。

8. **PR 提交。** 任务成功后的最后一步是 `git push`，再调用 GitHub API 开一个 PR，正文里附上计划和 diff 摘要。

## 生产实践

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## 交付产物

交付的技能位于 `outputs/skill-terminal-coding-agent.md`。给定一个仓库路径和一段任务描述，它在沙箱里运行完整的规划-行动-观察循环，返回一个 PR URL 外加一个 trace 包。本毕业项目的评分标准：

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 对比基线 | 你的 harness 与 mini-swe-agent 在 30 个匹配的 Python 任务上对比 |
| 20 | 架构清晰度 | 规划/行动/观察的分离、钩子面、工具 schema——对照 Live-SWE-agent 的布局评审 |
| 20 | 安全性 | 沙箱逃逸测试、权限提示、破坏性命令守卫通过红队测试 |
| 20 | 可观测性 | trace 完整性（100% 的工具调用都有 span）、每轮 token 记账 |
| 15 | 开发者体验 | 冷启动 < 2s、崩溃恢复能续上计划、Ctrl-C 能在工具执行中途干净地取消 |
| **100** | | |

## 练习

1. 把底层模型从 Claude Sonnet 4.7 换成用 vLLM 部署的 Qwen3-Coder-30B。对比 pass@1 和每任务美元成本。报告开源模型在哪些地方表现不足。

2. 添加一个 `reviewer` 子智能体，在 PR 提交前阅读 diff，并能发起一轮修订循环。测量误报的审查是否会把 SWE-bench 通过率拉到单智能体基线之下（提示：通常会）。

3. 压力测试沙箱：写一个尝试 `curl` 外部 URL 的任务，再写一个尝试在 worktree 之外写文件的任务。确认两者都被 PreToolUse 钩子拦截。记录这些尝试。

4. 用一个更小的模型（Haiku 4.5）实现 `PreCompact` 摘要。测量在 3 倍压缩下损失了多少计划保真度。

5. 把 MCP StreamableHTTP 传输层换成 stdio。对冷启动和单次调用延迟做基准测试。为纯本地使用场景选出赢家。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Harness | "智能体循环" | 包裹在模型外围的代码，负责分发工具、维护计划状态、强制执行预算 |
| Hook | "智能体事件监听器" | 由 harness 在八种生命周期事件之一触发运行的用户编写脚本 |
| Worktree | "Git 沙箱" | 位于另一路径的关联 git 检出；可随时丢弃，不影响主克隆 |
| TodoWrite | "计划状态" | 一份带类型的 pending/in-progress/done 条目列表，模型每轮重写 |
| StreamableHTTP | "MCP 传输层" | 2026 年 MCP 修订版：长连接 HTTP 双向流式传输；取代 SSE |
| Token ceiling | "上下文预算" | 每轮或每会话的输入+输出 token 上限；触发压缩或终止 |
| pass@1 | "单次尝试通过率" | 第一次运行就解决的 SWE-bench 任务比例，不重试、不偷看测试集 |

## 延伸阅读

- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) —— Anthropic 的参考 harness
- [Cursor 3 changelog](https://cursor.com/changelog) —— Agent Tabs 与 Composer 2 的产品说明
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) —— SWE-bench harness 对比用的最小基线
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) —— 用 Opus 4.5 在 SWE-bench Verified 上达到 79.2%
- [OpenCode](https://opencode.ai) —— 开源 harness，112k stars
- [SWE-bench Pro leaderboard](https://www.swebench.com) —— 本毕业项目瞄准的评测
- [Model Context Protocol 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) —— StreamableHTTP、能力元数据
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— 工具调用与 token 用量的 span 模式
