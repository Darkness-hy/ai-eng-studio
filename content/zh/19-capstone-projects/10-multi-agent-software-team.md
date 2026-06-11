# Capstone 10 — 多智能体软件工程团队

> SWE-AF 的工厂架构、MetaGPT 的角色化提示、AutoGen 0.4 的类型化 actor 图、Cognition 的 Devin、Factory 的 Droids，到 2026 年都收敛到了同一种形态：一个架构师负责规划，N 个编码者在并行 worktree 中干活，一个审查者把关，一个测试者验证。并行 worktree 把墙钟时间换成了吞吐量。共享状态和交接协议则成了故障面。这个毕业项目的任务是：搭建这样一支团队，在 SWE-bench Pro 上做评测，并报告哪些交接会出问题、出问题的频率有多高。

**Type:** Capstone
**Languages:** Python / TypeScript (agents), Shell (worktree scripts)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 16 (multi-agent), Phase 17 (infrastructure)
**Phases exercised:** P11 · P13 · P14 · P15 · P16 · P17
**Time:** 40 hours

## 问题背景

单智能体编码框架在大型任务上会遇到天花板。这不是因为单个智能体能力不行，而是因为 200k token 的上下文装不下一份架构方案、四个并行的代码库切片、审查者的评论，再加上测试输出。多智能体工厂把问题拆开：架构师负责方案，编码者在并行 worktree 中各自负责实现，审查者把关，测试者验证。SWE-AF 的「工厂」架构、MetaGPT 的角色体系、AutoGen 的类型化 actor 图——三种表述描述的都是同一种形态。

故障面在交接（handoff）上。架构师规划了编码者实现不了的东西；编码者产出互相冲突的 diff；审查者批准了一个臆造出来的修复；测试者和一个还在写代码的编码者发生竞态。你要搭建这样一支团队，在 50 个 SWE-bench Pro issue 上跑起来，追踪每一次交接，并公开发布事后复盘。

## 核心概念

角色就是类型化的智能体。**架构师（Architect）**（Claude Opus 4.7）阅读 issue、撰写方案，并把它拆解成带有显式接口定义的子任务。**编码者（Coders）**（Claude Sonnet 4.7，N 个并行实例，每个都在一个 `git worktree` + Daytona 沙箱中）各自独立地实现子任务。**审查者（Reviewer）**（GPT-5.4）阅读合并后的 diff，要么批准，要么提出具体的修改要求。**测试者（Tester）**（Gemini 2.5 Pro）在隔离环境中运行测试套件，连同产物一起报告通过/失败。

通信通过一块共享任务板（task board，基于文件或 Redis）进行。每个角色只消费它被允许处理的任务。交接采用 A2A 协议的类型化消息。协调层面要解决的问题包括：合并冲突的处理（设协调者角色或自动三方合并）、共享状态的同步（编码者一旦开工，方案就冻结；重新规划是独立事件），以及审查者把关（审查者不能批准自己写的或自己提出的改动）。

token 放大（token amplification）是隐藏成本。每多一道角色边界，就会多出摘要提示和交接上下文。一个 40 轮的单智能体运行，分到四个角色后会变成总计 160 轮。评分细则之所以专门考核相对单智能体基线的 token 效率，是因为问题不是「多智能体能不能跑通」，而是「按花的每一美元算，它赢不赢」。

## 架构

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## 技术栈

- 编排：LangGraph，共享状态 + 每个智能体一个子图
- 消息传递：A2A 协议（Google 2025），用于类型化的智能体间消息
- 模型：Opus 4.7（架构师）、Sonnet 4.7（编码者）、GPT-5.4（审查者）、Gemini 2.5 Pro（测试者）
- worktree 隔离：每个编码者一个 `git worktree add` + Daytona 沙箱
- 合并协调者：自研三方合并 + LLM 介入的冲突解决
- 评测：SWE-bench Pro（50 个 issue）、SWE-AF 场景、HumanEval++ 用于单元测试
- 可观测性：Langfuse，带角色标签的 span，按智能体核算 token
- 部署：K8s，每个角色一个独立 Deployment + 基于积压量的 HPA

## 从零实现

1. **任务板。** 基于文件的 JSONL，承载类型化消息：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。智能体按标签订阅。

2. **架构师。** 读取 GitHub issue，用一个方案模板驱动 Opus 4.7，模板要求给出显式的子任务接口（涉及的文件、公开函数、对测试的影响）。产出一条 `plan_request`，内含子任务的 DAG。

3. **编码者。** N 个并行 worker，各自从任务板上认领一个子任务。每个 worker 用 `git worktree add` 新建一个分支，外加一个 Daytona 沙箱。完成子任务实现后，产出 `diff_ready`，附上补丁和测试增量。

4. **合并协调者。** 所有编码者完成后，把 N 条分支三方合并到一个暂存分支。仅在存在文件级重叠时才引入 LLM 介入的冲突解决。

5. **审查者。** GPT-5.4 阅读合并后的 diff。不能批准自己写的 diff。产出 `approved`（无后续动作）或 `review_feedback`，其中带有具体修改要求，路由回对应的编码者。

6. **测试者。** Gemini 2.5 Pro 在干净沙箱中运行测试套件。捕获产物。产出 `test_passed` 或带堆栈跟踪的 `test_failed`。失败的测试回流到负责该失败子任务的编码者。

7. **交接核算。** 每条跨越角色边界的消息都在 Langfuse 中记录一个 span，附上载荷大小和所用模型。计算每个子任务的 token 放大率（coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens）。

8. **评测。** 在 50 个 SWE-bench Pro issue 上运行。与单智能体基线（一个 Sonnet 4.7 在单一 worktree 中工作）对比 pass@1 和每解决一个 issue 的美元成本。

9. **事后复盘。** 对每个失败的 issue，定位出问题的那次交接（方案太模糊、合并冲突、审查者误批、测试者抖动）。产出一份交接失败直方图。

## 生产实践

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## 交付产物

交付物是 `outputs/skill-multi-agent-team.md`。给定一个 issue URL 和并行度，团队产出一个可直接合并的 PR，并附带按角色核算的 token 数据。

| 权重 | 评分项 | 测量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 配对的 50 个 issue 子集上的 pass@1 |
| 20 | 并行加速比 | 墙钟时间对比单智能体基线 |
| 20 | 审查质量 | 注入缺陷探针上的误批率 |
| 20 | token 效率 | 每解决一个 issue 的总 token 数对比单智能体 |
| 15 | 协调工程 | 合并冲突解决、交接失败直方图 |
| **100** | | |

## 练习

1. 在运行中途往一个 diff 里注入一个明显的 bug（在主体逻辑之前多加一个 `return None`）。测量审查者的误批率。调优审查者提示词，直到误批率低于 5%。

2. 缩减到两个编码者（架构师 + 编码者 + 审查者 + 测试者，编码者串行执行两个子任务）。对比墙钟时间和通过率。

3. 用单写者约束替换合并协调者（让各子任务触及的文件集互不相交）。测量这给架构师带来的规划负担。

4. 把审查者从 GPT-5.4 换成 Claude Opus 4.7。测量误批率和 token 成本的变化。

5. 加入第五个角色：文档员（Haiku 4.5）。审查通过后，由它生成一条 changelog 条目。测量文档质量是否值得这笔额外的 token 开销。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 并行 worktree | 「隔离分支」 | 用 `git worktree add` 为每个编码者生成一棵全新的工作树 |
| 任务板 | 「共享消息总线」 | 由文件或 Redis 存储的类型化消息，智能体按需订阅 |
| 交接 | 「角色边界」 | 任何从一个角色的上下文跨到另一个角色上下文的消息 |
| token 放大 | 「多智能体开销」 | 同一任务下各角色总 token 数 / 单智能体 token 数 |
| A2A 协议 | 「agent 到 agent」 | Google 2025 年发布的类型化智能体间消息规范 |
| 合并协调者 | 「集成者」 | 执行三方合并并调解冲突的组件 |
| 误批 | 「审查者幻觉」 | 审查者批准了一个带有已知 bug 的 diff |

## 延伸阅读

- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — 2026 年多智能体工厂的参考实现
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — 基于角色的多智能体框架
- [AutoGen v0.4](https://github.com/microsoft/autogen) — Microsoft 的类型化 actor 框架
- [Cognition AI (Devin)](https://cognition.ai) — 参考产品
- [Factory Droids](https://www.factory.ai) — 另一个参考产品
- [Google A2A protocol](https://developers.google.com/agent-to-agent) — 智能体间消息规范
- [git worktree documentation](https://git-scm.com/docs/git-worktree) — 隔离机制的底层基础
- [SWE-bench Pro](https://www.swebench.com) — 评测目标
