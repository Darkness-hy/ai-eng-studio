# 毕业项目 09 —— 代码迁移智能体（仓库级语言 / 运行时升级）

> Amazon 的 MigrationBench（Java 8 升级到 17）和 Google 的 App Engine Py2 转 Py3 迁移器代表了 2026 年的水准。Moderne 的 OpenRewrite 在大规模场景下做确定性的 AST 重写。Grit 用 codemod 风格的 DSL 瞄准同一个问题。生产级模式是把两者结合：用确定性基座完成安全重写，再加一层智能体处理含糊的情况，配上按分支隔离的沙箱构建环境，以及一个在 PR 打开之前必须全部转绿的测试套件。这个毕业项目的目标是迁移 50 个真实仓库，并发布通过率和失败分类报告。

**Type:** Capstone
**Languages:** Python (agent), Java / Python (targets), TypeScript (dashboard)
**Prerequisites:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure)
**Phases exercised:** P5 · P7 · P11 · P13 · P14 · P15 · P17
**Time:** 30 hours

## 问题背景

大规模代码迁移是 2026 年编程智能体（coding agent）最干净的生产级应用之一。这里的判定标准一目了然（迁移之后测试套件是否通过？），回报真实可观（一次 Java 8 全量迁移是按人头计算的大工程），基准测试也是公开的（MigrationBench 的 50 仓库子集）。Moderne 的 OpenRewrite 负责确定性的部分。智能体层负责 OpenRewrite recipe 处理不了的一切：含糊的重写、构建系统漂移、长尾语法、传递依赖断裂。

你将构建一个智能体：给它一个 Java 8 仓库（或 Python 2 仓库），它产出一个 CI 全绿的已迁移分支。你要测量通过率、测试覆盖率保持情况、单仓库成本，并建立失败分类（failure taxonomy）。与「纯确定性方案」基线的并排对比，会告诉你智能体的价值到底在哪里。

## 核心概念

整条流水线分两层。**确定性基座**（deterministic substrate，Java 用 OpenRewrite，Python 用 libcst）安全地完成大部分机械性重写：import、方法签名、空安全修改、try-with-resources、废弃 API 替换。它速度快，产出可审计的 diff。**智能体层**（基于 Claude Opus 4.7 和 GPT-5.4-Codex 之上的 OpenAI Agents SDK 或 LangGraph）处理 recipe 搞不定的情况：构建文件升级（Maven/Gradle/pyproject）、传递依赖冲突、测试抖动、自定义注解。

每个仓库分到一个预装了目标运行时的 Daytona 沙箱。智能体迭代执行：跑构建、归类失败、应用修复、重跑。硬性限制：每仓库 30 分钟、8 美元、20 个智能体回合。如果所有测试通过且覆盖率变化不为负，该分支就打开一个 PR。否则，这个仓库被归入某个失败类别，并附上证据。

失败分类本身就是交付物。50 个仓库跑下来，到底什么坏了？传递依赖？自定义注解？构建工具版本？与迁移无关的测试抖动？每个类别都有计数和一个示例 diff。未来的 recipe 作者可以针对排名前三的类别下手。

## 架构

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## 技术栈

- 确定性基座：OpenRewrite（Java）或 libcst（Python）
- 智能体：OpenAI Agents SDK 或 LangGraph，底层模型为 Claude Opus 4.7 + GPT-5.4-Codex
- 沙箱：每个分支一个 Daytona devcontainer，预装目标运行时（Java 17 / Python 3.12）
- 构建系统：Maven、Gradle、uv（Python）
- 基准测试：Amazon MigrationBench 50 仓库子集（Java 8 到 17）、Google App Engine Py2 转 Py3 仓库
- 测试套件：并行运行器，覆盖率用 Jacoco（Java）或 coverage.py（Python）统计
- 可观测性：Langfuse + 每个仓库一份包含全部 diff 片段的 trace 包
- 仪表盘：失败分类仪表盘，展示各类别计数和示例 diff

## 从零实现

1. **Recipe 阶段。** 先跑 OpenRewrite（Java）或 libcst（Python）的 recipe。把 70-80% 的机械性迁移收掉。作为「recipe」commit 提交。

2. **构建试跑。** 在 Daytona 沙箱中安装目标运行时并跑构建。如果是绿的，直接跳到测试。如果是红的，移交给智能体。

3. **智能体循环。** LangGraph 配以下工具：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。智能体把失败归类（依赖、语法、测试、构建工具），应用针对性修复，再重跑。

4. **预算上限。** 每仓库 30 分钟墙钟时间、8 美元成本、20 个智能体回合。任何一项超限即中止，连同当前 diff 一起归入「budget_exhausted」类别。

5. **测试 + 覆盖率门控。** 构建转绿后，运行测试套件。把覆盖率与基础仓库对比。如果覆盖率下降超过 2%，归入「coverage_regression」类别。

6. **打开 PR。** 成功后推送分支，打开 PR，附上 diff，以及一份说明哪些 recipe 生效、哪些 commit 由智能体编写的摘要。

7. **失败分类。** 给每个失败的仓库打上类别标签：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。构建一个仪表盘。

8. **50 仓库全量运行。** 在 MigrationBench 子集上完整执行。报告各类别通过率、单仓库成本、覆盖率保持情况，以及与「纯确定性方案」基线的对比。

## 生产实践

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## 交付产物

交付物是 `outputs/skill-migration-agent.md`。给定一个仓库，它先执行确定性 recipe，再跑智能体循环，最终产出一个全绿的已迁移分支，或者把该仓库归入某个分类类别。

| 权重 | 评分项 | 测量方式 |
|:-:|---|---|
| 25 | MigrationBench 通过率 | 50 仓库子集 pass@1 |
| 20 | 测试覆盖率保持 | 相对基础仓库的平均覆盖率变化 |
| 20 | 单个迁移成功仓库的成本 | 通过的运行上的 $/仓库 |
| 20 | 智能体 / 确定性工具的集成度 | OpenRewrite 处理的修复与智能体编写的修复各占的比例 |
| 15 | 失败分析报告 | 分类体系的完整性与示例 |
| **100** | | |

## 练习

1. 只用 OpenRewrite 跑迁移流水线（不带智能体）。把通过率与完整流水线对比。找出只有靠智能体才能搞定的案例。

2. 实现一个「lint 干净度」检查：迁移之后跑风格 linter（Java 用 spotless，Python 用 ruff）。如果出现新的 lint 错误就让 PR 失败。测量「覆盖率保住了但风格退化了」的比例。

3. 加一个「最小 diff」优化器：智能体的分支通过测试之后，用第二遍处理裁掉不必要的改动。报告 diff 体积的缩减量。

4. 扩展到第三种迁移：Node 18 升级到 Node 22。复用沙箱封装；把 recipe 层换成自定义 codemod。

5. 把「首次构建转绿时间」（time-to-first-green-build，TTFGB）作为用户体验指标来测量。目标：p50 低于 10 分钟。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 确定性基座 | 「recipe 引擎」 | OpenRewrite / libcst：带安全保证的声明式 AST 重写 |
| Codemod | 「改代码的程序」 | 一条机械化修改源码的重写规则 |
| 构建漂移 | 「工具版本偏差」 | Maven / Gradle / uv 在大版本之间的细微行为差异 |
| 失败类别 | 「分类桶」 | 一个仓库迁移失败的标注原因：依赖、语法、测试、构建工具、预算 |
| 覆盖率变化 | 「覆盖率保持」 | 从基础分支到迁移分支的测试覆盖率百分比变化 |
| 智能体回合 | 「一轮工具调用」 | 智能体循环中一次「规划 -> 行动 -> 观察」的循环 |
| 预算耗尽 | 「撞到天花板」 | 该仓库用光了 30 分钟 / 8 美元 / 20 回合的额度仍未通过 |

## 延伸阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) —— 2026 年的权威基准
- [Moderne.io OpenRewrite platform](https://www.moderne.io) —— 确定性基座的参考实现
- [OpenRewrite documentation](https://docs.openrewrite.org) —— recipe 编写指南
- [Grit.io](https://www.grit.io) —— 另一种 codemod DSL
- [OpenAI sandboxed migration cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) —— Agents SDK 参考
- [Google App Engine Py2 to Py3 migrator](https://cloud.google.com/appengine) —— 另一个迁移基准
- [libcst](https://github.com/Instagram/LibCST) —— Python 确定性基座
- [Daytona sandboxes](https://daytona.io) —— 按分支隔离沙箱的参考实现
