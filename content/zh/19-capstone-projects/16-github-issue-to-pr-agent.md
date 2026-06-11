# Capstone 16 — GitHub Issue 到 PR 自主智能体

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud 和 Google Jules 在 2026 年都交付了同一种产品形态：给 issue 打个标签，就能得到一个 PR。在云端沙箱中运行智能体，验证测试通过，然后发布一个附带改动理由、可直接进入评审的 PR。难点在于：自动复现仓库的构建环境、防止凭据泄露、按仓库强制执行预算上限，以及确保智能体无法 force-push。本 Capstone 将构建自托管版本，并在成本和通过率上与各家托管方案对比。

**Type:** Capstone
**Languages:** Python (agent), TypeScript (GitHub App), YAML (Actions)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure)
**Phases exercised:** P11 · P13 · P14 · P15 · P17
**Time:** 30 hours

## 问题背景

异步云端编码智能体（async cloud coding agent）与交互式编码智能体（Capstone 01）是两个不同的产品类别。它的交互界面就是一个 GitHub 标签。你给 issue 打上 `@agent fix this` 标签，一个 worker 就会在云端沙箱中启动，克隆仓库、运行测试、编辑文件、验证结果，然后开出一个 PR，并把智能体的改动理由写进 PR 正文。没有交互式循环，也没有终端。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 和 Factory Droids 全都收敛到了这种形态。

工程挑战非常具体：环境复现（智能体必须在没有缓存开发镜像的情况下从零构建仓库）、不稳定测试（flaky tests，必须重跑或隔离）、凭据作用域控制（使用权限最小化的细粒度 GitHub App）、按仓库按天强制执行预算，以及禁止 force-push 的策略。本 Capstone 会在通过率、成本和安全性上与托管方案进行对比测量。

## 核心概念

触发器是一个 GitHub webhook（issue 标签或 PR 评论）。调度器（dispatcher）把任务入队到 ECS Fargate 或 Lambda。worker 把仓库拉取到 Daytona 或 E2B 沙箱中，沙箱使用根据仓库（语言、框架）推断出的通用 Dockerfile。智能体基于 Claude Opus 4.7 或 GPT-5.4-Codex 运行 mini-swe-agent 或 SWE-agent v2 循环。它不断迭代：阅读代码、提出修复、应用补丁、运行测试。

验证是门控步骤。完整 CI 必须先在沙箱内通过，PR 才会开出。系统会计算覆盖率增量（coverage delta）；如果负向超过阈值，PR 仍会开出，但会被打上 `needs-review` 标签。智能体把改动理由作为 PR 描述发布，外加一个 `@agent` 会话线程，评审者可以在其中 @ 提及智能体进行跟进。

安全性通过两个不同的 GitHub 层面来限定：App 提供短期有效的 installation token，权限为 `workflows: read` 以及范围很窄的仓库 contents/PR 权限；分支保护（而非 App 权限）强制执行「禁止直接写入 `main`」和「禁止 force-push」——App 绝不会被加入 bypass 列表。对 `.github/workflows` 的路径级只读访问并不是 GitHub App 真正支持的原语，因此智能体在文件编辑上的允许列表必须在 worker 端强制执行这一点。按仓库按天的预算上限在调度器处强制执行（例如每仓库每天最多 5 个 PR、每个 PR 上限 $20）。

## 架构

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## 技术栈

- 触发：带细粒度 token 的 GitHub App；webhook 接收器通过 Lambda 或 Fly.io 部署
- Worker：ECS Fargate 任务（或 GitHub Actions 自托管 runner）
- 沙箱：每个任务一个 Daytona devcontainer 或 E2B 沙箱
- 智能体循环：以 mini-swe-agent 为基线，或基于 Claude Opus 4.7 / GPT-5.4-Codex 的 SWE-agent v2
- 检索：tree-sitter 仓库地图（repo-map）+ ripgrep
- 验证：沙箱内完整 CI + 覆盖率增量门控
- 可观测性：Langfuse，按 PR 归档 trace 并在 PR 正文中给出链接
- 预算：每仓库每日美元上限；每仓库每日最大 PR 数

## 从零实现

1. **GitHub App。** 细粒度 installation token：issues 读+写、pull_requests 写、contents 读+写、workflows 读。分支保护（唯一能做到这一点的层面）强制执行「禁止直接推送到 `main`」和「禁止 force-push」；App 不在 bypass 列表中。由于 GitHub App 权限不支持路径级作用域，worker 通过对提议 diff 做允许列表检查来强制执行「禁止写入 `.github/workflows` 之下的任何内容」。

2. **Webhook 接收器。** Lambda 函数接收 issue 标签 / PR 评论的 webhook。按标签 `@agent fix this` 过滤。入队到 SQS。

3. **调度器。** 从 SQS 取出任务。强制执行每仓库每日预算。启动一个 ECS Fargate 任务，传入仓库 URL、issue 正文和一个全新的 Daytona 沙箱。

4. **环境推断。** 检测语言（Python、Node、Go、Rust）和包管理器（uv、pnpm、go mod、cargo）。如果仓库没有 Dockerfile，就即时生成一个。

5. **智能体循环。** 使用 Claude Opus 4.7 的 mini-swe-agent 或 SWE-agent v2。工具：ripgrep、tree-sitter 仓库地图、read_file、edit_file、run_tests、git。硬性限制：成本 $20、墙钟时间 30 分钟、智能体回合数 30。

6. **验证。** 循环结束后，在沙箱内运行完整测试套件。通过 jacoco / coverage.py 计算覆盖率增量。如果 CI 红灯：停止，不开 PR。如果覆盖率下降超过 2%：开 PR 并打上 `needs-review` 标签。

7. **PR 发布。** 推送智能体分支。通过 GitHub API 开 PR，内容包括：标题、改动理由、diff 摘要、trace URL、成本、回合数。

8. **凭据卫生。** Worker 使用短期有效的 GitHub App installation token 运行。日志在归档前会清洗掉密钥。

9. **评测。** 30 个难度各异的内部预置 issue。测量通过率、PR 质量（diff 大小、风格、覆盖率）、成本、延迟。在同一批 issue 上与 Cursor Background Agents 和 AWS Remote SWE Agents 对比。

## 生产实践

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## 交付产物

交付产物是 `outputs/skill-issue-to-pr.md`。一个 GitHub App + 异步云端 worker，把打了标签的 issue 转化为可直接进入评审的 PR，同时保证成本有界、凭据作用域受限。

| 权重 | 评分标准 | 测量方式 |
|:-:|---|---|
| 25 | 30 个 issue 上的通过率 | 端到端成功（CI 绿灯 + 覆盖率达标） |
| 20 | PR 质量 | diff 大小、覆盖率增量、风格一致性 |
| 20 | 每个已解决 issue 的成本与延迟 | 每个 PR 的美元开销与墙钟时间 |
| 20 | 安全性 | 作用域受限的 token、按仓库预算、禁止 force-push、凭据卫生 |
| 15 | 操作者体验 | 改动理由评论、可重试入口、@ 提及跟进 |
| **100** | | |

## 练习

1. 增加「修复不稳定测试」模式：标签 `@agent stabilize-flake TestX` 会在沙箱内把该测试运行 50 次，并提出一个能让它稳定下来的最小改动。

2. 在三个共享 issue 上与 Cursor Background Agents 对比成本。报告各工具分别在哪些场景占优。

3. 实现预算看板：每仓库每日成本、每用户成本。出现异常时告警。

4. 构建「演练（dry-run）」模式：开一个草稿 PR 但不运行 CI，让评审者可以低成本地查看方案。

5. 增加保留策略：超过 7 天未合并的 PR 分支自动删除。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| GitHub App | 「作用域受限的机器人身份」 | 拥有细粒度权限 + 短期有效 installation token 的 App |
| 异步云端智能体 | 「后台智能体」 | 在云端沙箱（而非终端）中运行的非交互式 worker |
| 环境推断 | 「Dockerfile 合成」 | 检测语言 + 包管理器，缺少 Dockerfile 时自动生成 |
| 验证 | 「沙箱内 CI」 | 在 worker 内运行完整测试套件之后才开 PR |
| 覆盖率增量 | 「覆盖率保持」 | 从基线分支到智能体分支的测试覆盖率百分比变化 |
| 按仓库预算 | 「每日上限」 | 在调度器处强制执行的美元上限与 PR 数量上限 |
| 改动理由 | 「PR 正文说明」 | 智能体对改了什么、为何而改的总结；PR 正文中必须包含 |

## 延伸阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 异步云端智能体的权威参考实现
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI 参考
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — 商业替代方案
- [OpenAI Codex (cloud)](https://openai.com/codex) — 托管竞品
- [Google Jules](https://jules.google) — Google 的托管版本
- [Factory Droids](https://www.factory.ai) — 另一个商业参考
- [GitHub App documentation](https://docs.github.com/en/apps) — 作用域受限的机器人身份
- [Daytona cloud sandboxes](https://daytona.io) — 参考沙箱
