# Skills 与 Agent SDK —— Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 回答"有哪些工具可用"，Skill 回答"如何完成一项任务"。2026 年的技术栈把两者分层叠加。Anthropic 的 Agent Skills（开放标准，2025 年 12 月发布）以 SKILL.md 形式分发，支持渐进式披露。OpenAI 的 Apps SDK 是 MCP 加上 widget 元数据。AGENTS.md（已进入 60,000+ 个仓库）位于仓库根目录，承载项目级智能体上下文。本课会厘清每一层各自覆盖什么，并构建一个最小的 SKILL.md + AGENTS.md 组合包，可在不同智能体之间通用。

**Type:** Learn
**Languages:** Python (stdlib, SKILL.md parser and loader)
**Prerequisites:** Phase 13 · 07 (MCP server)
**Time:** ~45 minutes

## 学习目标

- 区分三个层次：AGENTS.md（项目上下文）、SKILL.md（可复用经验）、MCP（工具）。
- 编写带 YAML frontmatter 和渐进式披露的 SKILL.md。
- 以文件系统方式把 skill 加载进智能体运行时。
- 将一个 skill 与 MCP 服务器、AGENTS.md 组合起来，使同一个包能在 Claude Code、Cursor 和 Codex 中通用。

## 问题背景

一位工程师把发布说明（release notes）的撰写流程提炼成一段多步骤提示词："读取最近合并的 PR。按领域分组。逐条摘要。按团队风格撰写一条 changelog。发到 Slack 草稿。"然后把它放进团队的 Notion 文档。

现在他想在 Claude Code、Cursor 和 Codex CLI 中都用上这套流程。每个智能体加载指令的方式各不相同：Claude Code 用斜杠命令，Cursor 用 rules，Codex 用 `.codex.md`。于是这位工程师把流程复制了三份，维护三份拷贝。

AGENTS.md 和 SKILL.md 联手解决了这个问题：

- **AGENTS.md** 位于仓库根目录。每个兼容的智能体在会话启动时都会读取它。"这个项目怎么运作？有哪些约定？用什么命令跑测试？"
- **SKILL.md** 是一个可移植的包：YAML frontmatter（name、description）+ markdown 正文 + 可选资源。支持 skill 的智能体按名称按需加载。
- **MCP**（Phase 13 · 06-14）负责 skill 需要调用的工具。

三个层次，一份可移植的产物。

## 核心概念

### AGENTS.md (agents.md)

2025 年末推出，到 2026 年 4 月已被 60,000+ 个仓库采用。一个文件，放在仓库根目录。格式如下：

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

智能体在会话启动时读取该文件，据此校准在这个项目中的行为。2026 年的所有主流编码智能体都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md 格式

Anthropic 的 Agent Skills（2025 年 12 月作为开放标准发布）：

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

frontmatter 声明 skill 的身份信息。正文是 skill 加载时呈现给模型的提示词。

### 渐进式披露

skill 可以引用子资源，智能体只在需要时才去获取。例如：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 写一句"风格规则见 style-guide.md"。智能体只在 skill 实际运行时才拉取 style-guide.md。这样可以避免把模型未必用得上的细节塞进提示词，造成膨胀。

### 文件系统发现

智能体运行时会扫描已知目录寻找 SKILL.md 文件：

- `~/.anthropic/skills/*/SKILL.md`
- Project `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

加载依据是文件夹名和 frontmatter 中的 `name`。Claude Code、Anthropic Claude Agent SDK 以及 SkillKit（跨智能体工具）都遵循这一模式。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）和 `claude-agent-sdk`（Python）在会话启动时加载 skill，并在运行时内把它们暴露为可调用的"智能体"。当用户调用某个 skill 时，智能体循环会分发到对应的 skill。

### OpenAI Apps SDK

2025 年 10 月推出，直接构建在 MCP 之上。它把 OpenAI 此前的 Connectors 和 Custom GPT Actions 统一到单一的开发者界面下。一个 Apps SDK 应用包含：

- 一个 MCP 服务器（工具、资源、提示词）。
- 加上供 ChatGPT 界面使用的 widget 元数据。
- 加上一个可选的 MCP Apps `ui://` 资源，用于交互式界面。

协议相同，体验更丰富。

### 借助 SkillKit 实现跨智能体可移植性

SkillKit 等跨智能体分发层工具能把一份 SKILL.md 翻译成 32+ 种 AI 智能体（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）各自的原生格式。单一事实来源，多个消费方。

### 三层技术栈

| 层 | 文件 | 加载时机 | 用途 |
|-------|------|-------------|---------|
| AGENTS.md | 仓库根目录 | 会话启动 | 项目级约定 |
| SKILL.md | skills 目录 | skill 被调用时 | 可复用工作流 |
| MCP server | 外部进程 | 需要工具时 | 可调用的动作 |

三者协同工作：智能体在会话启动时读取 AGENTS.md，用户调用某个 skill，skill 的指令中包含 MCP 工具调用，智能体再通过 MCP 客户端进行分发。

## 生产实践

`code/main.py` 提供了一个纯标准库实现的 SKILL.md 解析器和加载器。它在 `./skills/` 下发现 skill，解析 YAML frontmatter 加 markdown 正文，生成一个以 skill 名称为键的字典。随后它模拟一个智能体循环，按名称调用 `release-notes-writer`。

值得关注的点：

- YAML frontmatter 由一个极简的标准库解析器解析（不依赖 `pyyaml`）。
- skill 正文原样存储；调用时智能体把它前置到系统提示词中。
- 通过一个 `read_subresource` 函数演示渐进式披露——按需拉取被引用的文件。

## 交付产物

本课产出 `outputs/skill-agent-bundle.md`。给定一个工作流，该 skill 会生成 SKILL.md + AGENTS.md + MCP 服务器蓝图的组合包，可在不同智能体之间通用。

## 练习

1. 运行 `code/main.py`。在 `skills/` 下新增第二个 skill，确认加载器能识别到它。

2. 为本课程仓库编写一份 AGENTS.md。包含测试命令、风格约定以及 Phase 13 的思维模型。

3. 把你团队内部文档中的一个多步骤工作流移植成 SKILL.md。验证它能在 Claude Code 中加载。

4. 手工把这个 skill 翻译成 Cursor 和 Codex 的原生规则格式。统计格式之间的差异量——这正是 SkillKit 所自动化的翻译面。

5. 阅读 Anthropic 的 Agent Skills 博客文章。找出 Claude Agent SDK 中本课加载器未覆盖的一个特性。（提示：agent 子调用。）

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| SKILL.md | "skill 文件" | YAML frontmatter 加 markdown 正文，由智能体运行时加载 |
| AGENTS.md | "仓库根目录的智能体上下文" | 项目级约定文件，会话启动时读取 |
| 渐进式披露（Progressive disclosure） | "懒加载子资源" | skill 正文引用的文件只在需要时才拉取 |
| Frontmatter | "顶部的 YAML 块" | 以 `---` 分隔的元数据（name、description） |
| Claude Agent SDK | "Anthropic 的 skill 运行时" | `@anthropic-ai/claude-agent-sdk`，负责加载 skill 并路由 |
| OpenAI Apps SDK | "MCP + widget 元数据" | OpenAI 基于 MCP 加 ChatGPT UI 钩子构建的开发者界面 |
| Skill 发现 | "文件系统扫描" | 遍历已知目录寻找 SKILL.md，按名称建立索引 |
| 跨智能体可移植性 | "一个 skill 通吃多个智能体" | 借助 SkillKit 类工具把一份 SKILL.md 翻译给 32+ 种智能体 |
| Agent Skill | "可移植的经验" | 可复用的任务模板，独立于 MCP 的工具概念之外 |
| Apps SDK | "MCP 加 ChatGPT UI" | 把 Connectors 和 Custom GPT 统一到 MCP 之上 |

## 延伸阅读

- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月发布公告
- [Anthropic — Agent Skills docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md 格式参考
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — 面向 ChatGPT 的 MCP 开发者平台
- [agents.md](https://agents.md/) — AGENTS.md 格式与采用列表
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — 官方 skill 示例
