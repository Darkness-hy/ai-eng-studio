# 浏览器智能体与长程网页任务

> ChatGPT agent（2025 年 7 月）将 Operator 和 deep research 合并为一个浏览器/终端智能体，并以 68.9% 刷新了 BrowseComp 的 SOTA。OpenAI 于 2025 年 8 月 31 日关停了 Operator——这是产品层面的整合。Anthropic 收购 Vercept 后，Claude Sonnet 在 OSWorld 上的成绩从不足 15% 提升到 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修复了原版 WebArena 中 11.3 个百分点的假阴性率，并发布了包含 258 个任务的 Hard 子集。这些数字是真实的，但攻击面同样真实：OpenAI 的 preparedness 负责人公开表示，针对浏览器智能体的间接提示注入"不是一个能被彻底修复的 bug"。2025–2026 年有据可查的攻击包括：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks），以及 Perplexity Comet 中的一键劫持。

**Type:** Learn
**Languages:** Python (stdlib, indirect prompt-injection attack surface model)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~45 minutes

## 问题背景

浏览器智能体是一种长程智能体：它读取不可信内容，并执行具有实际后果的动作。智能体访问的每一个页面，都是一段并非由用户编写的输入；每个页面上的每一个表单，都是一条潜在的命令通道。2025–2026 年的攻击案例集表明这绝非假设：Tainted Memories 让攻击者通过精心构造的页面把恶意指令绑定到智能体的记忆中；HashJack 把命令藏在智能体访问的 URL 片段（fragment）里；Perplexity Comet 的劫持只需一次点击即可得手。

防御态势令人不安。OpenAI 的 preparedness 负责人把这层窗户纸捅破了：间接提示注入"不是一个能被彻底修复的 bug"。原因在于，这类攻击寄生在智能体"读取"与"行动"之间的边界上，而这条边界在架构层面本就模糊——模型读到的每一个 token，原则上都可能被当作一条指令。

本课会逐一点名攻击面，梳理基准测试格局（BrowseComp、OSWorld、WebArena-Verified），并对一个最小化的间接提示注入场景建模，以便你在第 14 课和第 18 课中能够推演真正的防御手段。

## 核心概念

### 2026 年的格局，每个系统一段话

**ChatGPT agent（OpenAI）。** 2025 年 7 月发布。统一了 Operator（网页浏览）和 Deep Research（数小时级研究）。2025 年 8 月 31 日关停了独立的 Operator。在 BrowseComp 上以 68.9% 达到 SOTA；在 OSWorld 和 WebArena-Verified 上也有亮眼成绩。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic 收购 Vercept，瞄准的是计算机操作（computer-use）能力。这让 Claude Sonnet 在 OSWorld 上的成绩从 <15% 提升到 72.5%。Claude Computer Use 以工具 API 的形式交付。

**Gemini 3 Pro with Browser Use（DeepMind）。** Browser Use 集成提供了计算机操作控制能力；FSF v3（2026 年 4 月，第 20 课）专门跟踪 ML 研发领域中的自主性。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 修复了一个早有定论的问题：原版 WebArena 存在约 11.3% 的假阴性率（实际已完成的任务被判为失败）。Verified 版本用人工校验过的成功标准重新评分，并新增了包含 258 个任务的 Hard 子集（ICLR 2026 论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs OSWorld vs WebArena

| 基准 | 衡量内容 | 时间跨度 |
|---|---|---|
| BrowseComp | 在时间压力下从开放网络中查找特定事实 | 分钟级 |
| OSWorld | 智能体操作完整桌面（鼠标、键盘、shell） | 数十分钟级 |
| WebArena-Verified | 在模拟站点中完成事务型网页任务 | 分钟级 |
| Hard 子集 | WebArena-Verified 中涉及多页面状态转移的任务 | 数十分钟级 |

这些是不同的维度。BrowseComp 高分只能说明智能体擅长查找事实，并不能说明它会订机票。OSWorld 的分数更接近"它能不能在我的桌面上干活"。WebArena-Verified 更接近"它能不能走完一个流程"。任何生产决策都需要选择与实际任务分布相匹配的基准。

### 攻击面，逐一点名

1. **间接提示注入（indirect prompt injection）。** 不可信的页面内容里包含指令。智能体读到了，智能体就执行了。公开案例：2024 年 Kai Greshake 等人的工作、2025 年 Tainted Memories 论文、2026 年 HashJack（Cato Networks）。
2. **URL 片段 / 查询串注入。** 被抓取 URL 的 `#fragment` 或查询字符串中藏有命令。它们从不会被可见地渲染，却仍在智能体的上下文之内。
3. **记忆绑定攻击。** 页面指示智能体写入一条持久化记忆（第 12 课讲持久状态）。下一个会话中，这条记忆在没有任何可见触发的情况下引爆载荷。
4. **针对已认证会话的 CSRF 式攻击。** Tainted Memories 一类：智能体在某处处于登录态；攻击者的页面发起改变状态的请求，智能体带着用户的 cookie 替它执行。
5. **一键劫持。** 一个看起来人畜无害的按钮搭载着后续载荷，智能体顺势执行。Comet 一类。
6. **智能体宿主层面的 Content-Security-Policy 漏洞。** 渲染层和工具层本身也可以成为攻击向量；"浏览器套浏览器"的智能体技术栈攻击面很宽。

### 为什么"无法彻底修复"

这类攻击与智能体的能力本身是同构的。智能体必须读取不可信内容才能完成工作；它读到的任何内容都可能包含指令；它执行的任何指令都可能偏离用户的真实意图。各类防御（信任边界、分类器、工具白名单、对高后果动作的人工审批）能抬高攻击成本、缩小爆炸半径，但无法把这一整类攻击关死。

这与勒布定理（Lob's theorem，第 8 课）背后的推理模式相同：智能体无法证明下一个 token 是安全的；它只能搭建一个让不安全 token 更容易被发现的系统。

### 真正能落地的防御姿态

- **读 / 写边界。** 读取永远不产生后果。写入（提交表单、发布内容、调用有副作用的工具）若由信任边界之外的内容触发，必须获得新一轮人工批准。
- **按任务划定工具白名单。** 智能体可以浏览网页，但除非某个工具被显式为该任务启用，否则它不能发起电汇。第 13 课讲预算。
- **会话隔离。** 浏览器智能体会话只使用受限作用域的凭证运行。不接触生产环境认证，不接触个人邮箱。每一条 HTTP 请求的日志都留存备查。
- **内容净化器。** 抓取到的 HTML 在拼接进模型上下文之前，先剔除已知的恶意模式。（能挡住低门槛攻击；挡不住精心构造的载荷。）
- **高后果动作上的人工介入（HITL）。** 先提议、后提交的模式（第 15 课）。
- **记忆上的金丝雀令牌（canary token）。** 一旦某条记忆条目被触发，用户能看到（第 14 课）。

## 生产实践

`code/main.py` 模拟了一次小型浏览器智能体运行，目标是三个合成页面。一个页面是良性的，一个在可见文本中嵌入了直接提示注入片段，一个带有 URL 片段注入（不可见，但位于智能体的上下文中）。脚本展示了：(a) 一个天真的智能体会怎么做，(b) 读/写边界能拦住什么，(c) 净化器能拦住什么，(d) 两者都拦不住什么。

## 交付产物

`outputs/skill-browser-agent-trust-boundary.md` 对一个拟议的浏览器智能体部署进行界定：它触及哪些信任区域、被授权写入什么，以及首次运行前必须就位哪些防御。

## 练习

1. 运行 `code/main.py`。找出哪种攻击净化器能拦住而读/写边界拦不住，又有哪种攻击只有读/写边界能拦住。

2. 扩展净化器，使其能检测一类 HashJack 式的 URL 片段注入。在带有合法片段的良性 URL 上测量误报率。

3. 选一个你熟悉的真实浏览器智能体工作流（例如"订一张机票"）。列出其中的每一次读和每一次写。标出哪些写操作需要 HITL，并说明原因。

4. 阅读 WebArena-Verified 的 ICLR 2026 论文。找出原版 WebArena 评分不可靠的一类任务，并解释 Verified 子集是如何解决的。

5. 为浏览器智能体场景设计一个记忆金丝雀。你会存什么、存在哪里、什么情况触发警报？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| 间接提示注入 | "页面文字有毒" | 智能体读取的页面中的不可信内容包含指令，智能体执行了这些指令 |
| Tainted Memories | "记忆攻击" | 智能体把攻击者提供的指令写入持久记忆；下一个会话中被触发 |
| HashJack | "URL 片段攻击" | 藏在 URL 片段 / 查询字符串中的载荷位于智能体上下文中，但不会被可见地渲染 |
| 一键劫持 | "坏按钮" | 一个可见的交互元素搭载后续载荷，智能体顺势执行 |
| BrowseComp | "网页搜索基准" | 在开放网络中查找特定事实；分钟级时间跨度 |
| OSWorld | "桌面基准" | 完整操作系统控制；多步 GUI 任务 |
| WebArena-Verified | "修复版网页任务基准" | ServiceNow 重新评分的 WebArena，附带 Hard 子集 |
| 读/写边界 | "副作用闸门" | 读取永远不产生后果；若内容来自信任边界之外，写入需要新一轮批准 |

## 延伸阅读

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator 与 deep research 的合并；BrowseComp SOTA。
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — Operator 的血统，以及后来演变成 ChatGPT agent 的架构。
- [Zhou et al. — WebArena](https://webarena.dev/) — 原版基准。
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 修复子集论文。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 包含对计算机操作智能体攻击面的讨论。
