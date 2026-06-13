# AI 辅导 · 服务端契约

前端的 AI 辅导浮窗是**休眠**的:只有当你在构建时设置了 `VITE_AI_TUTOR_ENDPOINT`,
浮窗才会出现。它把问题发到你自建的服务器,服务器用**你自己的 Claude 订阅**(在 Docker 里
跑 headless Claude Code)生成回答,流式返回。本文件是你那台服务器必须实现的接口。

参考实现就在 `server/tutor/`(FastAPI + claude-agent-sdk + Docker),照着跑即可。

---

## 1. HTTP 接口

**一个端点即可:** `POST {VITE_AI_TUTOR_ENDPOINT}`(例如 `https://your-host/chat`)。

### 请求

`Content-Type: application/json`。可选 `Authorization: Bearer <token>`(见 §4)。

```jsonc
{
  "message": "线性代数在这门课里为什么重要?",   // 必填:用户这轮的问题
  "history": [                                  // 选填:之前的对话(不含本轮)
    { "role": "user", "content": "上一条问题" },
    { "role": "assistant", "content": "上一条回答" }
  ],
  "lesson_id": "00-setup-and-tooling/01-dev-environment", // 选填:当前所在课(没有则 null)
  "context": "# 开发环境搭建\n\n<课文正文,已被前端截断到 ~4000 字>", // 选填:前端注入的 RAG 上下文
  "lang": "zh",                                 // "zh" | "en":期望的回答语言
  "stream": true                               // 前端总是传 true
}
```

字段说明:
- `message` —— 本轮问题,**必填**。
- `history` —— 多轮对话历史,按时间顺序;服务端应原样拼进对话。
- `lesson_id` / `context` —— **RAG 注入**,见 §3。两者都可能为 `null`(用户在非课程页提问)。
- `lang` —— 服务端应据此要求模型用中文或英文回答。
- `stream` —— 固定 `true`;前端按 SSE 解析。

### 响应(`stream: true`)

`Content-Type: text/event-stream`。按 SSE 规范,每个事件是 `data: <json>\n\n`。事件类型:

```
data: {"type":"delta","text":"线性"}

data: {"type":"delta","text":"代数"}

data: {"type":"done"}
```

- `{"type":"delta","text":"..."}` —— 一段增量文本;前端按到达顺序拼接显示。
- `{"type":"done"}` —— 本轮结束;前端停止等待。可省略(连接关闭也视为结束)。
- `{"type":"error","message":"..."}` —— 出错;前端把 `message` 显示为红色错误条并中止本轮。

> 前端解析器只认 `data:` 行,按空行 `\n\n` 分帧,JSON 解析失败的行直接跳过——
> 所以你可以安全地发心跳注释行(以 `:` 开头)或空 `data:` 行保活,前端会忽略。

### 响应(`stream: false`,可选)

如果将来要支持非流式,返回 `application/json`:`{ "reply": "完整回答" }`。
当前前端不会用到,但实现了无妨。

---

## 2. CORS(必须)

浮窗从浏览器**跨源**调用你的服务器,所以你必须处理 CORS,否则请求会被浏览器拦掉:

- 响应头 `Access-Control-Allow-Origin` 要包含你的站点源。生产是
  `https://darkness-hy.github.io`;本地开发是 `http://localhost:5180`。
  建议**精确 allowlist 这两个**,而不是 `*`(尤其当你启用了 `Authorization`)。
- 处理预检 `OPTIONS`:返回 `204`,带
  `Access-Control-Allow-Methods: POST, OPTIONS` 和
  `Access-Control-Allow-Headers: Content-Type, Authorization`。

参考实现用 FastAPI 的 `CORSMiddleware` 已经配好。

---

## 3. RAG 上下文注入

课程内容**前端本来就有**(全部是静态 JSON)。所以前端做了一层**轻量 RAG**:
当用户在某节课上提问时,前端把**当前课文**(标题 + 正文截断到 ~4000 字)放进 `context`,
并带上 `lesson_id`。你的服务端有两种用法,任选其一或叠加:

1. **直接用前端给的 `context`**(最省事,MVP 推荐)
   把 `context` 拼进发给 Claude 的提示里(参考实现就是这么做的)。即使你的服务端没有任何
   向量库,辅导也能"看到"当前这节课的内容。

2. **服务端自建 RAG**(更强,可选)
   你可以忽略前端 `context`,改用 `lesson_id` 去你自己的向量库/全文索引里检索更相关的多段
   内容(跨课、术语表等)再注入。课程 JSON 可从本仓库 `public/data/` 拿到:
   `index.json`(目录)、`lessons/<phase>/<lesson>.json`(每课全文 + 测验 + 代码)、
   `glossary.json`(术语)。建好索引后按 `lesson_id`/语义检索即可。

无论哪种,系统提示都应把模型**定位成"这门课的助教"**:基于课程内容回答,讲不清就举例、
拆步骤,不要编造课程里没有的 API。

---

## 4. 鉴权与防滥用

- 端点是公开的(静态站点谁都能调)。`VITE_AI_TUTOR_TOKEN` 会作为 `Authorization: Bearer`
  发出,但它**打包进了前端,是公开可见的**——别把它当机密。
- 真正的防滥用应在**服务端**做(参考实现**默认即开启**):
  - 限流:每 IP `TUTOR_RATE_PER_MIN`(默认 20/min)+ 跨所有 IP 的全局
    `TUTOR_RATE_GLOBAL_PER_MIN`(默认 60/min)——全局闸门才是保护你**唯一订阅额度**的关键,
    因为每 IP 限流可被轮换/伪造源 IP 绕过;
  - 请求体硬上限:`message ≤ 4000`、`history ≤ 50 条`(每条 `content ≤ 8000`)、`context ≤ 16000`、
    `user_profile ≤ 4000` 字符,超限在 spawn `claude` 之前直接返回 `422`;
  - `TUTOR_TRUST_PROXY`(默认 1)信任反代的 `X-Forwarded-For`;若 `:8787` 可被直连绕过反代,
    设为 `0`,否则每 IP 限流可被伪造头绕过;
  - 校验 `Origin` 头是否在你的 allowlist(配合 CORS);
  - 可选:在你和服务器之间放 Cloudflare/反代,加 WAF 与速率限制。
- 你的 **Claude 订阅 token 只存在服务端**(`CLAUDE_CODE_OAUTH_TOKEN`),永远不进前端。

---

## 4b. 多用户并发与隔离

前端是多人同时使用的,服务端必须能并发处理且彼此隔离。参考实现的做法(你自建时建议照做):

- **隔离靠无状态**:服务端**不保存任何用户的对话**。每个请求自带 `history`,各自 fork 一个
  `claude` 子进程、各自的 prompt 与 SSE 流,进程间无共享对话变量。所以即便多人同时用同一个
  订阅,也无法看到彼此内容——这比加锁更可靠,且天然满足"信息隔离"。
- **并发上限 + 排队**:用信号量限制同时在跑的 `claude` 调用(参考实现默认 3),避免突发把
  单个订阅打到限速;超出排队,排队过深直接回 `429`,前端提示"稍后再试"。
- **超时**:单请求超时即终止子进程,释放并发槽,防止卡死的请求长期占位。
- **一个订阅是硬天花板**:所有用户共享你那一份订阅的速率额度。隔离与正确性不受影响,但
  "同时能服务多少人"由订阅额度决定。

## 5. 模型与推理强度

- 默认模型 **`claude-sonnet-4-6`**,推理强度 **medium**。
- headless 调用(prompt 经 **stdin** 传入,避免参数长度限制):
  `claude -p --model claude-sonnet-4-6 --effort medium --tools "" --system-prompt "<完整系统提示>" --output-format stream-json --verbose --include-partial-messages`
  - `--tools ""` 禁用所有工具(纯文本问答,不读写文件、不执行命令);
  - `--effort medium` 中等推理强度;
  - **`--system-prompt`(整段替换)而非 `--append-system-prompt`(追加)**:claude CLI 是 Bun 编译的二进制,默认会带上 Claude Code 自己的系统提示(实测 ≈7870 token)。整段替换后只剩 ≈2065 token 的强制底座,每次请求省 ≈5800 token(详见 §5b);
  - `--output-format stream-json --verbose --include-partial-messages` 输出逐行 JSON(stream-json 要求 `--verbose`),服务端解析 `text_delta` 转成 SSE。
- 参考实现走的是 **CLI 子进程**(`asyncio.create_subprocess_exec`),不是 SDK;也可换用 Claude Agent SDK(Python/TS)。
- 中国大陆服务器要让 claude 出网,设**小写** `https_proxy`(Bun 只认小写,大写 `HTTPS_PROXY` 无效)。

---

## 5b. 助教的完整输入与 token 预算

每次 `/chat`,模型实际收到的输入由这几块拼成。Token 数为在 claude CLI 上**实测**(中文+代码内容约 0.58 token/字):

| 部分 | 内容 | Token |
| --- | --- | --- |
| Claude Code 强制底座 | 用 `--system-prompt` 后无法再去掉的部分 | ≈2065 |
| 助教人设 | 角色 + 回答语言指令 | ≈137 |
| 平台背景 | 课程定位 + 平台功能 + 20 阶段课程地图(由 `index.json` 动态生成) | ≈430 |
| `<learner>` 用户画像 | 昵称/定级/进度/测验/活跃/薄弱领域/低分课/徽章(前端 `buildUserProfile` 汇总) | ≈55–100 |
| `<course>` 当前课文 | 整篇课文 + 本课测验要点(前端 `lessonContext`,封顶 12000 字) | ≈2160(典型课;长课最多 ≈7000) |
| 用户消息 | 近 8 轮历史 + 本次提问 | ≈100–600/轮 |

- **典型单次请求 ≈ 5,200 token 输入**(本课 + 进阶画像 + 1 轮历史)。
- 历史:早期用 `--append-system-prompt` 时,仅 Claude Code 默认框架就占 ≈7870 token;切到 `--system-prompt` 后系统底座降到 ≈2065,整条请求从 ≈10,600 腰斩到 ≈5,200。
- **缓存**:底座、人设、平台背景、当前课文在多轮对话间稳定 → 命中 Anthropic prompt 缓存,按 ~10% 计费;用户画像与提问每轮变化,不缓存。
- 拼装位置:系统提示见 `server.py` 的 `system_prompt()` 与 `load_project_background()`;前端两块上下文见 `src/lib/tutor.ts` 的 `lessonContext()`(课文)与 `buildUserProfile()`(画像)。

系统提示结构(`--system-prompt` 整段传入):

```
[人设]      你是《从零开始的 AI 工程》的助教… 用简体中文回答。
[平台背景]   【关于本平台】…20 阶段课程地图…
<learner>   昵称/定级/进度/测验/活跃/薄弱领域/低分课/徽章
</learner>
<course>    # 课标题 + 整篇课文 + 本课测验要点
</course>
```

用户消息(经 stdin):近 8 轮 `学习者:…/助教:…` 历史 + 本次提问。

---

## 6. 部署速查

```bash
# 1) 在你的机器上,用 Claude 订阅登录并生成长期 token(一次)
claude setup-token                 # 输出一个一年有效的 OAuth token

# 2) 起服务(Docker)
cd server/tutor
echo "CLAUDE_CODE_OAUTH_TOKEN=<上一步的 token>" > .env
docker compose up -d --build       # 默认监听 :8787,端点为 /chat

# 3) 前端对接:在构建环境设
#    VITE_AI_TUTOR_ENDPOINT=https://<你的域名或IP>/chat
#    然后 npm run deploy:build 重新部署
```

想先不接 Claude、只验证前端连通?用 mock:

```bash
python3 server/tutor/mock_server.py 8765
VITE_AI_TUTOR_ENDPOINT=http://localhost:8765/chat npm run dev
```
