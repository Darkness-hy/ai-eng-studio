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
- 真正的防滥用应在**服务端**做:
  - 校验 `Origin` 头是否在你的 allowlist(配合 CORS);
  - 按 IP/会话**限流**(参考实现给了一个简单的内存限流示例);
  - 可选:在你和服务器之间放 Cloudflare/反代,加 WAF 与速率限制。
- 你的 **Claude 订阅 token 只存在服务端**(`CLAUDE_CODE_OAUTH_TOKEN`),永远不进前端。

---

## 5. 模型与推理强度

- 默认模型 **`claude-sonnet-4-6`**,推理强度 **medium**(用户要求)。
- headless 调用:`claude -p "<prompt>" --model claude-sonnet-4-6 --effort medium --tools "" --output-format stream-json`
  - `--tools ""` 禁用所有工具(纯文本问答,不读写文件、不执行命令);
  - `--effort medium` 设中等推理强度;
  - `--output-format stream-json` 输出逐行 JSON,服务端解析其中的 `text_delta` 转成 SSE。
- 也可用 **Claude Agent SDK**(Python/TS)替代 shell 调用,参考实现用的就是 Python SDK。

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
