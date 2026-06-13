# AI 辅导服务端

把 ai-eng-studio 的 AI 辅导浮窗接到**你自己的 Claude 订阅**。服务端在 Docker 里跑
headless Claude Code,前端通过 SSE 流式拿回答。完整接口见
[`docs/ai-tutor-server-contract.md`](../../docs/ai-tutor-server-contract.md)。

## 文件

| 文件 | 作用 |
| --- | --- |
| `server.py` | FastAPI 参考实现:`POST /chat` → 调 `claude -p` → SSE |
| `Dockerfile` / `docker-compose.yml` | 容器化(Node + Python + claude CLI) |
| `mock_server.py` | 无需 Claude 的假服务,用来先验证前端连通 |
| `requirements.txt` | Python 依赖 |

## 跑起来(3 步)

```bash
# 1) 用你的 Claude 订阅登录并生成长期 token(一次,token 约一年有效)
claude setup-token

# 2) 填 .env 并启动
cd server/tutor
printf 'CLAUDE_CODE_OAUTH_TOKEN=%s\n' '<上一步输出的 token>' > .env
docker compose up -d --build        # 监听 0.0.0.0:8787,端点 /chat

# 健康检查
curl http://localhost:8787/health   # {"ok":true,"model":"claude-sonnet-4-6","effort":"medium"}
```

然后给前端设端点并重新部署:

```bash
# 在前端项目根目录
echo 'VITE_AI_TUTOR_ENDPOINT=https://<你的域名或IP>/chat' >> .env.local   # 本地开发
# 生产:在 CI/构建环境注入同名变量,再 npm run deploy:build
```

> 生产建议在前面放一层 HTTPS 反代(Caddy/Nginx/Cloudflare),把 8787 暴露成
> `https://your-host/chat`,并确认 `TUTOR_ALLOWED_ORIGINS` 含
> `https://darkness-hy.github.io`。

## 多用户并发与隔离

设计为**多人同时使用、彼此信息隔离**:

- **隔离是结构性的**:服务端**不保存任何用户的对话**。每个请求自带 `history`,每个 `/chat`
  各 fork 一个 `claude` 子进程、各自的 prompt 与流。两个用户即便同时用同一个订阅,也**不可能**
  看到对方的内容——靠"服务端无记忆"实现,比加锁更可靠。
- **并发上限**:`TUTOR_MAX_CONCURRENCY`(默认 3)限制同时在跑的 `claude` 调用,避免一个班的
  突发把订阅瞬间打到限速。超出的请求排队,排队超过 `TUTOR_MAX_QUEUE`(默认 15)就返回
  `429 服务繁忙`,前端显示「助教正忙,请稍后再试」。
- **超时保护**:单个请求超过 `TUTOR_TIMEOUT`(默认 120s)即终止子进程、释放并发槽。
- **对话留存**:每完成一轮,向 `TUTOR_LOG_DIR/chat-YYYY-MM-DD.jsonl` 追加一行
  `{ts, ip, lesson_id, message, reply, status, ms}`,便于排查。docker compose 已把
  `./logs` 挂出容器,在宿主机直接看。
- **硬天花板**:所有人共享你**一个** Claude 订阅的速率额度。一个班(几十人偶发提问)单订阅
  够;真要几百人同时,需调高并发并准备多 token 轮换或改用 API key。

`/health` 返回里有 `inflight`(当前在跑数)和 `max_concurrency`,可用来观察负载。

## 配置项(环境变量)

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `CLAUDE_CODE_OAUTH_TOKEN` | (必填) | `claude setup-token` 生成的订阅 token,**只在服务端** |
| `TUTOR_MODEL` | `claude-sonnet-4-6` | 模型 |
| `TUTOR_EFFORT` | `medium` | 推理强度 low/medium/high |
| `TUTOR_MAX_CONCURRENCY` | `3` | 同时在跑的 claude 调用上限 |
| `TUTOR_MAX_QUEUE` | `15` | 排队等待上限,超出回 429 |
| `TUTOR_TIMEOUT` | `120` | 每请求秒数上限,超时终止并释放槽 |
| `TUTOR_LOG_DIR` | `logs` | 对话 JSONL 目录,设为空字符串可关闭留存 |
| `TUTOR_ALLOWED_ORIGINS` | Pages + localhost | CORS allowlist,逗号分隔 |
| `TUTOR_BEARER` | 空 | 设了就要求前端带 `Authorization: Bearer`(前端值公开,仅作弱校验) |
| `TUTOR_RATE_PER_MIN` | `0`(关闭) | >0 时启用每 IP 每分钟上限(代理后需正确透传 X-Forwarded-For) |

## 先验证前端(可选,不接 Claude)

```bash
python3 server/tutor/mock_server.py 8765
# 另开一个终端
VITE_AI_TUTOR_ENDPOINT=http://localhost:8765/chat npm run dev
# 打开任意课程页,右下角出现 AI 辅导浮窗,提问会看到流式假回答
```

## 备选:用 Agent SDK 而非 CLI

`server.py` 走的是 `claude -p ... --output-format stream-json`(接口最稳)。若想用
Python Agent SDK,把 `pip install claude-agent-sdk`,用
`query(prompt=..., options=ClaudeAgentOptions(model=MODEL, effort=EFFORT, allowed_tools=[],
system_prompt={"type":"preset","preset":"claude_code","append":sys_prompt},
include_partial_messages=True))` 遍历消息,取 `content_block_delta` 里的 `text_delta`
转成同样的 SSE 帧即可。鉴权同样用 `CLAUDE_CODE_OAUTH_TOKEN`。
