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

## 配置项(环境变量)

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `CLAUDE_CODE_OAUTH_TOKEN` | (必填) | `claude setup-token` 生成的订阅 token,**只在服务端** |
| `TUTOR_MODEL` | `claude-sonnet-4-6` | 模型 |
| `TUTOR_EFFORT` | `medium` | 推理强度 low/medium/high |
| `TUTOR_ALLOWED_ORIGINS` | Pages + localhost | CORS allowlist,逗号分隔 |
| `TUTOR_BEARER` | 空 | 设了就要求前端带 `Authorization: Bearer`(前端值公开,仅作弱校验) |
| `TUTOR_RATE_PER_MIN` | `20` | 每 IP 每分钟请求上限 |

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
