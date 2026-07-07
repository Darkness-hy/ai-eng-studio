# AI 辅导服务端

把 ai-eng-studio 的 AI 辅导浮窗接到 **DeepSeek API**。服务端通过
OpenAI-compatible streaming API 调模型,前端通过 SSE 流式拿回答。完整接口见
[`docs/ai-tutor-server-contract.md`](../../docs/ai-tutor-server-contract.md)。

## 文件

| 文件 | 作用 |
| --- | --- |
| `server.py` | FastAPI 参考实现:`POST /chat` → DeepSeek streaming → SSE |
| `Dockerfile` / `docker-compose.yml` | 容器化 Python 服务 |
| `mock_server.py` | 无需 Claude 的假服务,用来先验证前端连通 |
| `requirements.txt` | Python 依赖 |

## 跑起来(3 步)

```bash
# 1) 填 DeepSeek API key 并启动
cd server/tutor
printf 'DEEPSEEK_API_KEY=%s\n' '<你的 DeepSeek API key>' > .env
docker compose up -d --build        # 监听 0.0.0.0:8787,端点 /chat

# 健康检查
curl http://localhost:8787/health   # {"ok":true,"provider":"deepseek","model":"deepseek-v4-pro",...}
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
  都构造独立 prompt 并发起独立模型请求。两个用户即便同时使用同一个 API key,也不会共享服务端会话。
- **并发上限**:`TUTOR_MAX_CONCURRENCY`(默认 3)限制同时在跑的模型调用。超出的请求排队,
  排队超过 `TUTOR_MAX_QUEUE`(默认 15)就返回
  `429 服务繁忙`,前端显示「助教正忙,请稍后再试」。
- **超时保护**:单个请求超过 `TUTOR_TIMEOUT`(默认 120s)即中止流式请求、释放并发槽。
- **对话留存**:每完成一轮,向 `TUTOR_LOG_DIR/chat-YYYY-MM-DD.jsonl` 追加一行。
  **默认只记元数据** `{ts, ip, lesson_id, lang, status, ms}`(不落问答正文,避免隐私/密钥留存);
  设 `TUTOR_LOG_FULL=1` 才额外记录 `message`+`reply`。超过 `TUTOR_LOG_RETAIN_DAYS`(默认 14 天)
  的文件会被自动清理。docker compose 已把 `./logs` 挂出容器;请勿在助教对话里粘贴密钥/口令。
- **硬天花板**:所有人共享你**一个** DeepSeek API key 的速率和余额。一个班(几十人偶发提问)
  通常够用;真要几百人同时,需调高并发并准备额度/限流策略。

`/health` 返回里有 `inflight`(当前在跑数)和 `max_concurrency`,可用来观察负载。

## 配置项(环境变量)

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `TUTOR_PROVIDER` | `deepseek` | 默认 DeepSeek/OpenAI-compatible;设 `claude` 才走旧 Claude CLI |
| `DEEPSEEK_API_KEY` | (必填) | DeepSeek API key,**只在服务端** |
| `TUTOR_API_KEY` | 空 | 通用 API key 覆盖项;优先级高于 `DEEPSEEK_API_KEY` |
| `TUTOR_API_BASE` | `https://api.deepseek.com` | OpenAI-compatible base URL |
| `TUTOR_MODEL` | `deepseek-v4-pro` | 模型 |
| `TUTOR_TEMPERATURE` | `0.3` | 生成温度 |
| `TUTOR_THINKING` | `enabled` | DeepSeek thinking mode;设 `disabled` 可降低延迟 |
| `TUTOR_REASONING_EFFORT` | `medium` | DeepSeek 推理强度 |
| `TUTOR_MAX_TOKENS` | 空 | 可选输出 token 上限 |
| `TUTOR_EFFORT` | `medium` | 仅旧 Claude CLI provider 使用 |
| `TUTOR_MAX_CONCURRENCY` | `3` | 同时在跑的模型调用上限 |
| `TUTOR_MAX_QUEUE` | `15` | 排队等待上限,超出回 429 |
| `TUTOR_TIMEOUT` | `120` | 每请求秒数上限,超时终止并释放槽 |
| `TUTOR_LOG_DIR` | `logs` | 对话 JSONL 目录,设为空字符串可关闭留存 |
| `TUTOR_LOG_FULL` | `0` | `0` 只记元数据;`1` 额外记录问答正文(含隐私) |
| `TUTOR_LOG_RETAIN_DAYS` | `14` | 自动删除早于 N 天的日志(`0` = 永久保留) |
| `TUTOR_ALLOWED_ORIGINS` | Pages + localhost | CORS allowlist,逗号分隔 |
| `TUTOR_BEARER` | 空 | 设了就要求前端带 `Authorization: Bearer`(前端值公开,仅作弱校验) |
| `TUTOR_RATE_PER_MIN` | `20` | 每 IP 每分钟上限(`0` 关闭) |
| `TUTOR_RATE_GLOBAL_PER_MIN` | `60` | 跨所有 IP 的每分钟总上限,保护唯一订阅额度(`0` 关闭) |
| `TUTOR_TRUST_PROXY` | `1` | 信任反代的 `X-Forwarded-For`;`:8787` 可被直连时设 `0` |

## 先验证前端(可选,不接 Claude)

```bash
python3 server/tutor/mock_server.py 8765
# 另开一个终端
VITE_AI_TUTOR_ENDPOINT=http://localhost:8765/chat npm run dev
# 打开任意课程页,右下角出现 AI 辅导浮窗,提问会看到流式假回答
```

## 旧 Claude CLI fallback

`server.py` 仍保留旧分支:设 `TUTOR_PROVIDER=claude`、安装 `claude` CLI 并提供
`CLAUDE_CODE_OAUTH_TOKEN` 后,会回到原来的 `claude -p --output-format stream-json` 路径。
默认 Docker 镜像不再安装 Node/Claude CLI。
