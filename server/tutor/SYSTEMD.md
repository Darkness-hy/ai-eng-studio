# 裸装部署(不用 Docker)+ systemd 常驻

Docker 在这里只是「替你装好 Python 依赖」。你自己装 Python + venv,就不需要 Docker。

## 一、装运行时(全新服务器,一次)

```bash
# Python ≥ 3.9(server.py 已兼容 3.9)。多数 Linux 自带,确认一下:
python3 --version
```

## 二、先前台跑通(在 `tutor/` 目录)

`.env` 需要包含 DeepSeek API key:

```bash
printf 'DEEPSEEK_API_KEY=%s\n' '<你的 DeepSeek API key>' > .env
```

然后直接:

```bash
chmod +x run.sh
./run.sh            # 首次会建 .venv 装依赖,然后启动 :8787
# 另开一个终端验证:
curl localhost:8787/health     # 看到 max_concurrency/inflight 即正常
```

`run.sh` 干的事:检查 python → 建 venv 装依赖 → 把 `.env` 的 API key 导入环境
→ `uvicorn server:app`。

## 三、常驻(systemd,开机自启 + 崩溃自拉)

把下面存成 `/etc/systemd/system/tutor.service`,**把 `youruser` 和路径换成你的**
(假设代码在 `/home/youruser/.../server/tutor`):

```ini
[Unit]
Description=ai-eng-studio AI tutor
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser/ai-eng-studio/server/tutor
EnvironmentFile=/home/youruser/ai-eng-studio/server/tutor/.env
ExecStart=/home/youruser/ai-eng-studio/server/tutor/run.sh
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

> `run.sh` 会自动创建/更新 `.venv` 并加载 `.env`。`EnvironmentFile=.env` 也保留在 systemd
> 层,便于 `systemctl show`/日志排查时确认环境来源。

启用:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now tutor
sudo systemctl status tutor           # 看是否 active (running)
journalctl -u tutor -f                # 实时日志
```

改了代码后:`git pull` → `sudo systemctl restart tutor`。

## 四、对话日志在哪

`server.py` 把每轮问答写到 `WorkingDirectory/logs/chat-YYYY-MM-DD.jsonl`,即
`tutor/logs/`。排查时直接 `tail -f logs/chat-*.jsonl`。

## 五、还差一步:HTTPS

裸装只是把服务跑在 `:8787`(HTTP)。线上 HTTPS 页面调它仍需 HTTPS 端点——
用 Caddy 反代或 cloudflared 隧道(见 `../../docs/ai-tutor-server-contract.md` §Part D),
与用不用 Docker 无关。
