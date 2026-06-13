#!/usr/bin/env bash
# 裸装运行 AI 辅导服务端(不依赖 Docker)。幂等:首次会建 venv 并装依赖,
# 之后直接加载 .env 并启动 uvicorn。在 server/tutor/ 目录下执行: ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

# 1) 运行时检查
command -v python3 >/dev/null || { echo "✗ 需要 python3 (≥3.9)"; exit 1; }
command -v node    >/dev/null || { echo "✗ 需要 node (≥18) 来跑 claude CLI,先装 Node"; exit 1; }
command -v claude  >/dev/null || { echo "✗ 需要 claude CLI: npm i -g @anthropic-ai/claude-code"; exit 1; }

# 2) venv + 依赖(只在首次)
if [ ! -d .venv ]; then
  echo "→ 建 venv 并安装依赖…"
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi

# 3) 从 .env 导入订阅 token(claude 子进程会继承这个环境变量)
[ -f .env ] || { echo "✗ 缺少 .env(里面要有 CLAUDE_CODE_OAUTH_TOKEN=...)"; exit 1; }
set -a; . ./.env; set +a
[ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] || { echo "✗ .env 里没有 CLAUDE_CODE_OAUTH_TOKEN"; exit 1; }

# 4) 启动(exec 让信号直达 uvicorn,便于被 systemd/Ctrl-C 管理)
echo "→ 启动 http://0.0.0.0:${PORT:-8787}  (Ctrl-C 停止)"
exec .venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port "${PORT:-8787}"
