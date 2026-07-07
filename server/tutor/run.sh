#!/usr/bin/env bash
# 裸装运行 AI 辅导服务端(不依赖 Docker)。幂等:首次会建 venv 并装依赖,
# 之后直接加载 .env 并启动 uvicorn。在 server/tutor/ 目录下执行: ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

# 1) 运行时检查
command -v python3 >/dev/null || { echo "✗ 需要 python3 (≥3.9)"; exit 1; }

# 2) venv + 依赖
if [ ! -d .venv ]; then
  echo "→ 建 venv…"
  python3 -m venv .venv
fi
echo "→ 安装/更新依赖…"
.venv/bin/python -m pip install -q -r requirements.txt

# 3) 从 .env 导入模型 API key
[ -f .env ] || { echo "✗ 缺少 .env(里面要有 DEEPSEEK_API_KEY=...)"; exit 1; }
set -a; . ./.env; set +a
if [ "${TUTOR_PROVIDER:-deepseek}" != "claude" ] && [ -z "${DEEPSEEK_API_KEY:-${TUTOR_API_KEY:-}}" ]; then
  echo "✗ .env 里没有 DEEPSEEK_API_KEY 或 TUTOR_API_KEY"
  exit 1
fi

# 4) 启动(exec 让信号直达 uvicorn,便于被 systemd/Ctrl-C 管理)
echo "→ 启动 http://0.0.0.0:${PORT:-8787}  (Ctrl-C 停止)"
exec .venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port "${PORT:-8787}"
