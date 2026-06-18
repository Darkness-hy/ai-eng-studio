#!/usr/bin/env bash
# One-shot dev setup for ai-eng-studio on a fresh machine (Ubuntu/macOS).
# Run from the repo root:  ./setup.sh
# It never copies node_modules across machines — always installs fresh.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> ai-eng-studio setup"

# 1) Node >= 18
if ! command -v node >/dev/null 2>&1; then
  echo "✗ Node.js not found. On Ubuntu:"
  echo "    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt install -y nodejs"
  echo "  then re-run ./setup.sh"
  exit 1
fi
NODE_MAJOR=$(node -p "process.versions.node.split('.')[0]")
if [ "$NODE_MAJOR" -lt 18 ]; then
  echo "✗ Node $(node -v) is too old; need >= 18 (20 recommended)."
  exit 1
fi
echo "✓ Node $(node -v)"

# 2) Dependencies (fresh install — node_modules is OS-specific, never copy it over)
echo "==> npm install …"
npm install

# 3) .env.local — cloud/auth/AI-tutor are optional; without it the app runs local-only
if [ ! -f .env.local ]; then
  if [ -f .env.example ]; then
    cp .env.example .env.local
    echo "⚠ created .env.local from .env.example — fill in VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY"
    echo "  (Supabase → Settings → API) and VITE_AI_TUTOR_ENDPOINT, or leave blank for local-only mode."
  else
    echo "⚠ no .env.local — running in local-only mode (no cloud/auth/tutor)."
  fi
else
  echo "✓ .env.local present"
fi

# 4) Upstream course repo — only needed to RE-RUN build:content (public/data is committed)
if [ ! -d ../ai-engineering-from-scratch ]; then
  echo "ℹ upstream course repo not at ../ai-engineering-from-scratch"
  echo "  dev works without it (public/data is committed). Clone only if regenerating content:"
  echo "    (cd .. && git clone https://github.com/rohitg00/ai-engineering-from-scratch.git)"
fi

# 5) Sanity check
echo "==> verifying (tsc + lint) …"
if npx tsc -b >/dev/null 2>&1; then echo "✓ typecheck"; else echo "⚠ typecheck issues — run: npx tsc -b"; fi
if npm run lint >/dev/null 2>&1; then echo "✓ lint"; else echo "⚠ lint issues — run: npm run lint"; fi

echo ""
echo "✅ Done."
echo "   Dev:    npm run dev    →  http://localhost:5180"
echo "   Build:  npm run build"
echo "   Deploy: npm run deploy:build,  then force-push dist/ to the gh-pages branch"
