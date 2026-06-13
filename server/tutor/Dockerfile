# AI-tutor server: Python (FastAPI) + headless Claude Code CLI.
FROM node:22-slim

# Python for the FastAPI app
RUN apt-get update \
  && apt-get install -y --no-install-recommends python3 python3-pip python3-venv ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Headless Claude Code CLI (authenticated at runtime via CLAUDE_CODE_OAUTH_TOKEN)
RUN npm install -g @anthropic-ai/claude-code

WORKDIR /app
COPY requirements.txt .
RUN python3 -m venv /opt/venv \
  && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
ENV PATH="/opt/venv/bin:${PATH}"

COPY server.py .

ENV PORT=8787
EXPOSE 8787
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
