# AI-tutor server: Python (FastAPI) + DeepSeek/OpenAI-compatible streaming API.
FROM python:3.12-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python -m venv /opt/venv \
  && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
ENV PATH="/opt/venv/bin:${PATH}"

COPY server.py .

# Run as a non-root user; give it a writable HOME and logs dir.
# (If you bind-mount ./logs, `chown 10001 ./logs` on the host or logs won't be writable.)
RUN useradd --create-home --uid 10001 tutor \
  && mkdir -p /app/logs \
  && chown -R tutor:tutor /app
USER tutor
ENV HOME=/home/tutor

ENV PORT=8787
EXPOSE 8787
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
