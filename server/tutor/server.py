#!/usr/bin/env python3
"""Reference AI-tutor server for ai-eng-studio (multi-user hardened).

Implements docs/ai-tutor-server-contract.md. Runs headless Claude Code
(`claude -p ... --output-format stream-json`) authenticated with your Claude
SUBSCRIPTION via CLAUDE_CODE_OAUTH_TOKEN, and streams the answer back as SSE.

Multi-user design:
  * Isolation is structural — the server holds NO per-user conversation state.
    Each request carries its own `history`; each /chat spawns its own `claude`
    subprocess with its own prompt/stream. Two users can never see each other's
    content, even concurrently on one subscription.
  * Concurrency cap (TUTOR_MAX_CONCURRENCY) limits simultaneous `claude` calls so
    a class burst can't fork N processes or hammer your one subscription. Extra
    requests queue up to TUTOR_MAX_QUEUE; beyond that they get 429 (retry later).
  * Per-request timeout (TUTOR_TIMEOUT) frees a slot if a call hangs.
  * Transcript log (TUTOR_LOG_DIR) appends one JSONL line per finished turn for
    troubleshooting.

Env:
  CLAUDE_CODE_OAUTH_TOKEN   required — from `claude setup-token` (subscription)
  TUTOR_MODEL               default claude-sonnet-4-6
  TUTOR_EFFORT              default medium
  TUTOR_MAX_CONCURRENCY     default 3   (simultaneous claude calls)
  TUTOR_MAX_QUEUE           default 15  (extra waiters before 429)
  TUTOR_TIMEOUT             default 120 (seconds per request)
  TUTOR_LOG_DIR             default logs  (JSONL transcripts; "" to disable)
  TUTOR_ALLOWED_ORIGINS     comma list; default the Pages + localhost origins
  TUTOR_BEARER              optional shared secret; if set, require it
  TUTOR_RATE_PER_MIN        default 0 (disabled); >0 enables per-IP rate limit
  TUTOR_CLAUDE_BIN          default "claude" (override for testing)
  PORT                      default 8787
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

MODEL = os.environ.get("TUTOR_MODEL", "claude-sonnet-4-6")
EFFORT = os.environ.get("TUTOR_EFFORT", "medium")
BEARER = os.environ.get("TUTOR_BEARER", "")
RATE_PER_MIN = int(os.environ.get("TUTOR_RATE_PER_MIN", "0"))  # 0 = disabled
MAX_CONCURRENCY = int(os.environ.get("TUTOR_MAX_CONCURRENCY", "3"))
MAX_QUEUE = int(os.environ.get("TUTOR_MAX_QUEUE", "15"))
TIMEOUT = int(os.environ.get("TUTOR_TIMEOUT", "120"))
LOG_DIR = os.environ.get("TUTOR_LOG_DIR", "logs")
CLAUDE_BIN = os.environ.get("TUTOR_CLAUDE_BIN", "claude")
ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "TUTOR_ALLOWED_ORIGINS",
        "https://darkness-hy.github.io,http://localhost:5180",
    ).split(",")
    if o.strip()
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Concurrency control. asyncio is single-threaded, so the in-flight counter
# needs no lock as long as we never await between reading and updating it.
_sem = asyncio.Semaphore(MAX_CONCURRENCY)
_inflight = 0
_hits: Dict[str, deque] = defaultdict(deque)


def rate_ok(ip: str) -> bool:
    if RATE_PER_MIN <= 0:
        return True
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_PER_MIN:
        return False
    q.append(now)
    return True


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Turn] = []
    lesson_id: Optional[str] = None
    context: Optional[str] = None
    user_profile: Optional[str] = None  # 学习者的个人学习档案(前端汇总)
    lang: str = "zh"
    stream: bool = True


def system_prompt(lang: str, context: Optional[str], user_profile: Optional[str]) -> str:
    speak = "用简体中文回答" if lang == "zh" else "Answer in English"
    lines = [
        "你是开源课程《从零开始的 AI 工程》(ai-engineering-from-scratch) 的助教。",
        "基于课程内容回答学习者的问题:讲不清就举例、拆步骤;",
        "不要编造课程里不存在的 API 或结论。" + speak + "。",
    ]
    if user_profile:
        lines.append(
            "\n这位学习者的个人学习档案如下,请据此个性化你的回答(称呼、难度、进度与下一步建议);"
            "不要生硬复述这些数字,自然地用就好:\n<learner>\n" + user_profile + "\n</learner>"
        )
    if context:
        lines.append("\n以下是学习者当前所在课程的内容,作为回答依据:\n<course>\n" + context + "\n</course>")
    return "\n".join(lines)


def user_prompt(message: str, history: List[Turn]) -> str:
    parts: list[str] = []
    for t in history[-8:]:  # cap history to keep prompts bounded
        who = "学习者" if t.role == "user" else "助教"
        parts.append(f"{who}: {t.content}")
    parts.append(f"学习者: {message}")
    return "\n".join(parts)


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def client_ip(request: Request, xff: Optional[str]) -> str:
    if xff:  # behind a trusted reverse proxy / tunnel
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def log_turn(rec: dict) -> None:
    if not LOG_DIR:
        return
    try:
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with open(Path(LOG_DIR) / f"chat-{day}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


async def stream_claude(sys_prompt: str, prompt: str):
    """Spawn headless Claude Code; yield (sse_frame, text_chunk) tuples.

    Times out after TIMEOUT seconds (3.10-compatible, deadline-based) and always
    cleans up the subprocess so its concurrency slot is released.
    """
    cmd = [
        CLAUDE_BIN, "-p",
        "--model", MODEL,
        "--effort", EFFORT,
        "--tools", "",  # no tools — plain Q&A
        "--append-system-prompt", sys_prompt,
        "--output-format", "stream-json",
        "--verbose", "--include-partial-messages",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdin and proc.stdout
    proc.stdin.write(prompt.encode("utf-8"))
    await proc.stdin.drain()
    proc.stdin.close()

    deadline = time.time() + TIMEOUT
    got_text = False
    err_msg: Optional[str] = None
    try:
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                err_msg = "响应超时,请重试"
                break
            try:
                raw = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
            except asyncio.TimeoutError:
                err_msg = "响应超时,请重试"
                break
            if not raw:  # EOF
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "stream_event":
                ev = msg.get("event", {})
                if ev.get("type") == "content_block_delta":
                    delta = ev.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            got_text = True
                            yield sse({"type": "delta", "text": text}), text

        if err_msg is None and not got_text:
            # process ended without text — surface stderr if it failed
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
            if proc.returncode not in (0, None):
                stderr = b""
                if proc.stderr:
                    try:
                        stderr = await asyncio.wait_for(proc.stderr.read(), timeout=5)
                    except asyncio.TimeoutError:
                        stderr = b""
                err_msg = stderr.decode("utf-8", "replace")[:300] or "服务出错"

        yield (sse({"type": "error", "message": err_msg}) if err_msg else sse({"type": "done"})), ""
    finally:
        if proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass


@app.post("/chat")
async def chat(
    req: ChatRequest,
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_forwarded_for: Optional[str] = Header(default=None),
):
    global _inflight
    if BEARER and authorization != f"Bearer {BEARER}":
        raise HTTPException(status_code=401, detail="unauthorized")
    ip = client_ip(request, x_forwarded_for)
    if not rate_ok(ip):
        raise HTTPException(status_code=429, detail="rate limited")
    # bounded queue: cap total in-flight (running + waiting) so bursts fail fast
    if _inflight >= MAX_CONCURRENCY + MAX_QUEUE:
        raise HTTPException(status_code=429, detail="服务繁忙,请稍后再试")
    _inflight += 1

    sys_prompt = system_prompt(req.lang, req.context, req.user_profile)
    prompt = user_prompt(req.message, req.history)

    async def gen():
        global _inflight
        started = time.time()
        reply_parts: list[str] = []
        status = "ok"
        try:
            async with _sem:  # waits here when MAX_CONCURRENCY are already running
                async for frame, text in stream_claude(sys_prompt, prompt):
                    if text:
                        reply_parts.append(text)
                    elif '"type": "error"' in frame:
                        status = "error"
                    yield frame
        finally:
            _inflight -= 1
            log_turn({
                "ts": datetime.now(timezone.utc).isoformat(),
                "ip": ip,
                "lesson_id": req.lesson_id,
                "lang": req.lang,
                "message": req.message,
                "reply": "".join(reply_parts),
                "status": status,
                "ms": int((time.time() - started) * 1000),
            })

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/health")
async def health():
    return {
        "ok": True,
        "model": MODEL,
        "effort": EFFORT,
        "max_concurrency": MAX_CONCURRENCY,
        "inflight": _inflight,
    }
