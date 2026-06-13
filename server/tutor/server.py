#!/usr/bin/env python3
"""Reference AI-tutor server for ai-eng-studio.

Implements docs/ai-tutor-server-contract.md. Runs headless Claude Code
(`claude -p ... --output-format stream-json`) authenticated with your Claude
SUBSCRIPTION via CLAUDE_CODE_OAUTH_TOKEN, and streams the answer back as SSE.

Env:
  CLAUDE_CODE_OAUTH_TOKEN   required — from `claude setup-token` (subscription)
  TUTOR_MODEL               default claude-sonnet-4-6
  TUTOR_EFFORT              default medium
  TUTOR_ALLOWED_ORIGINS     comma list; default the Pages + localhost origins
  TUTOR_BEARER              optional shared secret; if set, require it
  TUTOR_RATE_PER_MIN        default 20 (per client IP)
  PORT                      default 8787
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

MODEL = os.environ.get("TUTOR_MODEL", "claude-sonnet-4-6")
EFFORT = os.environ.get("TUTOR_EFFORT", "medium")
BEARER = os.environ.get("TUTOR_BEARER", "")
RATE_PER_MIN = int(os.environ.get("TUTOR_RATE_PER_MIN", "20"))
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

_hits: dict[str, deque[float]] = defaultdict(deque)


def rate_ok(ip: str) -> bool:
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
    history: list[Turn] = []
    lesson_id: str | None = None
    context: str | None = None
    lang: str = "zh"
    stream: bool = True


def system_prompt(lang: str, context: str | None) -> str:
    speak = "用简体中文回答" if lang == "zh" else "Answer in English"
    lines = [
        "你是开源课程《从零开始的 AI 工程》(ai-engineering-from-scratch) 的助教。",
        "基于课程内容回答学习者的问题:讲不清就举例、拆步骤;",
        "不要编造课程里不存在的 API 或结论。" + speak + "。",
    ]
    if context:
        lines.append("\n以下是学习者当前所在课程的内容,作为回答依据:\n<course>\n" + context + "\n</course>")
    return "\n".join(lines)


def user_prompt(message: str, history: list[Turn]) -> str:
    parts: list[str] = []
    for t in history[-8:]:  # cap history to keep prompts bounded
        who = "学习者" if t.role == "user" else "助教"
        parts.append(f"{who}: {t.content}")
    parts.append(f"学习者: {message}")
    return "\n".join(parts)


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def stream_claude(sys_prompt: str, prompt: str):
    """Spawn headless Claude Code and yield SSE frames of text deltas."""
    cmd = [
        "claude", "-p",
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

    try:
        async for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            # stream-json: text deltas live in stream_event/content_block_delta
            if msg.get("type") == "stream_event":
                ev = msg.get("event", {})
                if ev.get("type") == "content_block_delta":
                    delta = ev.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            yield sse({"type": "delta", "text": text})
        yield sse({"type": "done"})
    finally:
        if proc.returncode is None:
            proc.terminate()
        err = (await proc.stderr.read()).decode("utf-8", "replace") if proc.stderr else ""
        if proc.returncode not in (0, None) and err:
            yield sse({"type": "error", "message": err[:300]})


@app.post("/chat")
async def chat(req: ChatRequest, request: Request, authorization: str | None = Header(default=None)):
    if BEARER:
        if authorization != f"Bearer {BEARER}":
            raise HTTPException(status_code=401, detail="unauthorized")
    ip = request.client.host if request.client else "unknown"
    if not rate_ok(ip):
        raise HTTPException(status_code=429, detail="rate limited")

    sys_prompt = system_prompt(req.lang, req.context)
    prompt = user_prompt(req.message, req.history)
    return StreamingResponse(
        stream_claude(sys_prompt, prompt),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/health")
async def health():
    return {"ok": True, "model": MODEL, "effort": EFFORT}
