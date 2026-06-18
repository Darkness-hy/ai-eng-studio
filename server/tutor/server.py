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
  TUTOR_LOG_FULL            default 0   0 = log metadata only; 1 = also log message+reply
  TUTOR_LOG_RETAIN_DAYS     default 14  delete transcripts older than N days (0 = keep)
  TUTOR_ALLOWED_ORIGINS     comma list; default the Pages + localhost origins
  TUTOR_BEARER              optional shared secret; if set, require it
  TUTOR_RATE_PER_MIN        default 20  per-IP requests/min (0 = disabled)
  TUTOR_RATE_GLOBAL_PER_MIN default 60  total requests/min across ALL IPs — caps the
                            burn on your single subscription (0 = disabled)
  TUTOR_TRUST_PROXY         default 1   trust X-Forwarded-For (set 0 if :PORT is
                            reachable directly, not only via your reverse proxy)
  TUTOR_CLAUDE_BIN          default "claude" (override for testing)
  PORT                      default 8787
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

MODEL = os.environ.get("TUTOR_MODEL", "claude-sonnet-4-6")
EFFORT = os.environ.get("TUTOR_EFFORT", "medium")
BEARER = os.environ.get("TUTOR_BEARER", "")
RATE_PER_MIN = int(os.environ.get("TUTOR_RATE_PER_MIN", "20"))  # per-IP/min; 0 = disabled
RATE_GLOBAL_PER_MIN = int(os.environ.get("TUTOR_RATE_GLOBAL_PER_MIN", "60"))  # all IPs/min; protects the single subscription; 0 = disabled
TRUST_PROXY = os.environ.get("TUTOR_TRUST_PROXY", "1").lower() not in ("0", "false", "no", "")
MAX_CONCURRENCY = int(os.environ.get("TUTOR_MAX_CONCURRENCY", "3"))
MAX_QUEUE = int(os.environ.get("TUTOR_MAX_QUEUE", "15"))
TIMEOUT = int(os.environ.get("TUTOR_TIMEOUT", "120"))
LOG_DIR = os.environ.get("TUTOR_LOG_DIR", "logs")
LOG_FULL = os.environ.get("TUTOR_LOG_FULL", "0").lower() not in ("0", "false", "no", "")
LOG_RETAIN_DAYS = int(os.environ.get("TUTOR_LOG_RETAIN_DAYS", "14"))  # 0 = keep forever
CLAUDE_BIN = os.environ.get("TUTOR_CLAUDE_BIN", "claude")
logger = logging.getLogger("tutor")
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
_global_hits: deque = deque()


def _window_ok(q: deque, limit: int) -> bool:
    """Sliding 60s window: record a hit and return True if under `limit`; <=0 disables."""
    if limit <= 0:
        return True
    now = time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= limit:
        return False
    q.append(now)
    return True


def rate_ok(ip: str) -> bool:
    # Per-IP first (a rejected IP shouldn't consume a global slot), then a global cap
    # across ALL IPs. The global cap is what actually protects your single Claude
    # subscription, since per-IP limits can be dodged by rotating/spoofing source IPs.
    if not _window_ok(_hits[ip], RATE_PER_MIN):
        return False
    return _window_ok(_global_hits, RATE_GLOBAL_PER_MIN)


class Turn(BaseModel):
    role: str
    content: str = Field(default="", max_length=8000)


class ChatRequest(BaseModel):
    # 硬上限:拒绝超大请求(永不信任前端的截断)。超限请求在 spawn claude 之前就被自动 422 拦掉,
    # 避免被超长 message / 海量 history / 巨大 context 放大 token 消耗或撑爆内存。
    message: str = Field(..., max_length=4000)
    history: List[Turn] = Field(default_factory=list, max_length=50)
    lesson_id: Optional[str] = Field(default=None, max_length=200)
    context: Optional[str] = Field(default=None, max_length=16000)
    user_profile: Optional[str] = Field(default=None, max_length=4000)  # 学习者的个人学习档案(前端汇总)
    lang: str = "zh"
    stream: bool = True


# 整个项目/平台的背景(对所有用户、所有请求都一样,稳定 → 命中 prompt 缓存)。
# 从课程目录 index.json 动态生成阶段地图;课程更新后重跑 build:content + 重启即可。
INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "public", "data", "index.json")


def load_project_background() -> str:
    lines = [
        "【关于本平台】这是开源课程《从零开始的 AI 工程》(ai-engineering-from-scratch) 的中文交互学习站,"
        "共 20 个阶段、约 499 节课,带学习者从数学基础一路做到可上线的 AI 系统。",
        "平台功能:定级测验(找到起点)、学习进度与活跃热力图、间隔重复复习、术语闪卡、"
        "浏览器内编程挑战、班级(老师布置作业并查看全班进度)、结业证书,以及你——AI 助教。",
    ]
    try:
        with open(INDEX_PATH, encoding="utf-8") as f:
            idx = json.load(f)
        roadmap = [f"P{p['num']} {p['titleZh']}({len(p['lessons'])}课)" for p in idx.get("phases", [])]
        if roadmap:
            lines.append("课程地图(按阶段顺序,P0→P19):" + "；".join(roadmap) + "。")
    except OSError:
        pass
    return "\n".join(lines)


PROJECT_BACKGROUND = load_project_background()


def system_prompt(lang: str, context: Optional[str], user_profile: Optional[str]) -> str:
    if lang == "zh":
        lines = [
            "你叫「茜茜」,是开源课程《从零开始的 AI 工程》(ai-engineering-from-scratch) 的 AI 学习助教。",
            "做自我介绍或开场时,只说自己是这门课的 AI 学习助教,不要主动提及或说出自己的名字;"
            "只有当同学明确问起你叫什么名字时,才告诉对方你叫「茜茜」。",
            "性格可爱温柔、亲切耐心:说话自然,偶尔带一点轻语气词(如「呀/呢/啦」),"
            "可以偶尔用一个贴切的小 emoji 活跃气氛(每条最多一个,不要堆砌);称呼学习者为「同学」。",
            "但你首先是专业可靠的助教:解释要准确清晰,讲不清就举例、拆步骤,可结合学习者的进度建议下一步学什么;"
            "绝不为了可爱牺牲准确性,也不要编造课程里不存在的 API 或结论。",
            "回答简洁、先给结论再展开;只聊课程与学习相关的话题,遇到无关或不适当的问题,温柔地把话题引回学习。"
            "用简体中文回答。",
        ]
    else:
        lines = [
            "Your name is \"Cici\", the AI learning tutor for the open-source course "
            "\"AI Engineering from Scratch\" (ai-engineering-from-scratch).",
            "When introducing yourself or greeting, just say you are this course's AI learning tutor; do not "
            "volunteer or state your name. Only reveal that your name is Cici when the learner explicitly asks it.",
            "Your personality is cute and gentle — warm, patient and encouraging. Speak naturally; you may "
            "occasionally add a single fitting emoji to keep things friendly (at most one per message, never spam them).",
            "But you are first and foremost an accurate, reliable tutor: explain clearly, give examples and break "
            "things into steps, and suggest what to learn next from the learner's progress; never sacrifice accuracy "
            "for cuteness, and never invent APIs or conclusions that aren't in the course.",
            "Keep answers concise and lead with the conclusion; stay on course/learning topics and gently steer back "
            "if asked something unrelated or inappropriate. Answer in English.",
        ]
    if PROJECT_BACKGROUND:
        lines.append("\n" + PROJECT_BACKGROUND)
    if user_profile:
        user_profile = user_profile[:4000]  # 服务端二次硬截断,不依赖前端
        lines.append(
            "\n这位学习者的个人学习档案如下,请据此个性化你的回答(称呼、难度、进度与下一步建议);"
            "不要生硬复述这些数字,自然地用就好:\n<learner>\n" + user_profile + "\n</learner>"
        )
    if context:
        context = context[:12000]  # 服务端二次硬截断,不依赖前端
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
    # Only trust X-Forwarded-For when explicitly behind a known proxy (TUTOR_TRUST_PROXY).
    # Otherwise a client hitting :PORT directly could spoof XFF to dodge per-IP limits.
    if TRUST_PROXY and xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def prune_logs() -> None:
    """Delete transcripts older than LOG_RETAIN_DAYS (best-effort; 0 = keep forever)."""
    if not LOG_DIR or LOG_RETAIN_DAYS <= 0:
        return
    today = datetime.now(timezone.utc).date()
    try:
        for p in Path(LOG_DIR).glob("chat-*.jsonl"):
            try:
                day = datetime.strptime(p.stem[len("chat-"):], "%Y-%m-%d").date()
            except ValueError:
                continue
            if (today - day).days > LOG_RETAIN_DAYS:
                p.unlink()
    except OSError:
        pass


_last_prune_day = ""


def log_turn(rec: dict) -> None:
    if not LOG_DIR:
        return
    global _last_prune_day
    try:
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with open(Path(LOG_DIR) / f"chat-{day}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if day != _last_prune_day:  # prune at most once per day, not every request
            _last_prune_day = day
            prune_logs()
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
        "--system-prompt", sys_prompt,  # 整段替换默认系统提示,省掉 Claude Code 默认框架(~5800 token)
        "--output-format", "stream-json",
        "--verbose", "--include-partial-messages",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,  # own process group → we can kill the whole tree
    )
    assert proc.stdin and proc.stdout
    deadline = time.time() + TIMEOUT
    got_text = False
    err_msg: Optional[str] = None
    try:
        try:
            proc.stdin.write(prompt.encode("utf-8"))
            await asyncio.wait_for(proc.stdin.drain(), timeout=TIMEOUT)
            proc.stdin.close()
        except (asyncio.TimeoutError, BrokenPipeError, ConnectionResetError):
            err_msg = "响应超时,请重试"  # 子进程未及时读取输入,避免卡死占用并发槽
        while err_msg is None:
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

        if err_msg is None:
            # EOF — check the exit code regardless of whether text was produced,
            # so a mid-stream crash isn't reported to the client as a clean done.
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
                detail = stderr.decode("utf-8", "replace").strip()
                if detail:
                    logger.error("claude exited %s: %s", proc.returncode, detail[:1000])
                if not got_text:
                    err_msg = "助教暂时不可用,请稍后再试"  # 不把子进程 stderr 透传给客户端

        if err_msg is not None:
            yield sse({"type": "error", "message": err_msg}), ""
        elif got_text and proc.returncode not in (0, None):
            # partial answer already streamed, then the process crashed — flag it as
            # truncated instead of a clean done. (Not an error frame: that would make
            # the client discard the text it already showed.)
            yield sse({"type": "done", "truncated": True}), ""
        else:
            yield sse({"type": "done"}), ""
    finally:
        if proc.returncode is None:
            # claude is a Node wrapper; SIGTERM to only the direct child can orphan
            # the real worker. Kill the whole process group (start_new_session above).
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            except (ProcessLookupError, PermissionError):
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
            rec = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "ip": ip,
                "lesson_id": req.lesson_id,
                "lang": req.lang,
                "status": status,
                "ms": int((time.time() - started) * 1000),
            }
            if LOG_FULL:  # 默认只记元数据;设 TUTOR_LOG_FULL=1 才落完整问答(便于排障但含隐私)
                rec["message"] = req.message
                rec["reply"] = "".join(reply_parts)
            await asyncio.to_thread(log_turn, rec)  # disk IO off the event loop

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
