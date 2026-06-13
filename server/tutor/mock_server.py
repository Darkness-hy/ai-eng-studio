#!/usr/bin/env python3
"""Mock AI-tutor server — implements the contract in docs/ai-tutor-server-contract.md
without any Claude/LLM. Use it to verify the frontend wiring (CORS, SSE streaming,
context injection) before you stand up the real Claude-Code-backed server.

Run:  python3 server/tutor/mock_server.py 8765
Then: VITE_AI_TUTOR_ENDPOINT=http://localhost:8765/chat npm run dev
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ALLOW_ORIGIN = "*"  # mock only — the real server should allowlist your Pages origin


def sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", ALLOW_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self) -> None:  # CORS preflight
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            req = {}
        message = req.get("message", "")
        zh = req.get("lang") == "zh"
        ctx_title = None
        if req.get("context"):
            first = str(req["context"]).splitlines()[0].lstrip("# ").strip()
            ctx_title = first or None

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self._cors()
        self.end_headers()

        if ctx_title:
            lead = (f"关于《{ctx_title}》——" if zh else f"On “{ctx_title}” — ")
        else:
            lead = ("" if zh else "")
        reply = lead + (
            f"这是模拟回答。你问的是:{message}。真实服务接通后,这里会是 Claude 的流式回答。"
            if zh
            else f"This is a mock reply. You asked: {message}. Real answers stream here once Claude is wired."
        )
        for ch in reply:
            self.wfile.write(sse({"type": "delta", "text": ch}))
            self.wfile.flush()
            time.sleep(0.01)
        self.wfile.write(sse({"type": "done"}))
        self.wfile.flush()

    def log_message(self, *_args) -> None:  # quiet
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"mock tutor server on http://localhost:{port}/chat")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
