from __future__ import annotations

import hashlib
from typing import Any

import rfc8785


def request_fingerprint(method: str, path: str, body: dict[str, Any]) -> str:
    normalized_path = "/" + "/".join(part for part in path.split("/") if part)
    canonical = rfc8785.dumps(body)
    payload = method.upper().encode("ascii") + b"\n" + normalized_path.encode("utf-8") + b"\n" + canonical
    return hashlib.sha256(payload).hexdigest()
