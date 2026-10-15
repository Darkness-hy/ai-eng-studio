from __future__ import annotations

from typing import Any


class DomainError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


def not_found(resource: str = "run") -> DomainError:
    code = "artifact_not_found" if resource == "artifact" else "run_not_found"
    return DomainError(code, "Resource not found", 404)


def version_conflict(expected: int, actual: int) -> DomainError:
    return DomainError(
        "version_conflict",
        "The run changed; refresh before retrying this action",
        409,
        {"expectedVersion": expected, "actualVersion": actual},
    )


def invalid_state(message: str) -> DomainError:
    return DomainError("invalid_state", message, 409)


def idempotency_conflict() -> DomainError:
    return DomainError(
        "idempotency_conflict",
        "The idempotency key was already used for a different request",
        409,
    )
