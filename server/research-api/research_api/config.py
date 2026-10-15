from __future__ import annotations

import os
from dataclasses import dataclass
from uuid import UUID


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    environment: str
    store_mode: str
    database_url: str | None
    supabase_url: str | None
    supabase_jwt_secret: str | None
    jwt_audience: str
    jwt_issuer: str | None
    dev_auth: bool
    dev_user_id: UUID
    allowed_origins: tuple[str, ...]
    runner_mode: str
    runner_image: str
    runtime_image_digest: str
    runner_timeout_seconds: int
    dispatcher_poll_seconds: float
    event_poll_seconds: float

    @classmethod
    def from_env(cls) -> Settings:
        environment = os.environ.get("RESEARCH_ENV", "development").strip().lower()
        store_mode = os.environ.get("RESEARCH_STORE", "memory").strip().lower()
        supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/") or None
        issuer = os.environ.get("SUPABASE_JWT_ISSUER", "").strip() or None
        if issuer is None and supabase_url:
            issuer = f"{supabase_url}/auth/v1"
        origins = tuple(
            origin.strip()
            for origin in os.environ.get(
                "RESEARCH_ALLOWED_ORIGINS",
                "http://localhost:5180,http://127.0.0.1:5180,https://darkness-hy.github.io",
            ).split(",")
            if origin.strip()
        )
        settings = cls(
            environment=environment,
            store_mode=store_mode,
            database_url=os.environ.get("SUPABASE_DB_URL") or os.environ.get("DATABASE_URL"),
            supabase_url=supabase_url,
            supabase_jwt_secret=os.environ.get("SUPABASE_JWT_SECRET"),
            jwt_audience=os.environ.get("SUPABASE_JWT_AUDIENCE", "authenticated"),
            jwt_issuer=issuer,
            dev_auth=_bool_env("RESEARCH_DEV_AUTH"),
            dev_user_id=UUID(os.environ.get("RESEARCH_DEV_USER_ID", "00000000-0000-4000-8000-000000000001")),
            allowed_origins=origins,
            runner_mode=os.environ.get("RESEARCH_RUNNER_MODE", "local").strip().lower(),
            runner_image=os.environ.get("RESEARCH_RUNNER_IMAGE", "ai-eng-studio-research-runner:0.4.0"),
            runtime_image_digest=os.environ.get("RESEARCH_RUNTIME_IMAGE_DIGEST", "local-dev-unpinned"),
            runner_timeout_seconds=int(os.environ.get("RESEARCH_RUNNER_TIMEOUT", "120")),
            dispatcher_poll_seconds=float(os.environ.get("RESEARCH_DISPATCHER_POLL_SECONDS", "1")),
            event_poll_seconds=float(os.environ.get("RESEARCH_EVENT_POLL_SECONDS", "0.25")),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.store_mode not in {"memory", "postgres"}:
            raise ValueError("RESEARCH_STORE must be memory or postgres")
        if self.runner_mode not in {"local", "docker"}:
            raise ValueError("RESEARCH_RUNNER_MODE must be local or docker")
        if self.store_mode == "postgres" and not self.database_url:
            raise ValueError("SUPABASE_DB_URL is required for RESEARCH_STORE=postgres")
        if self.environment == "production":
            if self.dev_auth:
                raise ValueError("RESEARCH_DEV_AUTH must be disabled in production")
            if self.store_mode != "postgres":
                raise ValueError("production requires RESEARCH_STORE=postgres")
            if self.runner_mode != "docker":
                raise ValueError("production requires RESEARCH_RUNNER_MODE=docker")
            if not self.runtime_image_digest.startswith("sha256:"):
                raise ValueError("production requires a sha256 RESEARCH_RUNTIME_IMAGE_DIGEST")
            if not self.jwt_issuer or not self.supabase_url:
                raise ValueError("SUPABASE_URL and JWT issuer are required in production")
