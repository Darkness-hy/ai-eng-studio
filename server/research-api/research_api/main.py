from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .auth import Authenticator, current_user
from .config import Settings
from .engine import FixedResearchEngine, ResearchEngine
from .errors import DomainError
from .fingerprint import request_fingerprint
from .models import (
    ArtifactCommandResult,
    ArtifactKind,
    ArtifactList,
    CommandAccepted,
    CreateRunRequest,
    ErrorBody,
    ErrorResponse,
    ExperimentJob,
    ExperimentSuiteAccepted,
    ExperimentSuiteRequest,
    HealthResponse,
    HypothesisInput,
    ListRunsResponse,
    ResearchEvent,
    RunSnapshot,
    RunStatus,
    TemplateList,
    VersionedCommand,
)
from .service import ResearchService
from .store import MemoryResearchStore, ResearchStore
from .template import template_definitions

logger = logging.getLogger("research.api")

IdempotencyKey = Annotated[
    str,
    Header(
        alias="Idempotency-Key",
        min_length=8,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    ),
]
UserId = Annotated[UUID, Depends(current_user)]


def encode_sse(event: ResearchEvent) -> str:
    data = json.dumps(
        event.model_dump(mode="json", by_alias=True),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"id: {event.id}\nevent: {event.event_type}\ndata: {data}\n\n"


def _request_id(request: Request) -> UUID:
    return getattr(request.state, "request_id", uuid4())


def _error_response(
    request: Request,
    code: str,
    message: str,
    status_code: int,
    details: dict | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(
            code=code,
            message=message,
            request_id=_request_id(request),
            details=details,
        )
    )
    return JSONResponse(
        body.model_dump(mode="json", by_alias=True, exclude_none=True),
        status_code=status_code,
        headers=headers,
    )


def create_app(
    settings: Settings | None = None,
    store: ResearchStore | None = None,
    engine: ResearchEngine | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        resolved_store = store
        if resolved_store is None:
            if resolved_settings.store_mode == "memory":
                resolved_store = MemoryResearchStore()
            else:
                from .postgres_store import PostgresResearchStore

                resolved_store = await PostgresResearchStore.connect(resolved_settings.database_url or "")
        resolved_engine = engine or FixedResearchEngine(
            mode=resolved_settings.runner_mode,
            image=resolved_settings.runner_image,
            runtime_image_digest=resolved_settings.runtime_image_digest,
            timeout_seconds=resolved_settings.runner_timeout_seconds,
        )
        application.state.settings = resolved_settings
        application.state.authenticator = Authenticator(resolved_settings)
        application.state.service = ResearchService(
            resolved_store,
            resolved_engine,
            dispatcher_poll_seconds=resolved_settings.dispatcher_poll_seconds,
            stale_job_seconds=resolved_settings.runner_timeout_seconds + 30,
        )
        application.state.service.start()
        yield
        await application.state.service.close()

    application = FastAPI(
        title="ai-eng-studio ResearchRun API",
        version="0.4.0",
        description=(
            "Bounded ResearchRun workflow for 17 fixed offline capstone experiments. "
            "The runner accepts no user code or arbitrary configuration and makes no paid API calls."
        ),
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "Last-Event-ID"],
    )

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        try:
            request.state.request_id = UUID(supplied) if supplied else uuid4()
        except ValueError:
            request.state.request_id = uuid4()
        response = await call_next(request)
        response.headers["X-Request-ID"] = str(request.state.request_id)
        return response

    @application.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        headers = {"Retry-After": "5"} if exc.status_code == 429 else None
        return _error_response(request, exc.code, exc.message, exc.status_code, exc.details, headers=headers)

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [
            {"location": list(item["loc"]), "message": item["msg"], "type": item["type"]}
            for item in exc.errors()
        ]
        return _error_response(
            request,
            "validation_failed",
            "Request validation failed",
            422,
            {"fields": fields},
        )

    @application.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict):
            code = str(exc.detail.get("code", "authentication_required"))
            message = str(exc.detail.get("message", "Request failed"))
        else:
            code = "authentication_required" if exc.status_code == 401 else "forbidden"
            message = str(exc.detail)
        return _error_response(request, code, message, exc.status_code, headers=exc.headers)

    @application.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled request error: request_id=%s", _request_id(request), exc_info=exc)
        return _error_response(request, "internal_error", "Internal server error", 500)

    def service(request: Request) -> ResearchService:
        return request.app.state.service

    @application.get("/health", response_model=HealthResponse, tags=["System"])
    async def health() -> HealthResponse:
        return HealthResponse()

    @application.get("/v1/research-templates", response_model=TemplateList, tags=["Templates"])
    async def list_templates(_user_id: UserId) -> TemplateList:
        return TemplateList(items=template_definitions())

    @application.get("/v1/research-runs", response_model=ListRunsResponse, tags=["Runs"])
    async def list_runs(
        request: Request,
        user_id: UserId,
        status: RunStatus | None = None,
        cursor: str | None = Query(default=None, min_length=1, max_length=512),
        limit: int = Query(default=20, ge=1, le=50),
    ) -> ListRunsResponse:
        del cursor
        return await service(request).store.list_runs(user_id, status=status, limit=limit)

    @application.post(
        "/v1/research-runs",
        response_model=RunSnapshot,
        status_code=201,
        tags=["Runs"],
    )
    async def create_run(
        request: Request,
        body: CreateRunRequest,
        user_id: UserId,
        idempotency_key: IdempotencyKey,
    ) -> RunSnapshot:
        fingerprint = request_fingerprint(
            request.method, request.url.path, body.model_dump(mode="json", by_alias=True)
        )
        return await service(request).store.create_run(user_id, idempotency_key, fingerprint, body)

    @application.get("/v1/research-runs/{run_id}", response_model=RunSnapshot, tags=["Runs"])
    async def get_run(request: Request, run_id: UUID, user_id: UserId) -> RunSnapshot:
        return await service(request).store.get_snapshot(user_id, run_id)

    @application.put("/v1/research-runs/{run_id}/hypothesis", response_model=RunSnapshot, tags=["Runs"])
    async def put_hypothesis(
        request: Request, run_id: UUID, body: HypothesisInput, user_id: UserId
    ) -> RunSnapshot:
        return await service(request).store.put_hypothesis(user_id, run_id, body)

    @application.get(
        "/v1/research-runs/{run_id}/experiment-jobs",
        response_model=dict[str, list[ExperimentJob]],
        tags=["Runs"],
    )
    async def list_jobs(request: Request, run_id: UUID, user_id: UserId) -> dict[str, list[ExperimentJob]]:
        snapshot = await service(request).store.get_snapshot(user_id, run_id)
        return {"items": snapshot.jobs}

    @application.post(
        "/v1/research-runs/{run_id}/experiment-jobs",
        response_model=ExperimentSuiteAccepted,
        status_code=202,
        tags=["Runs"],
    )
    async def enqueue_experiments(
        request: Request,
        run_id: UUID,
        body: ExperimentSuiteRequest,
        user_id: UserId,
        idempotency_key: IdempotencyKey,
    ) -> ExperimentSuiteAccepted:
        fingerprint = request_fingerprint(
            request.method, request.url.path, body.model_dump(mode="json", by_alias=True)
        )
        result = await service(request).store.enqueue_suite(
            user_id, run_id, idempotency_key, fingerprint, body.expected_version
        )
        service(request).schedule_jobs(result.jobs)
        return result

    async def begin_command(
        request: Request,
        run_id: UUID,
        body: VersionedCommand,
        user_id: UUID,
        idempotency_key: str,
        command: str,
    ) -> CommandAccepted:
        fingerprint = request_fingerprint(
            request.method, request.url.path, body.model_dump(mode="json", by_alias=True)
        )
        store_instance = service(request).store
        if command == "analysis":
            result = await store_instance.begin_analysis(
                user_id, run_id, idempotency_key, fingerprint, body.expected_version
            )
            if result.run.status == RunStatus.ANALYZING:
                service(request).schedule_analysis(run_id)
            return result
        if command == "verification":
            result = await store_instance.begin_verification(
                user_id, run_id, idempotency_key, fingerprint, body.expected_version
            )
            if result.run.status == RunStatus.VERIFYING:
                service(request).schedule_verification(run_id)
            return result
        if command == "retry":
            result, jobs = await store_instance.retry_run(
                user_id, run_id, idempotency_key, fingerprint, body.expected_version
            )
            service(request).schedule_jobs(jobs)
            return result
        result = await store_instance.cancel_run(
            user_id, run_id, idempotency_key, fingerprint, body.expected_version
        )
        await service(request).cancel_background(run_id)
        return result

    @application.post(
        "/v1/research-runs/{run_id}/analysis",
        response_model=CommandAccepted,
        status_code=202,
        tags=["Runs"],
    )
    async def analyze_run(
        request: Request,
        run_id: UUID,
        body: VersionedCommand,
        user_id: UserId,
        idempotency_key: IdempotencyKey,
    ) -> CommandAccepted:
        return await begin_command(request, run_id, body, user_id, idempotency_key, "analysis")

    @application.post(
        "/v1/research-runs/{run_id}/verify",
        response_model=CommandAccepted,
        status_code=202,
        tags=["Runs"],
    )
    async def verify_run(
        request: Request,
        run_id: UUID,
        body: VersionedCommand,
        user_id: UserId,
        idempotency_key: IdempotencyKey,
    ) -> CommandAccepted:
        return await begin_command(request, run_id, body, user_id, idempotency_key, "verification")

    @application.post(
        "/v1/research-runs/{run_id}/retry",
        response_model=CommandAccepted,
        status_code=202,
        tags=["Runs"],
    )
    async def retry_run(
        request: Request,
        run_id: UUID,
        body: VersionedCommand,
        user_id: UserId,
        idempotency_key: IdempotencyKey,
    ) -> CommandAccepted:
        return await begin_command(request, run_id, body, user_id, idempotency_key, "retry")

    @application.post(
        "/v1/research-runs/{run_id}/cancel",
        response_model=CommandAccepted,
        status_code=202,
        tags=["Runs"],
    )
    async def cancel_run(
        request: Request,
        run_id: UUID,
        body: VersionedCommand,
        user_id: UserId,
        idempotency_key: IdempotencyKey,
    ) -> CommandAccepted:
        return await begin_command(request, run_id, body, user_id, idempotency_key, "cancel")

    @application.get("/v1/research-runs/{run_id}/events", tags=["Runs"])
    async def stream_events(
        request: Request,
        run_id: UUID,
        user_id: UserId,
        after: str | None = Query(default=None, pattern=r"^[0-9]+$", max_length=19),
        last_event_id: Annotated[
            str | None,
            Header(alias="Last-Event-ID", pattern=r"^[0-9]+$", max_length=19),
        ] = None,
    ) -> StreamingResponse:
        cursor = int(after or last_event_id or "0")
        await service(request).store.list_events(user_id, run_id, cursor)

        async def event_stream() -> AsyncIterator[str]:
            nonlocal cursor
            last_heartbeat = time.monotonic()
            yield ": connected\n\n"
            while not await request.is_disconnected():
                events = await service(request).store.list_events(user_id, run_id, cursor)
                for event in events:
                    cursor = int(event.id)
                    yield encode_sse(event)
                    last_heartbeat = time.monotonic()
                if time.monotonic() - last_heartbeat >= 15:
                    yield ": heartbeat\n\n"
                    last_heartbeat = time.monotonic()
                await asyncio.sleep(resolved_settings.event_poll_seconds)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @application.get("/v1/research-runs/{run_id}/artifacts", response_model=ArtifactList, tags=["Artifacts"])
    async def list_artifacts(
        request: Request,
        run_id: UUID,
        user_id: UserId,
        kind: ArtifactKind | None = None,
    ) -> ArtifactList:
        return await service(request).store.list_artifacts(user_id, run_id, kind)

    async def publication_command(
        request: Request,
        artifact_id: UUID,
        body: VersionedCommand,
        user_id: UUID,
        idempotency_key: str,
        publish: bool,
    ) -> ArtifactCommandResult:
        fingerprint = request_fingerprint(
            request.method, request.url.path, body.model_dump(mode="json", by_alias=True)
        )
        operation = (
            service(request).store.publish_artifact if publish else service(request).store.unpublish_artifact
        )
        return await operation(user_id, artifact_id, idempotency_key, fingerprint, body.expected_version)

    @application.post(
        "/v1/research-artifacts/{artifact_id}/publish",
        response_model=ArtifactCommandResult,
        tags=["Artifacts"],
    )
    async def publish_artifact(
        request: Request,
        artifact_id: UUID,
        body: VersionedCommand,
        user_id: UserId,
        idempotency_key: IdempotencyKey,
    ) -> ArtifactCommandResult:
        return await publication_command(request, artifact_id, body, user_id, idempotency_key, publish=True)

    @application.delete(
        "/v1/research-artifacts/{artifact_id}/publication",
        response_model=ArtifactCommandResult,
        tags=["Artifacts"],
    )
    async def unpublish_artifact(
        request: Request,
        artifact_id: UUID,
        body: VersionedCommand,
        user_id: UserId,
        idempotency_key: IdempotencyKey,
    ) -> ArtifactCommandResult:
        return await publication_command(request, artifact_id, body, user_id, idempotency_key, publish=False)

    return application


app = create_app()
