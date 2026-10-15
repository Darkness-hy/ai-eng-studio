from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from research_api.errors import DomainError
from research_api.fingerprint import request_fingerprint
from research_api.models import (
    CreateRunRequest,
    ExpectedDirection,
    HypothesisInput,
    JobStatus,
    RunFailure,
)
from research_api.service import ResearchService
from research_api.store import MemoryResearchStore
from research_api.template import TEMPLATE_ID

from .conftest import StaticEngine

USER = UUID("00000000-0000-4000-8000-000000000001")
OTHER = UUID("00000000-0000-4000-8000-000000000002")


def create_request() -> CreateRunRequest:
    return CreateRunRequest(topic="Cross-repository code retrieval", templateId="codebase_rag_retrieval_v1")


def hypothesis(version: int) -> HypothesisInput:
    return HypothesisInput(
        expectedVersion=version,
        statement="The complete retrieval pipeline increases nDCG@10 over body-only BM25.",
        independentVariable="retrieval_pipeline",
        dependentMetric="ndcg_at_10",
        expectedDirection=ExpectedDirection.INCREASE,
        rationale="AST fields, local LSA retrieval, and reranking expose complementary code signals.",
    )


@pytest.mark.asyncio
async def test_owner_version_and_create_idempotency_are_enforced():
    store = MemoryResearchStore()
    request = create_request()
    fingerprint = request_fingerprint("POST", "/v1/research-runs", request.model_dump(by_alias=True))
    first = await store.create_run(USER, "create-store-001", fingerprint, request)
    replay = await store.create_run(USER, "create-store-001", fingerprint, request)
    assert replay.run.id == first.run.id

    with pytest.raises(DomainError) as wrong_owner:
        await store.get_snapshot(OTHER, first.run.id)
    assert wrong_owner.value.code == "run_not_found"

    with pytest.raises(DomainError) as stale:
        await store.put_hypothesis(USER, first.run.id, hypothesis(99))
    assert stale.value.code == "version_conflict"

    changed = CreateRunRequest(topic="Different topic", templateId="codebase_rag_retrieval_v1")
    changed_fingerprint = request_fingerprint("POST", "/v1/research-runs", changed.model_dump(by_alias=True))
    with pytest.raises(DomainError) as conflict:
        await store.create_run(USER, "create-store-001", changed_fingerprint, changed)
    assert conflict.value.code == "idempotency_conflict"


@pytest.mark.asyncio
async def test_retry_schedules_failed_attempt_and_untouched_sibling():
    store = MemoryResearchStore()
    request = create_request()
    create_hash = request_fingerprint("POST", "/v1/research-runs", request.model_dump(by_alias=True))
    snapshot = await store.create_run(USER, "create-retry-001", create_hash, request)
    snapshot = await store.put_hypothesis(USER, snapshot.run.id, hypothesis(snapshot.run.version))
    enqueue_hash = request_fingerprint(
        "POST",
        f"/v1/research-runs/{snapshot.run.id}/experiment-jobs",
        {"expectedVersion": snapshot.run.version, "benchmark": {"includeAllVariants": True}},
    )
    queued = await store.enqueue_suite(
        USER,
        snapshot.run.id,
        "enqueue-retry-001",
        enqueue_hash,
        snapshot.run.version,
    )
    baseline = next(item for item in queued.jobs if item.variant == "bm25_body")
    await store.mark_job_running(baseline.id)
    await store.finish_job(
        baseline.id,
        None,
        RunFailure(code="experiment_failed", message="The fixed experiment failed"),
    )
    failed = await store.get_snapshot(USER, snapshot.run.id)
    assert failed.run.status.value == "failed"

    retry_body = {"expectedVersion": failed.run.version}
    retry_hash = request_fingerprint("POST", f"/v1/research-runs/{snapshot.run.id}/retry", retry_body)
    accepted, jobs = await store.retry_run(
        USER,
        snapshot.run.id,
        "retry-store-001",
        retry_hash,
        failed.run.version,
    )
    assert accepted.run.status.value == "queued"
    assert {(job.variant, job.attempt, job.status) for job in jobs} == {
        ("bm25_body", 2, JobStatus.QUEUED),
        ("bm25_ast", 1, JobStatus.QUEUED),
        ("hybrid", 1, JobStatus.QUEUED),
        ("hybrid_rerank", 1, JobStatus.QUEUED),
    }


@pytest.mark.asyncio
async def test_dispatcher_recovers_queued_jobs_after_service_start(fixed_metrics):
    store = MemoryResearchStore()
    request = create_request()
    create_hash = request_fingerprint("POST", "/v1/research-runs", request.model_dump(by_alias=True))
    snapshot = await store.create_run(USER, "create-recover-001", create_hash, request)
    snapshot = await store.put_hypothesis(USER, snapshot.run.id, hypothesis(snapshot.run.version))
    enqueue_body = {
        "expectedVersion": snapshot.run.version,
        "benchmark": {"includeAllVariants": True},
    }
    queued = await store.enqueue_suite(
        USER,
        snapshot.run.id,
        "enqueue-recover-001",
        request_fingerprint("POST", f"/v1/research-runs/{snapshot.run.id}/experiment-jobs", enqueue_body),
        snapshot.run.version,
    )
    assert all(job.status == JobStatus.QUEUED for job in queued.jobs)

    service = ResearchService(
        store,
        StaticEngine({TEMPLATE_ID: fixed_metrics}),
        dispatcher_poll_seconds=0.01,
    )
    service.start()
    try:
        for _ in range(100):
            recovered = await store.get_snapshot(USER, snapshot.run.id)
            if recovered.run.status.value == "analysis_ready":
                break
            await asyncio.sleep(0.01)
        assert recovered.run.status.value == "analysis_ready"
    finally:
        await service.close()
