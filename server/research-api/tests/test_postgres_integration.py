from __future__ import annotations

import os
from uuid import UUID

import asyncpg
import pytest

from research_api.fingerprint import request_fingerprint
from research_api.models import (
    ArtifactKind,
    CreateRunRequest,
    ExpectedDirection,
    HypothesisInput,
)
from research_api.postgres_store import PostgresResearchStore
from research_api.template import analyze_results
from research_api.template_catalog import get_template

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL is required for PostgreSQL integration tests",
)

USER = UUID("00000000-0000-4000-8000-000000000011")
OTHER = UUID("00000000-0000-4000-8000-000000000012")


@pytest.mark.asyncio
async def test_postgres_vertical_slice_and_rls(all_fixed_metrics):
    database_url = os.environ["TEST_DATABASE_URL"]
    admin = await asyncpg.connect(database_url)
    try:
        await admin.execute(
            """
            truncate public.research_events, public.research_artifacts,
              public.research_experiment_jobs, public.research_hypotheses,
              public.research_runs, public.profiles, auth.users cascade
            """
        )
        await admin.executemany("insert into auth.users (id) values ($1)", [(USER,), (OTHER,)])
        await admin.executemany(
            """
            insert into public.profiles (id, email, display_name, role)
            values ($1, $2, $3, 'student')
            """,
            [
                (USER, "owner@example.com", "Owner"),
                (OTHER, "other@example.com", "Other"),
            ],
        )
    finally:
        await admin.close()

    store = await PostgresResearchStore.connect(database_url)
    try:
        create = CreateRunRequest(
            topic="Cross-repository code retrieval", templateId="codebase_rag_retrieval_v1"
        )
        create_hash = request_fingerprint(
            "POST", "/v1/research-runs", create.model_dump(mode="json", by_alias=True)
        )
        snapshot = await store.create_run(USER, "create-postgres-001", create_hash, create)
        run_id = snapshot.run.id
        snapshot = await store.put_hypothesis(
            USER,
            run_id,
            HypothesisInput(
                expectedVersion=1,
                statement="The complete retrieval pipeline increases nDCG@10 over body-only BM25.",
                independentVariable="retrieval_pipeline",
                dependentMetric="ndcg_at_10",
                expectedDirection=ExpectedDirection.INCREASE,
                rationale="AST fields, local LSA retrieval, and reranking expose complementary code signals.",
            ),
        )
        enqueue_body = {
            "expectedVersion": snapshot.run.version,
            "benchmark": {"includeAllVariants": True},
        }
        enqueue_hash = request_fingerprint(
            "POST", f"/v1/research-runs/{run_id}/experiment-jobs", enqueue_body
        )
        queued = await store.enqueue_suite(
            USER,
            run_id,
            "enqueue-postgres-001",
            enqueue_hash,
            snapshot.run.version,
        )
        for job in queued.jobs:
            await store.mark_job_running(job.id)
            await store.finish_job(job.id, all_fixed_metrics[create.template_id][job.variant], None)
        snapshot = await store.get_snapshot(USER, run_id)
        assert snapshot.run.status.value == "analysis_ready"

        analysis_hash = request_fingerprint(
            "POST",
            f"/v1/research-runs/{run_id}/analysis",
            {"expectedVersion": snapshot.run.version},
        )
        await store.begin_analysis(
            USER,
            run_id,
            "analysis-postgres-001",
            analysis_hash,
            snapshot.run.version,
        )
        run, direction, metrics = await store.analysis_inputs(run_id)
        analysis = analyze_results(run.template_id, metrics, direction)
        await store.complete_analysis(run_id, analysis)
        snapshot = await store.get_snapshot(USER, run_id)
        assert snapshot.run.status.value == "verification_required"

        verify_hash = request_fingerprint(
            "POST",
            f"/v1/research-runs/{run_id}/verify",
            {"expectedVersion": snapshot.run.version},
        )
        await store.begin_verification(
            USER,
            run_id,
            "verify-postgres-001",
            verify_hash,
            snapshot.run.version,
        )
        await store.complete_verification(
            run_id,
            {"passed": True, "verdict": analysis["verdict"]},
            {"title": "Cross-repository code retrieval", "verdict": analysis["verdict"]},
            True,
        )
        snapshot = await store.get_snapshot(USER, run_id)
        card = next(item for item in snapshot.artifacts if item.kind == ArtifactKind.RESEARCH_CARD)

        publish_hash = request_fingerprint(
            "POST",
            f"/v1/research-artifacts/{card.id}/publish",
            {"expectedVersion": snapshot.run.version},
        )
        published = await store.publish_artifact(
            USER,
            card.id,
            "publish-postgres-001",
            publish_hash,
            snapshot.run.version,
        )
        assert published.artifact.visibility.value == "public"

        admin = await asyncpg.connect(database_url)
        try:
            async with admin.transaction():
                await admin.execute("set local role authenticated")
                await admin.execute("select set_config('request.jwt.claim.sub', $1, true)", str(USER))
                assert (
                    await admin.fetchval("select count(*) from public.research_runs where id = $1", run_id)
                    == 1
                )
                await admin.execute("select set_config('request.jwt.claim.sub', $1, true)", str(OTHER))
                assert (
                    await admin.fetchval("select count(*) from public.research_runs where id = $1", run_id)
                    == 0
                )
                with pytest.raises(asyncpg.InsufficientPrivilegeError):
                    await admin.execute(
                        """
                        insert into public.research_runs (
                          user_id, idempotency_key, create_request_sha256, topic, research_question
                        ) values ($1, 'forbidden-insert', $2, 'Bad', 'Bad question text')
                        """,
                        OTHER,
                        "0" * 64,
                    )

            async with admin.transaction():
                await admin.execute("set local role anon")
                public_row = await admin.fetchrow(
                    """
                    select id, kind, content, visibility
                    from public.research_artifacts where id = $1
                    """,
                    card.id,
                )
                assert public_row["kind"] == "research_card"
                assert public_row["visibility"] == "public"

            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                async with admin.transaction():
                    await admin.execute("set local role anon")
                    await admin.fetchval(
                        "select user_id from public.research_artifacts where id = $1", card.id
                    )
        finally:
            await admin.close()

        unpublish_hash = request_fingerprint(
            "DELETE",
            f"/v1/research-artifacts/{card.id}/publication",
            {"expectedVersion": published.run_version},
        )
        unpublished = await store.unpublish_artifact(
            USER,
            card.id,
            "unpublish-postgres-001",
            unpublish_hash,
            published.run_version,
        )
        assert unpublished.artifact.visibility.value == "private"

        governance = get_template("mcp_registry_governance_v1")
        second_create = CreateRunRequest(
            topic="Offline MCP governance", templateId=governance.id
        )
        second_hash = request_fingerprint(
            "POST",
            "/v1/research-runs",
            second_create.model_dump(mode="json", by_alias=True),
        )
        second = await store.create_run(
            USER, "create-postgres-mcp-001", second_hash, second_create
        )
        second = await store.put_hypothesis(
            USER,
            second.run.id,
            HypothesisInput(
                expectedVersion=second.run.version,
                statement="Full governance improves policy utility over schema-only validation.",
                independentVariable=governance.independent_variable,
                dependentMetric=governance.primary_metric,
                expectedDirection=governance.expected_direction,
                rationale="Scope and PII controls cover failures that schema validation cannot detect.",
            ),
        )
        second_body = {
            "expectedVersion": second.run.version,
            "benchmark": {"includeAllVariants": True},
        }
        second_queue = await store.enqueue_suite(
            USER,
            second.run.id,
            "enqueue-postgres-mcp-001",
            request_fingerprint(
                "POST",
                f"/v1/research-runs/{second.run.id}/experiment-jobs",
                second_body,
            ),
            second.run.version,
        )
        for job in second_queue.jobs:
            await store.mark_job_running(job.id)
            await store.finish_job(
                job.id, all_fixed_metrics[governance.id][job.variant], None
            )
        second = await store.get_snapshot(USER, second.run.id)
        assert second.run.status.value == "analysis_ready"
        assert {job.variant for job in second.jobs} == {
            variant.id for variant in governance.variants
        }
    finally:
        await store.close()
