from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse
from uuid import UUID, uuid4

import asyncpg

from .errors import DomainError, idempotency_conflict, invalid_state, not_found, version_conflict
from .models import (
    Actor,
    ArtifactCommandResult,
    ArtifactKind,
    ArtifactList,
    ArtifactRecord,
    Budget,
    CommandAccepted,
    CreateRunRequest,
    EventRecord,
    ExpectedDirection,
    ExperimentJob,
    ExperimentMetrics,
    ExperimentSuiteAccepted,
    HypothesisInput,
    HypothesisRecord,
    HypothesisStatus,
    JobRecord,
    JobStatus,
    ListRunsResponse,
    ResearchArtifact,
    ResearchEvent,
    ResearchHypothesis,
    ResearchRun,
    ResearchStage,
    ResourceLimits,
    RunFailure,
    RunRecord,
    RunSnapshot,
    RunStatus,
    VerificationStatus,
    Visibility,
)
from .store import utcnow
from .template import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    canonical_json_sha256,
    experiment_config,
    source_bundle_sha256,
    template_variants,
)
from .template_catalog import get_template


async def _initialize_connection(connection: asyncpg.Connection) -> None:
    for type_name in ("json", "jsonb"):
        await connection.set_type_codec(
            type_name,
            schema="pg_catalog",
            encoder=lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":")),
            decoder=json.loads,
        )


class PostgresResearchStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def connect(cls, database_url: str) -> PostgresResearchStore:
        parsed = urlparse(database_url)
        local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        pool = await asyncpg.create_pool(
            database_url,
            min_size=1,
            max_size=5,
            command_timeout=30,
            ssl=None if local else "require",
            init=_initialize_connection,
        )
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()

    async def create_run(
        self, user_id: UUID, key: str, fingerprint: str, request: CreateRunRequest
    ) -> RunSnapshot:
        async with self._pool.acquire() as connection, connection.transaction():
            await self._advisory_lock(connection, user_id, "run.create", key)
            existing = await connection.fetchrow(
                """
                select * from public.research_runs
                where user_id = $1 and idempotency_key = $2
                """,
                user_id,
                key,
            )
            if existing:
                if existing["create_request_sha256"] != fingerprint:
                    raise idempotency_conflict()
                return await self._snapshot(connection, self._run(existing))

            try:
                template = get_template(request.template_id)
            except ValueError as exc:
                raise DomainError(
                    "template_not_found", "The research template was not found", 404
                ) from exc
            row = await connection.fetchrow(
                """
                insert into public.research_runs (
                  user_id, idempotency_key, create_request_sha256, topic,
                  research_question, engine, template_id, template_version,
                  status, current_stage, version, budget
                ) values ($1, $2, $3, $4, $5, 'minimal', $6, $7, 'draft', 'hypothesis', 1, $8)
                returning *
                """,
                user_id,
                key,
                fingerprint,
                request.topic.strip(),
                template.research_question,
                template.id,
                template.version,
                Budget().model_dump(mode="json", by_alias=True),
            )
            run = self._run(row)
            await self._create_artifact(
                connection,
                run,
                ResearchStage.TOPIC,
                ArtifactKind.TOPIC_BRIEF,
                {
                    "topic": run.topic,
                    "researchQuestion": template.research_question,
                    "templateId": template.id,
                    "templateVersion": template.version,
                },
            )
            await self._append_event(
                connection,
                run,
                Actor.USER,
                "run.created",
                ResearchStage.TOPIC,
                {"runVersion": 1, "templateId": template.id},
            )
            return await self._snapshot(connection, run)

    async def list_runs(
        self, user_id: UUID, status: RunStatus | None = None, limit: int = 20
    ) -> ListRunsResponse:
        async with self._pool.acquire() as connection:
            if status:
                rows = await connection.fetch(
                    """
                    select * from public.research_runs
                    where user_id = $1 and status = $2
                    order by created_at desc, id desc limit $3
                    """,
                    user_id,
                    status.value,
                    limit,
                )
            else:
                rows = await connection.fetch(
                    """
                    select * from public.research_runs
                    where user_id = $1
                    order by created_at desc, id desc limit $2
                    """,
                    user_id,
                    limit,
                )
            return ListRunsResponse(items=[self._public_run(self._run(row)) for row in rows])

    async def get_snapshot(self, user_id: UUID, run_id: UUID) -> RunSnapshot:
        async with self._pool.acquire() as connection:
            run = await self._owned_run(connection, user_id, run_id)
            return await self._snapshot(connection, run)

    async def put_hypothesis(self, user_id: UUID, run_id: UUID, request: HypothesisInput) -> RunSnapshot:
        async with self._pool.acquire() as connection, connection.transaction():
            run = await self._owned_run(connection, user_id, run_id, lock=True)
            self._check_version(run, request.expected_version)
            if run.status != RunStatus.DRAFT:
                raise invalid_state("A hypothesis can only be accepted from a draft run")
            template = get_template(run.template_id)
            if request.independent_variable != template.independent_variable:
                raise invalid_state("The independent variable does not match the selected template")
            if request.dependent_metric != template.primary_metric:
                raise invalid_state("The dependent metric does not match the selected template")
            now = utcnow()
            row = await connection.fetchrow(
                """
                insert into public.research_hypotheses (
                  run_id, user_id, statement, independent_variable, dependent_metric,
                  expected_direction, rationale, status, accepted_at
                ) values ($1, $2, $3, $4, $5, $6, $7, 'accepted', $8)
                returning *
                """,
                run.id,
                user_id,
                request.statement.strip(),
                request.independent_variable,
                request.dependent_metric,
                request.expected_direction.value,
                request.rationale.strip(),
                now,
            )
            hypothesis = self._hypothesis(row)
            updated = await connection.fetchrow(
                """
                update public.research_runs
                set status = 'hypothesis_ready', current_stage = 'experiment', version = version + 1
                where id = $1 and user_id = $2 returning *
                """,
                run.id,
                user_id,
            )
            run = self._run(updated)
            await self._create_artifact(
                connection,
                run,
                ResearchStage.HYPOTHESIS,
                ArtifactKind.HYPOTHESIS,
                hypothesis.model_dump(
                    mode="json",
                    by_alias=True,
                    exclude={"id", "run_id", "created_at", "updated_at", "accepted_at"},
                ),
            )
            await self._append_event(
                connection,
                run,
                Actor.USER,
                "hypothesis.accepted",
                ResearchStage.HYPOTHESIS,
                {"hypothesisId": str(hypothesis.id), "runVersion": run.version},
            )
            return await self._snapshot(connection, run)

    async def enqueue_suite(
        self,
        user_id: UUID,
        run_id: UUID,
        key: str,
        fingerprint: str,
        expected_version: int,
    ) -> ExperimentSuiteAccepted:
        async with self._pool.acquire() as connection, connection.transaction():
            await self._advisory_lock(connection, user_id, "experiment.enqueue", key)
            replay = await self._command_replay(connection, user_id, "experiment.enqueue", key, fingerprint)
            if replay:
                run = await self._owned_run(connection, user_id, run_id)
                jobs = await connection.fetch(
                    "select * from public.research_experiment_jobs where id = any($1::uuid[])",
                    [UUID(value) for value in replay.payload["jobIds"]],
                )
                by_id = {row["id"]: self._job(row) for row in jobs}
                ordered = [by_id[UUID(value)] for value in replay.payload["jobIds"]]
                return ExperimentSuiteAccepted(
                    run=self._public_run(run),
                    suite_id=UUID(replay.payload["suiteId"]),
                    jobs=[self._public_job(job) for job in ordered],
                    event_id=replay.id,
                )

            run = await self._owned_run(connection, user_id, run_id, lock=True)
            self._check_version(run, expected_version)
            if run.status != RunStatus.HYPOTHESIS_READY:
                raise invalid_state("The run is not ready to enqueue experiments")
            hypothesis_row = await connection.fetchrow(
                """
                select * from public.research_hypotheses
                where run_id = $1 and user_id = $2 and status = 'accepted'
                """,
                run.id,
                user_id,
            )
            if not hypothesis_row:
                raise invalid_state("An accepted hypothesis is required")
            hypothesis = self._hypothesis(hypothesis_row)
            suite_id = uuid4()
            jobs: list[JobRecord] = []
            variants = template_variants(run.template_id)
            for variant in variants:
                job_row = await connection.fetchrow(
                    """
                    insert into public.research_experiment_jobs (
                      run_id, user_id, hypothesis_id, suite_id, variant,
                      template_id, template_version, attempt, status,
                      bootstrap_seed, bootstrap_replicates, config, resource_limits
                    ) values ($1, $2, $3, $4, $5, $6, $7, 1, 'queued', $8, $9, $10, $11)
                    returning *
                    """,
                    run.id,
                    user_id,
                    hypothesis.id,
                    suite_id,
                    variant,
                    run.template_id,
                    run.template_version,
                    BOOTSTRAP_SEED,
                    BOOTSTRAP_REPLICATES,
                    experiment_config(run.template_id, variant).model_dump(
                        mode="json", by_alias=True
                    ),
                    ResourceLimits().model_dump(mode="json", by_alias=True),
                )
                jobs.append(self._job(job_row))
            updated = await connection.fetchrow(
                """
                update public.research_runs
                set status = 'queued', current_stage = 'experiment', version = version + 1
                where id = $1 and user_id = $2 returning *
                """,
                run.id,
                user_id,
            )
            run = self._run(updated)
            await self._create_artifact(
                connection,
                run,
                ResearchStage.EXPERIMENT,
                ArtifactKind.EXPERIMENT_SPEC,
                {
                    "templateId": run.template_id,
                    "templateVersion": run.template_version,
                    "suiteId": str(suite_id),
                    "bootstrapSeed": BOOTSTRAP_SEED,
                    "bootstrapReplicates": BOOTSTRAP_REPLICATES,
                    "variants": [
                        {
                            "id": job.variant,
                            "config": job.config.model_dump(mode="json", by_alias=True),
                        }
                        for job in jobs
                    ],
                },
            )
            await self._create_artifact(
                connection,
                run,
                ResearchStage.EXPERIMENT,
                ArtifactKind.SOURCE_BUNDLE,
                {
                    "sha256": source_bundle_sha256(run.template_id),
                    "templateVersion": run.template_version,
                },
            )
            event = await self._append_event(
                connection,
                run,
                Actor.USER,
                "experiment.suite_queued",
                ResearchStage.EXPERIMENT,
                {
                    "suiteId": str(suite_id),
                    "jobIds": [str(job.id) for job in jobs],
                    "runVersion": run.version,
                },
                command_name="experiment.enqueue",
                idempotency_key=key,
                request_sha256=fingerprint,
            )
            return ExperimentSuiteAccepted(
                run=self._public_run(run),
                suite_id=suite_id,
                jobs=[self._public_job(job) for job in jobs],
                event_id=event.id,
            )

    async def mark_job_running(self, job_id: UUID) -> JobRecord | None:
        async with self._pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                "select * from public.research_experiment_jobs where id = $1 for update", job_id
            )
            if not row or row["status"] != JobStatus.QUEUED.value:
                return None
            job = self._job(row)
            run_row = await connection.fetchrow(
                "select * from public.research_runs where id = $1 for update", job.run_id
            )
            run = self._run(run_row)
            if run.status in {RunStatus.CANCELLED, RunStatus.FAILED}:
                return None
            job_row = await connection.fetchrow(
                """
                update public.research_experiment_jobs
                set status = 'running', started_at = now()
                where id = $1 returning *
                """,
                job.id,
            )
            job = self._job(job_row)
            if run.status == RunStatus.QUEUED:
                run_row = await connection.fetchrow(
                    """
                    update public.research_runs
                    set status = 'running', version = version + 1
                    where id = $1 returning *
                    """,
                    run.id,
                )
                run = self._run(run_row)
            await self._append_event(
                connection,
                run,
                Actor.RUNNER,
                "experiment.job_started",
                ResearchStage.EXPERIMENT,
                {"jobId": str(job.id), "variant": job.variant, "runVersion": run.version},
            )
            return job

    async def list_queued_jobs(self, limit: int = 20) -> list[JobRecord]:
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                """
                select * from public.research_experiment_jobs
                where status = 'queued'
                order by queued_at, id limit $1
                """,
                limit,
            )
            return [self._job(row) for row in rows]

    async def list_stale_running_job_ids(self, older_than_seconds: int) -> list[UUID]:
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                """
                select id from public.research_experiment_jobs
                where status = 'running'
                  and started_at < now() - make_interval(secs => $1)
                order by started_at, id limit 20
                """,
                older_than_seconds,
            )
            return [row["id"] for row in rows]

    async def finish_job(
        self,
        job_id: UUID,
        metrics: ExperimentMetrics | None,
        failure: RunFailure | None,
        timed_out: bool = False,
    ) -> None:
        async with self._pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                "select * from public.research_experiment_jobs where id = $1 for update", job_id
            )
            if not row or row["status"] != JobStatus.RUNNING.value:
                return
            job = self._job(row)
            run_row = await connection.fetchrow(
                "select * from public.research_runs where id = $1 for update", job.run_id
            )
            run = self._run(run_row)
            status = (
                JobStatus.TIMED_OUT if timed_out else JobStatus.FAILED if failure else JobStatus.SUCCEEDED
            )
            finished = utcnow()
            duration_ms = max(
                0,
                int((finished - (job.started_at or finished)).total_seconds() * 1000),
            )
            job_row = await connection.fetchrow(
                """
                update public.research_experiment_jobs
                set status = $2, metrics = $3, error_code = $4, error_message = $5,
                    finished_at = $6, duration_ms = $7
                where id = $1 returning *
                """,
                job.id,
                status.value,
                metrics.model_dump(mode="json", by_alias=True) if metrics else None,
                failure.code if failure else None,
                failure.message if failure else None,
                finished,
                duration_ms,
            )
            job = self._job(job_row)
            await self._append_event(
                connection,
                run,
                Actor.RUNNER,
                f"experiment.job_{status.value}",
                ResearchStage.EXPERIMENT,
                {"jobId": str(job.id), "variant": job.variant, "attempt": job.attempt},
            )
            if failure:
                if run.status != RunStatus.FAILED:
                    updated = await connection.fetchrow(
                        """
                        update public.research_runs
                        set status = 'failed', current_stage = 'experiment',
                            failure_code = 'experiment_failed',
                            failure_message = 'The fixed experiment failed',
                            version = version + 1
                        where id = $1 returning *
                        """,
                        run.id,
                    )
                    run = self._run(updated)
                    await self._append_event(
                        connection,
                        run,
                        Actor.RUNNER,
                        "run.failed",
                        ResearchStage.EXPERIMENT,
                        {"code": "experiment_failed", "runVersion": run.version},
                    )
                return
            if run.status in {RunStatus.FAILED, RunStatus.CANCELLED}:
                return

            latest = await self._latest_jobs(connection, run.id)
            variants = template_variants(run.template_id)
            if all(
                latest.get(variant) and latest[variant].status == JobStatus.SUCCEEDED
                for variant in variants
            ):
                raw = {
                    variant: latest[variant].metrics.model_dump(mode="json", by_alias=True)
                    for variant in variants
                    if latest[variant].metrics
                }
                await self._create_artifact(
                    connection,
                    run,
                    ResearchStage.EXPERIMENT,
                    ArtifactKind.RAW_METRICS,
                    raw,
                )
                updated = await connection.fetchrow(
                    """
                    update public.research_runs
                    set status = 'analysis_ready', current_stage = 'analysis',
                        failure_code = null, failure_message = null, version = version + 1
                    where id = $1 returning *
                    """,
                    run.id,
                )
                run = self._run(updated)
                await self._append_event(
                    connection,
                    run,
                    Actor.RUNNER,
                    "experiment.suite_succeeded",
                    ResearchStage.EXPERIMENT,
                    {"runVersion": run.version},
                )

    async def begin_analysis(
        self, user_id: UUID, run_id: UUID, key: str, fingerprint: str, expected_version: int
    ) -> CommandAccepted:
        return await self._begin_command(
            user_id,
            run_id,
            key,
            fingerprint,
            expected_version,
            "analysis.start",
            RunStatus.ANALYSIS_READY,
            RunStatus.ANALYZING,
            ResearchStage.ANALYSIS,
            "analysis.started",
        )

    async def analysis_inputs(
        self, run_id: UUID
    ) -> tuple[ResearchRun, ExpectedDirection, dict[str, ExperimentMetrics]]:
        async with self._pool.acquire() as connection:
            run_row = await connection.fetchrow("select * from public.research_runs where id = $1", run_id)
            if not run_row or run_row["status"] != RunStatus.ANALYZING.value:
                raise invalid_state("The run is not analyzing")
            hypothesis_row = await connection.fetchrow(
                "select * from public.research_hypotheses where run_id = $1", run_id
            )
            hypothesis = self._hypothesis(hypothesis_row)
            return (
                self._public_run(self._run(run_row)),
                hypothesis.expected_direction,
                await self._latest_metrics(connection, run_id),
            )

    async def complete_analysis(self, run_id: UUID, content: dict[str, Any]) -> None:
        async with self._pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                "select * from public.research_runs where id = $1 for update", run_id
            )
            if not row or row["status"] != RunStatus.ANALYZING.value:
                return
            run = self._run(row)
            await self._create_artifact(
                connection, run, ResearchStage.ANALYSIS, ArtifactKind.ANALYSIS, content
            )
            updated = await connection.fetchrow(
                """
                update public.research_runs
                set status = 'verification_required', current_stage = 'verification',
                    version = version + 1
                where id = $1 returning *
                """,
                run.id,
            )
            run = self._run(updated)
            await self._append_event(
                connection,
                run,
                Actor.SYSTEM,
                "analysis.completed",
                ResearchStage.ANALYSIS,
                {"verdict": content["verdict"], "runVersion": run.version},
            )

    async def begin_verification(
        self, user_id: UUID, run_id: UUID, key: str, fingerprint: str, expected_version: int
    ) -> CommandAccepted:
        return await self._begin_command(
            user_id,
            run_id,
            key,
            fingerprint,
            expected_version,
            "verification.start",
            RunStatus.VERIFICATION_REQUIRED,
            RunStatus.VERIFYING,
            ResearchStage.VERIFICATION,
            "verification.started",
        )

    async def verification_inputs(
        self, run_id: UUID
    ) -> tuple[
        ResearchRun,
        ResearchHypothesis,
        dict[str, ExperimentMetrics],
        dict[str, Any],
    ]:
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow("select * from public.research_runs where id = $1", run_id)
            if not row or row["status"] != RunStatus.VERIFYING.value:
                raise invalid_state("The run is not verifying")
            run = self._run(row)
            hypothesis_row = await connection.fetchrow(
                "select * from public.research_hypotheses where run_id = $1", run.id
            )
            artifact_row = await connection.fetchrow(
                """
                select * from public.research_artifacts
                where run_id = $1 and kind = 'analysis'
                order by version desc limit 1
                """,
                run.id,
            )
            if not hypothesis_row or not artifact_row or artifact_row["content"] is None:
                raise invalid_state("The deterministic analysis artifact is missing")
            return (
                self._public_run(run),
                self._public_hypothesis(self._hypothesis(hypothesis_row)),
                await self._latest_metrics(connection, run.id),
                dict(artifact_row["content"]),
            )

    async def complete_verification(
        self,
        run_id: UUID,
        bundle: dict[str, Any],
        research_card: dict[str, Any] | None,
        passed: bool,
    ) -> None:
        async with self._pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                "select * from public.research_runs where id = $1 for update", run_id
            )
            if not row or row["status"] != RunStatus.VERIFYING.value:
                return
            run = self._run(row)
            await self._create_artifact(
                connection,
                run,
                ResearchStage.VERIFICATION,
                ArtifactKind.VERIFICATION_BUNDLE,
                bundle,
                verification_status=(VerificationStatus.PASSED if passed else VerificationStatus.FAILED),
            )
            if passed and research_card:
                await self._create_artifact(
                    connection,
                    run,
                    ResearchStage.PUBLICATION,
                    ArtifactKind.RESEARCH_CARD,
                    research_card,
                    verification_status=VerificationStatus.PASSED,
                )
            if passed:
                updated = await connection.fetchrow(
                    """
                    update public.research_runs
                    set status = 'completed', current_stage = 'publication',
                        completed_at = now(), failure_code = null, failure_message = null,
                        version = version + 1
                    where id = $1 returning *
                    """,
                    run.id,
                )
                event_type = "verification.passed"
            else:
                updated = await connection.fetchrow(
                    """
                    update public.research_runs
                    set status = 'failed', current_stage = 'verification',
                        failure_code = 'verification_failed',
                        failure_message = 'The reproduction check did not pass',
                        version = version + 1
                    where id = $1 returning *
                    """,
                    run.id,
                )
                event_type = "verification.failed"
            run = self._run(updated)
            await self._append_event(
                connection,
                run,
                Actor.VERIFIER,
                event_type,
                ResearchStage.VERIFICATION,
                {"passed": passed, "runVersion": run.version},
            )

    async def fail_run(self, run_id: UUID, code: str, message: str, stage: ResearchStage) -> None:
        async with self._pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                "select * from public.research_runs where id = $1 for update", run_id
            )
            if not row or row["status"] in {RunStatus.COMPLETED.value, RunStatus.CANCELLED.value}:
                return
            updated = await connection.fetchrow(
                """
                update public.research_runs
                set status = 'failed', current_stage = $2, failure_code = $3,
                    failure_message = $4, version = version + 1
                where id = $1 returning *
                """,
                run_id,
                stage.value,
                code,
                message,
            )
            run = self._run(updated)
            await self._append_event(
                connection,
                run,
                Actor.SYSTEM,
                "run.failed",
                stage,
                {"code": code, "runVersion": run.version},
            )

    async def retry_run(
        self, user_id: UUID, run_id: UUID, key: str, fingerprint: str, expected_version: int
    ) -> tuple[CommandAccepted, list[JobRecord]]:
        async with self._pool.acquire() as connection, connection.transaction():
            await self._advisory_lock(connection, user_id, "run.retry", key)
            replay = await self._command_replay(connection, user_id, "run.retry", key, fingerprint)
            if replay:
                run = await self._owned_run(connection, user_id, run_id)
                ids = [UUID(value) for value in replay.payload.get("jobIds", [])]
                rows = (
                    await connection.fetch(
                        "select * from public.research_experiment_jobs where id = any($1::uuid[])",
                        ids,
                    )
                    if ids
                    else []
                )
                by_id = {row["id"]: self._job(row) for row in rows}
                jobs = [by_id[item] for item in ids]
                return CommandAccepted(run=self._public_run(run), event_id=replay.id), jobs

            run = await self._owned_run(connection, user_id, run_id, lock=True)
            self._check_version(run, expected_version)
            if run.status != RunStatus.FAILED or not run.failure:
                raise invalid_state("Only a failed run can be retried")
            retry_jobs: list[JobRecord] = []
            if run.failure.code == "experiment_failed":
                latest = await self._latest_jobs(connection, run.id)
                if any(job.status == JobStatus.RUNNING for job in latest.values()):
                    raise invalid_state("Wait for active sibling jobs to settle before retrying")
                failed = [
                    job for job in latest.values() if job.status in {JobStatus.FAILED, JobStatus.TIMED_OUT}
                ]
                if not failed or any(job.attempt >= 2 for job in failed):
                    raise DomainError("budget_exceeded", "The retry budget is exhausted", 429)
                suite_id = uuid4()
                for previous in failed:
                    row = await connection.fetchrow(
                        """
                        insert into public.research_experiment_jobs (
                          run_id, user_id, hypothesis_id, suite_id, variant,
                          template_id, template_version, attempt, status,
                          bootstrap_seed, bootstrap_replicates, config, resource_limits
                        ) values ($1, $2, $3, $4, $5, $6, $7, $8, 'queued', $9, $10, $11, $12)
                        returning *
                        """,
                        run.id,
                        user_id,
                        previous.hypothesis_id,
                        suite_id,
                        previous.variant,
                        run.template_id,
                        run.template_version,
                        previous.attempt + 1,
                        BOOTSTRAP_SEED,
                        BOOTSTRAP_REPLICATES,
                        experiment_config(run.template_id, previous.variant).model_dump(
                            mode="json", by_alias=True
                        ),
                        ResourceLimits().model_dump(mode="json", by_alias=True),
                    )
                    retry_jobs.append(self._job(row))
                retry_jobs.extend(job for job in latest.values() if job.status == JobStatus.QUEUED)
                status = RunStatus.QUEUED
                stage = ResearchStage.EXPERIMENT
            elif run.failure.code == "verification_failed":
                status = RunStatus.VERIFICATION_REQUIRED
                stage = ResearchStage.VERIFICATION
            elif run.failure.code == "analysis_failed":
                status = RunStatus.ANALYSIS_READY
                stage = ResearchStage.ANALYSIS
            else:
                raise invalid_state("This failure cannot be retried")
            updated = await connection.fetchrow(
                """
                update public.research_runs
                set status = $2, current_stage = $3, retry_count = retry_count + 1,
                    failure_code = null, failure_message = null, version = version + 1
                where id = $1 returning *
                """,
                run.id,
                status.value,
                stage.value,
            )
            run = self._run(updated)
            event = await self._append_event(
                connection,
                run,
                Actor.USER,
                "run.retried",
                stage,
                {"jobIds": [str(job.id) for job in retry_jobs], "runVersion": run.version},
                command_name="run.retry",
                idempotency_key=key,
                request_sha256=fingerprint,
            )
            return CommandAccepted(run=self._public_run(run), event_id=event.id), retry_jobs

    async def cancel_run(
        self, user_id: UUID, run_id: UUID, key: str, fingerprint: str, expected_version: int
    ) -> CommandAccepted:
        async with self._pool.acquire() as connection, connection.transaction():
            await self._advisory_lock(connection, user_id, "run.cancel", key)
            replay = await self._command_replay(connection, user_id, "run.cancel", key, fingerprint)
            if replay:
                run = await self._owned_run(connection, user_id, run_id)
                return CommandAccepted(run=self._public_run(run), event_id=replay.id)
            run = await self._owned_run(connection, user_id, run_id, lock=True)
            self._check_version(run, expected_version)
            if run.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
                raise invalid_state("A terminal run cannot be cancelled")
            await connection.execute(
                """
                update public.research_experiment_jobs
                set status = 'cancelled', error_code = 'cancelled',
                    error_message = 'The run was cancelled', finished_at = now()
                where run_id = $1 and status in ('queued', 'running')
                """,
                run.id,
            )
            updated = await connection.fetchrow(
                """
                update public.research_runs
                set status = 'cancelled', cancelled_at = now(), version = version + 1
                where id = $1 returning *
                """,
                run.id,
            )
            run = self._run(updated)
            event = await self._append_event(
                connection,
                run,
                Actor.USER,
                "run.cancelled",
                run.current_stage,
                {"runVersion": run.version},
                command_name="run.cancel",
                idempotency_key=key,
                request_sha256=fingerprint,
            )
            return CommandAccepted(run=self._public_run(run), event_id=event.id)

    async def list_events(self, user_id: UUID, run_id: UUID, after: int = 0) -> list[ResearchEvent]:
        async with self._pool.acquire() as connection:
            await self._owned_run(connection, user_id, run_id)
            rows = await connection.fetch(
                """
                select * from public.research_events
                where run_id = $1 and user_id = $2 and id > $3
                order by id
                """,
                run_id,
                user_id,
                after,
            )
            return [self._public_event(self._event(row)) for row in rows]

    async def list_artifacts(
        self, user_id: UUID, run_id: UUID, kind: ArtifactKind | None = None
    ) -> ArtifactList:
        async with self._pool.acquire() as connection:
            await self._owned_run(connection, user_id, run_id)
            if kind:
                rows = await connection.fetch(
                    """
                    select * from public.research_artifacts
                    where run_id = $1 and user_id = $2 and kind = $3
                    order by created_at, id
                    """,
                    run_id,
                    user_id,
                    kind.value,
                )
            else:
                rows = await connection.fetch(
                    """
                    select * from public.research_artifacts
                    where run_id = $1 and user_id = $2
                    order by created_at, id
                    """,
                    run_id,
                    user_id,
                )
            return ArtifactList(items=[self._public_artifact(self._artifact(row)) for row in rows])

    async def publish_artifact(
        self,
        user_id: UUID,
        artifact_id: UUID,
        key: str,
        fingerprint: str,
        expected_version: int,
    ) -> ArtifactCommandResult:
        return await self._publication_command(
            user_id, artifact_id, key, fingerprint, expected_version, publish=True
        )

    async def unpublish_artifact(
        self,
        user_id: UUID,
        artifact_id: UUID,
        key: str,
        fingerprint: str,
        expected_version: int,
    ) -> ArtifactCommandResult:
        return await self._publication_command(
            user_id, artifact_id, key, fingerprint, expected_version, publish=False
        )

    async def _publication_command(
        self,
        user_id: UUID,
        artifact_id: UUID,
        key: str,
        fingerprint: str,
        expected_version: int,
        publish: bool,
    ) -> ArtifactCommandResult:
        function = "publish_research_artifact" if publish else "unpublish_research_artifact"
        try:
            async with self._pool.acquire() as connection:
                result = await connection.fetchval(
                    f"select public.{function}($1, $2, $3, $4, $5)",  # noqa: S608 - fixed allowlist
                    user_id,
                    artifact_id,
                    expected_version,
                    key,
                    fingerprint,
                )
                artifact_row = await connection.fetchrow(
                    """
                    select * from public.research_artifacts
                    where id = $1 and user_id = $2
                    """,
                    artifact_id,
                    user_id,
                )
                if not artifact_row:
                    raise not_found("artifact")
                return ArtifactCommandResult(
                    artifact=self._public_artifact(self._artifact(artifact_row)),
                    run_version=int(result["runVersion"]),
                    event_id=str(result["eventId"]),
                )
        except asyncpg.NoDataFoundError as exc:
            raise not_found("artifact") from exc
        except asyncpg.SerializationError as exc:
            raise version_conflict(expected_version, -1) from exc
        except asyncpg.UniqueViolationError as exc:
            raise idempotency_conflict() from exc
        except asyncpg.InvalidParameterValueError as exc:
            raise invalid_state("The artifact cannot change publication state") from exc

    async def _begin_command(
        self,
        user_id: UUID,
        run_id: UUID,
        key: str,
        fingerprint: str,
        expected_version: int,
        command_name: str,
        required_status: RunStatus,
        next_status: RunStatus,
        stage: ResearchStage,
        event_type: str,
    ) -> CommandAccepted:
        async with self._pool.acquire() as connection, connection.transaction():
            await self._advisory_lock(connection, user_id, command_name, key)
            replay = await self._command_replay(connection, user_id, command_name, key, fingerprint)
            if replay:
                run = await self._owned_run(connection, user_id, run_id)
                return CommandAccepted(run=self._public_run(run), event_id=replay.id)
            run = await self._owned_run(connection, user_id, run_id, lock=True)
            self._check_version(run, expected_version)
            if run.status != required_status:
                raise invalid_state(f"Command requires run state {required_status.value}")
            updated = await connection.fetchrow(
                """
                update public.research_runs
                set status = $2, current_stage = $3, version = version + 1
                where id = $1 returning *
                """,
                run.id,
                next_status.value,
                stage.value,
            )
            run = self._run(updated)
            event = await self._append_event(
                connection,
                run,
                Actor.USER,
                event_type,
                stage,
                {"runVersion": run.version},
                command_name=command_name,
                idempotency_key=key,
                request_sha256=fingerprint,
            )
            return CommandAccepted(run=self._public_run(run), event_id=event.id)

    @staticmethod
    async def _advisory_lock(connection: asyncpg.Connection, user_id: UUID, command: str, key: str) -> None:
        await connection.execute(
            "select pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"{user_id}:{command}:{key}",
        )

    async def _owned_run(
        self,
        connection: asyncpg.Connection,
        user_id: UUID,
        run_id: UUID,
        lock: bool = False,
    ) -> RunRecord:
        suffix = " for update" if lock else ""
        row = await connection.fetchrow(
            f"select * from public.research_runs where id = $1 and user_id = $2{suffix}",  # noqa: S608
            run_id,
            user_id,
        )
        if not row:
            raise not_found()
        return self._run(row)

    @staticmethod
    def _check_version(run: RunRecord, expected: int) -> None:
        if run.version != expected:
            raise version_conflict(expected, run.version)

    async def _command_replay(
        self,
        connection: asyncpg.Connection,
        user_id: UUID,
        command: str,
        key: str,
        fingerprint: str,
    ) -> EventRecord | None:
        row = await connection.fetchrow(
            """
            select * from public.research_events
            where user_id = $1 and command_name = $2 and idempotency_key = $3
            """,
            user_id,
            command,
            key,
        )
        if not row:
            return None
        event = self._event(row)
        if event.request_sha256 != fingerprint:
            raise idempotency_conflict()
        return event

    async def _append_event(
        self,
        connection: asyncpg.Connection,
        run: RunRecord,
        actor: Actor,
        event_type: str,
        stage: ResearchStage,
        payload: dict[str, Any],
        command_name: str | None = None,
        idempotency_key: str | None = None,
        request_sha256: str | None = None,
    ) -> EventRecord:
        row = await connection.fetchrow(
            """
            insert into public.research_events (
              run_id, user_id, actor, event_type, stage, command_name,
              idempotency_key, request_sha256, payload
            ) values ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            returning *
            """,
            run.id,
            run.user_id,
            actor.value,
            event_type,
            stage.value,
            command_name,
            idempotency_key,
            request_sha256,
            payload,
        )
        return self._event(row)

    async def _create_artifact(
        self,
        connection: asyncpg.Connection,
        run: RunRecord,
        stage: ResearchStage,
        kind: ArtifactKind,
        content: dict[str, Any],
        verification_status: VerificationStatus = VerificationStatus.UNVERIFIED,
    ) -> ArtifactRecord:
        version = await connection.fetchval(
            """
            select coalesce(max(version), 0) + 1
            from public.research_artifacts
            where run_id = $1 and kind = $2 and job_id is null
            """,
            run.id,
            kind.value,
        )
        row = await connection.fetchrow(
            """
            insert into public.research_artifacts (
              run_id, user_id, stage, kind, version, content, sha256,
              verification_status, visibility
            ) values ($1, $2, $3, $4, $5, $6, $7, $8, 'private')
            returning *
            """,
            run.id,
            run.user_id,
            stage.value,
            kind.value,
            version,
            content,
            canonical_json_sha256(content),
            verification_status.value,
        )
        return self._artifact(row)

    async def _latest_jobs(
        self, connection: asyncpg.Connection, run_id: UUID
    ) -> dict[str, JobRecord]:
        rows = await connection.fetch(
            """
            select distinct on (variant) *
            from public.research_experiment_jobs
            where run_id = $1
            order by variant, attempt desc
            """,
            run_id,
        )
        return {row["variant"]: self._job(row) for row in rows}

    async def _latest_metrics(
        self, connection: asyncpg.Connection, run_id: UUID
    ) -> dict[str, ExperimentMetrics]:
        run_row = await connection.fetchrow(
            "select template_id from public.research_runs where id = $1", run_id
        )
        if not run_row:
            raise not_found("Research run")
        variants = template_variants(run_row["template_id"])
        latest = await self._latest_jobs(connection, run_id)
        if any(
            variant not in latest
            or latest[variant].status != JobStatus.SUCCEEDED
            or not latest[variant].metrics
            for variant in variants
        ):
            raise invalid_state("All latest experiment variants must have succeeded")
        return {
            variant: latest[variant].metrics
            for variant in variants
            if latest[variant].metrics
        }

    async def _snapshot(self, connection: asyncpg.Connection, run: RunRecord) -> RunSnapshot:
        hypothesis_row = await connection.fetchrow(
            "select * from public.research_hypotheses where run_id = $1", run.id
        )
        job_rows = await connection.fetch(
            """
            select * from public.research_experiment_jobs
            where run_id = $1 order by queued_at, id
            """,
            run.id,
        )
        artifact_rows = await connection.fetch(
            """
            select * from public.research_artifacts
            where run_id = $1 order by created_at, id
            """,
            run.id,
        )
        event_id = await connection.fetchval(
            "select max(id) from public.research_events where run_id = $1", run.id
        )
        return RunSnapshot(
            run=self._public_run(run),
            hypothesis=(
                self._public_hypothesis(self._hypothesis(hypothesis_row)) if hypothesis_row else None
            ),
            jobs=[self._public_job(self._job(row)) for row in job_rows],
            artifacts=[self._public_artifact(self._artifact(row)) for row in artifact_rows],
            last_event_id=str(event_id) if event_id is not None else None,
        )

    @staticmethod
    def _run(row: asyncpg.Record) -> RunRecord:
        failure = (
            RunFailure(code=row["failure_code"], message=row["failure_message"])
            if row["failure_code"]
            else None
        )
        return RunRecord(
            id=row["id"],
            user_id=row["user_id"],
            idempotency_key=row["idempotency_key"],
            create_request_sha256=row["create_request_sha256"],
            topic=row["topic"],
            research_question=row["research_question"],
            engine=row["engine"],
            template_id=row["template_id"],
            template_version=row["template_version"],
            status=RunStatus(row["status"]),
            current_stage=ResearchStage(row["current_stage"]),
            version=row["version"],
            budget=Budget.model_validate(row["budget"]),
            retry_count=row["retry_count"],
            failure=failure,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
            cancelled_at=row["cancelled_at"],
        )

    @staticmethod
    def _hypothesis(row: asyncpg.Record) -> HypothesisRecord:
        return HypothesisRecord(
            id=row["id"],
            run_id=row["run_id"],
            user_id=row["user_id"],
            statement=row["statement"],
            independent_variable=row["independent_variable"],
            dependent_metric=row["dependent_metric"],
            expected_direction=ExpectedDirection(row["expected_direction"]),
            rationale=row["rationale"],
            status=HypothesisStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            accepted_at=row["accepted_at"],
        )

    @staticmethod
    def _job(row: asyncpg.Record) -> JobRecord:
        error = (
            RunFailure(code=row["error_code"], message=row["error_message"]) if row["error_code"] else None
        )
        return JobRecord(
            id=row["id"],
            run_id=row["run_id"],
            user_id=row["user_id"],
            hypothesis_id=row["hypothesis_id"],
            suite_id=row["suite_id"],
            variant=row["variant"],
            template_id=row["template_id"],
            template_version=row["template_version"],
            attempt=row["attempt"],
            status=JobStatus(row["status"]),
            bootstrap_seed=row["bootstrap_seed"],
            bootstrap_replicates=row["bootstrap_replicates"],
            config=(
                experiment_config(row["template_id"], row["variant"])
                if row["config"] is None
                else row["config"]
            ),
            resource_limits=ResourceLimits.model_validate(row["resource_limits"]),
            metrics=ExperimentMetrics.model_validate(row["metrics"]) if row["metrics"] else None,
            error=error,
            queued_at=row["queued_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            duration_ms=row["duration_ms"],
        )

    @staticmethod
    def _artifact(row: asyncpg.Record) -> ArtifactRecord:
        return ArtifactRecord(
            id=row["id"],
            run_id=row["run_id"],
            user_id=row["user_id"],
            job_id=row["job_id"],
            stage=ResearchStage(row["stage"]),
            kind=ArtifactKind(row["kind"]),
            version=row["version"],
            content=row["content"],
            storage_bucket=row["storage_bucket"],
            storage_path=row["storage_path"],
            filename=row["filename"],
            mime_type=row["mime_type"],
            byte_size=row["byte_size"],
            sha256=row["sha256"],
            verification_status=VerificationStatus(row["verification_status"]),
            visibility=Visibility(row["visibility"]),
            created_at=row["created_at"],
            published_at=row["published_at"],
        )

    @staticmethod
    def _event(row: asyncpg.Record) -> EventRecord:
        return EventRecord(
            id=str(row["id"]),
            run_id=row["run_id"],
            user_id=row["user_id"],
            actor=Actor(row["actor"]),
            event_type=row["event_type"],
            stage=ResearchStage(row["stage"]),
            command_name=row["command_name"],
            idempotency_key=row["idempotency_key"],
            request_sha256=row["request_sha256"],
            payload=row["payload"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _public_run(run: RunRecord) -> ResearchRun:
        return ResearchRun.model_validate(run.model_dump())

    @staticmethod
    def _public_hypothesis(hypothesis: HypothesisRecord) -> ResearchHypothesis:
        return ResearchHypothesis.model_validate(hypothesis.model_dump())

    @staticmethod
    def _public_job(job: JobRecord) -> ExperimentJob:
        return ExperimentJob.model_validate(job.model_dump())

    @staticmethod
    def _public_artifact(artifact: ArtifactRecord) -> ResearchArtifact:
        return ResearchArtifact.model_validate(artifact.model_dump())

    @staticmethod
    def _public_event(event: EventRecord) -> ResearchEvent:
        return ResearchEvent.model_validate(event.model_dump())
