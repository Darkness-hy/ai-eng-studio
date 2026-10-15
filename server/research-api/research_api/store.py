from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import UUID, uuid4

from .errors import DomainError, idempotency_conflict, invalid_state, not_found, version_conflict
from .models import (
    Actor,
    ArtifactCommandResult,
    ArtifactKind,
    ArtifactList,
    ArtifactRecord,
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
    RunFailure,
    RunRecord,
    RunSnapshot,
    RunStatus,
    VerificationStatus,
    Visibility,
)
from .template import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    canonical_json_sha256,
    experiment_config,
    source_bundle_sha256,
    template_variants,
)
from .template_catalog import get_template


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ResearchStore(Protocol):
    async def create_run(
        self, user_id: UUID, key: str, fingerprint: str, request: CreateRunRequest
    ) -> RunSnapshot: ...

    async def list_runs(
        self, user_id: UUID, status: RunStatus | None = None, limit: int = 20
    ) -> ListRunsResponse: ...

    async def get_snapshot(self, user_id: UUID, run_id: UUID) -> RunSnapshot: ...

    async def put_hypothesis(self, user_id: UUID, run_id: UUID, request: HypothesisInput) -> RunSnapshot: ...

    async def enqueue_suite(
        self,
        user_id: UUID,
        run_id: UUID,
        key: str,
        fingerprint: str,
        expected_version: int,
    ) -> ExperimentSuiteAccepted: ...

    async def mark_job_running(self, job_id: UUID) -> JobRecord | None: ...

    async def list_queued_jobs(self, limit: int = 20) -> list[JobRecord]: ...

    async def list_stale_running_job_ids(self, older_than_seconds: int) -> list[UUID]: ...

    async def finish_job(
        self,
        job_id: UUID,
        metrics: ExperimentMetrics | None,
        failure: RunFailure | None,
        timed_out: bool = False,
    ) -> None: ...

    async def begin_analysis(
        self, user_id: UUID, run_id: UUID, key: str, fingerprint: str, expected_version: int
    ) -> CommandAccepted: ...

    async def analysis_inputs(
        self, run_id: UUID
    ) -> tuple[ResearchRun, ExpectedDirection, dict[str, ExperimentMetrics]]: ...

    async def complete_analysis(self, run_id: UUID, content: dict[str, Any]) -> None: ...

    async def begin_verification(
        self, user_id: UUID, run_id: UUID, key: str, fingerprint: str, expected_version: int
    ) -> CommandAccepted: ...

    async def verification_inputs(
        self, run_id: UUID
    ) -> tuple[
        ResearchRun,
        ResearchHypothesis,
        dict[str, ExperimentMetrics],
        dict[str, Any],
    ]: ...

    async def complete_verification(
        self,
        run_id: UUID,
        bundle: dict[str, Any],
        research_card: dict[str, Any] | None,
        passed: bool,
    ) -> None: ...

    async def fail_run(self, run_id: UUID, code: str, message: str, stage: ResearchStage) -> None: ...

    async def retry_run(
        self, user_id: UUID, run_id: UUID, key: str, fingerprint: str, expected_version: int
    ) -> tuple[CommandAccepted, list[JobRecord]]: ...

    async def cancel_run(
        self, user_id: UUID, run_id: UUID, key: str, fingerprint: str, expected_version: int
    ) -> CommandAccepted: ...

    async def list_events(self, user_id: UUID, run_id: UUID, after: int = 0) -> list[ResearchEvent]: ...

    async def list_artifacts(
        self, user_id: UUID, run_id: UUID, kind: ArtifactKind | None = None
    ) -> ArtifactList: ...

    async def publish_artifact(
        self,
        user_id: UUID,
        artifact_id: UUID,
        key: str,
        fingerprint: str,
        expected_version: int,
    ) -> ArtifactCommandResult: ...

    async def unpublish_artifact(
        self,
        user_id: UUID,
        artifact_id: UUID,
        key: str,
        fingerprint: str,
        expected_version: int,
    ) -> ArtifactCommandResult: ...

    async def close(self) -> None: ...


class MemoryResearchStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._runs: dict[UUID, RunRecord] = {}
        self._hypotheses: dict[UUID, HypothesisRecord] = {}
        self._jobs: dict[UUID, JobRecord] = {}
        self._artifacts: dict[UUID, ArtifactRecord] = {}
        self._events: list[EventRecord] = []
        self._event_id = 0
        self._create_keys: dict[tuple[UUID, str], UUID] = {}
        self._command_keys: dict[tuple[UUID, str, str], EventRecord] = {}

    async def close(self) -> None:
        return None

    async def create_run(
        self, user_id: UUID, key: str, fingerprint: str, request: CreateRunRequest
    ) -> RunSnapshot:
        async with self._lock:
            existing_id = self._create_keys.get((user_id, key))
            if existing_id:
                existing = self._runs[existing_id]
                if existing.create_request_sha256 != fingerprint:
                    raise idempotency_conflict()
                return self._snapshot(existing)

            try:
                template = get_template(request.template_id)
            except ValueError as exc:
                raise DomainError("template_not_found", "The research template was not found", 404) from exc
            now = utcnow()
            run = RunRecord(
                id=uuid4(),
                user_id=user_id,
                idempotency_key=key,
                create_request_sha256=fingerprint,
                topic=request.topic.strip(),
                research_question=template.research_question,
                template_id=template.id,
                template_version=template.version,
                status=RunStatus.DRAFT,
                current_stage=ResearchStage.HYPOTHESIS,
                version=1,
                created_at=now,
                updated_at=now,
            )
            self._runs[run.id] = run
            self._create_keys[(user_id, key)] = run.id
            self._create_artifact(
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
            self._append_event(
                run,
                Actor.USER,
                "run.created",
                ResearchStage.TOPIC,
                {"runVersion": run.version, "templateId": template.id},
            )
            return self._snapshot(run)

    async def list_runs(
        self, user_id: UUID, status: RunStatus | None = None, limit: int = 20
    ) -> ListRunsResponse:
        async with self._lock:
            rows = [
                run
                for run in self._runs.values()
                if run.user_id == user_id and (status is None or run.status == status)
            ]
            rows.sort(key=lambda item: (item.created_at, str(item.id)), reverse=True)
            return ListRunsResponse(
                items=[ResearchRun.model_validate(item.model_dump()) for item in rows[:limit]],
                next_cursor=None,
            )

    async def get_snapshot(self, user_id: UUID, run_id: UUID) -> RunSnapshot:
        async with self._lock:
            return self._snapshot(self._owned_run(user_id, run_id))

    async def put_hypothesis(self, user_id: UUID, run_id: UUID, request: HypothesisInput) -> RunSnapshot:
        async with self._lock:
            run = self._owned_run(user_id, run_id)
            self._check_version(run, request.expected_version)
            if run.status != RunStatus.DRAFT:
                raise invalid_state("A hypothesis can only be accepted from a draft run")
            template = get_template(run.template_id)
            if request.independent_variable != template.independent_variable:
                raise invalid_state("The independent variable does not match the selected template")
            if request.dependent_metric != template.primary_metric:
                raise invalid_state("The dependent metric does not match the selected template")
            now = utcnow()
            hypothesis = HypothesisRecord(
                id=uuid4(),
                run_id=run.id,
                user_id=user_id,
                statement=request.statement.strip(),
                independent_variable=request.independent_variable,
                dependent_metric=request.dependent_metric,
                expected_direction=request.expected_direction,
                rationale=request.rationale.strip(),
                status=HypothesisStatus.ACCEPTED,
                created_at=now,
                updated_at=now,
                accepted_at=now,
            )
            self._hypotheses[run.id] = hypothesis
            run.status = RunStatus.HYPOTHESIS_READY
            run.current_stage = ResearchStage.EXPERIMENT
            self._bump(run)
            self._create_artifact(
                run,
                ResearchStage.HYPOTHESIS,
                ArtifactKind.HYPOTHESIS,
                hypothesis.model_dump(
                    mode="json",
                    by_alias=True,
                    exclude={"id", "run_id", "created_at", "updated_at", "accepted_at"},
                ),
            )
            self._append_event(
                run,
                Actor.USER,
                "hypothesis.accepted",
                ResearchStage.HYPOTHESIS,
                {"hypothesisId": str(hypothesis.id), "runVersion": run.version},
            )
            return self._snapshot(run)

    async def enqueue_suite(
        self,
        user_id: UUID,
        run_id: UUID,
        key: str,
        fingerprint: str,
        expected_version: int,
    ) -> ExperimentSuiteAccepted:
        async with self._lock:
            replay = self._command_replay(user_id, "experiment.enqueue", key, fingerprint)
            if replay:
                jobs = [self._jobs[UUID(item)] for item in replay.payload["jobIds"]]
                return ExperimentSuiteAccepted(
                    run=self._public_run(self._owned_run(user_id, run_id)),
                    suite_id=UUID(replay.payload["suiteId"]),
                    jobs=[self._public_job(item) for item in jobs],
                    event_id=replay.id,
                )

            run = self._owned_run(user_id, run_id)
            self._check_version(run, expected_version)
            if run.status != RunStatus.HYPOTHESIS_READY:
                raise invalid_state("The run is not ready to enqueue experiments")
            hypothesis = self._hypotheses.get(run.id)
            if not hypothesis or hypothesis.status != HypothesisStatus.ACCEPTED:
                raise invalid_state("An accepted hypothesis is required")
            suite_id = uuid4()
            now = utcnow()
            variants = template_variants(run.template_id)
            jobs = [
                JobRecord(
                    id=uuid4(),
                    run_id=run.id,
                    user_id=user_id,
                    hypothesis_id=hypothesis.id,
                    suite_id=suite_id,
                    variant=variant,
                    template_id=run.template_id,
                    template_version=run.template_version,
                    attempt=1,
                    status=JobStatus.QUEUED,
                    bootstrap_seed=BOOTSTRAP_SEED,
                    bootstrap_replicates=BOOTSTRAP_REPLICATES,
                    config=experiment_config(run.template_id, variant),
                    metrics=None,
                    error=None,
                    queued_at=now,
                )
                for variant in variants
            ]
            for job in jobs:
                self._jobs[job.id] = job
            run.status = RunStatus.QUEUED
            run.current_stage = ResearchStage.EXPERIMENT
            self._bump(run)
            spec = {
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
            }
            self._create_artifact(run, ResearchStage.EXPERIMENT, ArtifactKind.EXPERIMENT_SPEC, spec)
            self._create_artifact(
                run,
                ResearchStage.EXPERIMENT,
                ArtifactKind.SOURCE_BUNDLE,
                {
                    "sha256": source_bundle_sha256(run.template_id),
                    "templateVersion": run.template_version,
                },
            )
            event = self._append_event(
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
                jobs=[self._public_job(item) for item in jobs],
                event_id=event.id,
            )

    async def mark_job_running(self, job_id: UUID) -> JobRecord | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status != JobStatus.QUEUED:
                return None
            run = self._runs[job.run_id]
            if run.status in {RunStatus.CANCELLED, RunStatus.FAILED}:
                return None
            job.status = JobStatus.RUNNING
            job.started_at = utcnow()
            if run.status == RunStatus.QUEUED:
                run.status = RunStatus.RUNNING
                self._bump(run)
            self._append_event(
                run,
                Actor.RUNNER,
                "experiment.job_started",
                ResearchStage.EXPERIMENT,
                {"jobId": str(job.id), "variant": job.variant, "runVersion": run.version},
            )
            return job.model_copy(deep=True)

    async def list_queued_jobs(self, limit: int = 20) -> list[JobRecord]:
        async with self._lock:
            rows = [job for job in self._jobs.values() if job.status == JobStatus.QUEUED]
            rows.sort(key=lambda item: (item.queued_at, str(item.id)))
            return [item.model_copy(deep=True) for item in rows[:limit]]

    async def list_stale_running_job_ids(self, older_than_seconds: int) -> list[UUID]:
        async with self._lock:
            cutoff = utcnow().timestamp() - older_than_seconds
            return [
                job.id
                for job in self._jobs.values()
                if job.status == JobStatus.RUNNING
                and job.started_at is not None
                and job.started_at.timestamp() < cutoff
            ]

    async def finish_job(
        self,
        job_id: UUID,
        metrics: ExperimentMetrics | None,
        failure: RunFailure | None,
        timed_out: bool = False,
    ) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status != JobStatus.RUNNING:
                return
            run = self._runs[job.run_id]
            now = utcnow()
            job.finished_at = now
            job.duration_ms = max(0, int((now - (job.started_at or now)).total_seconds() * 1000))
            job.metrics = metrics
            job.error = failure
            job.status = (
                JobStatus.TIMED_OUT if timed_out else JobStatus.FAILED if failure else JobStatus.SUCCEEDED
            )
            self._append_event(
                run,
                Actor.RUNNER,
                f"experiment.job_{job.status.value}",
                ResearchStage.EXPERIMENT,
                {"jobId": str(job.id), "variant": job.variant, "attempt": job.attempt},
            )
            if failure:
                if run.status != RunStatus.FAILED:
                    run.status = RunStatus.FAILED
                    run.current_stage = ResearchStage.EXPERIMENT
                    run.failure = RunFailure(code="experiment_failed", message="The fixed experiment failed")
                    self._bump(run)
                    self._append_event(
                        run,
                        Actor.RUNNER,
                        "run.failed",
                        ResearchStage.EXPERIMENT,
                        {"code": "experiment_failed", "runVersion": run.version},
                    )
                return

            latest = self._latest_jobs(run.id)
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
                self._create_artifact(
                    run,
                    ResearchStage.EXPERIMENT,
                    ArtifactKind.RAW_METRICS,
                    raw,
                )
                run.status = RunStatus.ANALYSIS_READY
                run.current_stage = ResearchStage.ANALYSIS
                run.failure = None
                self._bump(run)
                self._append_event(
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
        async with self._lock:
            run = self._runs.get(run_id)
            if not run or run.status != RunStatus.ANALYZING:
                raise invalid_state("The run is not analyzing")
            hypothesis = self._hypotheses[run.id]
            return (
                self._public_run(run),
                hypothesis.expected_direction,
                self._latest_metrics(run.id),
            )

    async def complete_analysis(self, run_id: UUID, content: dict[str, Any]) -> None:
        async with self._lock:
            run = self._runs.get(run_id)
            if not run or run.status != RunStatus.ANALYZING:
                return
            self._create_artifact(run, ResearchStage.ANALYSIS, ArtifactKind.ANALYSIS, content)
            run.status = RunStatus.VERIFICATION_REQUIRED
            run.current_stage = ResearchStage.VERIFICATION
            self._bump(run)
            self._append_event(
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
        async with self._lock:
            run = self._runs.get(run_id)
            if not run or run.status != RunStatus.VERIFYING:
                raise invalid_state("The run is not verifying")
            hypothesis = self._hypotheses[run.id]
            analysis = self._latest_artifact(run.id, ArtifactKind.ANALYSIS)
            if not analysis or not analysis.content:
                raise invalid_state("The deterministic analysis artifact is missing")
            return (
                self._public_run(run),
                ResearchHypothesis.model_validate(hypothesis.model_dump()),
                self._latest_metrics(run.id),
                dict(analysis.content),
            )

    async def complete_verification(
        self,
        run_id: UUID,
        bundle: dict[str, Any],
        research_card: dict[str, Any] | None,
        passed: bool,
    ) -> None:
        async with self._lock:
            run = self._runs.get(run_id)
            if not run or run.status != RunStatus.VERIFYING:
                return
            verification = VerificationStatus.PASSED if passed else VerificationStatus.FAILED
            self._create_artifact(
                run,
                ResearchStage.VERIFICATION,
                ArtifactKind.VERIFICATION_BUNDLE,
                bundle,
                verification_status=verification,
            )
            if passed and research_card:
                self._create_artifact(
                    run,
                    ResearchStage.PUBLICATION,
                    ArtifactKind.RESEARCH_CARD,
                    research_card,
                    verification_status=VerificationStatus.PASSED,
                )
            if passed:
                run.status = RunStatus.COMPLETED
                run.current_stage = ResearchStage.PUBLICATION
                run.completed_at = utcnow()
                run.failure = None
                event_type = "verification.passed"
            else:
                run.status = RunStatus.FAILED
                run.current_stage = ResearchStage.VERIFICATION
                run.failure = RunFailure(
                    code="verification_failed", message="The reproduction check did not pass"
                )
                event_type = "verification.failed"
            self._bump(run)
            self._append_event(
                run,
                Actor.VERIFIER,
                event_type,
                ResearchStage.VERIFICATION,
                {"passed": passed, "runVersion": run.version},
            )

    async def fail_run(self, run_id: UUID, code: str, message: str, stage: ResearchStage) -> None:
        async with self._lock:
            run = self._runs.get(run_id)
            if not run or run.status in {RunStatus.COMPLETED, RunStatus.CANCELLED}:
                return
            run.status = RunStatus.FAILED
            run.current_stage = stage
            run.failure = RunFailure(code=code, message=message)
            self._bump(run)
            self._append_event(
                run,
                Actor.SYSTEM,
                "run.failed",
                stage,
                {"code": code, "runVersion": run.version},
            )

    async def retry_run(
        self, user_id: UUID, run_id: UUID, key: str, fingerprint: str, expected_version: int
    ) -> tuple[CommandAccepted, list[JobRecord]]:
        async with self._lock:
            replay = self._command_replay(user_id, "run.retry", key, fingerprint)
            if replay:
                run = self._owned_run(user_id, run_id)
                jobs = [self._jobs[UUID(item)] for item in replay.payload.get("jobIds", [])]
                return CommandAccepted(run=self._public_run(run), event_id=replay.id), [
                    item.model_copy(deep=True) for item in jobs
                ]
            run = self._owned_run(user_id, run_id)
            self._check_version(run, expected_version)
            if run.status != RunStatus.FAILED or not run.failure:
                raise invalid_state("Only a failed run can be retried")
            retry_jobs: list[JobRecord] = []
            if run.failure.code == "experiment_failed":
                latest = self._latest_jobs(run.id)
                if any(job.status == JobStatus.RUNNING for job in latest.values()):
                    raise invalid_state("Wait for active sibling jobs to settle before retrying")
                failed = [
                    job for job in latest.values() if job.status in {JobStatus.FAILED, JobStatus.TIMED_OUT}
                ]
                if not failed or any(job.attempt >= 2 for job in failed):
                    raise DomainError("budget_exceeded", "The retry budget is exhausted", 429)
                suite_id = uuid4()
                now = utcnow()
                for previous in failed:
                    job = JobRecord(
                        id=uuid4(),
                        run_id=run.id,
                        user_id=user_id,
                        hypothesis_id=previous.hypothesis_id,
                        suite_id=suite_id,
                        variant=previous.variant,
                        template_id=run.template_id,
                        template_version=run.template_version,
                        attempt=previous.attempt + 1,
                        status=JobStatus.QUEUED,
                        bootstrap_seed=BOOTSTRAP_SEED,
                        bootstrap_replicates=BOOTSTRAP_REPLICATES,
                        config=experiment_config(run.template_id, previous.variant),
                        metrics=None,
                        error=None,
                        queued_at=now,
                    )
                    self._jobs[job.id] = job
                    retry_jobs.append(job)
                retry_jobs.extend(job for job in latest.values() if job.status == JobStatus.QUEUED)
                run.status = RunStatus.QUEUED
                run.current_stage = ResearchStage.EXPERIMENT
            elif run.failure.code == "verification_failed":
                run.status = RunStatus.VERIFICATION_REQUIRED
                run.current_stage = ResearchStage.VERIFICATION
            elif run.failure.code == "analysis_failed":
                run.status = RunStatus.ANALYSIS_READY
                run.current_stage = ResearchStage.ANALYSIS
            else:
                raise invalid_state("This failure cannot be retried")
            run.retry_count += 1
            run.failure = None
            self._bump(run)
            event = self._append_event(
                run,
                Actor.USER,
                "run.retried",
                run.current_stage,
                {"jobIds": [str(item.id) for item in retry_jobs], "runVersion": run.version},
                command_name="run.retry",
                idempotency_key=key,
                request_sha256=fingerprint,
            )
            return CommandAccepted(run=self._public_run(run), event_id=event.id), [
                item.model_copy(deep=True) for item in retry_jobs
            ]

    async def cancel_run(
        self, user_id: UUID, run_id: UUID, key: str, fingerprint: str, expected_version: int
    ) -> CommandAccepted:
        async with self._lock:
            replay = self._command_replay(user_id, "run.cancel", key, fingerprint)
            if replay:
                return CommandAccepted(
                    run=self._public_run(self._owned_run(user_id, run_id)), event_id=replay.id
                )
            run = self._owned_run(user_id, run_id)
            self._check_version(run, expected_version)
            if run.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
                raise invalid_state("A terminal run cannot be cancelled")
            now = utcnow()
            for job in self._jobs_for_run(run.id):
                if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                    job.status = JobStatus.CANCELLED
                    job.finished_at = now
                    job.error = RunFailure(code="cancelled", message="The run was cancelled")
            run.status = RunStatus.CANCELLED
            run.cancelled_at = now
            self._bump(run)
            event = self._append_event(
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
        async with self._lock:
            self._owned_run(user_id, run_id)
            return [
                ResearchEvent.model_validate(item.model_dump())
                for item in self._events
                if item.run_id == run_id and int(item.id) > after
            ]

    async def list_artifacts(
        self, user_id: UUID, run_id: UUID, kind: ArtifactKind | None = None
    ) -> ArtifactList:
        async with self._lock:
            self._owned_run(user_id, run_id)
            rows = [item for item in self._artifacts_for_run(run_id) if kind is None or item.kind == kind]
            return ArtifactList(items=[self._public_artifact(item) for item in rows])

    async def publish_artifact(
        self,
        user_id: UUID,
        artifact_id: UUID,
        key: str,
        fingerprint: str,
        expected_version: int,
    ) -> ArtifactCommandResult:
        return await self._set_publication(
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
        return await self._set_publication(
            user_id, artifact_id, key, fingerprint, expected_version, publish=False
        )

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
        async with self._lock:
            replay = self._command_replay(user_id, command_name, key, fingerprint)
            if replay:
                return CommandAccepted(
                    run=self._public_run(self._owned_run(user_id, run_id)), event_id=replay.id
                )
            run = self._owned_run(user_id, run_id)
            self._check_version(run, expected_version)
            if run.status != required_status:
                raise invalid_state(f"Command requires run state {required_status.value}")
            run.status = next_status
            run.current_stage = stage
            self._bump(run)
            event = self._append_event(
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

    async def _set_publication(
        self,
        user_id: UUID,
        artifact_id: UUID,
        key: str,
        fingerprint: str,
        expected_version: int,
        publish: bool,
    ) -> ArtifactCommandResult:
        command = "artifact.publish" if publish else "artifact.unpublish"
        async with self._lock:
            replay = self._command_replay(user_id, command, key, fingerprint)
            if replay:
                artifact = self._artifacts.get(artifact_id)
                if not artifact or artifact.user_id != user_id:
                    raise not_found("artifact")
                return ArtifactCommandResult(
                    artifact=self._public_artifact(artifact),
                    run_version=int(replay.payload["runVersion"]),
                    event_id=replay.id,
                )
            artifact = self._artifacts.get(artifact_id)
            if not artifact or artifact.user_id != user_id:
                raise not_found("artifact")
            run = self._owned_run(user_id, artifact.run_id)
            self._check_version(run, expected_version)
            if publish:
                if run.status != RunStatus.COMPLETED:
                    raise invalid_state("Only a completed run can publish artifacts")
                if artifact.visibility != Visibility.PRIVATE:
                    raise invalid_state("The artifact is already public")
                if (
                    artifact.kind not in {ArtifactKind.VERIFICATION_BUNDLE, ArtifactKind.RESEARCH_CARD}
                    or artifact.verification_status != VerificationStatus.PASSED
                    or artifact.content is None
                    or artifact.storage_path is not None
                ):
                    raise invalid_state("The artifact is not publication-safe")
                artifact.visibility = Visibility.PUBLIC
                artifact.published_at = utcnow()
                event_type = "artifact.published"
            else:
                if artifact.visibility != Visibility.PUBLIC:
                    raise invalid_state("The artifact is not public")
                artifact.visibility = Visibility.PRIVATE
                artifact.published_at = None
                event_type = "artifact.unpublished"
            self._bump(run)
            event = self._append_event(
                run,
                Actor.USER,
                event_type,
                ResearchStage.PUBLICATION,
                {
                    "artifactId": str(artifact.id),
                    "visibility": artifact.visibility.value,
                    "runVersion": run.version,
                },
                command_name=command,
                idempotency_key=key,
                request_sha256=fingerprint,
            )
            return ArtifactCommandResult(
                artifact=self._public_artifact(artifact), run_version=run.version, event_id=event.id
            )

    def _owned_run(self, user_id: UUID, run_id: UUID) -> RunRecord:
        run = self._runs.get(run_id)
        if not run or run.user_id != user_id:
            raise not_found()
        return run

    @staticmethod
    def _check_version(run: RunRecord, expected: int) -> None:
        if run.version != expected:
            raise version_conflict(expected, run.version)

    @staticmethod
    def _bump(run: RunRecord) -> None:
        run.version += 1
        run.updated_at = utcnow()

    def _append_event(
        self,
        run: RunRecord,
        actor: Actor,
        event_type: str,
        stage: ResearchStage,
        payload: dict[str, Any],
        command_name: str | None = None,
        idempotency_key: str | None = None,
        request_sha256: str | None = None,
    ) -> EventRecord:
        self._event_id += 1
        event = EventRecord(
            id=str(self._event_id),
            run_id=run.id,
            user_id=run.user_id,
            actor=actor,
            event_type=event_type,
            stage=stage,
            payload=payload,
            created_at=utcnow(),
            command_name=command_name,
            idempotency_key=idempotency_key,
            request_sha256=request_sha256,
        )
        self._events.append(event)
        if command_name and idempotency_key:
            self._command_keys[(run.user_id, command_name, idempotency_key)] = event
        return event

    def _command_replay(self, user_id: UUID, command: str, key: str, fingerprint: str) -> EventRecord | None:
        event = self._command_keys.get((user_id, command, key))
        if event and event.request_sha256 != fingerprint:
            raise idempotency_conflict()
        return event

    def _create_artifact(
        self,
        run: RunRecord,
        stage: ResearchStage,
        kind: ArtifactKind,
        content: dict[str, Any],
        verification_status: VerificationStatus = VerificationStatus.UNVERIFIED,
    ) -> ArtifactRecord:
        version = 1 + max(
            (item.version for item in self._artifacts_for_run(run.id) if item.kind == kind),
            default=0,
        )
        artifact = ArtifactRecord(
            id=uuid4(),
            run_id=run.id,
            user_id=run.user_id,
            stage=stage,
            kind=kind,
            version=version,
            content=content,
            sha256=canonical_json_sha256(content),
            verification_status=verification_status,
            visibility=Visibility.PRIVATE,
            created_at=utcnow(),
        )
        self._artifacts[artifact.id] = artifact
        return artifact

    def _latest_jobs(self, run_id: UUID) -> dict[str, JobRecord]:
        latest: dict[str, JobRecord] = {}
        for job in self._jobs_for_run(run_id):
            current = latest.get(job.variant)
            if current is None or job.attempt > current.attempt:
                latest[job.variant] = job
        return latest

    def _latest_metrics(self, run_id: UUID) -> dict[str, ExperimentMetrics]:
        latest = self._latest_jobs(run_id)
        run = self._runs[run_id]
        variants = template_variants(run.template_id)
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

    def _jobs_for_run(self, run_id: UUID) -> list[JobRecord]:
        return sorted(
            (item for item in self._jobs.values() if item.run_id == run_id),
            key=lambda item: (item.queued_at, str(item.id)),
        )

    def _artifacts_for_run(self, run_id: UUID) -> list[ArtifactRecord]:
        return sorted(
            (item for item in self._artifacts.values() if item.run_id == run_id),
            key=lambda item: (item.created_at, str(item.id)),
        )

    def _latest_artifact(self, run_id: UUID, kind: ArtifactKind) -> ArtifactRecord | None:
        rows = [item for item in self._artifacts_for_run(run_id) if item.kind == kind]
        return max(rows, key=lambda item: item.version) if rows else None

    def _snapshot(self, run: RunRecord) -> RunSnapshot:
        hypothesis = self._hypotheses.get(run.id)
        events = [item for item in self._events if item.run_id == run.id]
        return RunSnapshot(
            run=self._public_run(run),
            hypothesis=ResearchHypothesis.model_validate(hypothesis.model_dump()) if hypothesis else None,
            jobs=[self._public_job(item) for item in self._jobs_for_run(run.id)],
            artifacts=[self._public_artifact(item) for item in self._artifacts_for_run(run.id)],
            last_event_id=events[-1].id if events else None,
        )

    @staticmethod
    def _public_run(run: RunRecord) -> ResearchRun:
        return ResearchRun.model_validate(run.model_dump())

    @staticmethod
    def _public_job(job: JobRecord) -> ExperimentJob:
        return ExperimentJob.model_validate(job.model_dump())

    @staticmethod
    def _public_artifact(artifact: ArtifactRecord) -> ResearchArtifact:
        return ResearchArtifact.model_validate(artifact.model_dump())
