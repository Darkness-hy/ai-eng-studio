from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from .engine import ResearchEngine, RunnerError
from .models import (
    CommandAccepted,
    ExperimentJob,
    ExperimentSuiteAccepted,
    ResearchStage,
    RunFailure,
)
from .store import ResearchStore
from .template import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    analyze_results,
    canonical_json_sha256,
    dependency_lock_sha256,
    experiment_config,
    source_bundle_sha256,
    template_variants,
    verify_reproduction,
)
from .template_catalog import get_template

logger = logging.getLogger("research.service")


class ResearchService:
    def __init__(
        self,
        store: ResearchStore,
        engine: ResearchEngine,
        dispatcher_poll_seconds: float = 1,
        stale_job_seconds: int = 150,
    ) -> None:
        self.store = store
        self.engine = engine
        self.dispatcher_poll_seconds = dispatcher_poll_seconds
        self.stale_job_seconds = stale_job_seconds
        self._tasks: set[asyncio.Task[None]] = set()
        self._tasks_by_run: dict[UUID, set[asyncio.Task[None]]] = {}
        self._task_keys: set[tuple[UUID, str]] = set()
        self._dispatcher: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._dispatcher is None:
            self._dispatcher = asyncio.create_task(self._dispatch_loop())

    def schedule_jobs(self, jobs: list[ExperimentJob]) -> None:
        for job in jobs:
            self._schedule(job.run_id, f"job:{job.id}", lambda job_id=job.id: self._run_job(job_id))

    def schedule_analysis(self, run_id: UUID) -> None:
        self._schedule(run_id, "analysis", lambda: self._analyze(run_id))

    def schedule_verification(self, run_id: UUID) -> None:
        self._schedule(run_id, "verification", lambda: self._verify(run_id))

    async def cancel_background(self, run_id: UUID) -> None:
        tasks = list(self._tasks_by_run.get(run_id, set()))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def wait_for_idle(self, run_id: UUID, timeout: float = 30) -> None:
        async def wait() -> None:
            while self._tasks_by_run.get(run_id):
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait(), timeout=timeout)

    async def close(self) -> None:
        if self._dispatcher:
            self._dispatcher.cancel()
            await asyncio.gather(self._dispatcher, return_exceptions=True)
            self._dispatcher = None
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.store.close()

    def _schedule(
        self,
        run_id: UUID,
        key: str,
        factory: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        task_key = (run_id, key)
        if task_key in self._task_keys:
            return
        self._task_keys.add(task_key)
        task = asyncio.create_task(factory())
        self._tasks.add(task)
        self._tasks_by_run.setdefault(run_id, set()).add(task)

        def done(completed: asyncio.Task[None]) -> None:
            self._tasks.discard(completed)
            self._task_keys.discard(task_key)
            run_tasks = self._tasks_by_run.get(run_id)
            if run_tasks:
                run_tasks.discard(completed)
                if not run_tasks:
                    self._tasks_by_run.pop(run_id, None)
            if not completed.cancelled() and completed.exception():
                exception = completed.exception()
                logger.error(
                    "background task failed",
                    exc_info=(type(exception), exception, exception.__traceback__),
                )

        task.add_done_callback(done)

    async def _dispatch_loop(self) -> None:
        while True:
            try:
                stale_ids = await self.store.list_stale_running_job_ids(self.stale_job_seconds)
                for job_id in stale_ids:
                    await self.store.finish_job(
                        job_id,
                        None,
                        RunFailure(
                            code="experiment_failed",
                            message="The fixed experiment exceeded its execution lease",
                        ),
                        timed_out=True,
                    )
                self.schedule_jobs(await self.store.list_queued_jobs())
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("queue dispatcher iteration failed")
            await asyncio.sleep(self.dispatcher_poll_seconds)

    async def _run_job(self, job_id: UUID) -> None:
        job = await self.store.mark_job_running(job_id)
        if not job:
            return
        try:
            metrics = await self.engine.run_variant(job.template_id, job.variant)
            await self.store.finish_job(job.id, metrics, None)
        except RunnerError as exc:
            await self.store.finish_job(
                job.id,
                None,
                RunFailure(code="experiment_failed", message=exc.safe_message),
                timed_out=exc.code == "experiment_timed_out",
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("unexpected fixed-runner failure: job=%s", job.id)
            await self.store.finish_job(
                job.id,
                None,
                RunFailure(code="experiment_failed", message="The fixed experiment failed"),
            )

    async def _analyze(self, run_id: UUID) -> None:
        try:
            run, direction, metrics = await self.store.analysis_inputs(run_id)
            content = analyze_results(run.template_id, metrics, direction)
            await self.store.complete_analysis(run_id, content)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("analysis failed: run=%s", run_id)
            await self.store.fail_run(
                run_id,
                "analysis_failed",
                "The deterministic analysis failed",
                ResearchStage.ANALYSIS,
            )

    async def _verify(self, run_id: UUID) -> None:
        try:
            run, hypothesis, recorded, analysis = await self.store.verification_inputs(run_id)
            template = get_template(run.template_id)
            variants = template_variants(run.template_id)
            reproduced = await self.engine.run_suite(run.template_id)
            reproduction = verify_reproduction(run.template_id, recorded, reproduced)
            recomputed = analyze_results(
                run.template_id, recorded, hypothesis.expected_direction
            )
            analysis_hash = canonical_json_sha256(analysis)
            recomputed_hash = canonical_json_sha256(recomputed)
            integrity_passed = analysis_hash == recomputed_hash
            passed = integrity_passed and reproduction["passed"]
            raw_metrics = {
                variant: metrics.model_dump(mode="json", by_alias=True)
                for variant, metrics in recorded.items()
            }
            verified_at = datetime.now(timezone.utc).isoformat()
            bundle = {
                "researchQuestion": run.research_question,
                "hypothesis": hypothesis.model_dump(
                    mode="json",
                    by_alias=True,
                    exclude={"id", "run_id", "created_at", "updated_at", "accepted_at"},
                ),
                "templateId": template.id,
                "templateVersion": template.version,
                "sourceBundleSha256": source_bundle_sha256(template.id),
                "dependencyLockSha256": dependency_lock_sha256(),
                "runtime": self.engine.runtime_fingerprint(),
                "bootstrapSeed": BOOTSTRAP_SEED,
                "bootstrapReplicates": BOOTSTRAP_REPLICATES,
                "configs": {
                    variant: experiment_config(template.id, variant).model_dump(
                        mode="json", by_alias=True
                    )
                    for variant in variants
                },
                "rawMetricsSha256": canonical_json_sha256(raw_metrics),
                "analysisSha256": analysis_hash,
                "integrity": {
                    "passed": integrity_passed,
                    "recomputedAnalysisSha256": recomputed_hash,
                },
                "reproduction": reproduction,
                "claimVerdict": analysis["verdict"],
                "limitations": analysis["limitations"],
                "externalValidation": analysis["externalValidation"],
                "zeroApiCost": True,
                "verifiedAt": verified_at,
                "verifierVersion": "zero-api-capstone-verifier/3.0.0",
            }
            research_card = None
            if passed:
                research_card = {
                    "title": run.topic,
                    "researchQuestion": run.research_question,
                    "hypothesis": hypothesis.statement,
                    "verdict": analysis["verdict"],
                    "primaryMetric": analysis["primaryMetric"],
                    "primaryMetricLabel": analysis["primaryMetricLabel"],
                    "bestVariant": analysis["bestVariant"],
                    "baselineVariant": analysis["baselineVariant"],
                    "candidateVariant": analysis["candidateVariant"],
                    "baselinePrimaryScore": analysis["baselinePrimaryScore"],
                    "candidatePrimaryScore": analysis["candidatePrimaryScore"],
                    "meanPrimaryMetricDelta": analysis["meanPrimaryMetricDelta"],
                    "confidenceInterval95": analysis["confidenceInterval95"],
                    "directionAgreementCount": analysis["directionAgreementCount"],
                    "bootstrapCount": BOOTSTRAP_REPLICATES,
                    "templateId": template.id,
                    "templateVersion": template.version,
                    "verifiedAt": verified_at,
                    "limitations": analysis["limitations"],
                    "externalValidation": analysis["externalValidation"],
                    "zeroApiCost": True,
                }
            await self.store.complete_verification(run_id, bundle, research_card, passed)
        except asyncio.CancelledError:
            raise
        except RunnerError:
            logger.exception("verification rerun failed: run=%s", run_id)
            await self.store.fail_run(
                run_id,
                "verification_failed",
                "The reproduction runner failed",
                ResearchStage.VERIFICATION,
            )
        except Exception:
            logger.exception("verification failed: run=%s", run_id)
            await self.store.fail_run(
                run_id,
                "verification_failed",
                "The verification step failed",
                ResearchStage.VERIFICATION,
            )


def accepted_jobs(result: ExperimentSuiteAccepted) -> list[ExperimentJob]:
    return result.jobs


def retry_jobs(result: tuple[CommandAccepted, list[ExperimentJob]]) -> list[ExperimentJob]:
    return result[1]
