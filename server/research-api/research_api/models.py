from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


def to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class APIModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
        allow_inf_nan=False,
    )


TemplateId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")]
VariantId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")]


class RunStatus(str, Enum):
    DRAFT = "draft"
    HYPOTHESIS_READY = "hypothesis_ready"
    QUEUED = "queued"
    RUNNING = "running"
    ANALYSIS_READY = "analysis_ready"
    ANALYZING = "analyzing"
    VERIFICATION_REQUIRED = "verification_required"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchStage(str, Enum):
    TOPIC = "topic"
    HYPOTHESIS = "hypothesis"
    EXPERIMENT = "experiment"
    ANALYSIS = "analysis"
    VERIFICATION = "verification"
    PUBLICATION = "publication"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class ArtifactKind(str, Enum):
    TOPIC_BRIEF = "topic_brief"
    HYPOTHESIS = "hypothesis"
    EXPERIMENT_SPEC = "experiment_spec"
    SOURCE_BUNDLE = "source_bundle"
    EXPERIMENT_LOG = "experiment_log"
    RAW_METRICS = "raw_metrics"
    ANALYSIS = "analysis"
    VERIFICATION_BUNDLE = "verification_bundle"
    RESEARCH_CARD = "research_card"


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    PASSED = "passed"
    FAILED = "failed"


class Visibility(str, Enum):
    PRIVATE = "private"
    PUBLIC = "public"


class HypothesisStatus(str, Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ExpectedDirection(str, Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    NO_CHANGE = "no_change"


class Actor(str, Enum):
    USER = "user"
    RUNNER = "runner"
    VERIFIER = "verifier"
    SYSTEM = "system"


class Budget(APIModel):
    cpu_seconds_per_job: Literal[120] = 120
    memory_mb_per_job: Literal[1024] = 1024
    max_concurrent_jobs: Literal[2] = 2
    max_attempts_per_variant: Literal[2] = 2


class ResourceLimits(APIModel):
    cpu_count: Literal[1] = 1
    memory_mb: Literal[1024] = 1024
    timeout_seconds: Literal[120] = 120
    network: Literal[False] = False


class ExperimentConfig(APIModel):
    benchmark_version: str = Field(min_length=1, max_length=100)
    method: VariantId
    parameters: dict[str, Any] = Field(default_factory=dict)


class CreateRunRequest(APIModel):
    topic: str = Field(min_length=3, max_length=200)
    template_id: TemplateId


class BenchmarkInput(APIModel):
    include_all_variants: Literal[True]


class ExperimentSuiteRequest(APIModel):
    expected_version: int = Field(ge=1)
    benchmark: BenchmarkInput


class HypothesisInput(APIModel):
    expected_version: int = Field(ge=1)
    statement: str = Field(min_length=10, max_length=1000)
    independent_variable: str = Field(min_length=1, max_length=100)
    dependent_metric: str = Field(min_length=1, max_length=100)
    expected_direction: ExpectedDirection
    rationale: str = Field(min_length=10, max_length=2000)


class VersionedCommand(APIModel):
    expected_version: int = Field(ge=1)


class RunFailure(APIModel):
    code: str
    message: str


class ResearchRun(APIModel):
    id: UUID
    topic: str
    research_question: str
    engine: Literal["minimal"] = "minimal"
    template_id: TemplateId
    template_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    status: RunStatus
    current_stage: ResearchStage
    version: int = Field(ge=1)
    budget: Budget = Field(default_factory=Budget)
    retry_count: int = Field(default=0, ge=0)
    failure: RunFailure | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    cancelled_at: datetime | None = None


class RunRecord(ResearchRun):
    user_id: UUID = Field(exclude=True)
    idempotency_key: str = Field(exclude=True)
    create_request_sha256: str = Field(exclude=True)


class ResearchHypothesis(APIModel):
    id: UUID
    run_id: UUID
    statement: str
    independent_variable: str
    dependent_metric: str
    expected_direction: ExpectedDirection
    rationale: str
    status: HypothesisStatus
    created_at: datetime
    updated_at: datetime
    accepted_at: datetime | None = None


class HypothesisRecord(ResearchHypothesis):
    user_id: UUID = Field(exclude=True)


class CaseMetric(APIModel):
    case_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    primary_score: float = Field(ge=0, le=1)
    secondary: dict[str, float] = Field(default_factory=dict)
    latency_ms: float = Field(ge=0)
    evidence: list[str] = Field(default_factory=list, max_length=12)


class BootstrapMetric(APIModel):
    replicate: int = Field(ge=0, lt=1000)
    primary_score: float = Field(ge=0, le=1)
    sample_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AggregateMetric(APIModel):
    primary_score_mean: float = Field(ge=0, le=1)
    primary_score_sample_std: float = Field(ge=0)
    secondary_means: dict[str, float] = Field(default_factory=dict)
    case_latency_ms_mean: float = Field(ge=0)
    case_latency_ms_p95: float = Field(ge=0)
    case_count: int = Field(ge=2, le=2000)


class ExperimentMetrics(APIModel):
    case_results: list[CaseMetric] = Field(min_length=2, max_length=2000)
    bootstrap_results: list[BootstrapMetric] = Field(min_length=1000, max_length=1000)
    aggregate: AggregateMetric
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExperimentJob(APIModel):
    id: UUID
    run_id: UUID
    hypothesis_id: UUID
    suite_id: UUID
    variant: VariantId
    template_id: TemplateId
    template_version: str
    attempt: int = Field(ge=1, le=2)
    status: JobStatus
    bootstrap_seed: Literal[20260711] = 20260711
    bootstrap_replicates: Literal[1000] = 1000
    config: ExperimentConfig
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    metrics: ExperimentMetrics | None = None
    error: RunFailure | None = None
    queued_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = Field(default=None, ge=0)


class JobRecord(ExperimentJob):
    user_id: UUID = Field(exclude=True)


class ResearchArtifact(APIModel):
    id: UUID
    run_id: UUID
    job_id: UUID | None = None
    stage: ResearchStage
    kind: ArtifactKind
    version: int = Field(ge=1)
    content: dict[str, Any] | None = None
    filename: str | None = None
    mime_type: str | None = None
    byte_size: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    verification_status: VerificationStatus
    visibility: Visibility
    download_url: str | None = None
    download_url_expires_at: datetime | None = None
    created_at: datetime
    published_at: datetime | None = None


class ArtifactRecord(ResearchArtifact):
    user_id: UUID = Field(exclude=True)
    storage_bucket: str | None = Field(default=None, exclude=True)
    storage_path: str | None = Field(default=None, exclude=True)


class ResearchEvent(APIModel):
    id: str = Field(pattern=r"^[0-9]+$")
    run_id: UUID
    actor: Actor
    event_type: str = Field(min_length=3, max_length=100)
    stage: ResearchStage
    payload: dict[str, Any]
    created_at: datetime


class EventRecord(ResearchEvent):
    user_id: UUID = Field(exclude=True)
    command_name: str | None = Field(default=None, exclude=True)
    idempotency_key: str | None = Field(default=None, exclude=True)
    request_sha256: str | None = Field(default=None, exclude=True)


class RunSnapshot(APIModel):
    run: ResearchRun
    hypothesis: ResearchHypothesis | None
    jobs: list[ExperimentJob]
    artifacts: list[ResearchArtifact]
    last_event_id: str | None


class ListRunsResponse(APIModel):
    items: list[ResearchRun]
    next_cursor: str | None = None


class ExperimentSuiteAccepted(APIModel):
    run: ResearchRun
    suite_id: UUID
    jobs: list[ExperimentJob] = Field(min_length=3, max_length=4)
    event_id: str


class CommandAccepted(APIModel):
    run: ResearchRun
    event_id: str


class ArtifactList(APIModel):
    items: list[ResearchArtifact]


class ArtifactCommandResult(APIModel):
    artifact: ResearchArtifact
    run_version: int
    event_id: str


class ResearchVariant(APIModel):
    id: VariantId
    label: str
    pipeline: list[str] = Field(min_length=1, max_length=8)


class ResearchTemplate(APIModel):
    id: TemplateId
    capstone_number: int = Field(ge=1, le=17)
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    name: str
    name_zh: str
    research_question: str
    dataset: str
    benchmark_kind: Literal["public_snapshot", "packaged_dataset", "deterministic_fixture"]
    provenance: list[str] = Field(min_length=1)
    independent_variable: str
    primary_metric: str
    primary_metric_label: str
    expected_direction: ExpectedDirection = ExpectedDirection.INCREASE
    effect_threshold: float = Field(ge=0, le=1)
    variants: list[ResearchVariant] = Field(min_length=3, max_length=4)
    bootstrap_seed: Literal[20260711] = 20260711
    bootstrap_replicates: Literal[1000] = 1000
    secondary_metrics: list[str]
    limitations: list[str] = Field(min_length=1)
    external_validation: list[str] = Field(default_factory=list)
    zero_api_cost: Literal[True] = True
    runtime_mode: Literal["offline"] = "offline"
    budget: Budget = Field(default_factory=Budget)


class TemplateList(APIModel):
    items: list[ResearchTemplate]


class HealthResponse(APIModel):
    status: Literal["ok"] = "ok"
    service: Literal["research-api"] = "research-api"


class ErrorBody(APIModel):
    code: str
    message: str
    request_id: UUID
    details: dict[str, Any] | None = None


class ErrorResponse(APIModel):
    error: ErrorBody
