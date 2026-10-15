from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import sklearn

from .code_rag_benchmark import DATA_PATH, PROVENANCE, STATISTICS, benchmark_sha256
from .code_rag_experiment import run_code_rag_variant
from .models import (
    AggregateMetric,
    BootstrapMetric,
    CaseMetric,
    ExpectedDirection,
    ExperimentConfig,
    ExperimentMetrics,
    ResearchTemplate,
)
from .portfolio_benchmarks import RawCaseResult, run_portfolio_variant
from .template_catalog import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    REPRODUCTION_TOLERANCE,
    get_template,
    list_templates,
)
from .template_catalog import (
    TEMPLATE_VERSION as CATALOG_TEMPLATE_VERSION,
)

TEMPLATE_ID = "codebase_rag_retrieval_v1"
TEMPLATE_VERSION = CATALOG_TEMPLATE_VERSION
CANONICAL_QUESTION = get_template(TEMPLATE_ID).research_question
VARIANTS = tuple(variant.id for variant in get_template(TEMPLATE_ID).variants)


def template_definition(template_id: str = TEMPLATE_ID) -> ResearchTemplate:
    return get_template(template_id)


def template_definitions() -> list[ResearchTemplate]:
    return list_templates()


def template_variants(template_id: str) -> tuple[str, ...]:
    return tuple(variant.id for variant in get_template(template_id).variants)


def experiment_config(template_id: str, variant: str) -> ExperimentConfig:
    template = get_template(template_id)
    descriptor = next((item for item in template.variants if item.id == variant), None)
    if descriptor is None:
        raise ValueError(f"unsupported variant for {template_id}: {variant}")
    return ExperimentConfig(
        benchmark_version=template.dataset,
        method=variant,
        parameters={
            "capstoneNumber": template.capstone_number,
            "offline": True,
            "pipeline": descriptor.pipeline,
        },
    )


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_raw_variant(template_id: str, variant: str) -> list[RawCaseResult]:
    if template_id == TEMPLATE_ID:
        return run_code_rag_variant(variant)
    return run_portfolio_variant(template_id, variant)


def _bootstrap(case_results: list[CaseMetric]) -> list[BootstrapMetric]:
    generator = random.Random(BOOTSTRAP_SEED)
    output: list[BootstrapMetric] = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        indices = [generator.randrange(len(case_results)) for _ in case_results]
        output.append(
            BootstrapMetric(
                replicate=replicate,
                primary_score=statistics.fmean(case_results[index].primary_score for index in indices),
                sample_sha256=hashlib.sha256(
                    json.dumps(indices, separators=(",", ":")).encode()
                ).hexdigest(),
            )
        )
    return output


def run_fixed_variant(template_id: str, variant: str) -> ExperimentMetrics:
    experiment_config(template_id, variant)
    started = time.perf_counter()
    raw_results = _run_raw_variant(template_id, variant)
    elapsed_ms = (time.perf_counter() - started) * 1000
    fallback_latency = elapsed_ms / max(1, len(raw_results))
    case_results = [
        CaseMetric(
            case_id=item.case_id,
            primary_score=item.primary_score,
            secondary=item.secondary,
            latency_ms=item.latency_ms or fallback_latency,
            evidence=list(item.evidence),
        )
        for item in raw_results
    ]
    bootstrap_results = _bootstrap(case_results)
    secondary_keys = sorted({key for item in case_results for key in item.secondary})
    latencies = sorted(item.latency_ms for item in case_results)
    p95_index = max(0, math.ceil(0.95 * len(latencies)) - 1)
    scores = [item.primary_score for item in case_results]
    aggregate = AggregateMetric(
        primary_score_mean=statistics.fmean(scores),
        primary_score_sample_std=statistics.stdev(scores),
        secondary_means={
            key: statistics.fmean(item.secondary.get(key, 0.0) for item in case_results)
            for key in secondary_keys
        },
        case_latency_ms_mean=statistics.fmean(latencies),
        case_latency_ms_p95=latencies[p95_index],
        case_count=len(case_results),
    )
    deterministic_payload = [
        {
            "caseId": item.case_id,
            "primaryScore": item.primary_score,
            "secondary": item.secondary,
            "evidence": item.evidence,
        }
        for item in case_results
    ]
    return ExperimentMetrics(
        case_results=case_results,
        bootstrap_results=bootstrap_results,
        aggregate=aggregate,
        result_sha256=canonical_json_sha256(deterministic_payload),
    )


def source_bundle_sha256(template_id: str) -> str:
    get_template(template_id)
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    paths = [
        root / "models.py",
        root / "runner_entry.py",
        root / "template.py",
        root / "template_catalog.py",
        root / "portfolio_benchmarks.py",
        root / "code_rag_experiment.py",
        root / "code_rag_benchmark.py",
    ]
    if template_id == TEMPLATE_ID:
        paths.append(DATA_PATH)
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def dependency_lock_sha256() -> str:
    lock = Path(__file__).resolve().parents[1] / "uv.lock"
    return hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else "unlocked-local-dev"


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def analyze_results(
    template_id: str,
    metrics: dict[str, ExperimentMetrics],
    expected_direction: ExpectedDirection,
) -> dict[str, Any]:
    template = get_template(template_id)
    variants = template_variants(template_id)
    baseline_id = variants[0]
    candidate_id = variants[-1]
    baseline = metrics[baseline_id]
    candidate = metrics[candidate_id]
    baseline_bootstrap = {item.replicate: item for item in baseline.bootstrap_results}
    candidate_bootstrap = {item.replicate: item for item in candidate.bootstrap_results}
    deltas = [
        candidate_bootstrap[replicate].primary_score
        - baseline_bootstrap[replicate].primary_score
        for replicate in range(BOOTSTRAP_REPLICATES)
    ]
    mean_delta = (
        candidate.aggregate.primary_score_mean - baseline.aggregate.primary_score_mean
    )
    alpha = 1 - CONFIDENCE_LEVEL
    confidence_lower = _percentile(deltas, alpha / 2)
    confidence_upper = _percentile(deltas, 1 - alpha / 2)
    threshold = template.effect_threshold
    if expected_direction == ExpectedDirection.INCREASE:
        agreements = sum(delta > 0 for delta in deltas)
        supported = mean_delta >= threshold and confidence_lower > 0
        partial = mean_delta > 0
    elif expected_direction == ExpectedDirection.DECREASE:
        agreements = sum(delta < 0 for delta in deltas)
        supported = mean_delta <= -threshold and confidence_upper < 0
        partial = mean_delta < 0
    else:
        agreements = sum(abs(delta) < threshold for delta in deltas)
        supported = abs(mean_delta) < threshold and confidence_lower <= 0 <= confidence_upper
        partial = abs(mean_delta) < threshold
    verdict = "supported" if supported else "partially_supported" if partial else "refuted"
    best_variant = max(variants, key=lambda item: metrics[item].aggregate.primary_score_mean)
    incremental_deltas = [
        {
            "from": previous,
            "to": current,
            "primaryMetricDelta": (
                metrics[current].aggregate.primary_score_mean
                - metrics[previous].aggregate.primary_score_mean
            ),
        }
        for previous, current in pairwise(variants)
    ]
    benchmark_statistics: dict[str, Any] = {
        "caseCount": baseline.aggregate.case_count,
        "benchmarkKind": template.benchmark_kind,
    }
    provenance: Any = template.provenance
    benchmark_digest = source_bundle_sha256(template_id)
    if template_id == TEMPLATE_ID:
        benchmark_statistics = {**STATISTICS, **benchmark_statistics}
        provenance = PROVENANCE
        benchmark_digest = benchmark_sha256()
    return {
        "templateId": template.id,
        "templateVersion": template.version,
        "benchmarkSha256": benchmark_digest,
        "benchmarkProvenance": provenance,
        "benchmarkStatistics": benchmark_statistics,
        "primaryMetric": template.primary_metric,
        "primaryMetricLabel": template.primary_metric_label,
        "expectedDirection": expected_direction.value,
        "variants": {
            variant: metrics[variant].aggregate.model_dump(mode="json", by_alias=True)
            for variant in variants
        },
        "baselineVariant": baseline_id,
        "candidateVariant": candidate_id,
        "bestVariant": best_variant,
        "incrementalDeltas": incremental_deltas,
        "pairedBootstrapDeltas": [
            {"replicate": replicate, "primaryMetricDelta": delta}
            for replicate, delta in enumerate(deltas)
        ],
        "baselinePrimaryScore": baseline.aggregate.primary_score_mean,
        "candidatePrimaryScore": candidate.aggregate.primary_score_mean,
        "meanPrimaryMetricDelta": mean_delta,
        "sampleStdPrimaryMetricDelta": statistics.stdev(deltas),
        "confidenceInterval95": {"lower": confidence_lower, "upper": confidence_upper},
        "directionAgreementCount": agreements,
        "claimRule": {
            "effectThreshold": threshold,
            "bootstrapSeed": BOOTSTRAP_SEED,
            "bootstrapReplicates": BOOTSTRAP_REPLICATES,
            "confidenceLevel": CONFIDENCE_LEVEL,
            "requiresDirectionalConfidenceInterval": True,
            "isSignificanceTest": False,
        },
        "verdict": verdict,
        "limitations": template.limitations,
        "externalValidation": template.external_validation,
        "zeroApiCost": True,
    }


def verify_reproduction(
    template_id: str,
    recorded: dict[str, ExperimentMetrics],
    reproduced: dict[str, ExperimentMetrics],
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    passed = True
    for variant in template_variants(template_id):
        old = recorded[variant]
        new = reproduced[variant]
        result_hash_match = old.result_sha256 == new.result_sha256
        primary_delta = abs(
            old.aggregate.primary_score_mean - new.aggregate.primary_score_mean
        )
        item_passed = result_hash_match and primary_delta <= REPRODUCTION_TOLERANCE
        passed = passed and item_passed
        comparisons.append(
            {
                "variant": variant,
                "resultHashMatch": result_hash_match,
                "primaryMetricAbsoluteDelta": primary_delta,
                "passed": item_passed,
            }
        )
    return {
        "passed": passed,
        "tolerance": REPRODUCTION_TOLERANCE,
        "latencyCompared": False,
        "comparisons": comparisons,
    }


def runtime_fingerprint() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "numpy": np.__version__,
        "scikitLearn": sklearn.__version__,
        "runtimeImageDigest": os.environ.get(
            "RESEARCH_RUNTIME_IMAGE_DIGEST", "local-dev-unpinned"
        ),
    }
