from __future__ import annotations

import pytest

from research_api.engine import FixedResearchEngine
from research_api.models import ExpectedDirection
from research_api.template import (
    REPRODUCTION_TOLERANCE,
    TEMPLATE_ID,
    analyze_results,
    run_fixed_variant,
    template_definition,
    template_definitions,
    verify_reproduction,
)


def test_code_rag_template_uses_predeclared_claim_rule(fixed_metrics):
    template = template_definition(TEMPLATE_ID)
    analysis = analyze_results(TEMPLATE_ID, fixed_metrics, ExpectedDirection.INCREASE)

    assert analysis["meanPrimaryMetricDelta"] > template.effect_threshold
    assert analysis["confidenceInterval95"]["lower"] > 0
    assert analysis["directionAgreementCount"] == 1000
    assert analysis["verdict"] == "supported"
    assert analysis["bestVariant"] == "hybrid"
    assert analysis["incrementalDeltas"][-1]["primaryMetricDelta"] < 0
    assert analysis["claimRule"] == {
        "effectThreshold": 0.02,
        "bootstrapSeed": 20260711,
        "bootstrapReplicates": 1000,
        "confidenceLevel": 0.95,
        "requiresDirectionalConfidenceInterval": True,
        "isSignificanceTest": False,
    }


def test_reproduction_ignores_latency_but_requires_result_hash(fixed_metrics):
    reproduced = {variant: metrics.model_copy(deep=True) for variant, metrics in fixed_metrics.items()}
    reproduced["bm25_body"].case_results[0].latency_ms += 9999
    result = verify_reproduction(TEMPLATE_ID, fixed_metrics, reproduced)
    assert result["passed"] is True
    assert result["latencyCompared"] is False
    assert result["tolerance"] == REPRODUCTION_TOLERANCE

    reproduced["bm25_body"].result_sha256 = "0" * 64
    assert verify_reproduction(TEMPLATE_ID, fixed_metrics, reproduced)["passed"] is False


def test_all_17_zero_api_templates_execute_and_reproduce(all_fixed_metrics):
    templates = template_definitions()
    assert len(templates) == 17
    assert {template.capstone_number for template in templates} == set(range(1, 18))

    for template in templates:
        recorded = all_fixed_metrics[template.id]
        reproduced = {
            variant.id: run_fixed_variant(template.id, variant.id)
            for variant in template.variants
        }
        assert set(recorded) == {variant.id for variant in template.variants}
        assert verify_reproduction(template.id, recorded, reproduced)["passed"] is True
        analysis = analyze_results(template.id, recorded, template.expected_direction)
        assert analysis["zeroApiCost"] is True
        assert analysis["verdict"] in {"supported", "partially_supported", "refuted"}


def test_docker_runner_uses_configured_immutable_digest(monkeypatch):
    monkeypatch.setattr("research_api.engine.shutil.which", lambda _name: "/usr/bin/docker")
    digest = "sha256:" + "a" * 64
    engine = FixedResearchEngine(
        mode="docker",
        image="mutable-runner:latest",
        runtime_image_digest=digest,
    )
    command, _environment = engine._command(TEMPLATE_ID, "bm25_body")
    assert digest in command
    assert "mutable-runner:latest" not in command


@pytest.mark.asyncio
async def test_local_runner_subprocess_returns_valid_metrics():
    engine = FixedResearchEngine(mode="local", timeout_seconds=30)
    metrics = await engine.run_variant(TEMPLATE_ID, "bm25_body")
    assert len(metrics.case_results) == 85
    assert len(metrics.bootstrap_results) == 1000
    assert metrics.aggregate.primary_score_mean > 0.7
    assert len(metrics.result_sha256) == 64
