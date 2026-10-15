from __future__ import annotations

import json
from pathlib import Path

from fastapi.routing import APIRoute

from research_api.main import create_app


def normalize_path(path: str) -> str:
    return path.replace("{run_id}", "{runId}").replace("{artifact_id}", "{artifactId}")


def test_implemented_methods_match_canonical_openapi(settings):
    root = Path(__file__).resolve().parents[3]
    canonical = json.loads((root / "docs/research-run-openapi.json").read_text())
    expected = {
        (normalize_path(path), method.upper())
        for path, path_item in canonical["paths"].items()
        for method in path_item
        if method.lower() in {"get", "post", "put", "patch", "delete"}
    }

    application = create_app(settings=settings)
    actual = {
        (normalize_path(route.path), method)
        for route in application.routes
        if isinstance(route, APIRoute) and not route.path.startswith(("/docs", "/redoc", "/openapi"))
        for method in route.methods
        if method != "HEAD"
    }
    assert actual == expected


def test_canonical_contract_contains_deterministic_result_hash():
    root = Path(__file__).resolve().parents[3]
    canonical = json.loads((root / "docs/research-run-openapi.json").read_text())
    metrics = canonical["components"]["schemas"]["ExperimentMetrics"]
    assert "resultSha256" in metrics["required"]
    assert metrics["properties"]["resultSha256"]["pattern"] == "^[0-9a-f]{64}$"
