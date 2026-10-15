from __future__ import annotations

from collections.abc import Iterator
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from research_api.config import Settings
from research_api.main import create_app
from research_api.models import ExperimentMetrics
from research_api.store import MemoryResearchStore
from research_api.template import TEMPLATE_ID, run_fixed_variant, template_definitions


class StaticEngine:
    def __init__(self, metrics: dict[str, dict[str, ExperimentMetrics]]) -> None:
        self._metrics = metrics

    async def run_variant(self, template_id: str, variant: str) -> ExperimentMetrics:
        return self._metrics[template_id][variant].model_copy(deep=True)

    async def run_suite(self, template_id: str) -> dict[str, ExperimentMetrics]:
        return {
            variant: metrics.model_copy(deep=True)
            for variant, metrics in self._metrics[template_id].items()
        }

    def runtime_fingerprint(self) -> dict[str, str]:
        return {
            "python": "test",
            "architecture": "test",
            "numpy": "test",
            "scikitLearn": "test",
            "runtimeImageDigest": "sha256:" + "0" * 64,
        }


@pytest.fixture(scope="session")
def all_fixed_metrics() -> dict[str, dict[str, ExperimentMetrics]]:
    return {
        template.id: {
            variant.id: run_fixed_variant(template.id, variant.id)
            for variant in template.variants
        }
        for template in template_definitions()
    }


@pytest.fixture(scope="session")
def fixed_metrics(
    all_fixed_metrics: dict[str, dict[str, ExperimentMetrics]],
) -> dict[str, ExperimentMetrics]:
    return all_fixed_metrics[TEMPLATE_ID]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        environment="test",
        store_mode="memory",
        database_url=None,
        supabase_url=None,
        supabase_jwt_secret=None,
        jwt_audience="authenticated",
        jwt_issuer=None,
        dev_auth=True,
        dev_user_id=UUID("00000000-0000-4000-8000-000000000001"),
        allowed_origins=("http://localhost:5180",),
        runner_mode="local",
        runner_image="unused-in-tests",
        runtime_image_digest="local-test-unpinned",
        runner_timeout_seconds=120,
        dispatcher_poll_seconds=0.01,
        event_poll_seconds=0.01,
    )


@pytest.fixture
def api_client(
    settings: Settings,
    all_fixed_metrics: dict[str, dict[str, ExperimentMetrics]],
) -> Iterator[tuple[TestClient, MemoryResearchStore, UUID]]:
    store = MemoryResearchStore()
    engine = StaticEngine(all_fixed_metrics)
    application = create_app(settings=settings, store=store, engine=engine)
    with TestClient(application) as client:
        yield client, store, settings.dev_user_id


def auth_headers(user_id: UUID, key: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer dev:{user_id}"}
    if key:
        headers["Idempotency-Key"] = key
    return headers
