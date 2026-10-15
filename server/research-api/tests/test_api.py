from __future__ import annotations

import time

from fastapi.testclient import TestClient

from .conftest import auth_headers


def wait_for_status(
    client: TestClient,
    headers: dict[str, str],
    run_id: str,
    expected: str,
    timeout: float = 5,
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/v1/research-runs/{run_id}", headers=headers)
        assert response.status_code == 200, response.text
        snapshot = response.json()
        if snapshot["run"]["status"] == expected:
            return snapshot
        if snapshot["run"]["status"] == "failed":
            raise AssertionError(snapshot)
        time.sleep(0.01)
    raise AssertionError(f"run did not reach {expected}")


def test_complete_cpu_vertical_slice_and_publication(api_client):
    client, _store, user_id = api_client
    auth = auth_headers(user_id)
    create_body = {
        "topic": "Cross-repository code retrieval",
        "templateId": "codebase_rag_retrieval_v1",
    }
    created = client.post(
        "/v1/research-runs",
        headers=auth_headers(user_id, "create-api-001"),
        json=create_body,
    )
    assert created.status_code == 201, created.text
    snapshot = created.json()
    run_id = snapshot["run"]["id"]
    assert snapshot["run"]["version"] == 1
    assert snapshot["run"]["currentStage"] == "hypothesis"

    replay = client.post(
        "/v1/research-runs",
        headers=auth_headers(user_id, "create-api-001"),
        json=create_body,
    )
    assert replay.status_code == 201
    assert replay.json()["run"]["id"] == run_id

    hypothesis = client.put(
        f"/v1/research-runs/{run_id}/hypothesis",
        headers=auth,
        json={
            "expectedVersion": 1,
            "statement": "The complete retrieval pipeline increases nDCG@10 over body-only BM25.",
            "independentVariable": "retrieval_pipeline",
            "dependentMetric": "ndcg_at_10",
            "expectedDirection": "increase",
            "rationale": "AST fields, local LSA retrieval, and reranking expose complementary code signals.",
        },
    )
    assert hypothesis.status_code == 200, hypothesis.text
    assert hypothesis.json()["run"]["version"] == 2

    queued = client.post(
        f"/v1/research-runs/{run_id}/experiment-jobs",
        headers=auth_headers(user_id, "enqueue-api-001"),
        json={"expectedVersion": 2, "benchmark": {"includeAllVariants": True}},
    )
    assert queued.status_code == 202, queued.text
    assert {item["variant"] for item in queued.json()["jobs"]} == {
        "bm25_body",
        "bm25_ast",
        "hybrid",
        "hybrid_rerank",
    }

    snapshot = wait_for_status(client, auth, run_id, "analysis_ready")
    assert snapshot["run"]["version"] == 5
    assert all(item["status"] == "succeeded" for item in snapshot["jobs"])

    analyzing = client.post(
        f"/v1/research-runs/{run_id}/analysis",
        headers=auth_headers(user_id, "analysis-api-001"),
        json={"expectedVersion": 5},
    )
    assert analyzing.status_code == 202, analyzing.text
    snapshot = wait_for_status(client, auth, run_id, "verification_required")
    assert snapshot["run"]["version"] == 7
    analysis = next(item for item in snapshot["artifacts"] if item["kind"] == "analysis")
    assert analysis["content"]["verdict"] == "supported"
    assert analysis["content"]["bestVariant"] == "hybrid"

    analysis_replay = client.post(
        f"/v1/research-runs/{run_id}/analysis",
        headers=auth_headers(user_id, "analysis-api-001"),
        json={"expectedVersion": 5},
    )
    assert analysis_replay.status_code == 202
    assert analysis_replay.json()["eventId"] == analyzing.json()["eventId"]
    time.sleep(0.05)
    assert client.get(f"/v1/research-runs/{run_id}", headers=auth).json()["run"]["status"] == (
        "verification_required"
    )

    verifying = client.post(
        f"/v1/research-runs/{run_id}/verify",
        headers=auth_headers(user_id, "verify-api-001"),
        json={"expectedVersion": 7},
    )
    assert verifying.status_code == 202, verifying.text
    snapshot = wait_for_status(client, auth, run_id, "completed")
    assert snapshot["run"]["version"] == 9
    card = next(item for item in snapshot["artifacts"] if item["kind"] == "research_card")
    assert card["verificationStatus"] == "passed"
    assert card["visibility"] == "private"

    published = client.post(
        f"/v1/research-artifacts/{card['id']}/publish",
        headers=auth_headers(user_id, "publish-api-001"),
        json={"expectedVersion": 9},
    )
    assert published.status_code == 200, published.text
    assert published.json()["artifact"]["visibility"] == "public"
    assert published.json()["runVersion"] == 10

    publish_replay = client.post(
        f"/v1/research-artifacts/{card['id']}/publish",
        headers=auth_headers(user_id, "publish-api-001"),
        json={"expectedVersion": 9},
    )
    assert publish_replay.status_code == 200
    assert publish_replay.json()["eventId"] == published.json()["eventId"]

    unpublished = client.request(
        "DELETE",
        f"/v1/research-artifacts/{card['id']}/publication",
        headers=auth_headers(user_id, "unpublish-api-001"),
        json={"expectedVersion": 10},
    )
    assert unpublished.status_code == 200, unpublished.text
    assert unpublished.json()["artifact"]["visibility"] == "private"
    assert unpublished.json()["runVersion"] == 11

    events = client.get(f"/v1/research-runs/{run_id}", headers=auth).json()
    assert int(events["lastEventId"]) >= 11


def test_auth_validation_and_forbidden_request_fields(api_client):
    client, _store, user_id = api_client
    assert client.get("/health").status_code == 200
    assert client.get("/v1/research-runs").status_code == 401
    assert (
        client.get(
            "/v1/research-runs",
            headers={"Authorization": "Bearer dev:00000000-0000-4000-8000-000000000002"},
        ).status_code
        == 401
    )
    response = client.post(
        "/v1/research-runs",
        headers=auth_headers(user_id, "create-api-bad-001"),
        json={
            "topic": "Cross-repository code retrieval",
            "templateId": "codebase_rag_retrieval_v1",
            "userId": str(user_id),
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_failed"


def test_template_catalog_and_non_code_capstone_vertical_slice(api_client):
    client, _store, user_id = api_client
    auth = auth_headers(user_id)
    catalog_response = client.get("/v1/research-templates", headers=auth)
    assert catalog_response.status_code == 200, catalog_response.text
    templates = catalog_response.json()["items"]
    assert len(templates) == 17
    assert {item["capstoneNumber"] for item in templates} == set(range(1, 18))
    assert all(item["zeroApiCost"] is True and item["runtimeMode"] == "offline" for item in templates)

    template = next(item for item in templates if item["id"] == "adaptive_tutor_bkt_v1")
    created = client.post(
        "/v1/research-runs",
        headers=auth_headers(user_id, "create-api-bkt-001"),
        json={"topic": "BKT spaced practice", "templateId": template["id"]},
    )
    assert created.status_code == 201, created.text
    snapshot = created.json()
    run_id = snapshot["run"]["id"]

    hypothesis = client.put(
        f"/v1/research-runs/{run_id}/hypothesis",
        headers=auth,
        json={
            "expectedVersion": 1,
            "statement": "BKT spaced practice improves the fixed learner utility over a linear curriculum.",
            "independentVariable": template["independentVariable"],
            "dependentMetric": template["primaryMetric"],
            "expectedDirection": template["expectedDirection"],
            "rationale": (
                "The policy revisits weak skills while the fixed learner traces "
                "preserve paired comparison."
            ),
        },
    )
    assert hypothesis.status_code == 200, hypothesis.text

    queued = client.post(
        f"/v1/research-runs/{run_id}/experiment-jobs",
        headers=auth_headers(user_id, "enqueue-api-bkt-001"),
        json={"expectedVersion": 2, "benchmark": {"includeAllVariants": True}},
    )
    assert queued.status_code == 202, queued.text
    assert {item["variant"] for item in queued.json()["jobs"]} == {
        item["id"] for item in template["variants"]
    }

    snapshot = wait_for_status(client, auth, run_id, "analysis_ready")
    analyzing = client.post(
        f"/v1/research-runs/{run_id}/analysis",
        headers=auth_headers(user_id, "analysis-api-bkt-001"),
        json={"expectedVersion": snapshot["run"]["version"]},
    )
    assert analyzing.status_code == 202, analyzing.text
    snapshot = wait_for_status(client, auth, run_id, "verification_required")

    verifying = client.post(
        f"/v1/research-runs/{run_id}/verify",
        headers=auth_headers(user_id, "verify-api-bkt-001"),
        json={"expectedVersion": snapshot["run"]["version"]},
    )
    assert verifying.status_code == 202, verifying.text
    snapshot = wait_for_status(client, auth, run_id, "completed")
    analysis = next(item for item in snapshot["artifacts"] if item["kind"] == "analysis")
    card = next(item for item in snapshot["artifacts"] if item["kind"] == "research_card")
    assert analysis["content"]["primaryMetric"] == template["primaryMetric"]
    assert card["content"]["zeroApiCost"] is True
