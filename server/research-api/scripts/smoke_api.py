from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any
from uuid import uuid4

import httpx


def request(
    client: httpx.Client,
    method: str,
    path: str,
    token: str,
    body: dict[str, Any] | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    if key:
        headers["Idempotency-Key"] = key
    response = client.request(method, path, headers=headers, json=body)
    if response.status_code >= 400:
        raise RuntimeError(f"{method} {path}: {response.status_code} {response.text}")
    return response.json()


def wait_for_status(
    client: httpx.Client,
    token: str,
    run_id: str,
    expected: str,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = request(client, "GET", f"/v1/research-runs/{run_id}", token)
        status = snapshot["run"]["status"]
        if status == expected:
            return snapshot
        if status == "failed":
            raise RuntimeError(json.dumps(snapshot["run"], ensure_ascii=False))
        time.sleep(0.1)
    raise TimeoutError(f"run {run_id} did not reach {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete ResearchRun P0 flow")
    parser.add_argument("--url", default="http://127.0.0.1:8790")
    parser.add_argument(
        "--token",
        default=os.environ.get(
            "RESEARCH_SMOKE_TOKEN",
            "dev:00000000-0000-4000-8000-000000000001",
        ),
    )
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--template", default="codebase_rag_retrieval_v1")
    parser.add_argument("--topic")
    args = parser.parse_args()
    suffix = uuid4().hex[:12]

    with httpx.Client(base_url=args.url, timeout=30) as client:
        catalog = request(client, "GET", "/v1/research-templates", args.token)
        template = next(
            (item for item in catalog["items"] if item["id"] == args.template),
            None,
        )
        if not template:
            raise RuntimeError(f"unknown template: {args.template}")
        baseline = template["variants"][0]["label"]
        candidate = template["variants"][-1]["label"]
        created = request(
            client,
            "POST",
            "/v1/research-runs",
            args.token,
            {
                "topic": args.topic or template["nameZh"],
                "templateId": template["id"],
            },
            f"create-{suffix}",
        )
        run_id = created["run"]["id"]
        hypothesis = request(
            client,
            "PUT",
            f"/v1/research-runs/{run_id}/hypothesis",
            args.token,
            {
                "expectedVersion": created["run"]["version"],
                "statement": (
                    f"{candidate} improves {template['primaryMetricLabel']} over {baseline}."
                ),
                "independentVariable": template["independentVariable"],
                "dependentMetric": template["primaryMetric"],
                "expectedDirection": template["expectedDirection"],
                "rationale": (
                    "The fixed variants add task-specific mechanisms while the frozen cases "
                    "preserve a paired comparison."
                ),
            },
        )
        request(
            client,
            "POST",
            f"/v1/research-runs/{run_id}/experiment-jobs",
            args.token,
            {
                "expectedVersion": hypothesis["run"]["version"],
                "benchmark": {"includeAllVariants": True},
            },
            f"enqueue-{suffix}",
        )
        snapshot = wait_for_status(client, args.token, run_id, "analysis_ready", args.timeout)
        request(
            client,
            "POST",
            f"/v1/research-runs/{run_id}/analysis",
            args.token,
            {"expectedVersion": snapshot["run"]["version"]},
            f"analysis-{suffix}",
        )
        snapshot = wait_for_status(client, args.token, run_id, "verification_required", args.timeout)
        request(
            client,
            "POST",
            f"/v1/research-runs/{run_id}/verify",
            args.token,
            {"expectedVersion": snapshot["run"]["version"]},
            f"verify-{suffix}",
        )
        snapshot = wait_for_status(client, args.token, run_id, "completed", args.timeout)
        analysis = next(item for item in snapshot["artifacts"] if item["kind"] == "analysis")
        card = next(item for item in snapshot["artifacts"] if item["kind"] == "research_card")
        published = request(
            client,
            "POST",
            f"/v1/research-artifacts/{card['id']}/publish",
            args.token,
            {"expectedVersion": snapshot["run"]["version"]},
            f"publish-{suffix}",
        )
        unpublished = request(
            client,
            "DELETE",
            f"/v1/research-artifacts/{card['id']}/publication",
            args.token,
            {"expectedVersion": published["runVersion"]},
            f"unpublish-{suffix}",
        )

    print(
        json.dumps(
            {
                "runId": run_id,
                "templateId": template["id"],
                "status": snapshot["run"]["status"],
                "verdict": analysis["content"]["verdict"],
                "bestVariant": analysis["content"]["bestVariant"],
                "primaryMetric": analysis["content"]["primaryMetric"],
                "meanPrimaryMetricDelta": analysis["content"]["meanPrimaryMetricDelta"],
                "confidenceInterval95": analysis["content"]["confidenceInterval95"],
                "directionAgreementCount": analysis["content"]["directionAgreementCount"],
                "artifacts": len(snapshot["artifacts"]),
                "publishedThenWithdrawn": unpublished["artifact"]["visibility"] == "private",
                "finalRunVersion": unpublished["runVersion"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
