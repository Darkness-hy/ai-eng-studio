from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import UUID

import jwt
import pytest
from fastapi import HTTPException

from research_api.auth import Authenticator
from research_api.fingerprint import request_fingerprint
from research_api.main import encode_sse
from research_api.models import Actor, ResearchEvent, ResearchStage


@pytest.mark.asyncio
async def test_hs256_supabase_token_validation(settings):
    issuer = "https://project.example/auth/v1"
    secret = "test-only-secret-that-is-long-enough"
    production_auth = replace(
        settings,
        dev_auth=False,
        supabase_jwt_secret=secret,
        jwt_issuer=issuer,
    )
    user_id = UUID("00000000-0000-4000-8000-000000000077")
    token = jwt.encode(
        {
            "sub": str(user_id),
            "aud": "authenticated",
            "iss": issuer,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        },
        secret,
        algorithm="HS256",
    )
    assert await Authenticator(production_auth).authenticate(token) == user_id

    wrong_audience = jwt.encode(
        {
            "sub": str(user_id),
            "aud": "wrong",
            "iss": issuer,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        },
        secret,
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as rejected:
        await Authenticator(production_auth).authenticate(wrong_audience)
    assert rejected.value.status_code == 401


def test_request_fingerprint_is_canonical_and_path_sensitive():
    first = request_fingerprint("post", "//v1/research-runs/", {"b": 2, "a": 1})
    second = request_fingerprint("POST", "/v1/research-runs", {"a": 1, "b": 2})
    different = request_fingerprint("POST", "/v1/other", {"a": 1, "b": 2})
    assert first == second
    assert different != first


def test_sse_encoding_uses_event_id_type_and_camel_case():
    event = ResearchEvent(
        id="1842",
        runId="4be266ef-8c68-4629-9e41-78f421e840ad",
        actor=Actor.RUNNER,
        eventType="experiment.job_succeeded",
        stage=ResearchStage.EXPERIMENT,
        payload={"variant": "hybrid"},
        createdAt="2026-07-11T08:00:00Z",
    )
    encoded = encode_sse(event)
    assert encoded.startswith("id: 1842\nevent: experiment.job_succeeded\ndata: ")
    assert '"runId":"4be266ef-8c68-4629-9e41-78f421e840ad"' in encoded
    assert encoded.endswith("\n\n")
