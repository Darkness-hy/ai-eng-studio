from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parent / "data" / "codesearchnet_python_human_v1.json.gz"
EXPECTED_SCHEMA_VERSION = "2.0.0"
EXPECTED_SOURCE_COMMIT = "106e827405c968597da938f6b373d30183918869"
EXPECTED_ANNOTATION_SHA256 = "0340af32b551ceadb74fec147f97642b7fedf3ff039e38fb86baff49ee899846"
EXPECTED_BENCHMARK_SHA256 = "a8c0adedf36a283410cb66a8a26bce057ee3a151a408d4366ada83c19c4af274"
CORPUS_VERSION = "codesearchnet-python-human-v1"
QUERY_SET_VERSION = "codesearchnet-human-eval-v1"
SOURCE_REF = f"github/CodeSearchNet@{EXPECTED_SOURCE_COMMIT}"


def _load() -> tuple[dict[str, Any], bytes]:
    with gzip.open(DATA_PATH, "rb") as benchmark_file:
        encoded = benchmark_file.read()
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != EXPECTED_BENCHMARK_SHA256:
        raise RuntimeError(
            f"CodeSearchNet benchmark digest mismatch: expected {EXPECTED_BENCHMARK_SHA256}, got {digest}"
        )
    payload = json.loads(encoded)
    if payload.get("schemaVersion") != EXPECTED_SCHEMA_VERSION:
        raise RuntimeError("unsupported CodeSearchNet benchmark schema")
    source = payload.get("source", {})
    if source.get("commit") != EXPECTED_SOURCE_COMMIT:
        raise RuntimeError("CodeSearchNet source commit mismatch")
    if source.get("annotationSha256") != EXPECTED_ANNOTATION_SHA256:
        raise RuntimeError("CodeSearchNet annotation digest mismatch")
    statistics = payload.get("statistics", {})
    if statistics.get("chunks") != 938 or statistics.get("testQueries") != 85:
        raise RuntimeError("unexpected frozen CodeSearchNet benchmark dimensions")
    return payload, encoded


BENCHMARK, _ENCODED_BENCHMARK = _load()
CHUNKS: list[dict[str, Any]] = BENCHMARK["chunks"]
ALL_QUERIES: list[dict[str, Any]] = BENCHMARK["queries"]
DEV_QUERIES: list[dict[str, Any]] = [item for item in ALL_QUERIES if item["split"] == "dev"]
QUERIES: list[dict[str, Any]] = [item for item in ALL_QUERIES if item["split"] == "test"]
PROVENANCE: dict[str, Any] = BENCHMARK["source"]
STATISTICS: dict[str, int] = BENCHMARK["statistics"]


def benchmark_sha256() -> str:
    return hashlib.sha256(_ENCODED_BENCHMARK).hexdigest()
