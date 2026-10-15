# ResearchRun API

FastAPI service for the bounded research-game workflow:

```text
topic -> hypothesis -> fixed offline experiment -> analysis -> verification -> publication
```

Version `0.4.0` exposes 17 fixed capstone templates. The service accepts no user
code, commands, URLs, packages, datasets, variants, or arbitrary experiment
configuration. Experiment, analysis, and verification make no paid API calls.
See [`docs/zero-api-capstone-matrix.md`](../../docs/zero-api-capstone-matrix.md)
for the implemented scope and external validation boundaries.

## Architecture

- Supabase access JWTs authenticate users from JWT `sub`.
- PostgreSQL transactions persist runs, immutable attempts, artifacts, and SSE events.
- A database dispatcher recovers queued jobs after service restart.
- Browser table access is read-only behind RLS.
- Development uses a sanitized local subprocess.
- Production requires an immutable Docker image with no network, read-only root,
  one CPU, 1 GiB memory, and a 120-second timeout.
- Analysis and verification are deterministic Python and do not accept LLM edits.

## Local Start

```bash
cd server/research-api
uv sync --all-groups
export RESEARCH_DEV_AUTH=1
export RESEARCH_STORE=memory
export RESEARCH_RUNNER_MODE=local
./run.sh
```

```bash
curl --noproxy '*' http://127.0.0.1:8790/health
.venv/bin/python scripts/smoke_api.py
```

The development bearer token is restricted to `RESEARCH_DEV_USER_ID`.
Development auth is rejected in production.

## Runner

```bash
docker build -f runner.Dockerfile \
  -t ai-eng-studio-research-runner:0.4.0 .
```

Run one variant through the restricted container:

```bash
docker run --rm --network none --read-only --cpus 1 --memory 1024m \
  --pids-limit 128 --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  ai-eng-studio-research-runner:0.4.0 \
  --template codebase_rag_retrieval_v1 --variant hybrid_rerank
```

Capstone 02 uses the committed CodeSearchNet snapshot. Rebuild that snapshot
only when intentionally revising its provenance:

```bash
uv run python scripts/build_codesearchnet_benchmark.py
```

The builder pins the official annotation hash and source commit. Runtime
containers remain offline.

## Production Settings

```text
RESEARCH_ENV=production
RESEARCH_STORE=postgres
RESEARCH_RUNNER_MODE=docker
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_DB_URL=postgresql://<server-only-credential>@<pooler>/<database>
SUPABASE_JWT_AUDIENCE=authenticated
RESEARCH_RUNTIME_IMAGE_DIGEST=sha256:<registry-image-digest>
```

Never expose database, JWT, or service-role secrets through `VITE_*` variables
or pass them into runner containers.

## Database

Apply these files in order to a non-production project:

1. [`supabase/013-research-runs.sql`](../../supabase/013-research-runs.sql)
2. [`supabase/014-zero-api-capstone-portfolio.sql`](../../supabase/014-zero-api-capstone-portfolio.sql)

They create the ResearchRun tables, RLS, private artifact bucket, atomic
publication commands, and the 17-template whitelist. The API uses a server-only
PostgreSQL connection; client reads remain governed by RLS.

## Verification

```bash
uv run ruff check research_api tests
uv run pytest -q
TEST_DATABASE_URL=postgresql://... uv run pytest -q
```

The PostgreSQL test truncates ResearchRun tables and test profiles. Never point
`TEST_DATABASE_URL` at production.

The canonical contract is
[`docs/research-run-openapi.json`](../../docs/research-run-openapi.json). The test
suite compares implemented method/path pairs and generic deterministic metrics
against it.
