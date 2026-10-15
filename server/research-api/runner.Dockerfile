FROM python:3.11.15-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app
COPY runner-requirements.txt ./
RUN pip install --no-cache-dir -r runner-requirements.txt \
    && useradd --create-home --uid 10001 runner

COPY research_api/__init__.py research_api/models.py research_api/template.py \
    research_api/template_catalog.py research_api/portfolio_benchmarks.py \
    research_api/code_rag_benchmark.py research_api/code_rag_experiment.py \
    research_api/runner_entry.py ./research_api/
COPY research_api/data ./research_api/data

USER runner
ENTRYPOINT ["python", "-m", "research_api.runner_entry"]
