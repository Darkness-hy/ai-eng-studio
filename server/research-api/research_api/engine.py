from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Protocol

from .models import ExperimentMetrics
from .template import template_variants

logger = logging.getLogger("research.runner")


class RunnerError(RuntimeError):
    def __init__(self, code: str, safe_message: str) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message


class ResearchEngine(Protocol):
    async def run_variant(self, template_id: str, variant: str) -> ExperimentMetrics: ...

    async def run_suite(self, template_id: str) -> dict[str, ExperimentMetrics]: ...

    def runtime_fingerprint(self) -> dict[str, str]: ...


class FixedResearchEngine:
    def __init__(
        self,
        mode: str = "local",
        image: str = "ai-eng-studio-research-runner:0.4.0",
        runtime_image_digest: str = "local-dev-unpinned",
        timeout_seconds: int = 120,
        max_concurrency: int = 2,
    ) -> None:
        if mode not in {"local", "docker"}:
            raise ValueError("runner mode must be local or docker")
        self.mode = mode
        self.image = image
        self.runtime_image_digest = runtime_image_digest
        self.timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._package_root = Path(__file__).resolve().parents[1]
        self._runtime: dict[str, str] = {}

    def runtime_fingerprint(self) -> dict[str, str]:
        return {
            **self._runtime,
            "runtimeImageDigest": self.runtime_image_digest,
        }

    async def run_suite(self, template_id: str) -> dict[str, ExperimentMetrics]:
        variants = template_variants(template_id)
        results = await asyncio.gather(
            *(self.run_variant(template_id, variant) for variant in variants)
        )
        return dict(zip(variants, results, strict=True))

    async def run_variant(self, template_id: str, variant: str) -> ExperimentMetrics:
        async with self._semaphore:
            command, env = self._command(template_id, variant)
            with tempfile.TemporaryDirectory(prefix="research-run-") as workdir:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=workdir,
                    env=env,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), timeout=self.timeout_seconds
                    )
                except TimeoutError as exc:
                    process.kill()
                    await process.wait()
                    raise RunnerError("experiment_timed_out", "The fixed experiment timed out") from exc
                except asyncio.CancelledError:
                    process.kill()
                    await process.wait()
                    raise

            if process.returncode != 0:
                detail = stderr.decode("utf-8", errors="replace")[-2000:]
                logger.error(
                    "fixed runner failed: mode=%s template=%s variant=%s detail=%s",
                    self.mode,
                    template_id,
                    variant,
                    detail,
                )
                raise RunnerError(
                    "experiment_failed",
                    "The fixed experiment failed",
                )
            try:
                payload = json.loads(stdout.decode("utf-8"))
                metrics = ExperimentMetrics.model_validate(payload["metrics"])
                runtime = payload["runtime"]
                if not isinstance(runtime, dict) or not all(
                    isinstance(key, str) and isinstance(value, str) for key, value in runtime.items()
                ):
                    raise ValueError("invalid runtime fingerprint")
                self._runtime = runtime
                return metrics
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                raise RunnerError("experiment_invalid_output", "The runner returned invalid metrics") from exc

    def _command(self, template_id: str, variant: str) -> tuple[list[str], dict[str, str]]:
        base_env = {
            "PATH": os.environ.get("PATH", ""),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        if self.mode == "local":
            base_env["PYTHONPATH"] = str(self._package_root)
            return [
                sys.executable,
                "-m",
                "research_api.runner_entry",
                "--template",
                template_id,
                "--variant",
                variant,
            ], base_env

        docker = shutil.which("docker")
        if not docker:
            raise RunnerError("runner_unavailable", "Docker is required for the production runner")
        image_reference = (
            self.runtime_image_digest
            if self.runtime_image_digest.startswith("sha256:")
            else self.image
        )
        command = [
            docker,
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cpus",
            "1",
            "--memory",
            "1024m",
            "--pids-limit",
            "128",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            image_reference,
            "--template",
            template_id,
            "--variant",
            variant,
        ]
        return command, base_env
