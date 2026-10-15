from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

async def verify(mode: str, image: str, digest: str) -> None:
    from research_api.engine import FixedResearchEngine
    from research_api.template import (
        analyze_results,
        run_fixed_variant,
        template_definitions,
        verify_reproduction,
    )

    engine = FixedResearchEngine(
        mode=mode,
        image=image,
        runtime_image_digest=digest,
        timeout_seconds=120,
    )
    summaries: list[dict[str, object]] = []
    for template in template_definitions():
        expected = {
            variant.id: run_fixed_variant(template.id, variant.id)
            for variant in template.variants
        }
        observed = await engine.run_suite(template.id)
        reproduction = verify_reproduction(template.id, expected, observed)
        if not reproduction["passed"]:
            raise RuntimeError(f"container reproduction failed for {template.id}")
        analysis = analyze_results(template.id, observed, template.expected_direction)
        summaries.append(
            {
                "capstoneNumber": template.capstone_number,
                "templateId": template.id,
                "variantCount": len(template.variants),
                "caseCount": next(iter(observed.values())).aggregate.case_count,
                "meanPrimaryMetricDelta": analysis["meanPrimaryMetricDelta"],
                "verdict": analysis["verdict"],
                "bestVariant": analysis["bestVariant"],
                "reproduced": True,
            }
        )
    print(json.dumps({"templates": summaries}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify every fixed zero-API capstone template")
    parser.add_argument("--mode", choices=("local", "docker"), default="local")
    parser.add_argument("--image", default="ai-eng-studio-research-runner:0.4.0")
    parser.add_argument("--digest", default="local-dev-unpinned")
    args = parser.parse_args()
    asyncio.run(verify(args.mode, args.image, args.digest))


if __name__ == "__main__":
    main()
