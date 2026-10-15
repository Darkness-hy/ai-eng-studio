from __future__ import annotations

import argparse
import json

from .template import run_fixed_variant, runtime_fingerprint, template_variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one fixed ResearchRun variant")
    parser.add_argument("--template", required=True)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    if args.variant not in template_variants(args.template):
        parser.error("variant is not enabled for the selected template")
    metrics = run_fixed_variant(args.template, args.variant)
    print(
        json.dumps(
            {
                "metrics": metrics.model_dump(mode="json", by_alias=True),
                "runtime": runtime_fingerprint(),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
