from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.paper_grade_entrypoint import run_catalog_entrypoint


def main() -> None:
    run_catalog_entrypoint(
        "Evaluate runtime efficiency metrics from formal result catalogs.",
        "efficiency_profile",
        [
            "candidates_per_second",
            "gpu_hours_per_10000_candidates",
            "peak_memory_gb",
            "latency_ms",
            "throughput_candidates_per_sec",
        ],
    )


if __name__ == "__main__":
    main()
