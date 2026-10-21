"""Write the controlled neural-world benchmark terminal decision report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, write_text_atomic  # noqa: E402


def report_controlled_world_benchmark_decision(*, config_path: Path, artifact_root: Path) -> Path:
    config = load_config(config_path)
    paths = dict(config["controlled_world_benchmark_qualification"]["output_paths"])
    audit_path = artifact_root / Path(paths["qualification_audit"])
    qualification = json.loads(audit_path.read_text(encoding="utf-8"))
    terminal = qualification["terminal_decision"]
    if terminal not in {"CONTROLLED_NEURAL_WORLD_BENCHMARK_QUALIFIED", "CONTROLLED_NEURAL_WORLD_BENCHMARK_NOT_QUALIFIED"}:
        raise ValueError("Qualification terminal is not exact.")
    report = "\n".join(
        [
            "# Controlled Neural-World Benchmark Qualification",
            "",
            f"Terminal decision: `{terminal}`",
            "",
            "Evidence class: controlled-world evidence only; not empirical, therapeutic, or clinical evidence.",
            "Access counters: six_context_outcome_access=0, protected_holdout_access=0, test_outcome_access=0.",
            f"Blocker count: {len(qualification['blockers'])}",
            "",
        ]
    )
    output = artifact_root / Path(paths["decision_report"])
    write_text_atomic(output, report)
    print(f"decision_report={output}")
    print(f"terminal_decision={terminal}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "Libs/configs/config_stack.yaml")
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    report_controlled_world_benchmark_decision(config_path=args.config.resolve(), artifact_root=args.artifact_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
