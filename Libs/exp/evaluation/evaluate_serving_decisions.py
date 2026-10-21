from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.paper_grade_entrypoint import run_catalog_entrypoint


def main() -> None:
    run_catalog_entrypoint(
        "Evaluate serving decision-label contract metrics from formal result catalogs.",
        "serving_decision_eval",
        [
            "decision_label_accuracy",
            "accept_precision",
            "active_learning_route_rate",
            "external_validation_route_rate",
            "insufficient_evidence_route_rate",
        ],
    )


if __name__ == "__main__":
    main()
