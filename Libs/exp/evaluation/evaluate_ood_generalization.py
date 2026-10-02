from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.paper_grade_entrypoint import run_catalog_entrypoint


def main() -> None:
    run_catalog_entrypoint(
        "Evaluate AD-only OOD generalization metrics from formal result catalogs.",
        "ood_generalization",
        [
            "ood_auroc",
            "source_holdout_auroc",
            "family_holdout_auroc",
            "regimen_holdout_auroc",
            "target_program_holdout_auroc",
            "ood_detection_auroc",
        ],
    )


if __name__ == "__main__":
    main()
