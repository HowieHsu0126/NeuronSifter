from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.paper_grade_entrypoint import run_catalog_entrypoint


def main() -> None:
    run_catalog_entrypoint("Run or summarize structured-action ablation metrics.", "action_ablation")


if __name__ == "__main__":
    main()
