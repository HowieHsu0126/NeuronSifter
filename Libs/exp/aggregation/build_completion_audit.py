from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.paper_grade_entrypoint import run_delegate_entrypoint


def main() -> None:
    run_delegate_entrypoint(
        "Build the paper-grade suite completion audit.",
        "paper_report_generation",
        "Libs/scripts/validators/validate_paper_grade_full_suite.py",
        ["--mode", "postrun"],
    )


if __name__ == "__main__":
    main()
