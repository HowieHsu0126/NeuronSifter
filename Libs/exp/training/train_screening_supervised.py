from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.paper_grade_entrypoint import parse_standard_args
from Libs.utils.exp.paper_grade_entrypoint import run_delegate_entrypoint


def main() -> None:
    args = parse_standard_args("Run the paper-grade supervised screening entrypoint.", "screening_supervised")
    delegate_path = "Libs/exp/training/train_shared_latent_benchmark.py"
    if str(args.model_family) != "shared_latent_dual_encoder":
        delegate_path = "Libs/exp/training/train_screening_system.py"
    delegate_args = []
    if str(args.experiment_family).strip():
        delegate_args.extend(["--disease-program", "ad_control"])
    run_delegate_entrypoint(
        "Run the paper-grade supervised screening entrypoint.",
        "screening_supervised",
        delegate_path,
        delegate_args,
    )


if __name__ == "__main__":
    main()
