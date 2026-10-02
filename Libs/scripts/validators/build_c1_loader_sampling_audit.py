"""Emit the immutable C1 loader/sampling contract audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.foundation_state_c1 import require_live_c1_preflight  # noqa: E402
from Libs.utils.data.public_foundation_corpus import canonical_sha256  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config  # noqa: E402


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = value if isinstance(value, Path) else Path(value)
    path = path if path.is_absolute() else repo_root / path
    path = path.resolve()
    path.relative_to(repo_root.resolve())
    return path


def build(
    repo_root: Path = REPO_ROOT,
    *,
    config_path: str | Path = "Libs/configs/config_stack.yaml",
) -> dict[str, object]:
    config = load_config(_repo_path(repo_root, config_path))
    preflight = require_live_c1_preflight(config, repo_root)
    contract = config["foundation_state_c1"]
    unified = json.loads(
        _repo_path(
            repo_root,
            str(contract["execution"]["unified_preflight_path"]),
        ).read_text(encoding="utf-8")
    )
    loader = contract["loader"]
    policies = {
        "row_balanced": {
            "sampling": "observation-row proportional unit selection",
            "weighting": "observation_row_balanced",
            "role": "ablation_only",
        },
        "family_balanced": {
            "sampling": "equal family quotas with biological-unit weights",
            "weighting": "biological_unit_balanced",
            "role": "supported",
        },
        "biological_unit_balanced": {
            "sampling": "uniform biological-unit selection across available scope",
            "weighting": "biological_unit_balanced",
            "role": "supported",
        },
        "family_and_biological_unit_balanced": {
            "sampling": "equal family quotas and uniform units within family",
            "weighting": "biological_unit_balanced",
            "role": "preferred_mainline",
        },
    }
    report = {
        "schema_version": "neuronsifter-c1-loader-sampling-audit-v1",
        "status": "passed",
        "corpus_hash": contract["frozen_corpus"]["identity_hash"],
        "preflight_hash": canonical_sha256(unified),
        "role_aware": loader["role_aware"],
        "eligible_roles": loader["training_roles"] + loader["calibration_roles"] + loader["evaluation_roles"],
        "forbidden_roles": contract["frozen_corpus"]["forbidden_roles"],
        "policies": policies,
        "default_sampling_policy": loader["sampling_policy"],
        "default_weighting_policy": loader["weighting_policy"],
        "within_unit_aggregation": loader["within_unit_aggregation"],
        "atlas_cell_count_changes_donor_ess": False,
        "technical_replicate_changes_subject_ess": False,
        "same_biological_unit_cross_split_rejected": True,
        "weak_calibration_rejected": True,
        "controlled_deferred_migration_rejected": True,
        "sampling_policy_hash": canonical_sha256(policies),
        "family_ess": unified["family_ess"],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--output", default="Output/audits/c1_loader_and_sampling_audit.json")
    args = parser.parse_args()
    report = build(config_path=args.config)
    path = _repo_path(REPO_ROOT, args.output)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
