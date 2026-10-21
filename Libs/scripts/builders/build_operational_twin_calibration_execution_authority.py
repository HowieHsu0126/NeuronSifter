"""Build the reviewed execution authority for calibration-world-only UQ fitting."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Libs.exp.training.train_operational_twin_devcal import (  # noqa: E402
    EXPECTED_ACCESS_POLICY,
    EXPECTED_CLAIM_BOUNDARY,
)
from Libs.utils.evaluation.operational_twin_candidate_execution import (  # noqa: E402
    compute_candidate_calibration_authority_uid,
    resolve_candidate_execution_protocol,
    require_future_execution_selection,
    candidate_calibration_source_paths,
    write_json_exclusive,
    require_clean_tracked_source,
)
from Libs.utils.evaluation.operational_twin_training_protocol import (  # noqa: E402
    candidate_source_identity_hash,
    load_candidate_protocol_artifacts,
    repo_file_identity,
)
from Libs.utils.exp_utils import load_config_stack  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROUND_DIR = Path(
    "orchestrator/rounds/round-37-train-and-calibrate-operational-twin"
)
DEFAULT_OUTPUT = Path(
    "Output/audits/operational_twin_calibration_execution_authority.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_candidate_authority(
    round_dir: Path,
    *,
    root: Path,
) -> dict[str, object]:
    config = load_config_stack(root / "Libs/configs/config_stack.yaml")
    resolved = resolve_candidate_execution_protocol(config, root)
    execution = dict(resolved.execution)
    source_commit = require_clean_tracked_source(root)
    protocol = dict(resolved.protocol)
    candidate, assignments, commands, preflight = load_candidate_protocol_artifacts(
        config,
        root,
        resolved_context=resolved,
    )
    execution_selection = require_future_execution_selection(config, root, round_dir)
    record = execution_selection["round_plan_record"]
    review_path = root / execution["protocol_review_contract"]["path"]
    source_paths = [root / path for path in candidate_calibration_source_paths(config, resolved, execution_selection)]
    payload: dict[str, object] = {
        "schema_version": "neuronsifter-operational-twin-candidate-lineage-calibration-execution-authority-v2",
        "successor_execution_binding": dict(resolved.binding),
        "execution_selection": execution_selection,
        "authority_uid": None,
        "authority_role": "development_calibration_execution",
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_round_id": record["round_id"],
        "source_commit": source_commit,
        "candidate_lineage_uid": protocol["candidate_lineage_uid"],
        "protocol_uid": protocol["protocol_uid"],
        "preflight_uid": preflight["preflight_uid"],
        "candidate_manifest_hash": candidate["candidate_manifest_hash"],
        "assignment_manifest_hash": assignments["assignment_manifest_hash"],
        "command_catalog_hash": commands["command_catalog_hash"],
        "source_identity_hash": candidate_source_identity_hash(preflight),
        "approved_protocol_review_identity": repo_file_identity(root, review_path),
        "frozen_protocol_construction_flag": {
            "calibration_fitting_allowed_in_protocol_freeze_round": bool(
                protocol["scientific_contract"]["selection_and_calibration"][
                    "calibration_fitting_allowed_in_this_round"
                ]
            ),
            "interpretation": "The frozen false value prohibited fitting during protocol construction; this execution authority permits only the preregistered calibration-world fit after restoring best_dev.pt.",
        },
        "calibration_fitting_authorized": True,
        "calibration_world_only": True,
        "checkpoint_selection_role": "development_only",
        "best_development_checkpoint_restore_required": True,
        "calibration_state_initialization_allowed": False,
        "training_state_modification_allowed": False,
        "allowed_method_ids": ["full_uncertainty_aware_action_conditioned"],
        "source_files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            }
            for path in source_paths
        ],
        "access_flags": EXPECTED_ACCESS_POLICY,
        "claim_flags": EXPECTED_CLAIM_BOUNDARY,
        "sealed_material_access": 0,
        "test_outcome_access": 0,
        "sufficient_training": False,
        "twin_qualification_pass": False,
        "formal_qualification": False,
    }
    payload["authority_uid"] = compute_candidate_calibration_authority_uid(payload)
    return payload


def build_authority(
    round_dir: Path,
    *,
    protocol_scope: str = "historical-v1",
    root: Path = REPO_ROOT,
) -> dict[str, object]:
    if protocol_scope == "candidate-lineage":
        return _build_candidate_authority(round_dir, root=root)
    if protocol_scope != "historical-v1":
        raise ValueError(f"Unsupported protocol scope: {protocol_scope}")
    selection_path = round_dir / "selection-record.json"
    record_path = round_dir / "round-plan-record.json"
    plan_path = round_dir / "plan.md"
    for path in (selection_path, record_path, plan_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Round execution authority source is absent: {path}")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    plan_text = plan_path.read_text(encoding="utf-8")
    expected_identity = {
        "milestone_id": "operational-twin-qualification",
        "direction_id": "train-and-calibrate-operational-twin",
        "extracted_item_id": "operational-twin-development-training-and-calibration",
    }
    for key, expected in expected_identity.items():
        if selection.get(key) != expected or record.get(key) != expected:
            raise ValueError(f"Round calibration authority identity mismatch for {key}.")
    required_plan_statements = (
        "Calibration fitting is calibration-world-only.",
        "The uncertainty-aware action-conditioned method may fit calibration-world-only UQ parameters",
    )
    if any(statement not in plan_text for statement in required_plan_statements):
        raise ValueError("Round plan does not explicitly authorize calibration-world-only UQ fitting.")

    config = load_config_stack(root / "Libs/configs/config_stack.yaml")
    protocol = dict(config["operational_twin_training_protocol"])
    preflight_path = root / str(protocol["output_roots"]["preflight_authority"])
    manifest_path = root / str(protocol["output_roots"]["assignment_manifest"])
    protocol_path = root / "Libs/configs/model/operational_twin_training_protocol.yaml"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or preflight.get("training_executed") is not False:
        raise ValueError("Frozen protocol preflight is not the approved pre-execution authority.")

    source_paths = (
        selection_path,
        record_path,
        plan_path,
        protocol_path,
        preflight_path,
        manifest_path,
    )
    return {
        "schema_version": "neuronsifter-operational-twin-calibration-execution-authority-v1",
        "authority_uid": "operational_twin_development_calibration_execution",
        "authority_role": "development_calibration_execution",
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_round_id": record["round_id"],
        "protocol_uid": protocol["protocol_uid"],
        "frozen_protocol_construction_flag": {
            "calibration_fitting_allowed_in_protocol_freeze_round": bool(
                protocol["selection_and_calibration"]["calibration_fitting_allowed_in_this_round"]
            ),
            "interpretation": "The frozen false value prohibited fitting during protocol construction; this later execution authority permits only the preregistered calibration-world fit.",
        },
        "calibration_fitting_authorized": True,
        "calibration_world_only": True,
        "checkpoint_selection_role": "development_only",
        "allowed_method_ids": ["full_uncertainty_aware_action_conditioned"],
        "source_files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            }
            for path in source_paths
        ],
        "access_flags": EXPECTED_ACCESS_POLICY,
        "claim_flags": EXPECTED_CLAIM_BOUNDARY,
        "sealed_material_access": 0,
        "test_outcome_access": 0,
        "sufficient_training": False,
        "twin_qualification_pass": False,
        "formal_qualification": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-scope",
        choices=("historical-v1", "candidate-lineage"),
        default="historical-v1",
    )
    parser.add_argument("--round-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    candidate_mode = args.protocol_scope == "candidate-lineage"
    if candidate_mode:
        if args.round_dir is None:
            parser.error("candidate-lineage requires the genuine later --round-dir")
        config = load_config_stack(REPO_ROOT / "Libs/configs/config_stack.yaml")
        resolved = resolve_candidate_execution_protocol(config, REPO_ROOT)
        expected_output = REPO_ROOT / resolved.execution["output_paths"]["calibration_execution_authority"]
        if args.output is not None and args.output.absolute() != expected_output:
            parser.error("candidate calibration authority output must be canonical")
        output = expected_output
        round_dir = args.round_dir
    else:
        round_dir = args.round_dir or REPO_ROOT / DEFAULT_ROUND_DIR
        output = args.output or REPO_ROOT / DEFAULT_OUTPUT
    payload = build_authority(
        round_dir.resolve(),
        protocol_scope=args.protocol_scope,
    )
    if candidate_mode:
        write_json_exclusive(output, payload)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": output.relative_to(REPO_ROOT).as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
