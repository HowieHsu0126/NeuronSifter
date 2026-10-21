"""Verify the bounded Operational Twin development/calibration artifacts."""

from __future__ import annotations

import argparse
from argparse import Namespace
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.exp.training.train_operational_twin_devcal import (  # noqa: E402
    EXPECTED_ACCESS_POLICY,
    EXPECTED_CLAIM_BOUNDARY,
    OperationalTwinTrainingClosedError,
    require_operational_twin_training_authorized,
    resolve_operational_twin_run_contract,
    tracked_reference_identity,
)
from Libs.models.modules.world_action_encoder import world_action_schema_hash  # noqa: E402
from Libs.utils.data.controlled_world_contracts import FAMILY_UIDS  # noqa: E402
from Libs.utils.data.controlled_world_tensorization import controlled_manifest_sha256, tensorize_manifest  # noqa: E402
from Libs.utils.evaluation.operational_twin_training_protocol import (  # noqa: E402
    REQUIRED_METHOD_IDS,
    canonical_hash,
    repo_path,
)
from Libs.utils.evaluation.operational_twin_training_evidence_ledger import (  # noqa: E402
    compute_operational_twin_training_evidence_ledger_sha256,
    validate_operational_twin_training_evidence_ledger,
)
from Libs.utils.exp_utils import load_config_stack, resolve_project_root  # noqa: E402


DEFAULT_TRAINING_EVIDENCE_LEDGER = Path(
    "Output/audits/operational_twin_training_evidence_ledger.json"
)
TRAINING_EVIDENCE_LEDGER_SCHEMA = Path(
    "Libs/configs/schema/operational_twin_training_evidence_ledger.schema.json"
)


def _append_blocker(
    blockers: list[dict[str, str]],
    code: str,
    message: str,
) -> None:
    blocker = {"code": code, "message": message}
    if blocker not in blockers:
        blockers.append(blocker)


def _repo_relative(root: Path, path: Path | str) -> str:
    candidate = Path(path)
    absolute = candidate if candidate.is_absolute() else root / candidate
    return absolute.resolve().relative_to(root.resolve()).as_posix()


def _git_head_blob(root: Path, relative_path: str) -> tuple[bytes | None, str]:
    result = subprocess.run(
        ["git", "cat-file", "blob", f"HEAD:{relative_path}"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return None, result.stderr.decode("utf-8", errors="replace").strip()
    return result.stdout, ""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_committed_training_evidence_ledger(
    *,
    root: Path,
    path: Path | str,
    blockers: list[dict[str, str]],
) -> tuple[dict[str, object], str]:
    try:
        relative_path = _repo_relative(root, path)
    except ValueError:
        _append_blocker(
            blockers,
            "training_evidence_ledger_path_invalid",
            f"Training evidence ledger path resolves outside the repository: {path}.",
        )
        return {}, str(path)

    absolute_path = root / relative_path
    committed_bytes, git_error = _git_head_blob(root, relative_path)
    if committed_bytes is None:
        code = (
            "training_evidence_ledger_untracked"
            if absolute_path.is_file()
            else "training_evidence_ledger_missing"
        )
        _append_blocker(
            blockers,
            code,
            (
                f"Training evidence ledger is not present in the Git HEAD tree: "
                f"{relative_path}. {git_error}"
            ).strip(),
        )
        return {}, relative_path

    if absolute_path.is_file():
        try:
            working_bytes = absolute_path.read_bytes()
        except OSError as error:
            _append_blocker(
                blockers,
                "training_evidence_ledger_read_failed",
                f"Training evidence ledger could not be read: {error}.",
            )
        else:
            if working_bytes != committed_bytes:
                _append_blocker(
                    blockers,
                    "training_evidence_ledger_worktree_drift",
                    (
                        "The working-tree training evidence ledger differs from "
                        f"the committed Git HEAD blob: {relative_path}."
                    ),
                )

    try:
        decoded = committed_bytes.decode("utf-8")
        value = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _append_blocker(
            blockers,
            "training_evidence_ledger_json_invalid",
            f"Committed training evidence ledger is not valid JSON: {error}.",
        )
        return {}, relative_path
    if not isinstance(value, Mapping):
        _append_blocker(
            blockers,
            "training_evidence_ledger_json_invalid",
            "Committed training evidence ledger must contain a JSON object.",
        )
        return {}, relative_path
    return dict(value), relative_path


def _schema_error_path(error: object) -> str:
    absolute_path = getattr(error, "absolute_path", ())
    rendered = ".".join(str(part) for part in absolute_path)
    return rendered or "root"


def _validate_training_evidence_ledger_schema(
    *,
    root: Path,
    ledger: Mapping[str, object],
    blockers: list[dict[str, str]],
) -> None:
    schema_path = root / TRAINING_EVIDENCE_LEDGER_SCHEMA
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        _append_blocker(
            blockers,
            "training_evidence_ledger_schema_unavailable",
            f"Training evidence ledger schema could not be loaded: {error}.",
        )
        return
    errors = sorted(
        Draft202012Validator(schema).iter_errors(ledger),
        key=lambda error: list(error.absolute_path),
    )
    for error in errors:
        _append_blocker(
            blockers,
            "training_evidence_ledger_schema_invalid",
            f"{_schema_error_path(error)}: {error.message}",
        )


def _validate_exact_protocol_identity(
    *,
    ledger: Mapping[str, object],
    protocol: Mapping[str, object],
    blockers: list[dict[str, str]],
) -> None:
    identity = ledger.get("protocol_identity")
    if not isinstance(identity, Mapping):
        return
    expected = {
        "protocol_uid": protocol.get("protocol_uid"),
        "protocol_schema_version": protocol.get("schema_version"),
        "protocol_semantic_sha256": canonical_hash(protocol),
    }
    for field, expected_value in expected.items():
        if identity.get(field) != expected_value:
            _append_blocker(
                blockers,
                f"{field}_mismatch",
                (
                    f"Training evidence ledger {field} does not match the "
                    "active frozen protocol."
                ),
            )



def _live_source_matches(
    *,
    root: Path,
    source: object,
    label: str,
    blockers: list[dict[str, str]],
) -> bool:
    if not isinstance(source, Mapping):
        return False
    path_value = source.get("path")
    if not isinstance(path_value, str) or not path_value:
        return False
    try:
        relative_path = _repo_relative(root, path_value)
    except ValueError:
        _append_blocker(
            blockers,
            "live_assignment_evidence_path_invalid",
            f"{label} resolves outside the repository: {path_value}.",
        )
        return False
    path = root / relative_path
    if not path.is_file():
        return False
    try:
        content = path.read_bytes()
    except OSError as error:
        _append_blocker(
            blockers,
            "live_assignment_evidence_read_failed",
            f"{label} could not be read: {error}.",
        )
        return True
    if (
        source.get("sha256") != _sha256_bytes(content)
        or source.get("size_bytes") != len(content)
    ):
        _append_blocker(
            blockers,
            "live_assignment_evidence_drift",
            f"{label} differs from the canonical ledger identity: {relative_path}.",
        )
    return True


def _cross_check_live_assignment_evidence(
    *,
    root: Path,
    ledger: Mapping[str, object],
    blockers: list[dict[str, str]],
) -> dict[str, object]:
    blocker_count_before = len(blockers)
    checked_paths: set[str] = set()
    present_count = 0
    assignments = ledger.get("assignments")
    if not isinstance(assignments, list):
        assignments = []
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            continue
        assignment_uid = str(assignment.get("assignment_uid", "<missing>"))
        evidence = assignment.get("evidence")
        if isinstance(evidence, Mapping):
            for role, source in evidence.items():
                if isinstance(source, Mapping) and isinstance(source.get("path"), str):
                    source_path = str(source["path"])
                    if source_path in checked_paths:
                        continue
                    checked_paths.add(source_path)
                present_count += int(
                    _live_source_matches(
                        root=root,
                        source=source,
                        label=f"{assignment_uid} {role} evidence",
                        blockers=blockers,
                    )
                )
        attempts = assignment.get("attempts")
        if isinstance(attempts, list):
            for attempt in attempts:
                if not isinstance(attempt, Mapping):
                    continue
                attempt_uid = str(attempt.get("attempt_uid", "<missing>"))
                for label, source in (
                    (f"{attempt_uid} attempt source", attempt.get("source")),
                    (
                        f"{attempt_uid} command receipt",
                        (
                            attempt.get("command_receipt", {}).get("source")
                            if isinstance(attempt.get("command_receipt"), Mapping)
                            else None
                        ),
                    ),
                ):
                    if isinstance(source, Mapping) and isinstance(source.get("path"), str):
                        source_path = str(source["path"])
                        if source_path in checked_paths:
                            continue
                        checked_paths.add(source_path)
                    present_count += int(
                        _live_source_matches(
                            root=root,
                            source=source,
                            label=label,
                            blockers=blockers,
                        )
                    )
        checkpoint = assignment.get("checkpoint")
        if isinstance(checkpoint, Mapping) and checkpoint.get("contract") == "checkpoint_identity":
            checkpoint_path = checkpoint.get("path")
            if isinstance(checkpoint_path, str) and checkpoint_path:
                checkpoint_source = {
                    "path": checkpoint_path,
                    "sha256": checkpoint.get("sha256"),
                    "size_bytes": checkpoint.get("size_bytes"),
                }
                present_count += int(
                    _live_source_matches(
                        root=root,
                        source=checkpoint_source,
                        label=f"{assignment_uid} external checkpoint",
                        blockers=blockers,
                    )
                )
    drift_count = len(blockers) - blocker_count_before
    return {
        "status": (
            "NOT_PRESENT"
            if present_count == 0
            else ("PASS" if drift_count == 0 else "BLOCKED")
        ),
        "present_source_count": present_count,
        "drift_blocker_count": drift_count,
    }


def _ledger_assignment_summary(
    ledger: Mapping[str, object],
) -> tuple[dict[str, dict[str, str]], list[dict[str, object]]]:
    matrix: dict[str, dict[str, str]] = {
        method_id: {} for method_id in REQUIRED_METHOD_IDS
    }
    records: list[dict[str, object]] = []
    assignments = ledger.get("assignments")
    if not isinstance(assignments, list):
        return matrix, records
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            continue
        assignment_uid = str(assignment.get("assignment_uid", ""))
        method_id = str(assignment.get("method_id", ""))
        seed = assignment.get("seed")
        terminal_status = str(assignment.get("terminal_status", "UNKNOWN"))
        try:
            seed_value = int(seed)
        except (TypeError, ValueError):
            seed_value = -1
        matrix.setdefault(method_id, {})[str(seed_value)] = terminal_status
        evidence = assignment.get("evidence")
        terminal_source = evidence.get("terminal") if isinstance(evidence, Mapping) else None
        terminal_path = (
            str(terminal_source.get("path", ""))
            if isinstance(terminal_source, Mapping)
            else ""
        )
        attempts = assignment.get("attempts")
        records.append(
            {
                "assignment_uid": assignment_uid,
                "method_id": method_id,
                "seed": seed_value,
                "status": terminal_status,
                "raw_status": terminal_status,
                "output_root": Path(terminal_path).parent.as_posix() if terminal_path else "",
                "attempt_count": len(attempts) if isinstance(attempts, list) else 0,
                "runtime_evidence_blockers": [],
                "budget_completion_evidence": None,
                "evaluation_status": None,
            }
        )
    return matrix, records


def _finite_walk(value: object, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_walk(child, f"{path}[{index}]")
    elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
        raise AssertionError(f"Non-finite numeric value at {path}")


def verify(*, write_artifact: bool = True) -> dict[str, object]:
    root = resolve_project_root()
    args = Namespace(
        config="Libs/configs/model/operational_twin.yaml",
        training_manifest="Input/processed/controlled_neural_world/open_action_identification/training_manifest.json",
        development_manifest="Input/processed/controlled_neural_world/open_action_identification/development_manifest.json",
        calibration_manifest="Input/processed/controlled_neural_world/open_action_identification/calibration_manifest.json",
        package_receipt="Input/processed/controlled_neural_world/open_action_identification/package_receipt.json",
        open_data_qualification="Output/audits/controlled_world_open_data_qualification.json",
        output_root="Output/results/operational_twin/development_calibration",
    )
    contract = resolve_operational_twin_run_contract(args, root, require_output=False)
    config = contract["config"]
    cfg = contract["cfg"]
    paths = contract["paths"]
    try:
        require_operational_twin_training_authorized(cfg, paths)
    except OperationalTwinTrainingClosedError as exc:
        summary = {
            "schema_version": "neuronsifter-operational-twin-artifact-verification-v1",
            "status": "BLOCKED",
            "terminal_decision": "OPEN_DATA_QUALIFICATION_PENDING",
            "reason": str(exc),
            "access_flags": EXPECTED_ACCESS_POLICY,
            "claim_flags": EXPECTED_CLAIM_BOUNDARY,
            "training_authorized": False,
            "sufficient_training": False,
            "twin_qualification_pass": False,
            "formal_qualification": False,
            "agent_work_open": False,
            "test_manifest_open": False,
        }
        return summary
    output_root = paths["output_root"]
    required = (
        "config.resolved.yaml",
        "hashes/config_hash.txt",
        "hashes/data_manifest_hashes.json",
        "hashes/seed_hash.txt",
        "hashes/world_action_schema_hash.txt",
        "checkpoints/last.pt",
        "checkpoints/best_dev.pt",
        "metrics/dev_metrics.jsonl",
        "metrics/calibration_metrics.jsonl",
        "provenance/runtime_provenance.json",
        "replay/replay_report.json",
    )
    missing = [relative for relative in required if not (output_root / relative).is_file()]
    if missing:
        raise AssertionError(f"Missing Operational Twin artifacts: {missing}")

    manifest_paths = {name: paths[name] for name in ("training", "development", "calibration")}
    manifest_hashes = json.loads((output_root / "hashes/data_manifest_hashes.json").read_text())
    if manifest_hashes != {
        name: controlled_manifest_sha256(path, name)
        for name, path in manifest_paths.items()
    }:
        raise AssertionError("Opened manifest hashes do not match canonical train/dev/cal manifests.")
    if set(manifest_hashes) != {"training", "development", "calibration"}:
        raise AssertionError("Only training, development, and calibration manifests may be hashed.")
    if (output_root / "hashes/world_action_schema_hash.txt").read_text().strip() != world_action_schema_hash():
        raise AssertionError("WorldAction schema hash is stale.")

    for role, path in manifest_paths.items():
        tensorized = tensorize_manifest(path, role, config, root)
        if {row["family_uid"] for row in tensorized["records"]} != set(FAMILY_UIDS):
            raise AssertionError(f"{role} does not cover the canonical family set.")
        for row in tensorized["records"]:
            if (row["input_mask"] & row["target_mask"]).any():
                raise AssertionError(f"Causal masks overlap for {row['world_uid']}")
            if not row["input_mask"].any() or not row["target_mask"].any():
                raise AssertionError(f"Causal masks are empty for {row['world_uid']}")

    dev_rows = [
        json.loads(line)
        for line in (output_root / "metrics/dev_metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(dev_rows) != 12:
        raise AssertionError("Development metric series must contain exactly 12 bounded epochs.")
    for row in dev_rows:
        _finite_walk(row)
    latest_dev = dev_rows[-1]
    if latest_dev["world_count"] != 72 or latest_dev["action_contrast_mean"] <= 0:
        raise AssertionError("Development world count or action contrast is invalid.")
    if "calibration_scale_by_family" in latest_dev:
        raise AssertionError("Calibration-only scaling leaked into development metrics.")

    calibration_rows = [
        json.loads(line)
        for line in (output_root / "metrics/calibration_metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(calibration_rows) != 1:
        raise AssertionError("Calibration metrics must contain exactly one bounded summary.")
    calibration = calibration_rows[0]
    _finite_walk(calibration)
    if calibration["world_count"] != 36:
        raise AssertionError("Calibration world count is invalid.")
    scales = calibration["calibration_scale_by_family"]
    if set(scales) != set(FAMILY_UIDS) or not all(float(value) > 0 for value in scales.values()):
        raise AssertionError("Calibration scales are incomplete or non-positive.")

    provenance = json.loads((output_root / "provenance/runtime_provenance.json").read_text())
    expected_receipts = {
        "open_action_package_receipt": tracked_reference_identity(root, paths["open_action_package_receipt"]),
        "open_data_qualification": tracked_reference_identity(root, paths["future_open_data_qualification"]),
    }
    if provenance.get("reference_receipt_identities") != expected_receipts:
        raise AssertionError("Receipt identities are not bound to the canonical committed Git-tree authority.")
    if provenance.get("access_flags") != EXPECTED_ACCESS_POLICY:
        raise AssertionError("Access flags are not exactly zero.")
    if provenance.get("claim_flags") != EXPECTED_CLAIM_BOUNDARY:
        raise AssertionError("Qualification or Agent-open claim flags changed.")
    device = provenance["device_provenance"]
    if device["device_type"] != "cuda" or device["cuda_available"] is not True:
        raise AssertionError("Canonical bounded training artifact is not CUDA-backed.")

    replay = json.loads((output_root / "replay/replay_report.json").read_text())
    if replay.get("status") != "PASS" or replay.get("deterministic_replay") is not True:
        raise AssertionError("Deterministic replay did not pass.")
    if replay.get("access_flags") != EXPECTED_ACCESS_POLICY:
        raise AssertionError("Replay access flags are not exactly zero.")
    if replay.get("claim_flags") != EXPECTED_CLAIM_BOUNDARY:
        raise AssertionError("Replay qualification or Agent-open claim flags changed.")

    summary = {
        "schema_version": "neuronsifter-operational-twin-artifact-verification-v1",
        "status": "PASS",
        "latest_dev_rmse": latest_dev["reconstruction_rmse"],
        "latest_dev_action_contrast_mean": latest_dev["action_contrast_mean"],
        "calibration_coverage_95": calibration["coverage_95"],
        "device": device,
        "access_flags": EXPECTED_ACCESS_POLICY,
        "claim_flags": EXPECTED_CLAIM_BOUNDARY,
        "receipt_identity_authority": "git_commit_tree_blob",
    }
    if write_artifact:
        verification_path = output_root / "verification/artifact_verification.json"
        verification_path.parent.mkdir(parents=True, exist_ok=True)
        verification_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def verify_formal_development_training(
    *,
    write_artifact: bool = True,
    training_evidence_ledger_path: Path | str = DEFAULT_TRAINING_EVIDENCE_LEDGER,
) -> dict[str, object]:
    """Verify formal training from the canonical committed evidence ledger.

    The ledger is the sole evidence authority in a clean checkout.  Live
    assignment artifacts are optional: when present, their content identities
    must agree with the ledger, but they never replace ledger evidence.
    """

    root = resolve_project_root()
    config = load_config_stack(root / "Libs/configs/config_stack.yaml")
    protocol = dict(config["operational_twin_training_protocol"])
    roots = dict(protocol["output_roots"])
    blockers: list[dict[str, str]] = []
    ledger, ledger_relative_path = _load_committed_training_evidence_ledger(
        root=root,
        path=training_evidence_ledger_path,
        blockers=blockers,
    )

    live_cross_check: dict[str, object] = {
        "status": "NOT_CHECKED",
        "present_source_count": 0,
        "drift_blocker_count": 0,
    }
    if ledger:
        _validate_training_evidence_ledger_schema(
            root=root,
            ledger=ledger,
            blockers=blockers,
        )
        expected_digest = compute_operational_twin_training_evidence_ledger_sha256(
            ledger
        )
        if ledger.get("ledger_sha256") != expected_digest:
            _append_blocker(
                blockers,
                "ledger_sha256_mismatch",
                "The ledger digest does not match its canonical content.",
            )

        reported_blockers = ledger.get("validation_blockers")
        if isinstance(reported_blockers, list):
            for blocker in reported_blockers:
                if isinstance(blocker, Mapping):
                    _append_blocker(
                        blockers,
                        str(blocker.get("code", "ledger_validation_blocker")),
                        str(
                            blocker.get(
                                "message",
                                "The ledger carries an unresolved validation blocker.",
                            )
                        ),
                    )
        try:
            semantic_blockers = validate_operational_twin_training_evidence_ledger(
                ledger,
                protocol,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            _append_blocker(
                blockers,
                "training_evidence_ledger_semantic_validation_error",
                f"Training evidence ledger semantic validation failed: {error}.",
            )
        else:
            for blocker in semantic_blockers:
                _append_blocker(
                    blockers,
                    blocker["code"],
                    blocker["message"],
                )
        _validate_exact_protocol_identity(
            ledger=ledger,
            protocol=protocol,
            blockers=blockers,
        )
        live_cross_check = _cross_check_live_assignment_evidence(
            root=root,
            ledger=ledger,
            blockers=blockers,
        )

    matrix, records = _ledger_assignment_summary(ledger)
    code_commits = ledger.get("code_commits")
    if not isinstance(code_commits, Mapping):
        code_commits = {}
    training_commits = code_commits.get("training", [])
    evaluation_commits = code_commits.get("evaluation", [])
    if not isinstance(training_commits, list):
        training_commits = []
    if not isinstance(evaluation_commits, list):
        evaluation_commits = []
    dependencies = ledger.get("dependencies")
    assignments = ledger.get("assignments")
    summary = {
        "schema_version": "neuronsifter-operational-twin-development-training-verification-v1",
        "status": "PASS" if ledger and not blockers else "BLOCKED",
        "evidence_authority": "committed_training_evidence_ledger",
        "training_evidence_ledger": {
            "path": ledger_relative_path,
            "ledger_uid": ledger.get("ledger_uid"),
            "ledger_sha256": ledger.get("ledger_sha256"),
            "ledger_status": ledger.get("status"),
            "git_head_required": True,
        },
        "assignment_count": len(assignments) if isinstance(assignments, list) else 0,
        "method_ids": list(REQUIRED_METHOD_IDS),
        "seeds": [int(seed) for seed in protocol["seeds"]],
        "terminal_matrix": matrix,
        "retained_assignments": records,
        "blockers": blockers,
        "training_code_commits": list(training_commits),
        "evaluation_code_commits": list(evaluation_commits),
        "tracked_dependency_count": (
            len(dependencies) if isinstance(dependencies, list) else 0
        ),
        "live_assignment_evidence_cross_check": live_cross_check,
        "sufficient_training": False,
        "twin_qualification_pass": False,
        "formal_qualification": False,
        "access_flags": EXPECTED_ACCESS_POLICY,
        "claim_flags": EXPECTED_CLAIM_BOUNDARY,
    }
    if write_artifact:
        path = (
            repo_path(root, roots["catalog_root"], must_exist=False)
            / "development_training_verification.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--open-only", action="store_true")
    parser.add_argument("--require-frozen-manifest", action="store_true")
    parser.add_argument("--require-all-assignments-terminal", action="store_true")
    parser.add_argument(
        "--training-evidence-ledger",
        type=Path,
        default=DEFAULT_TRAINING_EVIDENCE_LEDGER,
        help=(
            "Canonical committed Operational Twin training evidence ledger used "
            "for formal verification."
        ),
    )
    args = parser.parse_args()
    if args.require_frozen_manifest or args.require_all_assignments_terminal:
        summary = verify_formal_development_training(
            write_artifact=not args.read_only,
            training_evidence_ledger_path=args.training_evidence_ledger,
        )
    else:
        summary = verify(write_artifact=not args.read_only)
    print(json.dumps(summary, sort_keys=True))
    if summary.get("status") != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
