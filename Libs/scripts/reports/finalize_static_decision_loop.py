"""Finalize static-loop provenance without redispatching scientific tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.models.modules.static_information_policy import POLICY_UIDS  # noqa: E402
from Libs.scripts.experiments.run_static_decision_loop_campaign import (  # noqa: E402
    CalibrationState,
    EVIDENCE_BOUNDARY,
    RELIABILITY_PROXY_NAME,
    _aggregate_reliability,
    _read_tsv,
    _rel,
    _repo_path,
    materialize_round_receipt,
)
from Libs.utils.data.public_foundation_corpus import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import write_json, write_tsv  # noqa: E402


DEFAULT_CONTRACT = (
    REPO_ROOT / "Libs/configs/neuronsifter/static_decision_loop.yaml"
)


def _load_contract(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Static-loop contract must be a mapping: {path}")
    return payload


def _calibration_state(
    contract: Mapping[str, Any], report: Mapping[str, Any]
) -> CalibrationState:
    return CalibrationState(
        calibration_uid=str(report["calibration_uid"]),
        c1_scale=float(report["c1_scale"]),
        c2s_scale=float(report["c2s_scale"]),
        c1_calibration_units=tuple(report["c1_calibration_units"]),
        c1_evaluation_units=tuple(report["c1_evaluation_units"]),
        c2s_calibration_units=tuple(report["c2s_calibration_units"]),
        c2s_evaluation_units=tuple(report["c2s_evaluation_units"]),
        target_coverage=float(contract["support_calibration"]["target_interval_coverage"]),
        coverage_floor=float(contract["support_calibration"]["coverage_floor"]),
        maximum_calibration_error=float(
            contract["support_calibration"]["maximum_calibration_error"]
        ),
    )


def _paired_reliability(
    path: Path, calibration: CalibrationState
) -> dict[str, Any]:
    paired_rows = _read_tsv(path)
    if not paired_rows:
        raise ValueError(f"Scientific pre/post rows are empty: {path}")
    pre_rows = [
        {
            "domain": row["domain"],
            "metric_name": row["metric_name"],
            "metric_value": float(row["metric_pre"]),
        }
        for row in paired_rows
    ]
    post_rows = [
        {
            "domain": row["domain"],
            "metric_name": row["metric_name"],
            "metric_value": float(row["metric_post"]),
        }
        for row in paired_rows
    ]
    pre = _aggregate_reliability(pre_rows, calibration)
    post = _aggregate_reliability(post_rows, calibration)
    delta = float(post["reliability_proxy"] - pre["reliability_proxy"])
    return {
        "tri_score_available": False,
        "tri_change": delta,
        "tri_change_semantics": "calibrated_reliability_proxy_delta_not_model_tri",
        "reliability_proxy_name": RELIABILITY_PROXY_NAME,
        "reliability_proxy_pre": pre["reliability_proxy"],
        "reliability_proxy_post": post["reliability_proxy"],
        "reliability_proxy_delta": delta,
    }


def _write_derived_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Derived static-loop table is empty: {path}")
    fields = list(rows[0])
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    write_tsv(path, fields, rows)


def finalize(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    run_root: Path | None = None,
) -> dict[str, Any]:
    contract = _load_contract(contract_path)
    run_root = run_root or _repo_path(contract["outputs"]["run_root"])
    if not run_root.is_dir():
        raise FileNotFoundError(f"Static-loop run root is missing: {run_root}")
    calibration_path = _repo_path(contract["outputs"]["tri_calibration"])
    calibration_report = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration = _calibration_state(contract, calibration_report)
    receipt_records: dict[tuple[str, str], dict[str, Any]] = {}
    receipt_catalog: list[dict[str, Any]] = []
    reliability_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    source_job_ids: set[str] = set()
    expected_rounds = int(contract["static_policy"]["sequential_rounds"])
    for policy_uid in POLICY_UIDS:
        for round_index in range(1, expected_rounds + 1):
            round_uid = f"round_{round_index}"
            round_root = run_root / policy_uid / round_uid
            receipt = materialize_round_receipt(
                round_root,
                policy_uid=policy_uid,
                round_uid=round_uid,
            )
            receipt_path = round_root / "round_receipt.json"
            receipt_records[(policy_uid, round_uid)] = receipt
            receipt_catalog.append(
                {
                    "policy_uid": policy_uid,
                    "round_uid": round_uid,
                    "receipt_uid": receipt["receipt_uid"],
                    "receipt_ref": _rel(receipt_path),
                    "receipt_sha256": sha256_file(receipt_path),
                    "freeze_before_dispatch_verified": receipt[
                        "freeze_before_dispatch_verified"
                    ],
                    "controller_timestamp_source": receipt[
                        "controller_timestamp_source"
                    ],
                }
            )
            for binding in receipt["dispatch_bindings"]:
                raw = json.loads(
                    _repo_path(binding["dispatch_ref"]).read_text(encoding="utf-8")
                )
                if raw.get("slurm_job_id"):
                    source_job_ids.add(str(raw["slurm_job_id"]))
            reliability_pairs[(policy_uid, round_uid)] = _paired_reliability(
                round_root / f"{round_uid}_scientific_pre_post.tsv",
                calibration,
            )

    prospective_path = _repo_path(contract["outputs"]["prospective_manifest"])
    prospective_rows = _read_tsv(prospective_path)
    if len(prospective_rows) != len(receipt_records):
        raise ValueError("Prospective manifest does not cover every receipt-bound round")
    for row in prospective_rows:
        key = (str(row["policy_uid"]), str(row["round_uid"]))
        receipt = receipt_records.get(key)
        if receipt is None:
            raise ValueError(f"Prospective row has no receipt: {key}")
        receipt_path = run_root / key[0] / key[1] / "round_receipt.json"
        row.update(
            {
                "outcome_visible_before_freeze": False,
                "selection_before_outcome": True,
                "freeze_before_dispatch_verified": receipt[
                    "freeze_before_dispatch_verified"
                ],
                "input_manifest_ref": receipt["artifact_bindings"]["input_manifest"][
                    "path"
                ],
                "input_manifest_sha256": receipt["artifact_bindings"][
                    "input_manifest"
                ]["sha256"],
                "selected_action_ref": receipt["artifact_bindings"][
                    "selected_information_action"
                ]["path"],
                "selected_action_sha256": receipt["artifact_bindings"][
                    "selected_information_action"
                ]["sha256"],
                "round_receipt_ref": _rel(receipt_path),
                "round_receipt_sha256": sha256_file(receipt_path),
                "round_receipt_uid": receipt["receipt_uid"],
            }
        )
    _write_derived_tsv(prospective_path, prospective_rows)

    effects_path = _repo_path(contract["outputs"]["information_effects"])
    effect_rows = _read_tsv(effects_path)
    if len(effect_rows) != len(reliability_pairs):
        raise ValueError("Information-effect rows do not cover every scientific pair")
    for row in effect_rows:
        key = (str(row["policy_uid"]), str(row["round_uid"]))
        if key not in reliability_pairs:
            raise ValueError(f"Information-effect row lacks paired metrics: {key}")
        row.update(reliability_pairs[key])
    _write_derived_tsv(effects_path, effect_rows)

    architecture_path = _repo_path(contract["outputs"]["architecture_gate"])
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    bound_read_only_paths = {
        "posterior_lineage": _repo_path(contract["outputs"]["posterior_lineage"]),
        "tool_catalog": _repo_path(contract["outputs"]["tool_catalog"]),
        "decision_loop": _repo_path(contract["outputs"]["decision_loop"]),
        "architecture_gate": architecture_path,
    }
    catalog = {
        "schema_version": "neuronsifter-static-loop-recovery-catalog-v1",
        "status": "recovered_without_tool_redispatch",
        "scientific_terminal_state": architecture["status"],
        "round_receipt_count": len(receipt_catalog),
        "round_receipts": receipt_catalog,
        "source_slurm_job_ids": sorted(source_job_ids),
        "registered_tool_redispatch_executed": False,
        "raw_tool_outputs_operation": "read_only_hash_binding",
        "posterior_lineage_operation": "read_only_hash_binding",
        "scientific_decision_operation": "read_only_hash_binding",
        "bound_read_only_artifacts": {
            name: {"path": _rel(path), "sha256": sha256_file(path)}
            for name, path in bound_read_only_paths.items()
        },
        "derived_artifacts_updated": {
            "prospective_manifest": {
                "path": _rel(prospective_path),
                "sha256": sha256_file(prospective_path),
            },
            "information_effects": {
                "path": _rel(effects_path),
                "sha256": sha256_file(effects_path),
            },
        },
        "tri_measurement": {
            "model_tri_score_available": False,
            "reported_delta": "calibrated_reliability_proxy_delta",
            "reliability_proxy_name": RELIABILITY_PROXY_NAME,
        },
        **EVIDENCE_BOUNDARY,
    }
    recovery_path = _repo_path(contract["outputs"]["recovery_catalog"])
    write_json(recovery_path, catalog)
    return {
        "status": "passed",
        "scientific_terminal_state": architecture["status"],
        "round_receipt_count": len(receipt_catalog),
        "registered_tool_redispatch_executed": False,
        "recovery_catalog": _rel(recovery_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--run-root")
    args = parser.parse_args()
    result = finalize(
        Path(args.contract).resolve(),
        run_root=Path(args.run_root).resolve() if args.run_root else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
