"""Run the bounded pre-replay Round-20Q-R baseline challenge audit.

This entrypoint is intentionally narrower than a scientific C1/C2 challenge.
It records the source and adapter status of the frozen reduced registry and
re-runs the restricted C2 simple adapters on train/validation rows only.  The
restricted simulator domain is useful for integration evidence, but it is not
allowed to promote the Round-20Q-R baseline challenge or freeze thresholds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_restricted_c2_baseline_qualification import (  # noqa: E402
    build as build_restricted_c2_qualification,
)
from Libs.utils.data.raw_data_registry import write_json  # noqa: E402


DEFAULT_REGISTRY = "Libs/configs/experiments/baseline_registry.yaml"
DEFAULT_CONFIG = "Libs/configs/config_stack.yaml"
DEFAULT_OUTPUT_ROOT = "Output/audits/round20qr_minimal_challenge_v1"
DEFAULT_C2_OUTPUT = (
    "Output/audits/round20qr_minimal_challenge_v1/c2_baseline_qualification.json"
)
SCHEMA_VERSION = "neuronsifter-round20qr-minimal-challenge-v1"
ALLOWED_COMPUTE_NODES = {"cpu03", "cpu04"}
SIMPLE_C2_ADAPTERS = {"persistence", "linear_state_space", "gru"}


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _root_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _producer_identity(root: Path) -> dict[str, object]:
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "dirty": bool(_git(root, "status", "--porcelain", "--untracked-files=no")),
        "script": "Libs/scripts/builders/build_round20qr_minimal_challenge.py",
        "job_uid": os.environ.get("SLURM_JOB_ID", "manual"),
        "node": os.environ.get("SLURMD_NODENAME", "local"),
    }


def _write_tsv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _immutable_json(path: Path, payload: dict[str, object]) -> None:
    content = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"immutable artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _candidate_paths(root: Path, uid: str, entry: dict[str, object]) -> list[Path]:
    repository = str(entry.get("repository_or_package", ""))
    candidates: list[str] = []
    if uid in SIMPLE_C2_ADAPTERS:
        candidates.append("Libs/utils/evaluation/restricted_c2_baselines.py")
    if "scvi-tools" in repository:
        candidates.append("Libs/external/scvi")
    elif "scGPT" in repository or "scgpt" in repository:
        candidates.append("Libs/external/scgpt")
    elif repository.endswith("/uce"):
        candidates.append("Libs/external/uce")
    elif "neuraloperator" in repository:
        candidates.extend(
            [
                "Libs/external/neural_operator",
                "Libs/external/neural_operator/Transolver",
            ]
        )
    elif "Transolver_plus" in repository:
        candidates.append("Libs/external/neural_operator/Transolver_plus")
    elif "AI4Science-WestlakeU/wdno" in repository:
        candidates.append("Libs/external/neural_operator/wdno")
    return [_root_path(root, value) for value in candidates]


def _source_row(
    root: Path, entry: dict[str, object], c2_payload: dict[str, object]
) -> dict[str, object]:
    uid = str(entry["baseline_uid"])
    candidates = _candidate_paths(root, uid, entry)
    existing = next((path for path in candidates if path.exists()), None)
    adapter = next(
        (
            row
            for row in c2_payload.get("adapters", [])
            if str(row.get("baseline_uid")) == uid
        ),
        None,
    )
    if (
        uid in SIMPLE_C2_ADAPTERS
        and adapter
        and adapter.get("qualification_status") == "adapter_qualified"
    ):
        status = "qualified_adapter_smoke_only"
        reason = "restricted C2 train/validation adapter is qualified; formal challenge remains prohibited"
    elif not existing:
        status = "missing_source_or_adapter"
        reason = "no managed source checkout or project adapter is bound"
    elif (existing / ".git").exists():
        expected_pin = str(entry.get("tag_or_commit", ""))
        head = _git(existing, "rev-parse", "HEAD")
        dirty = bool(_git(existing, "status", "--porcelain"))
        if expected_pin in {"", "pending_source_lock", "source_pin_pending"}:
            status = "source_pin_pending"
            reason = (
                "managed checkout exists but the frozen entry has no exact source pin"
            )
        elif head != expected_pin:
            status = "source_pin_mismatch"
            reason = (
                f"managed checkout is {head or 'unresolved'}, expected {expected_pin}"
            )
        elif dirty:
            status = "source_checkout_dirty"
            reason = "managed checkout contains uncommitted changes"
        else:
            status = "source_checkout_identity_only"
            reason = "source identity is present; license/data/adapter qualification is still required"
    else:
        status = "source_checkout_unbound"
        reason = "candidate path exists without a Git-bound managed source"
    return {
        "baseline_uid": uid,
        "claim_ids": "|".join(str(value) for value in entry.get("claim_ids", [])),
        "baseline_class": str(entry.get("baseline_class", "")),
        "implementation_source": str(entry.get("implementation_source", "")),
        "repository_or_package": str(entry.get("repository_or_package", "")),
        "candidate_path": str(existing.relative_to(root))
        if existing and existing.is_relative_to(root)
        else (str(existing) if existing else ""),
        "candidate_bytes": existing.stat().st_size
        if existing and existing.is_file()
        else 0,
        "candidate_sha256": _sha256(existing)
        if existing and existing.is_file()
        else "",
        "candidate_git_commit": _git(existing, "rev-parse", "HEAD")
        if existing and (existing / ".git").exists()
        else "",
        "candidate_git_dirty": bool(_git(existing, "status", "--porcelain"))
        if existing and (existing / ".git").exists()
        else False,
        "adapter_qualification_status": str(adapter.get("qualification_status", ""))
        if adapter
        else "",
        "qualification_status": status,
        "formal_challenge_eligible": False,
        "reason": reason,
    }


def _run_c2_qualification(root: Path, output_path: Path) -> dict[str, object]:
    if output_path.exists():
        raise RuntimeError(
            f"append-only C2 qualification output already exists: {output_path}"
        )
    payload = build_restricted_c2_qualification(
        config_path=root / DEFAULT_CONFIG,
        manifest_root=root / "Input/manifests/c2",
        readiness_path=root / "Output/audits/c2_domain_readiness.json",
        integrity_path=root / "Output/audits/c2/current_clamp/integrity_audit.json",
        output_path=output_path,
        store_path=None,
        source_root=root,
    )
    write_json(output_path, payload)
    return payload


def build(
    root: Path = REPO_ROOT,
    *,
    registry_path: Path = REPO_ROOT / DEFAULT_REGISTRY,
    output_root: Path = REPO_ROOT / DEFAULT_OUTPUT_ROOT,
    c2_output: Path = REPO_ROOT / DEFAULT_C2_OUTPUT,
    enforce_compute_node: bool = True,
) -> dict[str, object]:
    hostname = os.environ.get("SLURMD_NODENAME", "").split(".", 1)[0]
    if enforce_compute_node and hostname not in ALLOWED_COMPUTE_NODES:
        raise RuntimeError("Round-20Q-R minimal challenge must run on cpu03 or cpu04")
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    reduction = registry.get("reduction_registry", {})
    entries = [dict(row) for row in reduction.get("mandatory", [])]
    if (
        len(entries) != 20
        or len({str(row.get("baseline_uid")) for row in entries}) != 20
    ):
        raise ValueError(
            "reduced mandatory registry must contain exactly 20 unique entries"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    c2_payload = _run_c2_qualification(root, c2_output)
    source_rows = [_source_row(root, entry, c2_payload) for entry in entries]
    simple_rows = [
        {
            "baseline_uid": str(adapter["baseline_uid"]),
            "claim_id": "C2",
            "split_role": "validation",
            "metric_name": "validation_mse",
            "metric_value": float(adapter["validation_mse"]),
            "prediction_sha256": str(adapter["validation_prediction_sha256"]),
            "result_status": "adapter_smoke_available",
            "final_holdout_accessed": False,
            "formal_challenge_eligible": False,
            "note": "train/validation-only restricted simulator adapter; not a formal baseline challenge",
        }
        for adapter in c2_payload.get("adapters", [])
    ]
    failures = [
        {
            "baseline_uid": row["baseline_uid"],
            "failure_class": "source_or_data_gate",
            "status": row["qualification_status"],
            "reason": row["reason"],
            "preserved": True,
        }
        for row in source_rows
        if row["qualification_status"] != "qualified_adapter_smoke_only"
    ]
    source_counts: dict[str, int] = {}
    for row in source_rows:
        key = str(row["qualification_status"])
        source_counts[key] = source_counts.get(key, 0) + 1
    gate = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked_round20qr_baseline_challenge",
        "round20qr_baseline_challenge_resolved": False,
        "simple_adapter_smoke_passed": bool(c2_payload.get("status") == "passed"),
        "simple_baseline_sanity_passed": False,
        "strong_baseline_results_available": False,
        "challenge_thresholds_frozen": False,
        "test_data_unseen": True,
        "formal_challenge_started": False,
        "headline_training_authorized": False,
        "mandatory_registry_count": len(entries),
        "mandatory_source_qualification_counts": source_counts,
        "c2_scope": c2_payload.get("claim_scope", ""),
        "c2_baseline_challenge_allowed": False,
        "blockers": [
            "c1_full_objective_and_baseline_runner_unavailable",
            "c2_restricted_simulator_scope_not_formal_challenge_domain",
            "mandatory_external_source_or_license_evidence_incomplete",
            "strong_baseline_results_unavailable",
            "challenge_thresholds_not_frozen",
        ],
        "next_action": "qualify_missing_mandatory_baseline_sources_and_run_minimal_challenge_without_test_tuning",
        "c2b_status": "blocked_insufficient_real_e2_e4",
    }
    producer = _producer_identity(root)
    root_manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_uid": "",
        "lineage_uid": output_root.name,
        "status": gate["status"],
        "producer": producer,
        "registry": {
            "path": str(registry_path.relative_to(root))
            if registry_path.is_relative_to(root)
            else str(registry_path),
            "sha256": _sha256(registry_path),
            "mandatory_count": len(entries),
        },
        "c2_qualification": {
            "path": str(c2_output.relative_to(root))
            if c2_output.is_relative_to(root)
            else str(c2_output),
            "sha256": _sha256(c2_output),
            "status": c2_payload.get("status"),
            "final_holdout_prediction_materialized": c2_payload.get(
                "split_access", {}
            ).get("final_holdout_prediction_materialized"),
        },
        "source_catalog": "mandatory_source_catalog.tsv",
        "simple_results": "simple_adapter_results.tsv",
        "failure_catalog": "failure_catalog.tsv",
        "gate": gate,
        "c2b_status": "blocked_insufficient_real_e2_e4",
    }
    root_manifest["artifact_uid"] = _canonical_sha256(
        {key: value for key, value in root_manifest.items() if key != "artifact_uid"}
    )
    fields = list(source_rows[0]) if source_rows else []
    _write_tsv(output_root / "mandatory_source_catalog.tsv", fields, source_rows)
    _write_tsv(
        output_root / "simple_adapter_results.tsv",
        list(simple_rows[0]) if simple_rows else ["baseline_uid"],
        simple_rows,
    )
    _write_tsv(
        output_root / "failure_catalog.tsv",
        list(failures[0])
        if failures
        else ["baseline_uid", "failure_class", "status", "reason", "preserved"],
        failures,
    )
    _immutable_json(output_root / "round20qr_minimal_challenge_gate.json", gate)
    _immutable_json(
        output_root / "round20qr_minimal_challenge_root_manifest.json", root_manifest
    )
    return {
        "gate": gate,
        "root_manifest": root_manifest,
        "source_rows": source_rows,
        "simple_rows": simple_rows,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fail-closed Round-20Q-R minimal baseline challenge audit."
    )
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--c2-output", default=DEFAULT_C2_OUTPUT)
    parser.add_argument("--no-compute-node-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    result = build(
        root,
        registry_path=_root_path(root, args.registry),
        output_root=_root_path(root, args.output_root),
        c2_output=_root_path(root, args.c2_output),
        enforce_compute_node=not args.no_compute_node_check,
    )
    print(
        json.dumps(
            {
                "status": result["gate"]["status"],
                "artifact_uid": result["root_manifest"]["artifact_uid"],
                "simple_adapter_smoke_passed": result["gate"][
                    "simple_adapter_smoke_passed"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
