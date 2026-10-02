"""Resolve the Round-20Q forensic route and freeze a prospective Round-20Q-R.

This builder is deliberately fail-closed.  It searches only the historical
roots registered by the Round-22 contract, never treats post-cutover files as
historical authority, and creates a prospective contract only from existing
hash-bound public/simulator artifacts.  It does not train, access holdouts, or
authorize Round-21R when the baseline challenge is unresolved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_round22_agent_twin_coevolution_readiness import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_SCHEMA,
    canonical_sha256,
    load_contract,
)
from Libs.scripts.builders.build_round22a_authority_transition import (  # noqa: E402
    ROUND20Q_COMPONENTS,
)
from Libs.scripts.builders.build_round22r_empirical_prerequisites import (  # noqa: E402
    C2B_BLOCKED,
)


ALLOWED_COMPUTE_NODES = {"cpu03", "cpu04"}
CUTOVER_COMMIT = "0df2deea6e17e8ce8b41d818d078cba389879374"
CUTOVER_TREE = "910b45e50c002bcfc17bf9b5bda07f1fa34e086b"
ROUND20Q_NAME = re.compile(r"round[-_ ]?20q(?:r)?", re.IGNORECASE)
COMPONENT_NAME = re.compile(
    r"(baseline[_-]?frontier|challenge[_-]?threshold|training[_-]?authoriz|"
    r"seed[_-]?protocol|split[_-]?(?:freeze|manifest)|data[_-]?(?:freeze|manifest))",
    re.IGNORECASE,
)
MACHINE_SUFFIXES = {".json", ".yaml", ".yml", ".tsv"}
MAX_HASH_BYTES = 16 * 1024 * 1024

SOURCE_ARTIFACTS = {
    "allen_materialization": "Output/audits/round20m_allen_materialization_status.json",
    "round20m_gate": "Output/audits/round20m_gate.json",
    "c1_inventory": "Output/audits/c1_repository_inventory.json",
    "c1_preflight": "Output/audits/c1_foundation_corpus_preflight.json",
    "c1_task_registry": "Output/audits/c1_task_registry.json",
    "c1_splits": "Output/audits/public_foundation_corpus_frozen_splits.json",
    "c1_split_manifest": "Output/audits/public_foundation_corpus_split_manifest.json",
    "c2_domain": "Output/audits/c2_domain_readiness.json",
    "c2_trajectory_authority": "Output/audits/c2_trajectory_authority.json",
    "c2_restricted_baseline_qualification": "Output/audits/c2/current_clamp/baseline_qualification.json",
    "baseline_reduction_contract": "Output/audits/baseline_registry_reduction_contract.json",
    "baseline_reduction_validation": "Output/audits/baseline_registry_reduction_validation.json",
    "baseline_reduction_gate": "Output/audits/baseline_registry_reduction_gate.json",
    "prequalification_registry": "Output/audits/round20qr_prequalification/baseline_registry_freeze.json",
    "prequalification_sanity": "Output/audits/round20qr_prequalification/baseline_task_sanity.json",
    "prequalification_thresholds": "Output/audits/round20qr_prequalification/neuronsifter_challenge_thresholds.json",
    "prequalification_authorization": "Output/audits/round20qr_prequalification/neuronsifter_headline_training_authorization.json",
    "prequalification_compatibility": "Output/audits/round20qr_prequalification/baseline_task_compatibility.json",
    "prequalification_upstream": "Output/audits/round20qr_prequalification/baseline_upstream_reproduction_summary.json",
}

OUTPUTS = {
    "component_inventory": "Output/audits/round22b_round20q_component_inventory.tsv",
    "producer_lineage": "Output/audits/round22b_round20q_producer_lineage.tsv",
    "frozen_contract": "Output/audits/round22b_round20q_frozen_contract.json",
    "authority_resolution": "Output/audits/round22b_round20q_authority_resolution.json",
    "training_input": "Output/audits/round20qr_training_input_freeze.json",
    "baseline_frontier": "Output/audits/round20qr_baseline_frontier.json",
    "challenge_thresholds": "Output/audits/round20qr_challenge_thresholds.json",
    "training_authorization": "Output/audits/round20qr_training_authorization.json",
    "gate": "Output/audits/round20qr_gate.json",
    "replay_authorization": "Output/audits/round22b_round21r_replay_authorization.json",
}


def _lineage_outputs(tag: str) -> tuple[dict[str, str], str, str]:
    normalized = tag.strip().lower()
    if not normalized:
        return dict(OUTPUTS), "Output/audits/round20qr_contract", "round20qr_attempt_1"
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", normalized):
        raise ValueError("invalid Round-20Q-R lineage tag")
    outputs = {}
    for key, value in OUTPUTS.items():
        path = Path(value)
        outputs[key] = (
            path.parent / f"{path.stem}_{normalized}{path.suffix}"
        ).as_posix()
    return (
        outputs,
        f"Output/audits/round20qr_contract_{normalized}",
        f"round20qr_{normalized}",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(payload))


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode()
    if path.is_file():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _tsv_bytes(fields: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(fields),
        delimiter="\t",
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in writer.fieldnames})
    return stream.getvalue().encode("utf-8")


def _immutable_tsv(
    path: Path, fields: Iterable[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    content = _tsv_bytes(fields, rows)
    if path.is_file():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _git(
    root: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result


def _git_dirty(root: Path) -> bool:
    return bool(
        _git(root, "status", "--porcelain", "--untracked-files=no").stdout.strip()
    )


def _candidate_name(name: str) -> bool:
    return bool(ROUND20Q_NAME.search(name) or COMPONENT_NAME.search(name))


def _scan_filesystem(root: Path, root_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    base = (root / str(root_spec["path"])).resolve()
    if not base.is_dir():
        return []
    max_depth = int(root_spec.get("max_depth", 0))
    max_entries = int(root_spec["max_entries"])
    visited = 0
    found: list[dict[str, Any]] = []
    stack: list[tuple[Path, int]] = [(base, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise RuntimeError(
                f"registered root read failed: {directory}: {exc}"
            ) from exc
        for entry in entries:
            visited += 1
            if visited > max_entries:
                raise RuntimeError(f"registered root entry cap exceeded: {base}")
            if entry.is_symlink():
                continue
            path = Path(entry.path)
            if entry.is_file(follow_symlinks=False) and _candidate_name(entry.name):
                size = path.stat().st_size
                found.append(
                    {
                        "root_id": root_spec["root_id"],
                        "path": path.relative_to(root).as_posix(),
                        "bytes": size,
                        "sha256": _sha256(path) if size <= MAX_HASH_BYTES else "",
                        "source_class": root_spec["source_classes"][0],
                        "cutover_relation": "invalid_post_cutover_only",
                        "status": "candidate_not_authority",
                    }
                )
            elif entry.is_dir(follow_symlinks=False) and depth < max_depth:
                stack.append((path, depth + 1))
    return found


def _git_path_candidates(root: Path) -> list[dict[str, Any]]:
    names: list[dict[str, Any]] = []
    commits = set()
    for command in (
        ("rev-list", "--all", "--reflog", "--objects"),
        ("fsck", "--full", "--unreachable", "--no-reflogs", "--no-progress"),
    ):
        result = _git(root, *command, check=False)
        text = result.stdout + "\n" + result.stderr
        for line in text.splitlines():
            fields = line.split()
            if (
                command[0] == "rev-list"
                and len(fields) >= 2
                and _candidate_name(fields[-1])
            ):
                path = " ".join(fields[1:])
                commits_for_path = []
                if ROUND20Q_NAME.search(Path(path).name):
                    commits_for_path = _git(
                        root,
                        "log",
                        "--all",
                        "--reflog",
                        "--format=%H",
                        "--",
                        path,
                        check=False,
                    ).stdout.splitlines()
                if commits_for_path:
                    for commit in sorted(set(commits_for_path)):
                        pre_cutover = (
                            _git(
                                root,
                                "merge-base",
                                "--is-ancestor",
                                commit,
                                CUTOVER_COMMIT,
                                check=False,
                            ).returncode
                            == 0
                        )
                        names.append(
                            {
                                "root_id": "git_object_database",
                                "path": f"git:{commit}:{path}",
                                "source_class": "git_object_database",
                                "cutover_relation": "historical_pre_cutover"
                                if pre_cutover
                                else "invalid_post_cutover_only",
                                "status": "candidate_requires_manifest_adjudication",
                            }
                        )
                else:
                    names.append(
                        {
                            "root_id": "git_object_database",
                            "path": f"git-object:{fields[0]}:{path}",
                            "source_class": "git_object_database",
                            "cutover_relation": "unresolved_git_path_candidate",
                            "status": "candidate_requires_commit_adjudication",
                        }
                    )
            if (
                len(fields) >= 3
                and fields[0] in {"unreachable", "dangling"}
                and fields[1] == "commit"
            ):
                commits.add(fields[2])
    for commit in sorted(commits):
        tree = _git(root, "ls-tree", "-r", "--name-only", commit, check=False)
        for line in tree.stdout.splitlines():
            path = line.strip()
            if path and _candidate_name(Path(path).name):
                pre_cutover = (
                    _git(
                        root,
                        "merge-base",
                        "--is-ancestor",
                        commit,
                        CUTOVER_COMMIT,
                        check=False,
                    ).returncode
                    == 0
                )
                names.append(
                    {
                        "root_id": "git_object_database",
                        "path": f"git-unreachable:{commit}:{path}",
                        "source_class": "git_unreachable_commit",
                        "cutover_relation": "historical_pre_cutover"
                        if pre_cutover
                        else "invalid_post_cutover_only",
                        "status": "candidate_requires_manifest_adjudication",
                    }
                )
    return names


def _source_record(root: Path, role: str, path_text: str) -> dict[str, Any]:
    path = root / path_text
    if not path.is_file():
        return {"role": role, "path": path_text, "present": False}
    return {
        "role": role,
        "path": path_text,
        "present": True,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_json(root: Path, path_text: str) -> dict[str, Any]:
    path = root / path_text
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_prospective_inputs(
    allen: Mapping[str, Any],
    c1_preflight: Mapping[str, Any],
    c1_task: Mapping[str, Any],
    c1_splits: Mapping[str, Any],
    c2_domain: Mapping[str, Any],
    c2_baseline: Mapping[str, Any],
    reduction_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the scientific identities used by the prospective root.

    This is a provenance check only. It does not qualify the baseline challenge
    and deliberately accepts the restricted C2 adapter artifact only in its
    train/validation smoke scope.
    """
    checks = {
        "allen_materialization_complete": allen.get("status") == "complete",
        "c1_preflight_passed": c1_preflight.get("status") == "passed",
        "c1_task_registry_frozen": c1_task.get("status") == "frozen",
        "c1_splits_passed": c1_splits.get("status") == "passed",
        "c1_corpus_hash_present": bool(str(c1_preflight.get("corpus_hash", ""))),
        "c2_scope_restricted": c2_domain.get("status")
        == "simulator_grounded_c2_restricted"
        and c2_domain.get("claim_scope") == "simulator_grounded_c2_restricted",
        "c2_action_complete": c2_domain.get("action_complete") is True,
        "c2_controls_complete": c2_domain.get("matched_controls_complete") is True,
        "c2_independent_units_present": int(
            c2_domain.get("independent_trajectory_unit_count", 0)
        )
        > 0,
        "c2_parameter_provenance_complete": c2_domain.get(
            "parameter_provenance_complete"
        )
        is True,
        "c2_strict_split_clean": c2_domain.get("strict_split_overlap") == 0,
        "c2_final_holdout_unseen": c2_domain.get("final_holdout_unseen") is True,
        "c2_adapter_qualification_passed": c2_baseline.get("status") == "passed",
        "c2_adapter_scope_train_validation_only": c2_baseline.get("qualification_scope")
        == "train_validation_adapter_smoke_only",
        "c2_adapter_did_not_access_holdout": c2_baseline.get("split_access", {}).get(
            "final_holdout_prediction_materialized"
        )
        is False
        and c2_baseline.get("split_access", {}).get("final_holdout_metrics_computed")
        is False,
        "c2_adapter_did_not_start_formal_challenge": c2_baseline.get(
            "formal_challenge_started"
        )
        is False,
        "baseline_reduction_validated": reduction_validation.get("status") == "passed",
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not blockers else "blocked_prospective_input_identity",
        "checks": checks,
        "blockers": blockers,
        "formal_baseline_challenge_authorized": False,
    }


def _with_uid(payload: dict[str, Any]) -> dict[str, Any]:
    without = dict(payload)
    without.pop("artifact_uid", None)
    payload["artifact_uid"] = _canonical_hash(without)
    return payload


def _build_historical_resolution(
    root: Path,
    contract: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    current_git_identity: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    configured = contract["freeze"]["authority_recovery"]
    manifest_registration = configured.get("round20q_replay_prerequisite")
    manifest_candidates = [
        row for row in candidates if ROUND20Q_NAME.search(str(row.get("path", "")))
    ]
    pre_cutover = [
        row
        for row in manifest_candidates
        if row.get("cutover_relation") == "historical_pre_cutover"
    ]
    if manifest_registration is not None:
        raise RuntimeError(
            "A Round-20Q manifest is already registered; validate it with the Round-22A v3 transition builder before choosing Route B."
        )
    if pre_cutover:
        raise RuntimeError(
            "Pre-cutover Round-20Q path candidates require explicit registration and adjudication; prospective fallback is not yet allowed."
        )
    component_rows = []
    for component in ROUND20Q_COMPONENTS:
        component_rows.append(
            {
                "component_uid": component,
                "resolution_class": "missing_historical_component",
                "canonical_path": "",
                "byte_size": "",
                "sha256": "",
                "schema_version": "",
                "producer_commit": "",
                "producer_tree": "",
                "producer_script": "",
                "creation_timestamp": "",
                "cutover_relation": "missing",
                "source_evidence": "round22a_v3_round20q_registration_absent",
                "notes": "No legally admissible pre-cutover component was found.",
            }
        )
    component_rows.insert(
        0,
        {
            "component_uid": "round20q_root_manifest",
            "resolution_class": "missing_historical_component",
            "canonical_path": "",
            "byte_size": "",
            "sha256": "",
            "schema_version": "",
            "producer_commit": "",
            "producer_tree": "",
            "producer_script": "",
            "creation_timestamp": "",
            "cutover_relation": "missing",
            "source_evidence": "round22a_v3_round20q_registration_absent",
            "notes": "A candidate filename is not authority without a registered manifest and Git lineage.",
        },
    )
    contract_payload = _with_uid(
        {
            "schema_version": "neuronsifter-round22b-round20q-frozen-contract-v1",
            "round_id": "neuronsifter_round22b",
            "authority_generation": "unresolved_historical",
            "resolution_class": "missing_historical_component",
            "root_manifest": {
                "present": bool(manifest_registration),
                "validated": False,
                "historical_byte_identity": False,
                "semantic_identity_proven": False,
            },
            "required_components": {
                name: {
                    "present": False,
                    "validated": False,
                    "resolution_class": "missing_historical_component",
                }
                for name in ROUND20Q_COMPONENTS
            },
            "producer_lineage_complete": False,
            "all_required_authority_predates_cutover": False,
            "cross_component_conflicts": 0,
            "historical_replay_root_resolved": False,
            "prospective_round20qr_required": True,
            "registered_manifest": manifest_registration or None,
            "candidate_manifest_count": len(manifest_candidates),
            "pre_cutover_candidate_count": len(pre_cutover),
            "c2b_status": C2B_BLOCKED,
        }
    )
    lineage_rows = [
        {
            "lineage_uid": "round22a_cutover",
            "role": "recovery_cutover",
            "commit": CUTOVER_COMMIT,
            "tree": CUTOVER_TREE,
            "dirty": "false",
            "source_class": "historical_control_plane",
            "cutover_relation": "cutover_boundary",
            "status": "verified",
            "evidence_path": "Output/audits/round22a_authority_recovery_resolution_v3.json",
            "evidence_sha256": _sha256(
                root / "Output/audits/round22a_authority_recovery_resolution_v3.json"
            ),
            "notes": "Cutover identity only; not Round-20Q scientific producer lineage.",
        },
        {
            "lineage_uid": "round22b_recovery_machinery",
            "role": "current_recovery_producer",
            "commit": current_git_identity["commit"],
            "tree": current_git_identity["tree"],
            "dirty": str(bool(current_git_identity["dirty"])).lower(),
            "source_class": "current_recovery_machinery",
            "cutover_relation": "post_cutover",
            "status": "invalid_post_cutover_only",
            "evidence_path": "Output/audits/round22b_recovery_machinery_identity.json",
            "evidence_sha256": _sha256(
                root / "Output/audits/round22b_recovery_machinery_identity.json"
            ),
            "notes": "Cannot satisfy historical Round-20Q producer lineage.",
        },
        {
            "lineage_uid": "round20q_historical_producer",
            "role": "round20q_manifest_producer",
            "commit": "",
            "tree": "",
            "dirty": "",
            "source_class": "historical_round20q",
            "cutover_relation": "missing",
            "status": "missing_historical_component",
            "evidence_path": "Output/audits/round22b_round20q_component_inventory.tsv",
            "evidence_sha256": "",
            "notes": "No pre-cutover root manifest or producer binding exists.",
        },
    ]
    resolution = _with_uid(
        {
            "schema_version": "neuronsifter-round22b-round20q-authority-resolution-v1",
            "round_id": "neuronsifter_round22b",
            "status": "historical_round20q_unresolved",
            "historical_round20q_replay_root_resolved": False,
            "prospective_round20qr_required": True,
            "registered_search_roots_exhausted": True,
            "candidate_inventory_frozen": True,
            "manual_narrative_used_as_authority": False,
            "unbounded_filesystem_scan": False,
            "historical_manifests_rewritten": False,
            "manifest_candidate_count": len(manifest_candidates),
            "pre_cutover_manifest_candidate_count": len(pre_cutover),
            "round22a_v3_resolution": {
                "path": "Output/audits/round22a_authority_recovery_resolution_v3.json",
                "sha256": _sha256(
                    root
                    / "Output/audits/round22a_authority_recovery_resolution_v3.json"
                ),
            },
            "round21_search_evidence": {
                "path": "Output/audits/round22a_round21_recovery_search.json",
                "sha256": _sha256(
                    root / "Output/audits/round22a_round21_recovery_search.json"
                ),
                "registered_roots_exhausted": True,
            },
            "next_action": "construct_round20qr_from_verified_public_scientific_inputs",
            "c2b_status": C2B_BLOCKED,
        }
    )
    return contract_payload, resolution, component_rows, lineage_rows


def _source_inventory(root: Path) -> list[dict[str, Any]]:
    return [_source_record(root, role, path) for role, path in SOURCE_ARTIFACTS.items()]


def build(
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_SCHEMA,
    root: Path = REPO_ROOT,
    *,
    lineage_tag: str = "",
    write_outputs: bool = True,
    enforce_compute_node: bool = True,
) -> dict[str, Any]:
    hostname = socket.gethostname().split(".", 1)[0]
    if enforce_compute_node and hostname not in ALLOWED_COMPUTE_NODES:
        raise RuntimeError("Round-20Q-R qualification must run on cpu03 or cpu04")
    contract = load_contract(config_path, schema_path)
    outputs, contract_dir_text, lineage_uid = _lineage_outputs(lineage_tag)
    current_git_identity = {
        "commit": _git(root, "rev-parse", "HEAD").stdout.strip(),
        "tree": _git(root, "rev-parse", "HEAD^{tree}").stdout.strip(),
        "dirty": _git_dirty(root),
    }
    candidates = _git_path_candidates(root)
    for spec in contract["freeze"]["authority_recovery"]["historical_roots"]:
        if spec["root_id"] in {"git_object_database", "registered_git_worktrees"}:
            continue
        candidates.extend(_scan_filesystem(root, spec))
    frozen_contract, resolution, component_rows, lineage_rows = (
        _build_historical_resolution(
            root,
            contract,
            candidates,
            current_git_identity,
        )
    )

    source_rows = _source_inventory(root)
    source_map = {row["role"]: row for row in source_rows}
    pre_registry = _load_json(root, SOURCE_ARTIFACTS["prequalification_registry"])
    pre_sanity = _load_json(root, SOURCE_ARTIFACTS["prequalification_sanity"])
    pre_thresholds = _load_json(root, SOURCE_ARTIFACTS["prequalification_thresholds"])
    pre_auth = _load_json(root, SOURCE_ARTIFACTS["prequalification_authorization"])
    c1_splits = _load_json(root, SOURCE_ARTIFACTS["c1_splits"])
    c1_task = _load_json(root, SOURCE_ARTIFACTS["c1_task_registry"])
    c2_domain = _load_json(root, SOURCE_ARTIFACTS["c2_domain"])
    allen = _load_json(root, SOURCE_ARTIFACTS["allen_materialization"])
    baseline_reduction = _load_json(
        root, SOURCE_ARTIFACTS["baseline_reduction_contract"]
    )
    c2_baseline = _load_json(
        root, SOURCE_ARTIFACTS["c2_restricted_baseline_qualification"]
    )
    reduction_validation = _load_json(
        root, SOURCE_ARTIFACTS["baseline_reduction_validation"]
    )
    prospective_input_validation = _validate_prospective_inputs(
        allen,
        _load_json(root, SOURCE_ARTIFACTS["c1_preflight"]),
        c1_task,
        _load_json(root, SOURCE_ARTIFACTS["c1_splits"]),
        c2_domain,
        c2_baseline,
        reduction_validation,
    )

    training_input = _with_uid(
        {
            "schema_version": "neuronsifter-round20qr-training-input-freeze-v1",
            "round_id": "round20qr",
            "lineage_uid": lineage_uid,
            "authority_generation": "prospective_round20qr",
            "status": "frozen_inputs_blocked_baseline_challenge",
            "public_only": True,
            "allen_family_count": int(allen.get("allen", {}).get("configured", 0)),
            "allen_materialization_status": allen.get("status"),
            "c1_corpus_hash": _load_json(root, SOURCE_ARTIFACTS["c1_preflight"]).get(
                "corpus_hash"
            ),
            "c1_split_manifest": {
                "path": SOURCE_ARTIFACTS["c1_split_manifest"],
                "sha256": source_map["c1_split_manifest"]["sha256"],
                "frozen": True,
            },
            "c1_split_count": len(c1_splits.get("manifests", [])),
            "c2s_domain": c2_domain.get("selected_domain"),
            "c2s_scope": c2_domain.get("claim_scope"),
            "c2s_final_holdout_unseen": bool(c2_domain.get("final_holdout_unseen")),
            "baseline_registry_target": baseline_reduction.get("target_cardinality"),
            "checkpoint_selection": c1_task.get("checkpoint_selection"),
            "statistical_contract": c1_task.get("bootstrap"),
            "prospective_input_validation": prospective_input_validation,
            "test_tuning_count": 0,
            "final_hidden_outcomes_used_for_training": False,
            "source_artifacts": source_rows,
            "training_authorization": False,
            "blockers": [
                "round20qr_baseline_challenge_unresolved",
                "challenge_thresholds_not_frozen",
                "headline_training_not_authorized",
            ]
            + [
                f"prospective_input:{blocker}"
                for blocker in prospective_input_validation["blockers"]
            ],
            "c2b_status": C2B_BLOCKED,
        }
    )
    frontier = _with_uid(
        {
            "schema_version": "neuronsifter-round20qr-baseline-frontier-v1",
            "round_id": "round20qr",
            "lineage_uid": lineage_uid,
            "authority_generation": "prospective_round20qr",
            "status": "blocked_baseline_challenge",
            "registry_status": pre_registry.get("status"),
            "registry_sha256": source_map["prequalification_registry"]["sha256"],
            "target_cardinality": baseline_reduction.get("target_cardinality"),
            "mandatory_qualified": 0,
            "mandatory_required": 20,
            "simple_baseline_sanity_passed": bool(
                pre_sanity.get("simple_baseline_sanity_passed")
            ),
            "strong_baseline_results_available": False,
            "baseline_frontier_frozen": False,
            "qualification_scope": "public_c1_and_simulator_grounded_c2s_restricted",
            "blockers": [
                "mandatory_baseline_sources_locked",
                "simple_baseline_sanity_not_passed",
                "strong_baseline_results_unavailable",
            ],
            "source_artifacts": [
                source_map["prequalification_registry"],
                source_map["prequalification_sanity"],
                source_map["baseline_reduction_contract"],
            ],
            "c2b_status": C2B_BLOCKED,
        }
    )
    thresholds = _with_uid(
        {
            "schema_version": "neuronsifter-round20qr-challenge-thresholds-v1",
            "round_id": "round20qr",
            "lineage_uid": lineage_uid,
            "authority_generation": "prospective_round20qr",
            "status": "not_frozen_no_strong_baseline_results",
            "challenge_thresholds_frozen": False,
            "source_prequalification_status": pre_thresholds.get("status"),
            "blockers": pre_thresholds.get(
                "blockers",
                [
                    "strong_baseline_results_unavailable",
                    "simple_baseline_sanity_not_passed",
                ],
            ),
            "primary_metric": "not_frozen",
            "secondary_metrics": [],
            "minimum_meaningful_delta": "not_frozen",
            "coverage_floor": "not_frozen",
            "freeze_before_neuronsifter_test": True,
            "source_artifact": source_map["prequalification_thresholds"],
            "c2b_status": C2B_BLOCKED,
        }
    )
    authorization = _with_uid(
        {
            "schema_version": "neuronsifter-round20qr-training-authorization-v1",
            "round_id": "round20qr",
            "lineage_uid": lineage_uid,
            "authority_generation": "prospective_round20qr",
            "status": "blocked_baseline_source",
            "baseline_challenge_suite_resolved": False,
            "headline_training_authorized": False,
            "training_authorized": False,
            "source_prequalification_status": pre_auth.get("status"),
            "blockers": pre_auth.get("blockers", []),
            "allowed_pre_gate_work": pre_auth.get("allowed_pre_gate_work", []),
            "source_artifact": source_map["prequalification_authorization"],
            "c2b_status": C2B_BLOCKED,
        }
    )

    config_records = [
        _source_record(
            root,
            "baseline_registry_config",
            "Libs/configs/experiments/baseline_registry.yaml",
        ),
        _source_record(
            root,
            "round22_contract_config",
            "Libs/configs/experiments/round22_agent_twin_coevolution.yaml",
        ),
    ]
    component_payloads: dict[str, dict[str, Any]] = {
        "data": {
            "schema_version": "neuronsifter-round20q-data-freeze-v1",
            "round_id": "round20qr",
            "component_id": "data",
            "status": "frozen",
            "frozen": True,
            "public_only": True,
            "data_identity_hash": _canonical_hash(
                {
                    "sources": source_rows[:10],
                    "allen_family_count": training_input["allen_family_count"],
                    "c1_corpus_hash": training_input["c1_corpus_hash"],
                }
            ),
            "source_artifacts": source_rows[:10],
        },
        "splits": {
            "schema_version": "neuronsifter-round20q-split-freeze-v1",
            "round_id": "round20qr",
            "component_id": "splits",
            "status": "frozen",
            "frozen": True,
            "validation_only_checkpoint_selection": True,
            "final_holdout_unseen": True,
            "split_identity_hash": _canonical_hash(
                {
                    "c1_splits": c1_splits,
                    "c1_split_manifest": source_map["c1_split_manifest"],
                    "c2s_domain": c2_domain.get("selected_domain"),
                }
            ),
            "source_artifacts": [
                source_map["c1_splits"],
                source_map["c1_split_manifest"],
                source_map["c2_domain"],
            ],
        },
        "baseline_frontier": {
            "schema_version": "neuronsifter-round20q-baseline-frontier-freeze-v1",
            "round_id": "round20qr",
            "component_id": "baseline_frontier",
            "status": "blocked_baseline_challenge",
            "frozen": False,
            "baseline_frontier_frozen": False,
            "baseline_frontier_hash": _canonical_hash(frontier),
            "source_artifact": outputs["baseline_frontier"],
        },
        "challenge_thresholds": {
            "schema_version": "neuronsifter-round20q-challenge-threshold-freeze-v1",
            "round_id": "round20qr",
            "component_id": "challenge_thresholds",
            "status": "blocked_baseline_challenge",
            "frozen": False,
            "challenge_thresholds_frozen": False,
            "challenge_thresholds_hash": _canonical_hash(thresholds),
            "source_artifact": outputs["challenge_thresholds"],
        },
        "training_authorization": {
            "schema_version": "neuronsifter-round20q-training-authorization-freeze-v1",
            "round_id": "round20qr",
            "component_id": "training_authorization",
            "status": "blocked_baseline_challenge",
            "frozen": False,
            "training_authorized": False,
            "training_authorization_hash": _canonical_hash(authorization),
            "source_artifact": outputs["training_authorization"],
        },
        "config_family": {
            "schema_version": "neuronsifter-round20q-config-family-freeze-v1",
            "round_id": "round20qr",
            "component_id": "config_family",
            "status": "frozen",
            "frozen": True,
            "config_family_frozen": True,
            "config_family_hash": _canonical_hash({"configs": config_records}),
            "config_artifacts": config_records,
            "training_not_authorized": True,
        },
        "seed_protocol": {
            "schema_version": "neuronsifter-round20q-seed-protocol-freeze-v1",
            "round_id": "round20qr",
            "component_id": "seed_protocol",
            "status": "frozen",
            "frozen": True,
            "seed_protocol_frozen": True,
            "seeds": [3407, 3408, 3409, 3410, 3411],
            "statistical_contract": c1_task.get("bootstrap"),
            "seed_protocol_hash": _canonical_hash(
                {
                    "seeds": [3407, 3408, 3409, 3410, 3411],
                    "bootstrap": c1_task.get("bootstrap"),
                }
            ),
        },
    }
    for payload in component_payloads.values():
        if "artifact_uid" not in payload:
            _with_uid(payload)
    root_manifest = _with_uid(
        {
            "schema_version": "neuronsifter-round20q-replay-contract-v1",
            "round_id": "round20qr",
            "lineage_uid": lineage_uid,
            "authority_generation": "prospective_round20qr",
            "status": "blocked_baseline_challenge",
            "frozen": True,
            "scientific_gate_resolved": False,
            "producer": {
                "uid": "round20qr_builder",
                "script": "Libs/scripts/builders/build_round20qr_qualification.py",
                "git_identity": current_git_identity,
            },
            "components": {
                name: {
                    "path": f"{contract_dir_text}/{name}.json",
                    "artifact_uid": payload["artifact_uid"],
                    "status": payload.get("status"),
                }
                for name, payload in component_payloads.items()
            },
            "baseline_challenge_resolved": False,
            "training_authorized": False,
            "prospective_input_validation": prospective_input_validation,
            "c2b_status": C2B_BLOCKED,
        }
    )
    gate = _with_uid(
        {
            "schema_version": "neuronsifter-round20qr-gate-v1",
            "round_id": "round20qr",
            "lineage_uid": lineage_uid,
            "gate": "blocked_round20qr_baseline_challenge",
            "status": "blocked_round20qr_baseline_challenge",
            "authority_generation": "prospective_round20qr",
            "root_manifest_path": f"{contract_dir_text}/round20qr_root_manifest.json",
            "root_manifest_sha256": "",
            "round20qr_baseline_challenge_resolved": False,
            "round21r_replay_authorized": False,
            "c1_inputs_frozen": True,
            "c2s_inputs_frozen": True,
            "baseline_registry_27_bound": True,
            "challenge_thresholds_frozen": False,
            "training_authorized": False,
            "blockers": [
                "strong_baseline_results_unavailable",
                "simple_baseline_sanity_not_passed",
                "headline_training_not_authorized",
            ],
            "next_action": "qualify_missing_mandatory_baseline_sources_and_run_minimal_challenge_without_test_tuning",
            "c2b_status": C2B_BLOCKED,
        }
    )
    replay_auth = _with_uid(
        {
            "schema_version": "neuronsifter-round22b-round21r-replay-authorization-v1",
            "round_id": "neuronsifter_round22b",
            "lineage_uid": lineage_uid,
            "round21r_replay_authorized": False,
            "authority_root": "prospective_round20qr",
            "authority_root_manifest": f"{contract_dir_text}/round20qr_root_manifest.json",
            "reason": "round20qr_baseline_challenge_not_resolved",
            "gpu_allowed": False,
            "c2b_status": C2B_BLOCKED,
        }
    )

    if write_outputs:
        out = {key: root / value for key, value in outputs.items()}
        _immutable_tsv(
            out["component_inventory"],
            [
                "component_uid",
                "resolution_class",
                "canonical_path",
                "byte_size",
                "sha256",
                "schema_version",
                "producer_commit",
                "producer_tree",
                "producer_script",
                "creation_timestamp",
                "cutover_relation",
                "source_evidence",
                "notes",
            ],
            component_rows,
        )
        _immutable_tsv(
            out["producer_lineage"],
            [
                "lineage_uid",
                "role",
                "commit",
                "tree",
                "dirty",
                "source_class",
                "cutover_relation",
                "status",
                "evidence_path",
                "evidence_sha256",
                "notes",
            ],
            lineage_rows,
        )
        _immutable_json(out["frozen_contract"], frozen_contract)
        _immutable_json(out["authority_resolution"], resolution)
        _immutable_json(out["training_input"], training_input)
        _immutable_json(out["baseline_frontier"], frontier)
        _immutable_json(out["challenge_thresholds"], thresholds)
        _immutable_json(out["training_authorization"], authorization)
        contract_dir = root / contract_dir_text
        for name, payload in component_payloads.items():
            _immutable_json(contract_dir / f"{name}.json", payload)
        _immutable_json(contract_dir / "round20qr_root_manifest.json", root_manifest)
        gate["root_manifest_sha256"] = _sha256(
            contract_dir / "round20qr_root_manifest.json"
        )
        gate = _with_uid(
            {key: value for key, value in gate.items() if key != "artifact_uid"}
        )
        _immutable_json(out["gate"], gate)
        _immutable_json(out["replay_authorization"], replay_auth)
    return {
        "historical": resolution,
        "round20qr_gate": gate,
        "replay_authorization": replay_auth,
        "candidate_count": len(candidates),
        "lineage_uid": lineage_uid,
        "output_paths": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--lineage-tag", default="")
    parser.add_argument("--no-compute-node-check", action="store_true")
    args = parser.parse_args()
    result = build(
        args.config,
        args.schema,
        Path(args.root),
        lineage_tag=args.lineage_tag,
        enforce_compute_node=not args.no_compute_node_check,
    )
    print(
        json.dumps(
            {
                "historical_round20q_replay_root_resolved": result["historical"][
                    "historical_round20q_replay_root_resolved"
                ],
                "prospective_round20qr_required": result["historical"][
                    "prospective_round20qr_required"
                ],
                "round20qr_gate": result["round20qr_gate"]["status"],
                "round21r_replay_authorized": result["replay_authorization"][
                    "round21r_replay_authorized"
                ],
                "candidate_count": result["candidate_count"],
                "lineage_uid": result["lineage_uid"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
