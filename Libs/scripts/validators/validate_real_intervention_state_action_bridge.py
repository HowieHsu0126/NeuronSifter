"""Validate the real-intervention state/action bridge without promoting U0 support."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config  # noqa: E402


BOUNDARY = {
    "evidence_role": "development_only",
    "publication_claim_eligible": False,
    "protected_holdout_access": 0,
}
CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
MISSING = {"explicit_unknown", "explicit_unavailable"}


def _repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"Configured path escapes repository: {value}")
    return path


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _binding(path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _binding_matches(binding: object, expected: Path) -> bool:
    if not isinstance(binding, Mapping) or not expected.is_file():
        return False
    try:
        return (
            str(binding.get("path")) == expected.relative_to(REPO_ROOT).as_posix()
            and int(binding.get("bytes", -1)) == expected.stat().st_size
            and str(binding.get("sha256")) == sha256_file(expected)
        )
    except (TypeError, ValueError, OSError):
        return False


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def validate(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    root = repo_root.resolve()
    if root != REPO_ROOT:
        raise ValueError("Bridge validation is identity-bound to the repository root")
    blockers: list[str] = []
    config = load_config(CONFIG_PATH)
    policy = config["real_intervention_evidence"]["state_action_bridge"]
    input_paths = {
        name: _repo_path(str(policy[key]))
        for name, key in (
            ("support_matrix", "input_support_matrix"),
            ("episode_catalog", "input_episode_catalog"),
            ("observation_manifest", "input_observation_manifest"),
            ("source_manifest", "input_source_manifest"),
            ("decision_sensitivity", "input_decision_sensitivity"),
        )
    }
    output_paths = {
        name: _repo_path(str(value))
        for name, value in policy["outputs"].items()
        if name != "processed_root"
    }
    for name, path in {**input_paths, **output_paths}.items():
        if not path.is_file():
            blockers.append(f"missing_artifact:{name}")
    if blockers:
        return {"status": "blocked", "blocker_count": len(blockers), "blockers": sorted(set(blockers))}

    try:
        support_rows = _tsv(input_paths["support_matrix"])
        input_episodes = _tsv(input_paths["episode_catalog"])
        input_observations = _tsv(input_paths["observation_manifest"])
        input_sources = _tsv(input_paths["source_manifest"])
        decision_sensitivity = _json(input_paths["decision_sensitivity"])
        manifest_rows = _tsv(output_paths["episode_manifest"])
        action_rows = _tsv(output_paths["action_catalog"])
        episode_rows = _tsv(output_paths["episode_catalog"])
        quality = _json(output_paths["quality_audit"])
        integrity = _json(output_paths["episode_integrity"])
    except (OSError, csv.Error, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"status": "blocked", "blocker_count": 1, "blockers": [f"read:{exc}"]}

    expected_episode_count = int(policy["expected_episode_count"])
    expected_source_count = int(policy["expected_source_count"])
    if len(support_rows) != expected_episode_count:
        blockers.append("support_count")
    if len(input_episodes) != expected_episode_count or len(episode_rows) != expected_episode_count:
        blockers.append("episode_count")
    if len(manifest_rows) != expected_source_count or len(input_sources) != expected_source_count:
        blockers.append("source_count")
    if not action_rows:
        blockers.append("empty_action_catalog")

    support_by_uid = {row.get("episode_uid", ""): row for row in support_rows}
    input_by_uid = {row.get("episode_uid", ""): row for row in input_episodes}
    output_by_uid = {row.get("episode_uid", ""): row for row in episode_rows}
    observation_uids = {row.get("observation_uid", "") for row in input_observations}
    source_by_key = {row.get("source_key", ""): row for row in input_sources}
    action_by_uid = {row.get("structured_action_uid", ""): row for row in action_rows}
    if len(output_by_uid) != len(episode_rows) or set(output_by_uid) != set(input_by_uid):
        blockers.append("episode_identity")
    if len(action_by_uid) != len(action_rows):
        blockers.append("action_identity")

    for row in episode_rows:
        uid = row.get("episode_uid", "")
        source = source_by_key.get(row.get("source_key", ""))
        support = support_by_uid.get(uid)
        if support is None or source is None:
            blockers.append(f"row_binding:{uid}")
            continue
        for field in ("publication_uid", "accession_uid", "biological_unit_uid", "baseline_state_uid", "world_action_uid", "post_action_observation_uid", "control_uid"):
            if not row.get(field):
                blockers.append(f"row_missing:{field}")
        if row.get("post_action_observation_uid") not in observation_uids:
            blockers.append(f"row_observation_join:{uid}")
        if row.get("structured_action_uid") != row.get("world_action_uid") or row.get("structured_action_uid") not in action_by_uid:
            blockers.append(f"row_action_join:{uid}")
        if row.get("support_primary_failure_class") != support.get("primary_failure_class"):
            blockers.append(f"row_support_class:{uid}")
        if row.get("state_projection_status") not in set(policy["allowed_state_projection_status"]):
            blockers.append(f"row_state_status:{uid}")
        if row.get("state_namespace") != policy["state_namespace"]:
            blockers.append(f"row_state_namespace:{uid}")
        if row.get("u0_executable_support") != "unsupported" or row.get("u0_projection_status") != policy["u0_projection_status"]:
            blockers.append(f"row_u0_promotion:{uid}")
        for field in ("target_set", "exposure_program_ref", "occupancy_program_ref"):
            value = str(row.get(field, "")).strip()
            if not value:
                blockers.append(f"row_silent_missing:{field}:{uid}")
        if row.get("action_semantics_status") not in set(policy["allowed_action_semantics_status"]):
            blockers.append(f"row_action_status:{uid}")
        if row.get("protected_holdout_access") != "0" or row.get("publication_claim_eligible") != "False" or row.get("evidence_role") != "development_only":
            blockers.append(f"row_boundary:{uid}")

    for row in action_rows:
        if row.get("action_semantics_status") == "full_structured_action" and row.get("missing_action_fields") != "none":
            blockers.append(f"action_full_missing:{row.get('structured_action_uid')}")
        if row.get("u0_action_compatible") != "False":
            blockers.append(f"action_u0_promotion:{row.get('structured_action_uid')}")
        if row.get("protected_holdout_access") != "0" or row.get("publication_claim_eligible") != "False" or row.get("evidence_role") != "development_only":
            blockers.append(f"action_boundary:{row.get('structured_action_uid')}")

    for row in manifest_rows:
        source = source_by_key.get(row.get("source_key", ""))
        if source is None or row.get("source_accession") != source.get("source_accession"):
            blockers.append(f"manifest_source:{row.get('source_key')}")
        if int(row.get("u0_supported_count", -1)) != 0 or _truthy(row.get("training_authorized")):
            blockers.append(f"manifest_promotion:{row.get('source_key')}")
        if row.get("protected_holdout_access") != "0" or row.get("publication_claim_eligible") != "False" or row.get("evidence_role") != "development_only":
            blockers.append(f"manifest_boundary:{row.get('source_key')}")

    for key, expected in BOUNDARY.items():
        if quality.get(key) != expected or integrity.get(key) != expected:
            blockers.append(f"boundary:{key}")
    if quality.get("status") != "resolved" or quality.get("state_namespace") != policy["state_namespace"]:
        blockers.append("quality_status")
    if quality.get("u0_supported_count") != int(policy["expected_u0_supported_count"]):
        blockers.append("quality_u0_support")
    if quality.get("training_authorization") != policy["training_authorization"] or quality.get("new_static_lineage_created") is not False:
        blockers.append("quality_training_boundary")
    if quality.get("round27_decision_sensitivity_bound") is not True:
        blockers.append("quality_decision_binding")
    if quality.get("decision_sensitive_world_action_domain_available") != decision_sensitivity.get(
        "decision_sensitive_world_action_domain_available"
    ):
        blockers.append("quality_decision_domain")
    if quality.get("prespecified_therapeutic_decision_objective_available") != decision_sensitivity.get(
        "prespecified_therapeutic_decision_objective_available"
    ):
        blockers.append("quality_decision_objective")
    if quality.get("model_training_domain_qualified") != decision_sensitivity.get("model_training_domain_qualified"):
        blockers.append("quality_training_domain")
    if quality.get("round28_decision_sensitive_benchmark_pending") is not True or quality.get("model_predictions_inspected") is not False:
        blockers.append("quality_decision_boundary")
    if integrity.get("status") != "passed" or integrity.get("u0_supported_count") != 0 or integrity.get("new_lineage_training_authorized") is not False:
        blockers.append("integrity_status")
    if integrity.get("episode_count") != expected_episode_count or integrity.get("action_count") != len(action_rows):
        blockers.append("integrity_counts")

    for name, path in input_paths.items():
        if not _binding_matches(quality.get("input_bindings", {}).get(name), path):
            blockers.append(f"quality_input_binding:{name}")
        if not _binding_matches(integrity.get("input_bindings", {}).get(name), path):
            blockers.append(f"integrity_input_binding:{name}")
    for name, path in output_paths.items():
        if name == "quality_audit":
            continue
        if not _binding_matches(quality.get("output_bindings", {}).get(name), path):
            blockers.append(f"quality_output_binding:{name}")
    unique = sorted(set(blockers))
    return {
        "status": "passed" if not unique else "blocked",
        "blocker_count": len(unique),
        "blockers": unique,
        "source_count": len(manifest_rows),
        "episode_count": len(episode_rows),
        "action_count": len(action_rows),
        "u0_supported_count": sum(row.get("u0_executable_support") == "supported" for row in episode_rows),
        **BOUNDARY,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    report = validate(args.repo_root)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
