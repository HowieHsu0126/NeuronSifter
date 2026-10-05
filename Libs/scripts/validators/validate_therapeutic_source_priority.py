"""Validate the metadata-only therapeutic source priority registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_therapeutic_source_priority import (  # noqa: E402
    BUILDER_PATH,
    CONFIG_PATH,
    OUTPUT_AUDIT_PATH,
    OUTPUT_TABLE_PATH,
    PRIORITY_CONFIG_PATH,
    SCHEMA_PATH,
    _artifact,
    _canonical_hash,
    _load_policy,
    build,
)


def validate(*, recompute: bool = False) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        policy = _load_policy()
        audit = json.loads(OUTPUT_AUDIT_PATH.read_text(encoding="utf-8"))
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "blocked", "passed": False, "blockers": [f"artifact_read:{exc}"]}

    for error in Draft202012Validator(schema).iter_errors(policy):
        blockers.append(f"policy_schema:{list(error.absolute_path)}:{error.message}")
    expected = {
        "policy_binding": (PRIORITY_CONFIG_PATH, "policy_binding"),
        "schema_binding": (SCHEMA_PATH, "schema_binding"),
        "implementation_binding": (BUILDER_PATH, "implementation_binding"),
    }
    if audit.get("policy_digest") != _canonical_hash(policy):
        blockers.append("policy_digest")
    for key, (path, label) in expected.items():
        if audit.get(key) != _artifact(path):
            blockers.append(label)
    registries = policy["registries"]
    for name in ("geo_accessions", "arrayexpress_accessions"):
        path = REPO_ROOT / str(registries[name])
        binding = audit.get("registry_bindings", {}).get(name)
        if not path.is_file() or binding != _artifact(path):
            blockers.append(f"registry_binding:{name}")
    if audit.get("priority_table_binding") != _artifact(OUTPUT_TABLE_PATH):
        blockers.append("priority_table_binding")
    if audit.get("status") != "source_priority_frozen_metadata_only":
        blockers.append("status")
    if audit.get("selection_basis") != "metadata_only":
        blockers.append("selection_basis")
    if audit.get("outcome_values_inspected") is not False:
        blockers.append("outcome_values_inspected")
    if audit.get("model_predictions_inspected") is not False:
        blockers.append("model_predictions_inspected")
    if audit.get("protected_holdout_access") != 0:
        blockers.append("protected_holdout_access")
    if audit.get("benchmark_split_created") is not False:
        blockers.append("benchmark_split_created")
    if audit.get("world_action_training_authorized") is not False:
        blockers.append("world_action_training_authorized")
    rows = audit.get("rows")
    if not isinstance(rows, list) or len(rows) != len(policy["candidates"]):
        blockers.append("row_count")
    else:
        ranks = [int(row.get("priority_rank", -1)) for row in rows if isinstance(row, Mapping)]
        if ranks != list(range(1, len(rows) + 1)):
            blockers.append("priority_ranks")
        selected = [row for row in rows if isinstance(row, Mapping) and row.get("selection_class") == "selected_for_materialization"]
        if len(selected) != audit.get("selected_for_materialization_count"):
            blockers.append("selected_count")
        if audit.get("expected_candidate_set_count") != sum(int(row["expected_candidate_set_count"]) for row in rows):
            blockers.append("expected_candidate_set_count")
        if audit.get("expected_candidate_pair_count") != sum(int(row["expected_candidate_pair_count"]) for row in rows):
            blockers.append("expected_candidate_pair_count")
        for row in rows:
            if row.get("expected_candidate_pair_count", 0) and not row.get("expected_candidate_set_count", 0):
                blockers.append(f"candidate_pair_without_set:{row.get('source_accession')}")
    unique = sorted(set(blockers))
    if recompute and not unique:
        rebuilt = build(write_output=False)
        for key in ("policy_digest", "source_count", "selected_for_materialization_count", "expected_candidate_set_count", "expected_candidate_pair_count", "selected_sources"):
            if rebuilt.get(key) != audit.get(key):
                unique.append(f"recompute:{key}")
        unique = sorted(set(unique))
    return {
        "schema_version": "neuronsifter-therapeutic-source-selection-validation-v1",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "source_count": audit.get("source_count"),
        "selected_for_materialization_count": audit.get("selected_for_materialization_count"),
        "expected_candidate_set_count": audit.get("expected_candidate_set_count"),
        "expected_candidate_pair_count": audit.get("expected_candidate_pair_count"),
        "protected_holdout_access": audit.get("protected_holdout_access"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    result = validate(recompute=args.recompute)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
