"""Validate the role-scoped Phase 1 entity/unit crosswalk artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_evidence import safe_repo_path, sha256_file
from Libs.utils.data.raw_data_registry import write_json


ARTIFACT = "Output/audits/public_foundation_corpus_entity_unit_crosswalk.json"
SCHEMA = "Libs/configs/schema/public_foundation_corpus_crosswalk.schema.json"
REQUIRED = {"gene", "protein", "compound", "cell_type", "brain_region", "assay", "disease", "publication", "species", "unit"}


def validate(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    root = repo_root.resolve()
    artifact_path = safe_repo_path(root, ARTIFACT)
    schema_path = safe_repo_path(root, SCHEMA)
    blockers: list[dict[str, str]] = []
    if not artifact_path.is_file() or not schema_path.is_file():
        return {"status": "blocked", "blocker_count": 1, "blockers": [{"code": "required_artifact_missing", "message": ARTIFACT}]}
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(artifact), key=lambda e: list(e.absolute_path))
    blockers.extend({"code": "json_schema_violation", "path": ".".join(map(str, e.absolute_path)), "message": e.message} for e in errors)
    counts = artifact.get("entity_type_counts", {})
    if set(artifact.get("required_entity_types", [])) != REQUIRED:
        blockers.append({"code": "required_entity_set_mismatch", "path": "required_entity_types", "message": "Required entity axes are not the locked Phase 1 set."})
    missing = [entity for entity in REQUIRED if int(counts.get(entity, 0)) <= 0]
    if missing:
        blockers.append({"code": "required_entity_crosswalk_incomplete", "path": "entity_type_counts", "message": ",".join(sorted(missing))})
    if artifact.get("unresolved_required_mapping_count") != 0 or artifact.get("unresolved_required_entities"):
        blockers.append({"code": "unresolved_required_mapping", "path": "unresolved_required_mapping_count", "message": "Required mapping remains unresolved."})
    if artifact.get("critical_mapping_conflict_count") != 0 or artifact.get("ambiguity_count") != 0:
        blockers.append({"code": "mapping_conflict_or_ambiguity", "path": "$", "message": "Critical mapping conflicts and ambiguities must be zero."})
    for source_id, report in artifact.get("source_role_reports", {}).items():
        if report.get("status") != "passed" or report.get("missing_axes"):
            blockers.append({"code": "source_role_crosswalk_incomplete", "path": f"source_role_reports.{source_id}", "message": str(report.get("missing_axes"))})
    binding = artifact.get("crosswalk_table_binding", {})
    if binding.get("sha256") != sha256_file(safe_repo_path(root, binding.get("path", ""))):
        blockers.append({"code": "crosswalk_table_checksum_mismatch", "path": "crosswalk_table_binding.sha256", "message": "TSV binding hash does not match."})
    # The builder's status is the artifact-level qualification decision; the
    # validator reports any independent contradiction as ``blocked`` above.
    return {"status": "passed" if not blockers else "blocked", "blocker_count": len(blockers), "blockers": blockers, "record_count": artifact.get("record_count"), "entity_type_counts": counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="Output/audits/public_foundation_corpus_crosswalk_validation.json")
    args = parser.parse_args()
    report = validate()
    write_json(safe_repo_path(REPO_ROOT, args.output), report)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
