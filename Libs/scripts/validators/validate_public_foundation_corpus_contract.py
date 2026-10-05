from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, write_json


DEFAULT_CONFIG_PATH = "Libs/configs/config_stack.yaml"
DEFAULT_SCHEMA_PATH = "Libs/configs/schema/public_foundation_corpus.schema.json"

EXPECTED_SEMANTIC_AXES = {
    "study",
    "biological_unit",
    "species",
    "disease_state",
    "tissue",
    "brain_region",
    "cell_type",
    "assay",
    "modality",
    "feature",
    "time",
    "dose",
    "unit",
    "provenance",
}
EXPECTED_DISEASE_PROGRAMS = {
    "ad_control": "AD",
    "pd_control": "PD",
    "ftd_control": "FTD",
    "als_control": "ALS",
    "neuroinflammation_control": "neuroinflammation",
}
EXPECTED_CORE_FAMILIES = {
    "cns_disease_aging_state_transcriptomics",
    "cns_cell_type_region_atlas_transcriptomics",
    "compound_dose_time_perturbation_transcriptomics",
}
EXPECTED_DEFERRED_MODALITIES = {
    "epigenomics",
    "genomics",
    "gwas",
    "proteomics",
    "metabolomics",
    "lipidomics",
    "electrophysiology",
    "morphology",
    "imaging",
}
EXPECTED_DEFERRED_ROLES = {"context_prior", "weak_calibration", "holdout_extension"}
EXPECTED_CONTROLLED_EXCLUSIONS = {
    "adni",
    "dbgap",
    "uk_biobank",
    "rosmap",
    "amp_ad",
    "restricted_synapse",
}
EXPECTED_CROSSWALK_ENTITIES = {
    "gene",
    "protein",
    "compound",
    "cell_type",
    "brain_region",
    "assay",
    "disease",
    "publication",
    "species",
    "unit",
}
EXPECTED_G1_REQUIRED_CONDITIONS = [
    "license_and_terms_verified",
    "version_and_checksums_verified",
    "provenance_verified",
    "entity_and_unit_audit_passed",
    "controlled_reachability_zero",
    "critical_mapping_conflicts_zero",
    "storage_cap_confirmed",
]
EXPECTED_SOURCE_RELEASE_FIELDS = {
    "source_id",
    "official_url",
    "release_version",
    "retrieval_date",
    "source_card_path",
    "terms_snapshot_path",
    "terms_snapshot_sha256",
    "license_snapshot_path",
    "license_snapshot_sha256",
    "raw_sha256",
    "processed_sha256",
    "parser_id",
    "parser_version",
    "mapping_version",
    "provenance_record_id",
    "training_evaluation_role",
    "holdout_group",
    "publication_accession_group",
    "access_class",
}
EXPECTED_ACTION_FIELDS = {
    "action_uid",
    "intervention_type",
    "target_set",
    "agent_set",
    "dose_value",
    "dose_unit",
    "schedule",
    "route",
    "assay_context",
    "state_scope",
    "source_publication",
    "source_accession",
    "exposure_program_ref",
    "occupancy_program_ref",
}


def _repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_contract_schema(schema_path: str | Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    path = _repo_path(schema_path)
    with path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    Draft202012Validator.check_schema(schema)
    return schema


def _path_text(parts: object) -> str:
    values = [str(value) for value in parts]
    return ".".join(values) if values else "$"


def _add_blocker(
    blockers: list[dict[str, str]],
    code: str,
    path: str,
    message: str,
) -> None:
    blockers.append({"code": code, "path": path, "message": message})


def _require(
    condition: bool,
    blockers: list[dict[str, str]],
    code: str,
    path: str,
    message: str,
) -> None:
    if not condition:
        _add_blocker(blockers, code, path, message)


def _row_ids(rows: list[dict[str, Any]], key: str) -> list[str]:
    return [str(row[key]) for row in rows]


def validate_public_foundation_corpus_payload(
    payload: object,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_schema = schema if schema is not None else load_contract_schema()
    blockers: list[dict[str, str]] = []
    validator = Draft202012Validator(active_schema, format_checker=FormatChecker())
    schema_errors = sorted(
        validator.iter_errors(payload),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    for error in schema_errors:
        _add_blocker(
            blockers,
            "json_schema_violation",
            _path_text(error.absolute_path),
            error.message,
        )

    if schema_errors or not isinstance(payload, dict):
        return {
            "contract_id": "public_foundation_corpus_v1",
            "scope": payload.get("scope") if isinstance(payload, dict) else None,
            "status": "blocked",
            "blocker_count": len(blockers),
            "blockers": blockers,
        }

    _require(
        payload["scope"] == "reduced_transcriptomics_core",
        blockers,
        "scope_mismatch",
        "scope",
        "V1 scope must remain reduced_transcriptomics_core.",
    )

    alignment = payload["semantic_alignment"]
    _require(
        set(alignment["required_axes"]) == EXPECTED_SEMANTIC_AXES,
        blockers,
        "semantic_axes_mismatch",
        "semantic_alignment.required_axes",
        "Every canonical observation must expose the complete semantic alignment axis set.",
    )

    disease_rows = payload["disease_programs"]
    disease_ids = _row_ids(disease_rows, "program_id")
    disease_mapping = {str(row["program_id"]): str(row["disease_family"]) for row in disease_rows}
    _require(
        len(disease_ids) == len(set(disease_ids)) == 5 and disease_mapping == EXPECTED_DISEASE_PROGRAMS,
        blockers,
        "disease_program_set_mismatch",
        "disease_programs",
        "The five approved disease/control programs and their families must be exact and unique.",
    )

    core_rows = payload["core_observation_families"]
    core_ids = _row_ids(core_rows, "family_id")
    _require(
        len(core_ids) == len(set(core_ids)) == 3 and set(core_ids) == EXPECTED_CORE_FAMILIES,
        blockers,
        "core_family_set_mismatch",
        "core_observation_families",
        "Exactly the three approved transcriptomics observation families are core.",
    )
    for index, row in enumerate(core_rows):
        path = f"core_observation_families.{index}"
        _require(
            row["modality"] == "transcriptomics"
            and row["minimum_independent_source_ids"] >= 2
            and row["independent_group_key"] == "publication_accession_group"
            and row["nonoverlapping_publication_accession_groups_required"] is True
            and row["minimum_cns_sources"] >= 1
            and row["mirrored_accessions_count_as_independent"] is False,
            blockers,
            "core_family_qualification_weakened",
            path,
            "Each core family requires two independent source IDs, nonoverlapping groups, and a CNS source; mirrors cannot double count.",
        )

    deferred_rows = payload["deferred_modalities"]
    deferred_ids = _row_ids(deferred_rows, "modality")
    _require(
        len(deferred_ids) == len(set(deferred_ids)) == 9
        and set(deferred_ids) == EXPECTED_DEFERRED_MODALITIES,
        blockers,
        "deferred_modality_set_mismatch",
        "deferred_modalities",
        "The auxiliary/deferred modality set must be exact and unique.",
    )
    for index, row in enumerate(deferred_rows):
        _require(
            set(row["allowed_roles"]) == EXPECTED_DEFERRED_ROLES
            and row["counts_toward_core_ess"] is False
            and row["counts_toward_core_g2"] is False
            and row["primary_supervision_allowed"] is False,
            blockers,
            "deferred_modality_role_escalation",
            f"deferred_modalities.{index}",
            "Deferred modalities may only be context priors, weak calibration, or holdout extensions.",
        )

    governance = payload["source_governance"]
    _require(
        set(governance["required_release_fields"]) == EXPECTED_SOURCE_RELEASE_FIELDS,
        blockers,
        "source_governance_fields_incomplete",
        "source_governance.required_release_fields",
        "Source releases must retain the complete licensing, checksum, parser, mapping, provenance, and holdout contract.",
    )

    exclusions = payload["controlled_exclusions"]
    excluded_ids = _row_ids(exclusions["sources"], "source_id")
    _require(
        len(excluded_ids) == len(set(excluded_ids)) == 6
        and set(excluded_ids) == EXPECTED_CONTROLLED_EXCLUSIONS
        and exclusions["public_mainline_dependency_allowed"] is False
        and exclusions["required_public_mainline_reachability_count"] == 0,
        blockers,
        "controlled_boundary_weakened",
        "controlled_exclusions",
        "All named controlled/restricted sources must remain audit-only and unreachable from the public mainline.",
    )

    crosswalk = payload["crosswalk_contract"]
    _require(
        set(crosswalk["entity_types"]) == EXPECTED_CROSSWALK_ENTITIES
        and set(crosswalk["unit_ontologies"]) == {"UCUM", "UO"}
        and crosswalk["critical_mapping_conflict_limit"] == 0,
        blockers,
        "crosswalk_contract_incomplete",
        "crosswalk_contract",
        "Canonical entity and unit crosswalks must be complete and permit zero critical conflicts.",
    )

    action = payload["structured_action_contract"]
    _require(
        set(action["required_fields"]) == EXPECTED_ACTION_FIELDS
        and action["incomplete_record_policy"] == "quarantine",
        blockers,
        "structured_action_contract_incomplete",
        "structured_action_contract",
        "Structured actions missing physical or provenance fields must be quarantined.",
    )

    quality = payload["quality_and_training_roles"]
    _require(
        quality["grade_roles"]
        == {
            "A": "main_training",
            "B": "main_training",
            "C": "weak_calibration",
            "D": "reject",
            "S": "stress_test_only",
        }
        and quality["main_training_grades"] == ["A", "B"]
        and quality["core_ess_grades"] == ["A", "B"],
        blockers,
        "qc_role_escalation",
        "quality_and_training_roles",
        "Only Grade A/B may enter main training and core ESS; Grade C remains weak calibration.",
    )
    synthetic_policy = quality["synthetic_teacher_policy"]
    _require(
        not any(bool(value) for value in synthetic_policy.values()),
        blockers,
        "synthetic_teacher_escalation",
        "quality_and_training_roles.synthetic_teacher_policy",
        "Synthetic teachers cannot count as observations, G2 evidence, or main training data.",
    )

    split_policy = payload["split_policy"]
    _require(
        split_policy["query_domain_status"] == "frozen"
        and split_policy["split_manifest_status"] == "frozen"
        and split_policy["deterministic_assignment"] == "hash_grouped_after_freeze"
        and split_policy["split_seed_required_at_freeze"] is True,
        blockers,
        "split_freeze_policy_mismatch",
        "split_policy",
        "The frozen query domain must use the declared deterministic assignment and seed contract.",
    )

    claim_policy = payload["cross_modal_claim_policy"]
    paired = claim_policy["paired_benchmark"]
    paired_is_real = (
        paired["status"] == "qualified"
        and paired["provenance_kind"] == "real_only"
        and paired["real_pair_count"] > 0
    )
    _require(
        not claim_policy["cross_modal_prediction_claim_allowed"] or paired_is_real,
        blockers,
        "cross_modal_claim_without_real_pairs",
        "cross_modal_claim_policy",
        "Cross-modal prediction may be claimed only with a qualified benchmark containing real pairs.",
    )
    _require(
        paired["status"] != "not_available" or paired["real_pair_count"] == 0,
        blockers,
        "paired_benchmark_status_count_mismatch",
        "cross_modal_claim_policy.paired_benchmark",
        "An unavailable paired benchmark must report zero real pairs.",
    )

    storage = payload["storage_policy"]
    _require(
        storage["counted_persistent_classes"] == ["raw", "processed", "canonical", "cache", "output"]
        and storage["decimal_byte_cap"] == 1_000_000_000_000
        and storage["cap_exceeded_gate_policy"] == "fail"
        and storage["scan_root_paths"] == {
            "raw": "Input/raw",
            "processed": "Input/processed",
            "canonical": "Input/canonical",
            "cache": "Input/cache",
            "output": "Output",
        }
        and storage["scan_excluded_paths"] == [
            "Output/audits/public_foundation_corpus_storage_ledger.json"
        ]
        and storage["previous_confirmed_usage_bytes"] == 422_655_337_981
        and storage["largest_directory_limit"] == 20
        and storage["storage_ledger_schema_path"]
        == "Libs/configs/schema/public_foundation_corpus_storage_ledger.schema.json"
        and storage["storage_ledger_path"]
        == "Output/audits/public_foundation_corpus_storage_ledger.json"
        and storage["storage_validation_path"]
        == "Output/audits/public_foundation_corpus_storage_validation.json",
        blockers,
        "storage_budget_weakened",
        "storage_policy",
        "The persistent raw, processed, canonical, cache, and Output footprint must fail above one decimal TB.",
    )

    gates = payload["gates"]
    _require(
        gates["public_corpus_g1"]["gate_id"] == "iclr_v2.public_corpus.G1"
        and gates["public_corpus_g1"]["current_status"] == "passed"
        and gates["public_corpus_g1"]["required_conditions"]
        == EXPECTED_G1_REQUIRED_CONDITIONS
        and gates["public_corpus_g2"]["gate_id"] == "iclr_v2.public_corpus.G2"
        and gates["legacy_simulation_readiness"]["gate_id"] == "legacy.simulation_readiness.G2",
        blockers,
        "gate_namespace_collision",
        "gates",
        "G1 must remain evidence-complete and public-corpus gates must use distinct locked namespaces.",
    )
    _require(
        gates["public_corpus_g2"]["scope"] == "reduced_transcriptomics_core"
        and gates["public_corpus_g2"]["passing_status"] == "PASS_WITH_SCOPE_LIMITATION"
        and gates["public_corpus_g2"]["full_multimodal_claim_allowed"] is False
        and gates["full_multimodal_extension"]["current_status"] == "not_passed",
        blockers,
        "multimodal_gate_overclaim",
        "gates",
        "Reduced-core G2 must retain its scope limitation and cannot imply full multimodal readiness.",
    )

    return {
        "contract_id": "public_foundation_corpus_v1",
        "scope": payload["scope"],
        "status": "pass" if not blockers else "blocked",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "core_family_count": len(core_rows),
        "disease_program_count": len(disease_rows),
        "deferred_modality_count": len(deferred_rows),
        "controlled_exclusion_count": len(exclusions["sources"]),
        "storage_decimal_byte_cap": storage["decimal_byte_cap"],
        "gate_status": {
            "iclr_v2.public_corpus.G1": gates["public_corpus_g1"]["current_status"],
            "iclr_v2.public_corpus.G2": gates["public_corpus_g2"]["current_status"],
            "iclr_v2.public_corpus.G2_full_multimodal": gates["full_multimodal_extension"]["current_status"],
        },
    }


def build_report(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    schema_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    payload = config.get("public_foundation_corpus")
    selected_schema_path = schema_path
    if selected_schema_path is None and isinstance(payload, dict):
        selected_schema_path = str(payload.get("schema_path", DEFAULT_SCHEMA_PATH))
    schema = load_contract_schema(selected_schema_path or DEFAULT_SCHEMA_PATH)
    report = validate_public_foundation_corpus_payload(payload, schema)
    report["config_path"] = str(config_path)
    report["schema_path"] = str(selected_schema_path or DEFAULT_SCHEMA_PATH)
    report["producer"] = "validate_public_foundation_corpus_contract.py"
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the alignment-first Public Foundation Corpus V1 machine contract."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--schema", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.config, args.schema)
    config = load_config(args.config)
    payload = config.get("public_foundation_corpus", {})
    output_value = args.output or payload.get(
        "audit_output_path",
        "Output/audits/public_foundation_corpus_contract_audit.json",
    )
    output_path = _repo_path(str(output_value))
    write_json(output_path, report)
    print(f"public_foundation_corpus_contract_status={report['status']}")
    print(f"public_foundation_corpus_contract_blocker_count={report['blocker_count']}")
    print(f"public_foundation_corpus_scope={report.get('scope')}")
    print(f"public_foundation_corpus_contract_output={output_path}")
    if report["blocker_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
