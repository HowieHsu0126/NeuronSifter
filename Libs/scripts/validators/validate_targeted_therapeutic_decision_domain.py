"""Validate the targeted matched-control therapeutic decision domain."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_targeted_therapeutic_decision_domain import (  # noqa: E402
    BUILDER_ID,
    TARGETED_CONFIG_PATH,
    _artifact,
    _input_paths,
    _policy_digest,
)
from Libs.utils.data.raw_data_registry import load_config, read_tsv  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"


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


def _binding_matches(binding: object, path: Path) -> bool:
    if not isinstance(binding, Mapping):
        return False
    try:
        return dict(binding) == _artifact(path)
    except (FileNotFoundError, OSError, ValueError):
        return False


def _bool_text(value: object) -> bool:
    return str(value).strip().lower() == "true"


def validate(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    if repo_root.resolve() != REPO_ROOT:
        raise ValueError("Therapeutic decision-domain validation is identity-bound to the repository root")
    config = load_config(CONFIG_PATH)
    policy = config["real_intervention_evidence"]["targeted_therapeutic_expansion"]
    if not isinstance(policy, Mapping):
        raise TypeError("Targeted therapeutic policy must be a mapping")
    outputs = {name: _repo_path(str(value)) for name, value in policy["outputs"].items()}
    planned_splits = {name: _repo_path(str(value)) for name, value in policy["planned_split_paths"].items()}
    blockers: list[str] = []

    required_output_names = {
        "source_manifest",
        "observation_manifest",
        "gene_outcomes",
        "candidate_set_manifest",
        "outcome_quality",
        "objective",
        "domain_summary",
        "split_audit",
    }
    if set(outputs) != required_output_names:
        blockers.append("output_contract_mismatch")
    missing = [name for name, path in outputs.items() if not path.is_file()]
    if missing:
        return {
            "status": "blocked",
            "blocker_count": len(missing),
            "blockers": sorted(f"missing_output:{name}" for name in missing),
            "evidence_role": "development_only",
            "publication_claim_eligible": False,
            "protected_holdout_access": 0,
        }

    objective = _json(outputs["objective"])
    summary = _json(outputs["domain_summary"])
    split_audit = _json(outputs["split_audit"])
    schema = _json(_repo_path(str(policy["summary_schema_path"])))
    source_rows = read_tsv(outputs["source_manifest"])
    observation_rows = read_tsv(outputs["observation_manifest"])
    gene_rows = read_tsv(outputs["gene_outcomes"])
    candidate_rows = read_tsv(outputs["candidate_set_manifest"])
    quality_rows = read_tsv(outputs["outcome_quality"])

    Draft202012Validator.check_schema(schema)
    for error in Draft202012Validator(schema).iter_errors(summary):
        location = ".".join(str(value) for value in error.absolute_path) or "root"
        blockers.append(f"summary_schema:{location}:{error.message}")

    if objective.get("policy_digest") != _policy_digest(policy):
        blockers.append("objective_policy_digest")
    if objective.get("objective", {}).get("freeze_status") != "frozen_before_outcome_materialization":
        blockers.append("objective_not_frozen_before_outcomes")
    objective_payload = objective.get("objective", {})
    if objective_payload.get("decision_unit") != "candidate_treatment_within_source_defined_biological_context":
        blockers.append("objective_decision_unit_scope")
    if objective_payload.get("reference_condition") != "source_declared_matched_control":
        blockers.append("objective_reference_condition_scope")
    if objective.get("outcome_values_inspected_for_objective_selection") is not False:
        blockers.append("objective_selection_inspected_outcomes")
    if objective.get("model_predictions_inspected") is not False:
        blockers.append("objective_model_prediction_inspection")
    if objective.get("evidence_role") != "development_only":
        blockers.append("objective_evidence_role")
    if objective.get("publication_claim_eligible") is not False or objective.get("protected_holdout_access") != 0:
        blockers.append("objective_boundary")
    if not _binding_matches(objective.get("config_binding"), TARGETED_CONFIG_PATH):
        blockers.append("objective_config_binding")

    required_accessions = [str(value) for value in policy["source_policy"]["selected_accessions"]]
    source_by_accession = {row.get("source_accession", ""): row for row in source_rows}
    if set(source_by_accession) != set(required_accessions) or len(source_rows) != len(required_accessions):
        blockers.append("source_accession_coverage")
    if len({row.get("source_key") for row in source_rows}) != len(source_rows):
        blockers.append("source_identity")
    observed_counts: dict[str, Counter[str]] = {}
    for row in observation_rows:
        accession = str(row.get("accession_uid", ""))
        observed_counts.setdefault(accession, Counter())[str(row.get("treatment", ""))] += 1
    source_contracts = policy["source_policy"]["sources"]
    for row in source_rows:
        if row.get("evidence_role") != "development_only" or row.get("publication_claim_eligible") != "False" or row.get("protected_holdout_access") != "0":
            blockers.append(f"source_boundary:{row.get('source_accession')}")
        accession = str(row.get("source_accession", ""))
        contract = source_contracts.get(accession)
        try:
            expected = Counter({str(key): int(value) for key, value in contract["expected_biological_replicates"].items()})
            observed = observed_counts.get(accession, Counter())
            if observed != expected:
                blockers.append(f"source_replicate_counts:{accession}")
            if int(row["selected_sample_count"]) != sum(expected.values()):
                blockers.append(f"source_selected_sample_count:{accession}")
            if int(row["control_sample_count"]) != expected[str(contract["control_condition"])]:
                blockers.append(f"source_control_sample_count:{accession}")
            if int(row["candidate_action_count"]) != len(contract["candidate_treatments"]):
                blockers.append(f"source_candidate_action_count:{accession}")
        except (KeyError, TypeError, ValueError):
            blockers.append(f"source_numeric_fields:{accession}")

    if len(observation_rows) != 37:
        blockers.append("observation_count")
    if len(gene_rows) != 888:
        blockers.append("gene_outcome_count")
    if len(candidate_rows) != 3:
        blockers.append("candidate_set_count")
    if len({row.get("episode_uid") for row in observation_rows}) != len(observation_rows):
        blockers.append("observation_identity")
    if len({(row.get("sample_uid"), row.get("gene_symbol")) for row in gene_rows}) != len(gene_rows):
        blockers.append("gene_outcome_identity")
    if any(
        row.get("evidence_role") != "development_only"
        or row.get("publication_claim_eligible") != "False"
        or row.get("protected_holdout_access") != "0"
        or row.get("raw_value_materialized") != "True"
        for row in observation_rows
    ):
        blockers.append("observation_boundary_or_materialization")
    if any(
        row.get("evidence_role") != "development_only"
        or row.get("publication_claim_eligible") != "False"
        or row.get("protected_holdout_access") != "0"
        for row in gene_rows
    ):
        blockers.append("gene_outcome_boundary")
    if any(
        row.get("matched_control_values_materialized") != "True"
        or row.get("model_predictions_inspected") != "False"
        or row.get("evidence_role") != "development_only"
        or row.get("publication_claim_eligible") != "False"
        or row.get("protected_holdout_access") != "0"
        for row in quality_rows
    ):
        blockers.append("quality_boundary")

    candidates_by_source = {row.get("source_accession", ""): row for row in candidate_rows}
    valid_classes = {
        "decision_sensitive",
        "weakly_decision_sensitive",
        "scientifically_informative_but_not_rankable",
        "null",
        "ambiguous",
    }
    for accession in required_accessions:
        row = candidates_by_source.get(accession)
        if row is None:
            blockers.append(f"candidate_source_missing:{accession}")
            continue
        if row.get("classification") not in valid_classes:
            blockers.append(f"candidate_classification:{accession}")
        if row.get("therapeutic_objective_prespecified") != "True":
            blockers.append(f"candidate_objective:{accession}")
        if row.get("matched_control_values_materialized") != "True":
            blockers.append(f"candidate_control_values:{accession}")
        if _bool_text(row.get("qualified_decision_candidate_set")) and row.get("candidate_action_count", "0") in {"0", "1"}:
            blockers.append(f"qualified_single_action:{accession}")
        try:
            expected = {
                str(key): int(value)
                for key, value in source_contracts[accession]["expected_biological_replicates"].items()
                if str(key) != str(source_contracts[accession]["control_condition"])
            }
            observed_candidates = json.loads(row["candidate_sample_counts"])
            if observed_candidates != expected:
                blockers.append(f"candidate_replicate_counts:{accession}")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            blockers.append(f"candidate_numeric_fields:{accession}")

    qualified_count = sum(_bool_text(row.get("qualified_decision_candidate_set")) for row in candidate_rows)
    split_created = bool(split_audit.get("split_files_created"))
    if summary.get("qualified_decision_candidate_set_count") != qualified_count:
        blockers.append("qualified_count")
    if summary.get("decision_sensitive_domain_available") is not (qualified_count > 0):
        blockers.append("decision_domain_flag")
    if summary.get("model_training_domain_qualified") is not split_created:
        blockers.append("training_domain_flag")
    if split_created:
        if not all(path.is_file() for path in planned_splits.values()):
            blockers.append("split_file_missing")
        if split_audit.get("accession_overlap_count") != 0 or split_audit.get("publication_overlap_count") != 0 or split_audit.get("biological_unit_overlap_count") != 0:
            blockers.append("split_leakage")
    elif any(path.exists() for path in planned_splits.values()):
        blockers.append("unqualified_split_file_present")

    historical = summary.get("historical_artifact_bindings", {})
    expected_historical = {
        "domain_summary": REPO_ROOT / "Output/audits/real_intervention_decision_domain.json",
        "context_group_qualification": REPO_ROOT / "Output/audits/real_intervention_context_group_qualification.tsv",
        "split_audit": REPO_ROOT / "Output/audits/real_intervention_decision_benchmark_split_audit.json",
    }
    for name, path in expected_historical.items():
        if not _binding_matches(historical.get(name), path):
            blockers.append(f"historical_artifact_drift:{name}")

    expected_inputs = {
        **_input_paths(policy),
        "objective": outputs["objective"],
    }
    expected_implementation = {
        "config": TARGETED_CONFIG_PATH,
        "schema": _repo_path(str(policy["summary_schema_path"])),
        "builder": REPO_ROOT / BUILDER_ID,
    }
    expected_outputs = {
        name: outputs[name]
        for name in (
            "source_manifest",
            "observation_manifest",
            "gene_outcomes",
            "candidate_set_manifest",
            "outcome_quality",
            "split_audit",
        )
    }
    for prefix, bindings, expected in (
        ("input", summary.get("input_bindings"), expected_inputs),
        ("implementation", summary.get("implementation_bindings"), expected_implementation),
        ("output", summary.get("output_bindings"), expected_outputs),
    ):
        if not isinstance(bindings, Mapping):
            blockers.append(f"{prefix}_bindings_missing")
            continue
        if set(bindings) != set(expected):
            blockers.append(f"{prefix}_binding_contract")
            continue
        for name, path in expected.items():
            if not _binding_matches(bindings.get(name), path):
                blockers.append(f"{prefix}_binding_drift:{name}")
    if summary.get("agent_in_twin_authorized") is not False or summary.get("coevolution_authorized") is not False:
        blockers.append("downstream_authorization")
    if summary.get("model_predictions_inspected") is not False or summary.get("publication_comparison_open") is not False:
        blockers.append("scope_boundary")

    unique = sorted(set(blockers))
    return {
        "schema_version": "neuronsifter-targeted-therapeutic-decision-domain-validation-v1",
        "status": "passed" if not unique else "blocked",
        "blocker_count": len(unique),
        "blockers": unique,
        "source_count": len(source_rows),
        "observation_count": len(observation_rows),
        "objective_gene_outcome_count": len(gene_rows),
        "candidate_set_count": len(candidate_rows),
        "qualified_decision_candidate_set_count": qualified_count,
        "decision_sensitive_domain_available": qualified_count > 0,
        "model_training_domain_qualified": split_created,
        "model_training_authorized": split_created,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
    }


def main() -> int:
    result = validate()
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
