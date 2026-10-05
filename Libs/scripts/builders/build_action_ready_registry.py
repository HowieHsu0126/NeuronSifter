from __future__ import annotations

import argparse
import csv
import gzip
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, read_tsv, resolve_repo_path, write_json, write_tsv
from Libs.utils.data.screening_pipeline import canonical_smiles_key, normalize_text, stable_uid
from Libs.utils.evaluation.strict_case_discovery import action_ready_manifest_path, action_ready_manifest_payload, assert_no_forced_anchor_overlap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the action-ready registry for the AD screening system pipeline.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def clue_proxy_scores() -> dict[str, float]:
    pertinfo_path = resolve_repo_path("Input/raw/clue_lincs/downloads/builds/TouchstoneV1.1/files/GSE92742_Broad_LINCS_pert_info.txt.gz")
    siginfo_path = resolve_repo_path("Input/raw/clue_lincs/downloads/builds/TouchstoneV1.1/files/GSE92742_Broad_LINCS_sig_info.txt.gz")
    pert_by_id: dict[str, dict[str, str]] = {}
    with gzip.open(pertinfo_path, "rt", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            pert_by_id[str(row["pert_id"])] = {
                "pert_iname": str(row["pert_iname"]),
                "inchi_key": str(row["inchi_key"]),
                "canonical_smiles": canonical_smiles_key(str(row["canonical_smiles"])),
            }
    signature_counts: dict[str, int] = defaultdict(int)
    with gzip.open(siginfo_path, "rt", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            signature_counts[str(row["pert_id"])] += 1
    max_signature_count = max(signature_counts.values()) if signature_counts else 1
    score_by_key: dict[str, float] = {}
    for pert_id, count in signature_counts.items():
        metadata = pert_by_id.get(pert_id, {})
        score = float(count / max_signature_count)
        for key in [
            str(metadata.get("inchi_key", "")).strip(),
            canonical_smiles_key(str(metadata.get("canonical_smiles", ""))),
            normalize_text(str(metadata.get("pert_iname", ""))),
        ]:
            if key and key not in score_by_key:
                score_by_key[key] = score
    return score_by_key


def forced_identifier_set(values: list[object]) -> set[str]:
    return {
        normalize_text(str(value))
        for value in values
        if str(value).strip()
    }


def join_unique(values: list[str]) -> str:
    return "|".join(sorted({value for value in values if value}))


def case_study_identifier_sets(config: dict) -> set[str]:
    compound_keys: set[str] = set()
    for validation_row in config["dynamics_ready_registry"]["case_validation_entries"]:
        intervention_label = str(validation_row.get("intervention_label", "")).strip()
        if intervention_label:
            compound_keys.add(normalize_text(intervention_label))
    for case_config in config.get("case_study_registry", {}).values():
        for compound_name in case_config.get("anchor_compounds", []):
            compound_keys.add(normalize_text(str(compound_name)))
    return compound_keys


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    assert_no_forced_anchor_overlap(config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    action_root = resolve_repo_path(config["screening_pipeline"]["action_ready_registry"]["local_root"])
    ensure_directory(action_root)
    drug_target_root = resolve_repo_path(config["screening_pipeline"]["drug_target_ready_registry"]["local_root"])
    pkpd_root = resolve_repo_path(config["screening_pipeline"]["pkpd_ready_registry"]["local_root"])
    relative_paths = config["screening_pipeline"]["action_ready_registry"]["relative_paths"]
    action_config = config["screening_pipeline"]["action_ready_registry"]
    minimum_coverage = float(action_config["minimum_atlas_program_coverage"])
    minimum_hits = int(action_config["minimum_atlas_program_feature_hits"])
    maximum_single_compounds = int(action_config["maximum_single_compounds"])
    maximum_pairwise_compounds = int(action_config["maximum_pairwise_compounds"])
    maximum_pairwise_regimens = int(action_config["maximum_pairwise_regimens"])
    forced_compound_keys = forced_identifier_set(action_config.get("forced_include_compounds", []))
    forced_target_keys = forced_identifier_set(action_config.get("forced_include_targets", []))
    case_compound_keys = case_study_identifier_sets(config)

    atlas_root = resolve_repo_path(config["atlas_prior_registry"]["local_root"])
    alignment_root = resolve_repo_path(config["alignment_ready_registry"]["local_root"])
    atlas_rows = read_tsv(atlas_root / config["atlas_prior_registry"]["relative_paths"]["program_catalog"])
    target_rows = read_tsv(drug_target_root / config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]["target_catalog"])
    evidence_rows = read_tsv(drug_target_root / config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]["target_evidence_catalog"])
    compound_rows = read_tsv(drug_target_root / config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]["compound_catalog"])
    edge_rows = read_tsv(drug_target_root / config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]["drug_target_edge_catalog"])
    compound_pk_rows = read_tsv(pkpd_root / config["screening_pipeline"]["pkpd_ready_registry"]["relative_paths"]["compound_pk_prior_catalog"])
    entity_rows = read_tsv(alignment_root / config["alignment_ready_registry"]["relative_paths"]["entity_catalog"])
    pipeline_logger.progress(
        "loaded_action_ready_inputs",
        atlas_program_count=len(atlas_rows),
        target_count=len(target_rows),
        compound_count=len(compound_rows),
        edge_count=len(edge_rows),
    )

    target_uid_by_entity = {row["canonical_entity_id"]: row["target_uid"] for row in target_rows}
    target_symbol_by_uid = {row["target_uid"]: row["canonical_entity_label"] for row in target_rows}
    evidence_score_by_target = defaultdict(float)
    for row in evidence_rows:
        evidence_score_by_target[row["target_uid"]] = max(evidence_score_by_target[row["target_uid"]], float(row["evidence_score"]))

    gene_entities = {
        row["canonical_entity_label"]: row["canonical_entity_id"]
        for row in entity_rows
        if row["entity_type"] == "gene"
    }

    atlas_mapping_rows: list[dict[str, object]] = []
    target_program_rows: list[dict[str, object]] = []
    exclusion_rows: list[dict[str, object]] = []
    atlas_program_payload: dict[str, dict[str, object]] = {}
    for atlas_row in atlas_rows:
        feature_ids = [value for value in str(atlas_row["feature_ids"]).split("|") if value]
        canonical_hits = [(gene_entities[feature_id], feature_id) for feature_id in feature_ids if feature_id in gene_entities]
        coverage_fraction = (len(canonical_hits) / len(feature_ids)) if feature_ids else 0.0
        inclusion_flag = "yes" if coverage_fraction >= minimum_coverage and len(canonical_hits) >= minimum_hits else "no"
        exclusion_reason = "" if inclusion_flag == "yes" else "insufficient_feature_mapping_coverage"
        if inclusion_flag == "no":
            exclusion_rows.append(
                {
                    "exclusion_uid": stable_uid("atlas_program_exclusion", atlas_row["atlas_entity_uid"]),
                    "exclusion_scope": "atlas_program",
                    "source_record_id": atlas_row["atlas_entity_uid"],
                    "exclusion_reason": exclusion_reason,
                    "note": f"coverage={coverage_fraction:.4f}; hits={len(canonical_hits)}",
                }
            )
        for canonical_entity_id, feature_id in canonical_hits:
            atlas_mapping_rows.append(
                {
                    "atlas_entity_uid": atlas_row["atlas_entity_uid"],
                    "atlas_name": atlas_row["atlas_name"],
                    "program_id": atlas_row["program_id"],
                    "canonical_entity_id": canonical_entity_id,
                    "canonical_entity_label": feature_id,
                    "feature_hit_count": len(canonical_hits),
                    "feature_total_count": len(feature_ids),
                    "coverage_fraction": coverage_fraction,
                    "inclusion_flag": inclusion_flag,
                    "exclusion_reason": exclusion_reason,
                    "note": atlas_row["program_label"],
                }
            )
        if inclusion_flag == "yes":
            atlas_program_payload[atlas_row["atlas_entity_uid"]] = {
                "atlas_row": atlas_row,
                "canonical_hits": canonical_hits,
                "coverage_fraction": coverage_fraction,
            }

    for atlas_entity_uid, payload in atlas_program_payload.items():
        atlas_row = payload["atlas_row"]
        coverage_fraction = payload["coverage_fraction"]
        for canonical_entity_id, feature_id in payload["canonical_hits"]:
            if canonical_entity_id not in target_uid_by_entity:
                continue
            target_uid = target_uid_by_entity[canonical_entity_id]
            target_program_rows.append(
                {
                    "target_program_uid": stable_uid("target_program", atlas_entity_uid, canonical_entity_id),
                    "canonical_entity_id": canonical_entity_id,
                    "canonical_entity_label": feature_id,
                    "atlas_entity_uid": atlas_entity_uid,
                    "atlas_name": atlas_row["atlas_name"],
                    "program_id": atlas_row["program_id"],
                    "mapped_feature_ids": feature_id,
                    "mapped_feature_count": 1,
                    "coverage_fraction": coverage_fraction,
                    "target_evidence_score": evidence_score_by_target[target_uid],
                    "target_class": next((row["target_class"] for row in target_rows if row["target_uid"] == target_uid), ""),
                    "source_repository": "atlas+drug_target",
                    "note": atlas_row["program_label"],
                }
            )

    covered_target_entity_ids = {
        str(row["canonical_entity_id"])
        for row in target_program_rows
    }
    for target_row in target_rows:
        canonical_entity_id = str(target_row["canonical_entity_id"])
        if canonical_entity_id in covered_target_entity_ids:
            continue
        target_uid = str(target_row["target_uid"])
        target_program_rows.append(
            {
                "target_program_uid": stable_uid("target_program_direct", target_uid),
                "canonical_entity_id": canonical_entity_id,
                "canonical_entity_label": str(target_row["canonical_entity_label"]),
                "atlas_entity_uid": "",
                "atlas_name": "",
                "program_id": "",
                "mapped_feature_ids": "",
                "mapped_feature_count": 0,
                "coverage_fraction": 0.0,
                "target_evidence_score": evidence_score_by_target[target_uid],
                "target_class": str(target_row["target_class"]),
                "source_repository": "drug_target_direct",
                "note": "direct_target_evidence",
            }
        )

    pk_ready_compounds = {row["compound_uid"] for row in compound_pk_rows}
    clue_score_by_key = clue_proxy_scores()
    edge_rows_by_compound: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in edge_rows:
        edge_rows_by_compound[row["compound_uid"]].append(row)
    compound_lookup = {row["compound_uid"]: row for row in compound_rows}

    drug_program_rows: list[dict[str, object]] = []
    for compound_row in sorted(compound_rows, key=lambda row: row["compound_uid"]):
        compound_uid = compound_row["compound_uid"]
        current_edges = edge_rows_by_compound.get(compound_uid, [])
        if not current_edges:
            exclusion_rows.append(
                {
                    "exclusion_uid": stable_uid("compound_without_targets", compound_uid),
                    "exclusion_scope": "drug_program",
                    "source_record_id": compound_uid,
                    "exclusion_reason": "compound_without_mapped_targets",
                    "note": compound_row["preferred_name"],
                }
            )
            continue

        target_uid_set = sorted({str(row["target_uid"]) for row in current_edges})
        target_symbol_set = sorted({target_symbol_by_uid[target_uid] for target_uid in target_uid_set})
        evidence_score = max(float(row["evidence_score"]) for row in current_edges)
        affinity_values = [
            float(row["affinity_value"])
            for row in current_edges
            if str(row["affinity_value"]).strip() not in {"", "nan"}
        ]
        aggregate_affinity_score = min(affinity_values) if affinity_values else np.nan
        clue_proxy_score = 0.0
        for key in [
            compound_row["inchi_key"],
            canonical_smiles_key(compound_row["canonical_smiles"]),
            normalize_text(compound_row["preferred_name"]),
        ]:
            if key and key in clue_score_by_key:
                clue_proxy_score = max(clue_proxy_score, float(clue_score_by_key[key]))

        compound_identifiers = [
            compound_uid,
            compound_row.get("preferred_name", ""),
            compound_row.get("canonical_compound_id", ""),
            compound_row.get("chembl_id", ""),
        ]
        compound_match = any(
            normalize_text(str(identifier)) in forced_compound_keys
            for identifier in compound_identifiers
            if str(identifier).strip()
        )
        target_match = any(
            normalize_text(target_symbol) in forced_target_keys
            for target_symbol in target_symbol_set
            if target_symbol
        )
        case_compound_match = any(
            normalize_text(str(identifier)) in case_compound_keys
            for identifier in compound_identifiers
            if str(identifier).strip()
        )
        if compound_match or target_match:
            selection_reason = "forced_include"
        elif case_compound_match:
            selection_reason = "case_study_seed"
        else:
            selection_reason = "ranked_selection"
        drug_program_rows.append(
            {
                "drug_program_uid": stable_uid("drug_program", compound_uid),
                "compound_uid": compound_uid,
                "target_uid_set": "|".join(target_uid_set),
                "target_symbol_set": "|".join(target_symbol_set),
                "target_count": len(target_uid_set),
                "aggregate_target_evidence_score": evidence_score,
                "aggregate_affinity_score": "" if np.isnan(aggregate_affinity_score) else aggregate_affinity_score,
                "clue_reversal_proxy_score": clue_proxy_score,
                "pk_prior_ready_flag": "yes" if compound_uid in pk_ready_compounds else "no",
                "murcko_scaffold": compound_row["murcko_scaffold"],
                "selection_reason": selection_reason,
                "note": compound_row["preferred_name"],
            }
        )

    drug_program_rows.sort(
        key=lambda row: (
            0 if row["selection_reason"] == "forced_include" else 1,
            -float(row["aggregate_target_evidence_score"]),
            row["drug_program_uid"],
        )
    )
    selected_single_rows = drug_program_rows[:maximum_single_compounds]
    selected_single_lookup = {row["drug_program_uid"] for row in selected_single_rows}
    for row in drug_program_rows:
        if row["selection_reason"] in {"forced_include", "case_study_seed"} and row["drug_program_uid"] not in selected_single_lookup:
            selected_single_rows.append(row)
            selected_single_lookup.add(row["drug_program_uid"])

    route_rows = read_tsv(pkpd_root / config["screening_pipeline"]["pkpd_ready_registry"]["relative_paths"]["route_catalog"])
    schedule_rows = read_tsv(pkpd_root / config["screening_pipeline"]["pkpd_ready_registry"]["relative_paths"]["schedule_catalog"])
    dose_rows = read_tsv(pkpd_root / config["screening_pipeline"]["pkpd_ready_registry"]["relative_paths"]["dose_grid_catalog"])
    regimen_rows: list[dict[str, object]] = []

    def pk_feasibility_score(drug_program_row: dict[str, object], route: str, schedule: str, dose_multiplier: float) -> float:
        route_scale = next(float(row["absorption_scale"]) for row in route_rows if row["route"] == route)
        schedule_scale = next(float(row["exposure_scale"]) for row in schedule_rows if row["schedule"] == schedule)
        pk_ready = 1.0 if drug_program_row["pk_prior_ready_flag"] == "yes" else 0.5
        return float(min(1.0, pk_ready * route_scale * schedule_scale * dose_multiplier / 2.0))

    for drug_program_row in selected_single_rows:
        for route_row in route_rows:
            for schedule_row in schedule_rows:
                for dose_row in dose_rows:
                    regimen_rows.append(
                        {
                            "regimen_uid": stable_uid("regimen", drug_program_row["drug_program_uid"], route_row["route"], schedule_row["schedule"], dose_row["dose_level_id"]),
                            "drug_program_uid_set": drug_program_row["drug_program_uid"],
                            "compound_uid_set": drug_program_row["compound_uid"],
                            "target_uid_set": drug_program_row["target_uid_set"],
                            "target_symbol_set": drug_program_row["target_symbol_set"],
                            "regimen_scaffold_set": drug_program_row["murcko_scaffold"],
                            "route": route_row["route"],
                            "schedule": schedule_row["schedule"],
                            "dose_level_id": dose_row["dose_level_id"],
                            "dose_multiplier": dose_row["dose_multiplier"],
                            "aggregate_target_evidence_score": drug_program_row["aggregate_target_evidence_score"],
                            "aggregate_affinity_score": drug_program_row["aggregate_affinity_score"],
                            "clue_reversal_proxy_score": drug_program_row["clue_reversal_proxy_score"],
                            "pk_feasibility_score": pk_feasibility_score(drug_program_row, route_row["route"], schedule_row["schedule"], float(dose_row["dose_multiplier"])),
                            "regimen_size": 1,
                            "selection_reason": drug_program_row["selection_reason"],
                            "note": "single_compound_regimen",
                        }
                    )

    pairwise_candidates = selected_single_rows[:maximum_pairwise_compounds]
    pair_counter = 0
    for left_row, right_row in combinations(pairwise_candidates, 2):
        if pair_counter >= maximum_pairwise_regimens:
            break
        pair_counter += 1
        for route_row in route_rows:
            for schedule_row in schedule_rows:
                for dose_row in dose_rows:
                    regimen_rows.append(
                        {
                            "regimen_uid": stable_uid("regimen_pair", left_row["drug_program_uid"], right_row["drug_program_uid"], route_row["route"], schedule_row["schedule"], dose_row["dose_level_id"]),
                            "drug_program_uid_set": join_unique([left_row["drug_program_uid"], right_row["drug_program_uid"]]),
                            "compound_uid_set": join_unique([left_row["compound_uid"], right_row["compound_uid"]]),
                            "target_uid_set": join_unique((left_row["target_uid_set"] + "|" + right_row["target_uid_set"]).split("|")),
                            "target_symbol_set": join_unique((left_row["target_symbol_set"] + "|" + right_row["target_symbol_set"]).split("|")),
                            "regimen_scaffold_set": join_unique([left_row["murcko_scaffold"], right_row["murcko_scaffold"]]),
                            "route": route_row["route"],
                            "schedule": schedule_row["schedule"],
                            "dose_level_id": dose_row["dose_level_id"],
                            "dose_multiplier": dose_row["dose_multiplier"],
                            "aggregate_target_evidence_score": (float(left_row["aggregate_target_evidence_score"]) + float(right_row["aggregate_target_evidence_score"])) / 2.0,
                            "aggregate_affinity_score": left_row["aggregate_affinity_score"] or right_row["aggregate_affinity_score"],
                            "clue_reversal_proxy_score": max(float(left_row["clue_reversal_proxy_score"]), float(right_row["clue_reversal_proxy_score"])),
                            "pk_feasibility_score": min(
                                pk_feasibility_score(left_row, route_row["route"], schedule_row["schedule"], float(dose_row["dose_multiplier"])),
                                pk_feasibility_score(right_row, route_row["route"], schedule_row["schedule"], float(dose_row["dose_multiplier"])),
                            ),
                            "regimen_size": 2,
                            "selection_reason": (
                                "forced_include"
                                if "forced_include" in {left_row["selection_reason"], right_row["selection_reason"]}
                                else (
                                    "pairwise_case_study_seed"
                                    if "case_study_seed" in {left_row["selection_reason"], right_row["selection_reason"]}
                                    else "pairwise_ranked_selection"
                                )
                            ),
                            "note": "pairwise_regimen",
                        }
                    )

    write_tsv(action_root / relative_paths["atlas_program_feature_mapping"], config["schemas"]["action_ready_atlas_mapping_fields"], atlas_mapping_rows)
    write_tsv(action_root / relative_paths["target_program_catalog"], config["schemas"]["action_ready_target_program_fields"], target_program_rows)
    write_tsv(action_root / relative_paths["drug_program_catalog"], config["schemas"]["action_ready_drug_program_fields"], drug_program_rows)
    write_tsv(action_root / relative_paths["regimen_catalog"], config["schemas"]["action_ready_regimen_fields"], regimen_rows)
    write_tsv(action_root / relative_paths["action_exclusion_catalog"], config["schemas"]["action_ready_exclusion_fields"], exclusion_rows)
    manifest_path = action_ready_manifest_path(config)
    write_json(manifest_path, action_ready_manifest_payload(config))
    pipeline_logger.artifact(
        "action_ready_registry_written",
        artifact=action_root,
        manifest_path=str(manifest_path),
        atlas_mapping_count=len(atlas_mapping_rows),
        target_program_count=len(target_program_rows),
        drug_program_count=len(drug_program_rows),
        selected_single_count=len(selected_single_rows),
        regimen_count=len(regimen_rows),
        exclusion_count=len(exclusion_rows),
    )
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=action_root,
        manifest_path=str(manifest_path),
        atlas_mapping_count=len(atlas_mapping_rows),
        target_program_count=len(target_program_rows),
        drug_program_count=len(drug_program_rows),
        selected_single_count=len(selected_single_rows),
        regimen_count=len(regimen_rows),
        exclusion_count=len(exclusion_rows),
    )


if __name__ == "__main__":
    main()
