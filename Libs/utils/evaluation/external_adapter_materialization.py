from __future__ import annotations

import csv
import importlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import requests

from Libs.utils.data.platform_contracts import stable_uid
from Libs.utils.data.raw_data_registry import (
    ensure_directory,
    read_tsv,
    repo_root,
    resolve_repo_path,
    sha256_of_file,
    today_iso,
    write_json,
    write_tsv,
)


def external_adapter_registry_config(config: dict) -> dict[str, object]:
    return config["validation"]["external_adapter_registry"]


def external_adapter_materialization_config(config: dict, source_key: str) -> dict[str, object]:
    return external_adapter_registry_config(config)["materialization"][source_key]


def external_adapter_output_path(config: dict, relative_path_key: str) -> Path:
    registry = external_adapter_registry_config(config)
    return resolve_repo_path(str(registry["local_root"])) / str(registry["relative_paths"][relative_path_key])


def artifact_path_text(path: Path) -> str:
    resolved_path = path.resolve()
    root = repo_root()
    if resolved_path == root or root in resolved_path.parents:
        return resolved_path.relative_to(root).as_posix()
    return resolved_path.as_posix()


def normalized_name(value: object) -> str:
    return " ".join(str(value).upper().replace("-", " ").replace("_", " ").split())


def alias_values(value: object, separators: list[str]) -> set[str]:
    aliases = {normalized_name(value)}
    for separator in separators:
        for token in str(value).split(separator):
            normalized = normalized_name(token)
            if normalized:
                aliases.add(normalized)
    return aliases


def suffix_stripped_aliases(value: str, suffixes: list[str]) -> set[str]:
    aliases = {value}
    for suffix in suffixes:
        suffix_text = normalized_name(suffix)
        if value.endswith(f" {suffix_text}"):
            aliases.add(value[: -len(suffix_text)].strip())
    return {alias for alias in aliases if alias}


def split_pipe(value: object, separator: str) -> list[str]:
    return [token.strip() for token in str(value).split(separator) if token.strip()]


def dgidb_target_universe(config: dict) -> tuple[list[str], dict[str, str]]:
    dgidb_config = external_adapter_materialization_config(config, "dgidb")
    target_rows = read_tsv(resolve_repo_path(str(dgidb_config["target_catalog_path"])))
    intervention_rows = read_tsv(resolve_repo_path(str(dgidb_config["intervention_program_catalog_path"])))
    target_lookup: dict[str, str] = {}
    for row in target_rows:
        for field_name in ["target_symbol", "canonical_entity_label"]:
            symbol = normalized_name(row[field_name])
            if symbol:
                target_lookup[symbol] = str(row["target_uid"])
    requested_symbols: set[str] = set()
    for row in intervention_rows:
        for label in split_pipe(row["target_program_label_set"], str(dgidb_config["target_label_separator"])):
            normalized = normalized_name(label)
            if normalized in target_lookup:
                requested_symbols.add(normalized)
    if not requested_symbols:
        raise ValueError("DGIdb materialization found no target symbols in the intervention-program target universe.")
    return sorted(requested_symbols), target_lookup


def dgidb_drug_universe(config: dict) -> dict[str, str]:
    dgidb_config = external_adapter_materialization_config(config, "dgidb")
    compound_rows = read_tsv(resolve_repo_path(str(dgidb_config["compound_catalog_path"])))
    intervention_rows = read_tsv(resolve_repo_path(str(dgidb_config["intervention_program_catalog_path"])))
    intervention_compounds: set[str] = set()
    for row in intervention_rows:
        intervention_compounds.update(split_pipe(row["compound_set"], str(dgidb_config["compound_set_separator"])))
    alias_separators = [str(value) for value in dgidb_config["drug_name_alias_separators"]]
    suffixes = [str(value) for value in dgidb_config["drug_name_suffixes_to_strip"]]
    drug_lookup: dict[str, str] = {}
    for row in compound_rows:
        compound_uid = str(row["compound_uid"])
        if compound_uid not in intervention_compounds:
            continue
        aliases: set[str] = set()
        for alias in alias_values(row["preferred_name"], alias_separators):
            aliases.update(suffix_stripped_aliases(alias, suffixes))
        for alias in aliases:
            drug_lookup[alias] = compound_uid
    if not drug_lookup:
        raise ValueError("DGIdb materialization found no compound names in the intervention-program drug universe.")
    return drug_lookup


def dgidb_batches(values: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("DGIdb query_batch_size must be positive.")
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def dgidb_graphql_payload(query: str, gene_symbols: list[str]) -> dict[str, object]:
    return {
        "query": query,
        "variables": {"names": gene_symbols},
    }


def request_dgidb_batch(
    session: requests.Session,
    api_url: str,
    query: str,
    gene_symbols: list[str],
    client_name: str,
    timeout_seconds: float,
) -> dict[str, object]:
    response = session.post(
        api_url,
        json=dgidb_graphql_payload(query, gene_symbols),
        headers={"dgidb-client-name": client_name},
        timeout=timeout_seconds,
    )
    if response.status_code != 200:
        raise RuntimeError(f"DGIdb API request failed with status {response.status_code}: {response.text[:500]}")
    payload = response.json()
    if "errors" in payload:
        raise RuntimeError(f"DGIdb API returned GraphQL errors: {json.dumps(payload['errors'], ensure_ascii=True)}")
    return payload


def parse_dgidb_gene_nodes(
    nodes: list[dict[str, object]],
    target_lookup: dict[str, str],
    drug_lookup: dict[str, str],
    dgidb_config: dict[str, object],
    retrieval_date: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    require_drug_overlap = bool(dgidb_config["require_drug_universe_overlap"])
    for node in nodes:
        query_gene = normalized_name(node["name"])
        matched_target_uid = target_lookup.get(query_gene, "")
        for interaction in node["interactions"]:
            drug = interaction["drug"]
            gene = interaction["gene"]
            drug_name = normalized_name(drug["name"])
            matched_compound_uid = drug_lookup.get(drug_name, "")
            drug_in_universe = "yes" if matched_compound_uid else "no"
            if require_drug_overlap and not matched_compound_uid:
                continue
            interaction_types = [
                str(row["type"])
                for row in interaction["interactionTypes"]
                if str(row.get("type", "")).strip()
            ]
            directionality = [
                str(row["directionality"])
                for row in interaction["interactionTypes"]
                if str(row.get("directionality", "")).strip()
            ]
            sources = [
                str(row["fullName"])
                for row in interaction["sources"]
                if str(row.get("fullName", "")).strip()
            ]
            pmids = [
                str(row["pmid"])
                for row in interaction["publications"]
                if str(row.get("pmid", "")).strip()
            ]
            citations = [
                str(row["citation"]).replace("\t", " ").replace("\n", " ")
                for row in interaction["publications"]
                if str(row.get("citation", "")).strip()
            ]
            gene_concept_id = str(gene["conceptId"])
            rows.append(
                {
                    "interaction_uid": stable_uid("dgidb_interaction", interaction["id"], matched_target_uid, matched_compound_uid),
                    "source_key": "dgidb",
                    "source_release": str(dgidb_config["source_release"]),
                    "retrieval_date": retrieval_date,
                    "query_scope": str(dgidb_config["query_scope"]),
                    "query_gene_symbol": query_gene,
                    "query_drug_name": drug_name,
                    "gene_name": str(gene["name"]),
                    "gene_concept_id": gene_concept_id,
                    "drug_name": str(drug["name"]),
                    "drug_concept_id": str(drug["conceptId"]),
                    "matched_target_uid": matched_target_uid,
                    "matched_compound_uid": matched_compound_uid,
                    "drug_in_neuronsifter_universe": drug_in_universe,
                    "interaction_id": str(interaction["id"]),
                    "interaction_score": float(interaction["interactionScore"]),
                    "interaction_type_set": "|".join(sorted(set(interaction_types))),
                    "directionality_set": "|".join(sorted(set(directionality))),
                    "source_db_set": "|".join(sorted(set(sources))),
                    "publication_pmid_set": "|".join(sorted(set(pmids))),
                    "publication_citation_set": "|".join(citations),
                    "provenance_level": "dgidb_api_target_universe_query",
                    "source_holdout_role": str(dgidb_config["source_holdout_role"]),
                    "train_flag": str(dgidb_config["train_flag"]),
                    "eval_flag": str(dgidb_config["eval_flag"]),
                    "public_url": str(dgidb_config["public_url_template"]).format(gene_concept_id=gene_concept_id),
                    "note": "DGIdb GraphQL interaction matched to NeuronSifter target/drug universe.",
                }
            )
    return rows


def build_dgidb_processed_catalog(config: dict, session: requests.Session | None = None) -> Path:
    dgidb_config = external_adapter_materialization_config(config, "dgidb")
    target_symbols, target_lookup = dgidb_target_universe(config)
    drug_lookup = dgidb_drug_universe(config)
    active_session = requests.Session() if session is None else session
    if session is None:
        active_session.trust_env = bool(dgidb_config["trust_environment_proxy"])
    retrieval_date = today_iso()
    rows: list[dict[str, object]] = []
    for gene_batch in dgidb_batches(target_symbols, int(dgidb_config["query_batch_size"])):
        payload = request_dgidb_batch(
            active_session,
            str(dgidb_config["api_url"]),
            str(dgidb_config["graphql_query"]),
            gene_batch,
            str(dgidb_config["request_header_client_name"]),
            float(dgidb_config["request_timeout_seconds"]),
        )
        nodes = payload["data"]["genes"]["nodes"]
        rows.extend(parse_dgidb_gene_nodes(nodes, target_lookup, drug_lookup, dgidb_config, retrieval_date))
    if not rows:
        raise ValueError("DGIdb API materialization produced no interactions after target/drug universe filtering.")
    output_path = external_adapter_output_path(config, "dgidb_drug_gene_interaction_catalog")
    write_tsv(output_path, config["schemas"]["dgidb_drug_gene_interaction_fields"], sorted(rows, key=lambda row: (row["query_gene_symbol"], row["drug_name"], row["interaction_id"])))
    manifest_path = external_adapter_output_path(config, "dgidb_api_query_manifest")
    write_json(
        manifest_path,
        {
            "source_key": "dgidb",
            "api_url": str(dgidb_config["api_url"]),
            "source_release": str(dgidb_config["source_release"]),
            "retrieval_date": retrieval_date,
            "target_symbol_count": len(target_symbols),
            "drug_alias_count": len(drug_lookup),
            "processed_interaction_count": len(rows),
            "output_path": artifact_path_text(output_path),
        },
    )
    return output_path


def finite_float(value: object, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite numeric value for {field_name}: {value}")
    return number


def external_adapter_row(config: dict, source_key: str) -> dict[str, object]:
    for row in external_adapter_registry_config(config)["adapters"]:
        if str(row["source_key"]) == source_key:
            return row
    raise ValueError(f"External adapter materialization config is missing source_key={source_key}.")


def external_adapter_processed_artifact_path(config: dict, source_key: str) -> Path:
    return resolve_repo_path(str(external_adapter_row(config, source_key)["processed_artifact_path"]))


def import_required_python_package(package_name: str) -> object:
    try:
        return importlib.import_module(package_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"Required external adapter Python package is unavailable: {package_name}") from exc


def load_compound_descriptor_matrix(
    feature_catalog_path: Path,
    descriptor_fields: list[str],
) -> tuple[list[str], np.ndarray]:
    if not feature_catalog_path.exists():
        raise FileNotFoundError(f"MoleculeNet/DeepChem descriptor feature catalog is missing: {feature_catalog_path}")
    descriptor_set = set(descriptor_fields)
    compound_features: dict[str, dict[str, float]] = {}
    with feature_catalog_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            feature_name = str(row["feature_name"])
            if feature_name not in descriptor_set:
                continue
            compound_uid = str(row["compound_uid"])
            compound_features.setdefault(compound_uid, {})[feature_name] = finite_float(row["feature_value"], feature_name)
    if not compound_features:
        raise ValueError(f"MoleculeNet/DeepChem descriptor feature catalog has no configured descriptor fields: {feature_catalog_path}")
    incomplete = [
        compound_uid
        for compound_uid, feature_values in compound_features.items()
        if set(feature_values) != descriptor_set
    ]
    if incomplete:
        raise ValueError(
            "MoleculeNet/DeepChem descriptor feature catalog has incomplete descriptor coverage for "
            f"{len(incomplete)} compounds; first={incomplete[0]}"
        )
    compound_uids = sorted(compound_features)
    matrix = np.asarray(
        [
            [compound_features[compound_uid][descriptor_name] for descriptor_name in descriptor_fields]
            for compound_uid in compound_uids
        ],
        dtype=np.float64,
    )
    return compound_uids, matrix


def descriptor_index(descriptor_fields: list[str]) -> dict[str, int]:
    return {descriptor_name: index for index, descriptor_name in enumerate(descriptor_fields)}


def molecular_property_scores(
    feature_matrix: np.ndarray,
    descriptor_fields: list[str],
    baseline_config: dict[str, object],
) -> dict[str, np.ndarray]:
    indices = descriptor_index(descriptor_fields)
    thresholds = baseline_config["cns_rule_thresholds"]
    lipinski_thresholds = baseline_config["lipinski_thresholds"]
    mw = feature_matrix[:, indices["molecular_weight"]]
    alogp = feature_matrix[:, indices["alogp"]]
    tpsa = feature_matrix[:, indices["tpsa"]]
    hba = feature_matrix[:, indices["hba"]]
    hbd = feature_matrix[:, indices["hbd"]]
    rotatable = feature_matrix[:, indices["rotatable_bonds"]]
    rings = feature_matrix[:, indices["ring_count"]]
    cns_rules = np.vstack(
        [
            mw <= float(thresholds["molecular_weight_upper"]),
            alogp >= float(thresholds["alogp_lower"]),
            alogp <= float(thresholds["alogp_upper"]),
            tpsa <= float(thresholds["tpsa_upper"]),
            hba <= float(thresholds["hba_upper"]),
            hbd <= float(thresholds["hbd_upper"]),
            rotatable <= float(thresholds["rotatable_bonds_upper"]),
            rings >= float(thresholds["ring_count_lower"]),
        ]
    ).T
    lipinski_rules = np.vstack(
        [
            mw <= float(lipinski_thresholds["molecular_weight_upper"]),
            alogp <= float(lipinski_thresholds["alogp_upper"]),
            hba <= float(lipinski_thresholds["hba_upper"]),
            hbd <= float(lipinski_thresholds["hbd_upper"]),
        ]
    ).T
    alogp_lower = float(thresholds["alogp_lower"])
    alogp_upper = float(thresholds["alogp_upper"])
    lipophilicity_window_score = np.where(
        alogp < alogp_lower,
        np.maximum(0.0, 1.0 - (alogp_lower - alogp) / max(abs(alogp_lower), 1.0)),
        np.where(
            alogp > alogp_upper,
            np.maximum(0.0, 1.0 - (alogp - alogp_upper) / max(abs(alogp_upper), 1.0)),
            1.0,
        ),
    )
    polar_surface_penalty = np.clip(tpsa / float(thresholds["tpsa_upper"]), 0.0, 1.0)
    cns_rule_pass_count = cns_rules.sum(axis=1).astype(np.int64)
    cns_rule_total_count = np.full(feature_matrix.shape[0], cns_rules.shape[1], dtype=np.int64)
    return {
        "lipinski_violation_count": lipinski_rules.shape[1] - lipinski_rules.sum(axis=1),
        "cns_rule_pass_count": cns_rule_pass_count,
        "cns_rule_total_count": cns_rule_total_count,
        "cns_property_score": cns_rule_pass_count / cns_rule_total_count,
        "lipophilicity_window_score": lipophilicity_window_score,
        "polar_surface_penalty": polar_surface_penalty,
    }


def build_moleculenet_deepchem_baseline_catalog(config: dict) -> Path:
    baseline_config = external_adapter_materialization_config(config, "moleculenet_deepchem")
    deepchem_module = import_required_python_package("deepchem")
    descriptor_fields = [str(value) for value in baseline_config["descriptor_fields"]]
    feature_catalog_path = resolve_repo_path(str(baseline_config["compound_feature_catalog_path"]))
    compound_uids, feature_matrix = load_compound_descriptor_matrix(feature_catalog_path, descriptor_fields)
    scores = molecular_property_scores(feature_matrix, descriptor_fields, baseline_config)
    probe_count = min(int(baseline_config["deepchem_probe_row_count"]), feature_matrix.shape[0])
    if probe_count <= 0:
        raise ValueError("MoleculeNet/DeepChem deepchem_probe_row_count must be positive.")
    dataset_class = getattr(deepchem_module.data, str(baseline_config["deepchem_dataset_class"]))
    probe_dataset = dataset_class(
        X=feature_matrix[:probe_count],
        y=np.asarray(scores["cns_property_score"][:probe_count], dtype=np.float64).reshape(-1, 1),
        ids=np.asarray(compound_uids[:probe_count], dtype=object),
    )
    if len(probe_dataset) != probe_count:
        raise ValueError("DeepChem NumpyDataset probe row count does not match descriptor matrix.")
    materialization_date = today_iso()
    indices = descriptor_index(descriptor_fields)
    rows: list[dict[str, object]] = []
    for row_index, compound_uid in enumerate(compound_uids):
        row = {
            "baseline_uid": stable_uid("moleculenet_deepchem_baseline", compound_uid, str(baseline_config["source_release"])),
            "source_key": "moleculenet_deepchem",
            "source_release": str(baseline_config["source_release"]),
            "materialization_date": materialization_date,
            "compound_uid": compound_uid,
            "feature_count": len(descriptor_fields),
            "molecular_weight": feature_matrix[row_index, indices["molecular_weight"]],
            "alogp": feature_matrix[row_index, indices["alogp"]],
            "tpsa": feature_matrix[row_index, indices["tpsa"]],
            "hba": feature_matrix[row_index, indices["hba"]],
            "hbd": feature_matrix[row_index, indices["hbd"]],
            "rotatable_bonds": feature_matrix[row_index, indices["rotatable_bonds"]],
            "ring_count": feature_matrix[row_index, indices["ring_count"]],
            "lipinski_violation_count": int(scores["lipinski_violation_count"][row_index]),
            "cns_rule_pass_count": int(scores["cns_rule_pass_count"][row_index]),
            "cns_rule_total_count": int(scores["cns_rule_total_count"][row_index]),
            "cns_property_score": float(scores["cns_property_score"][row_index]),
            "lipophilicity_window_score": float(scores["lipophilicity_window_score"][row_index]),
            "polar_surface_penalty": float(scores["polar_surface_penalty"][row_index]),
            "deepchem_dataset_class": str(baseline_config["deepchem_dataset_class"]),
            "deepchem_version": str(deepchem_module.__version__),
            "provenance_level": "deepchem_numpy_dataset_descriptor_property_baseline",
            "source_holdout_role": str(baseline_config["source_holdout_role"]),
            "train_flag": str(baseline_config["train_flag"]),
            "eval_flag": str(baseline_config["eval_flag"]),
            "public_url": str(baseline_config["public_url"]),
            "note": "MoleculeNet/DeepChem adapter materialized as a DeepChem NumpyDataset over NeuronSifter RDKit descriptor features.",
        }
        rows.append(row)
    output_path = external_adapter_processed_artifact_path(config, "moleculenet_deepchem")
    write_tsv(output_path, config["schemas"]["molecular_property_baseline_fields"], rows)
    return output_path


def clear_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    ensure_directory(path)


def write_rows(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_part_path(block_state: dict[str, object], part_path: Path) -> None:
    relative_path = artifact_path_text(part_path)
    observed_paths = [value for value in str(block_state["dose_response_part_path"]).split("|") if value]
    if relative_path not in observed_paths:
        observed_paths.append(relative_path)
    block_state["dose_response_part_path"] = "|".join(observed_paths)


def write_drugcomb_part(
    matrix_dir: Path,
    part_index: int,
    fields: list[str],
    rows: list[dict[str, object]],
    block_index: dict[str, dict[str, object]],
) -> Path:
    part_path = dose_part_path(matrix_dir, part_index)
    for item in rows:
        append_part_path(block_index[str(item["block_id"])], part_path)
    write_rows(part_path, fields, rows)
    return part_path


def dose_part_path(matrix_dir: Path, part_index: int) -> Path:
    return matrix_dir / f"part-{part_index:05d}.tsv"


def drugcomb_dose_response_row(
    row: dict[str, str],
    row_index: int,
    drugcomb_config: dict[str, object],
) -> dict[str, object]:
    return {
        "dose_response_uid": f"drugcomb_dose_response_{row_index:012d}",
        "source_key": "drugcomb",
        "source_release": str(drugcomb_config["source_release"]),
        "block_id": str(row["block_id"]),
        "drug_row": str(row["drug_row"]),
        "drug_col": str(row["drug_col"]),
        "drug_row_cid": str(row["drug_row_cid"]),
        "drug_col_cid": str(row["drug_col_cid"]),
        "conc_r": finite_float(row["conc_r"], "conc_r"),
        "conc_c": finite_float(row["conc_c"], "conc_c"),
        "inhibition": finite_float(row["inhibition"], "inhibition"),
        "conc_r_unit": str(row["conc_r_unit"]),
        "conc_c_unit": str(row["conc_c_unit"]),
        "cell_line_name": str(row["cell_line_name"]),
        "cellosaurus_accession": str(row["cellosaurus_accession"]),
        "study_name": str(row["study_name"]),
        "provenance_level": "drugcomb_public_full_dose_response_matrix",
        "source_holdout_role": str(drugcomb_config["source_holdout_role"]),
        "train_flag": str(drugcomb_config["train_flag"]),
        "eval_flag": str(drugcomb_config["eval_flag"]),
    }


def materialize_drugcomb_dose_matrix(
    config: dict,
    max_rows: int | None = None,
) -> tuple[Path, dict[str, dict[str, object]], dict[str, object]]:
    drugcomb_config = external_adapter_materialization_config(config, "drugcomb")
    source_path = resolve_repo_path(str(drugcomb_config["dose_response_csv_path"]))
    if not source_path.exists():
        raise FileNotFoundError(f"DrugComb dose-response source CSV is missing: {source_path}")
    matrix_dir = external_adapter_output_path(config, "drugcomb_dose_response_matrix_dir")
    clear_output_dir(matrix_dir)
    fields = config["schemas"]["drugcomb_dose_response_matrix_fields"]
    chunk_size = int(drugcomb_config["chunk_row_count"])
    if chunk_size <= 0:
        raise ValueError("DrugComb chunk_row_count must be positive.")
    block_index: dict[str, dict[str, object]] = {}
    part_rows: list[dict[str, object]] = []
    part_index = 0
    row_count = 0
    part_paths: list[str] = []
    current_block_id = ""
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            block_id = str(raw_row["block_id"])
            if current_block_id and block_id != current_block_id and len(part_rows) >= chunk_size:
                part_path = write_drugcomb_part(matrix_dir, part_index, fields, part_rows, block_index)
                part_paths.append(artifact_path_text(part_path))
                part_rows = []
                part_index += 1
            current_block_id = block_id
            row_count += 1
            block_state = block_index.setdefault(
                block_id,
                {
                    "drug_row_cid": str(raw_row["drug_row_cid"]),
                    "drug_col_cid": str(raw_row["drug_col_cid"]),
                    "dose_response_row_count": 0,
                    "dose_response_part_path": "",
                },
            )
            block_state["dose_response_row_count"] = int(block_state["dose_response_row_count"]) + 1
            output_row = drugcomb_dose_response_row(raw_row, row_count, drugcomb_config)
            part_rows.append(output_row)
            if max_rows is not None and row_count >= max_rows:
                break
    if part_rows:
        part_path = write_drugcomb_part(matrix_dir, part_index, fields, part_rows, block_index)
        part_paths.append(artifact_path_text(part_path))
    if row_count == 0:
        raise ValueError("DrugComb dose-response source CSV produced no rows.")
    manifest = {
        "source_key": "drugcomb",
        "source_release": str(drugcomb_config["source_release"]),
        "dose_response_source_path": artifact_path_text(source_path),
        "dose_response_source_sha256": sha256_of_file(source_path),
        "matrix_dir": artifact_path_text(matrix_dir),
        "dose_response_row_count": row_count,
        "block_count": len(block_index),
        "part_count": len(part_paths),
        "part_paths": part_paths,
        "max_rows": max_rows if max_rows is not None else "",
        "full_matrix_flag": "yes" if max_rows is None else "no",
    }
    write_json(external_adapter_output_path(config, "drugcomb_dose_response_manifest"), manifest)
    return matrix_dir, block_index, manifest


def drugcomb_combination_prior_row(
    row: dict[str, str],
    block_state: dict[str, object],
    drugcomb_config: dict[str, object],
) -> dict[str, object]:
    primary_metric = str(drugcomb_config["primary_synergy_metric"])
    return {
        "combination_uid": stable_uid("drugcomb_combination", row["block_id"], row["drug_row"], row["drug_col"], row["cell_line_name"]),
        "source_key": "drugcomb",
        "source_release": str(drugcomb_config["source_release"]),
        "block_id": str(row["block_id"]),
        "drug_row": str(row["drug_row"]),
        "drug_col": str(row["drug_col"]),
        "drug_row_cid": str(block_state.get("drug_row_cid", "")),
        "drug_col_cid": str(block_state.get("drug_col_cid", "")),
        "cell_line_name": str(row["cell_line_name"]),
        "study_name": str(row["study_name"]),
        "tissue_name": str(row["tissue_name"]),
        "conc_row_unit": str(row["conc_row_unit"]),
        "conc_col_unit": str(row["conc_col_unit"]),
        "css_ri": finite_float(row["css_ri"], "css_ri"),
        "s_sum": finite_float(row["S_sum"], "S_sum"),
        "s_mean": finite_float(row["S_mean"], "S_mean"),
        "s_max": finite_float(row["S_max"], "S_max"),
        "synergy_zip": finite_float(row["synergy_zip"], "synergy_zip"),
        "synergy_loewe": finite_float(row["synergy_loewe"], "synergy_loewe"),
        "synergy_hsa": finite_float(row["synergy_hsa"], "synergy_hsa"),
        "synergy_bliss": finite_float(row["synergy_bliss"], "synergy_bliss"),
        "drug_row_clinical_phase": str(row["drug_row_clinical_phase"]),
        "drug_col_clinical_phase": str(row["drug_col_clinical_phase"]),
        "drug_row_target_name": str(row["drug_row_target_name"]),
        "drug_col_target_name": str(row["drug_col_target_name"]),
        "primary_synergy_metric": primary_metric,
        "primary_synergy_value": finite_float(row[primary_metric], primary_metric),
        "dose_response_part_path": str(block_state["dose_response_part_path"]),
        "dose_response_row_count": int(block_state["dose_response_row_count"]),
        "provenance_level": "drugcomb_public_synergy_summary_linked_to_full_matrix",
        "source_holdout_role": str(drugcomb_config["source_holdout_role"]),
        "train_flag": str(drugcomb_config["train_flag"]),
        "eval_flag": str(drugcomb_config["eval_flag"]),
        "public_url": str(drugcomb_config["public_url"]),
        "note": "DrugComb synergy summary linked to materialized full dose-response matrix.",
    }


def build_drugcomb_processed_catalogs(config: dict, max_dose_response_rows: int | None = None) -> tuple[Path, Path]:
    drugcomb_config = external_adapter_materialization_config(config, "drugcomb")
    matrix_dir, block_index, manifest = materialize_drugcomb_dose_matrix(config, max_rows=max_dose_response_rows)
    summary_path = resolve_repo_path(str(drugcomb_config["synergy_summary_csv_path"]))
    if not summary_path.exists():
        raise FileNotFoundError(f"DrugComb synergy summary source CSV is missing: {summary_path}")
    rows: list[dict[str, object]] = []
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            block_id = str(raw_row["block_id"])
            if block_id not in block_index:
                continue
            rows.append(drugcomb_combination_prior_row(raw_row, block_index[block_id], drugcomb_config))
    if not rows:
        raise ValueError("DrugComb synergy summary did not overlap the materialized full dose-response matrix.")
    output_path = external_adapter_output_path(config, "drugcomb_combination_prior_catalog")
    write_tsv(output_path, config["schemas"]["drugcomb_combination_prior_fields"], rows)
    manifest["combination_prior_row_count"] = len(rows)
    manifest["combination_prior_output_path"] = artifact_path_text(output_path)
    manifest["synergy_summary_source_path"] = artifact_path_text(summary_path)
    manifest["synergy_summary_source_sha256"] = sha256_of_file(summary_path)
    write_json(external_adapter_output_path(config, "drugcomb_dose_response_manifest"), manifest)
    return output_path, matrix_dir
