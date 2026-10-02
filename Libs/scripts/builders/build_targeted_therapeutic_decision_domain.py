"""Materialize matched-control therapeutic outcomes and qualify the decision domain."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import openpyxl


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, write_json, write_tsv  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
TARGETED_CONFIG_PATH = REPO_ROOT / "Libs/configs/data/targeted_therapeutic_evidence.yaml"
BUILDER_ID = "Libs/scripts/builders/build_targeted_therapeutic_decision_domain.py"
BOUNDARY = {
    "evidence_role": "development_only",
    "publication_claim_eligible": False,
    "protected_holdout_access": 0,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"Configured path escapes the repository: {value}")
    return path


def _artifact(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _uid(prefix: str, values: Sequence[str]) -> str:
    digest = hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def _policy_digest(policy: Mapping[str, object]) -> str:
    payload = json.dumps(policy, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _input_paths(policy: Mapping[str, object]) -> dict[str, Path]:
    inputs = policy["inputs"]
    if not isinstance(inputs, Mapping):
        raise TypeError("Targeted therapeutic inputs must be a mapping")
    paths: dict[str, Path] = {}
    for source_key, source_inputs in inputs.items():
        if isinstance(source_inputs, Mapping):
            for asset_role, raw_path in source_inputs.items():
                paths[f"{source_key}:{asset_role}"] = _repo_path(str(raw_path))
        else:
            paths[str(source_key)] = _repo_path(str(source_inputs))
    return paths


def freeze_objective(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    if repo_root.resolve() != REPO_ROOT:
        raise ValueError("Therapeutic objective freeze is identity-bound to the repository root")
    config = load_config(CONFIG_PATH)
    policy = config["real_intervention_evidence"]["targeted_therapeutic_expansion"]
    if not isinstance(policy, Mapping):
        raise TypeError("Targeted therapeutic policy must be a mapping")
    source_policy = policy["source_policy"]
    objective = policy["therapeutic_decision_objective"]
    if not isinstance(source_policy, Mapping) or not isinstance(objective, Mapping):
        raise TypeError("Source policy and therapeutic objective must be mappings")
    if objective.get("freeze_status") != "frozen_before_outcome_materialization":
        raise ValueError("Therapeutic objective is not frozen before outcome materialization")
    if objective.get("decision_unit") != "candidate_treatment_within_source_defined_biological_context":
        raise ValueError("Therapeutic objective must be source-context scoped")
    if objective.get("reference_condition") != "source_declared_matched_control":
        raise ValueError("Therapeutic objective must use each source's declared matched control")
    if source_policy.get("model_predictions_inspected") is not False:
        raise ValueError("Source selection inspected model predictions")
    weights = [float(objective["primary_goal"]["weight"])] + [
        float(item["weight"]) for item in objective["secondary_goals"]
    ]
    if not np.isclose(sum(weights), 1.0):
        raise ValueError(f"Therapeutic objective weights must sum to one: {weights}")
    input_paths = _input_paths(policy)
    payload = {
        "schema_version": "neuronsifter-therapeutic-decision-objective-v1",
        "frozen_at": _now(),
        "objective": objective,
        "source_policy": source_policy,
        "policy_digest": _policy_digest(policy),
        "outcome_values_inspected_for_objective_selection": False,
        "model_predictions_inspected": False,
        "author_claims_used_as_ground_truth": False,
        "objective_scope_contract": {
            "decision_unit": "candidate_treatment_within_source_defined_biological_context",
            "reference_condition": "source_declared_matched_control",
            "control_resolution": "source_contract.control_condition",
        },
        "input_bindings": {name: _artifact(path) for name, path in input_paths.items()},
        "config_binding": _artifact(TARGETED_CONFIG_PATH),
        **BOUNDARY,
    }
    output_path = _repo_path(str(policy["outputs"]["objective"]))
    if output_path.exists():
        existing = _json(output_path)
        if existing.get("policy_digest") != payload["policy_digest"]:
            raise FileExistsError("Frozen therapeutic objective exists with a different policy digest")
        if existing.get("objective") != objective or existing.get("source_policy") != source_policy:
            raise ValueError("Frozen therapeutic objective payload drifted despite matching policy digest")
        if existing.get("config_binding") != payload["config_binding"]:
            migrated = dict(existing)
            migrated["config_binding_migration"] = {
                "migrated_at": _now(),
                "reason": "targeted_policy_moved_to_semantic_standalone_config",
                "previous_config_binding": existing.get("config_binding"),
                "policy_digest_unchanged": True,
                "objective_payload_unchanged": True,
            }
            migrated["config_binding"] = payload["config_binding"]
            write_json(output_path, migrated)
            return migrated
        return existing
    write_json(output_path, payload)
    return payload


def _parse_series_matrix(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    series: dict[str, str] = {}
    sample_rows: dict[str, list[list[str]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", errors="strict", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            key = row[0]
            values = [str(value).strip() for value in row[1:]]
            if key.startswith("!Series_"):
                series.setdefault(key.removeprefix("!Series_"), values[0] if values else "")
            elif key.startswith("!Sample_"):
                sample_rows[key.removeprefix("!Sample_")].append(values)
    accessions = sample_rows.get("geo_accession", [[]])[0]
    titles = sample_rows.get("title", [[]])[0]
    if not accessions or len(accessions) != len(titles):
        raise ValueError(f"GEO sample identity rows are incomplete: {path}")
    samples = [
        {"sample_accession": accession, "sample_title": title}
        for accession, title in zip(accessions, titles, strict=True)
    ]
    for characteristic_values in sample_rows.get("characteristics_ch1", []):
        if len(characteristic_values) != len(samples):
            raise ValueError(f"GEO characteristic row width drifted: {path}")
        for sample, raw_value in zip(samples, characteristic_values, strict=True):
            name, separator, value = raw_value.partition(":")
            if not separator:
                raise ValueError(f"GEO characteristic is not key/value: {raw_value!r}")
            sample[name.strip().lower()] = value.strip()
    description_rows = sample_rows.get("description", [])
    library_row = next(
        (values for values in description_rows if values and all(value.startswith("Library name:") for value in values)),
        None,
    )
    if library_row is None or len(library_row) != len(samples):
        raise ValueError(f"GEO matrix lacks a complete library-name row: {path}")
    for sample, raw_library in zip(samples, library_row, strict=True):
        sample["library_name"] = raw_library.split(":", 1)[1].strip()
    return series, samples


def _objective_components(objective: Mapping[str, object]) -> list[dict[str, object]]:
    primary = objective["primary_goal"]
    secondary = objective["secondary_goals"]
    if not isinstance(primary, Mapping) or not isinstance(secondary, list):
        raise TypeError("Therapeutic objective components are malformed")
    components = [dict(primary), *[dict(item) for item in secondary]]
    symbols = [str(symbol).upper() for item in components for symbol in item["gene_symbols"]]
    if len(symbols) != len(set(symbols)):
        raise ValueError("A gene appears in multiple therapeutic objective components")
    return components


def _selected_samples(
    accession: str,
    samples: Sequence[Mapping[str, str]],
    source_contract: Mapping[str, object],
) -> list[dict[str, str]]:
    allowed_treatments = {
        str(source_contract["control_condition"]),
        *[str(value) for value in source_contract["candidate_treatments"]],
    }
    genotype = str(source_contract["metadata_genotype_value"])
    selected = [
        dict(sample)
        for sample in samples
        if sample.get("genotype") == genotype and sample.get("treatment") in allowed_treatments
    ]
    observed_counts: dict[str, int] = defaultdict(int)
    for sample in selected:
        observed_counts[str(sample["treatment"])] += 1
    expected = {str(key): int(value) for key, value in source_contract["expected_biological_replicates"].items()}
    if dict(sorted(observed_counts.items())) != dict(sorted(expected.items())):
        raise ValueError(f"{accession} selected replicate counts drifted: {dict(observed_counts)} != {expected}")
    if len({sample["sample_accession"] for sample in selected}) != len(selected):
        raise ValueError(f"{accession} sample accessions are not unique")
    return selected


def _resolve_count_header(accession: str, header: Sequence[object], library_name: str) -> int:
    normalized = [str(value).strip() for value in header]
    if accession == "GSE338689":
        matches = [index for index, value in enumerate(normalized) if value.startswith(f"{library_name}_")]
    else:
        matches = [index for index, value in enumerate(normalized) if value == library_name]
    if len(matches) != 1:
        raise ValueError(f"{accession} library {library_name!r} maps to {len(matches)} count columns")
    return matches[0]


def _human_gene_mapping(path: Path, objective_symbols: set[str]) -> dict[str, str]:
    payload = _json(path)
    mapping = {
        str(record["id"]).split(".", 1)[0]: str(symbol).upper()
        for symbol, record in payload.items()
        if isinstance(record, Mapping) and str(symbol).upper() in objective_symbols
    }
    if set(mapping.values()) != objective_symbols:
        missing = sorted(objective_symbols - set(mapping.values()))
        raise ValueError(f"Ensembl objective mapping is incomplete: {missing}")
    return mapping


def _read_objective_counts(
    accession: str,
    path: Path,
    selected: Sequence[Mapping[str, str]],
    objective_symbols: set[str],
    ensembl_mapping: Mapping[str, str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    library_names = [str(sample["library_name"]) for sample in selected]
    totals = np.zeros(len(selected), dtype=np.float64)
    retained: dict[str, np.ndarray] = {}
    if path.suffix == ".xlsx":
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        worksheet = workbook.active
        rows = worksheet.iter_rows(values_only=True)
        header = next(rows)
        positions = [_resolve_count_header(accession, header, library) for library in library_names]
        for row in rows:
            gene_symbol = str(row[0]).strip().upper()
            values = np.asarray([float(row[position]) for position in positions], dtype=np.float64)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{accession} contains invalid counts for {row[0]!r}")
            totals += values
            if gene_symbol in objective_symbols:
                if gene_symbol in retained:
                    raise ValueError(f"{accession} contains duplicate objective gene: {gene_symbol}")
                retained[gene_symbol] = values
        workbook.close()
    else:
        with gzip.open(path, "rt", encoding="utf-8", errors="strict", newline="") as handle:
            rows = csv.reader(handle)
            header = next(rows)
            positions = [_resolve_count_header(accession, header, library) for library in library_names]
            for row in rows:
                gene_id = str(row[0]).split(".", 1)[0]
                values = np.asarray([float(row[position]) for position in positions], dtype=np.float64)
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError(f"{accession} contains invalid counts for {row[0]!r}")
                totals += values
                gene_symbol = ensembl_mapping.get(gene_id)
                if gene_symbol is not None:
                    if gene_symbol in retained:
                        raise ValueError(f"{accession} contains duplicate objective gene: {gene_symbol}")
                    retained[gene_symbol] = values
    if (totals <= 0).any():
        raise ValueError(f"{accession} contains an empty count library")
    if set(retained) != objective_symbols:
        missing = sorted(objective_symbols - set(retained))
        raise ValueError(f"{accession} lacks objective genes: {missing}")
    return totals, retained


def _score_from_means(
    action_mean: np.ndarray,
    control_mean: np.ndarray,
    gene_order: Sequence[str],
    components: Sequence[Mapping[str, object]],
) -> tuple[float, dict[str, float]]:
    position = {symbol: index for index, symbol in enumerate(gene_order)}
    effects: dict[str, float] = {}
    total = 0.0
    for component in components:
        indices = [position[str(symbol).upper()] for symbol in component["gene_symbols"]]
        raw_effect = float(np.median(action_mean[indices] - control_mean[indices]))
        signed_effect = raw_effect if component["direction"] == "maximize" else -raw_effect
        component_uid = str(component["component_uid"])
        effects[component_uid] = signed_effect
        total += float(component["weight"]) * signed_effect
    return total, effects


def bootstrap_candidate_scores(
    log2_cpm: np.ndarray,
    treatments: Sequence[str],
    control_condition: str,
    candidates: Sequence[str],
    gene_order: Sequence[str],
    components: Sequence[Mapping[str, object]],
    *,
    resamples: int,
    seed: int,
) -> tuple[dict[str, dict[str, object]], np.ndarray]:
    treatment_array = np.asarray(treatments, dtype=object)
    control_indices = np.flatnonzero(treatment_array == control_condition)
    action_indices = {candidate: np.flatnonzero(treatment_array == candidate) for candidate in candidates}
    if not len(control_indices) or any(not len(indices) for indices in action_indices.values()):
        raise ValueError("Matched control or candidate samples are absent")
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros((resamples, len(candidates)), dtype=np.float64)
    point_scores: list[float] = []
    point_components: list[dict[str, float]] = []
    control_mean = log2_cpm[control_indices].mean(axis=0)
    for candidate in candidates:
        score, effects = _score_from_means(
            log2_cpm[action_indices[candidate]].mean(axis=0), control_mean, gene_order, components
        )
        point_scores.append(score)
        point_components.append(effects)
    for draw_index in range(resamples):
        sampled_control = rng.choice(control_indices, size=len(control_indices), replace=True)
        control_draw_mean = log2_cpm[sampled_control].mean(axis=0)
        for candidate_index, candidate in enumerate(candidates):
            indices = action_indices[candidate]
            sampled_action = rng.choice(indices, size=len(indices), replace=True)
            bootstrap[draw_index, candidate_index], _ = _score_from_means(
                log2_cpm[sampled_action].mean(axis=0),
                control_draw_mean,
                gene_order,
                components,
            )
    top_indices = np.argmax(bootstrap, axis=1)
    results: dict[str, dict[str, object]] = {}
    for index, candidate in enumerate(candidates):
        results[candidate] = {
            "score": point_scores[index],
            "component_effects": point_components[index],
            "score_ci95": [float(value) for value in np.quantile(bootstrap[:, index], [0.025, 0.975])],
            "benefit_probability": float(np.mean(bootstrap[:, index] > 0.0)),
            "top_probability": float(np.mean(top_indices == index)) if len(candidates) >= 2 else 1.0,
        }
    return results, bootstrap


def classify_candidate_set(
    action_results: Mapping[str, Mapping[str, object]],
    bootstrap: np.ndarray,
    qualification: Mapping[str, object],
) -> tuple[str, str, float, list[float]]:
    candidates = list(action_results)
    scores = np.asarray([float(action_results[candidate]["score"]) for candidate in candidates])
    top_index = int(np.argmax(scores))
    top_candidate = candidates[top_index]
    if len(candidates) < 2:
        return "scientifically_informative_but_not_rankable", top_candidate, 0.0, [0.0, 0.0]
    other_indices = [index for index in range(len(candidates)) if index != top_index]
    point_margin = float(scores[top_index] - max(scores[other_indices]))
    margin_draws = bootstrap[:, top_index] - np.max(bootstrap[:, other_indices], axis=1)
    margin_ci = [float(value) for value in np.quantile(margin_draws, [0.025, 0.975])]
    top = action_results[top_candidate]
    if (
        float(top["top_probability"]) >= float(qualification["decision_sensitive_min_top_probability"])
        and float(top["benefit_probability"]) >= float(qualification["decision_sensitive_min_benefit_probability"])
        and margin_ci[0] > 0.0
    ):
        classification = "decision_sensitive"
    elif (
        float(top["top_probability"])
        >= float(qualification["weakly_decision_sensitive_min_top_probability"])
        and float(top["benefit_probability"])
        >= float(qualification["weakly_decision_sensitive_min_benefit_probability"])
        and point_margin >= float(qualification["rank_margin_threshold_log2_cpm"])
    ):
        classification = "weakly_decision_sensitive"
    elif float(np.max(np.abs(scores))) <= float(qualification["near_null_absolute_score_max_log2_cpm"]):
        classification = "null"
    else:
        classification = "ambiguous"
    return classification, top_candidate, point_margin, margin_ci


def _fieldnames(rows: Sequence[Mapping[str, object]]) -> list[str]:
    if not rows:
        raise ValueError("Cannot write an empty TSV")
    return list(rows[0])


def build(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    if repo_root.resolve() != REPO_ROOT:
        raise ValueError("Therapeutic decision-domain build is identity-bound to the repository root")
    config = load_config(CONFIG_PATH)
    policy = config["real_intervention_evidence"]["targeted_therapeutic_expansion"]
    if not isinstance(policy, Mapping):
        raise TypeError("Targeted therapeutic policy must be a mapping")
    outputs = {name: _repo_path(str(value)) for name, value in policy["outputs"].items()}
    objective_artifact = _json(outputs["objective"])
    if objective_artifact.get("policy_digest") != _policy_digest(policy):
        raise ValueError("Therapeutic objective freeze no longer matches the configured policy")
    if objective_artifact.get("outcome_values_inspected_for_objective_selection") is not False:
        raise ValueError("Therapeutic objective selection inspected outcome values")
    for name, path in _input_paths(policy).items():
        binding = objective_artifact["input_bindings"].get(name)
        if not isinstance(binding, Mapping) or binding != _artifact(path):
            raise ValueError(f"Therapeutic objective input binding drifted: {name}")

    source_policy = policy["source_policy"]
    objective = policy["therapeutic_decision_objective"]
    source_contracts = source_policy["sources"]
    components = _objective_components(objective)
    objective_symbols = {
        str(symbol).upper() for component in components for symbol in component["gene_symbols"]
    }
    input_paths = _input_paths(policy)
    ensembl_mapping = _human_gene_mapping(
        input_paths["ensembl_human_objective_gene_mapping"], objective_symbols
    )

    source_rows: list[dict[str, object]] = []
    observation_rows: list[dict[str, object]] = []
    gene_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    for accession in source_policy["selected_accessions"]:
        accession = str(accession)
        source_contract = source_contracts[accession]
        series, all_samples = _parse_series_matrix(input_paths[f"{accession}:series_matrix"])
        selected = _selected_samples(accession, all_samples, source_contract)
        count_path = input_paths[f"{accession}:counts"]
        totals, retained = _read_objective_counts(
            accession,
            count_path,
            selected,
            objective_symbols,
            ensembl_mapping,
        )
        gene_order = sorted(objective_symbols)
        counts = np.stack([retained[gene] for gene in gene_order], axis=1)
        log2_cpm = np.log2(counts / totals[:, None] * 1_000_000.0 + 1.0)
        treatments = [str(sample["treatment"]) for sample in selected]
        control = str(source_contract["control_condition"])
        candidates = [str(value) for value in source_contract["candidate_treatments"]]
        action_results, bootstrap = bootstrap_candidate_scores(
            log2_cpm,
            treatments,
            control,
            candidates,
            gene_order,
            components,
            resamples=int(objective["qualification"]["bootstrap_resamples"]),
            seed=int(objective["qualification"]["bootstrap_seed"]),
        )
        classification, top_candidate, decision_margin, decision_margin_ci = classify_candidate_set(
            action_results, bootstrap, objective["qualification"]
        )
        context = source_contract["biological_context"]
        publication_uid = (
            f"pubmed:{series['pubmed_id']}" if series.get("pubmed_id") else f"geo:{accession}"
        )
        control_group_uid = f"geo:{accession}:matched_control:{control}"
        candidate_set_uid = _uid(
            "therapeutic_candidate_set",
            [accession, str(context["species"]), str(context["genotype"]), str(context["tissue"])],
        )
        source_asset_bindings = {
            name.split(":", 1)[1]: _artifact(path)
            for name, path in input_paths.items()
            if name.startswith(f"{accession}:")
        }
        source_rows.append(
            {
                "source_key": accession.lower(),
                "source_repository": "GEO",
                "source_accession": accession,
                "publication_uid": publication_uid,
                "title": series.get("title", ""),
                "public_url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
                "species": context["species"],
                "genotype": context["genotype"],
                "tissue": context["tissue"],
                "assay": context["assay"],
                "split_role": source_contract["split_role"],
                "control_condition": control,
                "candidate_treatments": json.dumps(candidates, separators=(",", ":")),
                "selected_sample_count": len(selected),
                "control_sample_count": treatments.count(control),
                "candidate_action_count": len(candidates),
                "raw_asset_bindings": json.dumps(source_asset_bindings, sort_keys=True, separators=(",", ":")),
                "evidence_class": "E2_REAL_ENDPOINT",
                "evidence_grade": "B",
                **BOUNDARY,
            }
        )
        control_indices = np.flatnonzero(np.asarray(treatments, dtype=object) == control)
        control_gene_mean = log2_cpm[control_indices].mean(axis=0)
        component_by_gene: dict[str, str] = {}
        for component in components:
            for symbol in component["gene_symbols"]:
                component_by_gene[str(symbol).upper()] = str(component["component_uid"])
        for sample_index, sample in enumerate(selected):
            treatment = str(sample["treatment"])
            role = "matched_control" if treatment == control else "therapeutic_endpoint"
            action_uid = (
                f"no_action:{accession}:{control}"
                if treatment == control
                else f"geo_action:{accession}:{treatment.lower()}"
            )
            biological_uid = f"geo:{accession}:biological_unit:{sample['sample_accession']}"
            sample_score, sample_components = _score_from_means(
                log2_cpm[sample_index], control_gene_mean, gene_order, components
            )
            observation_rows.append(
                {
                    "episode_uid": _uid("therapeutic_episode", [accession, sample["sample_accession"]]),
                    "source_key": accession.lower(),
                    "publication_uid": publication_uid,
                    "accession_uid": accession,
                    "biological_unit_uid": biological_uid,
                    "sample_uid": f"geo:{sample['sample_accession']}",
                    "source_sample_accession": sample["sample_accession"],
                    "source_sample_title": sample["sample_title"],
                    "source_library_name": sample["library_name"],
                    "species": context["species"],
                    "genotype": context["genotype"],
                    "tissue": context["tissue"],
                    "sex": sample.get("sex", "explicit_unavailable"),
                    "batch": sample.get("batch", "explicit_unavailable"),
                    "treatment": treatment,
                    "observation_role": role,
                    "control_group_uid": control_group_uid,
                    "world_action_uid": action_uid,
                    "action_family_uid": action_uid,
                    "time_grid": f"[{context['treatment_duration']}]",
                    "measurement_namespace": "gene_symbol",
                    "value_unit": "log2_cpm_plus_one",
                    "objective_uid": objective["objective_uid"],
                    "component_scores": json.dumps(sample_components, sort_keys=True, separators=(",", ":")),
                    "therapeutic_score": sample_score,
                    "split_role": source_contract["split_role"],
                    "raw_value_materialized": True,
                    "evidence_class": "E2_REAL_ENDPOINT",
                    "evidence_grade": "B",
                    **BOUNDARY,
                }
            )
            for gene_index, gene_symbol in enumerate(gene_order):
                gene_rows.append(
                    {
                        "source_accession": accession,
                        "biological_unit_uid": biological_uid,
                        "sample_uid": f"geo:{sample['sample_accession']}",
                        "treatment": treatment,
                        "observation_role": role,
                        "control_group_uid": control_group_uid,
                        "objective_uid": objective["objective_uid"],
                        "component_uid": component_by_gene[gene_symbol],
                        "gene_symbol": gene_symbol,
                        "raw_count": counts[sample_index, gene_index],
                        "library_size": totals[sample_index],
                        "cpm": counts[sample_index, gene_index] / totals[sample_index] * 1_000_000.0,
                        "log2_cpm_plus_one": log2_cpm[sample_index, gene_index],
                        "matched_control_log2_cpm_mean": control_gene_mean[gene_index],
                        **BOUNDARY,
                    }
                )
        action_payload: dict[str, object] = {}
        for candidate, result in action_results.items():
            action_uid = f"geo_action:{accession}:{candidate.lower()}"
            action_payload[action_uid] = result
            quality_rows.append(
                {
                    "source_accession": accession,
                    "candidate_set_uid": candidate_set_uid,
                    "world_action_uid": action_uid,
                    "treatment": candidate,
                    "control_group_uid": control_group_uid,
                    "objective_uid": objective["objective_uid"],
                    "component_effects": json.dumps(result["component_effects"], sort_keys=True, separators=(",", ":")),
                    "therapeutic_score": result["score"],
                    "therapeutic_score_ci95": json.dumps(result["score_ci95"], separators=(",", ":")),
                    "benefit_probability": result["benefit_probability"],
                    "top_probability": result["top_probability"],
                    "matched_control_values_materialized": True,
                    "model_predictions_inspected": False,
                    **BOUNDARY,
                }
            )
        qualified = classification in {"decision_sensitive", "weakly_decision_sensitive"}
        candidate_rows.append(
            {
                "candidate_set_uid": candidate_set_uid,
                "source_accession": accession,
                "publication_uid": publication_uid,
                "biological_context": json.dumps(context, sort_keys=True, separators=(",", ":")),
                "control_group_uid": control_group_uid,
                "control_condition": control,
                "control_sample_count": treatments.count(control),
                "candidate_action_count": len(candidates),
                "candidate_action_uids": json.dumps(sorted(action_payload), separators=(",", ":")),
                "candidate_sample_counts": json.dumps(
                    {candidate: treatments.count(candidate) for candidate in candidates},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "action_results": json.dumps(action_payload, sort_keys=True, separators=(",", ":")),
                "top_action_uid": f"geo_action:{accession}:{top_candidate.lower()}",
                "decision_margin": decision_margin,
                "decision_margin_ci95": json.dumps(decision_margin_ci, separators=(",", ":")),
                "classification": classification,
                "qualified_decision_candidate_set": qualified,
                "matched_control_values_materialized": True,
                "therapeutic_objective_prespecified": True,
                "model_predictions_inspected": False,
                "split_role": source_contract["split_role"],
                **BOUNDARY,
            }
        )

    write_tsv(outputs["source_manifest"], _fieldnames(source_rows), source_rows)
    write_tsv(outputs["observation_manifest"], _fieldnames(observation_rows), observation_rows)
    write_tsv(outputs["gene_outcomes"], _fieldnames(gene_rows), gene_rows)
    write_tsv(outputs["candidate_set_manifest"], _fieldnames(candidate_rows), candidate_rows)
    write_tsv(outputs["outcome_quality"], _fieldnames(quality_rows), quality_rows)

    qualified_rows = [row for row in candidate_rows if bool(row["qualified_decision_candidate_set"])]
    role_rows = {
        role: [row for row in observation_rows if row["split_role"] == role]
        for role in ("train", "validation", "development_eval")
    }
    create_splits = bool(qualified_rows) and all(role_rows.values())
    planned_splits = {name: _repo_path(str(path)) for name, path in policy["planned_split_paths"].items()}
    for path in planned_splits.values():
        if path.exists():
            raise FileExistsError(f"Therapeutic benchmark split already exists: {path}")
    if create_splits:
        for role, rows in role_rows.items():
            write_tsv(planned_splits[role], _fieldnames(rows), rows)
    accessions_by_role = {
        role: sorted({str(row["accession_uid"]) for row in rows}) for role, rows in role_rows.items()
    }
    publications_by_role = {
        role: sorted({str(row["publication_uid"]) for row in rows}) for role, rows in role_rows.items()
    }
    units_by_role = {
        role: {str(row["biological_unit_uid"]) for row in rows} for role, rows in role_rows.items()
    }
    roles = list(role_rows)
    unit_overlap = sum(
        len(units_by_role[roles[left]] & units_by_role[roles[right]])
        for left in range(len(roles))
        for right in range(left + 1, len(roles))
    )
    split_audit = {
        "schema_version": "neuronsifter-therapeutic-decision-benchmark-split-audit-v1",
        "status": "passed" if create_splits else "not_created_no_qualified_candidate_sets",
        "split_files_created": create_splits,
        "development_splits_frozen": create_splits,
        "accessions_by_split": accessions_by_role,
        "publications_by_split": publications_by_role,
        "biological_unit_overlap_count": unit_overlap,
        "publication_overlap_count": sum(
            len(set(publications_by_role[roles[left]]) & set(publications_by_role[roles[right]]))
            for left in range(len(roles))
            for right in range(left + 1, len(roles))
        ),
        "accession_overlap_count": sum(
            len(set(accessions_by_role[roles[left]]) & set(accessions_by_role[roles[right]]))
            for left in range(len(roles))
            for right in range(left + 1, len(roles))
        ),
        "world_action_training_authorized": create_splits,
        "agent_in_twin_authorized": False,
        "model_predictions_inspected": False,
        **BOUNDARY,
    }
    write_json(outputs["split_audit"], split_audit)

    status = (
        "therapeutic_decision_domain_qualified"
        if create_splits
        else "blocked_insufficient_decision_sensitive_therapeutic_outcomes"
    )
    summary = {
        "schema_version": policy["schema_version"],
        "created_at": _now(),
        "status": status,
        "gate_uid": "real_intervention_decision_domain_qualified",
        "gate_resolved": True,
        "therapeutic_objective_prespecified": True,
        "matched_control_outcomes_materialized": True,
        "classification_is_based_on_real_outcomes": True,
        "decision_sensitive_domain_available": bool(qualified_rows),
        "model_training_domain_qualified": create_splits,
        "source_count": len(source_rows),
        "observation_count": len(observation_rows),
        "objective_gene_outcome_count": len(gene_rows),
        "candidate_set_count": len(candidate_rows),
        "multi_action_candidate_set_count": sum(int(row["candidate_action_count"] >= 2) for row in candidate_rows),
        "qualified_decision_candidate_set_count": len(qualified_rows),
        "candidate_classifications": {
            str(row["source_accession"]): str(row["classification"]) for row in candidate_rows
        },
        "development_splits_frozen": create_splits,
        "world_action_training_authorized": create_splits,
        "agent_in_twin_authorized": False,
        "coevolution_authorized": False,
        "model_predictions_inspected": False,
        "historical_decision_domain_artifacts_mutated": False,
        "historical_artifact_bindings": {
            "domain_summary": _artifact(REPO_ROOT / "Output/audits/real_intervention_decision_domain.json"),
            "context_group_qualification": _artifact(
                REPO_ROOT / "Output/audits/real_intervention_context_group_qualification.tsv"
            ),
            "split_audit": _artifact(
                REPO_ROOT / "Output/audits/real_intervention_decision_benchmark_split_audit.json"
            ),
        },
        "publication_comparison_open": False,
        "input_bindings": {
            **{name: _artifact(path) for name, path in _input_paths(policy).items()},
            "objective": _artifact(outputs["objective"]),
        },
        "implementation_bindings": {
            "config": _artifact(TARGETED_CONFIG_PATH),
            "schema": _artifact(_repo_path(str(policy["summary_schema_path"]))),
            "builder": _artifact(REPO_ROOT / BUILDER_ID),
        },
        "output_bindings": {
            name: _artifact(outputs[name])
            for name in (
                "source_manifest",
                "observation_manifest",
                "gene_outcomes",
                "candidate_set_manifest",
                "outcome_quality",
                "split_audit",
            )
        },
        **BOUNDARY,
    }
    if create_splits:
        summary["split_bindings"] = {name: _artifact(path) for name, path in planned_splits.items()}
    write_json(outputs["domain_summary"], summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-objective", action="store_true")
    args = parser.parse_args()
    result = freeze_objective() if args.freeze_objective else build()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
