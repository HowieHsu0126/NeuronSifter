"""Build a design-enriched, outcome-blind public CNS ranking intake."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable

from Bio import Entrez
from jsonschema import Draft202012Validator
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_retrospective_public_cns_ranking_intake import (  # noqa: E402
    _article_ids,
    _binding,
    _canonical_digest,
    _sha256,
)
from Libs.utils.data.raw_data_registry import load_config, write_json, write_tsv  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
PREREGISTRATION_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/retrospective_public_design_enriched_ranking_intake_preregistration.schema.json"
)
INTAKE_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/retrospective_public_design_enriched_ranking_intake.schema.json"
)
FROZEN_QUEUE_AUDIT_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_ranking_source_queue.json"
)
REFRESH_QUEUE_AUDIT_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_ranking_source_refresh_queue.json"
)
PRIOR_INTAKE_PREREGISTRATION_PATH = (
    REPO_ROOT / "Input/manifests/retrospective_public/cns_ranking_intake_preregistration.json"
)
PRIOR_INTAKE_AUDIT_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_cns_ranking_intake.json"
)
PRIOR_INTAKE_RECOVERY_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_cns_ranking_source_recovery.json"
)
PRIOR_METADATA_SNAPSHOT_PATH = (
    REPO_ROOT / "Input/raw/public/retrospective/cns_ranking_intake/pubmed_metadata_snapshot.json"
)
MATERIALIZATION_AUDIT_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_evidence_materialization.json"
)
REPLAY_AUDIT_PATH = REPO_ROOT / "Output/audits/retrospective_replay_benchmark.json"
EMPTY_DIGEST_PAYLOAD_PATH = (
    REPO_ROOT
    / "Output/audits/retrospective_public_design_enriched_ranking_intake_empty_digest_payload.json"
)
EMPTY_DIGEST_INVALIDATION_PATH = (
    REPO_ROOT
    / "Output/audits/retrospective_public_design_enriched_ranking_intake_empty_digest_invalidation.json"
)

EXCLUSION_TABLES = (
    (REPO_ROOT / "Input/manifests/retrospective_public/evidence_registry.tsv", "publication_uid"),
    (
        REPO_ROOT / "Output/audits/retrospective_public_evidence_rejection_catalog.tsv",
        "publication_uid",
    ),
    (
        REPO_ROOT / "Input/manifests/retrospective_public/ranking_informative_source_queue.tsv",
        "publication_uid",
    ),
    (
        REPO_ROOT / "Input/manifests/retrospective_public/ranking_source_refresh_queue.tsv",
        "publication_uid",
    ),
    (
        REPO_ROOT / "Input/manifests/retrospective_public/cns_ranking_intake_queue.tsv",
        "publication_uid",
    ),
)

QUEUE_FIELDS = (
    "intake_uid",
    "intake_sequence",
    "preregistration_digest",
    "parent_state_digest",
    "publication_uid",
    "source_identifier",
    "publication_date",
    "journal",
    "title",
    "source_accession",
    "doi",
    "paper_type",
    "source_quality_tier",
    "context_hint",
    "context_family",
    "objective_uid",
    "study_design_class",
    "named_intervention_entities",
    "named_intervention_entity_count",
    "therapeutic_arm_structure",
    "shared_control_expected",
    "same_endpoint_time_expected",
    "biological_replication_or_uncertainty_expected",
    "source_data_metadata_status",
    "independence_key",
    "ranking_informativeness_prior",
    "metadata_design_score",
    "experimental_numeric_evidence_status",
    "priority_order",
    "queue_status",
    "metadata_basis",
    "metadata_snapshot_date",
    "metadata_snapshot_path",
    "metadata_snapshot_sha256",
    "prior_metadata_universe_excluded",
    "preexposed_sources_excluded",
    "abstracts_retrieved_before_priority",
    "full_text_retrieved_before_priority",
    "figures_retrieved_before_priority",
    "supplementary_retrieved_before_priority",
    "outcome_values_used_for_priority",
    "observed_effect_used_for_priority",
    "p_values_used_for_priority",
    "author_winner_statement_used_for_priority",
    "protected_holdout_access",
)

DESIGN_PATTERNS: tuple[tuple[str, tuple[str, ...], str, str, int], ...] = (
    (
        "combination_vs_monotherapy",
        ("combination", "monotherapy"),
        "named combination and component intervention conditions",
        "high",
        0,
    ),
    (
        "direct_head_to_head",
        (" versus ", " vs ", " vs. ", "head-to-head", "comparison"),
        "at least two named intervention conditions in direct comparison",
        "high",
        1,
    ),
    (
        "dose_response",
        ("dose-response", "dose response", "concentration-response", "concentration response"),
        "multiple dose conditions across at least two named interventions",
        "medium_high",
        2,
    ),
    (
        "multi_arm_screen",
        ("screening", "screen", "high-throughput", "high throughput"),
        "multiple named intervention conditions in a common functional screen",
        "medium",
        3,
    ),
)

OBJECTIVE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("electrophysiology", ("electrophysiology", "excitability", "seizure", "firing", "electrical")),
    ("calcium_activity", ("calcium",)),
    ("neurite_morphology", ("neurite", "axon", "dendrite", "morphology")),
    ("synaptic_phenotype", ("synaptic", "synapse", "network activity")),
    ("cell_viability_survival", ("viability", "survival", "cell death", "cytotoxicity")),
    (
        "quantitative_pathology_phenotype",
        ("amyloid", "tau", "aggregate", "pathology", "inflammation", "microglia", "glial"),
    ),
)

CONTEXT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("alzheimer_disease", ("alzheimer", "amyloid", "tau")),
    ("parkinson_disease", ("parkinson", "dopaminergic", "midbrain")),
    ("amyotrophic_lateral_sclerosis", ("amyotrophic", " als ", "motor neuron")),
    ("frontotemporal_degeneration", ("frontotemporal", "tdp-43", "tdp43")),
    ("huntington_disease", ("huntington",)),
    ("rett_syndrome", ("rett",)),
    ("epilepsy_hyperexcitability", ("epilep", "seizure", "hyperexcit")),
    ("multiple_sclerosis", ("multiple sclerosis", "demyelin")),
    ("cns_injury", ("spinal cord injury", "brain injury", "stroke", "ischemia")),
    ("gaucher_neurological", ("gaucher",)),
    ("neurodegeneration_other", ("neurodegener",)),
)

CNS_MODEL_TERMS = (
    "neuron",
    "neuronal",
    "brain",
    "organoid",
    "microglia",
    "astrocyte",
    "synaptic",
    "synapse",
    "neurite",
    "axon",
    "dendrite",
    "electrophysiology",
    "calcium",
    "spinal cord",
)

ENTITY_SUFFIXES = (
    "mab",
    "nib",
    "pril",
    "sartan",
    "statin",
    "oxetine",
    "zole",
    "caine",
    "olol",
    "vir",
    "cycline",
    "mycin",
    "ersen",
    "parib",
    "ciclib",
    "gliptin",
    "gliflozin",
    "terol",
    "afil",
    "apine",
    "prazole",
    "glutide",
)

ENTITY_STOPWORDS = {
    "alzheimer",
    "parkinson",
    "amyotrophic",
    "als",
    "huntington",
    "rett",
    "frontotemporal",
    "neurodegeneration",
    "neurodegenerative",
    "epilepsy",
    "seizure",
    "brain",
    "spinal",
    "cord",
    "injury",
    "neuron",
    "neurons",
    "neuronal",
    "organoid",
    "organoids",
    "microglia",
    "astrocyte",
    "astrocytes",
    "synaptic",
    "synapse",
    "neurite",
    "axon",
    "dendrite",
    "calcium",
    "electrophysiology",
    "screen",
    "screening",
    "assay",
    "comparison",
    "comparative",
    "combination",
    "monotherapy",
    "therapy",
    "treatment",
    "drug",
    "drugs",
    "compound",
    "compounds",
    "disease",
    "model",
    "models",
    "human",
    "mouse",
    "rat",
    "ipsc",
    "gfp",
    "dmso",
    "vehicle",
    "control",
    "placebo",
    "survival",
    "viability",
    "pathology",
    "amyloid",
    "tau",
    "dose",
    "response",
    "high",
    "throughput",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _load_policy() -> dict[str, Any]:
    config = load_config(CONFIG_PATH)
    policy = config.get("retrospective_public_design_enriched_ranking_intake")
    if not isinstance(policy, dict):
        raise ValueError("Missing retrospective_public_design_enriched_ranking_intake configuration")
    authorization = policy["authorization"]
    search_plan = policy["search_plan"]
    if int(authorization["source_count_cap"]) != 5:
        raise ValueError("Design-enriched source cap drifted")
    if int(authorization["metadata_record_cap"]) != 150:
        raise ValueError("Design-enriched metadata cap drifted")
    if int(authorization["per_query_cap"]) != 25:
        raise ValueError("Design-enriched per-query cap drifted")
    if len(search_plan["queries"]) != int(search_plan["max_queries"]):
        raise ValueError("Design-enriched query count drifted")
    if int(search_plan["max_queries"]) * int(search_plan["max_ids_per_query"]) > int(
        search_plan["metadata_record_cap"]
    ):
        raise ValueError("Design-enriched query plan exceeds the metadata record cap")
    return policy


def _output_paths(policy: dict[str, Any]) -> dict[str, Path]:
    paths = {
        key: (REPO_ROOT / str(value)).resolve()
        for key, value in policy["outputs"].items()
    }
    for key, path in paths.items():
        if path != REPO_ROOT and REPO_ROOT not in path.parents:
            raise ValueError(f"Configured design-enriched output escapes repository: {key}")
    return paths


def _read_publication_uids(path: Path, field: str) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if field not in (reader.fieldnames or ()):
            raise ValueError(f"Missing {field} in exclusion table: {path}")
        return {
            str(row.get(field) or "").strip()
            for row in reader
            if str(row.get(field) or "").strip()
        }


def _assert_parent_state(policy: dict[str, Any]) -> dict[str, dict[str, object]]:
    expected = policy["parent_state"]
    frozen = _read_json(FROZEN_QUEUE_AUDIT_PATH)
    refresh = _read_json(REFRESH_QUEUE_AUDIT_PATH)
    prior_preregistration = _read_json(PRIOR_INTAKE_PREREGISTRATION_PATH)
    prior_intake = _read_json(PRIOR_INTAKE_AUDIT_PATH)
    prior_recovery = _read_json(PRIOR_INTAKE_RECOVERY_PATH)
    materialization = _read_json(MATERIALIZATION_AUDIT_PATH)
    replay = _read_json(REPLAY_AUDIT_PATH)
    observed = {
        "frozen_queue_digest": frozen.get("queue_digest"),
        "refresh_queue_digest": refresh.get("queue_digest"),
        "prior_intake_preregistration_digest": prior_preregistration.get("preregistration_digest"),
        "prior_intake_queue_digest": prior_intake.get("queue_digest"),
        "prior_intake_terminal_status": prior_recovery.get("status"),
        "status": materialization.get("status"),
        "accepted_records": materialization.get("accepted_record_count"),
        "rejected_records": materialization.get("rejected_record_count"),
        "independent_decision_sensitive_groups": replay.get("independent_decision_sensitive_group_count"),
        "required_groups": replay.get("minimum_independent_decision_sensitive_groups"),
        "split_files_created": replay.get("split_files_created"),
        "protected_holdout_access": replay.get("protected_holdout_access"),
    }
    required = {
        "frozen_queue_digest": expected["frozen_queue_digest"],
        "refresh_queue_digest": expected["refresh_queue_digest"],
        "prior_intake_preregistration_digest": expected["prior_intake_preregistration_digest"],
        "prior_intake_queue_digest": expected["prior_intake_queue_digest"],
        "prior_intake_terminal_status": "queue_exhausted_without_new_decision_sensitive_group",
        "status": expected["status"],
        "accepted_records": expected["accepted_records"],
        "rejected_records": expected["rejected_records"],
        "independent_decision_sensitive_groups": expected["independent_decision_sensitive_groups"],
        "required_groups": expected["required_groups"],
        "split_files_created": False,
        "protected_holdout_access": 0,
    }
    if observed != required:
        raise ValueError(f"Design-enriched parent state drifted: observed={observed} expected={required}")
    return {
        "frozen_queue_audit": _binding(FROZEN_QUEUE_AUDIT_PATH),
        "refresh_queue_audit": _binding(REFRESH_QUEUE_AUDIT_PATH),
        "prior_intake_preregistration": _binding(PRIOR_INTAKE_PREREGISTRATION_PATH),
        "prior_intake_audit": _binding(PRIOR_INTAKE_AUDIT_PATH),
        "prior_intake_recovery": _binding(PRIOR_INTAKE_RECOVERY_PATH),
        "materialization_audit": _binding(MATERIALIZATION_AUDIT_PATH),
        "replay_audit": _binding(REPLAY_AUDIT_PATH),
    }


def _collect_exclusions(policy: dict[str, Any]) -> dict[str, Any]:
    publication_uids: set[str] = set()
    for path, field in EXCLUSION_TABLES:
        publication_uids.update(_read_publication_uids(path, field))
    prior_snapshot = _read_json(PRIOR_METADATA_SNAPSHOT_PATH)
    prior_metadata_pmids = {
        str(pmid)
        for query in prior_snapshot["search_plan"]["queries"]
        for pmid in query["returned_pmids"]
    }
    prior_preregistration = _read_json(PRIOR_INTAKE_PREREGISTRATION_PATH)
    previous_preexposed_dois = {
        str(value).lower()
        for value in prior_preregistration.get("exclusions", {}).get("preexposed_dois", [])
    }
    pmids = {
        value.removeprefix("pubmed:")
        for value in publication_uids
        if value.startswith("pubmed:")
    }
    dois = {
        value.removeprefix("doi:").lower()
        for value in publication_uids
        if value.startswith("doi:")
    }
    return {
        "preexposed_publication_uids": sorted(publication_uids),
        "preexposed_pmids": sorted(pmids),
        "preexposed_dois": sorted(dois | previous_preexposed_dois),
        "prior_metadata_universe_pmids": sorted(prior_metadata_pmids, key=int),
        "prior_metadata_universe_count": len(prior_metadata_pmids),
        "excluded_source_policy": ["MDPI"],
        "excluded_doi_prefixes": list(policy["selection_policy"]["excluded_doi_prefixes"]),
    }


def _schema_errors(payload: dict[str, Any], schema_path: Path) -> list[str]:
    schema = _read_json(schema_path)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    return [f"{list(error.absolute_path)}: {error.message}" for error in errors]


def _validate_preregistration(payload: dict[str, Any], policy: dict[str, Any]) -> None:
    errors = _schema_errors(payload, PREREGISTRATION_SCHEMA_PATH)
    if errors:
        raise ValueError("Design-enriched preregistration schema errors: " + "; ".join(errors[:8]))
    digest_payload = dict(payload)
    observed_digest = str(digest_payload.pop("preregistration_digest"))
    if _canonical_digest(digest_payload) != observed_digest:
        raise ValueError("Design-enriched preregistration digest drifted")
    for field in ("intake_uid", "authorization", "parent_state", "search_plan", "selection_policy", "boundaries", "outputs"):
        if payload[field] != policy[field]:
            raise ValueError(f"Design-enriched preregistration {field} drifted")
    if payload["parent_state_digest"] != _canonical_digest(policy["parent_state"]):
        raise ValueError("Design-enriched parent-state digest drifted")


def preregister(*, write_output: bool = True) -> dict[str, Any]:
    policy = _load_policy()
    paths = _output_paths(policy)
    if paths["preregistration_manifest"].is_file():
        payload = _read_json(paths["preregistration_manifest"])
        _validate_preregistration(payload, policy)
        return payload
    payload: dict[str, Any] = {
        "schema_version": "neuronsifter-retrospective-public-design-enriched-ranking-intake-preregistration-v1",
        "status": "preregistered_before_metadata_retrieval",
        "intake_uid": policy["intake_uid"],
        "preregistered_at": _now(),
        "authorization": policy["authorization"],
        "parent_state": policy["parent_state"],
        "parent_state_digest": _canonical_digest(policy["parent_state"]),
        "parent_bindings": _assert_parent_state(policy),
        "search_plan": policy["search_plan"],
        "selection_policy": policy["selection_policy"],
        "exclusions": _collect_exclusions(policy),
        "boundaries": policy["boundaries"],
        "outputs": policy["outputs"],
    }
    payload["preregistration_digest"] = _canonical_digest(payload)
    _validate_preregistration(payload, policy)
    if write_output:
        write_json(paths["preregistration_manifest"], payload)
    return payload


def _entrez_email() -> str:
    configured = os.environ.get("NCBI_EMAIL", "").strip()
    if configured:
        return configured
    completed = subprocess.run(
        ["git", "config", "--get", "user.email"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    configured = completed.stdout.strip()
    if not configured:
        raise ValueError("NCBI metadata retrieval requires NCBI_EMAIL or git user.email")
    return configured


def _search_pubmed(query: str, *, retmax: int, sort: str) -> tuple[int, list[str]]:
    with Entrez.esearch(db="pubmed", term=query, retmax=retmax, sort=sort) as handle:
        result = Entrez.read(handle)
    return int(result["Count"]), [str(value) for value in result["IdList"]]


def _summaries(pmids: Iterable[str]) -> list[dict[str, Any]]:
    identifiers = list(pmids)
    records: list[dict[str, Any]] = []
    for start in range(0, len(identifiers), 50):
        batch = identifiers[start : start + 50]
        if not batch:
            continue
        with Entrez.esummary(db="pubmed", id=",".join(batch)) as handle:
            payload = Entrez.read(handle)
        for item in payload:
            article_ids = _article_ids(item)
            records.append(
                {
                    "pmid": str(item.get("Id") or article_ids.get("pubmed") or "").strip(),
                    "title": " ".join(str(item.get("Title") or "").split()),
                    "publication_date": str(item.get("PubDate") or "").strip(),
                    "journal": str(item.get("FullJournalName") or item.get("Source") or "").strip(),
                    "publication_types": sorted(str(value) for value in item.get("PubTypeList", [])),
                    "pmcid": article_ids.get("pmc", ""),
                    "doi": article_ids.get("doi", "").lower(),
                }
            )
    return records


def _europe_pmc_lite(pmid: str) -> dict[str, Any]:
    response = requests.get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={
            "query": f"EXT_ID:{pmid} AND SRC:MED",
            "format": "json",
            "resultType": "lite",
            "pageSize": "1",
        },
        timeout=(10, 30),
        headers={"User-Agent": "NeuronSifter-design-enriched-metadata-intake/1.0"},
    )
    response.raise_for_status()
    if len(response.content) > 512 * 1024:
        raise ValueError(f"Europe PMC metadata payload exceeded limit for PMID {pmid}")
    results = response.json().get("resultList", {}).get("result", [])
    if not results:
        return {
            "access_status": "not_found",
            "is_open_access": False,
            "has_supplement": False,
            "has_data": False,
        }
    record = results[0]
    return {
        "access_status": "resolved",
        "is_open_access": str(record.get("isOpenAccess") or "N") == "Y",
        "has_supplement": str(record.get("hasSuppl") or "N") == "Y",
        "has_data": str(record.get("hasData") or "N") == "Y",
    }


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(str(term).lower() in text for term in terms)


def _context_family(title: str) -> str | None:
    lowered = f" {title.lower()} "
    for family, terms in CONTEXT_PATTERNS:
        if _contains_any(lowered, terms):
            return family
    return None


def _objective_family(title: str) -> str | None:
    lowered = title.lower()
    for family, terms in OBJECTIVE_PATTERNS:
        if _contains_any(lowered, terms):
            return family
    return None


def _design_class(title: str) -> tuple[str, str, str, int] | None:
    lowered = f" {title.lower()} "
    for design_class, terms, arm_structure, prior, rank in DESIGN_PATTERNS:
        if design_class == "combination_vs_monotherapy":
            if all(term in lowered for term in terms):
                return design_class, arm_structure, prior, rank
        elif _contains_any(lowered, terms):
            return design_class, arm_structure, prior, rank
    return None


def _named_intervention_entities(title: str) -> list[str]:
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9-]{2,}\b", title)
    entities: list[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in ENTITY_STOPWORDS:
            continue
        has_digit_code = any(character.isdigit() for character in token) and any(
            character.isalpha() for character in token
        )
        has_drug_suffix = lowered.endswith(ENTITY_SUFFIXES)
        is_compact_uppercase = token.isupper() and 3 <= len(token) <= 12
        if has_digit_code or has_drug_suffix or is_compact_uppercase:
            entities.append(token)
    return list(dict.fromkeys(entities))


def _publication_year(value: str) -> int:
    match = re.search(r"\b(19|20)\d{2}\b", value)
    return int(match.group(0)) if match else 0


def _screen_record(
    record: dict[str, Any],
    *,
    query_uids: list[str],
    policy: dict[str, Any],
    exclusions: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    pmid = str(record.get("pmid") or "")
    doi = str(record.get("doi") or "").lower()
    title = str(record.get("title") or "")
    lowered = title.lower()
    if not pmid or not title:
        return None, "missing_identifier_or_title"
    if (
        pmid in exclusions["preexposed_pmids"]
        or pmid in exclusions["prior_metadata_universe_pmids"]
        or doi in exclusions["preexposed_dois"]
    ):
        return None, "prior_lineage_or_metadata_universe"
    if any(doi.startswith(prefix) for prefix in policy["selection_policy"]["excluded_doi_prefixes"]):
        return None, "source_policy_excluded"
    if not record.get("pmcid"):
        return None, "no_public_pmc_link"
    publication_types = {str(value) for value in record.get("publication_types", [])}
    if policy["selection_policy"]["required_publication_type"] not in publication_types:
        return None, "not_journal_article"
    if publication_types & set(policy["selection_policy"]["excluded_publication_types"]):
        return None, "excluded_publication_type"
    if _contains_any(lowered, policy["selection_policy"]["outcome_bearing_title_terms"]):
        return None, "outcome_bearing_title"
    if _contains_any(
        lowered,
        policy["selection_policy"]["non_original_or_non_interventional_title_terms"],
    ):
        return None, "non_original_or_non_interventional_title"
    context_family = _context_family(title)
    if context_family is None:
        return None, "no_disease_or_injury_context"
    if not _contains_any(lowered, CNS_MODEL_TERMS):
        return None, "no_cns_model_or_endpoint_signal"
    objective_family = _objective_family(title)
    if objective_family is None:
        return None, "no_prespecified_functional_endpoint_signal"
    design = _design_class(title)
    if design is None:
        return None, "no_prespecified_design_signal"
    entities = _named_intervention_entities(title)
    if len(entities) < int(policy["selection_policy"]["minimum_named_intervention_entities"]):
        return None, "fewer_than_two_named_intervention_entities"
    source_metadata = _europe_pmc_lite(pmid)
    if source_metadata["access_status"] != "resolved":
        raise RuntimeError(f"Europe PMC metadata enrichment unresolved for PMID {pmid}")
    design_class, arm_structure, prior, design_rank = design
    completeness_score = sum(
        bool(source_metadata[key])
        for key in ("is_open_access", "has_supplement", "has_data")
    )
    metadata_design_score = (
        len(entities) * 3
        + (4 - design_rank) * 2
        + len(query_uids)
        + completeness_score
    )
    return {
        **record,
        "query_uids": sorted(query_uids),
        "query_support_count": len(query_uids),
        "paper_type": "original_experimental_expected",
        "source_quality_tier": "primary_public_journal",
        "context_family": context_family,
        "objective_family": objective_family,
        "objective_uid": policy["selection_policy"]["objective_uids"][objective_family],
        "study_design_class": design_class,
        "therapeutic_arm_structure": arm_structure,
        "ranking_informativeness_prior": prior,
        "design_rank": design_rank,
        "named_intervention_entities": entities,
        "named_intervention_entity_count": len(entities),
        "source_data_metadata": source_metadata,
        "public_source_completeness": completeness_score,
        "metadata_design_score": metadata_design_score,
        "publication_year": _publication_year(str(record.get("publication_date") or "")),
    }, None


def _select_records(
    summaries: list[dict[str, Any]],
    *,
    query_membership: dict[str, list[str]],
    policy: dict[str, Any],
    exclusions: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], int]:
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for record in summaries:
        pmid = str(record.get("pmid") or "")
        screened, reason = _screen_record(
            record,
            query_uids=query_membership.get(pmid, []),
            policy=policy,
            exclusions=exclusions,
        )
        if screened is not None:
            eligible.append(screened)
        else:
            rejected.append(
                {
                    "pmid": pmid,
                    "reason": str(reason),
                    "title_sha256": hashlib.sha256(
                        str(record.get("title") or "").encode("utf-8")
                    ).hexdigest(),
                }
            )
    selected: list[dict[str, Any]] = []
    remaining = list(eligible)
    cap = int(policy["authorization"]["source_count_cap"])
    while remaining and len(selected) < cap:
        selected_contexts = {record["context_family"] for record in selected}
        remaining.sort(
            key=lambda item: (
                -int(item["named_intervention_entity_count"]),
                int(item["design_rank"]),
                item["context_family"] in selected_contexts,
                -int(item["query_support_count"]),
                -int(item["public_source_completeness"]),
                -int(item["publication_year"]),
                str(item["pmid"]),
            )
        )
        selected.append(remaining.pop(0))
    return selected, sorted(rejected, key=lambda item: (item["reason"], item["pmid"])), len(eligible)


def _queue_rows(
    selected: list[dict[str, Any]],
    *,
    preregistration: dict[str, Any],
    snapshot_path: Path,
    snapshot_sha256: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, record in enumerate(selected, start=1):
        pmid = str(record["pmid"])
        source_metadata = record["source_data_metadata"]
        rows.append(
            {
                "intake_uid": str(preregistration["intake_uid"]),
                "intake_sequence": str(index),
                "preregistration_digest": str(preregistration["preregistration_digest"]),
                "parent_state_digest": str(preregistration["parent_state_digest"]),
                "publication_uid": f"pubmed:{pmid}",
                "source_identifier": f"pmid:{pmid}",
                "publication_date": str(record["publication_date"]),
                "journal": str(record["journal"]),
                "title": str(record["title"]),
                "source_accession": str(record["pmcid"]),
                "doi": str(record["doi"]),
                "paper_type": str(record["paper_type"]),
                "source_quality_tier": str(record["source_quality_tier"]),
                "context_hint": f"{record['context_family']} design-enriched public CNS comparison",
                "context_family": str(record["context_family"]),
                "objective_uid": str(record["objective_uid"]),
                "study_design_class": str(record["study_design_class"]),
                "named_intervention_entities": ";".join(record["named_intervention_entities"]),
                "named_intervention_entity_count": str(record["named_intervention_entity_count"]),
                "therapeutic_arm_structure": str(record["therapeutic_arm_structure"]),
                "shared_control_expected": "true",
                "same_endpoint_time_expected": "true",
                "biological_replication_or_uncertainty_expected": "true",
                "source_data_metadata_status": (
                    "pmc_open=" + str(source_metadata["is_open_access"]).lower()
                    + ";supplement=" + str(source_metadata["has_supplement"]).lower()
                    + ";data=" + str(source_metadata["has_data"]).lower()
                ),
                "independence_key": f"pubmed:{pmid}",
                "ranking_informativeness_prior": str(record["ranking_informativeness_prior"]),
                "metadata_design_score": str(record["metadata_design_score"]),
                "experimental_numeric_evidence_status": "unknown_pre_recovery",
                "priority_order": str(index),
                "queue_status": "preregistered_for_public_artifact_recovery",
                "metadata_basis": (
                    "pubmed esearch and esummary plus europe pmc lite metadata only; "
                    "two named title entities; disease cns functional and design signals"
                ),
                "metadata_snapshot_date": str(preregistration["preregistered_at"])[:10],
                "metadata_snapshot_path": snapshot_path.relative_to(REPO_ROOT).as_posix(),
                "metadata_snapshot_sha256": snapshot_sha256,
                "prior_metadata_universe_excluded": "true",
                "preexposed_sources_excluded": "true",
                "abstracts_retrieved_before_priority": "false",
                "full_text_retrieved_before_priority": "false",
                "figures_retrieved_before_priority": "false",
                "supplementary_retrieved_before_priority": "false",
                "outcome_values_used_for_priority": "false",
                "observed_effect_used_for_priority": "false",
                "p_values_used_for_priority": "false",
                "author_winner_statement_used_for_priority": "false",
                "protected_holdout_access": "0",
            }
        )
    return rows


def _queue_digest(rows: list[dict[str, str]], preregistration: dict[str, Any]) -> str:
    return _canonical_digest(
        {
            "queue_digest_contract": "intake_envelope_v1",
            "intake_uid": preregistration["intake_uid"],
            "preregistration_digest": preregistration["preregistration_digest"],
            "parent_state_digest": preregistration["parent_state_digest"],
            "records": rows,
        }
    )


def repair_empty_queue_digest(*, write_output: bool = True) -> dict[str, Any]:
    policy = _load_policy()
    paths = _output_paths(policy)
    preregistration = preregister(write_output=False)
    audit = _read_json(paths["intake_audit"])
    with paths["source_queue"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != QUEUE_FIELDS:
            raise ValueError("Design-enriched queue columns drifted during digest repair")
        rows = [{key: str(value or "") for key, value in row.items()} for row in reader]
    if audit.get("status") != "metadata_only_queue_exhausted" or rows:
        raise ValueError("Empty-queue digest repair requires a canonical empty queue")
    legacy_digest = _canonical_digest(rows)
    if audit.get("queue_digest") != legacy_digest:
        raise ValueError("Current queue digest is not the legacy records-only digest")
    repaired = {
        **audit,
        "queue_digest": _queue_digest(rows, preregistration),
        "queue_digest_contract": "intake_envelope_v1",
    }
    errors = _schema_errors(repaired, INTAKE_SCHEMA_PATH)
    if errors:
        raise ValueError("Repaired design-enriched intake schema errors: " + "; ".join(errors[:8]))
    invalidation = {
        "schema_version": "neuronsifter-design-enriched-empty-queue-digest-invalidation-v1",
        "status": "invalidated_non_unique_records_only_empty_queue_digest",
        "scientific_result_changed": False,
        "metadata_universe_changed": False,
        "selection_changed": False,
        "queue_records_changed": False,
        "old_queue_digest": legacy_digest,
        "new_queue_digest": repaired["queue_digest"],
        "new_queue_digest_contract": "intake_envelope_v1",
        "old_audit_binding": _binding(paths["intake_audit"]),
        "repaired_at": _now(),
    }
    if write_output:
        write_json(EMPTY_DIGEST_PAYLOAD_PATH, audit)
        write_json(EMPTY_DIGEST_INVALIDATION_PATH, invalidation)
        write_json(paths["intake_audit"], repaired)
    return repaired


def discover(*, write_output: bool = True) -> dict[str, Any]:
    policy = _load_policy()
    paths = _output_paths(policy)
    preregistration = preregister(write_output=write_output)
    if paths["intake_audit"].is_file():
        return _read_json(paths["intake_audit"])
    if paths["metadata_snapshot"].exists() or paths["source_queue"].exists():
        raise ValueError("Partial design-enriched intake outputs exist without an audit")
    _assert_parent_state(policy)
    Entrez.email = _entrez_email()
    Entrez.tool = "NeuronSifterDesignEnrichedRankingIntake"
    api_key = os.environ.get("NCBI_API_KEY", "").strip()
    if api_key:
        Entrez.api_key = api_key
    search_plan = policy["search_plan"]
    query_results: list[dict[str, Any]] = []
    query_membership: dict[str, list[str]] = {}
    for query_spec in search_plan["queries"]:
        count, pmids = _search_pubmed(
            str(query_spec["query"]),
            retmax=int(search_plan["max_ids_per_query"]),
            sort=str(search_plan["sort"]),
        )
        query_uid = str(query_spec["query_uid"])
        query_results.append(
            {
                "query_uid": query_uid,
                "query": query_spec["query"],
                "count": count,
                "retmax": int(search_plan["max_ids_per_query"]),
                "returned_pmids": pmids,
            }
        )
        for pmid in pmids:
            query_membership.setdefault(pmid, []).append(query_uid)
    all_pmids = sorted(query_membership, key=int)
    if len(all_pmids) > int(search_plan["metadata_record_cap"]):
        raise ValueError("Resolved PubMed metadata universe exceeded the preregistered cap")
    metadata_records = _summaries(all_pmids)
    selected, rejected, eligible_count = _select_records(
        metadata_records,
        query_membership=query_membership,
        policy=policy,
        exclusions=preregistration["exclusions"],
    )
    snapshot = {
        "schema_version": "neuronsifter-design-enriched-pubmed-metadata-snapshot-v1",
        "source": "NCBI PubMed ESearch and ESummary plus Europe PMC lite metadata",
        "retrieved_at": _now(),
        "retrieval_mode": "metadata_only",
        "preregistration_binding": {
            "path": paths["preregistration_manifest"].relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(paths["preregistration_manifest"]),
            "preregistration_digest": preregistration["preregistration_digest"],
        },
        "search_plan": {
            "bounded_query_count": len(query_results),
            "maximum_ids_per_query": int(search_plan["max_ids_per_query"]),
            "metadata_record_cap": int(search_plan["metadata_record_cap"]),
            "queries": query_results,
        },
        "source_boundaries": {
            "abstracts_retrieved": False,
            "full_text_retrieved": False,
            "figures_retrieved": False,
            "supplementary_retrieved": False,
            "outcome_values_retrieved": False,
            "p_values_retrieved": False,
            "author_winner_statements_retrieved": False,
            "protected_holdout_access": 0,
        },
        "source_policy": {
            "excluded_sources": ["MDPI"],
            "excluded_doi_prefixes": policy["selection_policy"]["excluded_doi_prefixes"],
        },
        "exclusions": {
            **preregistration["exclusions"],
            "rejected_metadata_records": rejected,
        },
        "selection": {
            "source_count_cap": int(policy["authorization"]["source_count_cap"]),
            "unique_metadata_record_count": len(metadata_records),
            "eligible_metadata_record_count": eligible_count,
            "selected_pmids": [record["pmid"] for record in selected],
            "ranking_order": policy["selection_policy"]["ranking_order"],
            "numeric_evidence_score": "not_scored_pre_recovery",
            "outcome_values_used_for_priority": False,
            "observed_effect_used_for_priority": False,
            "p_values_used_for_priority": False,
            "author_winner_statement_used_for_priority": False,
        },
        "selected_records": selected,
    }
    if write_output:
        write_json(paths["metadata_snapshot"], snapshot)
        snapshot_binding = _binding(paths["metadata_snapshot"])
    else:
        serialized = json.dumps(snapshot, indent=2, ensure_ascii=True).encode("utf-8") + b"\n"
        snapshot_binding = {
            "path": paths["metadata_snapshot"].relative_to(REPO_ROOT).as_posix(),
            "bytes": len(serialized),
            "sha256": hashlib.sha256(serialized).hexdigest(),
        }
    rows = _queue_rows(
        selected,
        preregistration=preregistration,
        snapshot_path=paths["metadata_snapshot"],
        snapshot_sha256=str(snapshot_binding["sha256"]),
    )
    if write_output:
        write_tsv(paths["source_queue"], list(QUEUE_FIELDS), rows)
    status = "metadata_only_queue_frozen" if rows else "metadata_only_queue_exhausted"
    audit: dict[str, Any] = {
        "schema_version": "neuronsifter-retrospective-public-design-enriched-ranking-intake-v1",
        "status": status,
        "intake_uid": preregistration["intake_uid"],
        "preregistration_digest": preregistration["preregistration_digest"],
        "parent_state_digest": preregistration["parent_state_digest"],
        "source_count_cap": int(policy["authorization"]["source_count_cap"]),
        "metadata_record_cap": int(policy["authorization"]["metadata_record_cap"]),
        "record_count": len(rows),
        "queue_digest": _queue_digest(rows, preregistration),
        "queue_digest_contract": "intake_envelope_v1",
        "queue_path": paths["source_queue"].relative_to(REPO_ROOT).as_posix(),
        "metadata_snapshot_binding": snapshot_binding,
        "priority_order": [row["publication_uid"] for row in rows],
        "selection_summary": {
            "query_count": len(query_results),
            "returned_identifier_count": len(all_pmids),
            "metadata_record_count": len(metadata_records),
            "eligible_record_count": eligible_count,
            "selected_record_count": len(rows),
            "rejected_record_count": len(rejected),
        },
        "parent_state": preregistration["parent_state"],
        "historical_queues_unchanged": True,
        "metadata_network_access_performed": True,
        "retrieval_mode": "metadata_only",
        "abstracts_retrieved_before_priority": False,
        "full_text_retrieved_before_priority": False,
        "figures_retrieved_before_priority": False,
        "supplementary_retrieved_before_priority": False,
        "outcome_values_used_for_priority": False,
        "observed_effect_used_for_priority": False,
        "p_values_used_for_priority": False,
        "author_winner_statement_used_for_priority": False,
        "protected_holdout_access": 0,
        "formal_evidence_admitted": 0,
        "next_action": (
            "recover_preregistered_sources_in_priority_order"
            if rows
            else "stop_queue_exhausted_preserve_two_of_three_groups"
        ),
        "records": rows,
        "created_at": _now(),
    }
    errors = _schema_errors(audit, INTAKE_SCHEMA_PATH)
    if errors:
        raise ValueError("Design-enriched intake schema errors: " + "; ".join(errors[:8]))
    if write_output:
        write_json(paths["intake_audit"], audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preregister", "discover", "repair-empty-queue-digest"))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    if args.mode == "preregister":
        result = preregister(write_output=not args.no_write)
    elif args.mode == "repair-empty-queue-digest":
        result = repair_empty_queue_digest(write_output=not args.no_write)
    else:
        result = discover(write_output=not args.no_write)
    keys = (
        "status",
        "intake_uid",
        "preregistration_digest",
        "parent_state_digest",
        "source_count_cap",
        "metadata_record_cap",
        "record_count",
        "queue_digest",
        "next_action",
    )
    print(json.dumps({key: result[key] for key in keys if key in result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
