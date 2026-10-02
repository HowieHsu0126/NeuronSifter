"""Recover and fail-closed qualify the frozen public CNS ranking intake source."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

from jsonschema import Draft202012Validator
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_retrospective_public_cns_ranking_intake import (  # noqa: E402
    QUEUE_FIELDS,
)
from Libs.scripts.builders.build_retrospective_public_evidence_materialization import (  # noqa: E402
    MANIFEST_FIELDS,
    MANIFEST_PATH,
)
from Libs.utils.data.raw_data_registry import temporary_output_path, write_json, write_tsv  # noqa: E402


QUEUE_PATH = REPO_ROOT / "Input/manifests/retrospective_public/cns_ranking_intake_queue.tsv"
QUEUE_AUDIT_PATH = REPO_ROOT / "Output/audits/retrospective_public_cns_ranking_intake.json"
SOURCE_DIR = REPO_ROOT / "Input/raw/public/retrospective/public_cns_ranking_intake"
SOURCE_PATH = SOURCE_DIR / "pmc9635626_bioc.xml"
SOURCE_MANIFEST_PATH = SOURCE_DIR / "pmc9635626_source_artifact_manifest.json"
AUDIT_PATH = REPO_ROOT / "Output/audits/retrospective_public_cns_ranking_source_recovery.json"
SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/retrospective_public_cns_ranking_source_recovery.schema.json"
SOURCE_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/PMC9635626/unicode"
EXPECTED_QUEUE_DIGEST = "9349a8a111095e19b72195aad4240f4c4ae84e311945d458c003e5e414885448"
EXPECTED_PUBLICATION_UID = "pubmed:36102902"
EXPECTED_SOURCE_ACCESSION = "PMC9635626"
REJECTION_RECORD_UID = "pubmed36102902:protocol:no_therapeutic_ranking_context"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _read_queue() -> list[dict[str, str]]:
    with QUEUE_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != QUEUE_FIELDS:
            raise ValueError("Public CNS ranking intake queue columns drifted")
        rows = [{key: str(value or "") for key, value in row.items()} for row in reader]
    if len(rows) != 1:
        raise ValueError(f"Expected one frozen intake source, observed {len(rows)}")
    return rows


def _download_source() -> None:
    if SOURCE_PATH.is_file():
        return
    response = requests.get(
        SOURCE_URL,
        timeout=(10, 60),
        headers={"User-Agent": "NeuronSifter-public-source-recovery/1.0"},
    )
    response.raise_for_status()
    if not response.content.startswith(b"<?xml"):
        raise ValueError("Recovered PMC BioC payload is not XML")
    if len(response.content) > 10 * 1024 * 1024:
        raise ValueError("Recovered PMC BioC payload exceeds the bounded source limit")
    SOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = temporary_output_path(SOURCE_PATH)
    temporary_path.write_bytes(response.content)
    temporary_path.replace(SOURCE_PATH)


def _passage_texts(root: ET.Element) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for passage in root.findall(".//passage"):
        infons = {
            str(node.get("key") or ""): str(node.text or "")
            for node in passage.findall("infon")
        }
        text = " ".join(str(passage.findtext("text") or "").split())
        if text:
            records.append((infons.get("section_type") or infons.get("type") or "", text))
    return records


def _source_structure(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    document = root.find("document")
    if document is None:
        raise ValueError("PMC BioC XML has no document")
    document_id = str(document.findtext("id") or "")
    first_passage = document.find("passage")
    if first_passage is None:
        raise ValueError("PMC BioC XML has no passage metadata")
    infons = {
        str(node.get("key") or ""): str(node.text or "")
        for node in first_passage.findall("infon")
    }
    passages = _passage_texts(root)
    all_text = "\n".join(text for _, text in passages)
    lowered = all_text.lower()
    figure_texts = [text for section, text in passages if section == "FIG"]
    table_texts = [text for section, text in passages if section == "TABLE"]
    figure_seven = next(
        (
            text
            for text in figure_texts
            if "mean ± sd from one experiment in duplicate" in text.lower()
        ),
        "",
    )
    structure = {
        "document_id": document_id,
        "pmid": infons.get("article-id_pmid", ""),
        "pmcid": f"PMC{infons.get('article-id_pmc', '')}",
        "doi": infons.get("article-id_doi", ""),
        "license": infons.get("license", ""),
        "protocol_marker_count": lowered.count("basic protocol"),
        "figure_passage_count": len(figure_texts),
        "table_passage_count": len(table_texts),
        "positive_control_compounds": sorted(
            compound
            for compound in ("rotenone", "tetraoctylammonium bromide")
            if compound in lowered
        ),
        "vehicle_control_compounds": ["DMSO"] if "dmso" in lowered else [],
        "unspecified_test_compounds_present": "test compounds" in lowered,
        "disease_specific_model_present": False,
        "named_therapeutic_action_count": 0,
        "biological_replicate_count_recoverable": False,
        "demonstration_replication": (
            "one_experiment_in_duplicate" if figure_seven else "not_recoverable"
        ),
    }
    expected = {
        "document_id": "9635626",
        "pmid": "36102902",
        "pmcid": "PMC9635626",
        "doi": "10.1002/cpz1.542",
        "protocol_marker_count_minimum": 3,
        "figure_passage_count_minimum": 1,
        "positive_control_compounds": ["rotenone", "tetraoctylammonium bromide"],
        "vehicle_control_compounds": ["DMSO"],
        "unspecified_test_compounds_present": True,
        "demonstration_replication": "one_experiment_in_duplicate",
    }
    for key in ("document_id", "pmid", "pmcid", "doi", "positive_control_compounds", "vehicle_control_compounds", "unspecified_test_compounds_present", "demonstration_replication"):
        if structure[key] != expected[key]:
            raise ValueError(f"Recovered source structure drifted for {key}")
    if structure["protocol_marker_count"] < expected["protocol_marker_count_minimum"]:
        raise ValueError("Recovered source lacks the expected protocol structure")
    if structure["figure_passage_count"] < expected["figure_passage_count_minimum"]:
        raise ValueError("Recovered source lacks figure passages")
    return structure


def build(*, write_output: bool = True) -> dict[str, Any]:
    queue_audit = _read_json(QUEUE_AUDIT_PATH)
    rows = _read_queue()
    row = rows[0]
    if queue_audit.get("queue_digest") != EXPECTED_QUEUE_DIGEST:
        raise ValueError("Frozen CNS ranking intake digest drifted")
    if row["publication_uid"] != EXPECTED_PUBLICATION_UID:
        raise ValueError("Frozen CNS ranking intake publication drifted")
    if row["source_accession"] != EXPECTED_SOURCE_ACCESSION:
        raise ValueError("Frozen CNS ranking intake source accession drifted")
    _download_source()
    structure = _source_structure(SOURCE_PATH)
    qualification = {
        "status": "rejected_non_formal",
        "record_status": "rejected",
        "evidence_grade": "D",
        "evidence_role": "non_formal_rejection",
        "reason": (
            "preregistered source is a general neurite-outgrowth assay protocol; named conditions are "
            "neurotoxicant and vehicle controls, test compounds are unspecified, no disease-specific "
            "therapeutic action set is defined, and the quantitative demonstration is one experiment "
            "in duplicate rather than independent biological replication"
        ),
    }
    expected_manifest_fields = {
        "schema_version": "neuronsifter-public-cns-ranking-source-artifact-v1",
        "publication_uid": EXPECTED_PUBLICATION_UID,
        "source_accession": EXPECTED_SOURCE_ACCESSION,
        "source_url": SOURCE_URL,
        "queue_binding": {
            "path": QUEUE_PATH.relative_to(REPO_ROOT).as_posix(),
            "queue_digest": EXPECTED_QUEUE_DIGEST,
            "priority_order": 1,
        },
        "source_artifact": _binding(SOURCE_PATH),
        "source_structure": structure,
        "qualification": qualification,
        "citation_substitution_performed": False,
        "protected_holdout_access": 0,
    }
    if SOURCE_MANIFEST_PATH.is_file():
        source_manifest = _read_json(SOURCE_MANIFEST_PATH)
        for key, expected in expected_manifest_fields.items():
            if source_manifest.get(key) != expected:
                raise ValueError(f"Existing source manifest drifted for {key}")
        if not source_manifest.get("retrieved_at"):
            raise ValueError("Existing source manifest lacks retrieval time")
    else:
        source_manifest = {**expected_manifest_fields, "retrieved_at": _now()}
        if write_output:
            write_json(SOURCE_MANIFEST_PATH, source_manifest)
    manifest_binding = (
        _binding(SOURCE_MANIFEST_PATH)
        if write_output
        else {
            "path": SOURCE_MANIFEST_PATH.relative_to(REPO_ROOT).as_posix(),
            "bytes": len(json.dumps(source_manifest, indent=2, ensure_ascii=True).encode("utf-8")) + 1,
            "sha256": hashlib.sha256(
                (json.dumps(source_manifest, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
            ).hexdigest(),
        }
    )
    audit = {
        "schema_version": "neuronsifter-retrospective-public-cns-ranking-source-recovery-v1",
        "status": "queue_exhausted_without_new_decision_sensitive_group",
        "intake_uid": row["intake_uid"],
        "queue_digest": EXPECTED_QUEUE_DIGEST,
        "queue_record_count": 1,
        "queue_consumed_record_count": 1,
        "queue_remaining_record_count": 0,
        "publication_uid": EXPECTED_PUBLICATION_UID,
        "source_accession": EXPECTED_SOURCE_ACCESSION,
        "source_manifest_binding": manifest_binding,
        "source_artifact_binding": _binding(SOURCE_PATH),
        "source_structure": structure,
        "qualification": qualification,
        "citation_substitution_performed": False,
        "protected_holdout_access": 0,
        "formal_evidence_admitted": False,
        "candidate_set_created": False,
        "independent_decision_sensitive_group_count": 2,
        "required_group_count": 3,
        "next_action": "stop_new_intake_queue_exhausted_preserve_two_of_three_groups",
    }
    schema = _read_json(SCHEMA_PATH)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(audit),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        detail = "; ".join(f"{list(error.absolute_path)}: {error.message}" for error in errors[:8])
        raise ValueError(f"CNS ranking source recovery schema errors: {detail}")
    if write_output:
        write_json(AUDIT_PATH, audit)
    return audit


def _rejection_row(audit: dict[str, Any]) -> dict[str, str]:
    source = audit["source_artifact_binding"]
    return {
        "record_uid": REJECTION_RECORD_UID,
        "record_status": "rejected",
        "evidence_uid": "",
        "publication_uid": EXPECTED_PUBLICATION_UID,
        "study_uid": "study:PMC9635626:neurite_outgrowth_assay_protocol",
        "publication_date": "2022 Sep",
        "source_snapshot_uid": "pmc:PMC9635626:public_snapshot:20260827",
        "source_artifact_path": str(source["path"]),
        "source_locator": "PMC9635626:protocol:Figure7:rotenone_control_demonstration",
        "source_content_hash": str(source["sha256"]),
        "extraction_artifact_hash": "",
        "extraction_method": "bioc_xml_structure_and_source_semantics_audit",
        "extraction_qc_status": "rejected_protocol_nontherapeutic_control_demonstration",
        "public_access_status": "public",
        "measurement_protocol": "not_admitted_protocol_demonstration",
        "biological_context_uid": "unavailable_context",
        "common_control_uid": "unavailable_control",
        "structured_action_uid": "unavailable_action",
        "action_family_uid": "unavailable_family",
        "target_program_uid": "unavailable_target",
        "regimen_family_uid": "unavailable_regimen",
        "objective_uid": "unavailable_objective",
        "candidate_set_uid": "unavailable_candidate",
        "outcome_uid": "unavailable_outcome",
        "arm_mean": "unavailable_mean",
        "biological_replicate_count": "unavailable_n",
        "uncertainty": "unavailable_uncertainty",
        "uncertainty_kind": "unavailable_uncertainty_kind",
        "unit": "unavailable_unit",
        "outcome_visibility": "available_post_queue_freeze",
        "target_retrieval_blocked": "false",
        "tool_output_only": "false",
        "evidence_role": "non_formal_rejection",
        "evidence_grade": "D",
        "decision_domain_classification": "",
        "split_role": "",
        "rejection_reason": str(audit["qualification"]["reason"]),
    }


def materialize_rejection() -> dict[str, Any]:
    audit = build(write_output=True)
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError("Retrospective evidence materialization columns drifted")
        rows = [{key: str(value or "") for key, value in row.items()} for row in reader]
    expected = _rejection_row(audit)
    existing = [row for row in rows if row["record_uid"] == REJECTION_RECORD_UID]
    if existing:
        if existing != [expected]:
            raise ValueError("Existing protocol rejection row drifted")
        return {"status": "already_materialized", "record_uid": REJECTION_RECORD_UID}
    rows.append(expected)
    write_tsv(MANIFEST_PATH, list(MANIFEST_FIELDS), rows)
    return {
        "status": "rejection_materialized",
        "record_uid": REJECTION_RECORD_UID,
        "record_count": len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("recover", "materialize-rejection"))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    if args.mode == "recover":
        result = build(write_output=not args.no_write)
    else:
        if args.no_write:
            raise ValueError("materialize-rejection does not support --no-write")
        result = materialize_rejection()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
