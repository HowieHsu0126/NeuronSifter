"""Build the post-preregistration recovery audit for the bounded ranking refresh."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_retrospective_ranking_source_refresh_queue import (  # noqa: E402
    FROZEN_PARENT_QUEUE_DIGEST,
    build as build_refresh_queue,
)

SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/retrospective_public_ranking_source_recovery.schema.json"
SOURCE_MANIFEST_PATH = REPO_ROOT / "Input/raw/public/retrospective/ad_neuronal_hyperexcitability_screen/source_artifact_manifest.json"
AUDIT_PATH = REPO_ROOT / "Output/audits/retrospective_public_ranking_source_recovery.json"

EXPECTED_METADATA_PATH = "Input/raw/public/retrospective/ad_neuronal_hyperexcitability_screen/pubmed_metadata_snapshot.json"
EXPECTED_METADATA_SHA256 = "db04cf6c78e184f4ffb3cd41140f73f0217ce4f34c34ea83ba7e78a785df6de7"
EXPECTED_SOURCE_PATH = "Input/raw/public/retrospective/ad_neuronal_hyperexcitability_screen/pmc12449591_commentary.xml"
EXPECTED_SOURCE_SHA256 = "ca5ab2d59fddffe78d0312de1052cefea9d42269a3f366d73838d0507611b72c"
EXPECTED_REFRESH_QUEUE_DIGEST = "34db9a92c841568b7b8e26f922bac1d80cdeb6823fced6384ebf0788ae2034d4"


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


def _hash_binding(path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": _sha256(path),
    }


def _text(element: ET.Element | None) -> str:
    return "".join(element.itertext()).strip() if element is not None else ""


def _recover_structure(source_path: Path) -> dict[str, object]:
    root = ET.parse(source_path).getroot()
    article = root.find("article")
    if article is None:
        raise ValueError("PMC XML has no article element")
    article_type = article.get("article-type")
    title = _text(article.find("./front/article-meta/title-group/article-title"))
    subject_group = _text(article.find("./front/article-meta/article-categories/subj-group/subject"))
    article_ids = {
        item.get("pub-id-type"): _text(item)
        for item in article.findall("./front/article-meta/article-id")
    }
    references = [
        _text(item)
        for item in article.findall("./back/ref-list/ref/mixed-citation/article-title")
    ]
    reference_text = " ".join(references)
    return {
        "article_type": article_type,
        "subject_group": subject_group,
        "title": title,
        "pmid": article_ids.get("pmid"),
        "pmcid": article_ids.get("pmcid"),
        "doi": article_ids.get("doi"),
        "figure_count": len(article.findall(".//fig")),
        "table_count": len(article.findall(".//table-wrap")),
        "supplementary_material_count": len(article.findall(".//supplementary-material")),
        "quantitative_arm_table_present": False,
        "mean_n_uncertainty_recoverable": False,
        "article_reports_original_experiment": False,
        "references_knox_original_doi": (
            "10.1177/13872877251343321"
            if "Alzheimer’s disease-associated genotypes differentially influence chronic evoked seizure outcomes" in reference_text
            else ""
        ),
    }


def build(*, write_output: bool = True) -> dict[str, object]:
    source_manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    source_path = REPO_ROOT / EXPECTED_SOURCE_PATH
    metadata_path = REPO_ROOT / EXPECTED_METADATA_PATH
    source_binding = _binding(source_path)
    metadata_binding = _binding(metadata_path)
    if source_binding["sha256"] != EXPECTED_SOURCE_SHA256:
        raise ValueError("Recovered source artifact hash drifted")
    if metadata_binding["sha256"] != EXPECTED_METADATA_SHA256:
        raise ValueError("Metadata snapshot hash drifted")
    if source_manifest.get("publication_uid") != "pubmed:40754800":
        raise ValueError("Source artifact manifest publication binding drifted")
    if source_manifest.get("source_accession") != "PMC12449591":
        raise ValueError("Source artifact manifest accession binding drifted")
    if source_manifest.get("metadata_snapshot_binding") != _hash_binding(metadata_path):
        raise ValueError("Source artifact manifest metadata binding drifted")
    declared_source = source_manifest["source_artifact"]
    if declared_source != source_binding:
        raise ValueError("Source artifact manifest binding drifted")
    declared_qualification = source_manifest.get("qualification", {})
    qualification = {
        key: declared_qualification.get(key)
        for key in ("status", "record_status", "evidence_grade", "evidence_role", "reason")
    }
    if qualification != {
        "status": "rejected_non_formal",
        "record_status": "rejected",
        "evidence_grade": "D",
        "evidence_role": "non_formal_rejection",
        "reason": qualification["reason"],
    } or not qualification["reason"]:
        raise ValueError("Source artifact manifest qualification drifted")
    xml_structure = _recover_structure(source_path)
    expected_structure = source_manifest["xml_structure_audit"]
    for key, expected in expected_structure.items():
        if xml_structure.get(key) != expected:
            raise ValueError(f"Recovered XML structure drifted for {key}")
    refresh = build_refresh_queue(write_output=False)
    if refresh["parent_frozen_queue_digest"] != FROZEN_PARENT_QUEUE_DIGEST:
        raise ValueError("Frozen parent queue digest drifted")
    if refresh["queue_digest"] != EXPECTED_REFRESH_QUEUE_DIGEST:
        raise ValueError("Refresh queue digest drifted")
    audit: dict[str, object] = {
        "schema_version": "neuronsifter-retrospective-public-ranking-source-recovery-audit-v1",
        "status": "rejected_preregistered_source_commentary",
        "publication_uid": "pubmed:40754800",
        "source_accession": "PMC12449591",
        "parent_frozen_queue_digest": FROZEN_PARENT_QUEUE_DIGEST,
        "refresh_queue_digest": refresh["queue_digest"],
        "metadata_snapshot_binding": _hash_binding(metadata_path),
        "source_manifest_binding": _binding(SOURCE_MANIFEST_PATH),
        "source_artifact_binding": source_binding,
        "xml_structure": xml_structure,
        "qualification": qualification,
        "protected_holdout_access": 0,
        "formal_evidence_admitted": False,
        "candidate_set_created": False,
        "next_action": "stop_bounded_refresh_preserve_two_of_three_decision_sensitive_groups",
    }
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(audit), key=lambda error: list(error.absolute_path))
    if errors:
        detail = "; ".join(f"{list(error.absolute_path)}: {error.message}" for error in errors[:8])
        raise ValueError(f"Recovery audit schema error: {detail}")
    if write_output:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(write_output=not args.no_write), ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
