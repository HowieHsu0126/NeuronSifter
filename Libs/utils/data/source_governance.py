"""Deterministic parsers for the public metadata backbone.

These parsers normalize official source records without assigning observation
eligibility. PubMed records are deliberately restricted to publication
metadata and carry explicit fail-closed eligibility fields.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Iterable, Iterator, Mapping
import xml.etree.ElementTree as ET
import zipfile


MetadataRecord = dict[str, object]

PUBMED_EVIDENCE_ROLE = "publication_metadata_only"
PUBMED_RIGHTS_STATUS = "metadata_only_no_full_text"


class SourceParseError(ValueError):
    """Raised when source bytes cannot satisfy the canonical parser contract."""


def _require_source_context(source_id: str, release_identifier: str) -> None:
    if not source_id.strip():
        raise SourceParseError("source_id must be non-empty")
    if not release_identifier.strip():
        raise SourceParseError("release_identifier must be non-empty")


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        value = raw_value.strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _split_pipe_values(value: object) -> list[str]:
    return _deduplicate(str(value or "").split("|"))


def _normalized_header(value: str) -> str:
    normalized = value.lstrip("\ufeff").strip().casefold()
    return re.sub(r"_+", "_", re.sub(r"[\s-]+", "_", normalized))


def _field_map(fieldnames: Iterable[str | None]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_name in fieldnames:
        if raw_name is None:
            continue
        normalized = _normalized_header(raw_name)
        if normalized in fields:
            raise SourceParseError(f"duplicate normalized header: {normalized}")
        fields[normalized] = raw_name
    return fields


def _row_value(
    row: Mapping[str, object],
    fields: Mapping[str, str],
    field_name: str,
) -> str:
    original_name = fields.get(field_name)
    return str(row.get(original_name, "") if original_name else "").strip()


def _assert_unique_record_id(
    record_id: str,
    seen: set[str],
    *,
    source_id: str,
) -> None:
    if record_id in seen:
        raise SourceParseError(f"duplicate {source_id} record_id: {record_id}")
    seen.add(record_id)


def parse_hgnc_tsv(
    path: str | Path,
    *,
    release_identifier: str,
    source_id: str = "hgnc",
) -> list[MetadataRecord]:
    """Parse the official HGNC complete-set TSV using its underscore headers."""

    _require_source_context(source_id, release_identifier)
    source_path = Path(path)
    records: list[MetadataRecord] = []
    seen: set[str] = set()
    with source_path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = _field_map(reader.fieldnames or [])
        missing = sorted({"hgnc_id", "symbol"} - set(fields))
        if missing:
            raise SourceParseError(
                f"HGNC TSV is missing required underscore headers: {', '.join(missing)}"
            )
        for row_number, row in enumerate(reader, start=2):
            hgnc_id = _row_value(row, fields, "hgnc_id")
            symbol = _row_value(row, fields, "symbol")
            if not hgnc_id or not symbol:
                raise SourceParseError(
                    f"HGNC row {row_number} is missing hgnc_id or symbol"
                )
            _assert_unique_record_id(hgnc_id, seen, source_id=source_id)

            alias_symbols = _split_pipe_values(_row_value(row, fields, "alias_symbol"))
            previous_symbols = _split_pipe_values(_row_value(row, fields, "prev_symbol"))
            entrez_id = _row_value(row, fields, "entrez_id") or None
            ensembl_gene_id = _row_value(row, fields, "ensembl_gene_id") or None
            uniprot_ids = _split_pipe_values(_row_value(row, fields, "uniprot_ids"))
            refseq_accessions = _split_pipe_values(
                _row_value(row, fields, "refseq_accession")
            )

            external_ids = {
                "entrez_gene": [entrez_id] if entrez_id else [],
                "ensembl_gene": [ensembl_gene_id] if ensembl_gene_id else [],
                "uniprot": uniprot_ids,
                "refseq": refseq_accessions,
            }
            records.append(
                {
                    "source_id": source_id,
                    "release_identifier": release_identifier,
                    "record_type": "gene_identifier_metadata",
                    "record_id": hgnc_id,
                    "hgnc_id": hgnc_id,
                    "approved_symbol": symbol,
                    "approved_name": _row_value(row, fields, "name") or None,
                    "status": _row_value(row, fields, "status") or None,
                    "alias_symbols": alias_symbols,
                    "alias_names": _split_pipe_values(
                        _row_value(row, fields, "alias_name")
                    ),
                    "previous_symbols": previous_symbols,
                    "previous_names": _split_pipe_values(
                        _row_value(row, fields, "prev_name")
                    ),
                    "entrez_id": entrez_id,
                    "ensembl_gene_id": ensembl_gene_id,
                    "uniprot_ids": uniprot_ids,
                    "refseq_accessions": refseq_accessions,
                    "external_ids": external_ids,
                    "raw_record_hash": _canonical_hash(
                        {
                            _normalized_header(key): str(value or "")
                            for key, value in row.items()
                            if key is not None
                        }
                    ),
                }
            )
    return sorted(records, key=lambda record: str(record["record_id"]))


def _single_tag(
    tags: Mapping[str, list[str]],
    tag: str,
    *,
    required: bool,
    term_context: str,
) -> str | None:
    values = tags.get(tag, [])
    if required and len(values) != 1:
        raise SourceParseError(
            f"GO term {term_context} must contain exactly one {tag} tag"
        )
    if not required and len(values) > 1:
        raise SourceParseError(
            f"GO term {term_context} contains repeated scalar tag {tag}"
        )
    return values[0] if values else None


def _go_reference_id(value: str) -> str:
    return value.split("!", 1)[0].strip().split()[0]


def _go_relationship(value: str, term_id: str) -> dict[str, str]:
    semantic_value = value.split("!", 1)[0].strip()
    parts = semantic_value.split()
    if len(parts) < 2:
        raise SourceParseError(
            f"GO term {term_id} has malformed relationship: {value}"
        )
    return {"relation": parts[0], "target_id": parts[1], "raw_value": value}


def parse_go_obo(
    path: str | Path,
    *,
    release_identifier: str,
    source_id: str = "gene_ontology",
) -> list[MetadataRecord]:
    """Parse GO OBO terms while retaining every repeatable semantic tag."""

    _require_source_context(source_id, release_identifier)
    source_path = Path(path)
    records: list[MetadataRecord] = []
    seen: set[str] = set()
    stanza_type: str | None = None
    tags: defaultdict[str, list[str]] = defaultdict(list)
    stanza_lines: list[str] = []

    def flush_stanza() -> None:
        nonlocal tags, stanza_lines, stanza_type
        if stanza_type != "Term":
            tags = defaultdict(list)
            stanza_lines = []
            return
        term_context = tags.get("id", ["<missing-id>"])[0]
        term_id = _single_tag(
            tags, "id", required=True, term_context=term_context
        )
        name = _single_tag(tags, "name", required=True, term_context=term_context)
        assert term_id is not None and name is not None
        _assert_unique_record_id(term_id, seen, source_id=source_id)
        obsolete_text = _single_tag(
            tags, "is_obsolete", required=False, term_context=term_id
        )
        if obsolete_text not in {None, "true", "false"}:
            raise SourceParseError(
                f"GO term {term_id} has invalid is_obsolete value: {obsolete_text}"
            )

        is_a = list(tags.get("is_a", []))
        relationships = list(tags.get("relationship", []))
        replaced_by = [_go_reference_id(value) for value in tags.get("replaced_by", [])]
        consider = [_go_reference_id(value) for value in tags.get("consider", [])]
        records.append(
            {
                "source_id": source_id,
                "release_identifier": release_identifier,
                "record_type": "ontology_term_metadata",
                "record_id": term_id,
                "go_id": term_id,
                "name": name,
                "namespace": _single_tag(
                    tags, "namespace", required=False, term_context=term_id
                ),
                "definition": _single_tag(
                    tags, "def", required=False, term_context=term_id
                ),
                "alt_ids": list(tags.get("alt_id", [])),
                "synonyms": list(tags.get("synonym", [])),
                "xrefs": list(tags.get("xref", [])),
                "is_a": is_a,
                "is_a_ids": [_go_reference_id(value) for value in is_a],
                "relationships": relationships,
                "relationship_edges": [
                    _go_relationship(value, term_id) for value in relationships
                ],
                "is_obsolete": obsolete_text == "true",
                "replaced_by": replaced_by,
                "consider": consider,
                "comments": list(tags.get("comment", [])),
                "raw_tags": {
                    key: list(values) for key, values in sorted(tags.items())
                },
                "raw_record_hash": hashlib.sha256(
                    ("\n".join(stanza_lines) + "\n").encode("utf-8")
                ).hexdigest(),
            }
        )
        tags = defaultdict(list)
        stanza_lines = []

    with source_path.open("r", encoding="utf-8", errors="strict") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line.startswith("[") and line.endswith("]"):
                flush_stanza()
                stanza_type = line[1:-1]
                stanza_lines = [line]
                continue
            if stanza_type is None:
                continue
            stanza_lines.append(line)
            if not line or line.startswith("!") or ":" not in line:
                continue
            key, _, value = line.partition(":")
            tags[key.strip()].append(value.strip())
    flush_stanza()
    return sorted(records, key=lambda record: str(record["record_id"]))


def _reactome_lines(path: Path) -> Iterator[str]:
    if path.suffix.casefold() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = sorted(
                name for name in archive.namelist() if name.casefold().endswith(".gmt")
            )
            if len(members) != 1:
                raise SourceParseError(
                    "Reactome archive must contain exactly one GMT file"
                )
            with archive.open(members[0]) as raw_handle:
                with io.TextIOWrapper(raw_handle, encoding="utf-8", errors="strict") as handle:
                    yield from handle
        return
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        yield from handle


_REACTOME_STABLE_ID = re.compile(r"^R-[A-Z0-9]+-[0-9]+$")


def parse_reactome_gmt(
    path: str | Path,
    *,
    release_identifier: str,
    source_id: str = "reactome",
) -> list[MetadataRecord]:
    """Parse Reactome GMT rows as pathway name, stable ID, then entities."""

    _require_source_context(source_id, release_identifier)
    source_path = Path(path)
    records: list[MetadataRecord] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(_reactome_lines(source_path), start=1):
        line = raw_line.rstrip("\r\n")
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) < 2:
            raise SourceParseError(
                f"Reactome GMT row {line_number} has fewer than two fields"
            )
        pathway_name = fields[0].strip()
        stable_pathway_id = fields[1].strip()
        if _REACTOME_STABLE_ID.fullmatch(pathway_name) and not _REACTOME_STABLE_ID.fullmatch(
            stable_pathway_id
        ):
            raise SourceParseError(
                "Reactome GMT columns are reversed; expected pathway name then stable ID"
            )
        if not pathway_name or not _REACTOME_STABLE_ID.fullmatch(stable_pathway_id):
            raise SourceParseError(
                f"Reactome GMT row {line_number} has an invalid name or stable ID"
            )
        _assert_unique_record_id(stable_pathway_id, seen, source_id=source_id)
        records.append(
            {
                "source_id": source_id,
                "release_identifier": release_identifier,
                "record_type": "pathway_metadata",
                "record_id": stable_pathway_id,
                "stable_pathway_id": stable_pathway_id,
                "name": pathway_name,
                "entities": _deduplicate(fields[2:]),
                "raw_record_hash": hashlib.sha256(line.encode("utf-8")).hexdigest(),
            }
        )
    return sorted(records, key=lambda record: str(record["record_id"]))


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_text(element: ET.Element, local_name: str) -> str | None:
    for node in element.iter():
        if _local_name(node.tag) == local_name:
            text = "".join(node.itertext()).strip()
            if text:
                return text
    return None


def _journal_title(article: ET.Element) -> str | None:
    for node in article.iter():
        if _local_name(node.tag) != "Journal":
            continue
        return _first_text(node, "Title")
    return None


def _publication_year(article: ET.Element) -> str | None:
    for node in article.iter():
        if _local_name(node.tag) != "PubDate":
            continue
        year = _first_text(node, "Year")
        if year:
            return year
        medline_date = _first_text(node, "MedlineDate")
        match = re.search(r"\b(?:18|19|20|21)[0-9]{2}\b", medline_date or "")
        if match:
            return match.group(0)
    return None


def _article_ids(article: ET.Element) -> list[dict[str, str]]:
    identifiers: list[dict[str, str]] = []
    for node in article.iter():
        local_name = _local_name(node.tag)
        if local_name not in {"ArticleId", "ELocationID"}:
            continue
        value = "".join(node.itertext()).strip()
        id_type = str(node.attrib.get("IdType") or node.attrib.get("EIdType") or "").strip()
        if value and id_type:
            identifiers.append({"id_type": id_type.casefold(), "value": value})
    unique = {
        (identifier["id_type"], identifier["value"]): identifier
        for identifier in identifiers
    }
    return [unique[key] for key in sorted(unique)]


def _open_pubmed_xml(path: Path):
    with path.open("rb") as probe:
        gzip_encoded = probe.read(2) == b"\x1f\x8b"
    if gzip_encoded:
        return gzip.open(path, "rb")
    return path.open("rb")


def parse_pubmed_metadata_xml(
    path: str | Path,
    *,
    release_identifier: str,
    source_id: str = "pubmed",
) -> list[MetadataRecord]:
    """Parse PubMed bibliographic metadata without creating observations."""

    _require_source_context(source_id, release_identifier)
    source_path = Path(path)
    records: list[MetadataRecord] = []
    seen: set[str] = set()
    with _open_pubmed_xml(source_path) as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if _local_name(element.tag) not in {"PubmedArticle", "PubmedBookArticle"}:
                continue
            pmid = _first_text(element, "PMID")
            if not pmid:
                raise SourceParseError("PubMed article is missing PMID")
            _assert_unique_record_id(pmid, seen, source_id=source_id)
            identifiers = _article_ids(element)
            doi = next(
                (
                    identifier["value"]
                    for identifier in identifiers
                    if identifier["id_type"] == "doi"
                ),
                None,
            )
            languages = sorted(
                {
                    "".join(node.itertext()).strip()
                    for node in element.iter()
                    if _local_name(node.tag) == "Language"
                    and "".join(node.itertext()).strip()
                }
            )
            publication_types = sorted(
                {
                    "".join(node.itertext()).strip()
                    for node in element.iter()
                    if _local_name(node.tag) == "PublicationType"
                    and "".join(node.itertext()).strip()
                }
            )
            rights_sensitive_fields_present = any(
                _local_name(node.tag) in {"CopyrightInformation", "License"}
                for node in element.iter()
            )
            raw_bytes = ET.tostring(element, encoding="utf-8")
            records.append(
                {
                    "source_id": source_id,
                    "release_identifier": release_identifier,
                    "record_type": "publication_metadata",
                    "evidence_role": PUBMED_EVIDENCE_ROLE,
                    "record_id": pmid,
                    "publication_id": f"PMID:{pmid}",
                    "pmid": pmid,
                    "title": _first_text(element, "ArticleTitle"),
                    "journal_title": _journal_title(element),
                    "publication_year": _publication_year(element),
                    "doi": doi,
                    "article_ids": identifiers,
                    "languages": languages,
                    "publication_types": publication_types,
                    "article_rights_status": PUBMED_RIGHTS_STATUS,
                    "rights_sensitive_fields_present": rights_sensitive_fields_present,
                    "abstract_present": any(
                        _local_name(node.tag) == "AbstractText"
                        for node in element.iter()
                    ),
                    "observation_eligible": False,
                    "core_observation_eligible": False,
                    "counts_toward_observation_coverage": False,
                    "counts_toward_public_corpus_g2": False,
                    "raw_record_hash": hashlib.sha256(raw_bytes).hexdigest(),
                }
            )
            element.clear()
    return sorted(records, key=lambda record: str(record["record_id"]))
