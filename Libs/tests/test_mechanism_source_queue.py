from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from Libs.utils.data import mechanism_source_queue as msq


REPO_ROOT = Path(__file__).resolve().parents[2]


def _response(url: str, payload: str, *, content_type: str = "application/json") -> msq.FetchResult:
    return msq.FetchResult(
        requested_url=url,
        final_url=url,
        status_code=200,
        content_type=content_type,
        body=payload.encode(),
    )


def _slot(identity: str, *, intervention: str = "action-a", ordinal: int = 1) -> dict:
    return {
        "canonical_intervention_uid": intervention,
        "validation_uid": "0123456789ab",
        "family_uid": "publication_metadata",
        "family_priority": 1,
        "family_slot_ordinal": ordinal,
        "mechanism_relevance_tags": list(msq.MECHANISM_TAGS),
        "canonical_source_identity": identity,
        "source_type": "publication",
        "official_date": "2020-01-01T00:00:00Z",
        "disposition": "eligible",
        "duplicate_of": None,
        "query_uids": ["publication-query-0123456789abcdef"],
        "request_uids": ["metadata-request-0123456789abcdef0123"],
        "metadata": {
            "identifier_type": "pmid",
            "identifier_value": "1",
            "pmid": "1",
            "doi": None,
            "public_access_status": "public_metadata",
            "license_status": "public_metadata_terms",
            "metadata_sha256": "0" * 64,
            "response_sha256": "1" * 64,
            "response_bytes": 10,
        },
    }


def _frozen_payloads() -> tuple[dict, dict, dict]:
    preregistration = json.loads((REPO_ROOT / msq.PREREGISTRATION_PATH).read_text())
    queue = json.loads((REPO_ROOT / msq.QUEUE_PATH).read_text())
    chronology = json.loads((REPO_ROOT / msq.CHRONOLOGY_PATH).read_text())
    return preregistration, queue, chronology


def _summary_payload(pmids: list[str]) -> str:
    records = {
        pmid: {
            "uid": pmid,
            "sortpubdate": f"2020/01/{index:02d} 00:00",
            "title": "DENIED AND UNPROJECTED",
            "articleids": [
                {"idtype": "pubmed", "value": pmid},
                {"idtype": "doi", "value": f"10.1000/{pmid}"},
            ],
        }
        for index, pmid in enumerate(pmids, start=1)
    }
    return json.dumps({"result": {"uids": pmids, **records}})


def _build_publication_branch(branch: str) -> tuple[dict, dict, dict]:
    preregistration, _, _ = _frozen_payloads()
    query = deepcopy(preregistration["expanded_requests"][0])
    projection = deepcopy(preregistration["context_action_projections"][0])
    family = deepcopy(preregistration["source_families"][0])
    pmids = ["101", "102", "103", "104"]

    def fetcher(request: dict, policy: dict) -> msq.FetchResult:
        del policy
        is_search = "esearch.fcgi" in request["url"]
        if branch == "search_failure" and is_search:
            raise msq.MechanismSourceQueueError("synthetic search failure")
        if is_search:
            if branch == "empty_search":
                return _response(request["url"], '{"esearchresult":{"idlist":[]}}')
            if branch == "malformed_search":
                return _response(request["url"], '{"esearchresult":{}}')
            selected = pmids[:2] if branch in {"summary_failure", "malformed_summary", "partial_result"} else pmids
            return _response(
                request["url"], json.dumps({"esearchresult": {"idlist": selected}})
            )
        if branch == "summary_failure":
            raise msq.MechanismSourceQueueError("synthetic summary failure")
        if branch == "malformed_summary":
            return _response(request["url"], '{"result":{}}')
        selected = pmids[:2] if branch == "partial_result" else pmids
        return _response(request["url"], _summary_payload(selected))

    receipts: list[dict] = []
    slots, overflow = msq._publication_slots(
        projection=projection,
        family=family,
        query=query,
        fetcher=fetcher,
        now=lambda: "2026-08-31T00:00:00.000001Z",
        sleeper=lambda seconds: None,
        receipts=receipts,
        total_bytes=[0],
    )
    assert overflow == 0
    return (
        {
            "context_action_projections": [projection],
            "source_families": [family],
            "expanded_requests": [query],
        },
        {"slots": msq._finalize_slots(slots)},
        {"requests": receipts},
    )


def _build_and_validate_publication_branch(branch: str) -> list[dict]:
    preregistration, queue, chronology = _build_publication_branch(branch)
    msq._validate_common_slot_bindings(preregistration, queue)
    msq._validate_exact_request_chronology(preregistration, queue, chronology)
    return queue["slots"]


def _rebind_mutated_slot_metadata(slot: dict) -> None:
    metadata = slot["metadata"]
    if metadata["identifier_type"] == "unavailable":
        metadata["metadata_sha256"] = None
        return
    projected = {
        "identifier_type": metadata["identifier_type"],
        "identifier_value": metadata["identifier_value"],
        "pmid": metadata["pmid"],
        "doi": metadata["doi"],
        "official_date": slot["official_date"],
        "canonical_source_identity": slot["canonical_source_identity"],
        "public_access_status": metadata["public_access_status"],
        "license_status": metadata["license_status"],
    }
    metadata["metadata_sha256"] = msq.canonical_value_sha256(projected)


def test_contract_has_independent_policy_pin_and_exact_rosters() -> None:
    contract, _ = msq.validate_contract(REPO_ROOT)
    assert msq._code_policy_sha256() == msq.EXPECTED_CODE_POLICY_SHA256
    assert [row["family_uid"] for row in contract["families"]] == [row[1] for row in msq.FAMILY_ROWS]
    assert [row["slot_quota_per_intervention"] for row in contract["families"]] == [4, 2, 2, 1, 1, 2]
    assert [(row["intervention_label"], row["modality"]) for row in contract["projection"]["interventions"]] == list(msq.INTERVENTION_ROWS)
    assert contract["cutoff"]["official_date_at_or_before"] == "2021-03-02T23:59:59Z"


def test_all_four_schemas_are_strict_draft_2020_12() -> None:
    for relative in msq.EXPECTED_SCHEMA_SHA256:
        schema, _ = msq._schema(REPO_ROOT, relative)
        Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False


def test_upstream_authorities_and_access_counters_still_validate() -> None:
    contract, _ = msq.validate_contract(REPO_ROOT)
    records = msq.validate_upstream_authorities(REPO_ROOT, contract)
    assert len(records) == 8


def test_six_context_projector_reads_only_allowlisted_semantics() -> None:
    contract, _ = msq.validate_contract(REPO_ROOT)
    projections, sources = msq.project_context_actions(REPO_ROOT, contract)
    assert len(projections) == 6
    assert len(sources) == 9
    serialized = msq.canonical_json_bytes({"rows": projections}).decode()
    for forbidden in msq.FORBIDDEN_FIELD_NAMES:
        assert f'"{forbidden}"' not in serialized
    assert all(row["access_counters"] == msq.ZERO_COUNTERS for row in projections)
    by_label = {row["intervention_label"]: row for row in projections}
    assert by_label["VERUBECESTAT"]["route"] == "oral"
    assert by_label["LECANEMAB"]["route"] == "intravenous"
    assert all(row["dose_unit"] == "relative_reference_dose" for row in projections)


def test_exact_queries_are_deterministic_and_preregistered() -> None:
    contract, _ = msq.validate_contract(REPO_ROOT)
    projections, _ = msq.project_context_actions(REPO_ROOT, contract)
    first = msq._expanded_requests(contract, projections)
    second = msq._expanded_requests(contract, list(reversed(projections)))
    assert len(first) == 6
    assert {row["query_uid"] for row in first} == {row["query_uid"] for row in second}
    assert all("retmax=4" in row["search_url"] and "sort=pub_date" in row["search_url"] for row in first)
    assert all("2021%2F03%2F02" in row["search_url"] for row in first)


@pytest.mark.parametrize(
    ("url", "allowed"),
    [
        ("https://eutils.ncbi.nlm.nih.gov/a", True),
        ("http://eutils.ncbi.nlm.nih.gov/a", False),
        ("https://ncbi.nlm.nih.gov.evil.example/a", False),
        ("https://example.org/a", False),
    ],
)
def test_host_policy_fails_closed(url: str, allowed: bool) -> None:
    assert msq._host_allowed(url, ["ncbi.nlm.nih.gov"]) is allowed


def test_search_identifier_order_does_not_control_queue_order() -> None:
    left, overflow_left = msq._parse_search_ids(
        b'{"esearchresult":{"idlist":["20","3","11","2"]}}', 4
    )
    right, overflow_right = msq._parse_search_ids(
        b'{"esearchresult":{"idlist":["2","11","3","20"]}}', 4
    )
    assert left == right == ["2", "3", "11", "20"]
    assert overflow_left == overflow_right == 0


@pytest.mark.parametrize(
    "payload",
    [
        b'{"esearchresult":{"idlist":["1","1"]}}',
        b'{"esearchresult":{"idlist":["not-a-pmid"]}}',
        b'{"esearchresult":{}}',
        b'not-json',
    ],
)
def test_malformed_search_metadata_is_rejected(payload: bytes) -> None:
    with pytest.raises(msq.MechanismSourceQueueError):
        msq._parse_search_ids(payload, 4)


def test_summary_projection_normalizes_pmid_doi_and_ignores_title() -> None:
    payload = (
        '{"result":{"uids":["12"],"12":{"uid":"12",'
        '"sortpubdate":"2020/02/03 00:00","title":"DENIED VALUE",'
        '"articleids":[{"idtype":"pubmed","value":"12"},'
        '{"idtype":"doi","value":" 10.1234/ABC "}]}}}'
    ).encode()
    rows = msq._parse_summary_records(payload, ["12"], "a" * 64, len(payload))
    assert rows[0]["canonical_source_identity"] == "doi:10.1234/abc"
    assert rows[0]["pmid"] == "12"
    assert "DENIED VALUE" not in repr(rows)


@pytest.mark.parametrize("date_value", ["2021", "2021/03", "Spring 2021", ""])
def test_ambiguous_publication_dates_are_not_promoted(date_value: str) -> None:
    assert msq._parse_pubmed_date(date_value) is None


def test_duplicate_lineage_retains_later_slot_without_refill() -> None:
    first = _slot("doi:10.1/example", ordinal=1)
    second = _slot("doi:10.1/example", ordinal=2)
    finalized = msq._finalize_slots([second, first])
    assert len(finalized) == 2
    assert finalized[0]["disposition"] == "eligible"
    assert finalized[1]["disposition"] == "duplicate"
    assert finalized[1]["duplicate_of"] == finalized[0]["slot_uid"]


def test_structured_database_identities_remain_family_scoped() -> None:
    left = _slot("unavailable:chembl_compound_target_metadata:action-a:01")
    left["family_uid"] = "chembl_compound_target_metadata"
    left["family_priority"] = 3
    left["disposition"] = "source_date_ambiguous"
    right = deepcopy(left)
    right["family_uid"] = "dgidb_drug_gene_metadata"
    right["family_priority"] = 6
    right["canonical_source_identity"] = "unavailable:dgidb_drug_gene_metadata:action-a:01"
    finalized = msq._finalize_slots([right, left])
    assert all(row["disposition"] == "source_date_ambiguous" for row in finalized)


def test_forbidden_projection_fields_fail_before_value_logging() -> None:
    with pytest.raises(msq.MechanismSourceQueueError, match="forbidden metadata field"):
        msq._validate_no_forbidden_fields({"safe": {"expected_effect_direction": "DENIED"}})


def test_unexpected_content_type_becomes_one_failed_request_receipt() -> None:
    receipts: list[dict] = []
    total = [0]
    response, request_uid = msq._perform_request(
        query_uid="publication-query-0123456789abcdef",
        request_kind="pubmed_search",
        url="https://eutils.ncbi.nlm.nih.gov/test",
        headers={"Accept": "application/json"},
        policy=msq.REQUEST_POLICY,
        fetcher=lambda request, policy: _response(request["url"], "{}", content_type="text/html"),
        now=lambda: "2026-08-31T00:00:00.000001Z",
        receipts=receipts,
        total_bytes=total,
    )
    assert response is None
    assert receipts == [
        {
            "request_uid": request_uid,
            "query_uid": "publication-query-0123456789abcdef",
            "request_kind": "pubmed_search",
            "method": "GET",
            "url": "https://eutils.ncbi.nlm.nih.gov/test",
            "host": "eutils.ncbi.nlm.nih.gov",
            "started_at": "2026-08-31T00:00:00.000001Z",
            "completed_at": "2026-08-31T00:00:00.000001Z",
            "http_status": None,
            "content_type": None,
            "response_bytes": 0,
            "response_sha256": None,
            "status": "request_failed",
        }
    ]


def test_oversized_response_becomes_request_failure() -> None:
    receipts: list[dict] = []
    total = [0]
    policy = {**msq.REQUEST_POLICY, "maximum_response_bytes": 2}
    response, _ = msq._perform_request(
        query_uid="publication-query-0123456789abcdef",
        request_kind="pubmed_search",
        url="https://eutils.ncbi.nlm.nih.gov/test",
        headers={},
        policy=policy,
        fetcher=lambda request, policy: _response(request["url"], "{}{}"),
        now=lambda: "2026-08-31T00:00:00.000001Z",
        receipts=receipts,
        total_bytes=total,
    )
    assert response is None
    assert receipts[0]["status"] == "request_failed"


def test_request_budget_breach_is_rejected_before_fetch() -> None:
    receipts = [{} for _ in range(48)]
    called = False

    def fetcher(request: dict, policy: dict) -> msq.FetchResult:
        nonlocal called
        called = True
        return _response(request["url"], "{}")

    with pytest.raises(msq.MechanismSourceQueueError, match="request budget"):
        msq._perform_request(
            query_uid="publication-query-0123456789abcdef",
            request_kind="pubmed_search",
            url="https://eutils.ncbi.nlm.nih.gov/test",
            headers={},
            policy=msq.REQUEST_POLICY,
            fetcher=fetcher,
            now=lambda: "2026-08-31T00:00:00.000001Z",
            receipts=receipts,
            total_bytes=[0],
        )
    assert called is False


def test_placeholder_retains_cutoff_and_unavailable_slots() -> None:
    projection = {"intervention_program_uid": "action-a", "validation_uid": "0123456789ab"}
    contract, _ = msq.validate_contract(REPO_ROOT)
    for family in contract["families"][1:]:
        slot = msq._placeholder_slot(
            projection=projection,
            family=family,
            ordinal=1,
            disposition=family["preregistered_disposition"],
            query_uid=None,
            request_uids=[],
        )
        assert slot["disposition"] in {"cutoff_ineligible_release", "source_date_ambiguous"}
        assert slot["request_uids"] == []


def test_transactional_install_rolls_back_first_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    calls = 0
    real_replace = msq.os.replace

    def replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic failure")
        real_replace(source, destination)

    monkeypatch.setattr(msq.os, "replace", replace)
    with pytest.raises(msq.MechanismSourceQueueError, match="transactional"):
        msq._install_many([(first, b"one"), (second, b"two")])
    assert not first.exists()
    assert not second.exists()


def test_identical_single_artifact_rerun_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    assert msq._install_one(path, b"same") is True
    assert msq._install_one(path, b"same") is False
    with pytest.raises(msq.MechanismSourceQueueError, match="overwrite"):
        msq._install_one(path, b"different")


def test_no_source_content_opener_or_raw_response_persistence_api_exists() -> None:
    source = (REPO_ROOT / "Libs/utils/data/mechanism_source_queue.py").read_text(encoding="utf-8")
    assert "open_mechanism_source_content(" not in source
    assert '"raw_body"' not in source
    assert "write(response.body" not in source


def test_authorization_schema_rejects_open_content_and_downstream_flags() -> None:
    schema, _ = msq._schema(REPO_ROOT, msq.AUTHORIZATION_SCHEMA_PATH)
    minimal = {
        "schema_version": msq.AUTHORIZATION_SCHEMA_VERSION,
        "status": "MECHANISM_SOURCE_CONTENT_NOT_AUTHORIZED",
        "authorization_uid": "mechanism_source_content_authorization",
        "preregistration": {"path": "a", "file_sha256": "0" * 64, "semantic_sha256": "1" * 64},
        "queue": {"path": "b", "file_sha256": "2" * 64, "semantic_sha256": "3" * 64},
        "chronology": {"path": "c", "file_sha256": "4" * 64, "semantic_sha256": "5" * 64},
        "schema": {"path": msq.AUTHORIZATION_SCHEMA_PATH, "sha256": msq.EXPECTED_SCHEMA_SHA256[msq.AUTHORIZATION_SCHEMA_PATH]},
        "content_access_authorized": False,
        "access_counters": deepcopy(msq.ZERO_COUNTERS),
        "downstream_authorization": deepcopy(msq.DOWNSTREAM_AUTHORIZATION),
        "authorization_sha256": "6" * 64,
        "blockers": [],
    }
    assert not list(Draft202012Validator(schema).iter_errors(minimal))
    widened = deepcopy(minimal)
    widened["content_access_authorized"] = True
    assert list(Draft202012Validator(schema).iter_errors(widened))
    widened = deepcopy(minimal)
    widened["downstream_authorization"]["training_authorized"] = True
    assert list(Draft202012Validator(schema).iter_errors(widened))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("slot_count", 0),
        ("intervention_count", 5),
        ("family_count", 5),
    ],
)
def test_rehashed_declared_count_mutations_fail(field: str, value: int) -> None:
    preregistration, queue, _ = _frozen_payloads()
    queue["counts"][field] = value
    queue["queue_sha256"] = msq._self_hash(queue, "queue_sha256")
    with pytest.raises(msq.MechanismSourceQueueError, match="declared queue counts"):
        msq._validate_queue_derived_counts(queue, preregistration)


def test_rehashed_disposition_total_mutation_fails() -> None:
    preregistration, queue, _ = _frozen_payloads()
    queue["counts"]["dispositions"]["eligible"] -= 1
    queue["counts"]["dispositions"]["unavailable"] += 1
    queue["queue_sha256"] = msq._self_hash(queue, "queue_sha256")
    with pytest.raises(msq.MechanismSourceQueueError, match="declared queue counts"):
        msq._validate_queue_derived_counts(queue, preregistration)


def test_actual_per_intervention_cap_mutation_fails_even_with_rehashed_counts() -> None:
    preregistration, queue, _ = _frozen_payloads()
    left = queue["slots"][0]["canonical_intervention_uid"]
    replacement = next(
        slot["canonical_intervention_uid"]
        for slot in queue["slots"]
        if slot["canonical_intervention_uid"] != left
    )
    queue["slots"][0]["canonical_intervention_uid"] = replacement
    queue["counts"]["slot_count"] = len(queue["slots"])
    queue["queue_sha256"] = msq._self_hash(queue, "queue_sha256")
    with pytest.raises(msq.MechanismSourceQueueError, match="per-intervention"):
        msq._validate_queue_derived_counts(queue, preregistration)


def test_current_request_chronology_is_exactly_derived() -> None:
    preregistration, queue, chronology = _frozen_payloads()
    msq._validate_exact_request_chronology(preregistration, queue, chronology)


def test_rehashed_esearch_to_esummary_substitution_fails() -> None:
    preregistration, queue, chronology = _frozen_payloads()
    request = chronology["requests"][0]
    request["request_kind"] = "pubmed_summary"
    request["url"] = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&version=2.0&id=999"
    request["request_uid"] = msq._request_uid(
        request["query_uid"], request["request_kind"], request["url"]
    )
    chronology["chronology_sha256"] = msq._self_hash(chronology, "chronology_sha256")
    with pytest.raises(msq.MechanismSourceQueueError, match="request identity"):
        msq._validate_exact_request_chronology(preregistration, queue, chronology)


@pytest.mark.parametrize("mutation", ["uid", "url", "query_uid", "order"])
def test_rehashed_request_identity_and_sequence_mutations_fail(mutation: str) -> None:
    preregistration, queue, chronology = _frozen_payloads()
    if mutation == "uid":
        chronology["requests"][0]["request_uid"] = "metadata-request-" + "f" * 20
    elif mutation == "url":
        chronology["requests"][1]["url"] = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            "?db=pubmed&retmode=json&version=2.0&id=999"
        )
        chronology["requests"][1]["request_uid"] = msq._request_uid(
            chronology["requests"][1]["query_uid"],
            chronology["requests"][1]["request_kind"],
            chronology["requests"][1]["url"],
        )
    elif mutation == "query_uid":
        chronology["requests"][0]["query_uid"] = "publication-query-ffffffffffffffff"
        chronology["requests"][0]["request_uid"] = msq._request_uid(
            chronology["requests"][0]["query_uid"],
            chronology["requests"][0]["request_kind"],
            chronology["requests"][0]["url"],
        )
    else:
        chronology["requests"][0], chronology["requests"][1] = (
            chronology["requests"][1],
            chronology["requests"][0],
        )
    chronology["chronology_sha256"] = msq._self_hash(chronology, "chronology_sha256")
    with pytest.raises(msq.MechanismSourceQueueError, match="request identity"):
        msq._validate_exact_request_chronology(preregistration, queue, chronology)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_rehashed_request_cardinality_mutations_fail(mutation: str) -> None:
    preregistration, queue, chronology = _frozen_payloads()
    if mutation == "missing":
        chronology["requests"].pop()
    elif mutation == "extra":
        extra = deepcopy(chronology["requests"][-1])
        extra["request_uid"] = "metadata-request-" + "e" * 20
        chronology["requests"].append(extra)
    else:
        chronology["requests"].append(deepcopy(chronology["requests"][-1]))
    chronology["chronology_sha256"] = msq._self_hash(chronology, "chronology_sha256")
    with pytest.raises(msq.MechanismSourceQueueError):
        msq._validate_exact_request_chronology(preregistration, queue, chronology)


@pytest.mark.parametrize(
    ("branch", "expected_dispositions"),
    [
        ("search_failure", ["request_failed"] * 4),
        ("empty_search", ["unavailable"] * 4),
        ("malformed_search", ["forbidden_metadata"] * 4),
        ("summary_failure", ["request_failed", "request_failed", "unavailable", "unavailable"]),
        ("malformed_summary", ["forbidden_metadata", "forbidden_metadata", "unavailable", "unavailable"]),
        ("partial_result", ["eligible", "eligible", "unavailable", "unavailable"]),
        ("successful_summary", ["eligible"] * 4),
    ],
)
def test_production_publication_branches_validate_end_to_end(
    branch: str, expected_dispositions: list[str]
) -> None:
    slots = _build_and_validate_publication_branch(branch)
    assert [slot["disposition"] for slot in slots] == expected_dispositions


@pytest.mark.parametrize(
    ("branch", "mutation"),
    [
        ("search_failure", "doi"),
        ("empty_search", "date"),
        ("malformed_search", "access_license"),
        ("partial_result", "placeholder_identity"),
        ("partial_result", "response_metadata"),
        ("summary_failure", "doi"),
        ("summary_failure", "date"),
        ("summary_failure", "access_license"),
        ("summary_failure", "response_metadata"),
        ("malformed_summary", "doi"),
        ("malformed_summary", "date"),
        ("malformed_summary", "access_license"),
        ("malformed_summary", "response_metadata"),
    ],
)
def test_fully_rehashed_placeholder_metadata_mutations_fail(
    branch: str, mutation: str
) -> None:
    preregistration, queue, chronology = _build_publication_branch(branch)
    if branch in {"summary_failure", "malformed_summary"}:
        slot = next(row for row in queue["slots"] if row["metadata"]["pmid"] is not None)
    else:
        slot = next(row for row in reversed(queue["slots"]) if row["metadata"]["pmid"] is None)
    metadata = slot["metadata"]
    if mutation == "doi":
        metadata["doi"] = "10.9999/fabricated"
        metadata["identifier_type"] = "doi"
        metadata["identifier_value"] = metadata["doi"]
        slot["canonical_source_identity"] = f"doi:{metadata['doi']}"
    elif mutation == "date":
        slot["official_date"] = "2020-01-01T00:00:00Z"
    elif mutation == "access_license":
        metadata["public_access_status"] = "public_metadata"
        metadata["license_status"] = "public_metadata_terms"
    elif mutation == "response_metadata":
        metadata["response_sha256"] = "f" * 64
        metadata["response_bytes"] = 999
    else:
        slot["canonical_source_identity"] = "unavailable:publication_metadata:fabricated:99"
    if mutation in {"pmid", "doi"}:
        _rebind_mutated_slot_metadata(slot)
    queue["queue_sha256"] = msq._self_hash(queue, "queue_sha256")
    chronology["chronology_sha256"] = msq._self_hash(
        chronology, "chronology_sha256"
    )
    with pytest.raises(msq.MechanismSourceQueueError):
        msq._validate_exact_request_chronology(preregistration, queue, chronology)


@pytest.mark.parametrize(
    "family_uid",
    [
        "open_targets_target_metadata",
        "chembl_compound_target_metadata",
        "bindingdb_record_metadata",
        "drugcentral_drug_moa_metadata",
        "dgidb_drug_gene_metadata",
    ],
)
@pytest.mark.parametrize(
    "mutation",
    [
        "family_priority",
        "source_type",
        "validation_uid",
        "mechanism_tags",
        "ordinal",
        "source_identity",
        "query_lineage",
        "request_lineage",
        "pmid",
        "doi",
        "date",
        "access",
        "license",
        "metadata_hash",
        "response_hash",
        "response_bytes",
        "duplicate_lineage",
    ],
)
def test_fully_rehashed_disabled_family_placeholder_mutations_fail(
    family_uid: str, mutation: str
) -> None:
    preregistration, queue, _ = _frozen_payloads()
    slot = next(row for row in queue["slots"] if row["family_uid"] == family_uid)
    metadata = slot["metadata"]
    if mutation == "family_priority":
        slot["family_priority"] = 99
    elif mutation == "source_type":
        slot["source_type"] = "fabricated_source_type"
    elif mutation == "validation_uid":
        slot["validation_uid"] = "f" * 12
    elif mutation == "mechanism_tags":
        slot["mechanism_relevance_tags"] = list(reversed(msq.MECHANISM_TAGS))
    elif mutation == "ordinal":
        slot["family_slot_ordinal"] = 99
    elif mutation == "source_identity":
        slot["canonical_source_identity"] = f"fabricated:{family_uid}:identity"
    elif mutation == "query_lineage":
        slot["query_uids"] = ["publication-query-ffffffffffffffff"]
    elif mutation == "request_lineage":
        slot["request_uids"] = ["metadata-request-" + "f" * 20]
    elif mutation == "pmid":
        metadata["identifier_type"] = "pmid"
        metadata["identifier_value"] = "999"
        metadata["pmid"] = "999"
        slot["canonical_source_identity"] = "pmid:999"
    elif mutation == "doi":
        metadata["identifier_type"] = "doi"
        metadata["identifier_value"] = "10.9999/fabricated"
        metadata["doi"] = "10.9999/fabricated"
        slot["canonical_source_identity"] = "doi:10.9999/fabricated"
    elif mutation == "date":
        slot["official_date"] = "2020-01-01T00:00:00Z"
    elif mutation == "access":
        metadata["public_access_status"] = "public_metadata"
    elif mutation == "license":
        metadata["license_status"] = "public_metadata_terms"
    elif mutation == "metadata_hash":
        metadata["metadata_sha256"] = "a" * 64
    elif mutation == "response_hash":
        metadata["response_sha256"] = "b" * 64
    elif mutation == "response_bytes":
        metadata["response_bytes"] = 123
    else:
        slot["duplicate_of"] = "mechanism-source-slot-" + "c" * 20
    if mutation in {"pmid", "doi"}:
        _rebind_mutated_slot_metadata(slot)
    queue["slots"] = msq._finalize_slots(queue["slots"])
    queue["queue_sha256"] = msq._self_hash(queue, "queue_sha256")
    with pytest.raises(
        msq.MechanismSourceQueueError,
        match="disabled-family placeholder|family slot ordinal",
    ):
        msq._validate_disabled_family_placeholders(preregistration, queue)


def test_current_common_slot_bindings_and_publication_ordinals_pass() -> None:
    preregistration, queue, chronology = _frozen_payloads()
    msq._validate_common_slot_bindings(preregistration, queue)
    msq._validate_exact_request_chronology(preregistration, queue, chronology)


@pytest.mark.parametrize(
    "mutation",
    [
        "validation_uid",
        "family_priority",
        "source_type",
        "mechanism_tags",
        "query_lineage",
        "request_lineage",
        "summary_request_lineage",
    ],
)
def test_fully_rehashed_publication_common_field_mutations_fail(
    mutation: str,
) -> None:
    preregistration, queue, _ = _frozen_payloads()
    slot = next(row for row in queue["slots"] if row["family_uid"] == "publication_metadata")
    if mutation == "validation_uid":
        slot["validation_uid"] = "f" * 12
    elif mutation == "family_priority":
        slot["family_priority"] = 6
    elif mutation == "source_type":
        slot["source_type"] = "fabricated_source_type"
    elif mutation == "mechanism_tags":
        slot["mechanism_relevance_tags"] = list(reversed(msq.MECHANISM_TAGS))
    elif mutation == "query_lineage":
        slot["query_uids"] = ["publication-query-ffffffffffffffff"]
    elif mutation == "request_lineage":
        slot["request_uids"] = ["metadata-request-" + "f" * 20]
    else:
        slot["request_uids"][-1] = "metadata-request-" + "f" * 20
    queue["slots"] = msq._finalize_slots(queue["slots"])
    queue["queue_sha256"] = msq._self_hash(queue, "queue_sha256")
    with pytest.raises(msq.MechanismSourceQueueError, match="slot .* binding drift"):
        msq._validate_common_slot_bindings(preregistration, queue)


def test_fully_rehashed_publication_source_ordinal_swap_fails() -> None:
    preregistration, queue, _ = _frozen_payloads()
    intervention_uid = next(
        slot["canonical_intervention_uid"]
        for slot in queue["slots"]
        if slot["family_uid"] == "publication_metadata"
    )
    slots = [
        slot
        for slot in queue["slots"]
        if slot["family_uid"] == "publication_metadata"
        and slot["canonical_intervention_uid"] == intervention_uid
    ]
    slots[0]["family_slot_ordinal"], slots[1]["family_slot_ordinal"] = (
        slots[1]["family_slot_ordinal"],
        slots[0]["family_slot_ordinal"],
    )
    queue["slots"] = msq._finalize_slots(queue["slots"])
    queue["queue_sha256"] = msq._self_hash(queue, "queue_sha256")
    with pytest.raises(msq.MechanismSourceQueueError, match="source-to-ordinal"):
        msq._validate_common_slot_bindings(preregistration, queue)


def test_fully_rehashed_publication_queue_order_swap_fails() -> None:
    preregistration, queue, _ = _frozen_payloads()
    publication_indexes = [
        index
        for index, slot in enumerate(queue["slots"])
        if slot["family_uid"] == "publication_metadata"
    ]
    left, right = publication_indexes[:2]
    queue["slots"][left], queue["slots"][right] = (
        queue["slots"][right],
        queue["slots"][left],
    )
    for order, slot in enumerate(queue["slots"], start=1):
        slot["queue_order"] = order
    queue["queue_sha256"] = msq._self_hash(queue, "queue_sha256")
    with pytest.raises(msq.MechanismSourceQueueError, match="queue ordering"):
        msq._validate_common_slot_bindings(preregistration, queue)


def test_fully_rehashed_publication_to_disabled_family_substitution_fails() -> None:
    preregistration, queue, chronology = _frozen_payloads()
    publication = next(
        slot for slot in queue["slots"] if slot["family_uid"] == "publication_metadata"
    )
    disabled = next(
        slot
        for slot in queue["slots"]
        if slot["canonical_intervention_uid"] == publication["canonical_intervention_uid"]
        and slot["family_uid"] == "open_targets_target_metadata"
    )
    for field in ("family_uid", "family_priority", "source_type", "family_slot_ordinal"):
        publication[field], disabled[field] = disabled[field], publication[field]
    publication["query_uids"] = []
    publication["request_uids"] = []
    disabled["query_uids"] = [
        next(
            query["query_uid"]
            for query in preregistration["expanded_requests"]
            if query["canonical_intervention_uid"] == disabled["canonical_intervention_uid"]
        )
    ]
    disabled["request_uids"] = deepcopy(
        next(
            slot["request_uids"]
            for slot in queue["slots"]
            if slot["family_uid"] == "publication_metadata"
            and slot["canonical_intervention_uid"] == disabled["canonical_intervention_uid"]
        )
    )
    queue["slots"] = msq._finalize_slots(queue["slots"])
    queue["queue_sha256"] = msq._self_hash(queue, "queue_sha256")
    chronology["chronology_sha256"] = msq._self_hash(
        chronology, "chronology_sha256"
    )
    with pytest.raises(msq.MechanismSourceQueueError):
        msq._validate_queue_derived_counts(queue, preregistration)
        msq._validate_common_slot_bindings(preregistration, queue)
        msq._validate_disabled_family_placeholders(preregistration, queue)
        msq._validate_exact_request_chronology(preregistration, queue, chronology)
