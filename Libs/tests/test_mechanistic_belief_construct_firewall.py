"""Fail-closed tests for the mechanistic-belief construct firewall."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from typing import Any

from jsonschema import Draft202012Validator
import pytest
import yaml

from Libs.utils.data import mechanistic_belief_construct_firewall as firewall


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXED_TIME = "2026-08-31T04:00:00Z"
FIXED_CHECKOUT = {"branch": "synthetic-test", "head": "0" * 40}


def _copy_file(source_root: Path, target_root: Path, relative: str) -> None:
    destination = target_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / relative, destination)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    for relative in (
        firewall.CONTRACT_PATH,
        firewall.RECEIPT_SCHEMA_PATH,
        firewall.AUTHORITY_SCHEMA_PATH,
        *(path for path, _, _ in firewall.HISTORICAL_AUTHORITIES),
    ):
        _copy_file(REPO_ROOT, tmp_path, relative)
    return tmp_path


def _fake_body(publication: dict[str, Any]) -> bytes:
    kind = publication["projection_kind"]
    official_date = publication["official_date"]
    if kind == "pubmed_esummary_sortpubdate":
        uid = publication["identifier_value"]
        row = {"uid": uid, "sortpubdate": official_date.replace("-", "/") + " 00:00", "title": "must not persist"}
        return json.dumps({"result": {"uids": [uid], uid: row}}).encode()
    if kind == "elife_article_published":
        return json.dumps({"id": 69795, "published": official_date + "T00:00:00Z", "title": "must not persist"}).encode()
    year, month, day = (int(value) for value in official_date.split("-"))
    return json.dumps({"message": {"DOI": publication["identifier_value"], "published": {"date-parts": [[year, month, day]]}, "title": ["must not persist"]}}).encode()


def _fake_fetcher(calls: list[str] | None = None):
    def fetch(publication: dict[str, Any], policy: dict[str, Any]) -> firewall.FetchResult:
        assert policy["maximum_concurrent_requests"] == 1
        if calls is not None:
            calls.append(publication["publication_identity"])
        return firewall.FetchResult(
            requested_url=publication["requested_url"],
            final_url=publication["requested_url"],
            status_code=200,
            body=_fake_body(publication),
        )

    return fetch


def _build(repository: Path, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any], bool]:
    return firewall.build_construct_authority(
        repository_root=repository,
        fetcher=_fake_fetcher(),
        retrieved_at=FIXED_TIME,
        destination_checkout=FIXED_CHECKOUT,
        **kwargs,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(firewall.canonical_json_bytes(payload))


def _rehash(payload: dict[str, Any], field: str) -> None:
    payload[field] = firewall._self_hash(payload, field)


def _rebind_authority_to_receipt(
    repository: Path, receipt: dict[str, Any]
) -> None:
    receipt_path = repository / firewall.RECEIPT_PATH
    _rehash(receipt, "receipt_sha256")
    _write_json(receipt_path, receipt)
    receipt_raw = receipt_path.read_bytes()

    authority_path = repository / firewall.AUTHORITY_PATH
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["cutoff_receipt"]["file_sha256"] = firewall.sha256_bytes(receipt_raw)
    authority["cutoff_receipt"]["receipt_sha256"] = receipt["receipt_sha256"]
    _rehash(authority, "authority_sha256")
    _write_json(authority_path, authority)


def _rebind_authority_to_contract(repository: Path) -> None:
    contract_path = repository / firewall.CONTRACT_PATH
    contract_sha256 = firewall.sha256_file(contract_path)

    receipt_path = repository / firewall.RECEIPT_PATH
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["contract"]["sha256"] = contract_sha256
    _rebind_authority_to_receipt(repository, receipt)

    authority_path = repository / firewall.AUTHORITY_PATH
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["contract"]["sha256"] = contract_sha256
    _rehash(authority, "authority_sha256")
    _write_json(authority_path, authority)


def test_exact_six_build_is_schema_valid_metadata_only_and_idempotent(repository: Path) -> None:
    calls: list[str] = []
    receipt, authority, created = firewall.build_construct_authority(
        repository_root=repository,
        fetcher=_fake_fetcher(calls),
        retrieved_at=FIXED_TIME,
        destination_checkout=FIXED_CHECKOUT,
    )
    assert created is True
    assert calls == [identity for identity, *_ in firewall.PUBLICATIONS]
    assert len(receipt["records"]) == 6
    assert receipt["derivation"]["strictly_before_cutoff"] == firewall.CUTOFF
    assert receipt["access_counters"] == firewall.ACCESS_COUNTERS
    assert authority["method_uid"] == firewall.METHOD_UID
    assert authority["starting_authority"]["commit"] == firewall.STARTING_COMMIT
    assert authority["downstream_authorization"] == firewall.DOWNSTREAM_AUTHORIZATION
    serialized = (repository / firewall.RECEIPT_PATH).read_text(encoding="utf-8").lower()
    assert "must not persist" not in serialized
    assert all(not (set(record) & firewall.FORBIDDEN_FIELD_NAMES) for record in receipt["records"])

    second_calls: list[str] = []
    second_receipt, second_authority, second_created = firewall.build_construct_authority(
        repository_root=repository,
        fetcher=_fake_fetcher(second_calls),
        retrieved_at="2099-01-01T00:00:00Z",
        destination_checkout={"branch": "changed", "head": "1" * 40},
    )
    assert second_created is False
    assert second_calls == []
    assert second_receipt == receipt
    assert second_authority == authority


def test_schemas_are_strict_draft_2020_12(repository: Path) -> None:
    for relative in (firewall.RECEIPT_SCHEMA_PATH, firewall.AUTHORITY_SCHEMA_PATH):
        schema = json.loads((repository / relative).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("cutoff", "strictly_before_cutoff"), "2021-03-03T00:00:00Z"),
        (("publications", 0, "official_date"), "2026-06-22"),
        (("source_limits", "maximum_source_families"), 7),
        (("source_limits", "maximum_total_sources"), 161),
        (("source_limits", "maximum_sources_per_intervention"), 13),
        (("prior_roles", "maximum_sensitivity_roles"), 3),
        (("prior_roles", "required"), []),
        (("prior_roles", "optional_sensitivity"), ["conservative_support_sensitivity"]),
        (("prior_roles", "optional_sensitivity"), ["conservative_support_sensitivity", "broad_parameter_sensitivity", "added_role"]),
        (("prior_roles", "optional_sensitivity"), ["renamed_role"]),
        (("terminal_vocabulary",), firewall.TERMINALS[:-1]),
        (("terminal_vocabulary",), [*firewall.TERMINALS, "ADDED_TERMINAL"]),
        (("terminal_vocabulary", 0), "RENAMED_TERMINAL"),
        (("access_policy", "source_content_access"), 1),
    ],
)
def test_contract_cannot_self_widen(repository: Path, path: tuple[Any, ...], replacement: Any) -> None:
    contract_path = repository / firewall.CONTRACT_PATH
    payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    target: Any = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = replacement
    contract_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(firewall.ConstructFirewallError, match="contract identity drift"):
        firewall.validate_contract(repository)


def test_contract_rejects_missing_and_duplicate_publication_even_if_reformatted(repository: Path) -> None:
    contract_path = repository / firewall.CONTRACT_PATH
    payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    payload["publications"].pop()
    contract_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(firewall.ConstructFirewallError):
        firewall.validate_contract(repository)

    _copy_file(REPO_ROOT, repository, firewall.CONTRACT_PATH)
    payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    payload["publications"][1] = deepcopy(payload["publications"][0])
    contract_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(firewall.ConstructFirewallError):
        firewall.validate_contract(repository)


@pytest.mark.parametrize("field", ["title", "abstract", "result", "outcome", "ranking", "citation_context", "references", "raw_body"])
def test_receipt_rejects_forbidden_projection_fields(repository: Path, field: str) -> None:
    _build(repository)
    receipt_path = repository / firewall.RECEIPT_PATH
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["records"][0][field] = "forbidden"
    _rehash(receipt, "receipt_sha256")
    _write_json(receipt_path, receipt)
    with pytest.raises(firewall.ConstructFirewallError):
        firewall.validate_cutoff_receipt(repository_root=repository)


def test_receipt_rejects_cutoff_reorder_projection_and_access_drift(repository: Path) -> None:
    _build(repository)
    receipt_path = repository / firewall.RECEIPT_PATH
    original = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutations = []

    changed_cutoff = deepcopy(original)
    changed_cutoff["derivation"]["strictly_before_cutoff"] = "2021-03-03T00:00:00Z"
    mutations.append(changed_cutoff)
    reordered = deepcopy(original)
    reordered["records"][0], reordered["records"][1] = reordered["records"][1], reordered["records"][0]
    mutations.append(reordered)
    projected = deepcopy(original)
    projected["records"][0]["projected_metadata_sha256"] = "0" * 64
    mutations.append(projected)
    access = deepcopy(original)
    access["access_counters"]["six_context_outcome_access"] = 1
    mutations.append(access)

    for mutation in mutations:
        _rehash(mutation, "receipt_sha256")
        _write_json(receipt_path, mutation)
        with pytest.raises(firewall.ConstructFirewallError):
            firewall.validate_cutoff_receipt(repository_root=repository)
    _write_json(receipt_path, original)


@pytest.mark.parametrize(
    "mutation",
    [
        "add_title_to_projection_allowlist",
        "remove_required_projection_field",
        "rename_required_projection_field",
        "remove_forbidden_field",
        "add_forbidden_field",
        "rename_forbidden_field",
        "reorder_projection_allowlist",
        "reorder_forbidden_denylist",
        "change_access_kind",
        "authorize_content_from_metadata",
    ],
)
def test_rehashed_policy_drift_fails_both_validators_and_gate_before_open(
    repository: Path, mutation: str
) -> None:
    _build(repository)
    receipt_path = repository / firewall.RECEIPT_PATH
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    access_policy = receipt["access_policy"]
    if mutation == "add_title_to_projection_allowlist":
        access_policy["metadata_projection_fields"].append("title")
    elif mutation == "remove_required_projection_field":
        access_policy["metadata_projection_fields"].pop()
    elif mutation == "rename_required_projection_field":
        access_policy["metadata_projection_fields"][-1] = "renamed_validation_status"
    elif mutation == "remove_forbidden_field":
        access_policy["forbidden_persisted_fields"].pop()
    elif mutation == "add_forbidden_field":
        access_policy["forbidden_persisted_fields"].append("effect_size")
    elif mutation == "rename_forbidden_field":
        access_policy["forbidden_persisted_fields"][-1] = "renamed_raw_response"
    elif mutation == "reorder_projection_allowlist":
        access_policy["metadata_projection_fields"] = list(
            reversed(access_policy["metadata_projection_fields"])
        )
    elif mutation == "reorder_forbidden_denylist":
        access_policy["forbidden_persisted_fields"] = list(
            reversed(access_policy["forbidden_persisted_fields"])
        )
    elif mutation == "change_access_kind":
        access_policy["gate_access_kind"] = "official_metadata"
    else:
        access_policy["metadata_retrieval_authorizes_source_content"] = True
    _rebind_authority_to_receipt(repository, receipt)

    with pytest.raises(
        firewall.ConstructFirewallError, match="receipt metadata access policy drift"
    ):
        firewall.validate_cutoff_receipt(repository_root=repository)
    with pytest.raises(
        firewall.ConstructFirewallError, match="receipt metadata access policy drift"
    ):
        firewall.validate_construct_authority(repository_root=repository)

    input_root = repository / "Input" / "mechanism_sources"
    input_root.mkdir(parents=True)
    (input_root / "source.txt").write_bytes(b"synthetic")
    opener_calls: list[Path] = []
    with pytest.raises(
        firewall.ConstructFirewallError, match="receipt metadata access policy drift"
    ):
        firewall.open_mechanism_source_content(
            repository_root=repository,
            input_root=input_root,
            relative_path="source.txt",
            access_kind=firewall.GATE_ACCESS_KIND,
            opener=lambda path: opener_calls.append(path),
        )
    assert opener_calls == []


@pytest.mark.parametrize(
    "mutation",
    ["add_official_host", "remove_official_host", "reorder_official_hosts"],
)
def test_host_policy_is_semantically_pinned_after_full_rebinding(
    repository: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    _build(repository)
    contract_path = repository / firewall.CONTRACT_PATH
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    hosts = contract["retrieval_policy"]["allowed_host_suffixes"]
    if mutation == "add_official_host":
        hosts.append("example.com")
    elif mutation == "remove_official_host":
        hosts.pop()
    else:
        contract["retrieval_policy"]["allowed_host_suffixes"] = list(
            reversed(hosts)
        )
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        firewall,
        "EXPECTED_CONTRACT_SHA256",
        firewall.sha256_file(contract_path),
    )
    _rebind_authority_to_contract(repository)

    for validator in (
        firewall.validate_cutoff_receipt,
        firewall.validate_construct_authority,
    ):
        with pytest.raises(
            firewall.ConstructFirewallError,
            match="official metadata host or retrieval policy drift",
        ):
            validator(repository_root=repository)

    input_root = repository / "Input" / "mechanism_sources"
    input_root.mkdir(parents=True)
    (input_root / "source.txt").write_bytes(b"synthetic")
    opener_calls: list[Path] = []
    with pytest.raises(
        firewall.ConstructFirewallError,
        match="official metadata host or retrieval policy drift",
    ):
        firewall.open_mechanism_source_content(
            repository_root=repository,
            input_root=input_root,
            relative_path="source.txt",
            access_kind=firewall.GATE_ACCESS_KIND,
            opener=lambda path: opener_calls.append(path),
        )
    assert opener_calls == []


def test_authority_rejects_raw_hash_drift_and_historical_mutation(repository: Path) -> None:
    _build(repository)
    receipt_path = repository / firewall.RECEIPT_PATH
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["records"][0]["raw_response_sha256"] = "0" * 64
    _rehash(receipt, "receipt_sha256")
    _write_json(receipt_path, receipt)
    with pytest.raises(firewall.ConstructFirewallError, match="receipt binding drift|metadata binding drift"):
        firewall.validate_construct_authority(repository_root=repository)

    _build_fresh = repository / "fresh"
    _build_fresh.mkdir()
    for relative in (firewall.CONTRACT_PATH, firewall.RECEIPT_SCHEMA_PATH, firewall.AUTHORITY_SCHEMA_PATH, *(path for path, _, _ in firewall.HISTORICAL_AUTHORITIES)):
        _copy_file(REPO_ROOT, _build_fresh, relative)
    _build(_build_fresh)
    history_path = _build_fresh / firewall.HISTORICAL_AUTHORITIES[0][0]
    history_path.write_bytes(history_path.read_bytes() + b"\n")
    with pytest.raises(firewall.ConstructFirewallError, match="historical authority identity drift"):
        firewall.validate_construct_authority(repository_root=_build_fresh)


def test_date_drift_invalid_host_redirect_and_oversize_fail_before_outputs(repository: Path) -> None:
    def date_drift(publication: dict[str, Any], policy: dict[str, Any]) -> firewall.FetchResult:
        changed = dict(publication)
        changed["official_date"] = "2000-01-01"
        return firewall.FetchResult(publication["requested_url"], publication["requested_url"], 200, _fake_body(changed))

    with pytest.raises(firewall.ConstructFirewallError, match="date drift"):
        firewall.build_construct_authority(repository_root=repository, fetcher=date_drift, retrieved_at=FIXED_TIME, destination_checkout=FIXED_CHECKOUT)
    assert not (repository / firewall.RECEIPT_PATH).exists()
    assert not (repository / firewall.AUTHORITY_PATH).exists()

    def invalid_host(publication: dict[str, Any], policy: dict[str, Any]) -> firewall.FetchResult:
        return firewall.FetchResult(publication["requested_url"], "https://example.com/redirect", 200, _fake_body(publication), 1)

    with pytest.raises(firewall.ConstructFirewallError, match="host is not allowlisted"):
        firewall.build_construct_authority(repository_root=repository, fetcher=invalid_host, retrieved_at=FIXED_TIME, destination_checkout=FIXED_CHECKOUT)

    def excessive_redirects(publication: dict[str, Any], policy: dict[str, Any]) -> firewall.FetchResult:
        return firewall.FetchResult(publication["requested_url"], publication["requested_url"], 200, _fake_body(publication), int(policy["maximum_redirects"]) + 1)

    with pytest.raises(firewall.ConstructFirewallError, match="redirect limit"):
        firewall.build_construct_authority(repository_root=repository, fetcher=excessive_redirects, retrieved_at=FIXED_TIME, destination_checkout=FIXED_CHECKOUT)

    def oversized(publication: dict[str, Any], policy: dict[str, Any]) -> firewall.FetchResult:
        return firewall.FetchResult(publication["requested_url"], publication["requested_url"], 200, b"x" * (int(policy["maximum_response_bytes"]) + 1))

    with pytest.raises(firewall.ConstructFirewallError, match="response exceeds byte limit"):
        firewall.build_construct_authority(repository_root=repository, fetcher=oversized, retrieved_at=FIXED_TIME, destination_checkout=FIXED_CHECKOUT)


def test_transaction_rolls_back_and_drifted_existing_output_is_not_overwritten(repository: Path) -> None:
    calls = 0

    def fail_second(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic second install failure")
        os.replace(source, destination)

    with pytest.raises(firewall.ConstructFirewallError, match="transactional"):
        firewall.build_construct_authority(repository_root=repository, fetcher=_fake_fetcher(), retrieved_at=FIXED_TIME, destination_checkout=FIXED_CHECKOUT, replace=fail_second)
    assert not (repository / firewall.RECEIPT_PATH).exists()
    assert not (repository / firewall.AUTHORITY_PATH).exists()

    _build(repository)
    authority_path = repository / firewall.AUTHORITY_PATH
    authority_path.write_bytes(authority_path.read_bytes() + b" ")
    before = authority_path.read_bytes()
    with pytest.raises(firewall.ConstructFirewallError):
        _build(repository)
    assert authority_path.read_bytes() == before


def test_gate_validates_before_open_and_rejects_unsafe_paths(repository: Path) -> None:
    _build(repository)
    input_root = repository / "Input" / "mechanism_sources"
    input_root.mkdir(parents=True)
    content = input_root / "source.txt"
    content.write_bytes(b"synthetic mechanism bytes")
    calls: list[Path] = []

    def opener(path: Path) -> bytes:
        calls.append(path)
        return path.read_bytes()

    result = firewall.open_mechanism_source_content(repository_root=repository, input_root=input_root, relative_path="source.txt", access_kind=firewall.GATE_ACCESS_KIND, opener=opener)
    assert result == b"synthetic mechanism bytes"
    assert calls == [content]

    unsafe = ["../source.txt", "/absolute/source.txt", "protected/source.txt", "six_context/source.txt", "evaluation/source.txt"]
    for relative in unsafe:
        calls.clear()
        with pytest.raises(firewall.ConstructFirewallError):
            firewall.open_mechanism_source_content(repository_root=repository, input_root=input_root, relative_path=relative, access_kind=firewall.GATE_ACCESS_KIND, opener=opener)
        assert calls == []

    target = input_root / "target.txt"
    target.write_bytes(b"target")
    link = input_root / "link.txt"
    link.symlink_to(target)
    with pytest.raises(firewall.ConstructFirewallError, match="symlink"):
        firewall.open_mechanism_source_content(repository_root=repository, input_root=input_root, relative_path="link.txt", access_kind=firewall.GATE_ACCESS_KIND, opener=opener)


def test_gate_never_calls_opener_on_authority_or_access_kind_failure(repository: Path) -> None:
    _build(repository)
    input_root = repository / "Input" / "mechanism_sources"
    input_root.mkdir(parents=True)
    (input_root / "source.txt").write_bytes(b"synthetic")
    calls: list[Path] = []
    opener = lambda path: calls.append(path)  # noqa: E731

    with pytest.raises(firewall.ConstructFirewallError, match="access kind"):
        firewall.open_mechanism_source_content(repository_root=repository, input_root=input_root, relative_path="source.txt", access_kind="official_metadata", opener=opener)
    assert calls == []

    authority_path = repository / firewall.AUTHORITY_PATH
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["method_uid"] = "widened_method"
    _rehash(authority, "authority_sha256")
    _write_json(authority_path, authority)
    with pytest.raises(firewall.ConstructFirewallError):
        firewall.open_mechanism_source_content(repository_root=repository, input_root=input_root, relative_path="source.txt", access_kind=firewall.GATE_ACCESS_KIND, opener=opener)
    assert calls == []
