"""Fail-closed tests for Allen official metadata rights/release resolution."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import yaml

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.round20_track_contracts import ALLEN_FAMILY_UIDS
from Libs.utils.data import round20_allen_official_metadata_resolution as resolution


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / resolution.CONFIG_RELATIVE
SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/round20_allen_official_metadata_resolution.schema.json"


def _contract() -> dict[str, object]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[resolution.CONFIG_KEY]


def _fake_fetcher(
    source_uid: str,
    url: str,
    *,
    allowed_suffixes: list[str],
    timeout_seconds: int,
    max_bytes: int,
) -> resolution.FetchResult:
    assert resolution.is_official_url(url, allowed_suffixes)
    body = f"{source_uid}\n{url}\n".encode("utf-8")
    return resolution.FetchResult(
        source_uid=source_uid,
        url=url,
        final_url=url,
        status_code=200,
        content_type="text/html",
        headers={"etag": f'"{source_uid}"'},
        body=body[:max_bytes],
    )


def test_official_metadata_contract_is_canonical_and_in_config_stack() -> None:
    contract = _contract()
    assert resolution.validate_contract_definition(contract) == []
    assert load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")[resolution.CONFIG_KEY] == contract
    assert {row["family_uid"] for row in contract["family_metadata"]} == ALLEN_FAMILY_UIDS
    assert len(contract["source_snapshots"]) == 10
    assert contract["authorization"]["payload_download"] is False
    assert contract["source_policy"]["general_terms_satisfy_dataset_specific_rights"] is False


def test_official_metadata_contract_rejects_policy_and_identity_mutations() -> None:
    non_official = deepcopy(_contract())
    non_official["source_snapshots"][0]["url"] = "https://example.com/terms"
    assert "non_official_metadata_url" in {item["code"] for item in resolution.validate_contract_definition(non_official)}

    authorized_payload = deepcopy(_contract())
    authorized_payload["authorization"]["payload_download"] = True
    assert "unauthorized_execution" in {item["code"] for item in resolution.validate_contract_definition(authorized_payload)}

    assumed_release = deepcopy(_contract())
    assumed_release["family_metadata"][0]["release_identity_evidence"] = "assumed"
    assert "assumed_release_identity" in {item["code"] for item in resolution.validate_contract_definition(assumed_release)}

    missing_family = deepcopy(_contract())
    missing_family["family_metadata"].pop()
    assert "family_inventory_drift" in {item["code"] for item in resolution.validate_contract_definition(missing_family)}


def test_official_metadata_builds_blocked_artifacts_without_payload_retrieval(tmp_path: Path) -> None:
    contract = deepcopy(_contract())
    audit = resolution.emit_artifacts(contract, repository=tmp_path, fetcher=_fake_fetcher)
    blockers = resolution.validate_emitted_artifacts(contract, repository=tmp_path)
    assert blockers == []

    output = contract["output_artifacts"]
    rights = json.loads((tmp_path / output["rights_matrix"]).read_text(encoding="utf-8"))
    retrieval = json.loads((tmp_path / output["retrieval_manifest"]).read_text(encoding="utf-8"))
    release_rows = resolution.read_release_inventory(tmp_path / output["release_inventory"])
    audit_payload = json.loads((tmp_path / output["resolution_audit"]).read_text(encoding="utf-8"))

    assert audit["status"] == "blocked_official_rights_release_resolution"
    assert audit_payload["gate"] == "blocked_allen_12_family_rights_release_resolution_v2"
    assert rights["family_count"] == 12
    assert {row["family_uid"] for row in rights["records"]} == ALLEN_FAMILY_UIDS
    assert {row["family_uid"] for row in release_rows} == ALLEN_FAMILY_UIDS
    assert retrieval["payload_download_authorized"] is False
    assert all(entry["payload_download_authorized"] is False for entry in retrieval["entries"])
    assert all(str(row["decision"]).startswith("blocked_") for row in rights["records"])
    assert {"allen_terms_of_use", "allen_citation_policy"} <= set(audit_payload["terms_snapshot_hashes"])


def test_official_metadata_audit_schema_fixture_is_valid() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    fixture = resolution.build_schema_fixture()
    assert list(Draft202012Validator(schema).iter_errors(fixture)) == []
