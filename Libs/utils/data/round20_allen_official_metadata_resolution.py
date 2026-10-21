"""Official-metadata-only Allen rights and release resolution helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import csv
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from Libs.utils.data.round20_track_contracts import ALLEN_FAMILY_UIDS, RELEASE_FIELDS, RIGHTS_DECISIONS, RIGHTS_FIELDS
from Libs.utils.data.trajectory_contracts import is_sha256


CONFIG_KEY = "round20_allen_official_metadata_resolution"
CONFIG_RELATIVE = Path("Libs/configs/validation/round20_allen_official_metadata_resolution.yaml")
SCHEMA_VERSION = "neuronsifter-round20-allen-official-metadata-resolution-v1"
RIGHTS_SCHEMA_VERSION = "neuronsifter-round20-allen-rights-release-v1"
RELEASE_SCHEMA_VERSION = "neuronsifter-round20-allen-release-inventory-v1"
RETRIEVAL_SCHEMA_VERSION = "neuronsifter-round20-allen-retrieval-manifest-v1"
AUDIT_SCHEMA_VERSION = "neuronsifter-round20-allen-rights-release-resolution-v2"
UNRESOLVED_MARKERS = frozenset({"", "NOT_FROZEN", "NOT_AVAILABLE", "UNRESOLVED_OFFICIAL_IMMUTABLE_RELEASE", "UNRESOLVED_DATASET_SPECIFIC_LICENSE"})


@dataclass(frozen=True)
class FetchResult:
    source_uid: str
    url: str
    final_url: str
    status_code: int | None
    content_type: str
    headers: dict[str, str]
    body: bytes
    error: str | None = None
    redirected: bool = False
    blocked_redirect: str | None = None


class _BoundedRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: object, code: int, msg: str, headers: Mapping[str, str], newurl: str) -> Request | None:
        raise HTTPError(req.full_url, code, msg, headers, fp)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def sequence(value: object) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def blocker(code: str, message: str, **context: object) -> dict[str, object]:
    return {"code": code, "message": message, **context}


def sorted_blockers(blockers: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return sorted((dict(item) for item in blockers), key=lambda item: (str(item.get("family_uid", "")), str(item.get("source_uid", "")), str(item.get("code", ""))))


def host_for_url(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def is_official_url(url: str, allowed_suffixes: Sequence[str]) -> bool:
    host = host_for_url(url)
    suffixes = [str(value).lower() for value in allowed_suffixes]
    return bool(host) and any(host == suffix or host.endswith(f".{suffix}") for suffix in suffixes)


def validate_contract_definition(contract: Mapping[str, object]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if contract.get("version") != SCHEMA_VERSION:
        blockers.append(blocker("contract_version", "Official metadata resolution contract version is not frozen."))
    if contract.get("status") != "frozen_before_official_metadata_probe":
        blockers.append(blocker("contract_status", "Contract must be frozen before official metadata probing."))
    if contract.get("parent_repair_commit") != "4dfb0ee59a632d110d922bc1cd509fad7e36c3f1":
        blockers.append(blocker("parent_repair_commit", "Resolution contract must bind to the accepted repair commit."))
    if str(contract.get("parent_full_suite_job_id")) != "508644":
        blockers.append(blocker("parent_full_suite_job", "Resolution contract must bind to the accepted full-suite job."))

    policy = mapping(contract.get("source_policy"))
    allowed_suffixes = [str(value) for value in sequence(policy.get("allowed_host_suffixes"))]
    if policy.get("official_hosts_only") is not True or set(allowed_suffixes) != {"alleninstitute.org", "brain-map.org"}:
        blockers.append(blocker("official_host_policy", "Only official Allen host suffixes are allowed."))
    for key in ("general_terms_satisfy_dataset_specific_rights", "historical_release_hint_promotable", "mutable_latest_query_allowed", "redirect_to_non_official_host_allowed"):
        if policy.get(key) is not False:
            blockers.append(blocker("source_policy_fail_closed", f"{key} must remain false."))
    for key in ("release_identity_requires_official_immutable_evidence", "terms_and_release_snapshots_hash_required", "checked_at_execution_time_required"):
        if policy.get(key) is not True:
            blockers.append(blocker("source_policy_provenance", f"{key} must remain true."))

    configuration = mapping(contract.get("configuration"))
    if configuration.get("maximum_requests") != 48 or configuration.get("max_concurrent_requests") != 1:
        blockers.append(blocker("request_budget", "Request budget or serial concurrency drifted."))
    if configuration.get("request_timeout_seconds") != 20 or configuration.get("retry_count") != 0:
        blockers.append(blocker("request_retry_policy", "Timeout and retry policy must remain frozen."))

    snapshots = [mapping(value) for value in sequence(contract.get("source_snapshots"))]
    snapshot_uids = [str(value.get("source_uid", "")) for value in snapshots]
    if len(snapshots) != len(snapshot_uids) or len(set(snapshot_uids)) != len(snapshot_uids):
        blockers.append(blocker("source_snapshot_identity", "Source snapshot UIDs must be unique."))
    if len(snapshots) > int(configuration.get("maximum_requests", 0)):
        blockers.append(blocker("request_budget_exceeded", "Configured source snapshots exceed the frozen request budget."))
    for item in snapshots:
        url = str(item.get("url", ""))
        if not is_official_url(url, allowed_suffixes):
            blockers.append(blocker("non_official_metadata_url", "Metadata URL is outside the allowed official host suffixes.", source_uid=item.get("source_uid"), url=url))

    families = [mapping(value) for value in sequence(contract.get("family_metadata"))]
    family_uids = [str(value.get("family_uid", "")) for value in families]
    if len(families) != 12 or set(family_uids) != ALLEN_FAMILY_UIDS or len(family_uids) != len(set(family_uids)):
        blockers.append(blocker("family_inventory_drift", "Exactly 12 canonical Allen family records are required."))
    snapshot_uid_set = set(snapshot_uids)
    for family in families:
        family_uid = str(family.get("family_uid", ""))
        if family.get("official_snapshot_uid") not in snapshot_uid_set:
            blockers.append(blocker("family_snapshot_binding", "Family is not bound to a configured official snapshot.", family_uid=family_uid, source_uid=family.get("official_snapshot_uid")))
        endpoint = str(family.get("api_or_download_endpoint", ""))
        if endpoint.startswith("http") and not is_official_url(endpoint, allowed_suffixes):
            blockers.append(blocker("non_official_endpoint", "Family endpoint is outside the allowed official host suffixes.", family_uid=family_uid, endpoint=endpoint))
        if family.get("release_identity_evidence") == "assumed":
            blockers.append(blocker("assumed_release_identity", "Release identity may not be filled by assumption.", family_uid=family_uid))

    authorization = mapping(contract.get("authorization"))
    for key in ("payload_download", "training", "evidence_fabric_sealing", "dt_core_evaluation", "remote_push"):
        if authorization.get(key) is not False:
            blockers.append(blocker("unauthorized_execution", f"{key} must remain false.", authorization=key))
    if authorization.get("bounded_official_metadata_probe") is not True:
        blockers.append(blocker("metadata_probe_authorization", "Bounded official metadata probing must be explicitly authorized."))
    return sorted_blockers(blockers)


def fetch_official_snapshot(
    source_uid: str,
    url: str,
    *,
    allowed_suffixes: Sequence[str],
    timeout_seconds: int,
    max_bytes: int,
    user_agent: str = "NeuronSifter-Round20-MetadataResolver/1.0",
) -> FetchResult:
    if not is_official_url(url, allowed_suffixes):
        return FetchResult(source_uid, url, url, None, "", {}, b"", error="non_official_metadata_url")
    opener = build_opener(_BoundedRedirectHandler)
    current_url = url
    redirected = False
    for _ in range(5):
        request = Request(current_url, headers={"User-Agent": user_agent, "Accept": "text/html,application/json,application/xml,text/xml,text/csv,text/tab-separated-values;q=0.8,*/*;q=0.1"})
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                body = response.read(max_bytes + 1)
                headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
                if len(body) > max_bytes:
                    return FetchResult(source_uid, url, response.geturl(), response.status, headers.get("content-type", ""), headers, body[:max_bytes], error="snapshot_too_large", redirected=redirected)
                return FetchResult(source_uid, url, response.geturl(), response.status, headers.get("content-type", ""), headers, body, redirected=redirected)
        except HTTPError as exc:
            location = exc.headers.get("Location") if exc.headers else None
            if exc.code in {301, 302, 303, 307, 308} and location:
                if not is_official_url(location, allowed_suffixes):
                    return FetchResult(source_uid, url, current_url, exc.code, "", {}, b"", error="blocked_non_official_redirect", redirected=True, blocked_redirect=location)
                current_url = location
                redirected = True
                continue
            body = exc.read(max_bytes) if hasattr(exc, "read") else b""
            return FetchResult(source_uid, url, current_url, exc.code, str(exc.headers.get("Content-Type", "")) if exc.headers else "", {}, body, error=f"http_error_{exc.code}", redirected=redirected)
        except (URLError, TimeoutError, OSError) as exc:
            return FetchResult(source_uid, url, current_url, None, "", {}, b"", error=type(exc).__name__)
    return FetchResult(source_uid, url, current_url, None, "", {}, b"", error="redirect_limit_exceeded", redirected=redirected)


def write_snapshots(
    contract: Mapping[str, object],
    *,
    repository: Path,
    generated_at: str,
    fetcher: Callable[..., FetchResult] = fetch_official_snapshot,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    policy = mapping(contract.get("source_policy"))
    configuration = mapping(contract.get("configuration"))
    output = mapping(contract.get("output_artifacts"))
    snapshot_dir = repository / str(output.get("snapshot_dir"))
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    allowed_suffixes = [str(value) for value in sequence(policy.get("allowed_host_suffixes"))]
    max_bytes = int(configuration.get("maximum_snapshot_bytes_per_response", 1048576))
    timeout_seconds = int(configuration.get("request_timeout_seconds", 20))
    total_limit = int(configuration.get("maximum_total_snapshot_bytes", 25165824))
    total_bytes = 0
    blockers: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for item in [mapping(value) for value in sequence(contract.get("source_snapshots"))]:
        source_uid = str(item.get("source_uid"))
        url = str(item.get("url"))
        result = fetcher(source_uid, url, allowed_suffixes=allowed_suffixes, timeout_seconds=timeout_seconds, max_bytes=max_bytes)
        total_bytes += len(result.body)
        if total_bytes > total_limit:
            blockers.append(blocker("snapshot_total_budget_exceeded", "Official metadata snapshots exceed the frozen total byte budget.", source_uid=source_uid))
        suffix = "txt" if not result.content_type else ("json" if "json" in result.content_type else "html")
        snapshot_path = snapshot_dir / f"{source_uid}.{suffix}"
        snapshot_path.write_bytes(result.body)
        digest = sha256_bytes(result.body)
        if result.error:
            blockers.append(blocker("official_snapshot_unavailable", "Official metadata snapshot could not be captured cleanly.", source_uid=source_uid, url=url, error=result.error))
        record = {
            "source_uid": source_uid,
            "role": str(item.get("role")),
            "required": bool(item.get("required", False)),
            "url": url,
            "final_url": result.final_url,
            "host": host_for_url(result.final_url or url),
            "status_code": result.status_code,
            "content_type": result.content_type,
            "etag": result.headers.get("etag"),
            "last_modified": result.headers.get("last-modified"),
            "snapshot_path": str(snapshot_path.relative_to(repository)),
            "snapshot_sha256": digest,
            "snapshot_bytes": len(result.body),
            "captured_at": generated_at,
            "error": result.error,
            "redirected": result.redirected,
            "blocked_redirect": result.blocked_redirect,
        }
        records.append(record)
    return records, sorted_blockers(blockers)


def _family_blockers(family: Mapping[str, object], snapshot_by_uid: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    family_uid = str(family.get("family_uid", ""))
    blockers: list[dict[str, object]] = []
    snapshot_uid = str(family.get("official_snapshot_uid", ""))
    snapshot = mapping(snapshot_by_uid.get(snapshot_uid))
    if not snapshot or snapshot.get("error"):
        blockers.append(blocker("official_snapshot_unavailable", "Family official snapshot is unavailable.", family_uid=family_uid, source_uid=snapshot_uid))
    if str(family.get("release_uid", "")) in UNRESOLVED_MARKERS or str(family.get("release_date_or_version", "")) in UNRESOLVED_MARKERS:
        blockers.append(blocker("missing_immutable_release_identity", "Family lacks an official immutable release UID/date/version.", family_uid=family_uid))
    if str(family.get("dataset_specific_license", "")) in UNRESOLVED_MARKERS:
        blockers.append(blocker("unresolved_dataset_specific_terms", "Dataset-specific license or terms are unresolved.", family_uid=family_uid))
    third_party_license = str(family.get("third_party_license", ""))
    payload_host = str(family.get("payload_host", ""))
    if "THIRD_PARTY" in third_party_license or payload_host.startswith("third_party_"):
        blockers.append(blocker("unresolved_third_party_payload_terms", "Third-party payload hosting or license terms are unresolved.", family_uid=family_uid, payload_host=payload_host))
    return sorted_blockers(blockers)


def _decision_for_blockers(blockers_for_family: Sequence[Mapping[str, object]]) -> str:
    codes = {str(item.get("code")) for item in blockers_for_family}
    if "unresolved_third_party_payload_terms" in codes:
        return "blocked_third_party_terms"
    if "missing_immutable_release_identity" in codes:
        return "blocked_missing_release_identity"
    if "unresolved_dataset_specific_terms" in codes or "official_snapshot_unavailable" in codes:
        return "blocked_terms_ambiguity"
    return "approved_internal_noncommercial"


def build_resolution_payloads(
    contract: Mapping[str, object],
    *,
    repository: Path,
    generated_at: str,
    source_snapshots: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    snapshot_by_uid = {str(item.get("source_uid")): dict(item) for item in source_snapshots}
    families = [mapping(value) for value in sequence(contract.get("family_metadata"))]
    rights_rows: list[dict[str, object]] = []
    release_rows: list[dict[str, object]] = []
    retrieval_entries: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []
    for family in sorted(families, key=lambda value: str(value.get("family_uid"))):
        family_uid = str(family.get("family_uid"))
        official_snapshot = mapping(snapshot_by_uid.get(str(family.get("official_snapshot_uid"))))
        family_blockers = _family_blockers(family, snapshot_by_uid)
        blockers.extend(family_blockers)
        decision = _decision_for_blockers(family_blockers)
        blocker_codes = ",".join(sorted({str(item.get("code")) for item in family_blockers})) or "none"
        allow_internal = decision == "approved_internal_noncommercial"
        rights_rows.append(
            {
                "family_uid": family_uid,
                "dataset_and_release": f"{family.get('dataset_name')}; release_uid={family.get('release_uid')}; release_date_or_version={family.get('release_date_or_version')}",
                "official_host": str(official_snapshot.get("host", "")),
                "payload_host": str(family.get("payload_host")),
                "governing_terms": "allen_terms_of_use",
                "dataset_specific_license": str(family.get("dataset_specific_license")),
                "third_party_license": str(family.get("third_party_license")),
                "citation_policy": str(family.get("citation_policy")),
                "internal_download_allowed": allow_internal,
                "internal_transformation_allowed": allow_internal,
                "internal_training_allowed": allow_internal,
                "aggregate_reporting_allowed": allow_internal,
                "raw_redistribution_allowed": False,
                "processed_redistribution_allowed": False,
                "derived_feature_release_allowed": False,
                "checkpoint_release_allowed": False,
                "commercial_use_status": "not_resolved_dataset_specific" if not allow_internal else "not_approved_for_commercial_use",
                "required_attribution": "Allen Institute citation policy plus dataset-specific citation if resolved",
                "decision": decision,
                "blocker": blocker_codes,
                "official_snapshot_uid": str(family.get("official_snapshot_uid")),
                "official_snapshot_sha256": str(official_snapshot.get("snapshot_sha256", "")),
                "blocker_details": family_blockers,
            }
        )
        release_rows.append(
            {
                "family_uid": family_uid,
                "release_uid": str(family.get("release_uid")),
                "release_date_or_version": str(family.get("release_date_or_version")),
                "official_landing_page": str(official_snapshot.get("url", "")),
                "api_or_download_endpoint": str(family.get("api_or_download_endpoint")),
                "object_inventory": str(family.get("object_inventory")),
                "format": str(family.get("format")),
                "expected_size": str(family.get("expected_size")),
                "official_checksum": str(family.get("official_checksum")),
                "expected_biological_object_counts": str(family.get("expected_biological_object_counts")),
                "pagination_or_request_identity": str(family.get("pagination_or_request_identity")),
                "etag_or_last_modified": str(official_snapshot.get("etag") or official_snapshot.get("last_modified") or "NOT_AVAILABLE"),
            }
        )
        retrieval_entries.append(
            {
                "family_uid": family_uid,
                "release_uid": str(family.get("release_uid")),
                "api_or_download_endpoint": str(family.get("api_or_download_endpoint")),
                "payload_download_authorized": False,
                "retrieval_status": "blocked_before_payload_retrieval" if family_blockers else "not_started_payload_download_prohibited_by_contract",
                "blocker": blocker_codes,
            }
        )

    rights_matrix = {
        "schema_version": RIGHTS_SCHEMA_VERSION,
        "artifact_uid": "dt_round20_allen_rights_matrix_v2",
        "generated_at": generated_at,
        "family_count": len(rights_rows),
        "records": rights_rows,
    }
    retrieval_manifest = {
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "artifact_uid": "allen_retrieval_manifest_v2",
        "generated_at": generated_at,
        "payload_download_authorized": False,
        "entries": retrieval_entries,
        "source_snapshots": list(source_snapshots),
    }
    status = "passed" if not blockers else "blocked_official_rights_release_resolution"
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "artifact_uid": "dt_round20_allen_rights_release_resolution_v2",
        "status": status,
        "gate": "allen_12_family_rights_resolved" if not blockers else "blocked_allen_12_family_rights_release_resolution_v2",
        "generated_at": generated_at,
        "family_count": len(families),
        "families_evaluated": len(rights_rows),
        "families_with_approved_internal_download": sum(1 for row in rights_rows if row["internal_download_allowed"] is True),
        "families_blocked": sum(1 for row in rights_rows if str(row["decision"]).startswith("blocked_")),
        "official_snapshot_count": len(source_snapshots),
        "official_snapshot_success_count": sum(1 for row in source_snapshots if not row.get("error")),
        "terms_snapshot_hashes": {
            str(row.get("source_uid")): str(row.get("snapshot_sha256"))
            for row in source_snapshots
            if row.get("source_uid") in {"allen_terms_of_use", "allen_citation_policy"}
        },
        "blocker_count": len(blockers),
        "blockers": sorted_blockers(blockers),
        "authorization": {
            "payload_download": False,
            "training": False,
            "evidence_fabric_sealing": False,
            "dt_core_evaluation": False,
            "remote_push": False,
        },
        "next_action": "repair_family_specific_license_release_or_payload_host_blockers_before_raw_materialization" if blockers else "freeze_release_inventory_then_materialize_raw_payloads",
    }
    return {
        "rights_matrix": rights_matrix,
        "release_rows": release_rows,
        "retrieval_manifest": retrieval_manifest,
        "resolution_audit": audit,
    }


def write_release_inventory(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["family_uid", *sorted(RELEASE_FIELDS)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_release_inventory(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def emit_artifacts(contract: Mapping[str, object], *, repository: Path, fetcher: Callable[..., FetchResult] = fetch_official_snapshot) -> dict[str, object]:
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    contract_blockers = validate_contract_definition(contract)
    snapshots, snapshot_blockers = write_snapshots(contract, repository=repository, generated_at=generated_at, fetcher=fetcher)
    payloads = build_resolution_payloads(contract, repository=repository, generated_at=generated_at, source_snapshots=snapshots)
    audit = dict(payloads["resolution_audit"])
    all_blockers = sorted_blockers([*contract_blockers, *snapshot_blockers, *sequence(audit.get("blockers"))])
    audit["blockers"] = all_blockers
    audit["blocker_count"] = len(all_blockers)
    audit["status"] = "passed" if not all_blockers else "blocked_official_rights_release_resolution"
    audit["gate"] = "allen_12_family_rights_resolved" if not all_blockers else "blocked_allen_12_family_rights_release_resolution_v2"
    output = mapping(contract.get("output_artifacts"))
    write_json(repository / str(output.get("rights_matrix")), mapping(payloads["rights_matrix"]))
    write_release_inventory(repository / str(output.get("release_inventory")), sequence(payloads["release_rows"]))
    write_json(repository / str(output.get("retrieval_manifest")), mapping(payloads["retrieval_manifest"]))
    write_json(repository / str(output.get("resolution_audit")), audit)
    return audit


def validate_emitted_artifacts(contract: Mapping[str, object], *, repository: Path) -> list[dict[str, object]]:
    blockers = validate_contract_definition(contract)
    output = mapping(contract.get("output_artifacts"))
    rights_path = repository / str(output.get("rights_matrix"))
    release_path = repository / str(output.get("release_inventory"))
    retrieval_path = repository / str(output.get("retrieval_manifest"))
    audit_path = repository / str(output.get("resolution_audit"))
    for label, path in (("rights_matrix", rights_path), ("release_inventory", release_path), ("retrieval_manifest", retrieval_path), ("resolution_audit", audit_path)):
        if not path.is_file():
            blockers.append(blocker("artifact_missing", "Expected official metadata artifact is missing.", artifact=label, path=str(path.relative_to(repository))))
    if blockers:
        return sorted_blockers(blockers)

    rights = json.loads(rights_path.read_text(encoding="utf-8"))
    release_rows = read_release_inventory(release_path)
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    rights_records = [mapping(value) for value in sequence(rights.get("records"))]
    retrieval_entries = [mapping(value) for value in sequence(retrieval.get("entries"))]
    family_sets = [
        {str(row.get("family_uid")) for row in rights_records},
        {str(row.get("family_uid")) for row in release_rows},
        {str(row.get("family_uid")) for row in retrieval_entries},
    ]
    if any(value != ALLEN_FAMILY_UIDS for value in family_sets):
        blockers.append(blocker("family_inventory_drift", "Rights, release, and retrieval family inventories must all equal the canonical 12-family set."))
    for row in rights_records:
        missing = sorted(RIGHTS_FIELDS - set(row))
        if missing:
            blockers.append(blocker("rights_fields_missing", "Rights row lacks required fields.", family_uid=row.get("family_uid"), missing=missing))
        if row.get("decision") not in RIGHTS_DECISIONS:
            blockers.append(blocker("rights_decision_invalid", "Rights decision is outside the allowed decision set.", family_uid=row.get("family_uid"), decision=row.get("decision")))
        if row.get("internal_download_allowed") is True and str(row.get("decision", "")).startswith("blocked_"):
            blockers.append(blocker("blocked_family_authorized", "Blocked rights row cannot authorize internal download.", family_uid=row.get("family_uid")))
        if not is_sha256(row.get("official_snapshot_sha256")):
            blockers.append(blocker("snapshot_hash_missing", "Rights row lacks a source snapshot SHA-256.", family_uid=row.get("family_uid")))
    for row in release_rows:
        missing = sorted(RELEASE_FIELDS - set(row))
        if missing:
            blockers.append(blocker("release_fields_missing", "Release row lacks required fields.", family_uid=row.get("family_uid"), missing=missing))
        if str(row.get("release_uid")) == "latest":
            blockers.append(blocker("mutable_latest_query", "Mutable latest release identity is forbidden.", family_uid=row.get("family_uid")))
    if retrieval.get("payload_download_authorized") is not False:
        blockers.append(blocker("payload_retrieval_attempted", "Retrieval manifest must keep payload download disabled."))
    if audit.get("authorization", {}).get("payload_download") is not False or audit.get("authorization", {}).get("training") is not False:
        blockers.append(blocker("unauthorized_execution", "Resolution audit cannot authorize payload download or training."))
    return sorted_blockers(blockers)


def build_schema_fixture() -> dict[str, object]:
    now = "2026-08-06T00:00:00Z"
    snapshot = {
        "source_uid": "allen_terms_of_use",
        "role": "governing_terms",
        "required": True,
        "url": "https://alleninstitute.org/legal/terms-of-use",
        "final_url": "https://alleninstitute.org/legal/terms-of-use",
        "host": "alleninstitute.org",
        "status_code": 200,
        "content_type": "text/html",
        "etag": None,
        "last_modified": None,
        "snapshot_path": "Output/audits/round20/official_metadata_resolution/snapshots/allen_terms_of_use.html",
        "snapshot_sha256": sha256_bytes(b"terms"),
        "snapshot_bytes": 5,
        "captured_at": now,
        "error": None,
        "redirected": False,
        "blocked_redirect": None,
    }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "artifact_uid": "dt_round20_allen_rights_release_resolution_v2",
        "status": "blocked_official_rights_release_resolution",
        "gate": "blocked_allen_12_family_rights_release_resolution_v2",
        "generated_at": now,
        "family_count": 12,
        "families_evaluated": 12,
        "families_with_approved_internal_download": 0,
        "families_blocked": 12,
        "official_snapshot_count": 1,
        "official_snapshot_success_count": 1,
        "terms_snapshot_hashes": {"allen_terms_of_use": str(snapshot["snapshot_sha256"])},
        "blocker_count": 1,
        "blockers": [blocker("missing_immutable_release_identity", "example", family_uid="allen_abc_mouse")],
        "authorization": {"payload_download": False, "training": False, "evidence_fabric_sealing": False, "dt_core_evaluation": False, "remote_push": False},
        "next_action": "repair_family_specific_license_release_or_payload_host_blockers_before_raw_materialization",
    }
