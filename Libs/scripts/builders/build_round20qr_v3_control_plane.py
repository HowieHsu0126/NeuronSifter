"""Materialize the fail-closed Round-20QR v3 control-plane audit.

This builder does not qualify a baseline or run a performance challenge.  It
binds a clean producer candidate, records the effective network route, and
re-adjudicates the existing diagnostic source matrix so downstream replay
cannot be authorized while source identity or adapter evidence is missing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "neuronsifter-round20qr-v3-control-plane-v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Output/audits/round20qr_v3"
DEFAULT_REGISTRY = REPO_ROOT / "Libs/configs/experiments/baseline_registry.yaml"
DEFAULT_REDUCTION_SCHEMA = (
    REPO_ROOT / "Libs/configs/schema/baseline_registry_reduction.schema.json"
)
DEFAULT_DIAGNOSTIC_CATALOG = (
    REPO_ROOT
    / "Output/audits/round20qr_minimal_challenge_v2/mandatory_source_catalog.tsv"
)
DEFAULT_CHALLENGE_CONFIG = (
    REPO_ROOT / "Output/audits/round20qr_contract_v3/challenge_thresholds.json"
)
EXPECTED_BASELINE_COUNT = 20
EXPECTED_C2B_STATUS = "blocked_insufficient_real_e2_e4"
PROBE_TIMEOUT_SECONDS = 3
IDENTITY_FIELDS = (
    "baseline_uid",
    "claim_ids",
    "baseline_class",
    "implementation_source",
    "repository_or_package",
)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    if isinstance(payload, dict):
        payload = {
            key: value for key, value in payload.items() if key != "artifact_uid"
        }
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _worktree_state(root: Path) -> dict[str, Any]:
    # The producer candidate is an isolated detached worktree. Index checks
    # avoid materializing the repository's large tracked blobs; the explicit
    # untracked check rejects files that could shadow producer inputs.
    tracked = subprocess.run(
        ["git", "-C", str(root), "diff-index", "--quiet", "HEAD", "--"],
        check=False,
    )
    staged = subprocess.run(
        ["git", "-C", str(root), "diff-index", "--cached", "--quiet", "HEAD", "--"],
        check=False,
    )
    untracked = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--others",
            "--exclude-standard",
            "--directory",
            "--no-empty-directory",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    untracked_entries = [
        entry for entry in untracked.stdout.splitlines() if entry.strip()
    ]
    return {
        "tracked_clean": tracked.returncode == 0,
        "staged_clean": staged.returncode == 0,
        "untracked_check_succeeded": untracked.returncode == 0,
        "untracked_file_count": len(untracked_entries),
        "worktree_clean": (
            tracked.returncode == 0
            and staged.returncode == 0
            and untracked.returncode == 0
            and not untracked_entries
        ),
    }


def _clean_worktree(root: Path) -> bool:
    return bool(_worktree_state(root)["worktree_clean"])


def _write_immutable(path: Path, payload: Any) -> None:
    content = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"immutable artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_tsv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _relative_or_label(path: Path, root: Path, label: str) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(root.resolve()):
        return resolved.relative_to(root.resolve()).as_posix()
    return label


def _redact_proxy(value: str) -> str:
    if not value:
        return ""
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{host}{port}"


def _proxy_snapshot() -> dict[str, str]:
    names = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    )
    return {
        name: _redact_proxy(os.environ[name]) for name in names if os.environ.get(name)
    }


def _git_proxy_config(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "config", "--show-origin", "--get-regexp", "proxy"],
        check=False,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _pip_config() -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "config", "list"],
        check=False,
        capture_output=True,
        text=True,
    )
    return [
        line
        for line in result.stdout.splitlines()
        if line.strip()
        and "password" not in line.lower()
        and "token" not in line.lower()
    ]


def _probe_git(
    url: str,
    *,
    disable_proxy: bool,
    root: Path | None = None,
    timeout_seconds: int = PROBE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    env = os.environ.copy()
    command = ["git"]
    if root is not None:
        command.extend(["-C", str(root)])
    if disable_proxy:
        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "NO_PROXY",
            "no_proxy",
        ):
            env.pop(name, None)
        # These overrides keep a repository/global Git proxy from silently
        # re-entering a probe that is labeled as direct.
        command.extend(["-c", "http.proxy=", "-c", "https.proxy="])
    command.extend(["ls-remote", "--symref", url, "HEAD"])
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {"status": "network_timeout", "identity_known": False}
    stderr = result.stderr.strip()
    if result.returncode == 0:
        return {
            "status": "identity_resolved",
            "identity_known": True,
            "head_ref": result.stdout.strip(),
        }
    if "127.0.0.1" in stderr and "7898" in stderr:
        return {"status": "localhost_proxy_refused", "identity_known": False}
    return {
        "status": "network_error",
        "identity_known": False,
        "error": stderr[:500],
    }


def _official_urls(registry: dict[str, Any]) -> list[str]:
    urls: set[str] = set()
    for entry in registry.get("reduction_registry", {}).get("mandatory", []):
        value = str(entry.get("repository_or_package", "")).strip()
        if value.startswith("http://") or value.startswith("https://"):
            urls.add(value.rstrip("/"))
    return sorted(urls)


def _load_catalog(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _claim_ids_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    return ",".join(item.strip() for item in str(value).split(",") if item.strip())


def _identity_value(field: str, value: Any) -> str:
    if field == "claim_ids":
        return _claim_ids_text(value)
    return str(value or "").strip()


def _route_summary(probes: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    direct_route_known = any(
        probe.get("direct", {}).get("status") == "identity_resolved"
        for probe in probes.values()
    )
    proxy_route_known = any(
        probe.get("proxy", {}).get("status") == "identity_resolved"
        for probe in probes.values()
    )
    official_source_route_known = direct_route_known or proxy_route_known
    return {
        "direct_route_known": direct_route_known,
        "proxy_route_known": proxy_route_known,
        "official_source_route_known": official_source_route_known,
        "status": (
            "source_route_available_pending_qualification"
            if official_source_route_known
            else "blocked_external_source_route"
        ),
    }


def _producer_identity(
    root: Path,
    output_root: Path,
    registry: Path,
    reduction_schema: Path,
    challenge_config: Path,
) -> dict[str, Any]:
    tracked_files = (
        "Libs/scripts/builders/build_round20qr_v3_control_plane.py",
        "Libs/scripts/validators/validate_round20qr_v3_control_plane.py",
    )
    implementation = []
    for relative in tracked_files:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        implementation.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    worktree = _worktree_state(root)
    return {
        "schema_version": "neuronsifter-round20qr-v3-producer-identity-v1",
        "git_bound": True,
        "worktree_clean": worktree["worktree_clean"],
        "tracked_clean": worktree["tracked_clean"],
        "staged_clean": worktree["staged_clean"],
        "untracked_check_succeeded": worktree["untracked_check_succeeded"],
        "untracked_file_count": worktree["untracked_file_count"],
        "dirty_patch_required": not (
            worktree["tracked_clean"] and worktree["staged_clean"]
        ),
        "unrelated_user_files_modified": worktree["untracked_file_count"] > 0,
        "remote_push": False,
        "commit": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "implementation": implementation,
        "inputs": {
            "baseline_registry": {
                "label": "frozen_external_input/baseline_registry.yaml",
                "bytes": registry.stat().st_size,
                "sha256": _sha256(registry),
            },
            "reduction_schema": {
                "label": "frozen_external_input/baseline_registry_reduction.schema.json",
                "bytes": reduction_schema.stat().st_size,
                "sha256": _sha256(reduction_schema),
            },
            "challenge_config": {
                "label": "frozen_external_input/round20qr_challenge_thresholds.json",
                "bytes": challenge_config.stat().st_size,
                "sha256": _sha256(challenge_config),
            },
        },
        "output_root": str(output_root),
        "created_at": _utc_now(),
    }


def build(
    root: Path,
    output_root: Path,
    registry_path: Path,
    reduction_schema_path: Path,
    diagnostic_catalog_path: Path,
    challenge_config_path: Path,
    *,
    probe_network: bool = True,
    probe_timeout_seconds: int = PROBE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    entries = list(registry.get("reduction_registry", {}).get("mandatory", []))
    if len(entries) != EXPECTED_BASELINE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_BASELINE_COUNT} mandatory entries, found {len(entries)}"
        )
    if (
        len({str(row.get("baseline_uid")) for row in entries})
        != EXPECTED_BASELINE_COUNT
    ):
        raise ValueError("mandatory registry contains duplicate baseline_uid values")
    if not reduction_schema_path.is_file() or not challenge_config_path.is_file():
        raise FileNotFoundError("frozen v3 input schema/config is missing")

    git_proxy_config = _git_proxy_config(root)
    proxy_snapshot = _proxy_snapshot()
    proxy_route_configured = bool(proxy_snapshot or git_proxy_config)
    urls = _official_urls(registry)
    probes = {}
    for url in urls:
        probes[url] = {
            "proxy": (
                _probe_git(
                    url,
                    disable_proxy=False,
                    root=root,
                    timeout_seconds=probe_timeout_seconds,
                )
                if probe_network and proxy_route_configured
                else {"status": "not_configured", "identity_known": False}
            ),
            "direct": _probe_git(
                url,
                disable_proxy=True,
                root=root,
                timeout_seconds=probe_timeout_seconds,
            )
            if probe_network
            else {"status": "not_run"},
        }
    localhost_proxy_dependency = any(
        "127.0.0.1:7898" in value for value in proxy_snapshot.values()
    )
    route = _route_summary(probes)
    network_route = {
        "schema_version": "neuronsifter-round20qr-v3-network-route-v1",
        "created_at": _utc_now(),
        "proxy_environment": proxy_snapshot,
        "git_proxy_config": git_proxy_config,
        "pip_config": _pip_config(),
        "localhost_proxy_dependency": localhost_proxy_dependency,
        "proxy_route_configured": proxy_route_configured,
        "direct_route_attempted": probe_network,
        **route,
        "network_failures_distinguished_from_source_failures": True,
        "probe_timeout_seconds": probe_timeout_seconds,
        "probes": probes,
    }
    source_access = {
        "schema_version": "neuronsifter-round20qr-v3-source-access-v1",
        "created_at": _utc_now(),
        "official_source_route_known": route["official_source_route_known"],
        "localhost_proxy_dependency": localhost_proxy_dependency,
        "network_failures_distinguished_from_source_failures": True,
        "source_identity_queries": len(urls),
        "identity_resolved_count": sum(
            1
            for probe in probes.values()
            if probe.get("direct", {}).get("status") == "identity_resolved"
            or probe.get("proxy", {}).get("status") == "identity_resolved"
        ),
        "unresolved_network_failure_count": sum(
            1
            for probe in probes.values()
            if probe.get("direct", {}).get("status") != "identity_resolved"
            and probe.get("proxy", {}).get("status") != "identity_resolved"
        ),
        "status": route["status"],
        "probe_timeout_seconds": probe_timeout_seconds,
        "probes": probes,
    }

    diagnostic_rows = _load_catalog(diagnostic_catalog_path)
    if len(diagnostic_rows) != EXPECTED_BASELINE_COUNT:
        raise ValueError("diagnostic source catalog is not the frozen 20-entry matrix")
    catalog_by_uid = {str(row.get("baseline_uid", "")): row for row in diagnostic_rows}
    registry_by_uid = {str(row.get("baseline_uid", "")): row for row in entries}
    if len(catalog_by_uid) != EXPECTED_BASELINE_COUNT:
        raise ValueError(
            "diagnostic source catalog contains duplicate baseline_uid values"
        )
    if set(catalog_by_uid) != set(registry_by_uid):
        raise ValueError(
            "diagnostic source catalog UID set does not match frozen registry"
        )
    for uid, entry in registry_by_uid.items():
        diagnostic = catalog_by_uid[uid]
        for field in IDENTITY_FIELDS:
            if _identity_value(field, diagnostic.get(field)) != _identity_value(
                field, entry.get(field)
            ):
                raise ValueError(
                    f"diagnostic source catalog identity mismatch for {uid}:{field}"
                )
    source_fields = [
        "baseline_uid",
        "claim_ids",
        "baseline_class",
        "repository_or_package",
        "diagnostic_status",
        "source_identity_known",
        "version_pin_known",
        "license_decision_known",
        "adapter_status_known",
        "formal_challenge_eligible",
        "reason",
    ]
    source_rows = []
    for entry in entries:
        uid = str(entry["baseline_uid"])
        diagnostic = catalog_by_uid[uid]
        repository = str(entry.get("repository_or_package", ""))
        probe = probes.get(repository.rstrip("/"), {})
        identity_known = any(
            probe.get(channel, {}).get("status") == "identity_resolved"
            for channel in ("proxy", "direct")
        )
        source_rows.append(
            {
                "baseline_uid": uid,
                "claim_ids": _claim_ids_text(entry.get("claim_ids", [])),
                "baseline_class": entry.get("baseline_class", ""),
                "repository_or_package": repository,
                "diagnostic_status": diagnostic.get("qualification_status", ""),
                "source_identity_known": str(identity_known),
                "version_pin_known": "False",
                "license_decision_known": "False",
                "adapter_status_known": str(
                    bool(diagnostic.get("adapter_qualification_status"))
                ),
                "formal_challenge_eligible": "False",
                "reason": diagnostic.get("reason", "")
                or "v3 source qualification not started",
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    producer = _producer_identity(
        root, output_root, registry_path, reduction_schema_path, challenge_config_path
    )
    if producer["worktree_clean"] is not True:
        raise RuntimeError("producer candidate is not an exact clean worktree")
    _write_immutable(output_root / "network_route_manifest.json", network_route)
    _write_immutable(output_root / "source_access_preflight.json", source_access)
    _write_tsv(output_root / "mandatory_source_matrix.tsv", source_fields, source_rows)
    runtime = {
        "schema_version": "neuronsifter-round20qr-v3-runtime-dependency-v1",
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "pip_freeze": subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.splitlines(),
    }
    _write_immutable(output_root / "runtime_dependency_manifest.json", runtime)
    producer["runtime_dependency_manifest_sha256"] = _sha256(
        output_root / "runtime_dependency_manifest.json"
    )
    producer["worktree_clean"] = bool(producer["worktree_clean"])
    _write_immutable(output_root / "producer_identity.json", producer)

    counts: dict[str, int] = {}
    for row in diagnostic_rows:
        status = str(row.get("qualification_status", ""))
        counts[status] = counts.get(status, 0) + 1
    blockers = [
        "mandatory_source_or_adapter_evidence_incomplete",
        "version_pins_unresolved",
        "license_decisions_unresolved",
        "minimal_challenge_not_started",
    ]
    if not route["official_source_route_known"]:
        blockers.insert(0, "official_source_route_unresolved")
    gate = {
        "schema_version": "neuronsifter-round20qr-v3-gate-v1",
        "status": "blocked_round20qr_v3_source_qualification",
        "round20qr_v3_baseline_challenge_resolved": False,
        "mandatory_baselines": EXPECTED_BASELINE_COUNT,
        "source_identity_known": sum(
            row["source_identity_known"] == "True" for row in source_rows
        ),
        "version_pin_known": 0,
        "license_decision_known": 0,
        "adapter_status_known": sum(
            row["adapter_status_known"] == "True" for row in source_rows
        ),
        "simple_baseline_sanity_passed": False,
        "strong_baseline_results_available": False,
        "challenge_thresholds_frozen": False,
        "headline_training_authorized": False,
        "round21r_replay_authorized": False,
        "c3_c4_execution_allowed": False,
        "coevolution_execution_allowed": False,
        "source_counts_from_diagnostic_catalog": counts,
        "source_access_status": source_access["status"],
        "c2b_status": EXPECTED_C2B_STATUS,
        "blockers": blockers,
        "next_action": (
            "restore_approved_source_route_then_qualify_missing_mandatory_baseline_sources"
            if not route["official_source_route_known"]
            else "qualify_missing_mandatory_baseline_sources"
        ),
    }
    _write_immutable(output_root / "gate.json", gate)
    root_manifest = {
        "schema_version": "neuronsifter-round20qr-v3-root-v1",
        "artifact_uid": "",
        "status": gate["status"],
        "lineage_uid": "round20qr_v3",
        "producer": {
            "path": "producer_identity.json",
            "sha256": _sha256(output_root / "producer_identity.json"),
        },
        "network_route": {
            "path": "network_route_manifest.json",
            "sha256": _sha256(output_root / "network_route_manifest.json"),
        },
        "source_access": {
            "path": "source_access_preflight.json",
            "sha256": _sha256(output_root / "source_access_preflight.json"),
        },
        "source_matrix": {
            "path": "mandatory_source_matrix.tsv",
            "sha256": _sha256(output_root / "mandatory_source_matrix.tsv"),
        },
        "runtime": {
            "path": "runtime_dependency_manifest.json",
            "sha256": _sha256(output_root / "runtime_dependency_manifest.json"),
        },
        "gate": {"path": "gate.json", "sha256": _sha256(output_root / "gate.json")},
        "c2b_status": EXPECTED_C2B_STATUS,
        "round21r_replay_authorized": False,
    }
    root_manifest["artifact_uid"] = _canonical_sha256(root_manifest)
    _write_immutable(output_root / "root_manifest.json", root_manifest)
    replay = {
        "schema_version": "neuronsifter-round20qr-v3-replay-authorization-v1",
        "authority_root": "prospective_round20qr_v3",
        "round21r_replay_authorized": False,
        "gpu_allowed": False,
        "reason": "round20qr_v3_baseline_challenge_not_resolved",
        "root_manifest": "Output/audits/round20qr_v3/root_manifest.json",
        "c2b_status": EXPECTED_C2B_STATUS,
    }
    _write_immutable(output_root / "round21r_replay_authorization.json", replay)
    return {
        "gate": gate,
        "root_manifest": root_manifest,
        "source_access": source_access,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--reduction-schema", default=str(DEFAULT_REDUCTION_SCHEMA))
    parser.add_argument("--diagnostic-catalog", default=str(DEFAULT_DIAGNOSTIC_CATALOG))
    parser.add_argument("--challenge-config", default=str(DEFAULT_CHALLENGE_CONFIG))
    parser.add_argument("--skip-network-probe", action="store_true")
    parser.add_argument(
        "--probe-timeout-seconds", type=int, default=PROBE_TIMEOUT_SECONDS
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.probe_timeout_seconds <= 0:
        raise ValueError("--probe-timeout-seconds must be positive")
    result = build(
        Path(args.root).resolve(),
        Path(args.output_root).resolve(),
        Path(args.registry).resolve(),
        Path(args.reduction_schema).resolve(),
        Path(args.diagnostic_catalog).resolve(),
        Path(args.challenge_config).resolve(),
        probe_network=not args.skip_network_probe,
        probe_timeout_seconds=args.probe_timeout_seconds,
    )
    print(
        json.dumps(
            {
                "status": result["gate"]["status"],
                "artifact_uid": result["root_manifest"]["artifact_uid"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
