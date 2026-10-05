from __future__ import annotations

import glob
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Iterator, Mapping

from Libs.utils.data.controlled_research_config import CONTROLLED_RESEARCH_REQUIRED_METADATA
from Libs.utils.data.raw_data_registry import load_config_with_provenance


REPO_ROOT = Path(__file__).resolve().parents[3]
ROUTE_CLASSES = ("config", "experiment", "tool", "report_release", "anonymous_package")
ENV_INDIRECTION_PATTERN = re.compile(r"\$\{|\$\(|%\([^)]+\)s")
GOVERNANCE_CONTEXT_ROLES = {
    "controlled_exclusion_policy",
    "controlled_optional_report_contract",
    "g0_blocker_record",
    "isolation_contract",
}


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_mappings(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [_mapping(row) for row in value]


def _block(blockers: list[dict[str, object]], code: str, message: str, **context: object) -> None:
    blockers.append({"code": code, "message": message, "context": context})


def _resolve(path_value: object) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _label(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return resolved.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=True) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dotted_value(payload: Mapping[str, object], dotted_path: str) -> object:
    current: object = payload
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def _controlled_string_matches(value: str, contract: Mapping[str, object]) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    lowered = value.casefold()
    for identifier in [
        *contract.get("controlled_source_identifiers", []),
        *contract.get("controlled_source_classes", []),
    ]:
        marker = str(identifier)
        normalized = marker.casefold().replace("_", " ").replace("-", " ")
        candidate = lowered.replace("_", " ").replace("-", " ")
        if normalized and normalized in candidate:
            matches.append({"kind": "controlled_identifier", "value": value, "matched": marker})
    for path_root in contract.get("controlled_path_roots", []):
        marker = str(path_root)
        if marker.casefold() in lowered:
            matches.append({"kind": "controlled_path_root", "value": value, "matched": marker})
    for config_path in contract.get("controlled_config_paths", []):
        marker = str(config_path)
        if marker.casefold() in lowered:
            matches.append({"kind": "controlled_config_path", "value": value, "matched": marker})
    if ENV_INDIRECTION_PATTERN.search(value):
        matches.append({"kind": "unresolved_environment_indirection", "value": value, "matched": value})
    path_like_glob = glob.has_magic(value) and (
        "/" in value
        or value.endswith((".yaml", ".yml", ".json", ".tsv", ".csv"))
    )
    if path_like_glob:
        pattern = str(_resolve(value)) if not Path(value).is_absolute() else value
        controlled_paths = {_resolve(path).as_posix() for path in contract.get("controlled_config_paths", [])}
        for matched_path in glob.glob(pattern, recursive=False):
            resolved = Path(matched_path).resolve().as_posix()
            if resolved in controlled_paths:
                matches.append({"kind": "wildcard_controlled_config", "value": value, "matched": _label(Path(resolved))})
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in matches:
        key = json.dumps(row, sort_keys=True)
        if key not in seen:
            unique.append(row)
            seen.add(key)
    return unique


def _controlled_payload_matches(
    payload: object,
    contract: Mapping[str, object],
    path: str = "",
) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    boolean_keys = {str(value).casefold() for value in contract.get("controlled_boolean_keys", [])}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            child = f"{path}.{key_text}" if path else key_text
            for match in _controlled_string_matches(key_text, contract):
                matches.append({"value_path": child, "location": "mapping_key", **match})
            if value is True and key_text.casefold() in boolean_keys:
                matches.append(
                    {
                        "value_path": child,
                        "location": "boolean_value",
                        "kind": "controlled_boolean_enabled",
                        "value": "true",
                        "matched": key_text,
                    }
                )
            matches.extend(_controlled_payload_matches(value, contract, child))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            matches.extend(_controlled_payload_matches(value, contract, f"{path}[{index}]"))
    elif isinstance(payload, str):
        for match in _controlled_string_matches(payload, contract):
            matches.append({"value_path": path, "location": "string_value", **match})
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in matches:
        key = json.dumps(row, sort_keys=True, ensure_ascii=True)
        if key not in seen:
            unique.append(row)
            seen.add(key)
    return unique


def _route_findings(config: Mapping[str, object], contract: Mapping[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    blockers: list[dict[str, object]] = []
    findings: list[dict[str, object]] = []
    for route in _list_of_mappings(contract.get("public_routes")):
        route_id = str(route.get("route_id", ""))
        route_class = str(route.get("route_class", ""))
        dotted_path = str(route.get("dotted_path", ""))
        if route_class not in ROUTE_CLASSES:
            _block(blockers, "unknown_public_route_class", "A public route uses an unknown route class.", route_id=route_id, route_class=route_class)
            continue
        try:
            payload = _dotted_value(config, dotted_path)
        except KeyError:
            _block(blockers, "missing_public_route", "A configured public route does not resolve.", route_id=route_id, dotted_path=dotted_path)
            continue
        controlled_matches = _controlled_payload_matches(payload, contract, dotted_path)
        finding = {
            "route_id": route_id,
            "route_class": route_class,
            "dotted_path": dotted_path,
            "status": "blocked" if controlled_matches else "isolated",
            "controlled_matches": controlled_matches,
            "controlled_match_count": len(controlled_matches),
        }
        findings.append(finding)
        if controlled_matches:
            _block(
                blockers,
                "public_route_reaches_controlled_source",
                "A registered public executable route reaches a controlled source.",
                route_id=route_id,
                route_class=route_class,
                matches=controlled_matches,
            )
    return findings, blockers


def _governance_context_findings(
    config: Mapping[str, object],
    contract: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], set[str]]:
    blockers: list[dict[str, object]] = []
    findings: list[dict[str, object]] = []
    dotted_paths: set[str] = set()
    raw_contexts = contract.get("governance_only_contexts", [])
    if not isinstance(raw_contexts, list):
        _block(blockers, "invalid_governance_context_registry", "Governance-only contexts must be a list of explicit records.")
        return findings, blockers, dotted_paths
    for index, raw_context in enumerate(raw_contexts):
        if not isinstance(raw_context, Mapping):
            _block(
                blockers,
                "invalid_governance_context_record",
                "A governance-only context is not a structured record.",
                index=index,
            )
            continue
        context = dict(raw_context)
        dotted_path = str(context.get("dotted_path", ""))
        context_role = str(context.get("context_role", ""))
        metadata_valid = (
            bool(dotted_path)
            and context_role in GOVERNANCE_CONTEXT_ROLES
            and context.get("executable") is False
            and context.get("public_evidence_allowed") is False
        )
        if dotted_path in dotted_paths:
            metadata_valid = False
            _block(
                blockers,
                "duplicate_governance_context",
                "A governance-only context is declared more than once.",
                dotted_path=dotted_path,
            )
        dotted_paths.add(dotted_path)
        exists = True
        try:
            _dotted_value(config, dotted_path)
        except KeyError:
            exists = False
            _block(
                blockers,
                "missing_governance_context",
                "A governance-only context does not resolve in the public configuration.",
                dotted_path=dotted_path,
            )
        if not metadata_valid:
            _block(
                blockers,
                "invalid_governance_context_metadata",
                "A governance-only exclusion lacks a recognized non-executable, non-evidentiary role.",
                index=index,
                context=context,
            )
        findings.append(
            {
                "dotted_path": dotted_path,
                "context_role": context_role,
                "exists": exists,
                "executable": context.get("executable"),
                "public_evidence_allowed": context.get("public_evidence_allowed"),
                "status": "excluded_governance_only" if exists and metadata_valid else "blocked",
            }
        )
    dotted_paths.discard("")
    return findings, blockers, dotted_paths


def _unregistered_config_findings(
    config: Mapping[str, object],
    contract: Mapping[str, object],
    governance_paths: set[str],
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    registered_paths = {
        str(route.get("dotted_path", ""))
        for route in _list_of_mappings(contract.get("public_routes"))
        if route.get("dotted_path")
    }
    skipped_paths = governance_paths | registered_paths
    boolean_keys = {str(value).casefold() for value in contract.get("controlled_boolean_keys", [])}
    findings: list[dict[str, str]] = []

    def visit(payload: object, path: str) -> None:
        if path in skipped_paths:
            return
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                key_text = str(key)
                child = f"{path}.{key_text}" if path else key_text
                if child in skipped_paths:
                    continue
                for match in _controlled_string_matches(key_text, contract):
                    findings.append({"value_path": child, "location": "mapping_key", **match})
                if value is True and key_text.casefold() in boolean_keys:
                    findings.append(
                        {
                            "value_path": child,
                            "location": "boolean_value",
                            "kind": "controlled_boolean_enabled",
                            "value": "true",
                            "matched": key_text,
                        }
                    )
                visit(value, child)
            return
        if isinstance(payload, list):
            for index, value in enumerate(payload):
                visit(value, f"{path}[{index}]")
            return
        if isinstance(payload, str):
            for match in _controlled_string_matches(payload, contract):
                findings.append({"value_path": path, "location": "string_value", **match})

    visit(config, "")
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in findings:
        key = json.dumps(row, sort_keys=True, ensure_ascii=True)
        if key not in seen:
            unique.append(row)
            seen.add(key)
    blockers: list[dict[str, object]] = []
    if unique:
        _block(
            blockers,
            "unregistered_public_config_reaches_controlled_source",
            "A public configuration namespace outside the registered route graph reaches a controlled source.",
            matches=unique,
        )
    return unique, blockers


def _controlled_entrypoint_findings(contract: Mapping[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    blockers: list[dict[str, object]] = []
    findings: list[dict[str, object]] = []
    for path_value in contract.get("controlled_entrypoints", []):
        path = _resolve(path_value)
        relative = _label(path)
        if not path.is_file():
            _block(blockers, "missing_controlled_entrypoint", "A registered controlled entrypoint is missing.", path=relative)
            continue
        text = path.read_text(encoding="utf-8")
        checks = {
            "explicit_config_required": bool(re.search(r'add_argument\(["\']--config["\'],\s*required=True\)', text)),
            "controlled_loader_used": "load_controlled_research_config(args.config)" in text,
            "governance_preflight_used": "assert_controlled_governance_approval(config)" in text,
        }
        if path.name == "sync_adni_controlled_access.py":
            checks["credential_preflight_used"] = "assert_controlled_network_credentials(config)" in text
        failed = sorted(name for name, passed in checks.items() if not passed)
        findings.append({"path": relative, "status": "guarded" if not failed else "blocked", "checks": checks})
        if failed:
            _block(blockers, "controlled_entrypoint_guard_missing", "A controlled entrypoint can be invoked without a required guard.", path=relative, failed_checks=failed)
    return findings, blockers


def _static_contract_payload(contract: Mapping[str, object]) -> dict[str, object]:
    return {
        key: contract.get(key, [])
        for key in [
            "controlled_entrypoints",
            "governance_only_static_entrypoints",
            "static_scan_roots",
            "static_route_class_prefixes",
            "controlled_source_identifiers",
            "controlled_source_classes",
            "controlled_boolean_keys",
            "controlled_path_roots",
            "controlled_config_paths",
        ]
    }


def _controlled_literal_markers(text: str, contract: Mapping[str, object]) -> list[str]:
    lowered = text.casefold()
    markers = [
        *contract.get("controlled_source_identifiers", []),
        *contract.get("controlled_source_classes", []),
        *contract.get("controlled_path_roots", []),
        *contract.get("controlled_config_paths", []),
    ]
    return sorted({str(marker) for marker in markers if str(marker).casefold() in lowered})


def _static_route_class(path: Path, contract: Mapping[str, object]) -> str:
    relative = _label(path)
    candidates: list[tuple[int, str]] = []
    prefixes = _mapping(contract.get("static_route_class_prefixes"))
    for route_class, raw_values in prefixes.items():
        if str(route_class) not in ROUTE_CLASSES or not isinstance(raw_values, list):
            continue
        for raw_value in raw_values:
            prefix = str(raw_value).rstrip("/")
            if relative == prefix or relative.startswith(f"{prefix}/"):
                candidates.append((len(prefix), str(route_class)))
    return max(candidates)[1] if candidates else "config"


def _is_external_reference_path(path: Path) -> bool:
    external_root = REPO_ROOT / "Libs" / "external"
    resolved_path = path.resolve()
    return path.is_relative_to(external_root) or resolved_path.is_relative_to(external_root.resolve())


def _iter_static_scan_paths(root: Path) -> Iterator[Path]:
    for directory, child_directories, filenames in os.walk(root):
        current = Path(directory)
        child_directories[:] = sorted(
            child
            for child in child_directories
            if not _is_external_reference_path(current / child)
        )
        for filename in sorted(filenames):
            path = current / filename
            if path.suffix in {".py", ".sh"}:
                yield path


@lru_cache(maxsize=32)
def _cached_static_route_findings(contract_json: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    contract = json.loads(contract_json)
    blockers: list[dict[str, object]] = []
    findings: list[dict[str, object]] = []
    controlled_entrypoints = {_resolve(value) for value in contract.get("controlled_entrypoints", [])}
    governance_only_entrypoints = {_resolve(value) for value in contract.get("governance_only_static_entrypoints", [])}
    support_paths = {
        _resolve("Libs/utils/data/controlled_research_config.py"),
        _resolve("Libs/utils/data/public_mainline_isolation.py"),
    }
    for route_class, prefixes in _mapping(contract.get("static_route_class_prefixes")).items():
        if route_class not in ROUTE_CLASSES or not isinstance(prefixes, list):
            _block(
                blockers,
                "invalid_static_route_class_registry",
                "Static route-class prefixes must use known route classes and list values.",
                route_class=route_class,
            )
    for root_value in contract.get("static_scan_roots", []):
        root = _resolve(root_value)
        if not root.is_dir():
            _block(blockers, "missing_static_scan_root", "A configured static route root is missing.", path=_label(root))
            continue
        if _is_external_reference_path(root):
            _block(
                blockers,
                "external_reference_static_scan_root",
                "Third-party reference code cannot be a public-mainline static scan root.",
                path=_label(root),
            )
            continue
        for path in _iter_static_scan_paths(root):
            if (
                _is_external_reference_path(path)
                or path in controlled_entrypoints
                or path in governance_only_entrypoints
                or path in support_paths
                or "tests" in path.parts
            ):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            matches: list[str] = []
            if re.search(r'default\s*=\s*["\'][^"\']*config_stack_controlled_research\.yaml', text):
                matches.append("controlled_overlay_default")
            if re.search(r'(?:getenv|environ\.get)\([^\n]*(?:CONFIG|config)', text):
                matches.append("environment_conditioned_config_default")
            if re.search(r'(?:glob|rglob)\s*\([^\n]*\.(?:yaml|yml)', text) or re.search(
                r'(?:pkgutil\.iter_modules|import_module)\s*\([^\n]*(?:config|registry|plugin)',
                text,
                flags=re.IGNORECASE,
            ):
                matches.append("registry_or_wildcard_discovery")
            if re.search(r'^\s*(?:from\s+\S+\s+import\s+\*|import\s+\*)', text, flags=re.MULTILINE):
                matches.append("wildcard_import")
            controlled_literals = _controlled_literal_markers(text, contract)
            if controlled_literals:
                matches.append("controlled_literal")
            for key in contract.get("controlled_boolean_keys", []):
                if re.search(rf'\b{re.escape(str(key))}\b\s*[:=]\s*(?:true|True)', text):
                    matches.append("controlled_boolean_enabled")
                    break
            if matches:
                finding = {
                    "path": _label(path),
                    "route_class": _static_route_class(path, contract),
                    "status": "blocked",
                    "matches": sorted(set(matches)),
                    "controlled_literals": controlled_literals,
                }
                findings.append(finding)
                _block(blockers, "static_public_route_indirection", "A public code route has uncontrolled config or import indirection.", **finding)
    return findings, blockers


def _static_route_findings(contract: Mapping[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    contract_json = json.dumps(_static_contract_payload(contract), sort_keys=True, ensure_ascii=True)
    findings, blockers = _cached_static_route_findings(contract_json)
    return [dict(row) for row in findings], [dict(row) for row in blockers]


def validate_public_mainline_isolation_payload(
    config: Mapping[str, object],
    provenance: Mapping[str, object],
    config_path: str | Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    contract = _mapping(config.get("public_mainline_isolation"))
    blockers: list[dict[str, object]] = []
    config_file = _resolve(config_path)
    actual_entrypoint = str(provenance.get("entrypoint", ""))
    public_entrypoint = str(contract.get("public_entrypoint", ""))
    include_order = [str(value) for value in provenance.get("include_order", [])]
    include_graph = _mapping(provenance.get("include_graph"))
    controlled_paths = {str(value) for value in contract.get("controlled_config_paths", [])}
    controlled_sections = {str(value) for value in contract.get("controlled_section_keys", [])}
    controlled_nodes = sorted(set(include_order) & controlled_paths)
    loaded_controlled_sections = sorted(controlled_sections & set(config))
    if actual_entrypoint != public_entrypoint:
        _block(blockers, "public_entrypoint_mismatch", "The public isolation audit must start at the canonical public entrypoint.", expected=public_entrypoint, actual=actual_entrypoint)
    if controlled_nodes:
        _block(blockers, "public_config_reaches_controlled_node", "The public include graph reaches controlled configuration.", paths=controlled_nodes)
    if loaded_controlled_sections:
        _block(blockers, "public_config_loads_controlled_section", "The public merged config contains controlled sections.", sections=loaded_controlled_sections)

    isolated_overlays: list[dict[str, object]] = []
    for overlay in _list_of_mappings(contract.get("controlled_overlays")):
        overlay_path = _resolve(overlay.get("entrypoint", ""))
        row = {
            "overlay_id": overlay.get("overlay_id"),
            "entrypoint": _label(overlay_path),
            "exists": overlay_path.is_file(),
            "non_default": overlay.get("non_default") is True,
            "reachable_from_public": str(overlay.get("entrypoint", "")) in include_order,
            "status": "isolated",
        }
        if not row["exists"] or not row["non_default"] or row["reachable_from_public"]:
            row["status"] = "blocked"
            _block(blockers, "controlled_overlay_not_isolated", "A controlled overlay is missing, default, or reachable from public.", overlay=row)
        else:
            try:
                overlay_config, overlay_provenance = load_config_with_provenance(overlay_path)
                overlay_includes = set(overlay_provenance.get("include_order", []))
                required = {public_entrypoint, str(overlay.get("controlled_config", ""))}
                if not required.issubset(overlay_includes) or str(overlay.get("section_key", "")) not in overlay_config:
                    row["status"] = "blocked"
                    _block(blockers, "controlled_overlay_incomplete", "The explicit controlled overlay does not compose its declared public base and controlled config.", overlay_id=overlay.get("overlay_id"))
                metadata = _mapping(overlay_config.get("controlled_research_stack"))
                expected_metadata = dict(CONTROLLED_RESEARCH_REQUIRED_METADATA)
                metadata_mismatches = {
                    key: {"expected": expected, "actual": metadata.get(key)}
                    for key, expected in expected_metadata.items()
                    if metadata.get(key) != expected
                }
                row["governance_contract_valid"] = not metadata_mismatches
                if metadata_mismatches:
                    row["status"] = "blocked"
                    _block(
                        blockers,
                        "controlled_overlay_governance_invalid",
                        "The explicit controlled overlay does not preserve its non-public governance contract.",
                        overlay_id=overlay.get("overlay_id"),
                        mismatches=metadata_mismatches,
                    )
            except (FileNotFoundError, TypeError, ValueError) as error:
                row["status"] = "blocked"
                _block(blockers, "controlled_overlay_resolution_failed", "The explicit controlled overlay cannot be resolved.", overlay_id=overlay.get("overlay_id"), error=str(error))
        isolated_overlays.append(row)

    graph_blockers = [row for row in blockers if str(row.get("code", "")).startswith(("public_", "controlled_overlay"))]
    graph_report = {
        "schema_version": "neuronsifter-public-mainline-config-graph-v1",
        "status": "pass" if not graph_blockers else "blocked",
        "public_root": public_entrypoint,
        "actual_root": actual_entrypoint,
        "nodes": include_order,
        "edges": [
            {"source": str(source), "target": str(target)}
            for source, targets in include_graph.items()
            for target in targets
        ],
        "controlled_nodes_reachable": controlled_nodes,
        "controlled_nodes_reachable_count": len(controlled_nodes),
        "loaded_controlled_sections": loaded_controlled_sections,
        "isolated_controlled_overlays": isolated_overlays,
        "blockers": graph_blockers,
        "blocker_count": len(graph_blockers),
        "producer_command": "python Libs/scripts/validators/validate_public_mainline_isolation.py --config Libs/configs/config_stack.yaml",
    }

    governance_findings, governance_blockers, governance_paths = _governance_context_findings(config, contract)
    unregistered_findings, unregistered_blockers = _unregistered_config_findings(
        config,
        contract,
        governance_paths,
    )
    route_findings, route_blockers = _route_findings(config, contract)
    entrypoint_findings, entrypoint_blockers = _controlled_entrypoint_findings(contract)
    static_findings, static_blockers = _static_route_findings(contract)
    blockers.extend(governance_blockers)
    blockers.extend(unregistered_blockers)
    blockers.extend(route_blockers)
    blockers.extend(entrypoint_blockers)
    blockers.extend(static_blockers)
    route_class_counts = {route_class: 0 for route_class in ROUTE_CLASSES}
    route_class_counts["config"] = (
        len(controlled_nodes) + len(loaded_controlled_sections) + len(unregistered_findings)
    )
    for finding in route_findings:
        route_class = str(finding["route_class"])
        route_class_counts[route_class] += int(finding["controlled_match_count"])
    for finding in static_findings:
        route_class_counts[str(finding["route_class"])] += 1
    controlled_reachability_count = sum(route_class_counts.values())
    dedicated_validator_passed = not blockers and controlled_reachability_count == 0
    audit_report = {
        "schema_version": "neuronsifter-public-mainline-isolation-audit-v1",
        "status": "pass" if dedicated_validator_passed else "blocked",
        "config_path": _label(config_file),
        "config_sha256": _sha256_file(config_file),
        "resolved_config_sha256": _json_sha256(config),
        "config_graph_sha256": _json_sha256(graph_report),
        "route_class_counts": route_class_counts,
        "route_findings": route_findings,
        "unregistered_config_findings": unregistered_findings,
        "governance_context_findings": governance_findings,
        "controlled_entrypoint_findings": entrypoint_findings,
        "static_route_findings": static_findings,
        "controlled_reachability_count": controlled_reachability_count,
        "blockers": blockers,
        "blocker_count": len(blockers),
        "producer_command": "python Libs/scripts/validators/validate_public_mainline_isolation.py --config Libs/configs/config_stack.yaml",
    }
    remediation_report = {
        "schema_version": "neuronsifter-g0-public-mainline-remediation-v1",
        "blocker_id": "G0-PUBLIC-MAINLINE-CONTROLLED-INCLUDE",
        "original_status": "blocked",
        "candidate_status": "dedicated_validator_passed" if dedicated_validator_passed else "blocked",
        "dedicated_g0_rerun_required": not dedicated_validator_passed,
        "aggregate_g0_rerun_required": True,
        "dedicated_validator_passed": dedicated_validator_passed,
        "original_blocker_closed": dedicated_validator_passed,
        "phase1_open": False,
        "implementation_enabled": False,
        "isolation_audit_sha256": _json_sha256(audit_report),
        "controlled_reachability_count": controlled_reachability_count,
        "route_class_counts": route_class_counts,
        "status": "pass" if dedicated_validator_passed else "blocked",
        "blockers": blockers,
        "blocker_count": len(blockers),
        "immutable_blocker_ids": [
            "G0-NAMING-LEGACY-RESIDUALS",
            "G0-PUBLIC-MAINLINE-CONTROLLED-INCLUDE",
            "G0-NUMERIC-LINEAGE-DRIFT",
            "G0-CLAIM-GATED-WORDING",
        ],
        "producer_command": "python Libs/scripts/validators/validate_public_mainline_isolation.py --config Libs/configs/config_stack.yaml",
    }
    return graph_report, audit_report, remediation_report


def build_public_mainline_isolation_reports(
    config_path: str | Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    try:
        config, provenance = load_config_with_provenance(config_path)
    except (FileNotFoundError, TypeError, ValueError) as error:
        blocker = {
            "code": "config_include_resolution_failed",
            "message": "Public config include resolution failed closed.",
            "context": {"error": str(error)},
        }
        graph = {
            "schema_version": "neuronsifter-public-mainline-config-graph-v1",
            "status": "blocked",
            "public_root": str(config_path),
            "actual_root": str(config_path),
            "nodes": [],
            "edges": [],
            "controlled_nodes_reachable": [],
            "controlled_nodes_reachable_count": 0,
            "loaded_controlled_sections": [],
            "isolated_controlled_overlays": [],
            "blockers": [blocker],
            "blocker_count": 1,
            "producer_command": "python Libs/scripts/validators/validate_public_mainline_isolation.py --config Libs/configs/config_stack.yaml",
        }
        audit = {
            "schema_version": "neuronsifter-public-mainline-isolation-audit-v1",
            "status": "blocked",
            "config_path": str(config_path),
            "config_sha256": None,
            "resolved_config_sha256": None,
            "config_graph_sha256": _json_sha256(graph),
            "route_class_counts": {route_class: 0 for route_class in ROUTE_CLASSES},
            "route_findings": [],
            "unregistered_config_findings": [],
            "governance_context_findings": [],
            "controlled_entrypoint_findings": [],
            "static_route_findings": [],
            "controlled_reachability_count": 0,
            "blockers": [blocker],
            "blocker_count": 1,
            "producer_command": "python Libs/scripts/validators/validate_public_mainline_isolation.py --config Libs/configs/config_stack.yaml",
        }
        remediation = {
            "schema_version": "neuronsifter-g0-public-mainline-remediation-v1",
            "blocker_id": "G0-PUBLIC-MAINLINE-CONTROLLED-INCLUDE",
            "original_status": "blocked",
            "candidate_status": "blocked",
            "dedicated_g0_rerun_required": True,
            "aggregate_g0_rerun_required": True,
            "dedicated_validator_passed": False,
            "original_blocker_closed": False,
            "phase1_open": False,
            "implementation_enabled": False,
            "isolation_audit_sha256": _json_sha256(audit),
            "controlled_reachability_count": 0,
            "route_class_counts": {route_class: 0 for route_class in ROUTE_CLASSES},
            "status": "blocked",
            "blockers": [blocker],
            "blocker_count": 1,
            "immutable_blocker_ids": [
                "G0-NAMING-LEGACY-RESIDUALS",
                "G0-PUBLIC-MAINLINE-CONTROLLED-INCLUDE",
                "G0-NUMERIC-LINEAGE-DRIFT",
                "G0-CLAIM-GATED-WORDING",
            ],
            "producer_command": "python Libs/scripts/validators/validate_public_mainline_isolation.py --config Libs/configs/config_stack.yaml",
        }
        return graph, audit, remediation
    return validate_public_mainline_isolation_payload(config, provenance, config_path)
