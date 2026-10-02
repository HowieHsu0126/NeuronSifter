from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Mapping

from Libs.migrations.legacy_naming import canonical_replacement_for_legacy_symbol
from Libs.utils.data.raw_data_registry import write_json, write_text_atomic, write_tsv


REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_PATTERN = re.compile(r"mind(?:[-_ ]?)dino", re.IGNORECASE)
NUMBERED_ROUTE_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:bf|r)[0-9]+(?![A-Za-z0-9])", re.IGNORECASE)
INVENTORY_COLUMNS = [
    "file",
    "line",
    "matched_string",
    "symbol_or_text",
    "runtime_reachable",
    "public_facing",
    "category",
    "required_action",
    "canonical_replacement",
    "compatibility_required",
    "owner",
    "status",
]
ALLOWED_CATEGORIES = {
    "allowed_deprecated_alias",
    "allowed_checkpoint_migration",
    "allowed_historical_record",
    "allowed_test_fixture",
}
FORBIDDEN_CATEGORIES = {
    "forbidden_active_code",
    "forbidden_active_config",
    "forbidden_public_text",
    "forbidden_api_or_report",
    "forbidden_numbered_active_code",
    "forbidden_numbered_active_config",
    "forbidden_numbered_public_text",
    "forbidden_numbered_api_or_report",
}

NUMBERED_METADATA_MARKERS = (
    "schema_version",
    "corpus_version",
    "mapping_version",
    "unit_mapping_version",
    "parser_version",
    "contract_version",
    "model_contract_version",
    "store_format",
    "source_release",
    "release_version",
    "canonical_uid",
    "expected_canonical_uid",
    "sha256",
    "path",
    "root",
    "url",
    "historical",
    "parent_",
    "precision",
    "headline",
    "package",
    "r-",
)


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _policy(config: Mapping[str, object]) -> dict[str, object]:
    union = _mapping(config.get("iclr_design_contract_union"))
    policy = _mapping(union.get("g0_naming_audit"))
    if not policy:
        raise KeyError("iclr_design_contract_union.g0_naming_audit")
    return policy


def _relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _discover_files(config: Mapping[str, object], repo_root: Path) -> list[Path]:
    policy = _policy(config)
    suffixes = {str(value) for value in policy.get("text_suffixes", [])}
    excluded = tuple(str(value).rstrip("/") for value in policy.get("excluded_path_prefixes", []))
    files: set[Path] = set()
    for path_value in policy.get("scan_paths", []):
        candidate = repo_root / str(path_value)
        if candidate.is_file():
            files.add(candidate)
            continue
        if candidate.is_dir():
            for current_root, directory_names, file_names in os.walk(candidate):
                current_path = Path(current_root)
                directory_names[:] = [
                    name
                    for name in directory_names
                    if not any(
                        _relative(current_path / name, repo_root) == prefix
                        or _relative(current_path / name, repo_root).startswith(f"{prefix}/")
                        for prefix in excluded
                    )
                ]
                for file_name in file_names:
                    path = current_path / file_name
                    if not suffixes or path.suffix in suffixes:
                        files.add(path)
    return sorted(
        path
        for path in files
        if not any(
            _relative(path, repo_root) == prefix
            or _relative(path, repo_root).startswith(f"{prefix}/")
            for prefix in excluded
        )
    )


def _allowed_category(
    relative_path: str,
    policy: Mapping[str, object],
    line_number: int,
    union_compatibility_lines: set[int],
) -> str | None:
    if relative_path == "Libs/tests/test_g0_naming_migration.py":
        return "allowed_test_fixture"
    if relative_path == "Libs/configs/validation/iclr_design_contract_union.yaml" and line_number in union_compatibility_lines:
        return "allowed_deprecated_alias"
    for row_value in policy.get("legacy_import_shim_modules", []):
        row = _mapping(row_value)
        package_path = "/".join(str(value) for value in row.get("package_parts", []))
        module_stem = "_".join(str(value) for value in row.get("module_stem_parts", []))
        if relative_path == f"{package_path}/{module_stem}.py":
            return str(row.get("category", ""))
    for row_value in policy.get("allowed_path_categories", []):
        row = _mapping(row_value)
        prefix = str(row.get("path_prefix", "")).rstrip("/")
        if relative_path == prefix or relative_path.startswith(f"{prefix}/"):
            category = str(row.get("category", ""))
            if category == "allowed_deprecated_alias" and relative_path.startswith("Libs/migrations/") and "checkpoint" in Path(relative_path).name:
                return "allowed_checkpoint_migration"
            return category
    return None


def _forbidden_category(relative_path: str) -> str:
    if relative_path.startswith("Libs/configs/"):
        return "forbidden_active_config"
    if relative_path.startswith(("Libs/scripts/services/", "Libs/scripts/reports/", "Libs/exp/submission/")):
        return "forbidden_api_or_report"
    if relative_path.endswith(".sh") and relative_path.startswith("Libs/"):
        return "forbidden_api_or_report"
    if not relative_path.startswith("Libs/"):
        return "forbidden_public_text"
    return "forbidden_active_code"


def _forbidden_numbered_category(relative_path: str) -> str:
    base = _forbidden_category(relative_path)
    return f"forbidden_numbered_{base.removeprefix('forbidden_')}"


def _numbered_route_allowed(relative_path: str, line: str, policy: Mapping[str, object]) -> bool:
    normalized_line = line.lower()
    for marker in policy.get("numbered_route_metadata_markers", NUMBERED_METADATA_MARKERS):
        if str(marker).lower() in normalized_line:
            return True
    for prefix in policy.get("numbered_route_allowed_path_prefixes", []):
        value = str(prefix).rstrip("/")
        if relative_path == value or relative_path.startswith(f"{value}/"):
            return True
    return False


def _build_numbered_row(
    relative_path: str,
    line_number: int,
    matched_string: str,
    symbol_or_text: str,
    policy: Mapping[str, object],
) -> dict[str, object] | None:
    if _numbered_route_allowed(relative_path, symbol_or_text, policy):
        return None
    allowed_category = _allowed_category(relative_path, policy, line_number, set())
    if allowed_category is not None:
        return None
    row = _build_row(relative_path, line_number, matched_string, symbol_or_text, policy, set())
    row["category"] = _forbidden_numbered_category(relative_path)
    row["required_action"] = "replace_with_semantic_route_name"
    row["canonical_replacement"] = "semantic_route_name_without_release_number"
    row["compatibility_required"] = "false"
    row["status"] = "open"
    return row


def _union_compatibility_lines(text: str) -> set[int]:
    lines = text.splitlines()
    block_lines: set[int] = set()
    block_indent: int | None = None
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if stripped == "legacy_compatibility:":
            block_indent = indent
            continue
        if block_indent is not None and stripped and indent <= block_indent:
            block_indent = None
        if block_indent is not None:
            block_lines.add(line_number)
    return block_lines


def _build_row(
    relative_path: str,
    line_number: int,
    matched_string: str,
    symbol_or_text: str,
    policy: Mapping[str, object],
    union_compatibility_lines: set[int],
) -> dict[str, object]:
    category = _allowed_category(relative_path, policy, line_number, union_compatibility_lines)
    if category is None:
        category = _forbidden_category(relative_path)
    if category not in ALLOWED_CATEGORIES | FORBIDDEN_CATEGORIES:
        raise ValueError(f"Unknown naming residual category: {category}")
    allowed = category in ALLOWED_CATEGORIES
    public_facing = category in {"forbidden_public_text", "forbidden_api_or_report"}
    runtime_reachable = relative_path.startswith("Libs/") and category != "allowed_test_fixture"
    return {
        "file": relative_path,
        "line": line_number,
        "matched_string": matched_string,
        "symbol_or_text": symbol_or_text,
        "runtime_reachable": str(runtime_reachable).lower(),
        "public_facing": str(public_facing).lower(),
        "category": category,
        "required_action": "preserve_approved_context" if allowed else "replace_with_canonical_identity",
        "canonical_replacement": canonical_replacement_for_legacy_symbol(matched_string, symbol_or_text),
        "compatibility_required": str(category in {"allowed_deprecated_alias", "allowed_checkpoint_migration"}).lower(),
        "owner": str(policy.get("owner", "ccf-integrity-auditor")),
        "status": "allowed" if allowed else "open",
    }


def scan_naming_residuals(
    config: Mapping[str, object],
    repo_root: Path = REPO_ROOT,
) -> list[dict[str, object]]:
    policy = _policy(config)
    rows: list[dict[str, object]] = []
    for path in _discover_files(config, repo_root):
        relative_path = _relative(path, repo_root)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        union_lines = _union_compatibility_lines(text) if relative_path == "Libs/configs/validation/iclr_design_contract_union.yaml" else set()
        for match in LEGACY_PATTERN.finditer(relative_path):
            rows.append(
                _build_row(relative_path, 0, match.group(0), relative_path, policy, union_lines)
                )
        for match in NUMBERED_ROUTE_PATTERN.finditer(relative_path):
            row = _build_numbered_row(relative_path, 0, match.group(0), relative_path, policy)
            if row is not None:
                rows.append(row)
        for line_number, line in enumerate(text.splitlines(), start=1):
            for match in LEGACY_PATTERN.finditer(line):
                rows.append(
                    _build_row(relative_path, line_number, match.group(0), line.strip(), policy, union_lines)
                )
            for match in NUMBERED_ROUTE_PATTERN.finditer(line):
                row = _build_numbered_row(relative_path, line_number, match.group(0), line.strip(), policy)
                if row is not None:
                    rows.append(row)
    return rows


def scan_naming_residuals_at_revision(
    config: Mapping[str, object],
    revision: str,
    repo_root: Path = REPO_ROOT,
) -> list[dict[str, object]]:
    policy = dict(_policy(config))
    policy["legacy_import_shim_modules"] = []
    tree_result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", revision],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    all_paths = tree_result.stdout.splitlines()
    scan_paths = [str(value).rstrip("/") for value in policy.get("scan_paths", [])]
    excluded = [str(value).rstrip("/") for value in policy.get("excluded_path_prefixes", [])]
    suffixes = {str(value) for value in policy.get("text_suffixes", [])}

    def selected(path: str) -> bool:
        in_scope = any(path == root or path.startswith(f"{root}/") for root in scan_paths)
        is_excluded = any(path == prefix or path.startswith(f"{prefix}/") for prefix in excluded)
        return in_scope and not is_excluded and (not suffixes or Path(path).suffix in suffixes)

    selected_paths = sorted(path for path in all_paths if selected(path))
    union_path = "Libs/configs/validation/iclr_design_contract_union.yaml"
    union_result = subprocess.run(
        ["git", "show", f"{revision}:{union_path}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    union_lines = _union_compatibility_lines(union_result.stdout)
    rows: list[dict[str, object]] = []
    for relative_path in selected_paths:
        for match in LEGACY_PATTERN.finditer(relative_path):
            rows.append(_build_row(relative_path, 0, match.group(0), relative_path, policy, union_lines))
    grep_result = subprocess.run(
        ["git", "grep", "-n", "-I", "-i", "-E", r"mind[-_ ]?dino", revision, "--", *scan_paths],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if grep_result.returncode not in {0, 1}:
        raise RuntimeError(grep_result.stderr.strip() or "git grep failed")
    for output_line in grep_result.stdout.splitlines():
        _, relative_path, line_value, line = output_line.split(":", 3)
        if relative_path not in selected_paths:
            continue
        line_number = int(line_value)
        allowed_union_lines = union_lines if relative_path == union_path else set()
        for match in LEGACY_PATTERN.finditer(line):
            rows.append(_build_row(relative_path, line_number, match.group(0), line.strip(), policy, allowed_union_lines))
    return rows


def _row_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("file", "")),
        str(row.get("matched_string", "")),
        str(row.get("symbol_or_text", "")),
    )


def reconcile_inventory(
    baseline_rows: list[dict[str, str]],
    current_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    current_counts = Counter((_row_key(row), str(row.get("category", ""))) for row in current_rows)
    reconciled: list[dict[str, object]] = []
    for baseline in baseline_rows:
        row: dict[str, object] = dict(baseline)
        key = (_row_key(row), str(row.get("category", "")))
        if current_counts[key] > 0:
            current_counts[key] -= 1
            row["status"] = "allowed" if str(row.get("category")) in ALLOWED_CATEGORIES else "open"
        else:
            row["status"] = "removed" if str(row.get("category")) in ALLOWED_CATEGORIES else "remediated"
        reconciled.append(row)
    remaining = Counter(current_counts)
    for row in current_rows:
        key = (_row_key(row), str(row.get("category", "")))
        if remaining[key] > 0:
            reconciled.append(dict(row))
            remaining[key] -= 1
    return reconciled


def write_baseline_inventory(
    config: Mapping[str, object],
    output_path: Path,
    baseline_revision: str,
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    rows = scan_naming_residuals_at_revision(config, baseline_revision, repo_root)
    write_tsv(output_path, INVENTORY_COLUMNS, rows)
    category_counts = Counter(str(row["category"]) for row in rows)
    return {
        "status": "inventory_complete",
        "baseline_revision": baseline_revision,
        "output_path": _relative(output_path, repo_root),
        "row_count": len(rows),
        "forbidden_count": sum(category_counts[category] for category in FORBIDDEN_CATEGORIES),
        "allowed_count": sum(category_counts[category] for category in ALLOWED_CATEGORIES),
        "category_counts": dict(sorted(category_counts.items())),
        "sha256": _sha256(output_path),
    }


def _compatibility_contract_violations(
    config: Mapping[str, object],
    current_rows: list[dict[str, object]],
    repo_root: Path,
) -> list[dict[str, object]]:
    policy = _policy(config)
    contract = _mapping(policy.get("compatibility_contract"))
    required_tokens = [str(value) for value in contract.get("required_metadata_tokens", [])]
    warning_token = str(contract.get("required_warning_token", ""))
    compatibility_files = sorted(
        {
            str(row["file"])
            for row in current_rows
            if str(row["file"]).startswith(("Libs/compat/", "Libs/migrations/"))
            and str(row["category"]) in {"allowed_deprecated_alias", "allowed_checkpoint_migration"}
        }
    )
    violations: list[dict[str, object]] = []
    for relative_path in compatibility_files:
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        missing_tokens = [token for token in required_tokens if token not in text]
        if missing_tokens:
            violations.append(
                {
                    "code": "compatibility_metadata_missing",
                    "file": relative_path,
                    "missing_tokens": missing_tokens,
                }
            )
        if warning_token and warning_token not in text:
            violations.append(
                {
                    "code": "compatibility_warning_missing",
                    "file": relative_path,
                    "missing_token": warning_token,
                }
            )
    test_path = repo_root / str(contract.get("required_test_path", ""))
    if not test_path.is_file():
        violations.append(
            {
                "code": "compatibility_test_missing",
                "file": str(contract.get("required_test_path", "")),
            }
        )
    return violations


def build_naming_remediation_report(
    config: Mapping[str, object],
    config_path: Path,
    baseline_inventory_path: Path,
    inventory_output_path: Path,
    residual_output_path: Path,
    report_json_path: Path,
    report_markdown_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    baseline_revision = str(_policy(config).get("baseline_revision", ""))
    baseline_rows = scan_naming_residuals_at_revision(config, baseline_revision, repo_root)
    baseline_row_digest = hashlib.sha256(
        json.dumps(baseline_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    current_rows = scan_naming_residuals(config, repo_root)
    inventory_rows = reconcile_inventory(baseline_rows, current_rows)
    write_tsv(inventory_output_path, INVENTORY_COLUMNS, inventory_rows)
    inventory_output_hash = _sha256(inventory_output_path)
    write_tsv(residual_output_path, INVENTORY_COLUMNS, current_rows)
    current_counts = Counter(str(row["category"]) for row in current_rows)
    forbidden_counts = {category: current_counts[category] for category in sorted(FORBIDDEN_CATEGORIES)}
    allowed_counts = {category: current_counts[category] for category in sorted(ALLOWED_CATEGORIES)}
    forbidden_count = sum(forbidden_counts.values())
    compatibility_violations = _compatibility_contract_violations(config, current_rows, repo_root)
    passed = forbidden_count == 0 and not compatibility_violations
    report: dict[str, object] = {
        "schema_version": "neuronsifter-g0-naming-remediation-v1",
        "status": "pass" if passed else "blocked",
        "blocker_id": "G0-NAMING-LEGACY-RESIDUALS",
        "original_blocker_closed": passed,
        "config_path": _relative(config_path, repo_root),
        "config_entrypoint_hash": _sha256(config_path),
        "config_hash": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest(),
        "git_commit": _git_commit(repo_root),
        "producer_command": "python Libs/scripts/validators/validate_g0_legacy_naming.py --config Libs/configs/config_stack.yaml",
        "baseline_inventory_path": _relative(baseline_inventory_path, repo_root),
        "baseline_row_digest_sha256": baseline_row_digest,
        "baseline_revision": baseline_revision,
        "baseline_row_count": len(baseline_rows),
        "current_residual_count": len(current_rows),
        "forbidden_residual_count": forbidden_count,
        "forbidden_category_counts": forbidden_counts,
        "allowed_category_counts": allowed_counts,
        "allowed_context_policy": "explicit_path_allowlist_with_structural_union_compatibility_block",
        "inventory_output_path": _relative(inventory_output_path, repo_root),
        "inventory_output_hash": inventory_output_hash,
        "residual_output_path": _relative(residual_output_path, repo_root),
        "compatibility_policy": {
            "current_public_identity": "NeuronSifter",
            "legacy_output_allowed": False,
            "duplicate_current_registration_allowed": False,
            "contract_violations": compatibility_violations,
        },
        "blockers": [],
        "blocker_count": 0 if passed else 1,
    }
    if not passed:
        report["blockers"] = [
            {
                "blocker_id": "G0-NAMING-LEGACY-RESIDUALS",
                "reason": "forbidden residuals or incomplete compatibility contracts remain",
                "forbidden_count": forbidden_count,
                "compatibility_violation_count": len(compatibility_violations),
            }
        ]
    write_json(report_json_path, report)
    markdown = (
        "# G0 Naming Remediation Report\n\n"
        f"- Status: `{report['status']}`\n"
        f"- Immutable blocker: `{report['blocker_id']}`\n"
        f"- Baseline inventory rows: `{report['baseline_row_count']}`\n"
        f"- Current residuals: `{report['current_residual_count']}`\n"
        f"- Forbidden residuals: `{report['forbidden_residual_count']}`\n"
        f"- Allowed residual policy: `{report['allowed_context_policy']}`\n"
    )
    write_text_atomic(report_markdown_path, markdown)
    return report


def report_json(report: Mapping[str, object]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=True, sort_keys=False)
