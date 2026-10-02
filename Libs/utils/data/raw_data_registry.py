from __future__ import annotations

import builtins
import csv
import hashlib
import json
import os
import re
import uuid
import warnings
from datetime import date
from pathlib import Path
from typing import Iterable

import requests
import yaml


REQUEST_TIMEOUT_SECONDS = 120
REQUEST_HEADERS = {
    "User-Agent": "NeuronSifterDataRegistry/2026-04-10",
}
HTTP_SESSION = requests.Session()
HTTP_SESSION.trust_env = True
HTTP_SESSION.headers.update(REQUEST_HEADERS)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def deep_merge_dicts(base_payload: dict, override_payload: dict) -> dict:
    merged = dict(base_payload)
    for key, value in override_payload.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
            continue
        merged[key] = value
    return merged


def _resolve_config_path(config_path: str | Path, parent_path: Path | None = None) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path
    if parent_path is not None:
        candidate = (parent_path.parent / path).resolve()
        if candidate.exists():
            return candidate
    repo_candidate = (repo_root() / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (Path.cwd() / path).resolve()


def _config_path_label(config_path: Path) -> str:
    resolved_path = config_path.resolve()
    root = repo_root().resolve()
    if resolved_path.is_relative_to(root):
        return resolved_path.relative_to(root).as_posix()
    return str(resolved_path)


def _record_leaf_owners(
    payload: object,
    owner: str,
    leaf_owners: dict[str, list[str]],
    path_parts: tuple[str, ...] = (),
) -> None:
    if isinstance(payload, dict) and payload:
        for key, value in payload.items():
            _record_leaf_owners(value, owner, leaf_owners, (*path_parts, str(key)))
        return
    path = ".".join(path_parts)
    if path:
        leaf_owners.setdefault(path, []).append(owner)


def _load_yaml_payload(
    config_path: Path,
    *,
    active_stack: list[Path],
    seen_paths: set[Path],
    include_order: list[str],
    include_graph: dict[str, list[str]],
    leaf_owners: dict[str, list[str]],
) -> dict:
    resolved_path = config_path.resolve()
    path_label = _config_path_label(resolved_path)
    if resolved_path in active_stack:
        cycle_start = active_stack.index(resolved_path)
        cycle = [*active_stack[cycle_start:], resolved_path]
        cycle_text = " -> ".join(_config_path_label(path) for path in cycle)
        raise ValueError(f"Configuration include cycle detected: {cycle_text}")
    if resolved_path in seen_paths:
        raise ValueError(f"Configuration include path is repeated: {path_label}")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Configuration include is missing: {path_label}")

    seen_paths.add(resolved_path)
    active_stack.append(resolved_path)
    include_order.append(path_label)
    try:
        with resolved_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise TypeError(f"Configuration root must be a mapping: {resolved_path}")
        raw_includes = payload.pop("config_includes", [])
        if not isinstance(raw_includes, list):
            raise TypeError(f"config_includes must be a list: {resolved_path}")
        include_paths = [str(value) for value in raw_includes]
        include_graph[path_label] = []
        merged_payload: dict = {}
        for include_value in include_paths:
            include_path = _resolve_config_path(include_value, resolved_path)
            include_graph[path_label].append(_config_path_label(include_path))
            include_payload = _load_yaml_payload(
                include_path,
                active_stack=active_stack,
                seen_paths=seen_paths,
                include_order=include_order,
                include_graph=include_graph,
                leaf_owners=leaf_owners,
            )
            merged_payload = deep_merge_dicts(merged_payload, include_payload)
        _record_leaf_owners(payload, path_label, leaf_owners)
        return deep_merge_dicts(merged_payload, payload)
    finally:
        active_stack.pop()



def warning_category(category_name: str) -> type[Warning]:
    category = getattr(builtins, category_name)
    if not isinstance(category, type) or not issubclass(category, Warning):
        raise ValueError(f"Warning policy category must name a Warning subclass: {category_name}")
    return category


def apply_warning_policy(config: dict) -> None:
    policy = config.get("runtime_warning_policy")
    if policy is None:
        raise ValueError("Configuration is missing runtime_warning_policy")
    if not isinstance(policy, dict):
        raise TypeError("runtime_warning_policy must be a mapping")
    enabled = bool(policy["enabled"])
    if not enabled:
        return
    action = str(policy["action"])
    category = warning_category(str(policy["category"]))
    warnings.simplefilter(action, category)
    for filter_config in policy.get("external_package_filters", []):
        warnings.filterwarnings(
            str(filter_config["action"]),
            message=str(filter_config["message_regex"]),
            category=warning_category(str(filter_config["category"])),
            module=str(filter_config["module_regex"]),
        )

def load_config_with_provenance(config_path: str | Path) -> tuple[dict, dict[str, object]]:
    config_file = _resolve_config_path(config_path)
    include_order: list[str] = []
    include_graph: dict[str, list[str]] = {}
    leaf_owners: dict[str, list[str]] = {}
    config = _load_yaml_payload(
        config_file,
        active_stack=[],
        seen_paths=set(),
        include_order=include_order,
        include_graph=include_graph,
        leaf_owners=leaf_owners,
    )
    apply_warning_policy(config)
    provenance = {
        "entrypoint": _config_path_label(config_file),
        "include_order": include_order,
        "include_graph": include_graph,
        "leaf_owners": leaf_owners,
    }
    return config, provenance


def load_config(config_path: str | Path) -> dict:
    config, _ = load_config_with_provenance(config_path)
    return config


def resolve_repo_path(path_value: str) -> Path:
    return (repo_root() / path_value).resolve()


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def today_iso() -> str:
    return date.today().isoformat()


def temporary_output_path(path: Path) -> Path:
    return path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.part"


def write_json(path: Path, payload: object) -> None:
    ensure_directory(path.parent)
    temporary_path = temporary_output_path(path)
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, ensure_ascii=True)
        handle.write("\n")
    temporary_path.replace(path)


def write_text_atomic(path: Path, content: str) -> None:
    ensure_directory(path.parent)
    temporary_path = temporary_output_path(path)
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    ensure_directory(path.parent)
    temporary_path = temporary_output_path(path)
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temporary_path.replace(path)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def http_get(
    url: str,
    params: dict[str, object] | None = None,
    stream: bool = False,
    raise_for_status: bool = True,
) -> requests.Response:
    response = HTTP_SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS, stream=stream)
    if raise_for_status:
        response.raise_for_status()
    return response


def download_to_file(url: str, destination: Path) -> None:
    ensure_directory(destination.parent)
    temporary_path = destination.parent / f"{destination.name}.part"
    response = http_get(url, stream=True)
    try:
        with temporary_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
    finally:
        response.close()
    temporary_path.replace(destination)


def normalize_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def slugify(value: str) -> str:
    value = normalize_text(value)
    return value.replace(" ", "_")


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def infer_tissue_or_biofluid(text: str, keyword_map: dict[str, list[str]]) -> str:
    for label, keywords in keyword_map.items():
        if contains_any(text, keywords):
            return label
    return ""


def infer_sample_context(text: str, mechanistic_keywords: list[str], patient_keywords: list[str], species: str) -> str:
    normalized_species = normalize_text(species)
    if "homo sapiens" not in normalized_species and "human" not in normalized_species:
        return "mechanistic_model"
    if contains_any(text, mechanistic_keywords):
        return "mechanistic_model"
    if contains_any(text, patient_keywords):
        return "patient_cohort"
    return "patient_cohort"


def choose_diagnosis_label(text: str, query_terms: list[str]) -> str:
    normalized_text = normalize_text(text)
    for query_term in query_terms:
        if normalize_text(query_term) in normalized_text:
            return query_term
    return ""


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_checksums(source_root: Path, checksum_relative_path: str) -> None:
    checksum_path = source_root / checksum_relative_path
    rows: list[dict[str, object]] = []
    for file_path in sorted(source_root.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path == checksum_path:
            continue
        if file_path.name.endswith(".part"):
            continue
        rows.append(
            {
                "relative_path": file_path.relative_to(source_root).as_posix(),
                "sha256": sha256_of_file(file_path),
                "size_bytes": file_path.stat().st_size,
            }
        )
    write_tsv(checksum_path, ["relative_path", "sha256", "size_bytes"], rows)


def chunked(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]
