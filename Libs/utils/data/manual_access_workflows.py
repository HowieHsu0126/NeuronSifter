from __future__ import annotations

from pathlib import Path

from Libs.utils.data.raw_data_registry import ensure_directory, read_tsv, resolve_repo_path, write_json, write_tsv


def manual_workflow(config: dict, source_key: str) -> dict | None:
    workflows = config.get("manual_source_workflows", {})
    shared = dict(workflows.get("shared", {}))
    source_specific = workflows.get("sources", {}).get(source_key)
    if source_specific is None:
        return None
    merged = dict(shared)
    merged.update(source_specific)
    merged["requirements"] = list(source_specific.get("requirements", []))
    return merged


def manual_drop_root(source_root: Path, workflow: dict) -> Path:
    return source_root / workflow["drop_relative_dir"]


def manual_manifest_path(source_root: Path, workflow: dict) -> Path:
    return source_root / workflow["manifest_relative_path"]


def manual_checklist_path(source_root: Path, workflow: dict) -> Path:
    return source_root / workflow["checklist_relative_path"]


def manual_expected_root_text(source_key: str, workflow: dict) -> str:
    return f"Input/raw/{source_key}/{workflow['drop_relative_dir']}"


def required_pattern_for_requirement(requirement: dict) -> str:
    return str(requirement.get("required_artifact_pattern", ""))


def requirement_for_asset(workflow: dict | None, source_accession: str, asset_name: str) -> dict | None:
    if workflow is None:
        return None
    exact_match = None
    accession_match = None
    asset_match = None
    for requirement in workflow["requirements"]:
        same_accession = requirement.get("source_accession", "") == source_accession
        same_asset = requirement.get("asset_name", "") == asset_name
        if same_accession and same_asset:
            exact_match = requirement
            break
        if same_accession and requirement.get("asset_name", "") == "manual_collection_gate":
            accession_match = requirement
        if requirement.get("asset_name", "") == asset_name and not requirement.get("source_accession"):
            asset_match = requirement
    return exact_match or accession_match or asset_match


def checklist_payload(source_key: str, source_config: dict, workflow: dict) -> dict[str, object]:
    return {
        "source_key": source_key,
        "source_name": source_config["source_name"],
        "source_url": source_config["source_url"],
        "manual_drop_expected_root": manual_expected_root_text(source_key, workflow),
        "ingest_script": "Libs/scripts/data_acquisition/ingest_manual_source_drop.py",
        "requirements": workflow["requirements"],
    }


def manifest_template_rows(workflow: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for requirement in workflow["requirements"]:
        rows.append(
            {
                "source_accession": requirement.get("source_accession", ""),
                "asset_name": requirement.get("asset_name", ""),
                "relative_path": "",
                "asset_url": "",
                "license_or_terms_acknowledged_flag": "no",
                "note": requirement.get("description", ""),
            }
        )
    return rows


def ensure_manual_workflow_artifacts(
    source_key: str,
    source_config: dict,
    config: dict,
) -> None:
    workflow = manual_workflow(config, source_key)
    if workflow is None:
        return
    source_root = resolve_repo_path(source_config["local_root"])
    drop_root = manual_drop_root(source_root, workflow)
    ensure_directory(drop_root)

    manifest_path = manual_manifest_path(source_root, workflow)
    if not manifest_path.exists():
        write_tsv(manifest_path, config["schemas"]["manual_drop_manifest_fields"], manifest_template_rows(workflow))
    else:
        existing_rows = read_tsv(manifest_path)
        if not existing_rows:
            write_tsv(manifest_path, config["schemas"]["manual_drop_manifest_fields"], manifest_template_rows(workflow))

    checklist_path = manual_checklist_path(source_root, workflow)
    if not checklist_path.exists():
        write_json(checklist_path, checklist_payload(source_key, source_config, workflow))
