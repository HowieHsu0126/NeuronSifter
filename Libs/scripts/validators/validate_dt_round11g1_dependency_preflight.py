"""Validate Round-11G1 dependency closure before targeted pytest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import write_json  # noqa: E402
from Libs.utils.data.round11g_execution_inputs import load_execution_input_bundle  # noqa: E402
from Libs.utils.data.round11g_execution_inputs import resolve_execution_input  # noqa: E402
from Libs.utils.data.round11g_execution_inputs import run_execution_namespace_smoke  # noqa: E402
from Libs.utils.data.round11g_execution_inputs import sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Round-11G1 dependency-only preflight Gate.")
    parser.add_argument("--worktree-root", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--schema", default="Libs/configs/schema/round11g_execution_input_bundle.schema.json")
    parser.add_argument("--output", default="Output/audits/dt_round11g1_dependency_preflight.json")
    parser.add_argument("--resolution-output", default="Output/audits/dt_round11g1_input_resolution_validation.json")
    parser.add_argument(
        "--tracked-input-probe",
        default="Input/structures/predicted/APP_P05067_alphafold_v6.pdb",
    )
    parser.add_argument("--unlisted-processed-probe", required=True)
    return parser.parse_args()


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def main() -> int:
    args = parse_args()
    worktree_root = Path(args.worktree_root).resolve()
    output_path = (REPO_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    resolution_path = (
        (REPO_ROOT / args.resolution_output).resolve()
        if not Path(args.resolution_output).is_absolute()
        else Path(args.resolution_output)
    )
    schema_path = worktree_root / args.schema
    config_path = worktree_root / "Libs/configs/config_stack.yaml"
    blockers: list[dict[str, object]] = []
    bundle_path = Path(os.environ.get("NRNSIFTER_EXECUTION_INPUT_BUNDLE", ""))
    if not bundle_path.is_file():
        blockers.append({"code": "blocked_unknown_provenance", "message": "Execution input bundle is unavailable."})
        bundle_payload: dict[str, object] = {}
    else:
        bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle_payload:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema_errors = sorted(Draft202012Validator(schema).iter_errors(bundle_payload), key=lambda error: list(error.path))
        blockers.extend(
            {
                "code": "blocked_unknown_provenance",
                "message": error.message,
                "path": list(error.path),
            }
            for error in schema_errors
        )
    bundle = None
    resolution_rows: list[dict[str, object]] = []
    if not blockers:
        try:
            bundle = load_execution_input_bundle(
                config_stack_path=config_path,
                expected_code_commit=args.expected_code_commit,
            )
            for asset in bundle.assets:
                resolve_execution_input(
                    bundle,
                    config_path,
                    asset.asset_uid,
                    expected_role=asset.allowed_execution_role,
                )
                resolution_rows.append(
                    {
                        "asset_uid": asset.asset_uid,
                        "canonical_uri": asset.canonical_uri,
                        "role": asset.allowed_execution_role,
                        "content_hash": asset.content_hash,
                        "size": asset.size,
                        "producer_uid": asset.producer_uid,
                        "access_status": asset.license_or_access_status,
                        "resolution_status": "passed",
                        "physical_path_role": (
                            "processed_input_root" if asset.allowed_execution_role == "processed_test_input" else "materialized_audit_root"
                        ),
                    }
                )
        except PermissionError as error:
            blockers.append({"code": "blocked_license_or_access", "message": str(error)})
        except FileNotFoundError as error:
            blockers.append({"code": "blocked_missing_processed_input", "message": str(error)})
        except ValueError as error:
            message = str(error)
            code = "blocked_hash_mismatch" if "hash mismatch" in message else "blocked_unknown_provenance"
            blockers.append({"code": code, "message": message})
    observed_commit = _git_output(worktree_root, "rev-parse", "HEAD")
    dirty_status_before = _git_output(worktree_root, "status", "--porcelain=v1")
    if observed_commit != args.expected_code_commit:
        blockers.append(
            {
                "code": "blocked_hidden_root_dependency",
                "message": "Execution worktree commit does not match the input bundle identity.",
            }
        )
    if dirty_status_before:
        blockers.append(
            {
                "code": "blocked_hidden_root_dependency",
                "message": "Execution worktree is dirty before dependency preflight.",
                "paths": dirty_status_before.splitlines(),
            }
        )
    if shutil.which("bwrap") is None:
        blockers.append({"code": "blocked_environment_dependency", "message": "bubblewrap is unavailable."})
    source_text = (worktree_root / "Libs/utils/data/round11g_execution_inputs.py").read_text(encoding="utf-8")
    forbidden_patterns = ["rglob(", "glob(\"**", "parents[", "latest artifact", "first matching"]
    observed_forbidden = [pattern for pattern in forbidden_patterns if pattern in source_text]
    if observed_forbidden:
        blockers.append(
            {
                "code": "blocked_hidden_root_dependency",
                "message": "Execution input resolver contains a forbidden fallback-search pattern.",
                "patterns": observed_forbidden,
            }
        )

    namespace_smoke: dict[str, object] = {"status": "not_run", "checks": {}}
    if bundle is not None and not blockers:
        try:
            namespace_smoke = run_execution_namespace_smoke(
                worktree_root,
                bundle,
                tracked_input_path=args.tracked_input_probe,
                unlisted_processed_path=args.unlisted_processed_probe,
            )
            if namespace_smoke["status"] != "passed":
                blockers.append(
                    {
                        "code": "blocked_hidden_root_dependency",
                        "message": "Execution namespace smoke did not preserve the required visibility boundary.",
                        "checks": namespace_smoke.get("checks", {}),
                    }
                )
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            blockers.append(
                {
                    "code": "blocked_environment_dependency",
                    "message": f"Execution namespace smoke could not run: {type(error).__name__}",
                }
            )
    dirty_status_after = _git_output(worktree_root, "status", "--porcelain=v1")
    if dirty_status_after:
        blockers.append(
            {
                "code": "blocked_hidden_root_dependency",
                "message": "Execution worktree is dirty after dependency preflight.",
                "paths": dirty_status_after.splitlines(),
            }
        )

    resolution_report = {
        "schema_version": "dt-round11g1-input-resolution-validation-v1",
        "status": "passed" if not blockers else "blocked",
        "bundle_sha256": sha256_file(bundle_path) if bundle_path.is_file() else None,
        "resolved_asset_count": len(resolution_rows),
        "duplicate_logical_uid_count": 0,
        "hash_mismatch_count": 0 if not any(row["code"] == "blocked_hash_mismatch" for row in blockers) else 1,
        "unresolved_access_count": 0 if not any(row["code"] == "blocked_license_or_access" for row in blockers) else 1,
        "hidden_root_fallback_count": len(observed_forbidden),
        "namespace_smoke": namespace_smoke,
        "assets": resolution_rows,
        "blockers": blockers,
    }
    write_json(resolution_path, resolution_report)
    state = "dependency_closure_passed" if not blockers else str(blockers[0]["code"])
    preflight = {
        "schema_version": "dt-round11g1-dependency-preflight-v1",
        "state": state,
        "dependency_closure_passed": not blockers,
        "code_commit": observed_commit,
        "expected_code_commit": args.expected_code_commit,
        "execution_worktree_clean": not dirty_status_before and not dirty_status_after,
        "bundle_uid": bundle.bundle_uid if bundle is not None else None,
        "bundle_sha256": bundle.manifest_sha256 if bundle is not None else None,
        "asset_count": len(bundle.assets) if bundle is not None else 0,
        "missing_required_assets": 0 if not blockers else None,
        "unresolved_provenance": 0 if not blockers else None,
        "hash_mismatches": 0 if not blockers else None,
        "duplicate_logical_asset_uids": 0 if not blockers else None,
        "external_asset_substitution_count": 0,
        "hidden_dirty_root_dependency_count": len(observed_forbidden),
        "environment_dependencies_available": shutil.which("bwrap") is not None,
        "namespace_smoke": namespace_smoke,
        "namespace_smoke_passed": namespace_smoke.get("status") == "passed",
        "blockers": blockers,
    }
    write_json(output_path, preflight)
    print(json.dumps({"state": state, "blocker_count": len(blockers), "asset_count": preflight["asset_count"]}, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
