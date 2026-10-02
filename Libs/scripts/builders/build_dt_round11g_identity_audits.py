"""Build Round-11G root-scope and clean-execution identity records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.round11g_closure import (  # noqa: E402
    SCOPED_MANIFEST_FIELDS,
    build_round11f_identity_inventory,
    build_round11f_scoped_patch_manifest,
    build_round11g0_scoped_file_manifest,
    execution_identity,
    preserve_round11g0_scoped_changes,
    write_identity_audits,
)
from Libs.utils.data.raw_data_registry import write_tsv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument(
        "--include-execution-identity",
        action="store_true",
        help="Resolve config and capture an execution head only in a clean worktree.",
    )
    parser.add_argument(
        "--prepare-g0-scope",
        action="store_true",
        help="Write the authoritative scoped manifest and content-addressed recovery bundle.",
    )
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    inventory = build_round11f_identity_inventory(root)
    manifest = build_round11f_scoped_patch_manifest(root, inventory)
    recovery_status = "not_requested"
    if args.prepare_g0_scope:
        scoped_rows = build_round11g0_scoped_file_manifest(root)
        write_tsv(
            root / "Output/audits/dt_round11g0_scoped_file_manifest.tsv",
            SCOPED_MANIFEST_FIELDS,
            scoped_rows,
        )
        recovery_status = preserve_round11g0_scoped_changes(root, scoped_rows)["status"]
    if args.include_execution_identity:
        identity = execution_identity(root)
        write_identity_audits(root, inventory, manifest, identity)
        identity_status = identity["identity_status"]
    else:
        write_identity_audits(root, inventory, manifest, {"identity_status": "not_captured_root_is_not_execution_head"})
        identity_status = "not_captured_root_is_not_execution_head"
    print(
        json.dumps(
            {
                "root_dirty_status": inventory["root_dirty_status"],
                "identity_status": identity_status,
                "recovery_status": recovery_status,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
