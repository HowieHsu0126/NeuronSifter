#!/usr/bin/env python3
"""Materialize BASELINE_FIRST contract inputs into a clean, Git-bound candidate.

BASELINE_FIRST candidates are intentionally source-only, but runtime JSON schemas are
executable contract inputs rather than disposable generated payloads.  This
utility copies a small, predeclared set of such inputs from an authoritative
source tree, force-adds them to a new candidate commit, and records hashes.
It never rewrites an existing commit and refuses to operate on a dirty tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_CONTRACT_INPUTS = (
    "PROJECT_PREFIX",
    "Libs/configs/schema/agent_twin_wp2_development.schema.json",
    "Libs/configs/schema/baseline_first_contract.schema.json",
    "Libs/configs/schema/baseline_registry_reduction.schema.json",
    "Libs/configs/schema/public_foundation_corpus_storage_ledger.schema.json",
    "Libs/configs/schema/gate_writer_authority.schema.json",
    "Libs/configs/schema/phase1f_perturbation_pair_block.schema.json",
    "Libs/configs/schema/public_foundation_corpus_evidence.schema.json",
    "Libs/configs/schema/public_foundation_corpus_g2_m1_admission_audit.schema.json",
    "Libs/configs/schema/public_foundation_corpus.schema.json",
    "Libs/configs/schema/round11g_execution_input_bundle.schema.json",
    "Libs/configs/schema/evidence_union.schema.json",
)


class CandidateMaterializationError(RuntimeError):
    """Raised when the candidate cannot be materialized fail-closed."""


@dataclass(frozen=True)
class GitIdentity:
    commit: str
    tree: str


def _run_git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise CandidateMaterializationError(
            f"git {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def _assert_git_root(root: Path) -> None:
    if not root.is_dir():
        raise CandidateMaterializationError(f"candidate root is not a directory: {root}")
    if _run_git(root, "rev-parse", "--is-inside-work-tree") != "true":
        raise CandidateMaterializationError(f"not a Git worktree: {root}")


def _assert_clean(root: Path, label: str) -> None:
    status = _run_git(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise CandidateMaterializationError(f"{label} is dirty:\n{status}")


def _safe_relative(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute() or ".." in path.parts:
        raise CandidateMaterializationError(f"contract input escapes repository: {path_text}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(root: Path) -> GitIdentity:
    return GitIdentity(
        commit=_run_git(root, "rev-parse", "HEAD"),
        tree=_run_git(root, "rev-parse", "HEAD^{tree}"),
    )


def materialize(
    *,
    source_root: Path,
    candidate_root: Path,
    contract_inputs: Iterable[str] = DEFAULT_CONTRACT_INPUTS,
    commit_message: str = "BASELINE_FIRST: materialize runtime contract schemas",
    manifest_out: Path | None = None,
) -> dict:
    """Copy and commit contract inputs, returning a provenance manifest."""

    source_root = source_root.resolve()
    candidate_root = candidate_root.resolve()
    _assert_git_root(source_root)
    _assert_git_root(candidate_root)
    _assert_clean(source_root, "source root")
    _assert_clean(candidate_root, "candidate root")

    paths = tuple(_safe_relative(item) for item in contract_inputs)
    if not paths:
        raise CandidateMaterializationError("at least one contract input is required")

    records = []
    for relative in paths:
        source_path = source_root / relative
        candidate_path = candidate_root / relative
        if not source_path.is_file():
            raise CandidateMaterializationError(f"missing source contract input: {source_path}")
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, candidate_path)
        records.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256(source_path),
                "size_bytes": source_path.stat().st_size,
            }
        )

    _run_git(candidate_root, "add", "--force", "--", *(item.as_posix() for item in paths))
    staged = _run_git(candidate_root, "diff", "--cached", "--name-only").splitlines()
    expected = sorted(item.as_posix() for item in paths)
    if sorted(staged) != expected:
        raise CandidateMaterializationError(
            f"staged contract inputs differ; expected={expected}, observed={sorted(staged)}"
        )

    parent = _identity(candidate_root)
    _run_git(candidate_root, "commit", "--no-verify", "-m", commit_message)
    identity = _identity(candidate_root)
    _assert_clean(candidate_root, "candidate root after materialization")

    manifest = {
        "schema_version": "baseline_first_candidate_materialization_v1",
        "materialized_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "candidate_root": str(candidate_root),
        "parent_identity": {"commit": parent.commit, "tree": parent.tree},
        "candidate_identity": {"commit": identity.commit, "tree": identity.tree},
        "contract_inputs": records,
        "commit_message": commit_message,
    }
    if manifest_out is not None:
        manifest_out = manifest_out.resolve()
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--commit-message", default="BASELINE_FIRST: materialize runtime contract schemas")
    parser.add_argument("--contract-input", action="append", dest="contract_inputs")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        manifest = materialize(
            source_root=args.source_root,
            candidate_root=args.candidate_root,
            contract_inputs=args.contract_inputs or DEFAULT_CONTRACT_INPUTS,
            commit_message=args.commit_message,
            manifest_out=args.manifest_out,
        )
    except CandidateMaterializationError as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "materialized", **manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
