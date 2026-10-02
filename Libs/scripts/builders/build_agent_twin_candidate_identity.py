"""Freeze and audit a Git-backed Agent-Twin co-evolution candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = "Output/audits/agent_twin_update_contract.json"
DEFAULT_OUTPUT = "Output/audits/agent_twin_candidate_identity.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(worktree: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(worktree), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_candidate_identity(candidate_worktree: Path, contract_relative_path: str) -> dict[str, object]:
    if not candidate_worktree.is_dir():
        raise FileNotFoundError(f"Candidate worktree does not exist: {candidate_worktree}")
    candidate_commit = git(candidate_worktree, "rev-parse", "HEAD")
    candidate_tree_hash = git(candidate_worktree, "rev-parse", "HEAD^{tree}")
    worktree_status = git(candidate_worktree, "status", "--porcelain=v1")
    if worktree_status:
        raise ValueError("Candidate worktree is not exact-clean.")
    contract_path = candidate_worktree / contract_relative_path
    if not contract_path.is_file():
        raise FileNotFoundError(f"Candidate update contract is missing: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("Candidate update contract must be a JSON object.")
    tracked_paths = git(candidate_worktree, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    return {
        "schema_version": "neuronsifter-agent-twin-candidate-identity-v1",
        "candidate_commit": candidate_commit,
        "candidate_tree_hash": candidate_tree_hash,
        "candidate_worktree": str(candidate_worktree.resolve()),
        "candidate_worktree_clean": True,
        "candidate_identity_verified": True,
        "tracked_file_count": len(tracked_paths),
        "update_contract": {
            "path": contract_relative_path,
            "sha256": sha256_file(contract_path),
            "status": contract.get("status"),
            "formal_execution_allowed": contract.get("formal_execution_allowed"),
            "final_gate_passed": contract.get("final_gate_passed"),
            "blockers": contract.get("blockers"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an exact-clean Agent-Twin candidate identity audit.")
    parser.add_argument("--candidate-worktree", required=True)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_candidate_identity(Path(args.candidate_worktree), args.contract)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
