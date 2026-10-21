from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from Libs.utils.data.round11e_processed_input_linkage import INPUT_ROOT_ENV
from Libs.utils.data.round11e_processed_input_linkage import MANIFEST_ENV
from Libs.utils.data.round11e_processed_input_linkage import MANIFEST_SHA256_ENV
from Libs.utils.data.round11e_processed_input_linkage import build_bwrap_command
from Libs.utils.data.round11e_processed_input_linkage import build_processed_input_manifest
from Libs.utils.data.round11e_processed_input_linkage import load_processed_input_linkage
from Libs.utils.data.round11e_processed_input_linkage import normalize_logical_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    input_root = tmp_path / "processed"
    source_path = input_root / "simulation_audit" / "simulation_qc_summary.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text('{"status":"available"}\n', encoding="utf-8")
    logical_path = "Input/processed/simulation_audit/simulation_qc_summary.json"
    manifest = build_processed_input_manifest(
        input_root,
        [logical_path],
        discovery_trace_label="Output/audits/discovery.strace",
        discovery_trace_sha256="a" * 64,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    environment = {
        INPUT_ROOT_ENV: str(input_root),
        MANIFEST_ENV: str(manifest_path),
        MANIFEST_SHA256_ENV: _sha256(manifest_path),
    }
    return environment, input_root, source_path


def test_processed_input_linkage_requires_explicit_environment(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=INPUT_ROOT_ENV):
        load_processed_input_linkage({})


def test_processed_input_linkage_binds_only_hash_verified_entries(tmp_path: Path) -> None:
    environment, input_root, source_path = _manifest_environment(tmp_path)
    linkage = load_processed_input_linkage(environment)

    assert linkage.input_root == input_root.resolve()
    assert linkage.entries[0].logical_path == "Input/processed/simulation_audit/simulation_qc_summary.json"
    assert linkage.entries[0].physical_path == source_path.resolve()

    source_path.write_text('{"status":"different"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_processed_input_linkage(environment)


def test_processed_input_linkage_rejects_escaped_and_unbound_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repository-relative"):
        normalize_logical_path("Input/processed/../outside.tsv")
    with pytest.raises(ValueError, match="below Input/processed"):
        normalize_logical_path("Input/raw/source.tsv")

    input_root = tmp_path / "processed"
    input_root.mkdir()
    with pytest.raises(FileNotFoundError, match="unavailable"):
        build_processed_input_manifest(
            input_root,
            ["Input/processed/not_allowlisted.tsv"],
            discovery_trace_label="trace",
            discovery_trace_sha256="b" * 64,
        )


def test_bwrap_command_hides_source_root_and_maps_the_verified_destination(tmp_path: Path) -> None:
    environment, input_root, _ = _manifest_environment(tmp_path)
    linkage = load_processed_input_linkage(environment)
    worktree = tmp_path / "worktree"
    tracked_structure = worktree / "Input/structures/predicted/model.pdb"
    tracked_structure.parent.mkdir(parents=True)
    tracked_structure.write_text("MODEL\n", encoding="utf-8")

    command = build_bwrap_command(worktree, linkage, ["python", "-V"])

    destination = str(worktree / "Input/processed/simulation_audit/simulation_qc_summary.json")
    assert ["--ro-bind", str(linkage.entries[0].physical_path), destination] == command[
        command.index("--ro-bind", command.index("--ro-bind") + 1) : command.index("--ro-bind", command.index("--ro-bind") + 1) + 3
    ]
    assert ["--tmpfs", str(input_root.resolve())] in [command[index : index + 2] for index in range(len(command) - 1)]
    assert ["--tmpfs", str(worktree / "Input/processed")] in [
        command[index : index + 2] for index in range(len(command) - 1)
    ]
    assert str(tracked_structure) not in command
    assert "--unsetenv" in command
    unset_values = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--unsetenv"]
    assert "PYTHONPATH" in unset_values
