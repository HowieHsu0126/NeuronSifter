"""Fail-closed contracts for the controlled-world durable release."""

from pathlib import Path
import subprocess

import pytest

from Libs.scripts.validators.validate_controlled_world_benchmark import _durable_release_paths, _validate_durable_release


def test_durable_inventory_is_closed() -> None:
    paths = {
        "training_manifest": "a", "development_manifest": "b", "calibration_manifest": "c",
        "test_manifest": "d", "ood_manifest": "e", "development_headroom": "f",
        "freeze_receipt": "g", "qualification_audit": "h/q.json", "decision_report": "i",
        "access_receipt_directory": "j",
    }
    release = _durable_release_paths(Path("/tmp/root"), paths)
    assert len(release) == len(set(release)) == 14


def _paths() -> dict[str, object]:
    return {
        "training_manifest": "a", "development_manifest": "b", "calibration_manifest": "c",
        "test_manifest": "d", "ood_manifest": "e", "development_headroom": "f",
        "freeze_receipt": "g", "qualification_audit": "h/q.json", "decision_report": "i",
        "access_receipt_directory": "j",
    }


def _validate(root: Path, paths: dict[str, object]) -> None:
    _validate_durable_release(
        repository_root=root, artifact_root=root, paths=paths, config={}, manifests={},
        ood={}, headroom={}, freeze={}, qualification={},
    )


def test_durable_release_rejects_distinct_artifact_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="artifact_root == repository_root"):
        _validate_durable_release(
            repository_root=tmp_path / "repo", artifact_root=tmp_path / "artifacts", paths={},
            config={}, manifests={}, ood={}, headroom={}, freeze={}, qualification={},
        )


def test_durable_release_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="durable release file missing"):
        _validate(tmp_path, _paths())


def test_durable_release_rejects_untracked_file(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    first = _durable_release_paths(tmp_path, _paths())[0]
    first.write_text("untracked\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not Git-tracked"):
        _validate(tmp_path, _paths())


def test_durable_inventory_identifies_exact_files(tmp_path: Path) -> None:
    paths = _paths()
    names = sorted(path.resolve().name for path in _durable_release_paths(tmp_path, paths))
    assert names == sorted([
        "a", "b", "c", "d", "e", "f", "g", "i",
        "q.json",
        "controlled_neural_world_benchmark_validation.json",
        "twin_qualification_unopened_access_receipt.json",
        "agent_decision_unopened_access_receipt.json",
        "component_mechanism_unopened_access_receipt.json",
        "coevolution_robustness_unopened_access_receipt.json",
    ])
