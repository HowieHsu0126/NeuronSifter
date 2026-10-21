"""Deterministically replay the bounded Operational Twin dev/cal checkpoint."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader
import yaml

from Libs.exp.training.train_operational_twin_devcal import (
    EXPECTED_CLAIM_BOUNDARY,
    _canonical_hash,
    _seed_everything,
    _sha256_bytes,
    evaluate,
    require_operational_twin_training_authorized,
    resolve_operational_twin_run_contract,
    tracked_reference_identity,
)
from Libs.models.modules.world_action_encoder import WorldActionEncoder, world_action_schema_hash
from Libs.models.networks.operational_twin import OperationalTwin
from Libs.utils.data.controlled_world_tensorization import (
    ControlledWorldTensorDataset,
    controlled_manifest_sha256,
    tensorize_manifest,
)
from Libs.utils.exp_utils import resolve_project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "config",
        "training_manifest",
        "development_manifest",
        "calibration_manifest",
        "package_receipt",
        "open_data_qualification",
        "output_root",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--read-only", action="store_true")
    return parser.parse_args()


def _close(left: object, right: object, atol: float, rtol: float, path: str = "metrics") -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise AssertionError(f"Replay keys differ at {path}")
        for key in left:
            _close(left[key], right[key], atol, rtol, f"{path}.{key}")
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not torch.isclose(torch.tensor(float(left)), torch.tensor(float(right)), atol=atol, rtol=rtol):
            raise AssertionError(f"Replay mismatch at {path}: {left} != {right}")
    elif left != right:
        raise AssertionError(f"Replay mismatch at {path}: {left} != {right}")


def run(args: argparse.Namespace) -> dict[str, object]:
    root = resolve_project_root()
    contract = resolve_operational_twin_run_contract(args, root, require_output=False)
    config = contract["config"]
    cfg = contract["cfg"]
    paths = contract["paths"]
    require_operational_twin_training_authorized(cfg, paths)
    output_root = paths["output_root"]
    expected_hashes = json.loads((output_root / "hashes/data_manifest_hashes.json").read_text())
    manifest_paths = {name: paths[name] for name in ("training", "development", "calibration")}
    observed_hashes = {
        name: controlled_manifest_sha256(path, name)
        for name, path in manifest_paths.items()
    }
    if observed_hashes != expected_hashes or world_action_schema_hash() != (output_root / "hashes/world_action_schema_hash.txt").read_text().strip():
        raise AssertionError("Replay input or WorldAction schema hashes differ.")
    resolved_config = yaml.safe_dump(config, sort_keys=True)
    if _sha256_bytes(resolved_config.encode()) != (output_root / "hashes/config_hash.txt").read_text().strip():
        raise AssertionError("Replay config hash differs.")
    seed_hash = _canonical_hash({"seed": int(cfg["training"]["seed"]), "deterministic_kernels": cfg["training"]["deterministic_kernels"]})
    if seed_hash != (output_root / "hashes/seed_hash.txt").read_text().strip():
        raise AssertionError("Replay seed hash differs.")
    provenance = json.loads((output_root / "provenance/runtime_provenance.json").read_text())
    observed_receipt_identities = {
        "open_action_package_receipt": tracked_reference_identity(root, paths["open_action_package_receipt"]),
        "open_data_qualification": tracked_reference_identity(root, paths["future_open_data_qualification"]),
    }
    if provenance.get("reference_receipt_identities") != observed_receipt_identities:
        raise AssertionError("Replay receipt identities differ from the canonical committed Git-tree authority.")
    _seed_everything(int(cfg["training"]["seed"]), bool(cfg["training"]["deterministic_kernels"]))
    encoder = WorldActionEncoder()
    tensorized = {name: tensorize_manifest(path, name, config, root) for name, path in manifest_paths.items()}
    checkpoint = torch.load(output_root / "checkpoints/best_dev.pt", map_location="cpu", weights_only=False)
    model = OperationalTwin(len(checkpoint["feature_names"]), encoder.output_dim, checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    loaders = {name: DataLoader(ControlledWorldTensorDataset(value, encoder), batch_size=int(cfg["training"]["batch_size"]), shuffle=False) for name, value in tensorized.items()}
    dev = evaluate(model, loaders["development"], checkpoint["normalization_mean"], checkpoint["normalization_std"], torch.device("cpu"))
    calibration = evaluate(model, loaders["calibration"], checkpoint["normalization_mean"], checkpoint["normalization_std"], torch.device("cpu"), checkpoint["calibration_scale"])
    expected_dev = dict(checkpoint["dev_metrics"])
    expected_dev.pop("epoch", None)
    expected_dev.pop("training_nll", None)
    atol = float(cfg["replay"]["absolute_tolerance"])
    rtol = float(cfg["replay"]["relative_tolerance"])
    _close(dev, expected_dev, atol, rtol)
    expected_cal = dict(checkpoint["calibration_metrics"])
    expected_cal.pop("calibration_scale_by_family", None)
    _close(calibration, expected_cal, atol, rtol)
    report = {
        "status": "PASS",
        "deterministic_replay": True,
        "development_metrics": dev,
        "calibration_metrics": calibration,
        "manifest_hashes": observed_hashes,
        "reference_receipt_identities": observed_receipt_identities,
        "access_flags": cfg["access_policy"],
        "claim_flags": EXPECTED_CLAIM_BOUNDARY,
        "test_manifest_access": 0,
        "twin_qualification_pass": False,
        "formal_qualification": False,
        "agent_work_open": False,
    }
    if not getattr(args, "read_only", False):
        path = output_root / "replay/replay_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
