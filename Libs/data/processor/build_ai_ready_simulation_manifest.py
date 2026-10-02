from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import ensure_directory, load_config, temporary_output_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an AI-ready trajectory manifest and volume audit.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(REPO_ROOT))


def row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in handle) - 1


def chunk_store_shape(store_path: Path) -> list[int]:
    manifest_path = store_path / "store_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [int(value) for value in payload["arrays"]["trajectories"]["shape"]]


def zarr_v2_shape(store_path: Path) -> list[int]:
    payload = json.loads((store_path / "trajectories" / ".zarray").read_text(encoding="utf-8"))
    return [int(value) for value in payload["shape"]]


def trajectory_shape(store_path: Path) -> list[int]:
    if (store_path / "store_manifest.json").exists():
        return chunk_store_shape(store_path)
    return zarr_v2_shape(store_path)


def split_name(uid: str, split_policy: dict[str, object]) -> str:
    algorithm = str(split_policy["hash_algorithm"])
    modulus = int(split_policy["modulus"])
    digest = hashlib.new(algorithm, uid.encode("utf-8")).hexdigest()
    remainder = int(digest, 16) % modulus
    if remainder in {int(value) for value in split_policy["validation_remainders"]}:
        return "validation"
    if remainder in {int(value) for value in split_policy["test_remainders"]}:
        return "test"
    return "train"


def stream_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def manifest_fieldnames() -> list[str]:
    return [
        "ai_ready_uid",
        "source_collection",
        "source_uid",
        "disease_program",
        "teacher_family",
        "teacher_role",
        "calibration_role",
        "observed_or_synthetic",
        "trajectory_store_path",
        "trajectory_index",
        "trajectory_count_in_store",
        "species_count",
        "split_name",
        "ai_ready_status",
        "volume_status",
        "note",
    ]


def readiness_status(full_trajectory_count: int, volume_targets: dict[str, object], status_names: dict[str, object]) -> str:
    if full_trajectory_count >= int(volume_targets["final_full_microenvironment_trajectory_count"]):
        return str(status_names["million_ready"])
    if full_trajectory_count >= int(volume_targets["pilot_full_microenvironment_trajectory_count"]):
        return str(status_names["pilot_ready"])
    return str(status_names["shortfall"])


def is_full_microenvironment_training_row(
    row: dict[str, str],
    disease_gate: dict[str, object],
    active_disease_program: str,
    required_teacher_family: str,
    full_config: dict[str, object],
) -> bool:
    if bool(disease_gate["enabled"]) and row.get("disease_program", "") != active_disease_program:
        return False
    if row.get("teacher_family", "") != required_teacher_family:
        return False
    trajectory_store_value = row[str(full_config["trajectory_store_field"])].strip()
    trajectory_index_value = row[str(full_config["trajectory_index_field"])].strip()
    return bool(trajectory_store_value and trajectory_index_value)


def build_manifest(config: dict[str, object]) -> dict[str, object]:
    preprocessing_config = config["ai_ready_preprocessing"]
    source_collections = preprocessing_config["source_collections"]
    split_policy = preprocessing_config["split_policy"]
    volume_targets = preprocessing_config["volume_targets"]
    disease_gate = preprocessing_config["disease_gate"]
    status_names = preprocessing_config["readiness_status_names"]
    parameter_evidence = preprocessing_config["parameter_evidence_status"]
    active_disease_program = str(disease_gate["primary_disease_program"])
    output_root = repo_path(str(preprocessing_config["local_root"]))
    output_path = output_root / str(preprocessing_config["relative_paths"]["simulation_manifest"])
    audit_path = output_root / str(preprocessing_config["relative_paths"]["audit_report"])
    ensure_directory(output_root)

    full_store_path = repo_path(str(source_collections["full_microenvironment_teacher"]["required_trajectory_store"]))
    mesoscopic_store_path = repo_path(str(source_collections["mesoscopic_solver_teacher"]["required_trajectory_store"]))
    full_shape = trajectory_shape(full_store_path)
    mesoscopic_shape = trajectory_shape(mesoscopic_store_path)
    full_store_trajectory_count = int(full_shape[0])
    mesoscopic_trajectory_count = int(mesoscopic_shape[0])
    target_count = int(volume_targets["target_simulation_count"])
    full_config = source_collections["full_microenvironment_teacher"]
    full_catalog_path = repo_path(str(full_config["catalog_path"]))
    required_teacher_family = str(full_config["required_teacher_family"])
    full_trajectory_count = sum(
        1
        for row in stream_rows(full_catalog_path)
        if is_full_microenvironment_training_row(row, disease_gate, active_disease_program, required_teacher_family, full_config)
    )

    full_volume_status = "target_met" if full_trajectory_count >= int(volume_targets["minimum_full_microenvironment_trajectory_count"]) else "pilot_met" if full_trajectory_count >= int(volume_targets["pilot_full_microenvironment_trajectory_count"]) else "below_target"
    mesoscopic_volume_status = "compatibility_indexed" if not bool(volume_targets["mesoscopic_blocks_mainline_readiness"]) else "target_met" if mesoscopic_trajectory_count >= int(volume_targets["minimum_mesoscopic_teacher_trajectory_count"]) else "below_target"
    ai_ready_status = readiness_status(full_trajectory_count, volume_targets, status_names)

    total_manifest_rows = 0
    split_counts = {"train": 0, "validation": 0, "test": 0}
    temporary_path = temporary_output_path(output_path)
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fieldnames(), delimiter="\t")
        writer.writeheader()

        mesoscopic_config = source_collections["mesoscopic_solver_teacher"]
        mesoscopic_catalog_path = repo_path(str(mesoscopic_config["catalog_path"]))
        for trajectory_index, row in enumerate(stream_rows(mesoscopic_catalog_path)):
            source_uid = row[str(mesoscopic_config["uid_field"])]
            row_split = split_name(source_uid, split_policy)
            split_counts[row_split] += 1
            total_manifest_rows += 1
            writer.writerow(
                {
                    "ai_ready_uid": f"ai_ready_mesoscopic_{trajectory_index}",
                    "source_collection": "mesoscopic_solver_teacher",
                    "source_uid": source_uid,
                    "disease_program": row.get("disease_program", ""),
                    "teacher_family": row.get("teacher_family", ""),
                    "teacher_role": row.get("teacher_role", ""),
                    "calibration_role": row.get("calibration_role", ""),
                    "observed_or_synthetic": row.get("observed_or_synthetic", ""),
                    "trajectory_store_path": repo_relative(repo_path(row[str(mesoscopic_config["trajectory_store_field"])])),
                    "trajectory_index": trajectory_index,
                    "trajectory_count_in_store": mesoscopic_trajectory_count,
                    "species_count": int(mesoscopic_shape[4]),
                    "split_name": row_split,
                    "ai_ready_status": ai_ready_status,
                    "volume_status": mesoscopic_volume_status,
                    "note": "Mesoscopic solver trajectory indexed for AI loaders; volume target still controls readiness.",
                }
            )

        for row in stream_rows(full_catalog_path):
            if not is_full_microenvironment_training_row(row, disease_gate, active_disease_program, required_teacher_family, full_config):
                continue
            trajectory_store_value = row[str(full_config["trajectory_store_field"])].strip()
            trajectory_index_value = row[str(full_config["trajectory_index_field"])].strip()
            source_uid = row[str(full_config["uid_field"])]
            row_split = split_name(source_uid, split_policy)
            split_counts[row_split] += 1
            total_manifest_rows += 1
            writer.writerow(
                {
                    "ai_ready_uid": f"ai_ready_full_microenvironment_{trajectory_index_value}",
                    "source_collection": "full_microenvironment_teacher",
                    "source_uid": source_uid,
                    "disease_program": row.get("disease_program", ""),
                    "teacher_family": row.get("teacher_family", ""),
                    "teacher_role": row.get("teacher_role", ""),
                    "calibration_role": row.get("calibration_role", ""),
                    "observed_or_synthetic": row.get("observed_or_synthetic", ""),
                    "trajectory_store_path": repo_relative(repo_path(trajectory_store_value)),
                    "trajectory_index": int(trajectory_index_value),
                    "trajectory_count_in_store": full_store_trajectory_count,
                    "species_count": int(full_shape[4]),
                    "split_name": row_split,
                    "ai_ready_status": ai_ready_status,
                    "volume_status": full_volume_status,
                    "note": f"Full AD microenvironment trajectory indexed for AI loaders; {parameter_evidence['required_manifest_note']}.",
                }
            )
    temporary_path.replace(output_path)

    world_state_config = source_collections["world_state_observation"]
    world_state_catalog_path = repo_path(str(world_state_config["catalog_path"]))
    world_state_row_count = row_count(world_state_catalog_path)
    audit_payload = {
        "ai_ready_status": ai_ready_status,
        "target_simulation_count": target_count,
        "pilot_target_simulation_count": int(volume_targets["pilot_full_microenvironment_trajectory_count"]),
        "final_target_simulation_count": int(volume_targets["final_full_microenvironment_trajectory_count"]),
        "disease_gate": {
            "enabled": bool(disease_gate["enabled"]),
            "primary_disease_program": active_disease_program,
            "training_manifest_requires_primary_disease_only": bool(disease_gate["training_manifest_requires_primary_disease_only"]),
        },
        "parameter_evidence_status": parameter_evidence,
        "manifest_row_count": total_manifest_rows,
        "split_counts": split_counts,
        "mesoscopic_solver_teacher": {
            "catalog_path": repo_relative(repo_path(str(source_collections["mesoscopic_solver_teacher"]["catalog_path"]))),
            "catalog_row_count": row_count(repo_path(str(source_collections["mesoscopic_solver_teacher"]["catalog_path"]))),
            "trajectory_store_path": repo_relative(mesoscopic_store_path),
            "trajectory_shape": mesoscopic_shape,
            "trajectory_count": mesoscopic_trajectory_count,
            "volume_status": mesoscopic_volume_status,
        },
        "full_microenvironment_teacher": {
            "catalog_path": repo_relative(repo_path(str(source_collections["full_microenvironment_teacher"]["catalog_path"]))),
            "catalog_row_count": row_count(repo_path(str(source_collections["full_microenvironment_teacher"]["catalog_path"]))),
            "trajectory_store_path": repo_relative(full_store_path),
            "trajectory_shape": full_shape,
            "trajectory_count": full_store_trajectory_count,
            "ad_gated_training_trajectory_count": full_trajectory_count,
            "volume_status": full_volume_status,
        },
        "world_state_observation": {
            "catalog_path": repo_relative(world_state_catalog_path),
            "catalog_row_count": world_state_row_count,
        },
        "blocking_reasons": [
            reason
            for reason, active in [
                (f"full_microenvironment_trajectory_count_below_pilot_{volume_targets['pilot_full_microenvironment_trajectory_count']}", full_trajectory_count < int(volume_targets["pilot_full_microenvironment_trajectory_count"])),
                (f"full_microenvironment_trajectory_count_below_million_{volume_targets['final_full_microenvironment_trajectory_count']}", full_trajectory_count < int(volume_targets["final_full_microenvironment_trajectory_count"])),
                (f"mesoscopic_teacher_trajectory_count_below_{target_count}", bool(volume_targets["mesoscopic_blocks_mainline_readiness"]) and mesoscopic_volume_status != "target_met"),
            ]
            if active
        ],
        "next_preprocessing_action": "Launch configured staged million-scale trajectory generation on approved compute nodes, then rerun this manifest builder.",
    }
    write_json(audit_path, audit_payload)
    return {"manifest_path": repo_relative(output_path), "audit_path": repo_relative(audit_path), **audit_payload}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    payload = build_manifest(config)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
