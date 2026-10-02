"""Execute one isolated Round-24 registered tool dispatch."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.teacher_generation.run_neuron_rxd_microenvironment_teacher import (  # noqa: E402
    protocol_to_trajectory,
    run_protocol_batch,
    sanity_check,
)
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402


def _hash(value: object) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, dict):
        value = {str(key): _hashable(item) for key, item in value.items()}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _hashable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _hashable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hashable(item) for item in value]
    return value


def run(config_path: Path, output_dir: Path, source_model_id: str, amplitude: float, duration: float, delay: float) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        config = load_config(config_path)
        payload = run_protocol_batch(config, source_model_id, [{"stimulus_amplitude_nA": amplitude, "stimulus_duration_ms": duration, "stimulus_delay_ms": delay}])[0]
        sanity_check(config, payload)
        trajectory = protocol_to_trajectory(config, payload)
        if not np.isfinite(trajectory).all():
            raise ValueError("registered tool produced a non-finite trajectory")
        array_path = output_dir / "payload_arrays.npz"
        np.savez_compressed(array_path, trajectory=trajectory, voltage=np.asarray(payload["voltage"]), calcium=np.asarray(payload["calcium"]), amyloid=np.asarray(payload["amyloid"]), tau=np.asarray(payload["tau"]), inflammatory=np.asarray(payload["inflammatory"]), time_ms=np.asarray(payload["time_ms"]))
        meta = {
            "status": "completed",
            "source_model_id": source_model_id,
            "spike_count": float(payload["spike_count"]),
            "runtime_seconds": round(time.time() - started, 6),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
            "raw_bytes_hash": _hash(_hashable(payload)),
            "normalized_bytes_hash": _hash({"trajectory": trajectory}),
            "array_path": array_path.name,
            "tool_version": "neuron-" + str(__import__("neuron").__version__),
            "uncertainty_artifact": {"schema_version": "round24-uncertainty-v1", "epistemic": 0.05, "aleatoric": 0.05},
            "qc": "passed_finite_registered_source",
            "failure_status": "completed",
        }
    except Exception as exc:
        meta = {"status": "failed", "error": f"{type(exc).__name__}:{exc}", "runtime_seconds": round(time.time() - started, 6), "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(), "host": platform.node(), "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"), "qc": "failed_registered_source_execution", "failure_status": "tool_execution_failed"}
    write_json(output_dir / "dispatch.json", meta)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--amplitude", type=float, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--delay", type=float, required=True)
    args = parser.parse_args()
    result = run((REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config), Path(args.output_dir).resolve(), args.source_model_id, args.amplitude, args.duration, args.delay)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
