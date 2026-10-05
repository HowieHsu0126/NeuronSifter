from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.validators.validate_external_adapter_registry import (
    validate_external_adapter_rows,
    validate_header,
)
from Libs.utils.data.raw_data_registry import load_config, read_tsv, resolve_repo_path, write_json
from Libs.utils.evaluation.active_learning_loop import (
    active_learning_config,
    active_learning_root,
    mandatory_source_keys,
    round_path,
    validate_mandatory_adapters,
    validate_required_binaries,
)
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-fast preflight for NeuronSifter active-learning loop.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--strict-processed", action="store_true")
    parser.add_argument("--strict-binaries", action="store_true")
    return parser.parse_args()


def validate_active_learning_contract(config: dict) -> dict[str, object]:
    active_config = active_learning_config(config)
    if str(active_config["contract_name"]) != "decision_centric_simulator_portfolio_active_learning":
        raise ValueError(f"Unexpected active-learning contract: {active_config['contract_name']}")
    if int(active_config["full_round_count"]) != 3:
        raise ValueError("Active-learning full_round_count must be 3.")
    if int(active_config["full_tasks_per_round"]) != 128:
        raise ValueError("Active-learning full_tasks_per_round must be 128.")
    quota_sum = sum(float(value) for value in active_config["batch_quota"].values())
    if abs(quota_sum - float(active_config["quota_sum"])) > 1.0e-9:
        raise ValueError("Active-learning batch quota must sum to configured quota_sum.")
    simulator_families = {str(row["simulator_family"]) for row in active_config["simulator_portfolio"]}
    missing_simulators = sorted(set(active_config["mandatory_simulator_families"]) - simulator_families)
    if missing_simulators:
        raise ValueError(f"Active-learning simulator portfolio is missing mandatory families: {missing_simulators}")
    required_schema_keys = [
        "active_learning_candidate_pool_fields",
        "active_learning_acquisition_fields",
        "active_learning_simulation_request_fields",
        "active_learning_simulation_result_fields",
        "active_learning_teacher_update_fields",
        "active_learning_round_report_fields",
    ]
    missing_schema_keys = [schema_key for schema_key in required_schema_keys if schema_key not in config["schemas"]]
    if missing_schema_keys:
        raise ValueError(f"Active-learning schemas are missing: {missing_schema_keys}")
    required_request_defaults = [
        "time_grid",
        "region_grid",
        "compartment_grid",
        "species_list",
        "parameter_prior_uid",
        "uncertainty_policy",
    ]
    missing_request_defaults = [
        key for key in required_request_defaults if key not in active_config["simulation_request_defaults"]
    ]
    if missing_request_defaults:
        raise ValueError(f"Active-learning simulation request defaults are missing: {missing_request_defaults}")
    result_linking = active_config["result_linking"]
    missing_artifact_sources = sorted(set(active_config["mandatory_simulator_families"]) - set(result_linking["artifact_sources"]))
    if missing_artifact_sources:
        raise ValueError(f"Active-learning result linking lacks mandatory simulator families: {missing_artifact_sources}")
    missing_label_policies = sorted(
        {
            str(row["training_label_policy"])
            for row in active_config["simulator_portfolio"]
        }
        - set(result_linking["result_policy_by_training_label_policy"])
    )
    if missing_label_policies:
        raise ValueError(f"Active-learning result linking lacks training label policies: {missing_label_policies}")
    return {
        "contract_name": str(active_config["contract_name"]),
        "full_round_count": int(active_config["full_round_count"]),
        "full_tasks_per_round": int(active_config["full_tasks_per_round"]),
        "batch_quota": dict(active_config["batch_quota"]),
        "mandatory_source_count": len(mandatory_source_keys(config)),
        "simulator_family_count": len(simulator_families),
    }


def validate_strict_external_adapter_registry(config: dict, strict_processed: bool) -> None:
    if not strict_processed:
        return
    registry_config = config["validation"]["external_adapter_registry"]
    path = resolve_repo_path(str(registry_config["local_root"])) / str(registry_config["relative_paths"]["external_adapter_catalog"])
    if not path.exists():
        raise FileNotFoundError(f"Missing external adapter catalog: {path}")
    validate_header(path, config["schemas"]["external_adapter_fields"])
    validate_external_adapter_rows(read_tsv(path), config, strict_processed=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    contract_report = validate_active_learning_contract(config)
    validate_strict_external_adapter_registry(config, strict_processed=bool(args.strict_processed))
    adapter_validations = validate_mandatory_adapters(config, strict_processed=bool(args.strict_processed))
    binary_validations = validate_required_binaries(config) if bool(args.strict_binaries) else []
    report = {
        "status": "active_learning_preflight_passed",
        "contract": contract_report,
        "adapter_validations": adapter_validations,
        "binary_validations": binary_validations,
        "artifact_root": str(active_learning_root(config)),
        "round_report_paths": [
            str(round_path(config, "round_report_template", str(row["round_uid"])))
            for row in active_learning_config(config)["active_rounds"]
        ],
    }
    output_path = active_learning_root(config) / str(active_learning_config(config)["relative_paths"]["preflight_audit"])
    write_json(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=False))
    complete_script_logging(pipeline_logger, pipeline_started_at, artifact=output_path)


if __name__ == "__main__":
    main()
