from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.paper_grade_entrypoint import run_catalog_entrypoint, write_entrypoint_outputs, run_identifier, resolved_seed
from Libs.utils.exp_utils import build_standard_experiment_arg_parser, resolve_output_dir
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.exp.experiment_manifests import resolve_training_seed
from Libs.utils.model.screening_model_config import native_direct_mean_config
from Libs.utils.evaluation.native_rollout_metrics import metric_policy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_standard_experiment_arg_parser("Evaluate catalog or actual native predictive rollouts.")
    catalog_model_family = parser.get_default("model_family")
    parser.set_defaults(task="rollout_prediction", model_family=None)
    parser.add_argument("--execution-mode", choices=["catalog", "native"], default="catalog")
    parser.add_argument("--native-operation", choices=["predict", "fit", "check", "causal_prefix"], default="predict")
    parser.add_argument("--residual-artifact", default="")
    args = parser.parse_args(argv)
    if args.execution_mode != "native" and (args.native_operation != "predict" or args.residual_artifact):
        parser.error("Native operations/artifacts require --execution-mode native")
    if args.execution_mode == "catalog" and args.model_family is None:
        args.model_family = catalog_model_family
    return args


def resolve_native_identity(args: argparse.Namespace, config: dict) -> None:
    """Bind writer identity to the selected profile using existing declarations."""
    selected = config.get("selected_experiment_family", "")
    if not selected or (args.experiment_family and args.experiment_family != selected):
        raise ValueError("Native evaluation experiment label differs from effective config")
    profile_identities = {name for name, row in config["paper_grade_full_suite"]["models"].items()
                          if row.get("profile_name") == selected}
    profile_identities.update(row["baseline_id"] for row in config["paper_grade_baseline_registry"]["baselines"]
                              if row.get("profile_name") == selected)
    if len(profile_identities) != 1:
        raise ValueError("Native profile requires one unambiguous existing model/baseline identity")
    operation = getattr(args, "native_operation", "predict")
    kind = metric_policy(config)["predictor_kind"]
    if operation == "causal_prefix" and (kind != "native_diffusion" or args.residual_artifact):
        raise ValueError("Causal prefix requires native_diffusion without a residual artifact")
    if kind == "native_diffusion":
        identities = profile_identities
    elif kind in {"transolver_direct_mean", "transolver_fitted_residual", "linear_dynamics_direct_mean"}:
        if not native_direct_mean_config(config):
            raise ValueError("Native direct predictor requires its explicit opt-in")
        identities = profile_identities
    elif kind == "initial_state_persistence":
        identities = {row["baseline_id"] for row in config["paper_grade_baseline_registry"]["baselines"]
                      if row.get("native_evaluation_predictor") == kind}
    else:
        raise ValueError("Unsupported native evaluation predictor_kind")
    if len(identities) != 1:
        raise ValueError("Native profile requires one unambiguous existing model/baseline identity")
    resolved = identities.pop()
    if args.model_family is not None and args.model_family != resolved:
        raise ValueError(f"Native model label {args.model_family!r} conflicts with profile {selected!r}: {resolved!r}")
    args.model_family = resolved
    args.experiment_family = selected
    if operation == "causal_prefix":
        seed = resolved_seed(args, config)
        namespace = f"{selected}_causal_prefix"
        if not str(args.run_id).strip():
            args.run_id = f"{run_identifier(args, seed)}_{namespace}"
        if not str(args.output_dir).strip():
            args.output_dir = str(resolve_output_dir(args, config) / namespace)
    elif kind in {"initial_state_persistence", "transolver_direct_mean", "transolver_fitted_residual", "linear_dynamics_direct_mean"}:
        # The comparison and its conditioning checkpoint are distinct identities.
        # Extend the existing YAML-resolved default root; explicit user values win.
        seed = resolved_seed(args, config)
        namespace = selected
        if kind in {"transolver_direct_mean", "transolver_fitted_residual", "linear_dynamics_direct_mean"}:
            direct = native_direct_mean_config(config)
            common = load_config(direct["conditioning_config_path"])
            operation = getattr(args, "native_operation", "predict")
            mode = operation if operation != "predict" else "fitted_residual" if kind == "transolver_fitted_residual" else "mean_only"
            law_identity = config["paper_grade_full_suite"]["native_transolver_residual"]["frozen_artifact_sha256"]
            suffix = f"_law_{law_identity[:16] or 'missing'}" if mode in {"check", "fitted_residual"} else ""
            namespace = f'{selected}_{mode}_from_{common["selected_experiment_family"]}_seed_{resolve_training_seed(common, require_selected=True)["base_training_seed"]}{suffix}'
        if not str(args.run_id).strip():
            args.run_id = f"{run_identifier(args, seed)}_{namespace}"
        if not str(args.output_dir).strip():
            args.output_dir = str(resolve_output_dir(args, config) / namespace)


def main() -> None:
    args = parse_args()
    if args.execution_mode == "catalog":
        run_catalog_entrypoint("Evaluate rollout-prediction metrics from formal result catalogs.",
            "rollout_prediction", ["concentration_nrmse", "flux_nrmse", "rollout_nrmse",
                                   "trajectory_nrmse", "stability_violation_rate"], args=args)
        return
    config = load_config(args.config)
    resolve_training_seed(config, args.seed, require_selected=True)
    resolve_native_identity(args, config)
    if args.native_operation == "causal_prefix":
        # These are the five files written by the existing output consumer.
        output = resolve_output_dir(args, config)
        if any((output/name).exists() for name in (
                "metrics.json", "status.json", "run_metadata.json", "artifact_hashes.json", "artifact_manifest.json")):
            raise FileExistsError("Causal-prefix output already contains recorded evidence")
    if args.dry_run:
        payload, sources, status = {"metrics": {}, "source_status": "dry_run"}, [], "dry_run"
    else:
        if not args.checkpoint:
            raise ValueError("Native evaluation requires one explicit training-seed checkpoint")
        if args.native_operation in {"fit", "check"}:
            from Libs.utils.evaluation.native_transolver_calibration import run_calibration
            from Libs.utils.exp.paper_grade_entrypoint import default_output_dir
            if config["paper_grade_full_suite"]["native_rollout_evaluation"]["predictor_kind"] not in {
                    "transolver_direct_mean", "transolver_fitted_residual"}:
                raise ValueError("Residual operations require the explicit native direct predictor")
            configured = config["paper_grade_full_suite"]["native_transolver_residual"]["frozen_artifact_path"]
            artifact = Path(args.residual_artifact or configured) if args.residual_artifact or configured else (
                default_output_dir(args, config, resolved_seed(args, config)) / "residual_parameters.json")
            payload, sources = run_calibration(config, Path(args.checkpoint), args.native_operation, artifact)
            status = "materialized" if payload["status"] in {"FIT_COMPLETE", "CHECK_COMPLETE_DIAGNOSTICS_ONLY"} else "incomplete"
        elif args.native_operation == "causal_prefix":
            from Libs.exp.evaluation.evaluate_screening_system import evaluate_native_causal_prefix
            payload, sources = evaluate_native_causal_prefix(config, Path(args.checkpoint))
            status = "incomplete"
        else:
            from Libs.exp.evaluation.evaluate_screening_system import evaluate_native_rollout_prediction
            if args.residual_artifact and config["paper_grade_full_suite"]["native_rollout_evaluation"]["predictor_kind"] != "transolver_fitted_residual":
                raise ValueError("A prediction artifact requires the explicit fitted residual predictor")
            if args.residual_artifact:
                config["paper_grade_full_suite"]["native_transolver_residual"]["frozen_artifact_path"] = args.residual_artifact
            try:
                payload, sources = evaluate_native_rollout_prediction(config, Path(args.checkpoint))
            except (ValueError, OSError, TypeError, KeyError) as error:
                if config["paper_grade_full_suite"]["native_rollout_evaluation"]["predictor_kind"] != "transolver_fitted_residual":
                    raise
                payload, sources = {"status": "incomplete", "metrics": {}, "source_status": "invalid_requested_residual_law",
                                    "reason": str(error)}, []
            status = "materialized" if payload["status"] == "complete" else "incomplete"
    result = write_entrypoint_outputs(args, config, payload, sources, status)
    print(json.dumps(result, sort_keys=True))
    if status == "incomplete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
