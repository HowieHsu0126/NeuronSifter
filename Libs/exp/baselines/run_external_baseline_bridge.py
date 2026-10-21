from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.external_baseline_bridge import (
    evaluate_external_bridge,
    prepare_external_bridge_inputs,
    train_external_bridge,
)
from Libs.utils.data.raw_data_registry import ensure_directory, load_config
from Libs.utils.exp.runnable_baselines import baseline_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeuronSifter data bridges for vendored external baseline repositories.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--command", required=True, choices=["prepare_inputs", "train_baseline", "evaluate_baseline", "calibrate_baseline", "summarize_baselines", "fit_decision_reference", "validate_decision_reference", "fit_scale_policy", "calibrate_scale_policy", "evaluate_predose_queries", "evaluate_residual_controls", "collect_residual_calibration", "fit_adaptive_residual", "check_adaptive_residual", "calibrate_residual_selected_loss"])
    parser.add_argument("--disease-program")
    parser.add_argument("--fold-name")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol-scope",choices=["legacy","manuscript-native"],default="legacy")
    parser.add_argument("--native-input-manifest", type=Path)
    parser.add_argument("--conditioning-config")
    parser.add_argument("--conditioning-manifest")
    parser.add_argument("--training-manifest")
    parser.add_argument("--residual-artifact")
    parser.add_argument("--calibration-reservation", choices=["fit", "check", "threshold"],
        help="Declared calibration reservation for actual residual history collection")
    parser.add_argument("--collection-phase", choices=["adaptive_fit", "final_policy_check", "final_selected_loss"])
    parser.add_argument("--adaptive-collection", type=Path)
    parser.add_argument("--adaptive-fit-artifact", type=Path)
    parser.add_argument("--adaptive-check-artifact", type=Path)
    parser.add_argument("--selected-loss-calibration", type=Path)
    parser.add_argument("--scale-policy",type=Path)
    parser.add_argument("--scale-calibration",type=Path)
    parser.add_argument("--initial-action-index",type=int,help="Prespecified original-library initial execution, common across control comparisons")
    parser.add_argument("--optimize-initial-plan", action="store_true",
        help="Select t0 by public posterior terminal risk over all complete open-loop plans")
    parser.add_argument("--optimize-initial-joint-plan", action="store_true",
        help="Select t0 by complete depth-two query/control/STOP risk including query costs and latency")
    parser.add_argument("--adaptive-readout", action="store_true",
        help="Use the prospective original-action adaptive readout with validated offline starting scales; unqualified")
    parser.add_argument("--decision-reference",type=Path)
    parser.add_argument("--evaluation-role",choices=["development","validation"],default="validation")
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--comparison-left",nargs="+",type=Path)
    parser.add_argument("--comparison-right",nargs="+",type=Path)
    parser.add_argument("--comparison-metric",choices=["crps","normalized_rmse","held_readout_log_likelihood","factor_recovery_mse"],default="crps")
    args=parser.parse_args()
    if (args.optimize_initial_plan or args.optimize_initial_joint_plan) and (args.protocol_scope != "manuscript-native"
                                      or args.command not in {"evaluate_residual_controls", "collect_residual_calibration"}):
        parser.error("Initial optimization requires manuscript-native evaluate_residual_controls")
    if args.adaptive_readout and (args.protocol_scope != "manuscript-native"
            or args.command not in {"evaluate_residual_controls", "collect_residual_calibration"} or args.baseline_name != "residual_mpc"):
        parser.error("Adaptive readout requires manuscript-native residual_mpc evaluate_residual_controls")
    if args.command == "collect_residual_calibration":
        if (args.protocol_scope != "manuscript-native" or args.baseline_name != "residual_mpc"
                or args.calibration_reservation not in {"fit", "check", "threshold"} or not args.residual_artifact):
            parser.error("Residual collection requires manuscript-native residual_mpc, declared reservation and offline scales")
    elif args.calibration_reservation is not None:
        parser.error("Calibration reservation requires collect_residual_calibration")
    try:
        validate_adaptive_cli_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    if args.adaptive_readout and not args.residual_artifact:
        parser.error("Adaptive readout requires --residual-artifact; use explicit adaptive fit/check artifacts for fitted observables")
    if args.native_input_manifest and args.protocol_scope != "manuscript-native":
        parser.error("Native input staging requires --protocol-scope manuscript-native")
    if args.protocol_scope=="legacy" and (not args.disease_program or not args.fold_name):
        parser.error("Legacy bridge requires --disease-program and --fold-name")
    if args.protocol_scope=="legacy" and args.command in {"calibrate_baseline","summarize_baselines","fit_decision_reference","validate_decision_reference","fit_scale_policy","calibrate_scale_policy","evaluate_predose_queries","evaluate_residual_controls","collect_residual_calibration","fit_adaptive_residual","check_adaptive_residual","calibrate_residual_selected_loss"}:
        parser.error("Native comparison commands require --protocol-scope manuscript-native")
    return args



def validate_adaptive_cli_arguments(args):
    artifact_commands = {"fit_adaptive_residual", "check_adaptive_residual", "calibrate_residual_selected_loss"}
    phase = getattr(args, "collection_phase", None)
    fit = getattr(args, "adaptive_fit_artifact", None)
    check = getattr(args, "adaptive_check_artifact", None)
    selected = getattr(args,"selected_loss_calibration",None)
    if fit or check or selected:
        if (getattr(args,"protocol_scope","manuscript-native") != "manuscript-native"
                or args.baseline_name != "residual_mpc"
                or args.command not in artifact_commands | {"evaluate_residual_controls","collect_residual_calibration"}):
            raise ValueError("Adaptive artifacts require a supported manuscript-native residual_mpc command")
    if check and not fit:
        raise ValueError("Independent adaptive check requires --adaptive-fit-artifact")
    if selected and (not fit or not check):
        raise ValueError("Selected-loss bounds require --adaptive-fit-artifact and --adaptive-check-artifact")
    if selected and args.command != "evaluate_residual_controls":
        raise ValueError("--selected-loss-calibration requires evaluate_residual_controls")
    if args.command == "evaluate_residual_controls" and fit and not check:
        raise ValueError("Fitted observable evaluation requires --adaptive-check-artifact")
    if args.command == "check_adaptive_residual" and check:
        raise ValueError("Independent check generation cannot consume a prior check artifact")
    if phase == "final_policy_check" and check:
        raise ValueError("Independent check collection cannot consume a prior check artifact")
    if args.command in artifact_commands:
        if (args.protocol_scope != "manuscript-native" or args.baseline_name != "residual_mpc"
                or not getattr(args, "adaptive_collection", None) or phase is not None):
            raise ValueError("Adaptive artifact commands require residual_mpc and an actual --adaptive-collection")
        if args.command != "fit_adaptive_residual" and not fit:
            raise ValueError("Adaptive check/selected-loss calibration requires --adaptive-fit-artifact")
        if args.command == "calibrate_residual_selected_loss" and not check:
            raise ValueError("Selected-loss calibration requires --adaptive-check-artifact")
        if args.command == "fit_adaptive_residual" and (fit or check):
            raise ValueError("Adaptive fit uses only immutable fit histories")
    elif getattr(args, "adaptive_collection", None):
        raise ValueError("--adaptive-collection requires an adaptive artifact command")
    if phase is not None:
        expected = {"adaptive_fit": "fit", "final_policy_check": "check", "final_selected_loss": "threshold"}
        if args.command != "collect_residual_calibration" or expected[phase] != args.calibration_reservation:
            raise ValueError("Collection phase differs from the declared immutable reservation")
        if phase != "adaptive_fit" and not fit:
            raise ValueError("Final-policy collection requires --adaptive-fit-artifact")
        if phase == "final_selected_loss" and not check:
            raise ValueError("Threshold collection requires --adaptive-check-artifact")
        if phase == "adaptive_fit" and (fit or check):
            raise ValueError("Fit histories must use the inherited starting law")
    elif args.command == "collect_residual_calibration" and args.calibration_reservation == "threshold":
        raise ValueError("Threshold collection requires the final_selected_loss phase")
    if fit and args.command == "collect_residual_calibration" and phase is None:
        raise ValueError("Fitted-law collection requires an explicit final-policy phase")


def run_native_command(args,config:dict) -> int:
    validate_adaptive_cli_arguments(args)
    artifact_commands = {"fit_adaptive_residual", "check_adaptive_residual", "calibrate_residual_selected_loss"}
    if args.command in artifact_commands:
        import json
        from Libs.utils.evaluation.native_residual_control_evaluation import (
            fit_native_adaptive_residual, check_native_adaptive_residual, calibrate_native_residual_selected_loss,
        )
        fit = json.loads(args.adaptive_fit_artifact.read_text()) if args.adaptive_fit_artifact else None
        check = json.loads(args.adaptive_check_artifact.read_text()) if args.adaptive_check_artifact else None
        if args.command == "fit_adaptive_residual":
            result = fit_native_adaptive_residual(config, args.adaptive_collection, args.output_dir)
        elif args.command == "check_adaptive_residual":
            result = check_native_adaptive_residual(config, args.adaptive_collection, fit, args.output_dir)
        else:
            result = calibrate_native_residual_selected_loss(config, args.adaptive_collection, fit, check, args.output_dir)
        print(json.dumps({"status": result["status"], "artifact_id": result["artifact_id"]}))
        return 0 if result["status"] in {"FIT_COMPLETE", "CHECK_OBSERVED", "CALIBRATION_COMPLETE"} else 2
    collecting = args.command == "collect_residual_calibration"
    if collecting and (getattr(args, "protocol_scope", None) != "manuscript-native"
            or getattr(args, "baseline_name", None) != "residual_mpc"
            or getattr(args, "calibration_reservation", None) not in {"fit", "check", "threshold"}
            or not getattr(args, "residual_artifact", None)):
        raise ValueError("Residual collection requires its explicit purpose, declared reservation and offline scales")
    if not collecting and getattr(args, "calibration_reservation", None) is not None:
        raise ValueError("Calibration reservation requires collect_residual_calibration")
    if getattr(args, "adaptive_readout", False) and (args.command not in {"evaluate_residual_controls", "collect_residual_calibration"}
            or getattr(args, "baseline_name", None) != "residual_mpc"
            or getattr(args, "protocol_scope", "manuscript-native") != "manuscript-native"):
        raise ValueError("Adaptive readout requires manuscript-native residual_mpc evaluate_residual_controls")
    if (getattr(args, "optimize_initial_plan", False) or getattr(args, "optimize_initial_joint_plan", False)) and args.command not in {"evaluate_residual_controls", "collect_residual_calibration"}:
        raise ValueError("--optimize-initial-plan requires evaluate_residual_controls")
    from Libs.utils.model.native_scientific_source import native_pretrainer_method
    from Libs.utils.exp.native_comparator_training import (
        load_frozen_native_belief,fit_native_comparator,
    )
    from Libs.utils.evaluation.native_comparator_evaluation import (
        declared_dataset,load_native_comparator,calibrate_native_comparator,evaluate_native_comparator,
        summarize_native_comparisons,native_residual_partitions,admit_native_residual_calibration,
    )
    from Libs.models.modules.native_diffusion_comparators import local_reference
    from Libs.utils.exp.experiment_manifests import resolve_training_seed
    import json
    import torch

    if args.command=="summarize_baselines":
        if not args.comparison_left or not args.comparison_right:
            raise ValueError("Native comparisons require all five left and right evaluation reports")
        result=summarize_native_comparisons(config,args.comparison_left,args.comparison_right,
            args.comparison_metric,Path(args.output_dir).resolve(),args.evaluation_role)
        return 0 if result["status"]=="complete" else 2
    names={"wdno_adapter":"wdno","diffusion_pde_adapter":"diffusion_pde",
           "coda_no_adapter":"codano","wdno":"wdno","diffusion_pde":"diffusion_pde","codano":"codano","transolver":"transolver",
           "linear":"linear","persistence":"persistence","mamba_no":"mamba_no","mamba_neural_operator_adapter":"mamba_no"}
    names.update({name:name for name in ("tau_transport","treatment_history","same_compiler_solver","mechanistic_residual","native_deterministic","flattened_parameter_matched","gru","fno","native_wam","native_chemical_reference","native_action_id","native_target_id",
        "native_scalar_exposure","native_parallel","native_reversed","native_interleaved")})
    names.update({name:name for name in config.get("manuscript_native_protocol",{}).get("native_method_profiles",{})})
    names.update({name:name for name in config.get("native_rssm_prediction",{}).get("fitted_methods",[])})
    scale_methods=config.get("native_scale_routing",{}).get("methods",[])
    names.update({name:name for name in scale_methods})
    aliases=config.get("manuscript_native_protocol",{}).get("checkpoint_reuse",{})
    names.update(aliases)
    state_policy=config.get("native_state_comparison",{})
    names.update({name:name for name in state_policy.get("fitted_methods",[])})
    names.update(state_policy.get("checkpoint_aliases",{}))
    control_component = args.command in {"evaluate_residual_controls", "collect_residual_calibration"}
    if control_component:
        if args.baseline_name != "residual_mpc":
            raise ValueError("Residual control needs residual_mpc")
        from Libs.utils.evaluation.native_residual_control_evaluation import residual_initial_mode, residual_readout_mode
        readout_mode = residual_readout_mode(config, adaptive_readout=collecting or getattr(args, "adaptive_readout", False))
        if readout_mode == "original_action_adaptive_unqualified" and not getattr(args, "residual_artifact", None):
            raise ValueError("Adaptive readout requires a validated offline residual artifact; the offline starting artifact is required")
        residual_initial_mode(config, initial_index=getattr(args, "initial_action_index", None),
                              optimize_initial_plan=getattr(args, "optimize_initial_plan", False),
                              optimize_initial_joint_plan=getattr(args, "optimize_initial_joint_plan", False))
        # Evaluation identity differs from the unchanged selected predictor identity.
        names["residual_mpc"] = "mechanistic_residual"
    if args.baseline_name not in names:
        raise ValueError("No authentic native binding for required method: "+args.baseline_name)
    method=names[args.baseline_name]
    state_probe=method in state_policy.get("fitted_methods",[])
    if (args.baseline_name in aliases or args.baseline_name in state_policy.get("checkpoint_aliases",{})) and args.command=="train_baseline":
        raise ValueError("This control reuses its declared selected checkpoint and must not refit")
    if args.command not in {"fit_decision_reference","validate_decision_reference"}:
        resolve_training_seed(config,require_selected=True)
    if method in {"wdno","diffusion_pde","codano","fno"}:
        local_reference(config["native_controlled_comparators"]["references"][method])
    output=Path(args.output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    decision_policy=config.get("native_intervention_evaluation",{})
    decision_method=args.baseline_name in decision_policy.get("posterior_methods",[])
    predose_component=args.command=="evaluate_predose_queries"
    if predose_component and args.baseline_name not in {"posterior_joint", "mechanistic_residual"}:
        raise ValueError("The pre-dose component requires its declared selected joint or offline hybrid predictor")
    scale_method=args.baseline_name in scale_methods
    solver_method=method=="same_compiler_solver"
    if method=="adaptive_scale":
        from Libs.utils.evaluation.native_scale_routing import run_adaptive_scale_command
        if args.training_manifest or args.residual_artifact:
            raise ValueError("Adaptive SCALE reuses its declared checkpoints and selected-loss calibration")
        if not args.conditioning_config or not args.conditioning_manifest:
            raise ValueError("Adaptive SCALE requires its unchanged common conditioning bindings")
        result=run_adaptive_scale_command(config,args.command,Path(args.conditioning_config).resolve(),
            Path(args.conditioning_manifest).resolve(),args.decision_reference,output,torch.device(args.device),
            policy_path=args.scale_policy,calibration_path=args.scale_calibration,evaluation_role=args.evaluation_role)
        print(json.dumps({"status":result["status"],"additional_optimizer_updates":0}))
        return 0 if result["status"]=="complete" else 2
    if args.command in {"fit_decision_reference","validate_decision_reference"}:
        from Libs.utils.evaluation.native_intervention_evaluation import fit_dataset_public_reference, training_record_identities
        from Libs.utils.evaluation.native_intervention_decisions import validate_public_loss_reference
        if not decision_method or not args.decision_reference:
            raise ValueError("Public reference commands need a posterior control and explicit --decision-reference")
        if config["native_scientific_training"].get("source_scope","all")!="all":
            raise ValueError("This public reference is scoped to full-mainline training")
        if args.command=="fit_decision_reference":
            training,declarations=declared_dataset(config,"train")
            result=fit_dataset_public_reference(config,training,declarations,args.decision_reference)
            print(json.dumps({"status":result["status"],"reference_id":result["reference_id"],"fit_role":"train"}))
            return 0 if result["status"]=="defined" else 2
        reference=json.loads(args.decision_reference.read_text())
        training,declarations=declared_dataset(config,"train")
        validate_public_loss_reference(reference,config,set(declarations),training_record_identities(training))
        print(json.dumps({"status":"defined","reference_id":reference["reference_id"],"fit_role":"train"}))
        return 0
    if args.command=="prepare_inputs":
        training,_=declared_dataset(config,"train")
        development,_=declared_dataset(config,"development")
        print(json.dumps({"source":"admitted_native_registry","training_groups":len(training),
                          "development_groups":len(development),"scientific_twin":"NOT_RUN"}))
        return 0
    reference = None
    if decision_method or scale_method or solver_method or predose_component or control_component:
        from Libs.utils.evaluation.native_intervention_evaluation import training_record_identities
        from Libs.utils.evaluation.native_intervention_decisions import validate_public_loss_reference
        if args.command not in {"evaluate_baseline", "evaluate_predose_queries", "evaluate_residual_controls", "collect_residual_calibration"} or not args.decision_reference:
            raise ValueError("Native decision comparisons require evaluation and a declared public reference")
        reference = json.loads(args.decision_reference.read_text())
        training, training_declarations = declared_dataset(config, "train")
        validate_public_loss_reference(reference, config, set(training_declarations), training_record_identities(training))
    if not args.conditioning_config or not args.conditioning_manifest:
        raise ValueError("Native comparator requires exact conditioning config and development-selection manifest")
    device=torch.device(args.device)
    common=(config,Path(args.conditioning_config).resolve(),Path(args.conditioning_manifest).resolve(),device)
    if scale_method:
        from Libs.utils.evaluation.native_scale_routing import evaluate_native_scale_routing
        if args.training_manifest or args.residual_artifact:
            raise ValueError("Scale routes use their declared upstream checkpoint set and no residual-law override")
        dataset,declarations=declared_dataset(config,args.evaluation_role)
        result=evaluate_native_scale_routing(config,args.baseline_name,common[1],common[2],
            dataset,declarations,reference,output,device)
        return 0 if result["status"]=="complete" else 2
    if solver_method and (args.training_manifest or args.residual_artifact):
        raise ValueError("Frozen primary RHS has no transition checkpoint or residual calibration")
    if args.command=="train_baseline":
        if method in {"persistence","same_compiler_solver"}:
            raise ValueError("Persistence has no fitted transition; use evaluate_baseline")
        if native_pretrainer_method(config,method):
            raise ValueError("Native world-action training uses pretrain_microenvironment_world_model.py")
        belief,identity=load_frozen_native_belief(*common)
        training,_=declared_dataset(config,"train")
        development,_=declared_dataset(config,"development")
        result=fit_native_comparator(belief,method,training,development,config,output,identity,device)
        return 0 if result["selection"]["terminal"]=="DEVELOPMENT_PLATEAU_REQUIRES_SUFFICIENCY_REVIEW" else 2
    if not args.training_manifest and method not in {"persistence","same_compiler_solver"}:
        raise ValueError("Native evaluation/calibration requires its selected training manifest")
    if args.command=="calibrate_baseline" or collecting:
        native_residual_partitions(config)
        fit,fit_ids=declared_dataset(config,"calibration",reservation="fit")
        check,check_ids=declared_dataset(config,"calibration",reservation="check")
        admit_native_residual_calibration(config,fit,check)
        dataset=fit if not collecting or args.calibration_reservation == "fit" else check
        if collecting:
            from Libs.utils.evaluation.native_residual_control_evaluation import residual_collection_policy
            residual_collection_policy(config, args.calibration_reservation, getattr(args, "collection_phase", None))
            if args.calibration_reservation == "threshold":
                threshold, _ = declared_dataset(config, "calibration", reservation="threshold")
                dataset = threshold
    else:
        dataset,declarations=declared_dataset(config,args.evaluation_role)
    model,belief,binding=load_native_comparator(config,method,common[1],common[2],
        Path(args.training_manifest).resolve() if args.training_manifest else None,dataset,device)
    if collecting:
        from Libs.utils.evaluation.native_residual_control_evaluation import collect_native_residual_calibration
        result=collect_native_residual_calibration(config,model,belief,binding,fit,check,reference,output,
            reservation=args.calibration_reservation, residual_path=Path(args.residual_artifact).resolve(),
            initial_index=getattr(args, "initial_action_index", None),
            optimize_initial_plan=getattr(args, "optimize_initial_plan", False),
            optimize_initial_joint_plan=getattr(args, "optimize_initial_joint_plan", False),
            phase=getattr(args, "collection_phase", None),
            adaptive_artifact=json.loads(args.adaptive_fit_artifact.read_text()) if getattr(args, "adaptive_fit_artifact", None) else None,
            check_artifact=json.loads(args.adaptive_check_artifact.read_text()) if getattr(args, "adaptive_check_artifact", None) else None,
            threshold=threshold if args.calibration_reservation == "threshold" else None)
        return 0 if result["status"] in {"collection_complete", "software_collection_complete"} else 2
    if control_component:
        from Libs.utils.evaluation.native_residual_control_evaluation import evaluate_native_residual_controls
        result=evaluate_native_residual_controls(config,model,belief,binding,dataset,declarations,reference,output,
            initial_index=getattr(args, "initial_action_index", None),
            optimize_initial_plan=getattr(args, "optimize_initial_plan", False),
            optimize_initial_joint_plan=getattr(args, "optimize_initial_joint_plan", False),
            residual_path=Path(args.residual_artifact).resolve() if args.residual_artifact else None,
            adaptive_readout=getattr(args, "adaptive_readout", False),
            adaptive_artifact=json.loads(args.adaptive_fit_artifact.read_text()) if getattr(args, "adaptive_fit_artifact", None) else None,
            check_artifact=json.loads(args.adaptive_check_artifact.read_text()) if getattr(args, "adaptive_check_artifact", None) else None,
            selected_loss_calibration=json.loads(args.selected_loss_calibration.read_text()) if getattr(args, "selected_loss_calibration", None) else None)
        return 0 if result["status"]=="component_complete" else 2
    if predose_component:
        from Libs.utils.evaluation.native_predose_evaluation import evaluate_native_predose_queries
        result=evaluate_native_predose_queries(config,model,belief,binding,dataset,declarations,reference,output,
            residual_path=Path(args.residual_artifact).resolve() if args.residual_artifact else None)
        return 0 if result["status"]=="component_complete" else 2
    if args.command=="calibrate_baseline":
        result=calibrate_native_comparator(config,method,model,belief,binding,fit,check,output)
        return 0 if result.get("calibration_check_complete") else 2
    if decision_method:
        from Libs.utils.evaluation.native_intervention_evaluation import evaluate_native_interventions
        result=evaluate_native_interventions(config,args.baseline_name,model,belief,binding,dataset,
                                            declarations,reference,output)
        return 0 if result["status"]=="complete" else 2
    if state_probe:
        from Libs.utils.evaluation.native_state_evaluation import evaluate_native_state
        result=evaluate_native_state(config,args.baseline_name,model,binding,dataset,declarations,output)
        return 0 if result["status"]=="complete" else 2
    result=evaluate_native_comparator(config,method,model,belief,binding,dataset,declarations,output,
        Path(args.residual_artifact).resolve() if args.residual_artifact else None,decision_reference=reference)
    return 0 if result["status"]=="complete" else 2


def main() -> int | None:
    args = parse_args()
    config = load_config(args.config)
    if args.native_input_manifest:
        config.setdefault("native_input_staging", {})["manifest_path"] = str(args.native_input_manifest)
    if args.protocol_scope=="manuscript-native":
        return run_native_command(args,config)
    entry = baseline_entry(config, args.baseline_name)
    output_dir = Path(args.output_dir).resolve()
    ensure_directory(output_dir)
    manifest_path = output_dir / f"{args.baseline_name}_{args.disease_program}_{args.fold_name}_external_bundle_manifest.json"
    checkpoint_path = output_dir / "model_weights.pt"
    if args.command == "prepare_inputs":
        prepare_external_bridge_inputs(args.config, args.baseline_name, args.disease_program, args.fold_name, output_dir)
        return
    if not manifest_path.exists():
        prepare_external_bridge_inputs(args.config, args.baseline_name, args.disease_program, args.fold_name, output_dir)
    if args.command == "train_baseline":
        train_external_bridge(entry, manifest_path, output_dir)
        return
    if not checkpoint_path.exists():
        train_external_bridge(entry, manifest_path, output_dir)
    evaluate_external_bridge(entry, manifest_path, checkpoint_path, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
