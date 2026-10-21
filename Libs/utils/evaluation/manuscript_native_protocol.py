"""Manuscript-native profile, budget, roster and command bindings."""
from __future__ import annotations

import copy
import math
import hashlib
import json
import shlex
import subprocess
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

from Libs.utils.model.native_scientific_source import selected_group_definitions,group_definitions,planned_operation_counts,prior_particle_count,effective_stimulation_actions
from Libs.utils.model.neuronsifter_scale_registry import apply_neuronsifter_scale
from Libs.utils.exp.experiment_manifests import resolve_training_seed, native_training_seed_values
from Libs.utils.exp.native_learned_distillation import validate_distillation_policy
from Libs.models.modules.native_mechanistic_residual import native_residual_online_policy
from Libs.utils.model.native_future_controls import future_control_policy
from Libs.utils.model.screening_model_config import observation_feature_profile


def training_budget(config:dict,percentage:int) -> dict:
    """Plan a common update clock and report the distinct complete-group visits."""
    selected=copy.deepcopy(config)
    selected["native_scientific_training"]["training_subset_percentage"]=percentage
    groups=len(selected_group_definitions(selected,"train"))
    policy=selected["native_scientific_training"]
    batch=int(selected["screening_pipeline"]["screening_benchmark"]["training"]["memory_policy"]["effective_batch_size"])
    updates=math.ceil(groups/batch)
    maximum=int(policy["selection"]["max_steps"])
    minimum=int(policy["selection"]["minimum_steps"])
    epochs=math.ceil(maximum/updates)
    full_epochs,partial=divmod(maximum,updates)
    visits=full_epochs*groups+min(partial*batch,groups)
    if minimum>maximum or epochs<policy["selection"]["minimum_epochs"]:
        raise ValueError("Native fraction cannot reach its unchanged scientific stopping prerequisites")
    return {"training_subset_percentage":percentage,"selected_training_groups":groups,
        "effective_group_batch_size":batch,"optimizer_updates_per_full_epoch":updates,
        "max_optimizer_updates":maximum,"minimum_optimizer_updates":minimum,"max_epochs":epochs,
        "planned_group_visits_at_update_cap_without_early_stop":visits,
        "validation_interval_optimizer_steps":63,"patience_validation_snapshots":policy["selection"]["patience"],
        "warmup_optimizer_updates":policy["learning_rate_schedule"]["warmup_steps"],
        "learning_rate_schedule_horizon_updates":maximum,
        "group_visit_equality_across_fractions":False,"counts_are":"prospective_caps_not_executed"}


IMPLEMENTED_KINDS = {"native_pretrainer", "native_comparator", "native_identity_law", "native_state_probe", "native_checkpoint_alias", "native_posterior_decision"}


def content_hash(value) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(",",":")).encode()).hexdigest()


def method_roster(config:dict) -> list[dict]:
    """Keep every required row, including inapplicable empirical and missing controlled paths."""
    protocol=config["manuscript_native_protocol"]
    rows=copy.deepcopy(protocol["methods"])
    ids=[row["method_id"] for row in rows]
    if len(ids)!=len(set(ids)) or not ids:
        raise ValueError("Native protocol method identities must be unique and nonempty")
    for row in rows:
        implemented=row["implementation_kind"] in IMPLEMENTED_KINDS
        if implemented and row["implementation_blocker"] is not None:
            raise ValueError("Executable method has an unresolved implementation blocker")
        if not implemented and not row["implementation_blocker"]:
            raise ValueError("Unavailable method requires an exact implementation or evidence prerequisite")
        row["software_interface_bound"]=implemented
        row["scientific_execution_status"]="NOT_RUN"
        row["scientific_eligibility"]="awaiting_source_training_and_calibration" if implemented else "closed"
        row["fitted_checkpoint_reuse"]=("none_frozen_primary_transition_with_common_selected_belief"
            if row["method_id"]=="same_compiler_solver" else "same_method_seed_checkpoint_across_applicable_endpoints")
    return rows


def training_component_definition(config: dict, component: str) -> dict:
    """Resolve the internal predictor dependency without promoting its parent method."""
    protocol = config["manuscript_native_protocol"]
    definition = protocol["training_evidence"].get("predictor_components", {}).get(component)
    if (component != "residual_mpc_predictor" or not isinstance(definition, dict)
            or definition.get("owner_method") != "residual_mpc"
            or definition.get("predictor_method") != "mechanistic_residual"
            or definition.get("conditioning_method") != "native_wam"
            or definition.get("review_scope") != "residual_mpc_predictor_training"):
        raise ValueError("Unknown native predictor training component")
    roster = {row["method_id"]: row for row in method_roster(config)}
    if (definition["predictor_method"] in roster
            or roster[definition["owner_method"]]["software_interface_bound"]
            or not roster[definition["conditioning_method"]]["software_interface_bound"]):
        raise ValueError("Predictor training cannot change full residual MPC qualification")
    seeds = native_training_seed_values(protocol.get("seed_policy"), protocol["training_seeds"])
    if list(seeds) != [41, 42, 43]:
        raise ValueError("Predictor sufficiency requires exactly native seeds 41/42/43")
    return copy.deepcopy(definition)


def training_profile(config:dict,method:str,seed:int,percentage:int=100,*,budget:dict|None=None,
                     training_component: str | None = None) -> tuple[dict,dict]:
    protocol=config["manuscript_native_protocol"]
    declared={row["method_id"]:row for row in method_roster(config)}
    component = training_component_definition(config, training_component) if training_component else None
    if component is not None:
        if method != component["predictor_method"] or percentage != 100 or budget is not None:
            raise ValueError("Internal predictor requires its unchanged full scientific training profile")
    elif method not in declared or not declared[method]["software_interface_bound"]:
        raise ValueError("Method has no native executable training/evaluation binding: "+method)
    seeds = native_training_seed_values(protocol.get("seed_policy", "manuscript_native_five"), protocol["training_seeds"])
    if type(seed) is not int or seed not in seeds:
        raise ValueError("Profile seed is outside the declared native seeds")
    selected=copy.deepcopy(config)
    comparison=protocol["source_comparison"]
    scope=comparison["scope_methods"].get(method,"all")
    selected["native_scientific_training"]["source_scope"]=scope
    if scope!="all":
        if percentage!=100:raise ValueError("Source comparison has no extra fraction sweep")
        selected["native_diffusion_data"]["field_normalization"]["model_artifact_ref"]=str(
            Path(comparison["normalization_root"])/(scope+".json"))
        budget=None
    if method in protocol["chemical_comparison"]["method_ids"]:
        selected["native_scientific_training"]["family_scope"]=list(protocol["chemical_comparison"]["family_ids"])
        budget=training_budget(selected,percentage)
    else:
        selected["native_scientific_training"]["family_scope"]=list(selected["native_observable_bridge"]["quantity_profile"]["families"])
        budget=training_budget(selected,percentage) if budget is None else budget
    selected.pop("config_includes",None)
    name=f"{training_component or method}_fraction_{percentage}_seed_{seed}"
    profile_path=str(Path(protocol["profile_root"])/(name+".yaml"))
    output_method=(training_component or ("posterior_comparison" if method in selected["native_intervention_evaluation"]["posterior_methods"] else method))
    output=str(Path(protocol["execution_root"])/output_method/f"fraction_{percentage}"/f"seed_{seed}"/"primary")
    common_method=comparison["conditioning_methods"].get(scope,"native_wam")
    common_name=f"{common_method}_fraction_100_seed_{seed}"
    common_config=str(Path(protocol["profile_root"])/(common_name+".yaml"))
    common_manifest=str(Path(protocol["execution_root"])/common_method/"fraction_100"/f"seed_{seed}"/"primary"/"microenvironment_pretraining_manifest.json")
    native=selected["native_scientific_training"]
    native["training_subset_percentage"]=percentage
    native["selection"]["max_epochs"]=budget["max_epochs"]
    native["validation"]["interval_optimizer_steps"]=budget["validation_interval_optimizer_steps"]
    training_method=protocol.get("checkpoint_reuse",{}).get(method,method)
    native["conditioning"]={"mode":"joint"} if training_method in {"native_wam",*comparison["conditioning_methods"].values()} else {
        "mode":"frozen_selected_mainline","config_path":common_config,"manifest_path":common_manifest}
    benchmark=selected["screening_pipeline"]["screening_benchmark"]
    benchmark["artifact_root"]=output
    train=benchmark["training"]
    train.update(random_seed=seed,seed_policy=protocol.get("seed_policy", "manuscript_native_five"))
    train["zarr_loader"]["formal_artifact_subdir_template"]=""
    train["zarr_loader"]["pretraining_epoch_count"]=budget["max_epochs"]
    selected["selected_formal_seed"]=seed
    for key in ("formal_seed_values","seeds"):selected["paper_grade_full_suite"][key]=list(protocol["training_seeds"])
    selected["paper_grade_baseline_registry"]["seed_values"]=list(protocol["training_seeds"])
    variant=protocol["native_method_profiles"].get(method,{})
    scale=variant.get("checkpoint_scale",protocol["reference_scale"])
    inference=variant.get("inference_profile",protocol["reference_inference_profile"])
    learned=selected["native_learned_distillation"]
    learned["enabled"]=method=="student"
    learned["teacher_config"]=common_config if learned["enabled"] else None
    learned["teacher_manifest"]=common_manifest if learned["enabled"] else None
    registry=selected["neuronsifter_scale_registry"]
    registry["checkpoint_scales"][scale]["training_memory_policy"]=copy.deepcopy(train["memory_policy"])
    registry["inference_profiles"][inference]["sampler_budget"].update(
        sample_count=native["predictive_budget"]["conditional_draws_per_particle"],
        denoising_step_count=native["predictive_budget"]["categorical_reverse_steps"],
        training_sample_count=registry["quantized_mddm_main_rollout"]["partial_mask_schedule"]["training_sample_count"],
        training_denoising_step_count=registry["quantized_mddm_main_rollout"]["partial_mask_schedule"]["training_denoising_step_count"])
    registry["quantized_mddm_main_rollout"]["partial_mask_schedule"]["sampler_seed"]=seed
    selected=apply_neuronsifter_scale(selected,scale,inference)
    backbone=selected["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]
    backbone["transition_backbone"]["diffusion_sampler"]["sampler_seed"]=seed
    backbone["transition_backbone"]["quantized_mddm_rollout"].setdefault("runtime_overrides",{})["sampler_seed"]=seed
    if method in protocol["native_method_profiles"]:
        variant=protocol["native_method_profiles"][method]
        backbone["action_representation"]["active_policy"]=variant["action_policy"]
        splitting=backbone["transition_backbone"]["operator_splitting"]
        splitting["composition_mode"]=variant["composition_mode"]
        order=list(splitting["composition_order"])
        if variant["composition_order"]=="reversed":order.reverse()
        elif variant["composition_order"]=="even_then_odd":order=order[::2]+order[1::2]
        elif variant["composition_order"]!="declared":raise ValueError("Unknown native expert ordering")
        splitting["composition_order"]=order
    selected["native_profile_identity"]={"method":method,"percentage":percentage,"seed":seed,
        "protocol_uid":protocol["protocol_uid"],"checkpoint_scale":scale,"inference_profile":inference}
    if training_component:
        selected["native_profile_identity"]["training_component"] = training_component
    scale_dependencies=[]
    if method in selected.get("native_scale_routing",{}).get("methods",[]):
        checkpoints={}
        for stage in selected["native_scale_routing"]["stage_order"]:
            upstream=protocol.get("checkpoint_reuse",{}).get(stage,stage)
            stage_profile=f"{stage}_fraction_100_seed_{seed}"
            checkpoints[stage]={
                "config_path":str(Path(protocol["profile_root"])/(stage_profile+".yaml")),
                "training_manifest":str(Path(protocol["execution_root"])/upstream/"fraction_100"/f"seed_{seed}"/"primary"/"microenvironment_pretraining_manifest.json")}
            scale_dependencies.append(f"{upstream}_fraction_100_seed_{seed}:fit")
        selected["native_scale_routing"]["checkpoints"]=checkpoints
    binding=resolve_training_seed(selected,require_selected=True)
    reused=protocol.get("checkpoint_reuse",{}).get(method)
    manifest_name="microenvironment_pretraining_manifest.json" if method in protocol["native_method_profiles"] else "native_comparator_training_manifest.json"
    specification={"profile_id":name,"method":method,"seed":seed,"training_subset_percentage":percentage,
        "config_path":profile_path,"output_root":output,"checkpoint_scale":scale,"inference_profile":inference,
        "family_scope":selected["native_scientific_training"]["family_scope"],
        "source_scope":scope,"conditioning_method":common_method,"joint_belief_fit":native["conditioning"]["mode"]=="joint",
        "training_manifest":str(Path(output)/manifest_name),"conditioning_config":common_config,
        "conditioning_manifest":common_manifest,"training_seed_binding":binding,"budget":copy.deepcopy(budget),
        "effective_configuration_sha256":content_hash(selected)}
    if training_component:
        specification.update(training_component=training_component, training_owner_method=component["owner_method"])
    if scale_dependencies:
        specification.update(scale_fit_dependencies=list(dict.fromkeys(scale_dependencies)),
                             training_manifest=None,additional_optimizer_updates=0)
        if method=="adaptive_scale":
            specification.update(scale_policy_artifact=str(Path(output)/"policy_fit"/"native_adaptive_scale_policy.json"),
                scale_calibration_artifact=str(Path(output)/"calibration"/"native_scale_selected_calibration.json"))
    if reused:
        reused_manifest="microenvironment_pretraining_manifest.json" if reused in protocol["native_method_profiles"] else "native_comparator_training_manifest.json"
        specification["training_manifest"]=str(Path(protocol["execution_root"])/reused/f"fraction_{percentage}"/f"seed_{seed}"/"primary"/reused_manifest)
        specification["checkpoint_reuse_method"]=reused
        specification["checkpoint_reuse_profile"]=f"{reused}_fraction_{percentage}_seed_{seed}"
        specification["additional_optimizer_updates"]=0
    if method=="same_compiler_solver":
        specification.update(training_manifest=None,additional_optimizer_updates=0,
            transition_fit="not_applicable_frozen_primary_rhs",conditional_paths_per_particle=1)
    return selected,specification


def command_catalog(config:dict,profiles:list[dict]) -> list[dict]:
    protocol=config["manuscript_native_protocol"];commands=[]
    def add(uid,argv,dependencies,resource,outputs,**metadata):
        commands.append({"command_uid":uid,"argv":argv,"shell_command":shlex.join(argv),
            "dependencies":dependencies,"resource_profile":resource,"outputs":outputs,
            "dispatch_authorized":False,"execution_status":"NOT_RUN",**metadata})
    add("source_production_and_base_admission",["python","Libs/scripts/builders/build_controlled_native_source.py",
        "--config",protocol["source_config_path"]],[],"source",[config["native_observable_bridge"]["output_root"]],
        command_gate="complete_frozen_source_definition_and_candidate_execution_authorization",
        completion_predicate="source_receipt.source_status=complete_and_native_consumption.status=complete",
        continuation_argv=["python","Libs/scripts/builders/build_controlled_native_source.py",
            "--config",protocol["source_config_path"],"--continue-source"],
        maximum_groups_per_invocation=config["native_diffusion_data"]["group_storage"].get("maximum_groups_per_invocation"),
        source_budget_scope="cumulative_active_source_clock_across_verified_group_boundaries")
    add("named_parameter_admission",["python","Libs/scripts/builders/build_controlled_native_source.py",
        "--config",protocol["config_path"],"--consume-only"],["source_production_and_base_admission"],"cpu",
        [config["native_diffusion_data"]["registry_path"]])
    decision_policy=config["native_intervention_evaluation"]
    decision_methods=set(decision_policy["posterior_methods"])
    scale_methods=set(config.get("native_scale_routing",{}).get("methods",[]))
    if any(row["method"] in decision_methods|scale_methods|{"same_compiler_solver"} for row in profiles):
        add("public_cns_reference",["python","Libs/exp/baselines/run_external_baseline_bridge.py",
            "--protocol-scope","manuscript-native","--config",protocol["config_path"],
            "--baseline-name","posterior_joint","--command","fit_decision_reference",
            "--output-dir",str(Path(decision_policy["reference_artifact"]).parent),
            "--decision-reference",decision_policy["reference_artifact"]],
            ["named_parameter_admission"],"cpu",[decision_policy["reference_artifact"]],
            reference_role="train")
    scale_methods=set(config.get("native_scale_routing",{}).get("methods",[]))
    means=set(config["native_controlled_comparators"]["conditional_mean_methods"])
    native=set(protocol["native_method_profiles"])
    emitted_posterior=set()
    for row in profiles:
        method=row["method"];uid=row["profile_id"];output=Path(row["output_root"])
        if method in decision_methods:
            key=(row["seed"],row["training_subset_percentage"])
            if key in emitted_posterior:continue
            emitted_posterior.add(key)
        common_uid=f"{row.get('conditioning_method','native_wam')}_fraction_100_seed_{row['seed']}"
        dependencies=["named_parameter_admission"]
        if not row.get("joint_belief_fit",method=="native_wam"):dependencies.append(common_uid+":fit")
        bridge=["python","Libs/exp/baselines/run_external_baseline_bridge.py","--protocol-scope","manuscript-native",
            "--config",row["config_path"],"--baseline-name",method,"--device","cuda",
            "--conditioning-config",row["conditioning_config"],"--conditioning-manifest",row["conditioning_manifest"]]
        if method in scale_methods:
            scale_bridge=[*bridge,"--decision-reference",decision_policy["reference_artifact"]]
            scale_dependencies=[*row["scale_fit_dependencies"],"public_cns_reference"]
            if method=="adaptive_scale":
                add(uid+":fit_policy",[*scale_bridge,"--command","fit_scale_policy",
                    "--output-dir",str(output/"policy_fit")],scale_dependencies,"evaluation",
                    [row["scale_policy_artifact"]],profile_id=uid,additional_optimizer_updates=0)
                scale_bridge.extend(["--scale-policy",row["scale_policy_artifact"]])
                add(uid+":calibrate",[*scale_bridge,"--command","calibrate_scale_policy",
                    "--output-dir",str(output/"calibration")],[uid+":fit_policy"],"evaluation",
                    [row["scale_calibration_artifact"]],profile_id=uid,additional_optimizer_updates=0)
                scale_bridge.extend(["--scale-calibration",row["scale_calibration_artifact"]])
                scale_dependencies=[uid+":calibrate"]
            add(uid+":evaluate",[*scale_bridge,"--command","evaluate_baseline",
                "--output-dir",str(output/"evaluation"),"--evaluation-role","validation"],
                scale_dependencies,"evaluation",
                [str(output/"evaluation"/"native_scale_evaluation.json")],profile_id=uid,
                additional_optimizer_updates=0)
            continue
        if method=="same_compiler_solver":
            bridge[bridge.index("--device")+1]="cpu"
            add(uid+":evaluate",[*bridge,"--command","evaluate_baseline",
                "--decision-reference",decision_policy["reference_artifact"],
                "--output-dir",str(output/"evaluation"),"--evaluation-role","validation"],
                [*dependencies,"public_cns_reference"],"cpu",
                [str(output/"evaluation"/"native_comparator_evaluation.json")],
                profile_id=uid,additional_optimizer_updates=0,conditional_paths_per_particle=1)
            continue
        if method in native:
            fit=["python","Libs/exp/training/pretrain_microenvironment_world_model.py","--config",row["config_path"],
                "--checkpoint-scale",row["checkpoint_scale"],"--inference-profile",row["inference_profile"],
                "--distributed-strategy","single_process","--device","cuda"]
        else:fit=[*bridge,"--command","train_baseline","--output-dir",str(output)]
        if row.get("checkpoint_reuse_profile"):
            dependencies=[row["checkpoint_reuse_profile"]+":fit"]
        elif method!="persistence":
            add(uid+":fit",fit,dependencies,"training",[row["training_manifest"]],
                profile_id=uid,attempt_policy=protocol["attempts"])
            dependencies=[uid+":fit"]
        # Each native_wam fraction owns its fitted belief; external/native ablations use full-data shared belief.
        if row.get("joint_belief_fit",method=="native_wam"):
            bridge[bridge.index("--conditioning-config")+1]=row["config_path"]
            bridge[bridge.index("--conditioning-manifest")+1]=row["training_manifest"]
        if method!="persistence":bridge.extend(["--training-manifest",row["training_manifest"]])
        residual=output/"calibration"/"native_comparator_residual.json"
        if method in means:
            add(uid+":calibrate",[*bridge,"--command","calibrate_baseline","--output-dir",str(output/"calibration")],
                dependencies,"evaluation",[str(residual)],profile_id=uid)
            dependencies=[uid+":calibrate"]
        if method in decision_methods:
            bridge.extend(["--decision-reference",decision_policy["reference_artifact"]])
            dependencies=[*dependencies,"public_cns_reference"]
        evaluate=[*bridge,"--command","evaluate_baseline","--output-dir",str(output/"evaluation"),"--evaluation-role","validation"]
        if method in means:evaluate.extend(["--residual-artifact",str(residual)])
        state_method=row.get("checkpoint_reuse_method",method) in config.get("native_state_comparison",{}).get("fitted_methods",[])
        evaluation_name=("native_intervention_evaluation.json" if method in decision_methods else
                         "native_state_evaluation.json" if state_method else "native_comparator_evaluation.json")
        evaluation_uid=(f"posterior_comparison_fraction_{row['training_subset_percentage']}_seed_{row['seed']}:evaluate"
                        if method in decision_methods else uid+":evaluate")
        outputs=([str(output/"evaluation"/name/evaluation_name) for name in decision_policy["posterior_methods"]]
                 if method in decision_methods else [str(output/"evaluation"/evaluation_name)])
        add(evaluation_uid,evaluate,dependencies,"evaluation",outputs,profile_id=uid)
    # One evidence command covers the full matrix and deduplicates reused fits.
    fit_dependencies = list(dict.fromkeys(row["command_uid"] for row in commands
        if row["command_uid"].endswith(":fit") and "fraction_100_seed_" in row["command_uid"]))
    add("manuscript_native_training_evidence", ["python",
        "Libs/scripts/builders/build_operational_twin_training_sufficiency.py",
        "--protocol-scope", "manuscript-native", "--config", protocol["config_path"],
        "--output-dir", protocol["training_evidence"]["output_root"]],
        fit_dependencies, "cpu", [str(Path(protocol["training_evidence"]["output_root"])/
        "manuscript_native_training_evidence.json")],
        completion_predicate="artifact_integrity_only_independent_sufficiency_review_pending",
        sufficient_training=False, scientific_twin="NOT_RUN")
    return commands


def validate_definition(config:dict) -> dict:
    protocol=config["manuscript_native_protocol"]
    if protocol["schema_version"] != "manuscript_native_protocol.v1":
        raise ValueError("Native protocol schema differs")
    seeds = native_training_seed_values(protocol.get("seed_policy", "manuscript_native_five"), protocol["training_seeds"])
    statistics = config["manuscript_native_statistics"]
    if native_training_seed_values(statistics.get("seed_policy", "manuscript_native_five"), statistics["training_seeds"]) != seeds or statistics.get("seed_policy", "manuscript_native_five") != protocol.get("seed_policy", "manuscript_native_five"):
        raise ValueError("Native protocol and paired-statistics seed policies differ")
    if protocol["scientific_twin"]!="NOT_RUN" or protocol["formal_qualification"] is not False:
        raise ValueError("Protocol definition cannot promote scientific qualification")
    # Evaluation import is delayed to keep the profile/runner import boundary acyclic.
    from Libs.utils.evaluation.native_scale_routing import scale_routing_policy
    scale_routing_policy(config)
    native_residual_online_policy(config)
    future_control_policy(config)
    groups=group_definitions(config);counts=planned_operation_counts(config)
    if any(config["native_observable_bridge"]["budget"][key]!=value for key,value in counts.items()):
        raise ValueError("Prospective source counts differ from the canonical planner")
    reservations={key:{row["sample_uid"] for row in groups if row["calibration_reservation"]==key} for key in ("fit","check","threshold")}
    if any(len(ids)!=2048 for ids in reservations.values()) or any(reservations[a]&reservations[b] for a,b in (("fit","check"),("fit","threshold"),("check","threshold"))):
        raise ValueError("Fit/check/threshold groups must be independently declared at512 per family")
    roles={role:{row["sample_uid"] for row in groups if row["role"]==role} for role in ("train","development","validation")}
    if any(reservations["threshold"]&ids for ids in roles.values()):raise ValueError("Threshold groups leaked into another role")
    if prior_particle_count(config)!=32 or config["native_scientific_training"]["predictive_budget"] ["weighted_forecast_paths_per_action"]!=64:
        raise ValueError("Full prior and forecast budget must retain32 times2 paths")
    rows=method_roster(config)
    required=set(config["schemas"]["manuscript_native_protocol"]["required_method_ids"])
    if {row["method_id"] for row in rows}!=required:raise ValueError("Full manuscript method roster differs from its schema")
    for key in ("masked_diffusion","crps","paired_action_contrast","physical_constraint","teacher_distillation","held_sensing","source_nuisance_invariance","router"):
        value=config["native_scientific_training"]["objectives"].get(key)
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:raise ValueError("Missing finite scientific objective: "+key)
    for key in ("observation_likelihood_weight","categorical_kl_weight"):
        value=config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["joint_objectives"][key]
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:raise ValueError("Unbound ordinary belief objective: "+key)
    rssm=config["native_rssm_prediction"]
    if (rssm["fitted_methods"]!=["planet","dreamerv3"]
            or protocol["checkpoint_reuse"].get("dreamer")!="planet"):
        raise ValueError("Matched Gaussian RSSM predictions must share one fitted identity")
    required=config["schemas"]["manuscript_native_protocol"]["rssm_prediction_required_fields"]
    if set(required)-set(rssm):raise ValueError("RSSM prediction scope/law declarations are incomplete")
    gaussian=config["native_controlled_comparators"]["references"]["planet"]["hyperparameters"]
    discrete=config["native_controlled_comparators"]["references"]["dreamerv3"]["hyperparameters"]
    if (gaussian["observation_stddev"]!=1 or gaussian["overshooting_distance"]!=0
            or gaussian["free_nats"]!=3 or gaussian["divergence_weight"]!=1
            or discrete["dynamics_weight"]!=1 or discrete["representation_weight"]!=.1
            or discrete["unimix"]!=.01 or discrete["absolute_posterior"] is not False
            or discrete["deterministic_size"]%discrete["blocks"]):
        raise ValueError("RSSM default core/loss differs from its declared primary reference")
    learned=config["native_learned_distillation"]
    validate_distillation_policy(learned)
    required=config["schemas"]["manuscript_native_protocol"]["learned_distillation_required_fields"]
    if set(required)-set(learned):
        raise ValueError("Learned distillation declaration is incomplete")
    if protocol["checkpoint_reuse"].get("large")!="native_wam":
        raise ValueError("Large must reuse the selected native WAM without another fit")
    for method in ("student","student_no_distillation","large","xl"):
        if method not in protocol["native_method_profiles"]:
            raise ValueError("Native scale profile is missing: "+method)
    state=config["native_state_comparison"]
    blocks=[index for block in state["readout_blocks"] for index in block]
    profile=observation_feature_profile(config)
    if profile["status"]!="bound":raise ValueError("Manuscript requires an explicit observation feature profile")
    width=profile["observation_feature_width"]
    if sorted(blocks)!=list(range(width)) or len(set(blocks))!=len(blocks):
        raise ValueError("State readout blocks must partition the registered feature width")
    if (set(state["objectives"])!={"retained_readout","held_readout","categorical_kl"}
            or any(not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or value<=0
                   for value in state["objectives"].values())):
        raise ValueError("State probe objectives must be positive finite observation losses")
    for method in ("state_single_source","state_mixed_source"):
        if method in state.get("checkpoint_aliases",{}) or method in protocol.get("checkpoint_reuse",{}):
            raise ValueError("Matched source state controls require independent source-isolated fits")
    training_sources=sorted({(row["source_uid"],row["source_snapshot_ref"],row["source_law"])
                             for row in groups if row["role"]=="train"})
    return {"effective_stimulation_actions":effective_stimulation_actions(config,groups),
        "counts":counts,"role_group_counts":dict(Counter(row["role"] for row in groups)),
        "defined_training_source_contracts":[list(row) for row in training_sources],
        "defined_training_source_contract_count":len(training_sources),
        "source_admission_evaluated":False,
        "multiple_training_source_contracts_defined":len(training_sources)>=2,
        "calibration_group_counts":{key:len(value) for key,value in reservations.items()},
        "source_law_counts":dict(Counter(row["source_law"] for row in groups)),
        "partitions":copy.deepcopy(config["native_observable_bridge"]["group_partitions"]),
        "independent_group_identity_rule":"family_partition_replicate_before_outcomes",
        "threshold_excluded_from_training_normalization_selection_and_residual_fit_check":True}


def build_package(config:dict,root:Path,authoritative_date:str,*,require_committed:bool=True) -> tuple[dict,dict[str,dict]]:
    date.fromisoformat(authoritative_date)
    protocol=config["manuscript_native_protocol"]
    source=validate_definition(config)
    roster=method_roster(config)
    budgets={fraction:training_budget(config,fraction) for fraction in protocol["fraction_axis"]["percentages"]}
    configurations={};profiles=[]
    for row in roster:
        if not row["software_interface_bound"]:continue
        fractions=protocol["fraction_axis"]["percentages"] if row["method_id"] in protocol["fraction_axis"]["method_ids"] else [100]
        for percentage in fractions:
            for seed in protocol["training_seeds"]:
                effective,spec=training_profile(config,row["method_id"],seed,percentage,budget=budgets[percentage])
                configurations[spec["config_path"]]=effective;profiles.append(spec)
    components = {}
    for component in protocol["training_evidence"].get("predictor_components", {}):
        definition = training_component_definition(config, component)
        component_profiles = []
        for seed in protocol["training_seeds"]:
            effective, spec = training_profile(config, definition["predictor_method"], seed,
                                               training_component=component)
            configurations[spec["config_path"]] = effective
            component_profiles.append(spec)
        components[component] = {"definition": definition, "profiles": component_profiles,
            "commands": [row for row in command_catalog(config, component_profiles)
                         if row["command_uid"].endswith(":fit")],
            "scope": "internal_predictor_training_dependency", "full_method_qualified": False}
    paths=list(dict.fromkeys([protocol["config_path"],protocol["source_config_path"],
        *config["native_observable_bridge"]["freeze_paths"],*protocol["source_and_code_binding_paths"],
        *[path for entry in config["native_controlled_comparators"]["references"].values() for path in entry.get("source_modules",[])]]))
    git=["git","-c","gc.auto=0","-c","maintenance.auto=false"]
    commit=subprocess.check_output([*git,"rev-parse","HEAD"],cwd=root,text=True).strip()
    identities={}
    for relative in paths:
        data=(root/relative).read_bytes()
        if require_committed:
            old=subprocess.check_output([*git,"show",commit+":"+relative],cwd=root)
            if data!=old:raise ValueError("Commit native source/config/code before protocol publication: "+relative)
        identities[relative]=hashlib.sha256(data).hexdigest()
    blocked=[row for row in roster if row["implementation_kind"]=="controlled_adapter_missing"]
    design_blockers=[] if source["multiple_training_source_contracts_defined"] else [{
        "method_id":"state_single_source","contrast":"multisource_pretraining_benefit",
        "blocker":config["native_state_comparison"]["single_source"]["required_contrast_prerequisite"],
        "defined_training_source_contract_count":len(source["defined_training_source_contracts"])}]
    package={"schema_version":protocol["schema_version"],"protocol_uid":protocol["protocol_uid"],
        "authoritative_date":authoritative_date,"source_commit":commit,"source_files":identities,
        "effective_configuration_sha256":content_hash(config),"definition_validation":"PASS",
        "source_definition":source,"training_budgets":budgets,"methods":roster,"profiles":profiles,
        "training_components":components,
        "commands":command_catalog(config,profiles),"controlled_matrix_status":"BLOCKED" if blocked or design_blockers else "DEFINED_AWAITING_EXECUTION",
        "controlled_design_blockers":design_blockers,
        "controlled_implementation_blockers":[{"method_id":row["method_id"],"blocker":row["implementation_blocker"]} for row in blocked],
        "execution_prerequisites":protocol["execution_prerequisites"],"scientific_twin":"NOT_RUN",
        "formal_qualification":False,"dispatch_authorized":False,"executed_source_operations":0,
        "metrics":copy.deepcopy(protocol["metrics"]),"statistics":copy.deepcopy(protocol["statistics"]),
        "resources":copy.deepcopy(protocol["resources"]),"source_resource_sufficiency":"UNMEASURED_AT_FULL_PRIOR",
        "protected_access_counters":{name:0 for name in ("sealed","protected","six_context","AgentTrace","Agent","Tool","Memory","CoEvolution")}}
    package["protocol_content_sha256"]=content_hash(package)
    return package,configurations


def publication_config(config:dict,output_dir:Path|None) -> dict:
    if output_dir is None:return config
    selected=copy.deepcopy(config)
    selected["manuscript_native_protocol"]["artifact_root"]=str(output_dir.resolve())
    selected["manuscript_native_protocol"]["profile_root"]=str(output_dir.resolve()/"profiles")
    return selected


def publish_package(config:dict,root:Path,authoritative_date:str,*,output_dir:Path|None=None) -> dict:
    config=publication_config(config,output_dir)
    package,profiles=build_package(config,root,authoritative_date)
    output=root/config["manuscript_native_protocol"]["artifact_root"]
    if (output/"protocol.json").exists():raise FileExistsError("Native protocol publication already exists")
    for relative,payload in profiles.items():
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(yaml.safe_dump(payload,sort_keys=False))
    output.mkdir(parents=True,exist_ok=True)
    (output/"protocol.json").write_text(json.dumps(package,indent=2,sort_keys=True,allow_nan=False)+"\n")
    (output/"commands.json").write_text(json.dumps(package["commands"],indent=2,sort_keys=True)+"\n")
    return {"status":"PASS","definition_validation":"PASS","controlled_matrix_status":package["controlled_matrix_status"],
        "profile_count":len(profiles),"command_count":len(package["commands"]),"protocol_path":str(output/"protocol.json"),
        "scientific_twin":"NOT_RUN","formal_qualification":False,"dispatch_authorized":False}


def validate_package(config:dict,root:Path,authoritative_date:str,*,output_dir:Path|None=None) -> dict:
    config=publication_config(config,output_dir)
    expected,profiles=build_package(config,root,authoritative_date)
    output=root/config["manuscript_native_protocol"]["artifact_root"]
    actual=json.loads((output/"protocol.json").read_text())
    if actual!=json.loads(json.dumps(expected)):raise ValueError("Published native protocol differs from its committed definition")
    if json.loads((output/"commands.json").read_text())!=expected["commands"]:raise ValueError("Command catalog differs from native protocol")
    for relative,payload in profiles.items():
        if yaml.safe_load((root/relative).read_text())!=payload:raise ValueError("Effective native profile changed: "+relative)
    return {"status":"PASS","definition_validation":"PASS","controlled_matrix_status":expected["controlled_matrix_status"],
        "profile_count":len(profiles),"scientific_twin":"NOT_RUN","formal_qualification":False}
