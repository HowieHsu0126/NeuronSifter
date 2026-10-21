"""Native-comparator dataset roles, checkpoint evaluation and residual calibration."""
from __future__ import annotations

from Libs.utils.exp.experiment_manifests import native_training_seed_values

import copy
import json
import math

import numpy as np
from pathlib import Path

import torch

from Libs.utils.model.native_training_selection import training_checkpoint_path
from Libs.exp.training.pretrain_microenvironment_world_model import NativeJointGroupDataset,MicroenvironmentPretrainingModel
from Libs.models.modules.native_field_comparators import NativeWorldActionComparator
from Libs.utils.model.neuronsifter_scale_registry import apply_neuronsifter_scale
from Libs.utils.data.raw_data_registry import write_json,resolve_repo_path
from Libs.utils.exp.experiment_manifests import resolve_training_seed,validate_training_seed_checkpoint
from Libs.utils.exp.native_learned_distillation import validate_saved_learned_teacher
from Libs.utils.exp.native_comparator_training import (
    load_frozen_native_belief,prepare_comparator_group,make_comparator,
    comparator_identity,comparator_mean,
)
from Libs.utils.model.native_direct_mean import file_sha256
from Libs.utils.model.native_scientific_source import group_definitions,selected_group_definitions,native_training_profile,native_pretrainer_method
from Libs.utils.model.native_scientific_objectives import stratified_mixture_crps
from Libs.utils.model.native_transolver_residual import (
    CONTROLLED_LAW,episode_statistics,fit_scales,residual_draws,check_diagnostics,
)
from Libs.utils.model.dynamics_fields import node_adjacency_matrix
from Libs.utils.data.native_diffusion_data import load_family_registries, load_joint_sources
from Libs.utils.model.native_comparator_inputs import load_native_prediction_inputs, build_native_comparator_inputs
from Libs.utils.evaluation.native_intervention_decisions import (
    NativePublicCNSLoss, compiled_action_burden, summarize_posterior_action_loss,
)
from Libs.utils.evaluation.native_intervention_evaluation import _ranking, loss_intervals


def declared_dataset(config:dict,role:str,*,reservation:str|None=None):
    selected=selected_group_definitions(config,role,reservation)
    ids={row["sample_uid"] for row in selected}
    return NativeJointGroupDataset(config,role,group_uids=ids),{row["sample_uid"]:row for row in selected}


def load_native_comparator(config:dict,method:str,conditioning_config:Path,
                           conditioning_manifest:Path,training_manifest:Path,dataset,device,*,frozen_conditioning=None):
    # Native scale routing reuses the common belief already checked by this loader.
    belief,conditioning=(load_frozen_native_belief(config,conditioning_config,conditioning_manifest,device)
                         if frozen_conditioning is None else frozen_conditioning)
    identity=comparator_identity(config,method,conditioning)
    if method=="same_compiler_solver":
        if training_manifest is not None:
            raise ValueError("Frozen primary RHS has no transition checkpoint")
        model=make_comparator(config,method,None,conditioning).to(device).eval()
        return model,belief,{"model_identity":identity,"checkpoint_path":None,"checkpoint_sha256":None,
            "training_manifest":None,"normalization_id":belief.field_normalization.identity,
            "transition_fit":"not_applicable_frozen_primary_rhs","optimizer_updates":0}
    if method=="persistence":
        inputs,_,_,_=prepare_comparator_group(dataset[0]["native_joint_group"],belief,config,device)
        model=make_comparator(config,method,inputs,conditioning).to(device)
        return model,belief,{"model_identity":identity,"checkpoint_path":None,"checkpoint_sha256":None,
            "training_manifest":None,"normalization_id":belief.field_normalization.identity,
            "transition_fit":"not_applicable_identity_dynamics"}
    report=json.loads(training_manifest.read_text())
    if native_pretrainer_method(config,method):
        selection=report.get("selection",{})
        if selection.get("selection_role")!="development" or not selection.get("best"):
            raise ValueError("Native world-action comparison requires development-selected training")
        if report.get("training_seed_binding")!=resolve_training_seed(config,require_selected=True):
            raise ValueError("Native world-action selected seed mismatch")
        effective=apply_neuronsifter_scale(config,report["checkpoint_scale"],report["inference_profile"]) if "checkpoint_scale" in report else config
        path=training_checkpoint_path(training_manifest,selection,selection["best"]["checkpoint_path"])
        joint_fit = effective["native_scientific_training"].get("conditioning",{"mode":"joint"})["mode"] == "joint"
        if joint_fit and (str(path)!=conditioning["checkpoint_path"]
                or file_sha256(path)!=conditioning["checkpoint_sha256"]):
            raise ValueError("Native WAM evaluation must use its identical selected common checkpoint")
        checkpoint=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
        validate_training_seed_checkpoint(checkpoint,resolve_training_seed(config,require_selected=True))
        expected_profile=native_training_profile(effective,belief.field_normalization.identity)
        if (checkpoint.get("native_scientific_profile")!=expected_profile
                or report.get("native_scientific_profile")!=expected_profile):
            raise ValueError("Native world-action checkpoint/profile or training-domain mismatch")
        if not joint_fit and checkpoint.get("native_frozen_conditioning")!=conditioning:
            raise ValueError("Native ablation was not fitted against the common selected belief")
        validate_saved_learned_teacher(checkpoint,report,effective,conditioning)
        mainline=MicroenvironmentPretrainingModel(effective,False).to(device)
        mainline.load_state_dict(checkpoint["model_state_dict"],strict=True)
        mainline.requires_grad_(False).eval()
        model=NativeWorldActionComparator(mainline,effective)
        return model,belief,{"model_identity":identity,"checkpoint_path":str(path),
            "checkpoint_sha256":file_sha256(path),"training_manifest":str(training_manifest),
            "normalization_id":belief.field_normalization.identity}
    if report.get("identity")!=identity or report.get("selection",{}).get("selection_role")!="development":
        raise ValueError("Comparator training manifest identity/selection mismatch")
    best=report["selection"].get("best")
    if not best:raise ValueError("Comparator has no development-selected checkpoint")
    path=training_checkpoint_path(training_manifest,report["selection"],best["checkpoint_path"])
    checkpoint=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
    validate_training_seed_checkpoint(checkpoint,resolve_training_seed(config,require_selected=True))
    if checkpoint.get("native_comparator")!=identity:
        raise ValueError("Comparator checkpoint identity mismatch")
    inputs=None
    if method == "mechanistic_residual":
        # Factory dimensions come from admitted public support, never a held
        # target array. The evaluator loads target fields only after forecasting.
        child, positions, _ = dataset.groups[0]
        teacher = child.native_records[positions[0]]["teacher"]
        uid = teacher["posterior_uid"]
        raw = load_joint_sources(child.native_registry, [uid], input_locations=dataset.input_locations)[uid]
        posterior = belief(raw)
        inputs = build_native_comparator_inputs(raw, posterior, belief.field_normalization, config)
    elif method not in config.get("native_state_comparison",{}).get("fitted_methods",[]):
        inputs,_,_,_=prepare_comparator_group(dataset[0]["native_joint_group"],belief,config,device)
    model=make_comparator(config,method,inputs,conditioning,belief=belief).to(device)
    model.load_state_dict(checkpoint["model_state_dict"],strict=True);model.eval()
    binding={"model_identity":identity,"checkpoint_path":str(path),"checkpoint_sha256":file_sha256(path),
             "training_manifest":str(training_manifest),"normalization_id":belief.field_normalization.identity}
    return model,belief,binding


def weighted_native_scores(samples:torch.Tensor,weights:torch.Tensor,target:torch.Tensor,
                           mask:torch.Tensor,normalizer,nominal_levels:list[float],*,
                            deterministic_conditional:bool=False) -> dict:
    """Exact mixture-stratum finite-ensemble CRPS and weighted quantiles."""
    if samples.ndim!=7 or samples.shape[2:]!=target.shape or mask.shape!=target.shape:
        raise ValueError("Native scores need particle/draw/action/time/grid axes")
    normalized=normalizer.normalize(samples,mask)
    actual=normalizer.normalize(target,mask)
    j,d,a=samples.shape[:3]
    if deterministic_conditional:
        if (d!=1 or weights.shape!=(j,) or mask.dtype!=torch.bool
                or not mask.flatten(1).any(-1).all()
                or not all(torch.isfinite(value).all() for value in (normalized,actual,weights))
                or (weights<0).any() or not torch.allclose(weights.sum(),weights.new_tensor(1.))):
            raise ValueError("Deterministic conditional scores need one path per complete weighted tuple")
    else:
        scores=stratified_mixture_crps(normalized,normalized.new_zeros(j,d,a),weights,actual,mask)
    mass=(weights.to(device=normalized.device,dtype=torch.float64) if deterministic_conditional
          else weights.to(normalized)).repeat_interleave(d)/d
    if deterministic_conditional:
        mass=mass/mass.sum()
    pool=normalized.flatten(0,1)
    average=(pool*mass.reshape(-1,*([1]*(pool.ndim-1)))).sum(0)
    rmse=torch.where(mask,(average-actual).square(),0).sum()/mask.sum()
    ordered,indices=pool.sort(dim=0)
    expanded=mass.reshape(-1,*([1]*(pool.ndim-1))).expand_as(pool)
    sorted_mass=expanded.gather(0,indices)
    cdf=sorted_mass.cumsum(0)
    if deterministic_conditional:
        first=((pool-actual).abs()*expanded).sum(0)
        half_pair=(ordered*sorted_mass*(2*cdf-sorted_mass-mass.sum())).sum(0)
        pointwise=first-half_pair
        scores=torch.where(mask,pointwise,0).flatten(1).sum(-1)/mask.flatten(1).sum(-1)
    coverages={}
    for level in nominal_levels:
        if not 0<level<1:raise ValueError("Coverage level must be a probability")
        quantiles=[]
        for q in ((1-level)/2,1-(1-level)/2):
            position=(cdf<q).sum(0).clamp_max(j*d-1)
            quantiles.append(ordered.gather(0,position[None]).squeeze(0))
        coverage=((actual>=quantiles[0]) & (actual<=quantiles[1]) & mask).sum()/mask.sum()
        coverages[str(level)]=float(coverage)
    return {"crps":float(scores.mean()),"normalized_rmse":float(rmse.sqrt()),
            "predictive_coverage_by_nominal":coverages}


def native_residual_partitions(config: dict) -> dict:
    """Read declared independent reservations without loading a teacher array."""
    policy = config["native_controlled_comparators"]["residual_calibration"]
    if (policy["law"] != CONTROLLED_LAW or policy["species_count"] != 26
            or policy["fit_role"] != "calibration" or policy["fit_reservation"] != "fit"
            or policy["check_reservation"] != "check"):
        raise ValueError("Unsupported native26 residual calibration role/law policy")
    result = {}
    for reservation in ("fit", "check"):
        try:
            rows = selected_group_definitions(config, "calibration", reservation)
        except ValueError as error:
            raise ValueError("Invalid or missing declared independent calibration-"+reservation+" groups") from error
        result[reservation] = {row["sample_uid"]: row for row in rows}
        if (not rows or len(result[reservation]) != len(rows)
                or any(row["role"] != "calibration" or row["calibration_reservation"] != reservation for row in rows)):
            raise ValueError("Incomplete or misassigned native calibration reservation")
    threshold = {row["sample_uid"] for row in group_definitions(config, role_filter="calibration")
                 if row["role"] == "calibration" and row["calibration_reservation"] == "threshold"}
    fit_ids, check_ids = set(result["fit"]), set(result["check"])
    if fit_ids & check_ids or (fit_ids | check_ids) & threshold:
        raise ValueError("Native residual fit/check/threshold reservations overlap")
    return result


def admit_native_residual_calibration(config: dict, fit, check) -> dict:
    """Validate complete dataset metadata before either calibration role is read."""
    partitions = native_residual_partitions(config)
    for reservation, dataset in (("fit", fit), ("check", check)):
        declared = partitions[reservation]
        ids = dataset.group_uids
        if (dataset.split_name != "calibration" or len(ids) != len(set(ids))
                or set(ids) != set(declared) or len(dataset.groups) != len(ids)):
            raise ValueError("Residual calibration dataset differs from its complete declared "+reservation+" role")
        for uid, (child, positions, _) in zip(ids, dataset.groups):
            rows = [child.native_records[position]["teacher"] for position in positions]
            expected = declared[uid]
            if (not rows or any(row["sample_uid"] != uid or row["split_name"] != "calibration"
                    or row["source_holdout_role"] != "calibration"
                    or row.get("calibration_reservation") != reservation
                    or row["family_uid"] != expected["family_uid"] for row in rows)):
                raise ValueError("Residual calibration metadata has another group/role/reservation")
            if (len({row["regimen_uid"] for row in rows}) != len(rows)
                    or ("action_indices" in expected and len(rows) != len(expected["action_indices"]))):
                raise ValueError("Residual calibration needs each complete paired action group")
    return partitions


def _validate_native_residual_summaries(artifact: dict, config: dict, partitions: dict) -> None:
    """Check the existing diagnostic structure, never calibration quality thresholds."""
    def count(value):
        if type(value) is not int or value < 0:
            raise ValueError("Residual diagnostic counts must be nonnegative integers")
        return value

    def mean_summary(summary):
        if not isinstance(summary, dict):
            raise ValueError("Residual PIT summary must be an object")
        n = count(summary["count"])
        value = summary["mean"]
        if ((n == 0 and value is not None) or (n > 0 and
                (type(value) not in (int, float) or not math.isfinite(value)))):
            raise ValueError("Residual PIT mean must be finite when observed and None when missing")
        return n

    species = [row["quantity_id"] for row in config["native_observable_bridge"]["quantity_profile"]["quantities"]]
    covariance_kinds = {"adjacent_retained_times", "graph_edges", "cross_species_pairs"}
    for row in artifact["checks"]:
        pit, covariance = row["pit"], row["covariance"]
        if (not isinstance(pit, dict) or not isinstance(covariance, dict)
                or set(covariance) != covariance_kinds):
            raise ValueError("Residual checks need PIT and the three declared covariance diagnostics")
        by_species = pit["by_species"]
        if (not isinstance(by_species, list) or len(by_species) != 26 or len(species) != 26
                or any(not isinstance(item, dict) for item in by_species)
                or [item["species_id"] for item in by_species] != species
                or count(pit["t0_continuous_pit_count"]) != 0
                or pit["kind"] != "exact_marginal_mixture_cdf"):
            raise ValueError("Residual PIT must preserve the 26 quantity identities and t0 atom")
        if mean_summary(pit["future"]) != sum(mean_summary(item) for item in by_species):
            raise ValueError("Residual PIT future and per-quantity counts disagree")
        for summary in covariance.values():
            if not isinstance(summary, dict):
                raise ValueError("Residual covariance summary must be an object")
            declared = count(summary["declared_stratum_count"])
            measured = count(summary["measured_stratum_count"])
            missing = count(summary["missing_stratum_count"])
            n = count(summary["count"])
            if declared != measured+missing or measured > n or ((n == 0) != (measured == 0)):
                raise ValueError("Residual covariance stratum/coordinate counts disagree")
            for key in ("centered_product_mean", "total_covariance_mean"):
                value = summary[key]
                if ((n == 0 and value is not None) or (n > 0 and
                        (type(value) not in (int, float) or not math.isfinite(value)))):
                    raise ValueError("Residual covariance means must be finite when observed and None when missing")
    information = artifact["counts"]
    if not isinstance(information, dict):
        raise ValueError("Residual information counts must be an object")
    groups, episodes = information["groups"], information["episodes"]
    if (not isinstance(groups, list) or not isinstance(episodes, list)
            or any(not isinstance(row, dict) for row in groups+episodes)
            or len(groups) != len(partitions["fit"])
            or {row["group_uid"] for row in groups} != set(partitions["fit"])):
        raise ValueError("Residual information does not cover its declared independent fit groups")
    for episode in episodes:
        n = count(episode["future_coordinate_count"])
        if episode["group_uid"] not in partitions["fit"] or episode["eligible"] is not (n > 0):
            raise ValueError("Residual information episode eligibility/group is inconsistent")
    eligible = [row for row in episodes if row["eligible"]]
    if (count(information["declared_group_count"]) != len(groups)
            or count(information["declared_episode_count"]) != len(episodes)
            or count(information["eligible_episode_count"]) != len(eligible)
            or count(information["eligible_group_count"]) != len({row["group_uid"] for row in eligible})
            or count(information["future_coordinate_count"]) != sum(row["future_coordinate_count"] for row in episodes)
            or information["future_coordinate_count"] != sum(count(value) for value in information["per_species_future_coordinate_count"])):
        raise ValueError("Residual information totals disagree with its episodes/quantities")
    for group in groups:
        members = [row for row in episodes if row["group_uid"] == group["group_uid"]]
        if (count(group["declared_episode_count"]) != len(members)
                or count(group["eligible_episode_count"]) != sum(row["eligible"] for row in members)):
            raise ValueError("Residual information group counts disagree with its episodes")


def load_native_comparator_residual(path: Path, config: dict, binding: dict) -> dict:
    """Validate the existing controlled26 fit/check contract for both consumers."""
    partitions = native_residual_partitions(config)
    policy = config["native_controlled_comparators"]["residual_calibration"]
    artifact = json.loads(Path(path).read_text())
    if not isinstance(artifact, dict):
        raise ValueError("Native26 residual artifact must be an object")
    required = {"schema_version", "law", "binding", "status", "scales", "counts", "diagnostics",
                "fit_group_uids", "check_group_uids", "reference_time_hours", "numerical", "checks",
                "calibration_check_complete"}
    if (not required <= set(artifact) or artifact["schema_version"] != "native_comparator_residual.v1"
            or artifact["law"] != CONTROLLED_LAW or artifact["binding"] != binding
            or artifact["status"] != "FIT_COMPLETE" or artifact["calibration_check_complete"] is not True
            or artifact["reference_time_hours"] != policy["reference_time_hours"]
            or artifact["numerical"] != policy["numerical"]):
        raise ValueError("Residual artifact schema, predictor/normalizer binding or numerical policy mismatch")
    for reservation in ("fit", "check"):
        supplied = artifact[reservation+"_group_uids"]
        if (not isinstance(supplied, list) or len(supplied) != len(set(supplied))
                or set(supplied) != set(partitions[reservation])):
            raise ValueError("Residual artifact differs from its complete declared "+reservation+" groups")
    checks = artifact["checks"]
    check_fields = {"group_uid", "pit", "covariance", "scientific_check_pass", "interpretation"}
    if (not isinstance(checks, list) or len(checks) != len(partitions["check"])
            or any(not isinstance(row, dict) or not check_fields <= set(row) for row in checks)
            or {row.get("group_uid") for row in checks} != set(partitions["check"])
            or any(row.get("scientific_check_pass") is not None for row in checks)):
        raise ValueError("Native residual check requires all independent diagnostics; it has no scientific PASS threshold")
    try:
        _validate_native_residual_summaries(artifact, config, partitions)
        scales = torch.as_tensor(artifact["scales"], dtype=torch.float64)
        counts, diagnostics = artifact["counts"], artifact["diagnostics"]
        eigenvalues = torch.as_tensor(diagnostics["hessian_eigenvalues"], dtype=torch.float64)
        coordinates = counts["per_species_future_coordinate_count"]
        if (scales.shape != (26,) or eigenvalues.shape != (26,)
                or not torch.isfinite(scales).all() or not (scales > 0).all()
                or not torch.isfinite(eigenvalues).all()):
            raise ValueError("Residual law requires 26 positive finite scales and finite Hessian eigenvalues")
        numerical = policy["numerical"]
        threshold = max(numerical["hessian_absolute_threshold"],
                        numerical["hessian_relative_threshold"]*float(eigenvalues.max()))
        if (len(coordinates) != 26 or any(type(value) is not int or value <= 0 for value in coordinates)
                or counts["declared_group_count"] != len(partitions["fit"])
                or counts["eligible_group_count"] != counts["declared_group_count"]
                or not 0 <= diagnostics["gradient_max_abs"] <= numerical["gradient_tolerance"]
                or not float(eigenvalues.min()) > threshold
                or diagnostics["hessian_required_minimum"] != threshold
                or not math.isfinite(diagnostics["objective"])):
            raise ValueError("Residual artifact lacks complete converged positive identifiable information")
    except (KeyError, TypeError, RuntimeError) as error:
        raise ValueError("Malformed native26 residual numerical evidence") from error
    return artifact


def draw_native_comparator_residual(config: dict, inputs, means: torch.Tensor,
                                    normalization, artifact: dict) -> torch.Tensor:
    """Draw the declared unbounded readout law, without altering mechanical states."""
    b, j, a = inputs.group_shape
    expected = (j, a, *inputs.validity.shape[1:])
    if (b != 1 or j != 32 or means.shape != expected or expected[-1] != 26
            or normalization.identity != inputs.normalization_id
            or artifact["binding"]["normalization_id"] != normalization.identity):
        raise ValueError("Native residual draws require all32 tuples, complete native26 actions/time and shared normalization")
    original = inputs.posterior.support.initial_field
    if original.shape != (b, j, *means.shape[-3:]):
        raise ValueError("Residual t0 needs the original complete 32-tuple public initial field")
    mask = inputs.validity.reshape(b, j, a, *inputs.validity.shape[1:])[0]
    normalized = normalization.normalize(means, mask).cpu()
    d = config["native_scientific_training"]["predictive_budget"]["conditional_draws_per_particle"]
    if type(d) is not int or d <= 0:
        raise ValueError("Native residual conditional draw budget must be a positive integer")
    indices = torch.arange(j).repeat_interleave(d)
    samples = residual_draws(normalized, artifact["scales"], inputs.grid["time_hours"],
        artifact["reference_time_hours"], indices, episode_uid=inputs.sample_uids[0],
        residual_seed=config["native_controlled_comparators"]["residual_calibration"]["residual_seed"],
        draw_ids=list(range(j*d)), law=CONTROLLED_LAW)
    samples = normalization.inverse(samples, mask.cpu()[indices]).to(means.device)
    original = original[0, :, None].expand(j, a, *means.shape[-3:]).to(samples)
    anchored = torch.where(mask[:, :, 0].to(samples.device), original, 0.)
    samples[:, :, 0] = anchored[indices.to(samples.device)]
    return samples.reshape(j, d, *means.shape[1:])


def calibrate_native_comparator(config,method,model,belief,binding,fit,check,output:Path):
    """Fit only on calibration-fit; check once without any refitting."""
    if method not in config["native_controlled_comparators"]["conditional_mean_methods"]:
        raise ValueError("Brownian residual calibration is restricted to native conditional-mean comparators")
    admit_native_residual_calibration(config,fit,check)
    policy=config["native_controlled_comparators"]["residual_calibration"]
    path=output/"native_comparator_residual.json"
    if path.exists():raise FileExistsError("Residual fit already exists for this attempt")
    declared=[]
    normalizer=belief.field_normalization
    rows=[]
    for index in range(len(fit)):
        group=fit[index]["native_joint_group"]
        inputs,target,mask,_=prepare_comparator_group(group,belief,config,next(belief.parameters()).device)
        with torch.no_grad():_,means=comparator_mean(model,method,inputs,normalizer,config,0)
        uid=inputs.sample_uids[0];declared.append(uid)
        statistics=episode_statistics(
            normalizer.normalize(means[0],mask[0]).cpu(),inputs.weights[0].cpu(),
            normalizer.normalize(target[0],mask[0]).cpu(),mask[0].cpu(),
            inputs.grid["time_hours"],policy["reference_time_hours"],uid,uid)
        rows.append(statistics)
    fitted=fit_scales(rows,declared,policy["numerical"],species_count=26)
    artifact={"schema_version":"native_comparator_residual.v1","law":CONTROLLED_LAW,
        "binding":binding,"fit_group_uids":declared,"check_group_uids":[],
        "reference_time_hours":policy["reference_time_hours"],"numerical":policy["numerical"],
        "scientific_twin":"NOT_RUN","formal_qualification":False,**fitted,"checks":[]}
    write_json(path,artifact)
    if fitted["status"]!="FIT_COMPLETE":return artifact
    scales=torch.tensor(fitted["scales"],dtype=torch.float64)
    for index in range(len(check)):
        inputs,target,mask,_=prepare_comparator_group(check[index]["native_joint_group"],belief,config,
                                                      next(belief.parameters()).device)
        uid=inputs.sample_uids[0]
        if uid in declared:raise ValueError("Residual fit/check groups overlap")
        with torch.no_grad():_,means=comparator_mean(model,method,inputs,normalizer,config,0)
        diagnostics=check_diagnostics(
            normalizer.normalize(means[0],mask[0]).cpu(),inputs.weights[0].cpu(),
            normalizer.normalize(target[0],mask[0]).cpu(),mask[0].cpu(),inputs.grid["time_hours"],
            policy["reference_time_hours"],scales,node_adjacency_matrix(config),inputs.grid["species_id"])
        covariance={}
        for kind,entries in diagnostics["covariance"].items():
            measured=[row for row in entries if row["count"]]
            count=sum(row["count"] for row in measured)
            covariance[kind]={"declared_stratum_count":len(entries),"measured_stratum_count":len(measured),
                "missing_stratum_count":len(entries)-len(measured),"count":count,
                "centered_product_mean":sum(row["count"]*row["centered_product_mean"] for row in measured)/count if count else None,
                "total_covariance_mean":sum(row["count"]*row["total_covariance_mean"] for row in measured)/count if count else None}
        artifact["check_group_uids"].append(uid)
        artifact["checks"].append({"group_uid":uid,"pit":diagnostics["pit"],"covariance":covariance,
            "scientific_check_pass":None,"interpretation":diagnostics["interpretation"]})
    artifact["calibration_check_complete"]=len(artifact["checks"])==len(check)
    write_json(path,artifact)
    return artifact


def evaluate_native_comparator(config,method,model,belief,binding,dataset,declarations,output:Path,
                               residual_path:Path|None=None,*,decision_reference:dict|None=None):
    if method=="same_compiler_solver":
        if residual_path is not None or decision_reference is None:
            raise ValueError("Frozen RHS evaluation needs a public reference and no residual calibration")
        return _evaluate_native_solver(config,model,belief,binding,dataset,declarations,output,decision_reference)
    policy=config["native_controlled_comparators"]
    residual=None
    if residual_path is not None:
        residual=load_native_comparator_residual(residual_path,config,binding)
    seed=int(resolve_training_seed(config,require_selected=True)["model_seed"])
    rows=[]
    report={"schema_version":"native_comparator_evaluation.v1","method":method,"binding":binding,
        "declared_group_count":len(dataset),"groups":rows,"scientific_twin":"NOT_RUN","formal_qualification":False}
    if residual is not None:
        report["residual_calibration"]={"artifact_path":str(residual_path),"law":residual["law"],
            "fit_status":residual["status"],"calibration_check_complete":True,"scientific_check_pass":None,
            "physical_state_certification":"not_certified_unclipped_teacher_readout"}
    if method=="treatment_history":
        report.update(predictive_law=model.prediction_kind,method_mapping=model.method_mapping,
                      reported_uncertainty_calibrated=False)
    if method in config.get("native_rssm_prediction",{}).get("fitted_methods",[]):
        report.update(predictive_law=model.prediction_kind,reported_uncertainty_calibrated=False,
            source_readout_likelihood=config["native_rssm_prediction"]["observation_measurement_law"],
            behavior_learning_implemented=False)
    path=output/"native_comparator_evaluation.json"
    if path.exists():raise FileExistsError("Evaluation attempt already exists")
    write_json(path,report)
    for index in range(len(dataset)):
        declared_uid=dataset.group_uids[index]
        declaration=declarations[declared_uid]
        identity={"status":"started","dataset_index":index,"group_uid":declared_uid,
                  "partition_uid":declaration["partition_uid"],"family_uid":declaration["family_uid"]}
        rows.append(identity)
        try:
            inputs,target,mask,_=prepare_comparator_group(dataset[index]["native_joint_group"],belief,config,
                                                          next(belief.parameters()).device)
            uid=inputs.sample_uids[0]
            if uid!=declared_uid:
                raise ValueError("Loaded group differs from its declared dataset identity")
            identity.update(group_uid=uid,partition_uid=declaration["partition_uid"],
                            family_uid=inputs.grid["family_uid"],regimen_uids=list(inputs.regimen_uids))
            with torch.no_grad():
                mean,law=comparator_mean(model,method,inputs,belief.field_normalization,config,
                                         seed+config["native_scientific_training"]["validation_stream_offset"]+index)
            if method in config["native_controlled_comparators"]["conditional_mean_methods"] and residual is None:
                error=((mean-target)/belief.field_normalization.scale.to(mean)).square()
                identity.update(status="residual_unbound",metrics={"normalized_rmse":
                    float((torch.where(mask,error,0).sum()/mask.sum()).sqrt()),"crps":None})
                continue
            if method in config["native_controlled_comparators"]["conditional_mean_methods"]:
                samples=draw_native_comparator_residual(config,inputs,law[0],belief.field_normalization,residual)
            else:
                samples=law["particle_trajectory_samples"][:,0].transpose(0,1)
            metrics=weighted_native_scores(samples,inputs.weights[0],target[0],mask[0],
                                           belief.field_normalization,policy["nominal_levels"])
            identity.update(status="defined",metrics=metrics)
            if isinstance(law,dict) and law.get("execution"):
                identity["sampling_execution"]=law["execution"]
            if isinstance(law,dict) and law.get("action_conditioning"):
                identity["action_conditioning"]=law["action_conditioning"]
        except Exception as error:
            identity.update(status="failed",error=type(error).__name__+": "+str(error))
        finally:
            write_json(path,report)
    report["status"]="complete" if all(row["status"]=="defined" for row in rows) else "incomplete"
    report["defined_group_count"]=sum(row["status"]=="defined" for row in rows)
    report["failed_or_unavailable_group_count"]=len(rows)-report["defined_group_count"]
    write_json(path,report)
    return report



@torch.no_grad()
def _evaluate_native_solver(config,model,belief,binding,dataset,declarations,output,reference):
    """Forecast once per tuple; fix public-loss choices before loading scoring targets."""
    path=Path(output)/"native_comparator_evaluation.json"
    if path.exists():raise FileExistsError("Evaluation attempt already exists")
    if config["native_scientific_training"].get("source_scope","all")!="all":
        raise ValueError("Frozen RHS uses the full-mainline public-loss reference")
    loss_function=NativePublicCNSLoss(reference,config)
    registries=load_family_registries(config)
    rows=[]
    report={"schema_version":"native_comparator_evaluation.v1","method":"same_compiler_solver",
        "binding":binding,"reference_id":reference["reference_id"],"declared_group_count":len(declarations),
        "groups":rows,"predictive_law":model.prediction_kind,"conditional_draws_per_particle":1,
        "uncertainty_source":"original_complete_tuple_posterior_pushforward",
        "reported_uncertainty_calibrated":False,"numerical_accuracy_qualified":False,
        "truth_boundary":"initial_support_and_programs_then_fixed_selection_then_target_scoring",
        "scientific_twin":"NOT_RUN","formal_qualification":False,"status":"started"}
    write_json(path,report)
    for uid in dataset.group_uids:
        row={"group_uid":uid,"status":"started"};rows.append(row)
        try:
            declaration=declarations[uid]
            inputs=load_native_prediction_inputs(config,declaration,belief,registries)
            _,law=comparator_mean(model,"same_compiler_solver",inputs,belief.field_normalization,config,0)
            samples=law["particle_trajectory_samples"][:,0].transpose(0,1)
            weights=inputs.weights[0]
            source_mask=inputs.validity.reshape(*inputs.group_shape,*inputs.validity.shape[1:])[0].all(0)
            burdens=[]
            for action,regimen in enumerate(inputs.regimen_uids):
                values=[compiled_action_burden(records[action],inputs.grid["family_uid"],config)
                        for records in inputs.programs["records"][0]]
                if any(value!=values[0] for value in values[1:]):
                    raise ValueError("Intervention burden differs across complete posterior tuples")
                burdens.append(dict(values[0],regimen_uid=regimen))
            feasible=[i for i,item in enumerate(burdens) if item["status"]=="defined"]
            row.update(family_uid=inputs.grid["family_uid"],partition_uid=declaration["partition_uid"],
                source_uid=inputs.posterior.support.source_uid,regimen_uids=list(inputs.regimen_uids),
                execution=law["execution"],particle_count=len(weights),conditional_draws_per_particle=1,
                posterior_weights=weights.cpu().tolist(),burdens=burdens,
                declared_action_count=len(burdens),supported_action_count=len(feasible),
                withheld_action_count=len(burdens)-len(feasible))
            if not feasible:
                row.update(status="insufficient_evidence",reason="no_supported_public_loss_action")
                continue
            cost=samples.new_tensor([burdens[i]["normalized_burden"] for i in feasible])
            predicted=loss_function(samples[:,:,feasible],source_mask[feasible],inputs.grid,cost)
            decision=summarize_posterior_action_loss(predicted["loss"],weights)
            choice=decision["selected_index"]
            row["public_decision"]={"selected_regimen_uid":inputs.regimen_uids[feasible[choice]],
                "supported_regimen_uids":[inputs.regimen_uids[i] for i in feasible],
                "predictive_risks":decision["predictive_risks"].cpu().tolist(),
                "selected_optimal_mass":decision["selected_optimal_mass"],
                "tied_optimal_mass":decision["tied_optimal_mass"],"tie_break":decision["tie_break"],
                "predictive_clipped_fractions":(predicted["clipped_fraction"].double().mean(1)*
                    (weights.double()/weights.double().sum())[:,None,None]).sum(0).cpu().tolist()}
            # Prediction and choice are fixed before the dataset target can be opened.
            index=dataset.group_uids.index(uid)
            checked,target,mask,_=prepare_comparator_group(
                dataset[index]["native_joint_group"],belief,config,next(belief.parameters()).device)
            if (checked.sample_uids!=inputs.sample_uids or checked.regimen_uids!=inputs.regimen_uids
                    or checked.posterior_uid!=inputs.posterior_uid or not torch.equal(checked.weights,weights[None])
                    or not torch.equal(mask[0],source_mask)):
                raise ValueError("Scoring target differs from the fixed source-only forecast")
            observed=loss_function(target[0,feasible],mask[0,feasible],inputs.grid,cost)
            ranking=_ranking(decision["predictive_risks"].cpu().numpy(),observed["loss"].cpu().numpy())
            if ranking.pop("selected_supported_index")!=choice:
                raise ValueError("Held outcomes changed the fixed solver selection")
            ranking["supported_set_intervention_regret"]=ranking.pop("intervention_regret")
            row["public_decision"].update(metrics=ranking,observed_public_losses=observed["loss"].cpu().tolist(),
                observed_clipped_fractions=observed["clipped_fraction"].cpu().tolist(),
                maximum_supported_loss=predicted["maximum_supported_loss"],
                loss_intervals_by_nominal=loss_intervals(predicted["loss"],weights,observed["loss"],
                    config["native_controlled_comparators"]["nominal_levels"]))
            row.update(status="defined" if len(feasible)==len(burdens) else "defined_on_supported_actions",
                metrics=weighted_native_scores(samples,weights,target[0],mask[0],belief.field_normalization,
                    config["native_controlled_comparators"]["nominal_levels"],deterministic_conditional=True))
        except (ValueError,KeyError,RuntimeError,FloatingPointError,OSError) as error:
            row.update(status="failed",failure={"type":type(error).__name__,"message":str(error)})
        finally:write_json(path,report)
    seen=[row["group_uid"] for row in rows]
    report["defined_group_count"]=sum(row["status"] in {"defined","defined_on_supported_actions"} for row in rows)
    report["fully_supported_group_count"]=sum(row["status"]=="defined" for row in rows)
    report["failed_or_unavailable_group_count"]=len(declarations)-report["defined_group_count"]
    report["status"]=("complete" if len(seen)==len(set(seen)) and set(seen)==set(declarations)
                      and report["fully_supported_group_count"]==len(declarations) else "incomplete")
    write_json(path,report)
    return report

def paired_seed_group_summary(left: list[dict], right: list[dict], declared_group_uids: list[str],
                              metric: str, policy: dict) -> dict:
    """Average the matched training seeds inside each group before resampling groups."""
    seed_policy = policy.get("seed_policy", "manuscript_native_five")
    seeds = native_training_seed_values(seed_policy, policy["training_seeds"])
    bootstrap_seed = 41 if seed_policy == "manuscript_native_three" else 3407
    if (policy["bootstrap_replicates"]!=2000
            or policy["bootstrap_seed"]!=bootstrap_seed or policy["interval"]!=[.025,.975]
            or policy["order"]!="average_training_seeds_then_paired_independent_groups"):
        raise ValueError("Native manuscript statistics require the declared native seed/group contract")
    if not declared_group_uids or len(set(declared_group_uids))!=len(declared_group_uids):
        raise ValueError("Independent groups must be declared once before metrics")
    declared=set(declared_group_uids)
    def index(reports):
        if len({report["method"] for report in reports})!=1:
            raise ValueError("Each seed bundle must contain one declared comparison method")
        rows={}
        for report in reports:
            if report["binding"]["model_identity"]["method"]!=report["method"]:
                raise ValueError("Native comparison method/checkpoint identity mismatch")
            seed_binding = report["binding"]["model_identity"]["seed_binding"]
            seed = seed_binding["model_seed"]
            if type(seed) is not int or seed not in seeds or seed in rows:
                raise ValueError("Duplicate or undeclared native training seed")
            if (seed_binding.get("seed_policy") != policy.get("seed_policy", "manuscript_native_five")
                    or seed_binding.get("allowed_training_seeds") != seeds
                    or any(type(seed_binding.get(key)) is not int or seed_binding[key] != seed
                           for key in ("selected_formal_seed", "base_training_seed"))):
                raise ValueError("Comparison training seed binding differs from its declared native policy")
            observed={}
            for row in report["groups"]:
                uid=row.get("group_uid")
                if uid not in declared or uid in observed:
                    raise ValueError("Duplicate or undeclared native evaluation group")
                observed[uid]=row
            if set(observed)!=declared or report["declared_group_count"]!=len(declared):
                raise ValueError("Native evaluation changed the declared group denominator")
            rows[seed]=observed
        if set(rows)!=set(seeds):
            raise ValueError("Seed averaging requires every declared matched training seed")
        return rows
    lhs,rhs=index(left),index(right)
    left_binding={row["binding"]["model_identity"]["seed_binding"]["model_seed"]:row["binding"] for row in left}
    right_binding={row["binding"]["model_identity"]["seed_binding"]["model_seed"]:row["binding"] for row in right}
    for seed in seeds:
        a,b=left_binding[seed],right_binding[seed]
        if (not a.get("normalization_id") or a["normalization_id"]!=b.get("normalization_id")
                or a["model_identity"].get("conditioning")!=b["model_identity"].get("conditioning")):
            raise ValueError("Paired comparators need the same fitted conditioning and normalization per seed")
    differences=[];groups=[];unavailable=[]
    for uid in declared_group_uids:
        values=[]
        for seed in seeds:
            a,b=lhs[seed][uid],rhs[seed][uid]
            x,y=a.get("metrics",{}).get(metric),b.get("metrics",{}).get(metric)
            if (a["status"]!="defined" or b["status"]!="defined"
                    or not isinstance(x,(int,float)) or not isinstance(y,(int,float))
                    or not math.isfinite(x) or not math.isfinite(y)):
                unavailable.append({"group_uid":uid,"seed":seed})
            else:
                values.append(float(x)-float(y))
        if len(values)==len(seeds):
            differences.append(float(np.mean(values)))
            groups.append({"group_uid":uid,"seed_averaged_paired_difference":differences[-1]})
    result={"schema_version":"native_paired_seed_group_statistics.v1","metric":metric,
        "policy":policy,"declared_group_count":len(declared_group_uids),
        "defined_group_count":len(groups),"unavailable_seed_group_pairs":unavailable,"groups":groups,
        "status":"incomplete","mean_difference":None,"paired_group_interval":None,
        "scientific_twin":"NOT_RUN","formal_qualification":False}
    if unavailable:
        return result
    if len(groups)<2:
        raise ValueError("Paired group uncertainty requires multiple independent groups")
    values=np.asarray(differences,dtype=np.float64)
    rng=np.random.default_rng(policy["bootstrap_seed"])
    draws=np.empty(policy["bootstrap_replicates"])
    for index in range(len(draws)):
        draws[index]=values[rng.integers(0,len(values),size=len(values))].mean()
    result.update(status="complete",mean_difference=float(values.mean()),
        paired_group_interval=np.quantile(draws,policy["interval"]).tolist())
    return result


def summarize_native_comparisons(config:dict,left_paths:list[Path],right_paths:list[Path],
                                 metric:str,output:Path,role:str="validation") -> dict:
    """Read all declared groups, then publish every predeclared partition contrast."""
    if role not in {"development","validation"}:
        raise ValueError("Comparisons cannot use training or calibration groups")
    left=[json.loads(path.read_text()) for path in left_paths]
    right=[json.loads(path.read_text()) for path in right_paths]
    declared=[row for row in group_definitions(config) if row["role"]==role]
    uids={row["sample_uid"] for row in declared}
    if not uids or len(uids)!=len(declared):
        raise ValueError("Native comparison group declarations are invalid")
    for report in [*left,*right]:
        rows=report["groups"]
        if (report["declared_group_count"]!=len(declared) or len(rows)!=len(declared)
                or {row.get("group_uid") for row in rows}!=uids):
            raise ValueError("Comparison input changed the full predeclared denominator")
    policy=config["manuscript_native_statistics"]
    result={"schema_version":"native_comparator_pairwise_summary.v1","role":role,"metric":metric,
        "declared_group_count":len(declared),"input_reports":{"left":[str(p) for p in left_paths],
        "right":[str(p) for p in right_paths]},"partitions":[],
        "scientific_twin":"NOT_RUN","formal_qualification":False}
    for partition in sorted({row["partition_uid"] for row in declared}):
        selected=[row["sample_uid"] for row in declared if row["partition_uid"]==partition]
        keep=set(selected)
        def subset(reports):
            return [{**report,"declared_group_count":len(selected),
                     "groups":[row for row in report["groups"] if row["group_uid"] in keep]}
                    for report in reports]
        summary=paired_seed_group_summary(subset(left),subset(right),selected,metric,policy)
        result["partitions"].append({"partition_uid":partition,**summary})
    result["status"]="complete" if all(row["status"]=="complete" for row in result["partitions"]) else "incomplete"
    path=output/"native_comparator_pairwise_summary.json"
    if path.exists():
        raise FileExistsError("Native comparison output already exists")
    write_json(path,result)
    return result
