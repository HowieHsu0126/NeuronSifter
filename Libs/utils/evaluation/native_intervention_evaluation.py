"""Native predictive intervention comparisons under a shared public CNS loss."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from Libs.utils.data.raw_data_registry import write_json
from Libs.utils.evaluation.native_intervention_decisions import (
    compiled_action_burden, fit_public_loss_reference, posterior_control, NativePublicCNSLoss,
)
from Libs.utils.exp.experiment_manifests import resolve_training_seed
from Libs.utils.exp.native_comparator_training import prepare_comparator_group


def training_record_identities(dataset) -> list[dict]:
    """Use admitted teacher metadata only; no trajectory array or source replay."""
    keys=("sample_uid","trajectory_uid","family_uid","qc_grade","content_hash")
    return [{key:child.native_records[position]["teacher"][key] for key in keys}
            for child,positions,_ in dataset.groups for position in positions]


def fit_dataset_public_reference(config, dataset, declarations, output: Path) -> dict:
    if config["native_scientific_training"].get("source_scope", "all") != "all":
        raise ValueError("This public reference is scoped to the full-mainline comparison, not isolated source arms")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Public CNS reference already exists")
    def records():
        for index in range(len(dataset)):
            group = dataset[index]["native_joint_group"]
            child, positions, _ = dataset.groups[index]
            teachers = [child.native_records[position]["teacher"] for position in positions]
            source = group["joint_source"]["support"]
            if source["role"] != "train":
                raise ValueError("Public reference may only consume training source targets")
            for item, teacher in zip(group["items"], teachers):
                uid = teacher["sample_uid"]
                if (uid not in declarations or declarations[uid]["role"] != "train"
                        or item["trajectory_uid"] != teacher["trajectory_uid"]
                        or teacher["split_name"] != "train"):
                    raise ValueError("Public reference teacher/source declaration mismatch")
                yield {**{key:teacher[key] for key in ("sample_uid","trajectory_uid","family_uid","qc_grade","content_hash")},
                    "role":"train", "grid":child.native_registry["grid"],
                    "trajectory":item["trajectory"], "quantity_support":item["quantity_support"]}
    artifact = fit_public_loss_reference(config, records(), set(declarations))
    write_json(output, artifact)
    return artifact


def _ranking(predicted: np.ndarray, observed: np.ndarray) -> dict:
    if predicted.shape != observed.shape or len(predicted) < 1:
        raise ValueError("Predicted and observed intervention risks need one shared supported action set")
    selected = int(predicted.argmin())
    pairs = [(.5 if predicted[i] == predicted[j] or observed[i] == observed[j] else
              float((predicted[i]-predicted[j])*(observed[i]-observed[j]) > 0))
             for i in range(len(predicted)) for j in range(i)]
    left, right = rankdata(predicted), rankdata(observed)
    correlation = (float(np.corrcoef(left,right)[0,1]) if left.std() > 0 and right.std() > 0 else None)
    return {"selected_supported_index":selected, "intervention_regret":float(observed[selected]-observed.min()),
            "selected_observed_loss":float(observed[selected]), "spearman":correlation,
            "ordering_accuracy":sum(pairs)/len(pairs) if pairs else None,
            "tie_break":"declared_action_order"}


def loss_intervals(draws, weights, actual, nominal_levels):
    """Weighted empirical intervals with no inherited calibration claim."""
    j,d,actions=draws.shape
    mass=weights.double().repeat_interleave(d)/d
    ordered,order=draws.double().reshape(j*d,actions).sort(0)
    cdf=mass[:,None].expand_as(ordered).gather(0,order).cumsum(0)
    result={}
    for level in nominal_levels:
        if not 0<level<1:raise ValueError("Predictive loss coverage needs a proper nominal level")
        bounds=[]
        for q in ((1-level)/2,(1+level)/2):
            position=(cdf<q).sum(0).clamp_max(j*d-1)
            bounds.append(ordered.gather(0,position[None]).squeeze(0))
        covered=(actual>=bounds[0]) & (actual<=bounds[1])
        result[str(level)]={"lower":bounds[0].cpu().tolist(),"upper":bounds[1].cpu().tolist(),
            "covered":covered.cpu().tolist(),"covered_action_count":int(covered.sum()),
            "supported_action_count":actions,"coverage":float(covered.double().mean()),
            "independent_calibration_qualified":False}
    return result


@torch.no_grad()
def evaluate_native_interventions(config, method, model, belief, binding, dataset, declarations,
                                  reference: dict, output: Path) -> dict:
    """Sample each group once, then evaluate all posterior-weight controls."""
    methods=config["native_intervention_evaluation"]["posterior_methods"]
    if method not in methods:raise ValueError("Unknown native posterior intervention comparison")
    if config["native_scientific_training"].get("source_scope","all")!="all":
        raise ValueError("Full-mainline public-risk reference cannot be used by isolated source comparisons")
    paths={name:Path(output)/name/"native_intervention_evaluation.json" for name in methods}
    if any(path.exists() for path in paths.values()):raise FileExistsError("Native posterior comparison already exists")
    loss_function=NativePublicCNSLoss(reference,config)
    seed=int(resolve_training_seed(config,require_selected=True)["model_seed"])
    reports={name:{"schema_version":"native_intervention_evaluation.v1","method":name,"binding":binding,
        "reference_id":reference["reference_id"],"declared_group_count":len(declarations),"groups":[],
        "upstream_calibration":"shared_selected_belief","post_transform_calibration":"unevaluated",
        "point_uncertainty":"state_and_parameter_mass_collapsed_conditional_diffusion_retained" if name=="posterior_point" else None,
        "dependence_scope":"five_declared_factor_marginals_on_unchanged_valid_tuples" if name=="posterior_dependence_ablated" else None,
        "truth_boundary":"public_functional_of_held_trajectory_after_fixed_predicted_selection_no_hidden_optimum",
        "sampling":"one_shared_conditional_draw_set_per_group_for_all_controls",
        "scientific_twin":"NOT_RUN","formal_qualification":False,"status":"started"} for name in methods}
    sample_calls=0
    for index in range(len(dataset)):
        uid=dataset.group_uids[index]
        rows={name:{"group_uid":uid,"status":"started"} for name in methods}
        for name in methods:reports[name]["groups"].append(rows[name])
        try:
            if uid not in declarations or declarations[uid]["role"] not in {"development","validation"}:
                raise ValueError("Native ranking needs declared development or untouched validation groups")
            group=dataset[index]["native_joint_group"]
            inputs,target,mask,_=prepare_comparator_group(group,belief,config,next(belief.parameters()).device)
            budget=config["native_scientific_training"]["predictive_budget"]
            law=model.sample_native(inputs,belief.field_normalization,budget["conditional_draws_per_particle"],
                budget["categorical_reverse_steps"],seed+config["native_scientific_training"]["validation_stream_offset"]+index)
            sample_calls+=1
            samples=law["particle_trajectory_samples"][:,0].transpose(0,1)
            burdens=[]
            for ai,regimen in enumerate(inputs.regimen_uids):
                candidates=[compiled_action_burden(records[ai],inputs.grid["family_uid"],config)
                            for records in inputs.programs["records"][0]]
                if any(candidate!=candidates[0] for candidate in candidates[1:]):
                    raise ValueError("Intervention burden changes across physical posterior candidates")
                burdens.append(dict(candidates[0],regimen_uid=regimen))
            feasible=[i for i,item in enumerate(burdens) if item["status"]=="defined"]
            for row in rows.values():
                row.update(family_uid=inputs.grid["family_uid"],source_uid=inputs.posterior.support.source_uid,
                    burdens=burdens,declared_action_count=len(burdens),supported_action_count=len(feasible),
                    withheld_action_count=len(burdens)-len(feasible),
                    conditional_draws_per_particle=samples.shape[1],particle_count=samples.shape[0])
            if not feasible:
                for row in rows.values():row.update(status="insufficient_evidence",reason="no_action_within_public_burden_support")
                continue
            cost=samples.new_tensor([burdens[i]["normalized_burden"] for i in feasible])
            predicted=loss_function(samples[:,:,feasible],mask[0,feasible],inputs.grid,cost)
            controls,risks,choices={},{},{}
            # Fix every predicted choice before computing any held-target loss.
            for name in methods:
                control=posterior_control(inputs.posterior,name);controls[name]=control
                risk=(predicted["loss"].mean(1)*control.weights[0,:,None]).sum(0)
                risks[name]=risk;choices[name]=int(risk.argmin())
                rows[name].update(posterior_weights=control.weights[0].cpu().tolist(),
                    selected_regimen_uid=inputs.regimen_uids[feasible[choices[name]]],
                    supported_regimen_uids=[inputs.regimen_uids[i] for i in feasible],
                    predictive_risks=risk.cpu().tolist(),
                    predictive_clipped_fractions=(predicted["clipped_fraction"].mean(1)*
                        control.weights[0,:,None,None]).sum(0).cpu().tolist())
            observed=loss_function(target[0,feasible],mask[0,feasible],inputs.grid,cost)
            for name in methods:
                rankings=_ranking(risks[name].cpu().numpy(),observed["loss"].cpu().numpy())
                if rankings.pop("selected_supported_index")!=choices[name]:
                    raise ValueError("Target evaluation attempted to change the fixed predicted choice")
                rankings["supported_set_intervention_regret"]=rankings.pop("intervention_regret")
                rows[name].update(status="defined" if len(feasible)==len(burdens) else "defined_on_supported_actions",
                    observed_public_losses=observed["loss"].cpu().tolist(),
                    observed_clipped_fractions=observed["clipped_fraction"].cpu().tolist(),
                    loss_intervals_by_nominal=loss_intervals(predicted["loss"],controls[name].weights[0],
                        observed["loss"],config["native_controlled_comparators"]["nominal_levels"]),
                    metrics=rankings,maximum_supported_loss=predicted["maximum_supported_loss"])
        except (ValueError,KeyError,RuntimeError,FloatingPointError) as error:
            for row in rows.values():row.update(status="failed",failure={"type":type(error).__name__,"message":str(error)})
        finally:
            for name in methods:write_json(paths[name],reports[name])
    for name,report in reports.items():
        seen=[row["group_uid"] for row in report["groups"]]
        if len(seen)!=len(set(seen)) or set(seen)!=set(declarations):
            report["denominator_failure"]="missing_duplicate_or_undeclared_groups"
        report["defined_group_count"]=sum(row["status"] in {"defined","defined_on_supported_actions"} for row in report["groups"])
        report["fully_supported_group_count"]=sum(row["status"]=="defined" for row in report["groups"])
        report["shared_sampling_calls"]=sample_calls
        report["status"]=("complete" if report["fully_supported_group_count"]==len(declarations)
                          and not report.get("denominator_failure") else "incomplete")
        write_json(paths[name],report)
    return {"status":"complete" if all(report["status"]=="complete" for report in reports.values()) else "incomplete",
            "shared_sampling_calls":sample_calls,"reports":reports,"report_paths":{k:str(v) for k,v in paths.items()}}
