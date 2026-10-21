"""Checkpoint-bound CLI software check using previously completed retained inputs."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch

from Libs.utils.model.native_training_selection import training_checkpoint_path
import yaml

from Libs.exp.baselines import run_external_baseline_bridge as cli
from Libs.exp.training.pretrain_microenvironment_world_model import joint_inference_modules
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.native_field_normalization import NativeFieldNormalization
from Libs.utils.data.joint_physical_parameters import bind_candidate_parameters,parameter_schema_from_config
from Libs.utils.data.native_diffusion_data import parse_joint_source
from Libs.utils.data.native_observable_profile import family_grid,quantity_support
from Libs.utils.exp.experiment_manifests import resolve_training_seed
from Libs.utils.exp.native_comparator_training import load_frozen_native_belief
from Libs.utils.evaluation import native_comparator_evaluation as evaluation


def retained_group_metadata(origin: Path, selected: list) -> tuple:
    """Expose actual public split/action declarations before reading a target."""
    roles = {row["role"] for _, row in selected}
    if len(roles) != 1:
        raise ValueError("Retained metadata requires one nonempty declared role")
    groups = []
    for index, declaration in selected:
        raw = json.loads((origin/f"group_{index}_conditioning.json").read_text())
        actions = raw["particle_programs"]["regimen_uids"]
        if len(actions) < 2 or len(set(actions)) != len(actions):
            raise ValueError("Retained metadata requires its complete unique paired actions")
        teachers = [{"sample_uid": declaration["sample_uid"], "family_uid": declaration["family_uid"],
            "split_name": declaration["role"], "source_holdout_role": declaration["role"],
            "calibration_reservation": declaration["calibration_reservation"], "regimen_uid": uid} for uid in actions]
        groups.append((SimpleNamespace(native_records=[{"teacher": row} for row in teachers]), list(range(len(teachers))), None))
    return next(iter(roles)), groups


def run(origin:Path,output:Path,*,method="codano",full_conditioner=False,calibrate=True):
    output.mkdir(parents=True,exist_ok=True)
    config=load_config(str(origin/"native_config.yaml"))
    current=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    config["native_controlled_comparators"]=copy.deepcopy(current["native_controlled_comparators"])
    config["native_state_comparison"]=copy.deepcopy(current["native_state_comparison"])
    config["native_rssm_prediction"]=copy.deepcopy(current["native_rssm_prediction"])
    config["manuscript_native_protocol"]=copy.deepcopy(current["manuscript_native_protocol"])
    rssm=config["native_controlled_comparators"]["references"]
    rssm["planet"]["hyperparameters"].update(deterministic_size=12,stochastic_size=4,hidden_size=12)
    rssm["dreamerv3"]["hyperparameters"].update(deterministic_size=16,hidden_size=8,
        stochastic_size=3,classes=4,blocks=2,vector_units=8)
    config["native_state_comparison"]["pca_components"]=2
    training=config["screening_pipeline"]["screening_benchmark"]["training"]
    training.update(hidden_dim=12,dropout_rate=0.,random_seed=40,seed_policy="manuscript_native_five")
    training["batch_size"]=1
    training["memory_policy"].update(effective_batch_size=1,micro_batch_size=1,gradient_accumulation_steps=1)
    config["selected_formal_seed"]=40
    for key in ("formal_seed_values","seeds"):config["paper_grade_full_suite"][key]=[40,41,42,43,44]
    config["paper_grade_baseline_registry"]["seed_values"]=[40,41,42,43,44]
    config["schemas"]["native_training_seed_binding"]=current["schemas"]["native_training_seed_binding"]
    posterior=config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]
    posterior.update(physical_parameter_conditioning="source_declared_named_candidates",
                     posterior_hidden_dim=12,posterior_dropout_rate=0.)
    config["native_scientific_training"]["selection"].update(
        max_epochs=3,minimum_epochs=2,minimum_steps=2,patience=1,min_delta=1e12,max_steps=100,max_seconds=300.)
    config["native_scientific_training"]["learning_rate_schedule"].update(warmup_steps=1)
    h=config["native_controlled_comparators"]["references"]["codano"]["hyperparameters"]
    h.update(n_layers=1,lifting_channels=4,hidden_variable_codimension=2,projection_channels=4,
             conditioning_channels=2,positional_encoding_dim=2,positional_encoding_modes=[2,2],n_modes=[2,2],n_heads=1)
    if full_conditioner:
        from Libs.exp.training.pretrain_microenvironment_world_model import MicroenvironmentPretrainingModel
        backbone=config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
        backbone.update(operator_hidden_dim=12,operator_depth=1,attention_head_count=3,mechanism_state_token_count=6)
        config["native_diffusion_data"]["posterior_context_width"]=12
        refs=config["native_controlled_comparators"]["references"]
        refs["gru"]["hyperparameters"]["hidden_dim"]=8
        refs["flattened_parameter_matched"]["hyperparameters"].update(hidden_dim=4,depth=1,head_count=1)
        refs["fno"]["hyperparameters"].update(hidden_channels=4,n_layers=1,n_modes=[2,2],conditioning_channels=2)
    config_path=output/"fixture_config.yaml"
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    binding=resolve_training_seed(config,require_selected=True)
    artifact_path=origin/"train_normalization.json";data=artifact_path.read_bytes()
    normalizer=NativeFieldNormalization({"kind":"training_scaler","artifact_ref":str(artifact_path),
        "sha256":hashlib.sha256(data).hexdigest(),"artifact":json.loads(data)},26)
    torch.manual_seed(40)
    modules,_=joint_inference_modules(config,12,26)
    state={}
    for name,module in zip(("observation_encoder","retained_mask_projection","world_state_posterior"),modules):
        state.update({name+"."+key:value for key,value in module.state_dict().items()})
    state.update({"field_normalization."+key:value for key,value in normalizer.state_dict().items()})
    parameter_metadata={}
    if full_conditioner:
        common=MicroenvironmentPretrainingModel(config,False)
        assert sum(p.numel() for p in common.parameters())<1000000
        state=common.state_dict()
        parameter_metadata={"actual_total_parameter_count":sum(p.numel() for p in common.parameters())}
    common_path=output/"diagnostic_belief.pt"
    torch.save({"model_state_dict":state,"training_seed_binding":binding},common_path)
    common_manifest=output/"diagnostic_belief_selection.json"
    common_manifest.write_text(json.dumps({"training_seed_binding":binding,**parameter_metadata,
        "selection":{"selection_role":"development","best":{"checkpoint_path":str(common_path)},
                     "terminal":"SOFTWARE_FIXTURE_NOT_SCIENTIFIC_TRAINING"}}))
    receipt=json.loads((origin/"source_receipt.json").read_text())
    schema=parameter_schema_from_config(config)
    cache={}

    class RetainedGroups:
        def __init__(self,selected):
            self.selected=selected
            self.group_uids=[row["sample_uid"] for _,row in selected]
            self.split_name,self.groups=retained_group_metadata(origin,selected)
            self.family_group_counts=dict(Counter(row["family_uid"] for _,row in selected))
        def __len__(self):return len(self.selected)
        def __getitem__(self,index):
            gi,group=self.selected[index]
            if gi not in cache:
                raw=json.loads((origin/f"group_{gi}_conditioning.json").read_text())
                producer=json.loads((origin/f"group_{gi}_producer.json").read_text())
                bound=bind_candidate_parameters(raw,group,producer,schema,
                    record_ref=str(origin/f"group_{gi}_producer.json"),
                    producer_commit=receipt["source_identity"]["producer_commit"])
                support,_=parse_joint_source(bound)
                grid=family_grid(support.quantity_profile,support.source_family_uid)
                mask=torch.from_numpy(quantity_support(grid))
                cases=[case for case in receipt["cases"] if case["group_index"]==gi]
                items=[]
                for ai,case in enumerate(cases):
                    with np.load(origin/case["raw_reference"]) as payload:
                        target=torch.from_numpy(payload["trajectory"]).float()
                    items.append({"trajectory":target,"quantity_support":mask,
                        "uncertainty":torch.tensor(case["numerical"]["uncertainty"]),
                        "teacher_confidence":torch.tensor(case["numerical"]["teacher_confidence"]),
                        "teacher_loss_weight":torch.tensor(.5),
                        "normalization_id":normalizer.identity,
                        "regimen_uid":bound["particle_programs"]["regimen_uids"][ai]})
                cache[gi]={"native_joint_group":{"items":items,"joint_source":bound}}
            return cache[gi]

    original=evaluation.declared_dataset
    def declared(_config,role,*,reservation=None):
        selected=[(i,row) for i,row in enumerate(receipt["groups"]) if row["role"]==role
                  and (reservation is None or row["calibration_reservation"]==reservation)]
        return RetainedGroups(selected),{row["sample_uid"]:row for _,row in selected}
    evaluation.declared_dataset=declared
    original_argv=sys.argv
    try:
        base=["run_external_baseline_bridge.py","--protocol-scope","manuscript-native",
              "--config",str(config_path),"--baseline-name",method,"--device","cpu",
              "--conditioning-config",str(config_path),"--conditioning-manifest",str(common_manifest)]
        sys.argv=[*base,"--command","prepare_inputs","--output-dir",str(output/"prepared")]
        assert cli.main()==0
        train_root=output/"training"
        sys.argv=[*base,"--command","train_baseline","--output-dir",str(train_root)]
        code=cli.main()
        assert code==0
        report=json.loads((train_root/"native_comparator_training_manifest.json").read_text())
        best,last=report["selection"]["best"],report["selection"]["last"]
        assert best["checkpoint_path"]!=last["checkpoint_path"]
        assert last["steps"]==8 and report["executed_steps"]==8
        best_state=torch.load(training_checkpoint_path(train_root/"native_comparator_training_manifest.json",report["selection"],best["checkpoint_path"]),map_location="cpu",weights_only=False)
        last_state=torch.load(training_checkpoint_path(train_root/"native_comparator_training_manifest.json",report["selection"],last["checkpoint_path"]),map_location="cpu",weights_only=False)
        assert any(not torch.equal(value,last_state["model_state_dict"][key])
                   for key,value in best_state["model_state_dict"].items())
        eval_root=output/"evaluation"
        sys.argv=[*base,"--command","evaluate_baseline","--output-dir",str(eval_root),
                  "--training-manifest",str(train_root/"native_comparator_training_manifest.json")]
        if method in config["native_state_comparison"]["fitted_methods"]:
            evaluation_code=cli.main()
            observed=json.loads((eval_root/"native_state_evaluation.json").read_text())
            if evaluation_code!=0:print("STATE_EVALUATION_FAILURE",json.dumps(observed),flush=True)
            assert evaluation_code==0
            assert observed["status"]=="complete" and observed["defined_group_count"]==4
            assert all(row["metrics"]["factor_mse"] for row in observed["groups"])
            held_dataset,_=declared(config,"validation")
            loaded,_,_=evaluation.load_native_comparator(config,method,config_path,common_manifest,
                train_root/"native_comparator_training_manifest.json",held_dataset,torch.device("cpu"))
            probe_source=held_dataset[0]["native_joint_group"]["joint_source"]
            removed=loaded.blocks[0]
            visible=set(range(loaded.width))-set(removed)
            changed=copy.deepcopy(probe_source)
            for observation in changed["observations"]:
                for column,slot in enumerate(observation["encoder_indices"]):
                    if slot in removed:
                        for values in observation["values"]:values[column]=1e6
            original_read=Path.read_text
            def forbid_oracle_read(path,*args,**kwargs):
                if str(path).endswith("_producer.json"):
                    raise AssertionError("State fitting/inference attempted oracle file access")
                return original_read(path,*args,**kwargs)
            Path.read_text=forbid_oracle_read
            try:
                loaded.eval()
                _,_,before=loaded.infer(probe_source,visible_indices=visible)
                _,_,after=loaded.infer(changed,visible_indices=visible)
                assert torch.equal(before,after)
                assert torch.isfinite(loaded.loss(probe_source))
            finally:
                Path.read_text=original_read
            if method=="state_pca":
                from Libs.models.modules.native_state_probe import NativeStateProbe
                common,_=load_frozen_native_belief(config,config_path,common_manifest,torch.device("cpu"))
                unfitted=NativeStateProbe(method,common,config)
                try:unfitted.fit_pca(held_dataset)
                except ValueError as error:assert "training observations" in str(error)
                else:raise AssertionError("PCA accepted validation observations for fitting")
            return {"method":method,"status":"state_probe_training_and_evaluation_passed",
                "held_value_invariance":True,"oracle_reads_blocked_during_fit_and_inference":True,
                "executed_optimizer_updates":report["executed_steps"],"best_epoch":best["epoch"],"last_epoch":last["epoch"],
                "evaluation":observed,"fit":report["state_probe_fit"],
                "common_conditioner":"tiny_untrained_inference_fixture",
                "source_admission":"retained_group_in_memory_adapter","source_operations":0,
                "scientific_twin":"NOT_RUN","formal_qualification":False}
        if method in config["native_rssm_prediction"]["fitted_methods"]:
            code=cli.main()
            observed=json.loads((eval_root/"native_comparator_evaluation.json").read_text())
            if code:print("RSSM_EVALUATION_FAILURE",json.dumps(observed),flush=True)
            assert code==0 and observed["defined_group_count"]==4
            assert observed["reported_uncertainty_calibrated"] is False
            assert observed["behavior_learning_implemented"] is False
            assert all(row["status"]=="defined" and np.isfinite(row["metrics"]["crps"]) for row in observed["groups"])
            held,_=declared(config,"validation")
            loaded,belief,_=evaluation.load_native_comparator(config,method,config_path,common_manifest,
                train_root/"native_comparator_training_manifest.json",held,torch.device("cpu"))
            from Libs.tests.native_rssm_integration import verify_forecast_contract
            checks=verify_forecast_contract(loaded,belief,held[0]["native_joint_group"],config)
            wrong="dreamerv3" if method=="planet" else "planet"
            try:evaluation.load_native_comparator(config,wrong,config_path,common_manifest,
                train_root/"native_comparator_training_manifest.json",held,torch.device("cpu"))
            except ValueError as error:assert "identity" in str(error)
            else:raise AssertionError("A different RSSM method checkpoint was accepted")
            if method=="planet":
                alias=[*base];alias[alias.index("--baseline-name")+1]="dreamer"
                sys.argv=[*alias,"--command","train_baseline","--output-dir",str(output/"forbidden_refit")]
                try:cli.main()
                except ValueError as error:assert "must not refit" in str(error)
                else:raise AssertionError("Matched Dreamer predictor refitted the same Gaussian core")
                sys.argv=[*alias,"--command","evaluate_baseline","--output-dir",str(output/"dreamer_evaluation"),
                    "--training-manifest",str(train_root/"native_comparator_training_manifest.json")]
                assert cli.main()==0
                alias_result=json.loads((output/"dreamer_evaluation"/"native_comparator_evaluation.json").read_text())
                assert alias_result["binding"]==observed["binding"]
                assert alias_result["method"]=="planet"
                assert [row["metrics"] for row in alias_result["groups"]]==[row["metrics"] for row in observed["groups"]]
            return {"method":method,"status":"rssm_training_reload_and_prediction_passed",
                "training_steps":report["executed_steps"],"best_epoch":best["epoch"],"last_epoch":last["epoch"],
                "predictor_parameter_count":report["actual_predictor_parameter_count"],
                "prediction_law":observed["predictive_law"],"evaluation_groups":4,"evaluation":observed,"training":report,**checks,
                "source_operations":0,"scientific_twin":"NOT_RUN","formal_qualification":False}
        assert cli.main()==2
        observed=json.loads((eval_root/"native_comparator_evaluation.json").read_text())
        assert observed["status"]=="incomplete"
        assert len(observed["groups"])==4
        assert all(row["status"]=="residual_unbound" and row["metrics"]["crps"] is None
                   for row in observed["groups"])
        if not calibrate:
            return {"method":method,"status":"mean_training_and_reload_passed_predictive_law_unbound",
                "executed_optimizer_updates":report["executed_steps"],"best_epoch":best["epoch"],"last_epoch":last["epoch"],
                "parameter_match":report.get("parameter_match"),"evaluation":observed,
                "common_conditioner":"tiny_untrained_full_model_fixture",
                "source_admission":"retained_group_in_memory_adapter","source_operations":0,
                "scientific_twin":"NOT_RUN","formal_qualification":False}
        calibration_root=output/"calibration"
        sys.argv=[*base,"--command","calibrate_baseline","--output-dir",str(calibration_root),
                  "--training-manifest",str(train_root/"native_comparator_training_manifest.json")]
        assert cli.main()==0
        residual_path=calibration_root/"native_comparator_residual.json"
        calibrated=json.loads(residual_path.read_text())
        assert calibrated["status"]=="FIT_COMPLETE" and calibrated["calibration_check_complete"]
        assert len(calibrated["scales"])==26 and min(calibrated["scales"])>0
        assert len(calibrated["fit_group_uids"])==4 and len(calibrated["check_group_uids"])==4
        assert not set(calibrated["fit_group_uids"]) & set(calibrated["check_group_uids"])
        assert all(row["scientific_check_pass"] is None for row in calibrated["checks"])
        predictive_root=output/"predictive_evaluation"
        sys.argv=[*base,"--command","evaluate_baseline","--output-dir",str(predictive_root),
                  "--training-manifest",str(train_root/"native_comparator_training_manifest.json"),
                  "--residual-artifact",str(residual_path)]
        assert cli.main()==0
        predictive=json.loads((predictive_root/"native_comparator_evaluation.json").read_text())
        assert predictive["status"]=="complete" and predictive["defined_group_count"]==4
        assert all(row["status"]=="defined" and np.isfinite(row["metrics"]["crps"])
                   for row in predictive["groups"])
        mismatch=copy.deepcopy(config)
        mismatch["selected_formal_seed"]=41
        mismatch["screening_pipeline"]["screening_benchmark"]["training"]["random_seed"]=41
        try:load_frozen_native_belief(mismatch,config_path,common_manifest,torch.device("cpu"))
        except ValueError as error:assert "seed" in str(error)
        else:raise AssertionError("Different fitted seed was accepted")
        result={"status":"passed","scope":"real_cli_with_explicit_retained_group_fixture_adapter",
            "training_steps":report["executed_steps"],"development_best_epoch":best["epoch"],
            "last_epoch":last["epoch"],"distinct_immutable_checkpoints":True,
            "evaluation_groups":len(observed["groups"]),"evaluation_status":"residual_unbound",
            "prepared_training_groups":4,"prepared_development_groups":4,
            "calibration_fit_groups":len(calibrated["fit_group_uids"]),
            "calibration_check_groups":len(calibrated["check_group_uids"]),
            "predictive_defined_groups":predictive["defined_group_count"],
            "scientific_calibration_qualification":False,
            "source_operations":0,"scientific_twin":"NOT_RUN","formal_qualification":False}
        print("NATIVE_COMPARATOR_CLI",json.dumps(result),flush=True)
        return result
    finally:
        evaluation.declared_dataset=original
        sys.argv=original_argv


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained-fixture",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.retained_fixture,args.output.parent/"cli_fixture")
    args.output.write_text(json.dumps(result,indent=2)+"\n")
