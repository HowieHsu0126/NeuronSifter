"""Actual tiny selected native teacher and Student checks on retained software inputs."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

from Libs.utils.model.native_training_selection import training_checkpoint_path
from torch.utils.data import DataLoader
import yaml

from Libs.exp.baselines import run_external_baseline_bridge as cli
from Libs.exp.training import pretrain_microenvironment_world_model as training
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.joint_physical_parameters import bind_candidate_parameters, parameter_schema_from_config
from Libs.utils.data.native_diffusion_data import parse_joint_source
from Libs.utils.data.native_observable_profile import family_grid, quantity_support
from Libs.utils.evaluation import native_comparator_evaluation as evaluation
from Libs.utils.exp.experiment_manifests import resolve_training_seed
from Libs.utils.exp.native_comparator_training import bind_frozen_training_belief
from Libs.utils.exp.native_learned_distillation import load_selected_native_teacher, selected_teacher_identity


def run(origin: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    config=load_config(str(origin/"native_config.yaml"))
    current=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    for key in ("native_controlled_comparators","native_scientific_training",
                "native_learned_distillation","manuscript_native_protocol"):
        config[key]=copy.deepcopy(current[key])
    config.pop("config_includes",None)
    config["selected_formal_seed"]=40
    for key in ("formal_seed_values","seeds"):config["paper_grade_full_suite"][key]=[40,41,42,43,44]
    config["paper_grade_baseline_registry"]["seed_values"]=[40,41,42,43,44]
    config["schemas"]["native_training_seed_binding"]=current["schemas"]["native_training_seed_binding"]
    benchmark=config["screening_pipeline"]["screening_benchmark"]
    benchmark["training"].update(hidden_dim=12,dropout_rate=0.,random_seed=40,seed_policy="manuscript_native_five",batch_size=1)
    benchmark["training"]["memory_policy"].update(effective_batch_size=1,micro_batch_size=1,gradient_accumulation_steps=1)
    architecture=benchmark["model_architecture"]
    architecture["world_state_posterior"].update(physical_parameter_conditioning="source_declared_named_candidates",
        posterior_hidden_dim=12,posterior_dropout_rate=0.)
    architecture["rollout_operator"]["transition_backbone"].update(operator_hidden_dim=12,operator_depth=1,
        attention_head_count=3,mechanism_state_token_count=6,operator_dropout_rate=0.)
    config["native_diffusion_data"]["posterior_context_width"]=12
    config["neuronsifter_scale_registry"]["quantized_mddm_main_rollout"]["partial_mask_schedule"].update(
        training_sample_count=2,training_denoising_step_count=1,inference_sample_count=2,inference_denoising_step_count=1)
    policy=config["native_scientific_training"]
    policy["conditioning"]={"mode":"joint"}
    policy["selection"].update(max_epochs=3,minimum_epochs=2,minimum_steps=2,patience=1,
        min_delta=1e12,max_steps=100,max_seconds=600.)
    policy["learning_rate_schedule"].update(warmup_steps=1)
    policy["validation"].update(sample_count=2,denoising_step_count=1)
    policy["predictive_budget"].update(conditional_draws_per_particle=2,categorical_reverse_steps=1)
    config_path=output/"tiny_teacher.yaml";config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    binding=resolve_training_seed(config,require_selected=True);device=torch.device("cpu")
    torch.manual_seed(40)
    teacher=training.MicroenvironmentPretrainingModel(config,False)
    assert sum(p.numel() for p in teacher.parameters())<1000000
    teacher_initial={key:value.detach().clone() for key,value in teacher.named_parameters()}
    receipt=json.loads((origin/"source_receipt.json").read_text())
    schema=parameter_schema_from_config(config);cache={}

    class RetainedGroups:
        def __init__(self,role):
            self.selected=[(i,row) for i,row in enumerate(receipt["groups"]) if row["role"]==role]
            self.group_uids=[row["sample_uid"] for _,row in self.selected]
            self.family_group_counts=dict(Counter(row["family_uid"] for _,row in self.selected))
        def __len__(self):return len(self.selected)
        def __getitem__(self,index):
            gi,definition=self.selected[index]
            if gi not in cache:
                raw=json.loads((origin/f"group_{gi}_conditioning.json").read_text())
                producer=json.loads((origin/f"group_{gi}_producer.json").read_text())
                source=bind_candidate_parameters(raw,definition,producer,schema,
                    record_ref=str(origin/f"group_{gi}_producer.json"),
                    producer_commit=receipt["source_identity"]["producer_commit"])
                support,_=parse_joint_source(source)
                mask=torch.from_numpy(quantity_support(family_grid(support.quantity_profile,support.source_family_uid)))
                items=[]
                for ai,case in enumerate(row for row in receipt["cases"] if row["group_index"]==gi):
                    with np.load(origin/case["raw_reference"]) as data:
                        target=torch.from_numpy(data["trajectory"]).float()
                    items.append({"trajectory":target,"quantity_support":mask,
                        "uncertainty":torch.tensor(case["numerical"]["uncertainty"]),
                        "teacher_confidence":torch.tensor(case["numerical"]["teacher_confidence"]),
                        "teacher_loss_weight":torch.tensor(.5),"sample_uid":definition["sample_uid"],
                        "source_family_uid":definition["family_uid"],"normalization_id":teacher.field_normalization.identity,
                        "regimen_uid":source["particle_programs"]["regimen_uids"][ai]})
                cache[gi]={"native_joint_group":{"joint_source":source,"items":items}}
            return cache[gi]
    datasets={role:RetainedGroups(role) for role in ("train","development","validation")}
    assert all(len(dataset)==4 for dataset in datasets.values())
    assert not set(datasets["train"].group_uids)&set(datasets["development"].group_uids)
    original_dataset=training.NativeJointGroupDataset
    original_declarations=evaluation.declared_dataset
    original_argv=sys.argv
    training.NativeJointGroupDataset=lambda cfg,role:datasets[role]
    def declared(cfg,role,**kwargs):
        dataset=datasets[role]
        return dataset,{row["sample_uid"]:row for _,row in dataset.selected}
    evaluation.declared_dataset=declared

    def fit(model,cfg,directory,learned_teacher=None):
        directory.mkdir()
        loader=DataLoader(datasets["train"],batch_size=1,shuffle=False,**training.data_loader_kwargs(cfg))
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4,weight_decay=1e-4)
        if learned_teacher is not None:
            ids={id(p) for group in optimizer.param_groups for p in group["params"]}
            assert not ids&{id(p) for p in learned_teacher.parameters()}
        result=training.fit_scientific_native_groups(model,optimizer,loader,cfg,device,
            cfg["screening_pipeline"]["screening_benchmark"]["training"]["memory_policy"],
            directory,"tiny_software_fixture",binding,learned_teacher=learned_teacher)
        assert result["executed_steps"]==8
        assert result["selection"]["best"]["steps"]==4 and result["selection"]["last"]["steps"]==8
        assert result["selection"]["best"]["checkpoint_path"]!=result["selection"]["last"]["checkpoint_path"]
        return result,directory/"microenvironment_pretraining_manifest.json"

    try:
        teacher_report,manifest=fit(teacher,config,output/"teacher_training")
        assert any(not torch.equal(teacher_initial[name],p) for name,p in teacher.named_parameters())
        assert teacher_report["learned_teacher"] is None and teacher_report["learned_teacher_forward_calls"]==0
        results=[];selected_states={}
        for method in ("student","student_no_distillation"):
            selected=copy.deepcopy(config)
            selected["native_scientific_training"]["conditioning"]={"mode":"frozen_selected_mainline",
                "config_path":str(config_path),"manifest_path":str(manifest)}
            selected["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"][
                "transition_backbone"].update(operator_hidden_dim=8,attention_head_count=2)
            selected["native_learned_distillation"].update(enabled=method=="student",
                teacher_config=str(config_path) if method=="student" else None,
                teacher_manifest=str(manifest) if method=="student" else None)
            selected_path=output/(method+".yaml");selected_path.write_text(yaml.safe_dump(selected,sort_keys=False))
            torch.manual_seed(41)
            model=training.MicroenvironmentPretrainingModel(selected,False)
            assert sum(p.numel() for p in model.parameters())<1000000
            conditioning=bind_frozen_training_belief(model,selected,device)
            frozen={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
            rng=torch.random.get_rng_state()
            learned=load_selected_native_teacher(model,selected,device)
            assert torch.equal(rng,torch.random.get_rng_state())
            calls=[]
            if learned is not None:
                teacher_snapshot={name:p.detach().clone() for name,p in learned.named_parameters()}
                original_logits=learned.quantized_mddm.logits
                def counted_logits(*args,**kwargs):
                    calls.append(1)
                    return original_logits(*args,**kwargs)
                learned.quantized_mddm.logits=counted_logits
                assert model.native_learned_teacher_identity["selected_teacher_optimizer_updates"]==4
                bad=copy.deepcopy(selected);bad_manifest=output/"untrained_teacher_selection.json"
                invalid=copy.deepcopy(teacher_report);invalid["selection"]["best"]["steps"]=0;invalid["executed_steps"]=0
                bad_manifest.write_text(json.dumps(invalid))
                bad["native_learned_distillation"]["teacher_manifest"]=str(bad_manifest)
                bad_conditioning=dict(conditioning,manifest_path=str(bad_manifest))
                try:selected_teacher_identity(bad,bad_conditioning)
                except ValueError as error:assert "actually fitted" in str(error)
                else:raise AssertionError("Untrained learned teacher was admitted")
            report,student_manifest=fit(model,selected,output/(method+"_training"),learned)
            assert all(torch.equal(dict(model.named_parameters())[name],value) for name,value in frozen.items())
            if learned is not None:
                assert len(calls)==report["learned_teacher_forward_calls"]>0
                assert all(p.grad is None and not p.requires_grad and torch.equal(teacher_snapshot[name],p)
                    for name,p in learned.named_parameters())
                assert not learned.training
            else:assert report["learned_teacher_forward_calls"]==0
            # The unchanged development objective never consumes the learned teacher.
            model.eval();group=datasets["development"][0]["native_joint_group"]
            count_before=len(calls)
            loss,metrics,_=training.scientific_joint_group_loss(model,group,selected,device)
            assert "learned_teacher_kl" not in metrics and len(calls)==count_before
            disabled=copy.deepcopy(selected);disabled["native_learned_distillation"]["enabled"]=False
            ordinary,_,_=training.scientific_joint_group_loss(model,group,disabled,device)
            torch.testing.assert_close(loss,ordinary,rtol=0,atol=0)
            checkpoint=torch.load(training_checkpoint_path(student_manifest,report["selection"],report["selection"]["best"]["checkpoint_path"]),map_location="cpu",weights_only=False)
            assert not any("learned_teacher" in key for key in checkpoint["model_state_dict"])
            selected_states[method]=checkpoint["model_state_dict"]
            base=["run_external_baseline_bridge.py","--protocol-scope","manuscript-native","--config",str(selected_path),
                "--baseline-name",method,"--device","cpu","--conditioning-config",str(config_path),
                "--conditioning-manifest",str(manifest),"--training-manifest",str(student_manifest)]
            sys.argv=[*base,"--command","evaluate_baseline","--output-dir",str(output/(method+"_evaluation"))]
            assert cli.main()==0
            evaluated=json.loads((output/(method+"_evaluation")/"native_comparator_evaluation.json").read_text())
            assert evaluated["status"]=="complete" and len(evaluated["groups"])==4
            assert all(row["status"]=="defined" for row in evaluated["groups"])
            if learned is not None:
                corrupt=copy.deepcopy(report);corrupt["learned_teacher"]["binding"]["checkpoint_sha256"]="0"*64
                corrupt_path=output/"wrong_teacher_identity.json";corrupt_path.write_text(json.dumps(corrupt))
                try:evaluation.load_native_comparator(selected,method,config_path,manifest,corrupt_path,datasets["validation"],device)
                except ValueError as error:assert "learned-teacher" in str(error)
                else:raise AssertionError("Mismatched saved learned teacher was accepted")
            results.append({"method":method,"optimizer_updates":8,"selected_updates":4,
                "learned_teacher_forward_calls":report["learned_teacher_forward_calls"],
                "evaluation_group_count":4,"frozen_teacher":learned is not None})
            print("TRAIN_SELECTED_EVALUATED",method,json.dumps(results[-1]),flush=True)
        assert any(not torch.equal(value,selected_states["student_no_distillation"][key])
            for key,value in selected_states["student"].items() if key.startswith("operator.") and torch.is_tensor(value))
        return {"teacher_optimizer_updates":8,"teacher_selected_updates":4,"teacher_train_groups":4,
            "teacher_development_groups":4,"methods":results,"source_operations":0,
            "training_scope":"tiny_software_fixture","formal_qualification":False,"scientific_twin":"NOT_RUN"}
    finally:
        training.NativeJointGroupDataset=original_dataset
        evaluation.declared_dataset=original_declarations
        sys.argv=original_argv
