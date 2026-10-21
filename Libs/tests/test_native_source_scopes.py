"""Matched source scope consumers on artificial source-format software fixtures."""
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data.native_field_normalization import (
    fit_multigrid, bind_artifact, normalization_spec, NativeFieldNormalization,
)
from Libs.utils.data.native_observable_profile import observable_profile, family_grid, quantity_support
from Libs.utils.data.joint_physical_parameters import bind_candidate_parameters, parameter_schema_from_config
from Libs.utils.evaluation.manuscript_native_protocol import training_profile, command_catalog
from Libs.utils.model import native_scientific_source as source
from Libs.tests.test_native_romani_integration import initialize_fixture


def test_matched_groups_conditions_actions_and_independent_dependencies():
    cfg=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    profiles={}
    for method in ("native_single_source_wam","native_mixed_source_wam","state_single_source","state_mixed_source","state_joint"):
        selected,spec=training_profile(cfg,method,41)
        profiles[method]=(selected,spec)
    pure=source.selected_group_definitions(profiles["state_single_source"][0],"train")
    mixed=source.selected_group_definitions(profiles["state_mixed_source"][0],"train")
    assert len(pure)==len(mixed)==3250
    assert len(source.selected_group_definitions(profiles["state_joint"][0],"train"))==4000
    for groups in (pure,mixed):
        electrical=[g for g in groups if g["family_uid"]==source.ELECTRICAL]
        assert Counter(g["condition"] for g in electrical)=={"control":125,"ad_positive":125}
        assert len(electrical)==250 and sum(source.action_count(cfg,g) for g in groups)==9750
        assert sorted(Counter(g["family_uid"] for g in groups).values())==[250,1000,1000,1000]
    assert all(g["source_variant"]!="romani" for g in pure)
    assert Counter(g["source_variant"] for g in mixed)=={"primary":3125,"romani":125}
    assert {g["sample_uid"] for g in pure if g["family_uid"]!=source.ELECTRICAL}=={
        g["sample_uid"] for g in mixed if g["family_uid"]!=source.ELECTRICAL}
    assert profiles["state_single_source"][1]["budget"]==profiles["state_mixed_source"][1]["budget"]
    commands=command_catalog(cfg,[row[1] for row in profiles.values()])
    byid={row["command_uid"]:row for row in commands}
    for scope,base,probe in (("pure_shared","native_single_source_wam","state_single_source"),
                             ("mixed_shared","native_mixed_source_wam","state_mixed_source")):
        selected,base_spec=profiles[base];_,probe_spec=profiles[probe]
        assert selected["native_scientific_training"]["conditioning"]["mode"]=="joint"
        assert selected["native_diffusion_data"]["field_normalization"]["model_artifact_ref"].endswith(scope+".json")
        assert byid[base_spec["profile_id"]+":fit"]["dependencies"]==["named_parameter_admission"]
        assert base_spec["profile_id"]+":fit" in byid[probe_spec["profile_id"]+":fit"]["dependencies"]
        assert not any("native_wam_fraction_100" in value for value in byid[probe_spec["profile_id"]+":fit"]["dependencies"])


def fixture_scalers(tmp_path, *, reuse_existing=False):
    cfg=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    cfg.pop("config_includes",None)
    settings=cfg["native_observable_bridge"]
    settings["group_partitions"]=[dict(settings["group_partitions"][0],groups_per_family=8),
                                  dict(settings["group_partitions"][1],groups_per_family=1)]
    cfg["manuscript_native_protocol"]["source_comparison"]["electrical_groups_per_arm"]=2
    cfg["manuscript_native_protocol"]["source_comparison"]["normalization_root"]=str(tmp_path/"scopes")
    cfg["native_diffusion_data"]["source_family_uid"]=None
    groups=source.group_definitions(cfg)
    profile=observable_profile(cfg)
    raw_by_uid={};generated_by_uid={};targets={}
    for gi,group in enumerate(groups):
        if group["family_uid"]!=source.ELECTRICAL:
            continue
        current=source.scoped_group_config(cfg,group)
        generated=source.draw_group(current,group,gi,physical_initializer=initialize_fixture
                                    if group["source_variant"]=="romani" else None)
        records=[[source.compiled_program(current,group,generated["support"],atom,
                    generated["parameters"][atom],source.action_definition(current,source.ELECTRICAL,ai,group),"software_fixture")
                  for ai in range(3)] for atom in range(32)]
        raw=source.joint_source(current,group,generated,records)
        producer=tmp_path/(hashlib.sha256(group["sample_uid"].encode()).hexdigest()+".json")
        producer_payload={"hidden_index":generated["hidden_index"],"parameters":generated["parameters"]}
        if reuse_existing:
            assert json.loads(producer.read_text())==producer_payload
        else:
            producer.write_text(json.dumps(producer_payload))
        raw=bind_candidate_parameters(raw,group,{"parameters":generated["parameters"]},parameter_schema_from_config(cfg),
                                     record_ref=str(producer),producer_commit="software_fixture_not_source_evidence")
        raw_by_uid[group["sample_uid"]]=raw;generated_by_uid[group["sample_uid"]]=generated
    batches=[]
    for family in profile["families"]:
        grid=family_grid(profile,family);rows=[];values=[];masks=[]
        for gi,group in enumerate(groups):
            if group["family_uid"]!=family:continue
            mask=quantity_support(grid,group["source_uid"])
            raw=raw_by_uid.get(group["sample_uid"])
            initial=(np.asarray(raw["support"]["initial_field"])[0] if raw else
                     np.broadcast_to(np.where(mask[0],.1,0.),(32,*mask.shape[1:])).copy())
            for ai in range(3):
                ramp=np.linspace(0.,1.,len(grid["time_hours"]))[:,None,None,None]
                target=np.where(mask,initial[0][None]+ramp*(.04+.02*gi+.003*ai),0.).astype(np.float32)
                uid=group["sample_uid"]+":fixture_action:"+str(ai)
                teacher={"trajectory_uid":uid,"source_uid":group["source_uid"],"sample_uid":group["sample_uid"],
                    "regimen_uid":raw["particle_programs"]["regimen_uids"][ai] if raw else uid+":regimen",
                    "posterior_uid":group["posterior_uid"],"family_uid":family,
                    "content_hash":hashlib.sha256(target.tobytes()).hexdigest(),"trajectory_store_path":"software_fixture",
                    "trajectory_index":len(rows),"source_snapshot_ref":group["source_snapshot_ref"],
                    "split_name":group["role"],"source_holdout_role":group["role"],"qc_grade":"A","sample_weight":1.}
                target_record={"teacher":teacher,"joint_source":raw or {"support":{
                    "sample_uids":[group["sample_uid"]],"initial_field":[initial.tolist()],
                    "support_uid":group["support_uid"],"particle_uids":[group["particle_uids"]]},
                    "source_posterior_uid":group["posterior_uid"]}}
                targets.setdefault(group["sample_uid"],[]).append((teacher,target,mask))
                if group["role"]=="train":
                    rows.append(target_record);values.append(target);masks.append(mask)
        batches.append((grid,rows,{"trajectory":np.stack(values),"quantity_support":np.stack(masks)}))
    basepath=tmp_path/"base.json"
    if not reuse_existing:
        base=fit_multigrid(batches,cfg,streaming=True)
        basepath.write_text(json.dumps(base))
    from Libs.scripts.builders.build_controlled_native_source import scientific_index
    refs={};grids={}
    for grid,_,_ in batches:
        bound=bind_artifact(grid,basepath,cfg);grids[grid["family_uid"]]=bound
        path=tmp_path/(grid["family_uid"]+".json")
        payload={"schema_version":native.SCHEMA_VERSION,"grid":bound,"sources":[],
            "teachers":[],"programs":[],"observations":[],"joint_sources":[],"posterior_family":"finite_joint_particles"}
        if reuse_existing:
            assert json.loads(path.read_text())==payload
        else:
            path.write_text(json.dumps(payload))
        refs[grid["family_uid"]]=str(path)
    index=tmp_path/"index.json"
    if reuse_existing:
        assert json.loads(index.read_text())==scientific_index(cfg,refs)
    else:
        index.write_text(json.dumps(scientific_index(cfg,refs)))
    cfg["native_diffusion_data"]["registry_path"]=str(index)
    scoped={}
    for scope in ("pure_shared","mixed_shared"):
        current=copy.deepcopy(cfg);current["native_scientific_training"].update(source_scope=scope,training_subset_percentage=100)
        ids={g["sample_uid"] for g in source.selected_group_definitions(current,"train")}
        filtered=[]
        for grid,rows,arrays in batches:
            take=[i for i,row in enumerate(rows) if row["teacher"]["sample_uid"] in ids]
            filtered.append((grid,[rows[i] for i in take],{k:v[take] for k,v in arrays.items()}))
        path=tmp_path/(scope+".json")
        if reuse_existing:
            artifact=json.loads(path.read_text())
            assert artifact["training_data_scope"]==source.training_data_scope(current)
            assert artifact["source_grid_normalization_sha256"]==hashlib.sha256(basepath.read_bytes()).hexdigest()
        else:
            artifact=fit_multigrid(filtered,current,streaming=True)
            artifact.update(training_data_scope=source.training_data_scope(current),
                            source_grid_normalization_sha256=hashlib.sha256(basepath.read_bytes()).hexdigest())
            path.write_text(json.dumps(artifact))
        current["native_diffusion_data"]["field_normalization"]["model_artifact_ref"]=str(path)
        scoped[scope]=current
    return cfg,scoped,grids,raw_by_uid,targets,groups


def test_scoped_scalers_preserve_source_grid_and_reject_cross_scope(tmp_path):
    cfg,scoped,grids,raw,targets,groups=fixture_scalers(tmp_path)
    grid=grids[source.ELECTRICAL];before=copy.deepcopy(grid)
    specs={scope:normalization_spec(grid,current) for scope,current in scoped.items()}
    assert specs["pure_shared"]["sha256"]!=specs["mixed_shared"]["sha256"]
    for scope,spec in specs.items():
        model=NativeFieldNormalization(spec,26)
        model.assert_binding(grid)
        assert len({row["sample_uid"] for row in spec["artifact"]["training_records"]})==26
        assert len(spec["artifact"]["training_records"])==78
        expected={g["sample_uid"] for g in source.selected_group_definitions(scoped[scope],"train")}
        assert {row["sample_uid"] for row in spec["artifact"]["training_records"]}==expected
        if scope=="pure_shared":
            assert all(row["source_uid"]!=cfg["native_romani_source"]["source_uid"] for row in spec["artifact"]["training_records"])
    changed=copy.deepcopy(scoped["pure_shared"])
    changed["native_diffusion_data"]["field_normalization"]["model_artifact_ref"]=scoped["mixed_shared"]["native_diffusion_data"]["field_normalization"]["model_artifact_ref"]
    with pytest.raises(ValueError,match="scope differs"):
        normalization_spec(grid,changed)
    assert grid==before


def test_tiny_isolated_native_fit_strict_reload_and_state_probe(tmp_path):
    from Libs.exp.training import pretrain_microenvironment_world_model as training
    from Libs.utils.exp.experiment_manifests import resolve_training_seed
    from Libs.utils.exp.native_comparator_training import load_frozen_native_belief, fit_native_comparator, comparator_identity
    from Libs.utils.evaluation.native_comparator_evaluation import load_native_comparator
    from Libs.utils.evaluation.native_state_evaluation import evaluate_native_state
    resume=os.environ.get("NRNSIFT_SCOPE_RESUME")
    if resume:
        tmp_path=Path(resume)
    _,scoped,grids,raws,targets,groups=fixture_scalers(tmp_path,reuse_existing=bool(resume))
    receipts=[];counts=[]
    for scope in ("pure_shared","mixed_shared"):
        cfg=scoped[scope];cfg["selected_formal_seed"]=40
        traincfg=cfg["screening_pipeline"]["screening_benchmark"]["training"]
        traincfg.update(random_seed=40,seed_policy="manuscript_native_five",batch_size=1)
        traincfg["memory_policy"].update(effective_batch_size=1,micro_batch_size=1,gradient_accumulation_steps=1)
        cfg["native_scientific_training"]["conditioning"]={"mode":"joint"}
        cfg["native_scientific_training"]["selection"].update(max_epochs=1,minimum_epochs=1,minimum_steps=1,max_steps=2,max_seconds=600)
        cfg["native_scientific_training"]["validation"].update(sample_count=2,denoising_step_count=1,interval_optimizer_steps=1)
        cfg["native_scientific_training"]["predictive_budget"].update(conditional_draws_per_particle=2,weighted_forecast_paths_per_action=64,categorical_reverse_steps=1)
        cfg["native_scientific_training"]["learning_rate_schedule"]["warmup_steps"]=1
        backbone=cfg["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
        backbone.update(operator_hidden_dim=12,operator_depth=1,attention_head_count=3,mechanism_state_token_count=6)
        cfg["native_diffusion_data"]["posterior_context_width"]=12
        assert training.training_memory_policy(cfg,False)["effective_batch_size"]==1
        for candidate in ("native_single_source_wam","native_mixed_source_wam","state_single_source","state_mixed_source"):
            identity=comparator_identity(cfg,candidate,{})
            assert identity["reference"]["implementation_kind"]=="maintained_project_native"
            assert identity["training_data_scope"]==source.training_data_scope(cfg)
        torch.manual_seed(40)
        model=training.MicroenvironmentPretrainingModel(cfg,False)
        model.quantized_mddm.training_sample_count=2;model.quantized_mddm.training_denoising_step_count=1
        model.quantized_mddm.inference_sample_count=2;model.quantized_mddm.inference_denoising_step_count=1
        model.native_scientific_profile=source.native_training_profile(cfg,model.field_normalization.identity)
        counts.append(sum(p.numel() for p in model.parameters()))
        selected=[g for g in source.selected_group_definitions(cfg,"train") if g["family_uid"]==source.ELECTRICAL]
        development=next(g for g in groups if g["role"]=="development" and g["family_uid"]==source.ELECTRICAL)
        def item(group):
            uid=group["sample_uid"];rows=[]
            for teacher,target,mask in targets[uid]:
                rows.append({"trajectory":torch.from_numpy(target),"quantity_support":torch.from_numpy(mask),
                    "uncertainty":torch.ones(26)*.01,"teacher_confidence":torch.tensor(1.),
                    "teacher_loss_weight":torch.tensor(1.),"sample_ids":torch.tensor(native.stable_numeric_id(teacher["trajectory_uid"])),
                    "normalization_id":model.field_normalization.identity,**{key:teacher[key] for key in
                    ("trajectory_uid","sample_uid","regimen_uid","posterior_uid")},"source_family_uid":source.ELECTRICAL})
            return {"native_joint_group":{"items":rows,"joint_source":raws[uid]}}
        class FixtureGroups(list):
            def __init__(self, declarations):
                super().__init__(item(g) for g in declarations)
                self.family_group_counts=dict(Counter(g["family_uid"] for g in declarations))
                self.group_uids=[g["sample_uid"] for g in declarations]
        train=FixtureGroups(selected);dev=FixtureGroups([development])
        output=tmp_path/scope
        manifest_path=output/"manifest.json"
        seed=resolve_training_seed(cfg,require_selected=True)
        scores=None
        reused=bool(resume and scope=="pure_shared")
        if reused:
            manifest=json.loads(manifest_path.read_text())
            assert manifest["executed_steps"]==2
            assert manifest["training_seed_binding"]==seed
            assert manifest["native_scientific_profile"]==model.native_scientific_profile
            assert manifest["selection"]["best"]["steps"]==2
            assert Path(manifest["selection"]["best"]["checkpoint_path"])==output/"base_2.pt"
        else:
            output.mkdir()
            optimizer=training.create_adamw_optimizer(model,learning_rate=1e-4,weight_decay=1e-4)
            scores=[];checkpoints=[]
            for step,group in enumerate(train,1):
                model.train()
                loss,_,status=training.scientific_joint_group_loss(model,group["native_joint_group"],cfg,torch.device("cpu"))
                assert status["particle_count"]==32 and status["paired_target_count"]==3
                loss.backward();training.execute_optimizer_step(model,optimizer,cfg)
                model.eval()
                with torch.no_grad():
                    score,_,_=training.scientific_joint_group_loss(model,dev[0]["native_joint_group"],cfg,torch.device("cpu"))
                scores.append(float(score))
                checkpoints.append(training.save_training_checkpoint(model,optimizer,output,"base_"+str(step),step,step,
                                    "single_process",training_seed_binding=seed,immutable=True))
            best=int(np.argmin(scores));chosen={"checkpoint_path":str(checkpoints[best]),"steps":best+1}
            manifest={"training_seed_binding":seed,"executed_steps":2,"native_scientific_profile":model.native_scientific_profile,
                "selection":{"selection_role":"development","best":chosen,"last":{"checkpoint_path":str(checkpoints[-1]),"steps":2},
                             "development_scores":scores,
                             "terminal":"software_fixture_two_update_budget","sufficient_training":False}}
            manifest_path.write_text(json.dumps(manifest))
        path=output/("config.ordered.yaml" if reused else "config.yaml")
        path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        belief,binding=load_frozen_native_belief(cfg,path,manifest_path,"cpu")
        method="native_single_source_wam" if scope=="pure_shared" else "native_mixed_source_wam"
        loaded,_,reload_binding=load_native_comparator(cfg,method,path,manifest_path,manifest_path,None,"cpu")
        assert reload_binding["normalization_id"]==model.field_normalization.identity
        assert loaded.mainline.world_state_posterior.joint_candidate_parameter_projection.in_features==442
        print(scope+": native and frozen belief strict reload passed; reused="+str(reused),flush=True)
        probe_method="state_single_source" if scope=="pure_shared" else "state_mixed_source"
        probe_report=fit_native_comparator(belief,probe_method,train,dev,cfg,output/"probe",binding,"cpu")
        probe,_,probe_binding=load_native_comparator(cfg,probe_method,path,manifest_path,
                                    output/"probe/native_comparator_training_manifest.json",None,"cpu")
        report=evaluate_native_state(cfg,probe_method,probe,probe_binding,dev,
                                    {development["sample_uid"]:development},output/"evaluation")
        assert all(row["status"]=="defined" for row in report["groups"])
        receipts.append({"scope":scope,"normalization_id":model.field_normalization.identity,"native_updates":2,
                         "native_parameter_count":counts[-1],"state_updates":probe_report["executed_steps"],
                         "development_scores":scores,"reused_native_checkpoint":reused,
                         "selected_checkpoint_path":manifest["selection"]["best"]["checkpoint_path"],
                         "development_score_status":"not_retained_by_original_fixture" if reused else "recorded",
                         "strict_reload":True,"source_grid_unchanged":grids[source.ELECTRICAL]["normalization"]["sha256"]==model.field_normalization.spec["source_grid_normalization_sha256"]})
        print(scope+": independent probe fit/reload and state evaluation passed",flush=True)
        if os.environ.get("NRNSIFT_SCOPE_EVIDENCE"):
            Path(os.environ["NRNSIFT_SCOPE_EVIDENCE"]).write_text(json.dumps({
                "scope":"artificial_software_fixture_no_source_trajectory","status":"partial",
                "results":receipts,"scientific_source_admission":"NOT_RUN"},indent=2))
        if scope=="pure_shared":pure=(copy.deepcopy(cfg),path,manifest_path)
        del model,loaded,belief,probe,train,dev
    assert counts[0]==counts[1]
    with pytest.raises(ValueError,match="differently scoped belief"):
        load_frozen_native_belief(pure[0],path,manifest_path,"cpu")
    if os.environ.get("NRNSIFT_SCOPE_EVIDENCE"):
        Path(os.environ["NRNSIFT_SCOPE_EVIDENCE"]).write_text(json.dumps({"scope":"artificial_software_fixture_no_source_trajectory",
            "status":"complete","cross_scope_rejected":True,"results":receipts,"scientific_source_admission":"NOT_RUN"},indent=2))


def test_native_dataset_items_uses_scoped_transform_and_original_grid(tmp_path,monkeypatch):
    from Libs.exp.training import pretrain_microenvironment_world_model as training
    saved=os.environ.get("NRNSIFT_SCOPE_RESUME")
    if saved:
        root=Path(saved)
        scoped={scope:load_config(str(root/scope/("config.ordered.yaml" if scope=="pure_shared" else "config.yaml")))
                for scope in ("pure_shared","mixed_shared")}
        grid=json.loads((root/(source.ELECTRICAL+".json")).read_text())["grid"]
    else:
        _,scoped,grids,_,_,_=fixture_scalers(tmp_path)
        grid=grids[source.ELECTRICAL]
    original=copy.deepcopy(grid)
    teacher={"trajectory_uid":"fixture:trajectory","sample_uid":"fixture:sample",
             "regimen_uid":"fixture:regimen","posterior_uid":"fixture:posterior"}
    shape=quantity_support(grid).shape
    arrays={"trajectory":np.zeros((1,*shape),dtype=np.float32),
            "quantity_support":quantity_support(grid)[None],"uncertainty":np.ones((1,26),dtype=np.float32)}
    def retained_records(registry, records, config, *, input_locations=None, normalization_context=None):
        assert input_locations is None and normalization_context is None
        return arrays, {}
    monkeypatch.setattr(training, "load_records", retained_records)
    seen=[]
    for scope,cfg in scoped.items():
        spec=normalization_spec(grid,cfg)
        NativeFieldNormalization(spec,26).assert_binding(grid)
        items=training.native_dataset_items({"grid":grid},[{"teacher":teacher}],cfg,[1.])
        assert items[0]["normalization_id"]==spec["sha256"]
        assert items[0]["normalization_id"]!=grid["normalization"]["sha256"]
        assert torch.equal(items[0]["trajectory"],torch.from_numpy(arrays["trajectory"][0]))
        seen.append(items[0]["normalization_id"])
    assert seen[0]!=seen[1] and grid==original
