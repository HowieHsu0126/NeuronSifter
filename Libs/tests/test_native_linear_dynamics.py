"""Source-free affine dynamics and maintained direct-consumer evidence."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from Libs.models.modules.native_linear_dynamics import NativeLinearDynamics, LINEAR_DYNAMICS_SPEC
from Libs.utils.data.native_field_normalization import NativeFieldNormalization
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.model.native_direct_mean import predict_direct_mean
from Libs.utils.model.screening_model_config import native_direct_mean_config, screening_model_kwargs_from_config
from Libs.utils.exp.experiment_manifests import resolve_training_seed, load_experiment_family
from Libs.exp.training import pretrain_microenvironment_world_model as training
from Libs.exp.evaluation import evaluate_screening_system as evaluation
from Libs.tests.test_expert_composition import ten_expert_case
from Libs.tests.test_native_field_normalization import joint_normalized_case
from Libs.tests.test_native_transolver_mean import setup, point_training_config, MEMORY_POLICY
from Libs.tests.test_native_rollout_evaluation import run_cli

ROOT = Path(__file__).resolve().parents[2]


def mathematical_case(state_count=2, clock=(1.,1.3,2.)):
    domain={"region_id":["region"],"compartment_id":["compartment"],
            "species_id":[f"species-{i}" for i in range(state_count)],
            "species_units":{f"species-{i}":"fixture_unit" for i in range(state_count)},
            "time_hours":list(clock),"physical_time_unit":"hour"}
    # Known fixed buffers isolate the affine math; persisted R60 admission is
    # exercised by the real full-grid conditioning/checkpoint fixtures below.
    normalizer=NativeFieldNormalization({"kind":"physical"},state_count)
    normalizer.active=True
    normalizer.artifact={"domain":copy.deepcopy(domain)}
    normalizer.mean=torch.arange(state_count,dtype=torch.float64)+3
    normalizer.scale=torch.arange(state_count,dtype=torch.float64)+2
    model=NativeLinearDynamics(domain,2).double()
    context={"initial_state":(normalizer.mean+normalizer.scale*torch.arange(1,state_count+1,dtype=torch.float64)).reshape(1,1,1,1,-1),
             "posterior_context":torch.tensor([.2,.4],dtype=torch.float64).reshape(1,1,1,1,2),
             "occupancy":torch.full((1,1,len(clock),1,1,1),.3,dtype=torch.float64),
             "support":torch.ones(1,1,len(clock),1,1,1,dtype=torch.float64),
             "physical_time":torch.tensor(clock,dtype=torch.float64).reshape(1,1,-1,1,1,1)}
    return model,normalizer,context


@pytest.mark.parametrize("rates", [[.2],[-.2,.3],[0.,0.]])
def test_linear_analytic_diagonal_and_zero_forcing(rates):
    model,norm,context=mathematical_case(len(rates))
    with torch.no_grad():
        model.A.copy_(torch.diag(torch.tensor(rates,dtype=torch.float64)))
        model.B[:,0]=torch.arange(1,len(rates)+1,dtype=torch.float64)
        model.C.fill_(.2)
        model.b.fill_(.1)
    result=model(context,norm)
    x0=norm.normalize(context["initial_state"]).flatten()
    forcing=model.B[:,0]*.3+model.B[:,1]+model.C@context["posterior_context"].flatten()+model.b
    expected=[]
    for time in norm.artifact["domain"]["time_hours"]:
        elapsed=time-1.
        values=[]
        for i,rate in enumerate(rates):
            phi=math.exp(rate*elapsed)
            psi=elapsed if rate==0 else math.expm1(rate*elapsed)/rate
            values.append(phi*x0[i].item()+psi*forcing[i].item())
        expected.append(values)
    torch.testing.assert_close(result["normalized_trajectory"].reshape(len(expected),-1),torch.tensor(expected,dtype=torch.float64),rtol=1e-12,atol=1e-12)
    assert torch.equal(result["field_trajectory"][:,:,0],context["initial_state"])
    torch.testing.assert_close(result["field_trajectory"][:,:,1:],norm.inverse(result["normalized_trajectory"][:,:,1:]))
    assert result["router_diagnostics"]["status"]=="not_applicable"
    assert not {"router_entropy","expert_utilization"} & set(result["router_diagnostics"])


def test_linear_coupled_prefix_and_forward_local_transition_reuse(monkeypatch):
    model,norm,context=mathematical_case(2,(0.,.5,1.,2.))
    with torch.no_grad():
        model.A[0,1]=.4
        model.B[:,0]=torch.tensor([.2,.7])
        model.b.copy_(torch.tensor([.1,-.2]))
    context={key:value.expand(1,2,*value.shape[2:]).clone() for key,value in context.items()}
    context["occupancy"][:,1,2:]=.9  # same through knot2, different forcing afterwards
    calls=[]
    original=torch.matrix_exp
    def recorded(matrix):
        assert matrix.dtype==torch.float64
        calls.append(matrix.detach().clone())
        return original(matrix)
    monkeypatch.setattr(torch,"matrix_exp",recorded)
    with torch.autocast(device_type="cpu",dtype=torch.bfloat16):
        result=model(context,norm)
    assert len(calls)==2
    assert torch.equal(result["field_trajectory"][:,0,:3],result["field_trajectory"][:,1,:3])
    assert not torch.equal(result["field_trajectory"][:,0,3:],result["field_trajectory"][:,1,3:])
    expected=torch.tensor([1.,2.],dtype=torch.float64)
    rows=[expected.clone()]
    for delta,p in ((.5,.3),(.5,.3),(1.,.3)):
        u=torch.tensor([.2*p+.1,.7*p-.2],dtype=torch.float64)
        expected=torch.stack((expected[0]+.4*delta*expected[1]+delta*u[0]+.2*delta**2*u[1],expected[1]+delta*u[1]))
        rows.append(expected)
    torch.testing.assert_close(result["normalized_trajectory"][0,0].reshape(4,2),torch.stack(rows),rtol=1e-8,atol=1e-9)
    model(context,norm)
    assert len(calls)==4  # nothing retained across calls


def test_linear_zero_initialization_gradients_and_full_state_parameters():
    model,norm,context=mathematical_case(3)
    assert sum(p.numel() for p in model.parameters()) == 3*(3+2+2+1)
    assert all(not torch.count_nonzero(p) for p in model.parameters())
    output=model(context,norm)
    assert torch.equal(output["field_trajectory"],context["initial_state"].unsqueeze(2).expand(-1,-1,3,-1,-1,-1))
    output["normalized_trajectory"][:,:,-1].sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad) for p in model.parameters())
    gradient=model.A.grad[0,1].item()
    epsilon=1e-6
    with torch.no_grad():
        model.A[0,1]=epsilon
        upper=model(context,norm)["normalized_trajectory"][:,:,-1].sum()
        model.A[0,1]=-epsilon
        lower=model(context,norm)["normalized_trajectory"][:,:,-1].sum()
    assert gradient==pytest.approx(((upper-lower)/(2*epsilon)).item(),rel=1e-7)


@pytest.mark.parametrize("bad",["missing","initial_shape","context_shape","program_shape","nonfinite",
                                  "fraction","duplicate_time","decreasing_time","wrong_time","wrong_domain","unbound_normalization"])
def test_linear_invalid_inputs_reject(bad):
    model,norm,context=mathematical_case()
    if bad=="missing": context.pop("support")
    elif bad=="initial_shape": context["initial_state"]=context["initial_state"][...,:1]
    elif bad=="context_shape": context["posterior_context"]=context["posterior_context"][...,:1]
    elif bad=="program_shape": context["occupancy"]=context["occupancy"][:,:,:2]
    elif bad=="nonfinite": context["posterior_context"].fill_(float("nan"))
    elif bad=="fraction": context["support"].fill_(1.1)
    elif bad=="duplicate_time": context["physical_time"][:,:,1]=1.
    elif bad=="decreasing_time": context["physical_time"][:,:,1]=.9
    elif bad=="wrong_time": context["physical_time"][:,:,1]=1.4
    elif bad=="wrong_domain": norm.artifact["domain"]["species_id"].reverse()
    else: norm.active=False
    with pytest.raises(ValueError):model(context,norm)


def test_linear_state_identity_and_nonfinite_checkpoint_reject():
    model,_,_=mathematical_case()
    for field in ("definition","domain","posterior_context_dim"):
        state=copy.deepcopy(model.state_dict())
        state["_extra_state"][field]={} if field!="posterior_context_dim" else 3
        with pytest.raises(ValueError,match="identity"):model.load_state_dict(state,strict=False)
    state=copy.deepcopy(model.state_dict());state["A"][0,0]=float("nan")
    with pytest.raises(ValueError,match="finite"):model.load_state_dict(state)
    state=copy.deepcopy(model.state_dict());state["C"]=state["C"][:,:1]
    with pytest.raises(ValueError,match="shape"):model.load_state_dict(state)


def linear_config(config):
    config=copy.deepcopy(config)
    old=config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["native_direct_mean"]
    canonical=load_config("Libs/configs/config_stack.yaml")
    template=copy.deepcopy(canonical["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["linear_dynamics_template"])
    for key in ("conditioning_config_path","conditioning_checkpoint_path","posterior_context_dim"):
        template["native_direct_mean"][key]=old[key]
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]=template
    config["selected_experiment_family"]="rollout_operator_linear_dynamics"
    config["paper_grade_full_suite"]["native_rollout_evaluation"]["predictor_kind"]="linear_dynamics_direct_mean"
    return config


def bind_membrane_fixture(config, directory):
    from Libs.models.modules.world_state import JointParticleSupport
    from Libs.tests.test_joint_world_posterior import local_membrane_case,raw_joint_source
    from Libs.tests.test_physical_occupancy import make_packet
    from Libs.scripts.builders.build_sample_observation_registry import produce_controlled_observations
    from Libs.scripts.teacher_generation.build_osp_pbpk_teacher_rollouts import build_native_physical_source
    from Libs.utils.model.microenvironment_teacher import write_native_microenvironment_store
    config=copy.deepcopy(config)
    registry=copy.deepcopy(native.load_registry(config))
    records=native.join_registry(registry,config)
    arrays,_=native.load_records(registry,records,config)
    source=registry["joint_sources"][0]
    support=JointParticleSupport.from_payload(source["support"])
    keys=("channel_uid","observation_uid","observed","encoder_indices","library_offset","detection_limit","normalization_ref","modality","assay_uid")
    channels=[{key:row[key] for key in keys if row.get(key) is not None} for row in source["observations"]]
    support,channels=local_membrane_case(support,channels,registry["grid"])
    production=produce_controlled_observations(support,channels,latent_seed=41,observation_seed=42)
    packet=make_packet(directory/"exports",config,raw_joint_source(support,production))
    source=build_native_physical_source(config,packet,directory/"exports")
    registry["joint_sources"]=[source]
    store=directory/"source"
    for index,row in enumerate(registry["teachers"]):
        program=source["particle_programs"]["records"][index//2][0][index%2]
        row.update({key:program[key] for key in ("program_uid","posterior_uid","action_uid","target_program_uid")})
        row.update(trajectory_store_path=str(store),store_format="zarr_v2",trajectory_index=index)
    write_native_microenvironment_store(store,config,registry,arrays,"zarr_v2")
    config["native_diffusion_data"]["registry_path"]=str(store/".zattrs")
    return config


def train_linear(config,directory):
    model=training.MicroenvironmentPretrainingModel(config,False)
    assert model.operator is None and model.quantized_mddm is None and model.world_state_posterior is None
    assert set(dict(model.named_parameters()))=={"direct_mean.A","direct_mean.B","direct_mean.C","direct_mean.b"}
    assert model.direct_mean.A.shape==(540,540)
    assert model.direct_mean.B.shape==(540,72) and model.direct_mean.C.shape==(540,432)
    dataset=training.MicroenvironmentTeacherFamilyDataset(config,"train")
    batch=training.collate_native_sources(dataset.__getitems__([0,1]))
    before={key:p.detach().clone() for key,p in model.named_parameters()}
    frozen=copy.deepcopy(model.conditioning["model"].state_dict())
    optimizer=training.create_adamw_optimizer(model,learning_rate=.0001,weight_decay=0.)
    metrics,report=training.run_pretraining_step(model,batch,config,torch.device("cpu"),MEMORY_POLICY)
    assert report["router_status"]=="not_applicable" and metrics["router_regularization_loss"]==0
    training.execute_optimizer_step(model,optimizer,config)
    assert any(not torch.equal(before[key],p) for key,p in model.named_parameters())
    assert all(torch.equal(value,model.conditioning["model"].state_dict()[key]) for key,value in frozen.items())
    assert all(not p.requires_grad and p.grad is None for p in model.conditioning["model"].parameters())
    directory.mkdir()
    checkpoint=training.save_training_checkpoint(model,optimizer,directory,"linear",1,1,"single_process",
        training_seed_binding=resolve_training_seed(config,require_selected=True))
    return model.eval(),checkpoint,batch,report


@pytest.fixture(scope="module")
def linear_joint_case(ten_expert_case,tmp_path_factory):
    directory=tmp_path_factory.mktemp("linear_joint")
    evaluation_config,training_config,*_=setup(ten_expert_case,directory)
    evaluation_config=bind_membrane_fixture(linear_config(evaluation_config),directory/"development_membrane")
    training_config=bind_membrane_fixture(linear_config(training_config),directory/"training_membrane")
    model,checkpoint,batch,report=train_linear(training_config,directory/"trained")
    return directory,evaluation_config,training_config,model,checkpoint,batch,report


def test_linear_real_joint_optimizer_checkpoint_cli_and_target_isolation(linear_joint_case,tmp_path,monkeypatch):
    directory,config,train_config,model,checkpoint,batch,report=linear_joint_case
    assert sum(p.numel() for p in model.parameters())==564300
    saved=torch.load(checkpoint,map_location="cpu")
    assert saved["executed_steps"]==1 and saved["optimizer_state_dict"]["state"]
    assert not any("conditioning" in key for key in saved["model_state_dict"])
    captures=[]
    original=evaluation.predict_native_episode
    def capture(*args,**kwargs):
        result=original(*args,**kwargs);captures.append(result);return result
    monkeypatch.setattr(evaluation,"predict_native_episode",capture)
    payload,_=evaluation.evaluate_native_rollout_prediction(config,checkpoint)
    assert payload["status"]=="incomplete" and payload["counts"]["episode_count"]==2
    assert payload["metrics"]["normalized_rmse"]["count"]==2 and payload["counts"]["realized_draw_count"]==0
    assert payload["provenance"]["predictor_kind"]=="linear_dynamics_direct_mean"
    assert payload["provenance"]["predictor_checkpoint"]["trainable_parameter_count"]==564300
    for result in captures:
        assert result["predictive_law_draw_count"]==0 and result["posterior_evaluation_draw_count"]==2
        assert result["local_membrane_status"]["electrical_evolution_status"]=="unbound"
        assert result["local_membrane"]["snapshot"]["current_time_ms"] in ([.05],[.1])
        assert all(item["status"]=="not_applicable" for item in result["router_diagnostics"])
    for name in ("crps","predictive_coverage","predictive_calibration_error"):
        assert payload["metrics"][name]=={"mean":None,"count":0}
    first=[result["field_trajectory"].clone() for result in captures]
    loader=native.load_records
    def changed_targets(*args,**kwargs):
        arrays,bindings=loader(*args,**kwargs)
        if kwargs.get("include_teacher",True):
            arrays=copy.deepcopy(arrays);arrays["trajectory"][:,1:]+=17
        return arrays,bindings
    monkeypatch.setattr(native,"load_records",changed_targets)
    captures.clear()
    changed,_=evaluation.evaluate_native_rollout_prediction(config,checkpoint)
    assert all(torch.equal(before,result["field_trajectory"]) for before,result in zip(first,captures))
    assert changed["metrics"]["normalized_rmse"]["mean"]!=payload["metrics"]["normalized_rmse"]["mean"]
    result,output=run_cli(config,checkpoint,tmp_path)
    assert result.returncode==2,result.stdout+result.stderr
    cli=json.loads((output/"metrics.json").read_text())
    assert cli["counts"]["episode_count"]==2 and cli["metrics"]["normalized_rmse"]["count"]==2
    assert cli["provenance"]["residual_likelihood_status"]=="unbound"
    assert json.loads((output/"artifact_manifest.json").read_text())["postrun_status"]=="incomplete"


def test_linear_real_point_optimizer_and_cli(ten_expert_case,tmp_path):
    config,_,*_=setup(ten_expert_case,tmp_path,point=True)
    training_config=point_training_config(ten_expert_case,tmp_path,config)
    config,training_config=linear_config(config),linear_config(training_config)
    _,checkpoint,_,_=train_linear(training_config,tmp_path/"trained")
    result,output=run_cli(config,checkpoint,tmp_path)
    assert result.returncode==2,result.stdout+result.stderr
    payload=json.loads((output/"metrics.json").read_text())
    assert payload["counts"]["episode_count"]==1 and payload["metrics"]["normalized_rmse"]["count"]==1
    provenance=payload["episodes"][0]["prediction_provenance"]
    assert provenance["posterior_evaluation_draw_count"]==provenance["predictive_law_draw_count"]==0
    assert provenance["particle_draw_indices"] is None


def test_linear_weighted_loss_and_reordered_whole_particles(linear_joint_case):
    _,config,_,model,_,_,_=linear_joint_case
    target=torch.randn(2,3,6,6,15)
    prediction=torch.randn(1,2,3,6,6,15,requires_grad=True)
    batch={"trajectory":target,"uncertainty":torch.tensor([[.5]*15,[2.]*15]),
           "teacher_confidence":torch.tensor([.25,.75]),"teacher_loss_weight":torch.tensor([.5,1.])}
    output={"normalized_trajectory":prediction,"router_diagnostics":{"status":"not_applicable","router_regularization_loss":torch.tensor(0.)}}
    loss,_=training.direct_mean_objective(output,batch,model.field_normalization,config)
    per=(prediction[0,:,1:]-model.field_normalization.normalize(target[:,1:])).square().flatten(1).mean(1)
    expected=(per*torch.tensor([2.,.5])*torch.tensor([.125,.75])).sum()/.875
    torch.testing.assert_close(loss,expected)
    batch["teacher_confidence"].zero_()
    zero,_=training.direct_mean_objective(output,batch,model.field_normalization,config)
    zero.backward()
    assert zero==0 and not torch.count_nonzero(prediction.grad)
    registry=native.load_registry(config)
    def predict(actions):
        records=native.select_records(registry,config,"development",["sample-a"],actions)
        arrays,_=native.load_records(registry,records,config,include_teacher=False)
        _,prepared=evaluation.prepare_native_episode(model.conditioning["model"],records,config,arrays)
        out=predict_direct_mean(model,prepared,model.conditioning["model"].field_rollout_operator.quantized_mddm_rollout_operator)
        assert out["posterior_uid"]==prepared["joint_posterior"].posterior_uid
        assert torch.equal(out["particle_weights"],prepared["joint_posterior"].weights)
        return out
    first,reverse=predict(["regimen-a","regimen-b"]),predict(["regimen-b","regimen-a"])
    assert torch.equal(first["particle_draw_indices"],reverse["particle_draw_indices"])
    assert torch.equal(first["particle_conditional_means"].flip(2),reverse["particle_conditional_means"])
    assert torch.equal(first["field_trajectory"].flip(1),reverse["field_trajectory"])


@pytest.mark.parametrize("mutation",["equation","model","profile","grid","normalization","conditioner","seed","context"])
def test_linear_checkpoint_mismatch_before_targets(linear_joint_case,monkeypatch,tmp_path,mutation):
    _,config,_,model,checkpoint,_,_=linear_joint_case
    payload,selected=copy.deepcopy(torch.load(checkpoint,map_location="cpu")),copy.deepcopy(config)
    identity=payload["native_direct_mean"]["model_identity"]
    if mutation=="equation":identity["backbone"]["native_direct_mean"]["linear_dynamics"]["program_hold"]="right_endpoint"
    elif mutation=="model":identity["backbone"]["kind"]="transolver_microenvironment_operator"
    elif mutation=="profile":selected["selected_experiment_family"]="rollout_operator_transolver"
    elif mutation=="grid":identity["linear_domain"]["species_id"].reverse()
    elif mutation=="normalization":identity["normalization_id"]="wrong"
    elif mutation=="conditioner":payload["native_direct_mean"]["conditioning_identity"]["checkpoint_sha256"]="wrong"
    elif mutation=="seed":payload["training_seed_binding"]["model_seed"]+=1
    else:identity["source_context_width"]+=1
    invalid=tmp_path/"invalid.pt"
    torch.save(payload,invalid)
    monkeypatch.setattr(native,"load_records",lambda *a,**k:pytest.fail("invalid checkpoint reached target arrays"))
    with pytest.raises(ValueError):evaluation.evaluate_native_rollout_prediction(selected,invalid)


def test_linear_residual_and_unsupported_entrypoints_reject_before_consumption(linear_joint_case,monkeypatch,tmp_path):
    from Libs.utils.evaluation.native_rollout_metrics import metric_policy
    from Libs.utils.evaluation.native_transolver_calibration import run_calibration,load_residual_artifact,predict_fitted_residual
    _,config,_,_,checkpoint,_,_=linear_joint_case
    monkeypatch.setattr(native,"load_registry",lambda *a,**k:pytest.fail("misuse reached a source registry"))
    for operation in ("fit","check"):
        result,_=run_calibration(config,checkpoint,operation,tmp_path/"absent.json")
        assert result["status"]==("FIT_FAILED" if operation=="fit" else "CHECK_FAILED")
        assert "Transolver" in result["reason"]
    with pytest.raises(ValueError,match="Transolver"):load_residual_artifact(tmp_path/"absent.json",config,{})
    with pytest.raises(ValueError,match="Transolver"):predict_fitted_residual(
        {"predictor_kind":"linear_dynamics_direct_mean"},{},None,None,None,{}, {},episode_uid="unused")
    wrong=copy.deepcopy(config)
    wrong["paper_grade_full_suite"]["native_rollout_evaluation"]["predictor_kind"]="transolver_fitted_residual"
    with pytest.raises(ValueError,match="selector"):metric_policy(wrong)
    with pytest.raises(ValueError,match="screening/serving"):screening_model_kwargs_from_config(config)
    wrong=copy.deepcopy(config)
    wrong["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["native_direct_mean"]["router_regularization_weight"]=1.
    with pytest.raises(ValueError,match="router"):native_direct_mean_config(wrong)
    wrong=copy.deepcopy(config)
    wrong["paper_grade_full_suite"]["native_rollout_evaluation"]["source_role"]="calibration"
    with pytest.raises(ValueError,match="development"):evaluation.evaluate_native_rollout_prediction(wrong,checkpoint)


def test_linear_canonical_profile_schema_and_shared_five_seeds():
    config=load_config("Libs/configs/config_stack.yaml")
    profile=load_experiment_family("rollout_operator_linear_dynamics",config)
    assert profile["runtime"]["seed_values"]==[40,41,42,43,44]
    assert profile["config_overrides"]["paper_grade_full_suite"]["native_rollout_evaluation"]["predictor_kind"]=="linear_dynamics_direct_mean"
    entry=next(row for row in config["paper_grade_baseline_registry"]["baselines"] if row["baseline_id"]=="native_linear_dynamics")
    assert entry["reference_only"] and not entry["main_table_runnable"] and not entry["appendix_runnable"]
    assert "native_linear_dynamics" not in config["paper_grade_full_suite"]["models"]
    assert config["schemas"]["native_linear_dynamics"]["definition"]==LINEAR_DYNAMICS_SPEC
    template=config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["linear_dynamics_template"]
    assert template["native_direct_mean"]["router_regularization_weight"]==0


def test_linear_identity_requires_bound_r60(monkeypatch):
    from Libs.utils.model import native_direct_mean
    config=load_config("Libs/configs/config_stack.yaml")
    backbone=config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    selected=copy.deepcopy(backbone["linear_dynamics_template"])
    selected["native_direct_mean"].update(posterior_context_dim=12,
        conditioning_config_path="unused-config",conditioning_checkpoint_path="unused-checkpoint")
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]=selected
    config["selected_experiment_family"]="rollout_operator_linear_dynamics"
    monkeypatch.setattr(native_direct_mean,"normalization_from_config",lambda config:{"kind":"physical"})
    with pytest.raises(ValueError,match="bound fixed R60"):
        native_direct_mean.direct_model_identity(config)


def test_linear_materialized_profile_pretraining_main_and_checkpoint_consumer(ten_expert_case,tmp_path):
    config,_,*_=setup(ten_expert_case,tmp_path,point=True)
    train_config=linear_config(point_training_config(ten_expert_case,tmp_path,config))
    config=linear_config(config)
    artifact=tmp_path/"pretraining-main"
    train_config["logging_registry"]["console_enabled"]=True
    benchmark=train_config["screening_pipeline"]["screening_benchmark"]
    benchmark["artifact_root"]=str(artifact)
    train=benchmark["training"]
    train.update(batch_size=2,learning_rate=.0001,weight_decay=0.)
    train["data_loader"].update(num_workers=0,persistent_workers=False,pin_memory=False)
    train["memory_policy"].update(micro_batch_size=2,effective_batch_size=2,gradient_accumulation_steps=1)
    train["memory_policy"]["mixed_precision"]["enabled"]=False
    train["zarr_loader"].update(pretraining_epoch_count=1,pretraining_batches_per_epoch=1,
        checkpoint_every_epoch=True,formal_artifact_subdir_template="")
    paths=train_config["paper_grade_full_suite"]["output_paths"]
    paths["runtime_efficiency_profile_json"]=str(tmp_path/"runtime.json")
    paths["runtime_efficiency_profile_md"]=str(tmp_path/"runtime.md")
    # Bind the canonical profile's existing template and output override to this
    # disposable fixture. No source or reference-capacity declaration is changed.
    backbone=benchmark["model_architecture"]["rollout_operator"]["transition_backbone"]
    backbone["linear_dynamics_template"]=copy.deepcopy(backbone)
    train_config["paper_experiments"]["rollout_operator_linear_dynamics"]["runtime"]["artifact_root"]=str(artifact)
    raw=tmp_path/"base.yaml"
    raw.write_text(yaml.safe_dump(train_config,sort_keys=False))
    cfg=tmp_path/"materialized.yaml"
    result=subprocess.run([sys.executable,str(ROOT/"Libs/exp/orchestration/prepare_experiment_config.py"),
        "--base-config",str(raw),"--profile","rollout_operator_linear_dynamics","--seed","42",
        "--output-config",str(cfg)],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    materialized=load_config(cfg)
    policy=native_direct_mean_config(materialized)
    assert policy["linear_dynamics"]==LINEAR_DYNAMICS_SPEC
    assert policy["conditioning_checkpoint_path"]==native_direct_mean_config(config)["conditioning_checkpoint_path"]
    assert resolve_training_seed(materialized,require_selected=True)["model_seed"]==42
    assert training.validate_default_parameter_budget(materialized)==564300
    result=subprocess.run([sys.executable,str(ROOT/"Libs/exp/training/pretrain_microenvironment_world_model.py"),
        "--config",str(cfg),"--device","cpu"],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    saved=list(artifact.glob("microenvironment_pretraining*_epoch_1.pt"))
    assert len(saved)==1
    checkpoint=torch.load(saved[0],map_location="cpu")
    assert checkpoint["executed_steps"]==1 and checkpoint["optimizer_state_dict"]["state"]
    manifest=json.loads((artifact/"microenvironment_pretraining_manifest.json").read_text())
    assert manifest["executed_steps"]==1 and manifest["direct_normalized_mse"]>0
    assert manifest["active_parameter_estimate"]==manifest["actual_trainable_parameter_count"]==manifest["actual_total_parameter_count"]==564300
    assert manifest["checkpoint_scale"]=="native_linear_dynamics"
    assert manifest["router_status"]=="not_applicable" and "expert_utilization" not in manifest
    assert manifest["native_direct_mean"]["conditioning_identity"]["loaded_parameter_count"]>0
    assert manifest["data_loader"]["collate_fn"]=="collate_native_sources"
    cli,output=run_cli(config,saved[0],tmp_path)
    assert cli.returncode==2,cli.stdout+cli.stderr
    payload=json.loads((output/"metrics.json").read_text())
    assert payload["status"]=="incomplete" and payload["metrics"]["normalized_rmse"]["count"]==1,payload
