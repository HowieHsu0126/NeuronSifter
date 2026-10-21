from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data._utils.collate import default_collate

from Libs.tests.test_native_diffusion_data import producer_case
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data.native_field_normalization import (
    NativeFieldNormalization, fit_transform, fit_haar, normalization_from_config,
    representation, validate_artifact,
)

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT/"Libs/scripts/builders/build_native_diffusion_training_data.py"


def fixture_config(tmp_path):
    config,registry,arrays,store = producer_case(tmp_path)
    schedule = config["neuronsifter_scale_registry"]["quantized_mddm_main_rollout"]["partial_mask_schedule"]
    schedule.update(training_sample_count=2,training_denoising_step_count=2,
                    inference_sample_count=2,inference_denoising_step_count=2)
    return config,registry,arrays,store


def run_builder(config,tmp_path,*args):
    config_path=tmp_path/"config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return subprocess.run([sys.executable,str(BUILDER),"--config",str(config_path),
        "--max-rows","4",*args],text=True,capture_output=True)


def bind_fixture(config,tmp_path,store_format="zarr_v2"):
    artifact=tmp_path/"normalization.json"
    completed=run_builder(config,tmp_path,"--split","train","--fit-field-normalization","--output",str(artifact))
    assert completed.returncode==0,completed.stdout+completed.stderr
    store=tmp_path/("normalized-"+store_format)
    completed=run_builder(config,tmp_path,"--split","train","--normalization-artifact",str(artifact),
        "--store-format",store_format,"--output",str(store))
    assert completed.returncode==0,completed.stdout+completed.stderr
    config=copy.deepcopy(config)
    descriptor=store/(".zattrs" if store_format=="zarr_v2" else "store_manifest.json")
    config["native_diffusion_data"]["registry_path"]=str(descriptor)
    return config,artifact,store


def test_weighted_mean_support_constant_and_roundtrip(tmp_path):
    config,registry,_,_=fixture_config(tmp_path)
    records=native.select_records(registry,config,"train",["sample-a"],["regimen-a","regimen-b"])
    arrays,_=native.load_records(registry,records,config)
    records[0]["teacher"].update(qc_grade="A",sample_weight=1.)
    arrays["trajectory"][:]=7.
    arrays["initial_state"][:]=7.
    for index,values in enumerate(([10.,12.,14.],[20.,22.,24.])):
        arrays["trajectory"][index,...,0]=np.asarray(values)[:,None,None]
    arrays["initial_state"][...,0]=8.
    # Signed values and a separate exactly constant channel are retained.
    arrays["trajectory"][...,1]=-arrays["trajectory"][...,0]
    arrays["initial_state"][...,1]=-8.
    artifact=fit_transform(registry["grid"],records,arrays,config)
    assert artifact["mean"][0]==pytest.approx(46/3)
    assert artifact["population_sd"][0]==pytest.approx(np.sqrt(224)/3)
    assert artifact["mean"][1]==pytest.approx(-46/3)
    assert artifact["constant_channels"][2] is True and artifact["scale"][2]==1.
    validate_artifact(artifact,registry["grid"],representation(config))
    for index in range(2):
        residual=torch.tensor(arrays["trajectory"][index]-arrays["initial_state"][index],dtype=torch.float64)
        coefficients=fit_haar(residual,artifact["representation"]["wavelet_levels"])
        normalized=coefficients/torch.tensor(artifact["scale"])
        assert normalized.min()>=-.25 and normalized.max()<=.25
    x=torch.tensor(arrays["trajectory"])
    mean=torch.tensor(artifact["mean"],dtype=x.dtype)
    scale=torch.tensor(artifact["scale"],dtype=x.dtype)
    torch.testing.assert_close((x-mean)/scale*scale+mean,x)
    assert np.all(np.asarray(artifact["support_scale"])<=np.asarray(artifact["scale"]))
    path=tmp_path/"constant.json"
    path.write_text(json.dumps(artifact))
    from Libs.utils.data.native_field_normalization import bind_artifact,normalization_spec
    norm=NativeFieldNormalization(normalization_spec(bind_artifact(registry["grid"],path,config),config),15)
    departed=torch.full((1,1,3,6,6,15),7.)
    departed[...,2]=8.
    assert norm.excursions(departed)["constant_channel_excursion_count"][0,0,2]==108


@pytest.mark.parametrize("store_format",["zarr_v2","neuronsifter_chunk_store"])
def test_real_builder_store_and_pretrainer(tmp_path,store_format):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    config,registry,arrays,_=fixture_config(tmp_path)
    config,artifact,store=bind_fixture(config,tmp_path,store_format)
    derived=native.load_registry(config)
    restored,_=native.load_records(derived,native.select_records(derived,config,"train"),config)
    np.testing.assert_array_equal(restored["trajectory"],arrays["trajectory"])
    spec=normalization_from_config(config)
    assert spec["artifact"]==json.loads(artifact.read_text())
    model=pretrain.MicroenvironmentPretrainingModel(config,False)
    dataset=pretrain.MicroenvironmentTeacherFamilyDataset(config,"train")
    batch=default_collate(dataset.__getitems__([0,1]))
    captured=[]
    def capture(module,args,kwargs,output):
        captured.append((kwargs,output))
    handle=model.quantized_mddm.register_forward_hook(capture,with_kwargs=True)
    metrics,report=pretrain.run_pretraining_step(model,batch,config,torch.device("cpu"),
        {"gradient_accumulation_steps":1,"mixed_precision":{"enabled":False,"device_type":"cpu","dtype":"float32"}})
    handle.remove()
    assert np.isfinite(metrics["total_loss"])
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.parameters())
    kwargs,output=captured[0]
    expected=(output["training_reconstruction"]-kwargs["teacher_trajectory"]).square().mean((2,3,4,5))
    expected=(expected*kwargs["teacher_confidence"]).mean()
    torch.testing.assert_close(output["sampler_diagnostics"]["mddm_inverse_reconstruction_error"],expected)
    assert torch.equal(output["trajectory_mean"][:,:,0],kwargs["initial_state"])
    assert torch.equal(output["trajectory_std"][:,:,0],torch.zeros_like(kwargs["initial_state"]))
    diagnostics=output["field_normalization_diagnostics"]
    assert diagnostics["coefficient_preclamp_low_count"].sum()==0
    assert diagnostics["coefficient_preclamp_high_count"].sum()==0
    assert "field_normalization_diagnostics" in report
    assert model.field_normalization.identity==derived["grid"]["normalization"]["sha256"]
    # Checkpoint constants are consumed and cannot replace model-owned values.
    clone=pretrain.MicroenvironmentPretrainingModel(config,False)
    clone.load_state_dict(model.state_dict())
    broken=copy.deepcopy(model.state_dict())
    broken["field_normalization.scale"][0]*=2
    with pytest.raises(ValueError,match="checkpoint numerical transform"):
        clone.load_state_dict(broken,strict=False)
    cached=native.screening_native_payload(config,"train",["sample-a","sample-b"],["regimen-a","regimen-b"])
    assert str(artifact) in cached["source_bindings"]
    before=artifact.read_bytes()
    artifact.write_bytes(before+b" ")
    with pytest.raises(ValueError,match="artifact bytes changed"):
        normalization_from_config(config)
    with pytest.raises(ValueError,match="artifact bytes changed|cache_stale"):
        native.validate_cached_native({"fixture":{"train":{"sample_uids":cached["sample_uids"],"native_diffusion":cached}}},
                                     cached["regimen_uids"],config)


@pytest.mark.parametrize("role",["development","validation","calibration"])
def test_fit_role_rejected_before_array_reads(tmp_path,monkeypatch,role):
    from Libs.scripts.builders import build_native_diffusion_training_data as builder
    config,_,_,_=fixture_config(tmp_path)
    path=tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(builder,"load_records",lambda *args:pytest.fail("nontraining payload read"))
    monkeypatch.setattr(sys,"argv",["builder","--config",str(path),"--fit-field-normalization",
        "--split",role,"--output",str(tmp_path/"forbidden.json")])
    with pytest.raises(SystemExit) as error:
        builder.main()
    assert error.value.code==2 and not (tmp_path/"forbidden.json").exists()


def test_poison_nontraining_row_is_not_read(tmp_path,monkeypatch):
    config,registry,_,_=fixture_config(tmp_path)
    poison=copy.deepcopy(registry["teachers"][-1])
    registry["teachers"][-1].update(split_name="development",source_holdout_role="development",
        trajectory_store_path=str(tmp_path/"POISON_NO_READ"))
    from Libs.utils.model import dynamics_store
    loader=dynamics_store.load_zarr_rows
    seen=[]
    def read(path,name,indices):
        assert "POISON" not in str(path)
        seen.extend(indices.tolist())
        return loader(path,name,indices)
    monkeypatch.setattr(dynamics_store,"load_zarr_rows",read)
    records=native.select_records(registry,config,"train")
    arrays,_=native.load_records(registry,records,config)
    artifact=fit_transform(registry["grid"],records,arrays,config)
    assert poison["trajectory_uid"] not in [row["trajectory_uid"] for row in artifact["training_records"]]
    assert 3 not in seen


@pytest.mark.parametrize("failure",["nonfinite","zero_weight","units","time","duplicate","old_scaler","bad_codebook"])
def test_invalid_fit_input_rejects(tmp_path,failure):
    config,registry,_,_=fixture_config(tmp_path)
    records=native.select_records(registry,config,"train")
    arrays,_=native.load_records(registry,records,config)
    if failure=="nonfinite": arrays["trajectory"][0,0,0,0,0]=float("nan")
    if failure=="zero_weight": records[0]["teacher"]["sample_weight"]=0.
    if failure=="units": registry["grid"]["species_units"]={}
    if failure=="time": registry["grid"]["time_hours"]=[0.,2.,1.]
    if failure=="duplicate": records[1]["teacher"]["content_hash"]=records[0]["teacher"]["content_hash"]
    if failure=="old_scaler": registry["grid"]["normalization"]={"kind":"training_scaler","fit_role":"train","artifact_ref":"absent","sha256":"tag"}
    if failure=="bad_codebook": config["neuronsifter_scale_registry"]["quantized_mddm_main_rollout"]["residual_tokenization"]["residual_min_value"]=1.
    with pytest.raises(ValueError):
        fit_transform(registry["grid"],records,arrays,config)


@pytest.mark.parametrize("length",[3,4,5,6])
def test_actual_haar_order_and_unit_rescaling(tmp_path,length):
    config,registry,_,_=fixture_config(tmp_path)
    records=native.select_records(registry,config,"train")
    arrays,_=native.load_records(registry,records,config)
    registry["grid"]["time_hours"]=list(map(float,range(length)))
    arrays["trajectory"]=np.stack([np.stack([arrays["trajectory"][i,0]+t*.2 for t in range(length)]) for i in range(4)])
    artifact=fit_transform(registry["grid"],records,arrays,config)
    scaled={key:value.copy() for key,value in arrays.items()}
    scaled["trajectory"]*=1000.
    scaled["initial_state"]*=1000.
    scaled_grid=copy.deepcopy(registry["grid"])
    scaled_config=copy.deepcopy(config)
    scaled_grid["species_units"]={key:"fixture_milliunit" for key in scaled_grid["species_id"]}
    scaled_config["native_diffusion_data"]["species_units"]=scaled_grid["species_units"]
    changed=fit_transform(scaled_grid,records,scaled,scaled_config)
    np.testing.assert_allclose(changed["mean"],np.asarray(artifact["mean"])*1000,rtol=1e-6)
    np.testing.assert_allclose(changed["scale"],np.asarray(artifact["scale"])*1000,rtol=1e-6)
    from Libs.models.modules.neuronsifter_world_action import GraphSpectralTemporalWaveletTokenizer
    tokenizer=GraphSpectralTemporalWaveletTokenizer(1,1,artifact["representation"]["wavelet_levels"],1.,False)
    from Libs.utils.data.native_field_normalization import pack_field
    raw=torch.tensor(arrays["trajectory"][0]-arrays["initial_state"][0],dtype=torch.float64)
    exact,layout=pack_field(raw,tokenizer)
    torch.testing.assert_close(exact,fit_haar(raw,tokenizer.wavelet_level_count))
    assert layout["lengths"][0]==length
    assert np.asarray(changed["constant_channels"]).tolist()==artifact["constant_channels"]


def test_preclamp_excursion_errors_use_untruncated_source(tmp_path):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    config,_,_,_=fixture_config(tmp_path)
    config,_,_=bind_fixture(config,tmp_path)
    model=pretrain.MicroenvironmentPretrainingModel(config,False)
    dataset=pretrain.MicroenvironmentTeacherFamilyDataset(config,"train")
    batch=default_collate(dataset.__getitems__([0]))
    raw=batch["trajectory"].clone()+100.
    context={key:batch[key].unsqueeze(1) for key in ("occupancy","support","physical_time","posterior_context")}
    initial=batch["initial_state"].unsqueeze(1)
    output=model.quantized_mddm(torch.zeros_like(raw).unsqueeze(1),True,initial_state=initial,
        teacher_trajectory=raw.unsqueeze(1),teacher_confidence=torch.ones(1,1),conditioning=context,
        native_operator=model.operator,adjacency=model.node_adjacency,
        coordinates=torch.zeros(6,6,int(config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["node_coordinate_dim"])),field_normalization=model.field_normalization)
    diagnostics=output["field_normalization_diagnostics"]
    coefficients,_=model.quantized_mddm.pack(raw[0].double()-initial[0,0].double(),model.operator)
    coefficients/=model.field_normalization.scale
    assert torch.equal(diagnostics["coefficient_preclamp_high_count"][0,0],(coefficients>.25).sum((0,1,2)))
    torch.testing.assert_close(diagnostics["saturation_squared_error_sum"][0,0],
        (coefficients-coefficients.clamp(-.25,.25)).square().sum((0,1,2)))
    expected=(output["training_reconstruction"]-raw.unsqueeze(1)).square().mean()
    torch.testing.assert_close(output["sampler_diagnostics"]["mddm_inverse_reconstruction_error"],expected)
    assert diagnostics["source_range_excursion_count"].sum()>0
    assert torch.equal(raw,batch["trajectory"]+100.)
    inference=model.quantized_mddm(torch.zeros_like(raw).unsqueeze(1),False,initial_state=initial,
        conditioning=context,native_operator=model.operator,adjacency=model.node_adjacency,
        coordinates=torch.zeros(6,6,int(config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["node_coordinate_dim"])),field_normalization=model.field_normalization,return_samples=True)
    assert inference["field_normalization_diagnostics"]["teacher_reference_available"].sum()==0
    assert "saturation_squared_error_sum" not in inference["field_normalization_diagnostics"]
    assert "mddm_inverse_reconstruction_error" not in inference["sampler_diagnostics"]
    torch.testing.assert_close(inference["trajectory_std"],inference["sample_trajectories"].std(0,unbiased=False))
    torch.testing.assert_close(inference["trajectory_quantiles"],
        torch.quantile(inference["sample_trajectories"],torch.tensor(model.quantized_mddm.quantile_levels),dim=0))


@pytest.fixture(scope="module")
def joint_normalized_case(tmp_path_factory):
    from Libs.tests.test_joint_world_posterior import joint_registry_case
    from Libs.tests.test_physical_occupancy import make_packet
    from Libs.scripts.teacher_generation.build_osp_pbpk_teacher_rollouts import build_native_physical_source
    from Libs.utils.model.microenvironment_teacher import write_native_microenvironment_store
    directory=tmp_path_factory.mktemp("normalized_joint")
    _,config,registry,arrays,source=joint_registry_case(directory)
    architecture=config["screening_pipeline"]["screening_benchmark"]["model_architecture"]
    architecture["rollout_operator"]["transition_backbone"]["mechanism_moe"]["router_loss_weight"]=0.01
    architecture["multimodal_moe"]["enabled"]=False
    # The serving constructor needs a source-declared structural-prior catalog.
    # Keep this controlled fixture wholly node-local instead of resolving Input/.
    import pandas as pd
    prior=directory/"fixture-prior"
    prior.mkdir()
    config["structural_prior_registry"]["local_root"]=str(prior)
    config["structural_prior_registry"]["relative_paths"]["region_prior_catalog"]="regions.tsv"
    features=config["structural_prior_registry"]["region_feature_names"]
    pd.DataFrame([{"disease_program":"ad_control","region_id":region["region_id"],
        "structural_prior_uid":"fixture-prior-"+region["region_id"],
        **{name:float(index+1)/6 for name in features}}
        for index,region in enumerate(config["spatial_atlas_registry"]["regions"])]).to_csv(
            prior/"regions.tsv",sep="\t",index=False)
    config["neuronsifter_scale_registry"]["quantized_mddm_main_rollout"]["partial_mask_schedule"].update(
        training_sample_count=2,training_denoising_step_count=2,inference_sample_count=2,inference_denoising_step_count=2)
    packet=make_packet(directory/"exports",config,source)
    source=build_native_physical_source(config,packet,directory/"exports")
    registry["joint_sources"]=[source]
    raw_store=directory/"physical-raw"
    for index,teacher in enumerate(registry["teachers"]):
        record=source["particle_programs"]["records"][index//2][0][index%2]
        teacher.update({key:record[key] for key in ("program_uid","posterior_uid","action_uid","target_program_uid")})
        teacher["trajectory_store_path"]=str(raw_store)
    write_native_microenvironment_store(raw_store,config,registry,arrays)
    Path(config["native_diffusion_data"]["registry_path"]).write_text(json.dumps(registry))
    original=json.dumps(source,sort_keys=True)
    normalized,artifact,store=bind_fixture(config,directory)
    validation_config=directory/"validator-config.yaml"
    validation_config.write_text(yaml.safe_dump(normalized))
    completed=subprocess.run([sys.executable,
        str(ROOT/"Libs/scripts/validators/validate_full_microenvironment_teacher_registry.py"),
        "--config",str(validation_config)],text=True,capture_output=True)
    assert completed.returncode==0,completed.stdout+completed.stderr
    assert "4 records; no array payload read" in completed.stdout
    return directory,normalized,artifact,source,original,arrays


def screening_kwargs(config):
    from Libs.utils.model.screening_model_config import screening_model_kwargs_from_config
    from Libs.utils.model.dynamics_fields import node_adjacency_matrix
    from Libs.exp.training.train_screening_system import rollout_model_kwargs
    kwargs=screening_model_kwargs_from_config(config)
    kwargs.update(rollout_model_kwargs(config,rollout_step_count_override=3))
    groups=kwargs["rollout_update_policy"]["action_representation"]["feature_groups"]
    feature_width=max(index for indices in groups.values() for index in indices)+1
    kwargs.update(patient_state_dim=4,target_feature_dim=3,regimen_feature_dim=feature_width,hidden_dim=12,latent_dim=8,
        dropout_rate=0.,compartment_count=6,rollout_step_count=3,adjacency_matrix=node_adjacency_matrix(config).tolist(),
        tri_weight_by_component={key:1. for key in ("calibration","coverage","pkpd_physics_consistency","ranking_stability","ood_score")},
        field_normalization_config=normalization_from_config(config))
    return kwargs


def test_joint_raw_physical_programs_cache_projection_and_pretraining(joint_normalized_case,monkeypatch):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    from Libs.exp.training import train_screening_system as screening
    directory,config,artifact,source,original,_=joint_normalized_case
    model=pretrain.MicroenvironmentPretrainingModel(config,False)
    projection=[]
    handle=model.world_state_posterior.joint_field_projection.register_forward_pre_hook(
        lambda module,args:projection.append(args[0].detach().clone()))
    posterior=model.infer_joint_source(source)
    handle.remove()
    assert posterior.support.to_payload()==source["support"]
    assert json.dumps(source,sort_keys=True)==original
    torch.testing.assert_close(projection[0],model.field_normalization.normalize(posterior.support.initial_field))
    assert native.bind_joint_programs(posterior,source)["records"]==source["particle_programs"]["records"]
    dataset=pretrain.MicroenvironmentTeacherFamilyDataset(config,"train")
    metrics,report=pretrain.run_pretraining_step(model,pretrain.collate_native_sources(dataset.__getitems__([3,0])),
        config,torch.device("cpu"),{"gradient_accumulation_steps":1,
            "mixed_precision":{"enabled":False,"device_type":"cpu","dtype":"float32"}})
    assert np.isfinite(metrics["total_loss"]) and report["belief_evidence_status"]["calibration"]=="unverified"
    assert len(report["source_rows"])==2
    for row in report["source_rows"]:
        assert len(row["field_normalization_diagnostics"])==2
        for diagnostic in row["field_normalization_diagnostics"]:
            np.testing.assert_array_equal(diagnostic["source_reference_denominator"],np.full((1,1,15),108))
            np.testing.assert_array_equal(diagnostic["teacher_reference_available"],[[1.]])
    cached=native.screening_native_payload(config,"train",["sample-b","sample-a"],["regimen-b","regimen-a"])
    payload={"selected_target_rows":[{"target_program_uid":"target"}],
        "selected_regimen_rows":[{"regimen_uid":uid} for uid in cached["regimen_uids"]],
        "modality_payload":{"fixture":{"train":{"sample_uids":cached["sample_uids"],
            "state_features":np.zeros((2,4),dtype=np.float32),"native_diffusion":cached}}},
        "label_payload":{"fixture":{"train":{"target_labels":np.ones((2,1),dtype=np.float32),
            "regimen_labels":np.ones((2,2),dtype=np.float32)}}}}
    monkeypatch.setattr(screening,"source_artifact_hashes",lambda *args:{})
    monkeypatch.setattr(screening,"freeze_signature_sha256",lambda *args:"controlled-fixture")
    config_path=directory/"cache-config.yaml"
    cache_root=directory/"fold-cache"
    screening.write_scope_training_cache(config,config_path,["controlled-fixture"],cache_root,"fixture","train",11,payload,0.)
    restored,_,_=screening.load_scope_training_cache(config,config_path,cache_root,"fixture","train",11)
    actual=restored["modality_payload"]["fixture"]["train"]["native_diffusion"]
    assert actual["grid"]["normalization"]==cached["grid"]["normalization"]
    assert actual["joint_source"]==cached["joint_source"]
    assert str(artifact) in actual["source_bindings"]


def screening_fixture_bundle(model,config,regimens):
    import pandas as pd
    grid=native.load_registry(config)["grid"]
    bundle={"screening_model":model,"regimen_index_by_uid":{"regimen-a":0,"regimen-b":1},
        "regimen_feature_matrix":regimens,
        "regimen_feature_frame":pd.DataFrame([{"regimen_uid":uid,"target_symbol_set":"target",
            "route":"intravenous","schedule":"fixture","dose_level_id":"fixture"} for uid in ("regimen-a","regimen-b")]),
        "region_catalog":pd.DataFrame({"region_id":grid["region_id"],"region_label":grid["region_id"]}),
        "species_catalog":pd.DataFrame({"species_id":grid["species_id"]}),
        "preprocessing_manifest":{"active_disease_program":"ad_control"},
        "region_prior_catalog":pd.DataFrame({"region_id":grid["region_id"],"disease_program":["ad_control"]*6,
            "program_support_score":[0.]*6,"disease_vulnerability_score":[0.]*6,"transport_centrality_score":[0.]*6})}
    return bundle


def test_joint_screening_serving_checkpoint_and_source_uq(joint_normalized_case,tmp_path):
    from Libs.models.networks.screening_system import ADScreeningSystem
    from Libs.scripts.services import serve_screening_api as serving
    from Libs.exp.training.train_screening_system import native_diffusion_training_kwargs
    _,config,_,source,original,_=joint_normalized_case
    kwargs=screening_kwargs(config)
    model=ADScreeningSystem(**kwargs).eval()
    selected=native.select_joint_source(source,["sample-a"],["regimen-a","regimen-b"])
    regimens=torch.rand(2,kwargs["regimen_feature_dim"])
    cached=native.screening_native_payload(config,"train",["sample-a"],["regimen-a","regimen-b"])
    training=native_diffusion_training_kwargs(model,{"labels":[0],"native_diffusion":cached},
        slice(None),[0,1],torch.device("cpu"),split_name="train")
    output=model.score_regimens(torch.zeros(1,4),regimens,**training)
    torch.testing.assert_close(output["field_trajectory_std"].square(),
        output["within_diffusion_variance"]+output["between_particle_variance"],atol=1e-7,rtol=1e-5)
    samples=output["particle_trajectory_samples"]
    means=samples.mean(0)
    torch.testing.assert_close(means,output["particle_field_trajectory"])
    assert output["field_normalization_diagnostics"]["coefficient_preclamp_high_count"].shape[:3]==(1,2,2)
    source_initial=torch.tensor(selected["support"]["initial_field"])
    assert torch.equal(output["particle_field_trajectory"][:,:,:,0],source_initial[:,:,None].expand(-1,-1,2,-1,-1,-1))
    from Libs.utils.data.joint_observation import retained_encoder_inputs
    raw_support,observations=native.parse_joint_source(selected)
    values,masks=retained_encoder_inputs(observations,4)
    encoded=model.sample_evidence_encoder.encode_retained(values,masks)
    raw_posterior=model.world_state_posterior.infer_joint(encoded,values,raw_support,observations,
        field_normalization=model.field_rollout_operator.field_normalization)
    torch.testing.assert_close(raw_posterior.weights,training["joint_posterior"].weights)
    assert raw_posterior.support.to_payload()==training["joint_posterior"].support.to_payload()
    bad=copy.deepcopy(cached)
    bad["grid"]["normalization"]["sha256"]="wrong"
    with pytest.raises(ValueError,match="batch/cache/model"):
        native_diffusion_training_kwargs(model,{"labels":[0],"native_diffusion":bad},
            slice(None),[0,1],torch.device("cpu"),split_name="train")
    checkpoint=tmp_path/"screening.pt"
    torch.save({"model_kwargs":kwargs,"model_state":model.state_dict()},checkpoint)
    restored=serving.load_screening_model(checkpoint,config)
    assert restored.field_rollout_operator.field_normalization.identity==model.field_rollout_operator.field_normalization.identity
    bad_state=copy.deepcopy(model.state_dict())
    bad_state["field_rollout_operator.field_normalization.mean"][0]+=1.
    torch.save({"model_kwargs":kwargs,"model_state":bad_state},tmp_path/"wrong.pt")
    with pytest.raises(ValueError,match="checkpoint numerical transform"):
        serving.load_screening_model(tmp_path/"wrong.pt",config)
    for buffer in ("codebook_values","codebook_boundaries"):
        tampered=copy.deepcopy(model.state_dict())
        key="field_rollout_operator.quantized_mddm_rollout_operator."+buffer
        tampered[key].zero_()
        checkpoint_path=tmp_path/(buffer+".pt")
        torch.save({"model_kwargs":kwargs,"model_state":tampered},checkpoint_path)
        with pytest.raises(ValueError,match="Checkpoint quantizer "+buffer):
            serving.load_screening_model(checkpoint_path,config)
    bundle=screening_fixture_bundle(restored,config,regimens)
    payload={"joint_source":selected,"regimen_uids":["regimen-a","regimen-b"]}
    json.dumps(serving.infer_world_state(bundle,payload),allow_nan=False)
    native_kwargs=serving.native_diffusion_serving_kwargs(bundle,payload,payload["regimen_uids"])
    sequence=restored.rollout(torch.zeros(1,4),regimens.unsqueeze(0),
        joint_posterior=native_kwargs["joint_posterior"],particle_programs=native_kwargs["particle_programs"],
        regimen_uids=payload["regimen_uids"])
    assert torch.equal(sequence["particle_field_trajectory_sequence"][:,:,1,0],
                       sequence["particle_field_trajectory_sequence"][:,:,0,-1])
    result=serving.rollout_regimens(bundle,payload)
    json.dumps(result,allow_nan=False)
    assert "field_normalization_diagnostics" in result
    assert "saturation_squared_error_sum" not in result["field_normalization_diagnostics"]
    assert json.dumps(source,sort_keys=True)==original


def test_source_draw_mean_quantile_std_under_unit_conversion(tmp_path):
    import hashlib
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    config,_,_,_=fixture_config(tmp_path)
    config,_,_=bind_fixture(config,tmp_path)
    model=pretrain.MicroenvironmentPretrainingModel(config,False)
    dataset=pretrain.MicroenvironmentTeacherFamilyDataset(config,"train")
    batch=default_collate(dataset.__getitems__([0]))
    original=model.field_normalization
    converted=copy.deepcopy(original.artifact)
    multiplier=1000.
    for key in ("mean","population_sd","scale","source_min","source_max","weighted_sum",
                "haar_positive_max","haar_negative_max","support_scale"):
        converted[key]=(np.asarray(converted[key])*multiplier).tolist()
    converted["weighted_m2"]=(np.asarray(converted["weighted_m2"])*multiplier**2).tolist()
    converted["domain"]["species_units"]={key:"fixture_milliunit" for key in converted["domain"]["species_id"]}
    path=tmp_path/"unit-converted.json"
    path.write_text(json.dumps(converted))
    other=NativeFieldNormalization({"kind":"training_scaler","artifact_ref":str(path),
        "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"artifact":converted},15)
    context={key:batch[key].unsqueeze(1) for key in ("occupancy","support","physical_time","posterior_context")}
    initial=batch["initial_state"].unsqueeze(1).double()
    def sample(normalization,field):
        return model.quantized_mddm(torch.zeros(1,1,3,6,6,15,dtype=torch.float64),False,
            initial_state=field,conditioning=context,native_operator=model.operator,adjacency=model.node_adjacency,
            coordinates=torch.zeros(6,6,int(config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["node_coordinate_dim"])),field_normalization=normalization,return_samples=True)
    base=sample(original,initial)
    changed=sample(other,initial*multiplier)
    for key in ("sample_trajectories","trajectory_mean","trajectory_quantiles","trajectory_std"):
        torch.testing.assert_close(changed[key],base[key]*multiplier,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(changed["trajectory_std"].square(),base["trajectory_std"].square()*multiplier**2,
                               rtol=1e-10,atol=1e-10)


@pytest.fixture(scope="module")
def normalized_runtime_case(tmp_path_factory):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    directory=tmp_path_factory.mktemp("normalization_runtime")
    config,_,_,_=fixture_config(directory)
    config,_,_=bind_fixture(config,directory)
    model=pretrain.MicroenvironmentPretrainingModel(config,False).eval()
    dataset=pretrain.MicroenvironmentTeacherFamilyDataset(config,"train")
    batch=default_collate(dataset.__getitems__([0,1,2,3]))
    return config,model,batch


def unexpected_denoising(*args,**kwargs):
    raise AssertionError("An invalid runtime binding reached denoising")


@pytest.mark.parametrize("mutation",["dilated","second_action","one_grid_cell","time_count","region_count","compartment_count"])
def test_active_runtime_time_grid_checks_every_value_and_cardinality(normalized_runtime_case,monkeypatch,mutation):
    _,model,batch=normalized_runtime_case
    carrier=torch.zeros_like(batch["trajectory"]).reshape(2,2,3,6,6,15)
    initial=batch["initial_state"].reshape(2,2,6,6,15)
    context={key:batch[key].reshape(2,2,*batch[key].shape[1:]).clone()
             for key in ("occupancy","support","physical_time","posterior_context")}
    if mutation=="dilated":
        context["physical_time"]*=2
    elif mutation=="second_action":
        context["physical_time"][1,1]*=2
    elif mutation=="one_grid_cell":
        context["physical_time"][1,1,-1,-1,-1,0]+=1
    else:
        axis={"time_count":2,"region_count":3,"compartment_count":4}[mutation]
        selection=[slice(None)]*6
        selection[axis]=slice(None,-1)
        carrier=carrier[tuple(selection)]
        for key in ("occupancy","support","physical_time"):
            context[key]=context[key][tuple(selection)]
        if axis!=2:
            initial_selection=[slice(None)]*5
            initial_selection[axis-1]=slice(None,-1)
            initial=initial[tuple(initial_selection)]
            context["posterior_context"]=context["posterior_context"][tuple(initial_selection)]
    monkeypatch.setattr(model.quantized_mddm,"logits",unexpected_denoising)
    with pytest.raises(ValueError,match="conditioning physical_time.*fitted source"):
        model.quantized_mddm(carrier,False,initial_state=initial,conditioning=context,
            native_operator=model.operator,adjacency=model.node_adjacency,coordinates=torch.zeros(6,6,2),
            field_normalization=model.field_normalization)


def test_actual_pretraining_rejects_changed_bound_time(normalized_runtime_case,monkeypatch):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    config,model,batch=normalized_runtime_case
    altered=dict(batch,physical_time=batch["physical_time"]*2)
    monkeypatch.setattr(model.quantized_mddm,"logits",unexpected_denoising)
    with pytest.raises(ValueError,match="conditioning physical_time values differ from fitted source hours"):
        pretrain.run_pretraining_step(model,altered,config,torch.device("cpu"),
            {"gradient_accumulation_steps":1,"mixed_precision":{"enabled":False,"device_type":"cpu","dtype":"float32"}})
    torch.testing.assert_close(batch["physical_time"][:, :, 0, 0, 0],
                               torch.tensor([[0.,1.,2.]]).expand(4,-1))


def test_actual_serving_rejects_valid_program_on_another_time_grid(joint_normalized_case,tmp_path,monkeypatch):
    from Libs.models.networks.screening_system import ADScreeningSystem
    from Libs.scripts.services import serve_screening_api as serving
    from Libs.scripts.teacher_generation.build_osp_pbpk_teacher_rollouts import build_native_physical_source
    from Libs.tests.test_physical_occupancy import make_packet
    _,config,_,source,original,_=joint_normalized_case
    selected=native.select_joint_source(source,["sample-a"],["regimen-a","regimen-b"])
    directory=tmp_path/"alternate-time-exports"
    packet=make_packet(directory,config,selected)
    packet["time_grid_hours"]=[0.,.5,1.]
    for export in packet["exports"]:
        export["binding"]["interpolation"]="linear_inside_coverage"
    alternate=build_native_physical_source(config,packet,directory)
    native.parse_joint_source(alternate)
    assert alternate["support"]==selected["support"]
    assert alternate["observations"]==selected["observations"]
    model=ADScreeningSystem(**screening_kwargs(config)).eval()
    regimens=torch.rand(2,model.regimen_feature_dim)
    bundle=screening_fixture_bundle(model,config,regimens)
    payload={"joint_source":alternate,"regimen_uids":["regimen-a","regimen-b"]}
    # This coherent, in-coverage physical program passes the genuine serving
    # admission front door; only the active fitted time domain rejects it.
    serving.native_diffusion_serving_kwargs(bundle,payload,payload["regimen_uids"])
    monkeypatch.setattr(model.field_rollout_operator.quantized_mddm_rollout_operator,"logits",unexpected_denoising)
    with pytest.raises(ValueError,match="conditioning physical_time values differ from fitted source hours"):
        serving.rollout_regimens(bundle,payload)
    assert json.dumps(source,sort_keys=True)==original


def test_physical_identity_keeps_unbound_monotone_time_grid():
    from Libs.tests.test_native_masked_diffusion import make_native_case
    sampler,carrier,kwargs=make_native_case(batch=2,actions=2)
    kwargs.pop("teacher_trajectory")
    kwargs.pop("teacher_confidence")
    context=dict(kwargs["conditioning"])
    context["physical_time"]=context["physical_time"]*2
    output=sampler(carrier,False,**dict(kwargs,conditioning=context),
                   field_normalization=NativeFieldNormalization(None,7),return_samples=True)
    assert torch.isfinite(output["trajectory_mean"]).all()
    assert torch.equal(output["trajectory_mean"][:,:,0],kwargs["initial_state"])
    assert output["field_normalization_diagnostics"]["field_reference_available"].sum()==0


@pytest.mark.parametrize("buffer",["codebook_values","codebook_boundaries"])
def test_real_checkpoint_and_quantization_reject_buffer_substitution(normalized_runtime_case,buffer):
    _,original,_=normalized_runtime_case
    model=copy.deepcopy(original)
    sampler=model.quantized_mddm
    coefficients=torch.tensor([-.2,-.1,0.,.1,.2])
    expected=torch.tensor([6,19,31,44,57])
    assert torch.equal(sampler.quantize_residual(coefficients)[1],expected)
    valid=copy.deepcopy(model.state_dict())
    corrupt=copy.deepcopy(valid)
    corrupt["quantized_mddm."+buffer].zero_()
    with pytest.raises(ValueError,match="Checkpoint quantizer "+buffer):
        model.load_state_dict(corrupt,strict=False)
    model.load_state_dict(valid,strict=False)
    assert torch.equal(sampler.quantize_residual(coefficients)[1],expected)
    # Valid module dtype conversion keeps the original codebook/midpoint rule.
    sampler.double()
    model.field_normalization.assert_representation(model.operator,sampler)
    assert torch.equal(sampler.quantize_residual(coefficients.double())[1],expected)
    getattr(sampler,buffer).zero_()
    with pytest.raises(ValueError,match="Native quantizer "+buffer):
        model.field_normalization.assert_representation(model.operator,sampler)
    with pytest.raises(ValueError,match="Native quantizer "+buffer):
        sampler.quantize_residual(coefficients.double())
