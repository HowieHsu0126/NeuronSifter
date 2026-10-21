from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from Libs.tests.test_expert_composition import ten_expert_case, control_config
from Libs.tests.test_native_field_normalization import joint_normalized_case, screening_kwargs
from Libs.models.networks.screening_system import ADScreeningSystem
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.model.microenvironment_teacher import write_native_microenvironment_store
from Libs.utils.evaluation.native_rollout_metrics import validate_native_metric_payload
from Libs.exp.evaluation import evaluate_screening_system as evaluation
from Libs.exp.training.train_screening_system import screening_training_seed_binding
from Libs.exp.training.pretrain_microenvironment_world_model import save_portable_training_checkpoint

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ["dynamics_mainline_ad", "rollout_operator_reversed_experts", "rollout_operator_interleaved_experts", "action_conditioning_target_identity"]


def development_case(case, directory, profile=PROFILES[0], point=False):
    directory.mkdir(parents=True, exist_ok=True)
    config = control_config(case[1], profile)
    registry = copy.deepcopy(native.load_registry(config))
    source = registry["joint_sources"][0]
    source["support"]["role"] = "development"
    source["support"]["prior_weights"] = [[.2,.8] for _ in source["support"]["sample_uids"]]
    for observation in source["observations"]:
        observation["role"] = "development"
    store_format = "zarr_v2" if point else "neuronsifter_chunk_store"
    store = directory / "development-source"
    arrays = {key: np.array(value,copy=True) for key,value in case[-1].items()}
    for row in registry["sources"]:
        row["split_evidence_ref"] = "fixture-declared-development-split"
        row["predictive_evaluation_units"] = {"episode_field":"sample_uid", "group_field":"family_uid",
            "group_kind":"action_family", "independent_episode_units":True,
            "evidence_ref":"controlled-fixture-independent-episode-design"}
    for index, teacher in enumerate(registry["teachers"]):
        teacher.update(split_name="development", source_holdout_role="development",
            trajectory_uid="development-" + teacher["trajectory_uid"], family_uid="development-" + teacher["family_uid"],
            trajectory_store_path=str(store), store_format=store_format, trajectory_index=index,
            validity_mask={"kind":"all_valid","evidence_ref":"fixture-complete-target-field"})
    if point:
        config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["family"] = "deterministic_compatibility"
        registry.pop("joint_sources")
        registry["posterior_family"] = "deterministic_compatibility"
        registry["teachers"] = registry["teachers"][:2]
        registry["programs"] = copy.deepcopy(source["particle_programs"]["records"][0][0])
        first = registry["programs"][0]
        registry["observations"] = [{"posterior_uid": first["posterior_uid"], "sample_uid": first["sample_uid"],
            "source_uid": first["source_uid"], "origin":"observation_derived", "observation_ref":"fixture-readout",
            "source_snapshot_ref":source["support"]["source_snapshot_ref"], "initial_time_hours":0,
            "store_path":str(store), "store_format":store_format, "observation_index":0,
            "screening_features":[.2,.4,.6,.8]}]
        arrays = {key:value[:2] for key,value in arrays.items()}
        arrays.update(initial_state=np.asarray(source["support"]["initial_field"])[0:1,0].astype(np.float32),
                      posterior_context=np.zeros((1,6,6,12),dtype=np.float32))
        for key in ("occupancy","support","physical_time"):
            arrays[key] = np.asarray([row["physical_program"][key] for row in registry["programs"]],dtype=np.float32)
        for index, program in enumerate(registry["programs"]):
            program.update(store_path=str(store),store_format=store_format,program_index=index)
    training = config["screening_pipeline"]["screening_benchmark"]["training"]
    training.update(seed_policy="manuscript_native_five",random_seed=41)
    config["selected_formal_seed"] = 41
    policy = config["paper_grade_full_suite"]["native_rollout_evaluation"]
    policy.update(source_role="development",sample_uids=["sample-a"] if point else ["sample-a","sample-b"],
                  regimen_uids=["regimen-a","regimen-b"],checkpoint_scope="folds",checkpoint_split_name="fixture")
    config["logging_registry"].update(file_enabled=False,console_enabled=False,root_dir=str(directory/"logs"))
    config["ai_ready_preprocessing"]["simulation_audit"]["local_root"] = str(directory/"no-current-qc")
    kwargs = screening_kwargs(config)
    # The existing producer saves this complete metadata universe after training;
    # this disposable zero-training checkpoint uses the same authority format.
    rows = [{"regimen_uid": uid, "route": "intravenous", "schedule": "daily",
        "aggregate_target_evidence_score": .4, "aggregate_affinity_score": .3,
        "clue_reversal_proxy_score": .2, "pk_feasibility_score": .5, "dose_multiplier": i+1,
        "regimen_size": 1, "target_uid_set": "target", "compound_uid_set": "agent-"+str(i),
        "regimen_scaffold_set": "fixture-"+str(i), "dose_level_id": "dose_100",
        "selection_reason": "ranked_selection", "note":"single_compound_regimen"} for i, uid in enumerate(["regimen-a", "regimen-b"])]
    pd.DataFrame(rows).to_parquet(directory/"selected_regimens.parquet", index=False)
    write_native_microenvironment_store(store,config,registry,arrays,store_format)
    config["native_diffusion_data"]["registry_path"] = str(store/(".zattrs" if point else "store_manifest.json"))
    binding = screening_training_seed_binding(config,"folds","fixture")
    torch.manual_seed(binding["model_seed"])
    model = ADScreeningSystem(**kwargs).eval()
    assert len(model.field_rollout_operator.neuronsifter_world_action_operator.blocks[0].operator_splitting_expert_bank.expert_names)==10
    checkpoint = save_portable_training_checkpoint(directory/"fixture.pt",
        {"model_kwargs":kwargs,"model_state":model.state_dict(),"training_seed_binding":binding,
         "model_contract_version":config["screening_pipeline"]["screening_benchmark"]["model_contract_version"],
         "fixture_only":True})
    (directory/"training_manifest.json").write_text(json.dumps({
        "scope":"folds", "split_name":"fixture", "training_seed_binding":binding, "selected_regimen_count":2,
        "model_contract_version":config["screening_pipeline"]["screening_benchmark"]["model_contract_version"]}))
    return config,registry,arrays,store,store_format,checkpoint


def run_cli(config, checkpoint, directory, mode="native"):
    path = directory/"config.yaml"
    path.write_text(yaml.safe_dump(config,sort_keys=False))
    output = directory/"evaluation"
    result = subprocess.run([sys.executable,str(ROOT/"Libs/exp/evaluation/evaluate_rollout_prediction.py"),
        "--execution-mode",mode,"--config",str(path),"--checkpoint",str(checkpoint),
        "--output-dir",str(output),"--freeze-manifest",str(directory/"absent-freeze.json")],
        cwd=ROOT,text=True,capture_output=True)
    return result,output


@pytest.mark.parametrize("profile",PROFILES)
def test_real_native_cli_retains_episode_metrics_and_manifest(ten_expert_case,tmp_path,profile):
    config,registry,_,_,_,checkpoint = development_case(ten_expert_case,tmp_path,profile)
    frozen = Path(registry["grid"]["normalization"]["artifact_ref"])
    before = frozen.read_bytes()
    result,output = run_cli(config,checkpoint,tmp_path)
    assert result.returncode==0,result.stdout+result.stderr
    payload = json.loads((output/"metrics.json").read_text())
    validate_native_metric_payload(payload,config)
    assert payload["counts"]["episode_count"]==payload["counts"]["defined_episode_count"]==2
    assert payload["counts"]["group_count"]==2 and payload["counts"]["realized_draw_count"]==2
    assert payload["counts"]["valid_cell_count"]==6480
    assert payload["provenance"]["selected_experiment_family"]==profile
    assert payload["provenance"]["training_seed_binding"]["selected_formal_seed"]==41
    assert payload["protocol"]["quantile_levels"]==[.05,.5,.95]
    assert payload["metrics"]["predictive_calibration_error"]["mean"]==pytest.approx(abs(payload["metrics"]["predictive_coverage"]["mean"]-.9))
    for row,sample in zip(payload["episodes"],["sample-a","sample-b"]):
        assert row["episode_uid"]==row["sample_uid"]==sample
        assert len(row["trajectory_uids"])==len(row["regimen_uids"])==2
        assert row["prediction_provenance"]["particle_draw_indices"]
    assert json.loads((output/"status.json").read_text())["status"]=="materialized"
    assert json.loads((output/"artifact_manifest.json").read_text())["postrun_status"]=="materialized"
    assert json.loads((output/"run_metadata.json").read_text())["experiment_family"]==profile
    assert frozen.read_bytes()==before


def test_real_target_read_order_mixture_selection_and_perturbation(ten_expert_case,tmp_path,monkeypatch):
    config,registry,arrays,store,store_format,checkpoint = development_case(ten_expert_case,tmp_path)
    events,predictions = [],[]
    read = native.load_records
    predict = evaluation.predict_native_episode
    def watched_read(*args,**kwargs):
        events.append(("read",kwargs.get("include_teacher",True)))
        return read(*args,**kwargs)
    def watched_predict(*args,**kwargs):
        events.append(("predict",None))
        output = predict(*args,**kwargs)
        predictions.append(output)
        if len(predictions) == 1:
            alternative = predict(*args[:-1], torch.full_like(args[-1], 7.), **kwargs)
            assert torch.equal(output["field_trajectory_samples"], alternative["field_trajectory_samples"])
        assert float(output["router_diagnostics"]["mddm_teacher_loss_evaluated"])==0
        return output
    monkeypatch.setattr(native,"load_records",watched_read)
    monkeypatch.setattr(evaluation,"predict_native_episode",watched_predict)
    first,_ = evaluation.evaluate_native_rollout_prediction(config,checkpoint)
    assert events==[("read",False),("predict",None),("read",True)]*2
    for output in predictions:
        mixture = output["field_trajectory_samples"]
        indices = output["particle_draw_indices"]
        assert not torch.allclose(output["particle_weights"],torch.full_like(output["particle_weights"],.5))
        for draw in range(mixture.shape[0]):
            assert torch.equal(mixture[draw,0],output["particle_trajectory_samples"][draw,0,indices[0,draw]])
    saved = [output["field_trajectory_samples"].clone() for output in predictions]
    arrays["trajectory"] += 1.0
    for row in registry["teachers"]:
        row["content_hash"] += "-changed-target"
    write_native_microenvironment_store(store,config,registry,arrays,store_format)
    events.clear()
    predictions.clear()
    second,_ = evaluation.evaluate_native_rollout_prediction(config,checkpoint)
    assert events==[("read",False),("predict",None),("read",True)]*2
    assert all(torch.equal(value,output["field_trajectory_samples"]) for value,output in zip(saved,predictions))
    assert first["metrics"]["crps"]["mean"] != second["metrics"]["crps"]["mean"]
    assert first["counts"]==second["counts"]


def test_actual_point_compatibility_and_existing_loader(ten_expert_case,tmp_path):
    from Libs.scripts.services.serve_screening_api import load_screening_model
    config,_,_,_,_,checkpoint = development_case(ten_expert_case,tmp_path,point=True)
    model = load_screening_model(checkpoint,config)  # Existing two-argument loader remains supported.
    assert model.world_state_posterior.family=="deterministic_compatibility"
    payload,_ = evaluation.evaluate_native_rollout_prediction(config,checkpoint)
    assert payload["status"]=="complete",payload
    assert payload["counts"]["episode_count"]==1
    assert payload["episodes"][0]["prediction_provenance"]["particle_draw_indices"] is None


@pytest.mark.parametrize("mutation",["role","protected","units"])
def test_source_admission_precedes_payload_reads(ten_expert_case,tmp_path,monkeypatch,mutation):
    config,registry,_,_,_,checkpoint = development_case(ten_expert_case,tmp_path)
    if mutation=="role":
        config["paper_grade_full_suite"]["native_rollout_evaluation"]["source_role"]="test"
    elif mutation=="protected":
        registry["sources"][0]["protected"]=True
    else:
        registry["sources"][0]["predictive_evaluation_units"].pop("evidence_ref")
    path = tmp_path/"admission.json"
    path.write_text(json.dumps(registry))
    config["native_diffusion_data"]["registry_path"]=str(path)
    def forbidden(*args,**kwargs):
        raise AssertionError("payload read before source admission")
    monkeypatch.setattr(native,"load_records",forbidden)
    with pytest.raises(ValueError):
        evaluation.evaluate_native_rollout_prediction(config,checkpoint)


@pytest.mark.parametrize("mutation",["masked","missing_mask","one_draw"])
def test_real_cli_preserves_incomplete_episode_denominators(ten_expert_case,tmp_path,mutation):
    config,registry,arrays,store,store_format,checkpoint = development_case(ten_expert_case,tmp_path)
    if mutation=="one_draw":
        config["neuronsifter_scale_registry"]["quantized_mddm_main_rollout"]["partial_mask_schedule"]["inference_sample_count"]=1
        kwargs = screening_kwargs(config)
        torch.manual_seed(screening_training_seed_binding(config,"folds","fixture")["model_seed"])
        model = ADScreeningSystem(**kwargs).eval()
        save_portable_training_checkpoint(checkpoint,{"model_kwargs":kwargs,"model_state":model.state_dict(),
            "training_seed_binding":screening_training_seed_binding(config,"folds","fixture"),
            "model_contract_version":config["screening_pipeline"]["screening_benchmark"]["model_contract_version"]})
    else:
        for row in registry["teachers"][:2]:
            if mutation=="masked":
                row["validity_mask"]={"kind":"explicit","evidence_ref":"fixture-all-missing",
                                      "values":np.zeros((3,6,6,15),dtype=bool).tolist()}
            else:
                row.pop("validity_mask")
        write_native_microenvironment_store(store,config,registry,arrays,store_format)
    result,output = run_cli(config,checkpoint,tmp_path)
    assert result.returncode==2,result.stdout+result.stderr
    payload=json.loads((output/"metrics.json").read_text())
    assert payload["status"]=="incomplete"
    assert payload["counts"]["episode_count"]==2
    assert payload["counts"]["defined_episode_count"]==(0 if mutation=="one_draw" else 1)
    assert payload["counts"]["failed_episode_count"]==(1 if mutation=="missing_mask" else 0)
    assert payload["counts"]["undefined_episode_count"]==({"masked":1,"one_draw":2,"missing_mask":0}[mutation])
    assert json.loads((output/"status.json").read_text())["status"]=="incomplete"


@pytest.mark.parametrize("mutation",["seed","order","normalization","feature_context","feature_universe"])
def test_native_checkpoint_config_conflicts_reject(ten_expert_case,tmp_path,mutation):
    config,_,_,_,_,checkpoint=development_case(ten_expert_case,tmp_path)
    if mutation=="seed":
        config["selected_formal_seed"]=44
        config["screening_pipeline"]["screening_benchmark"]["training"]["random_seed"]=44
    elif mutation=="order":
        config=control_config(config,"rollout_operator_reversed_experts")
    elif mutation == "feature_context":
        path = tmp_path/"training_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["split_name"] = "different-training-scope"
        path.write_text(json.dumps(manifest))
    elif mutation == "feature_universe":
        path = tmp_path/"selected_regimens.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "regimen_uid"] = "absent-action"
        frame.to_parquet(path, index=False)
    else:
        payload=torch.load(checkpoint,map_location="cpu")
        key=next(key for key in payload["model_state"] if key.endswith("field_normalization.scale"))
        payload["model_state"][key]=payload["model_state"][key]+1
        save_portable_training_checkpoint(checkpoint,payload)
    with pytest.raises(ValueError):
        evaluation.evaluate_native_rollout_prediction(config,checkpoint)


def test_default_catalog_cli_preserves_existing_collection(ten_expert_case, tmp_path):
    config = copy.deepcopy(ten_expert_case[1])
    config["logging_registry"].update(file_enabled=False, console_enabled=False, root_dir=str(tmp_path/"logs"))
    config["ai_ready_preprocessing"]["simulation_audit"]["local_root"] = str(tmp_path/"absent-qc")
    config["screening_pipeline"]["screening_benchmark"]["training"].update(
        seed_policy="manuscript_native_five", random_seed=41)
    config["selected_formal_seed"] = 41
    catalog = tmp_path/"catalog.tsv"
    catalog.write_text("metric_name\tmetric_value\nrollout_nrmse\t2\nrollout_nrmse\t4\nother_metric\t99\n")
    config["paper_grade_full_suite"]["models"]["neuronsifter_mainline"]["result_catalog_template"] = str(catalog)
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    output = tmp_path/"catalog-output"
    result = subprocess.run([sys.executable, str(ROOT/"Libs/exp/evaluation/evaluate_rollout_prediction.py"),
        "--config", str(path), "--output-dir", str(output),
        "--freeze-manifest", str(tmp_path/"absent-freeze.json")], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads((output/"metrics.json").read_text())
    assert payload["source_status"] == "catalog_available"
    assert payload["metrics"] == {"rollout_nrmse": {"values":[2.,4.], "mean":3., "count":2}}
    assert json.loads((output/"artifact_manifest.json").read_text())["postrun_status"] == "materialized"



@pytest.mark.parametrize("mutation", ["missing", "agent_schema"])
def test_native_target_evaluation_rejects_checkpoint_identity_before_arrays(ten_expert_case, tmp_path, monkeypatch, mutation):
    config, _, _, _, _, checkpoint = development_case(ten_expert_case, tmp_path, "action_conditioning_target_identity")
    payload = torch.load(checkpoint, map_location="cpu")
    key = next(name for name in payload["model_state"] if name.endswith("action_conditioning_identity"))
    if mutation == "missing":
        payload["model_state"].pop(key)
    else:
        identity = json.loads(bytes(payload["model_state"][key].tolist()))
        identity["action_identity_schema"] = "canonical_agent_set_sha256_float4.v1"
        payload["model_state"][key] = torch.tensor(list(json.dumps(identity).encode()), dtype=torch.uint8)
    save_portable_training_checkpoint(checkpoint, payload)
    monkeypatch.setattr(native, "load_records", lambda *args, **kwargs: pytest.fail("Invalid target checkpoint reached array reads"))
    with pytest.raises(ValueError, match="identity"):
        evaluation.evaluate_native_rollout_prediction(config, checkpoint)
