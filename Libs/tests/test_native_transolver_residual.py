from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from Libs.tests.test_expert_composition import ten_expert_case
from Libs.tests.test_native_field_normalization import joint_normalized_case
from Libs.tests.test_native_transolver_mean import setup, train_and_save, point_training_config
from Libs.tests.test_physical_occupancy import reseal
from Libs.utils.data.physical_occupancy import compile_physical_record
from Libs.exp.evaluation import evaluate_screening_system as evaluation
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.model.microenvironment_teacher import write_native_microenvironment_store
from Libs.utils.model.native_direct_mean import load_direct_predictor, predict_direct_mean
from Libs.utils.model.native_transolver_residual import (
    LAW, episode_statistics, episode_log_likelihood, residual_objective, residual_draws,
    fit_scales, check_diagnostics)
from Libs.utils.evaluation.native_transolver_calibration import (
    residual_policy, admit_partition, conditional_fields, run_calibration,
    load_residual_artifact, prediction_binding, predict_fitted_residual)

ROOT = Path(__file__).resolve().parents[2]


def policy():
    return residual_policy(load_config("Libs/configs/config_stack.yaml"))


def test_masked_whole_mixture_matches_dense_reference():
    generator = torch.Generator().manual_seed(821)
    means = torch.randn((3,2,4,1,1,2),generator=generator,dtype=torch.float64)
    target = torch.randn((2,4,1,1,2),generator=generator,dtype=torch.float64)
    weights = torch.tensor([.1,.3,.6],dtype=torch.float64)
    times = [19.,21.,26.,36.]
    mask = torch.ones_like(target,dtype=torch.bool)
    mask[0,2,0,0,0] = False
    mask[1,1,0,0,1] = False
    eta = torch.tensor([-.3,.4],dtype=torch.float64,requires_grad=True)
    stats = episode_statistics(means,weights,target,mask,times,672.,"episode","group")
    actual = episode_log_likelihood(eta,stats)
    logs = []
    for ji in range(3):
        total = weights[ji].log()
        for ai in range(2):
            for si in range(2):
                keep = mask[ai,:,0,0,si].clone(); keep[0] = False
                t = torch.tensor(times,dtype=torch.float64)[keep]-times[0]
                covariance = torch.minimum(t[:,None],t[None,:])*eta[si].exp().square()/672.
                total = total+torch.distributions.MultivariateNormal(
                    means[ji,ai,keep,0,0,si], covariance_matrix=covariance).log_prob(target[ai,keep,0,0,si])
        logs.append(total)
    torch.testing.assert_close(actual,torch.logsumexp(torch.stack(logs),0))
    target[:,0] += 10000
    mask[:,0] = False
    changed = episode_statistics(means,weights,target,mask,times,672.,"episode","group")
    torch.testing.assert_close(actual,episode_log_likelihood(eta,changed))
    seconds = episode_statistics(means,weights,target,mask,[x*3600 for x in times],672.*3600,"episode","group")
    torch.testing.assert_close(actual,episode_log_likelihood(eta,seconds))
    assert stats.count == 10


def test_equal_group_episode_objective_and_empty_units():
    means = torch.zeros(1,1,3,1,1,15,dtype=torch.float64)
    y = means[0]+1
    mask = torch.ones_like(y,dtype=torch.bool)
    rows = [episode_statistics(means,[1.],y*k,mask,[0,10,20],672,str(k),group)
            for k,group in [(1,"a"),(2,"a"),(3,"b")]]
    eta = torch.zeros(15,dtype=torch.float64)
    values = [-episode_log_likelihood(eta,row)/row.count for row in rows]
    torch.testing.assert_close(residual_objective(eta,rows), ((values[0]+values[1])/2+values[2])/2)
    empty = episode_statistics(means,[1.],y,mask & False,[0,10,20],672,"empty","empty")
    with pytest.raises(ValueError,match="Empty"):
        episode_log_likelihood(eta,empty)
    failed = fit_scales(rows+[empty],["a","b","empty"],policy()["numerical"])
    assert failed["status"] == "FIT_FAILED" and failed["scales"] is None


@pytest.mark.parametrize("failure",["species","collapsed","nonfinite","budget","iterations"])
def test_whole_law_fit_failures(failure):
    generator = torch.Generator().manual_seed(47)
    means = torch.zeros(1,1,3,3,3,15,dtype=torch.float64)
    target = torch.randn(means.shape[1:],generator=generator,dtype=torch.float64)
    mask = torch.ones_like(target,dtype=torch.bool)
    if failure == "species": mask[...,-1] = False
    if failure == "collapsed": target.zero_()
    if failure == "nonfinite":
        target[0,1,0,0,0] = float("nan")
        with pytest.raises(ValueError,match="finite"):
            episode_statistics(means,[1.],target,mask,[0,10,20],672,"e","g")
        return
    row = episode_statistics(means,[1.],target,mask,[0,10,20],672,"e","g")
    n = policy()["numerical"]
    if failure == "budget": n["max_closure_evaluations"] = 1
    if failure == "iterations": n["max_iterations"] = 1
    fit = fit_scales([row],["g"],n)
    assert fit["status"] == "FIT_FAILED" and fit["scales"] is None
    assert fit["cost"]["optimizer_closure_evaluations"] <= n["max_closure_evaluations"]


def test_residual_path_rng_covariance_initial_and_prefix():
    means = torch.zeros(2,2,4,1,1,2,dtype=torch.float64)
    means[1] = -10.
    times = [0.,12.,24.,48.]
    scales = [.7,1.3]
    count = 2048
    indices = torch.arange(count) % 2
    draws = residual_draws(means,scales,times,672.,indices,
        episode_uid="whole",residual_seed=32,draw_ids=list(range(count)))
    assert torch.equal(draws[:,:,0],means[indices,:,0])
    assert (draws < 0).any()  # retained raw physical violation, no clipping/resampling
    piece = residual_draws(means,scales,times,672.,indices[100:110],
        episode_uid="whole",residual_seed=32,draw_ids=list(range(100,110)))
    assert torch.equal(piece,draws[100:110])
    prefix = residual_draws(means[:,:,:3],scales,times[:3],672.,indices[:10],
        episode_uid="whole",residual_seed=32,draw_ids=list(range(10)))
    assert torch.equal(prefix,draws[:10,:,:3])
    residual = draws-means[indices]
    covariance = torch.cov(residual[:,0,1:3,0,0,0].T)[0,1]
    assert float(covariance) == pytest.approx(scales[0]**2*12/672,rel=.1)
    cross = (residual[:,0,1,0,0,0]*residual[:,1,1,0,0,0]).mean()
    assert abs(float(cross)) < .001
    second_units = residual_draws(means,scales,[t*3600 for t in times],672*3600,indices[:10],
        episode_uid="whole",residual_seed=32,draw_ids=list(range(10)))
    torch.testing.assert_close(second_units,draws[:10])


def test_check_total_covariance_and_missing_strata():
    means = torch.zeros(2,1,3,1,2,2,dtype=torch.float64)
    means[1] = 2
    target = means[0]+1
    mask = torch.ones_like(target,dtype=torch.bool); mask[:,1,0,1,1] = False
    out = check_diagnostics(means,[.25,.75],target,mask,[0,12,24],672,[1.,2.],
                            [[1.,1.],[1.,1.]],["a","b"])
    assert out["pit"]["t0_continuous_pit_count"] == 0
    assert out["pit"]["future"]["count"] == 7
    assert out["covariance"]["graph_edges"][0]["total_covariance_mean"] == pytest.approx(.75)
    assert out["covariance"]["cross_species_pairs"][0]["total_covariance_mean"] == pytest.approx(.75)
    temporal = out["covariance"]["adjacent_retained_times"][0]
    assert temporal["total_covariance_mean"] == pytest.approx(.75+12/672)
    missing = check_diagnostics(means,[.25,.75],target,mask & False,[0,12,24],672,[1.,2.],
                               [[1.,1.],[1.,1.]],["a","b"])
    assert all(row["status"] == "missing" for rows in missing["covariance"].values() for row in rows)


def cli(config, checkpoint, directory, operation, artifact=None):
    directory.mkdir(parents=True,exist_ok=True)
    cfg = directory/"config.yaml"; cfg.write_text(yaml.safe_dump(config,sort_keys=False))
    args = [sys.executable,str(ROOT/"Libs/exp/evaluation/evaluate_rollout_prediction.py"),
        "--execution-mode","native","--native-operation",operation,"--config",str(cfg),
        "--checkpoint",str(checkpoint),"--output-dir",str(directory/"out"),
        "--freeze-manifest",str(directory/"absent-freeze")]
    if artifact is not None: args += ["--residual-artifact",str(artifact)]
    result = subprocess.run(args,cwd=ROOT,text=True,capture_output=True)
    path = directory/"out"/"metrics.json"
    return result,json.loads(path.read_text()) if path.exists() else None


def calibration_fixture(case, directory, point=False):
    config,train_config,registry,arrays,development_store,fmt,_ = setup(case,directory,point=point)
    if point:
        train_config = point_training_config(case,directory,config)
    development_registry, development_arrays = copy.deepcopy(registry), copy.deepcopy(arrays)
    store = directory/"calibration-source"
    if point:
        observation = copy.deepcopy(registry["observations"][0])
        old_posterior = observation["posterior_uid"]
        new_posterior = old_posterior+"-independent-check"
        def clone(value):
            if isinstance(value,dict): return {key:clone(item) for key,item in value.items()}
            if isinstance(value,list): return [clone(item) for item in value]
            if value == "sample-a": return "sample-b"
            if value == old_posterior: return new_posterior
            return value
        check_observation = clone(observation); check_observation["observation_index"] = 1
        registry["observations"].append(check_observation)
        checks_programs = [compile_physical_record(reseal(clone(row["physical_program"]["inputs"])))
                           for row in registry["programs"]]
        registry["programs"].extend(checks_programs)
        checks = [clone(row) for row in registry["teachers"]]
        for row,program in zip(checks,checks_programs):
            row["trajectory_uid"] += "-independent-check"
            row["program_uid"] = program["program_uid"]
        registry["teachers"].extend(checks)
        arrays = {key:np.concatenate([value,value.copy()],axis=0) for key,value in arrays.items()}
        for i,row in enumerate(registry["observations"]): row.update(store_path=str(store),observation_index=i)
        for i,row in enumerate(registry["programs"]): row.update(store_path=str(store),store_format=fmt,program_index=i)
    for i,row in enumerate(registry["teachers"]): row.update(trajectory_store_path=str(store),trajectory_index=i)
    config["native_diffusion_data"]["registry_path"] = str(store/(".zattrs" if point else "store_manifest.json"))
    _, checkpoint, _ = train_and_save(train_config,directory/"trained")
    source = registry["sources"][0]
    source["residual_readout"] = {"kind":"independent_mechanistic_teacher_readout",
        "missingness":"ignorable_given_conditions_and_mask","independent_groups":True,
        "evidence_ref":"synthetic-independent-readout-generator"}
    # Produce fresh fixture calibration episodes from the authentic source schema.
    # These are synthetic records, never promotion of an actual development source.
    if not point:
        registry["joint_sources"][0]["support"]["role"] = "calibration"
        for observation in registry["joint_sources"][0]["observations"]:
            observation["role"] = "calibration"
    for row in registry["teachers"]:
        row.update(split_name="calibration",source_holdout_role="calibration",
                   family_uid="fit-group" if row["sample_uid"] == "sample-a" else "check-group")
    p = config["paper_grade_full_suite"]["native_transolver_residual"]
    p.update(residual_seed=41,fit_groups=[{"group_uid":"fit-group","sample_uids":["sample-a"]}],
             check_groups=[{"group_uid":"check-group","sample_uids":["sample-b"]}])
    write_native_microenvironment_store(store,config,registry,arrays,fmt)
    loaded = load_direct_predictor(config,checkpoint,["regimen-a","regimen-b"])
    law = torch.linspace(.5,1.2,15,dtype=torch.float64)
    for bi,sample in enumerate(["sample-a","sample-b"]):
        records = native.select_records(registry,config,"calibration",[sample],["regimen-a","regimen-b"])
        inputs,_ = native.load_records(registry,records,config,include_teacher=False)
        with torch.no_grad():
            _, prepared = evaluation.prepare_native_episode(loaded["model"],records,config,inputs)
            pred = predict_direct_mean(loaded["predictor"],prepared,loaded["model"].field_rollout_operator.quantized_mddm_rollout_operator)
        means,_ = conditional_fields(pred)
        normalizer = loaded["model"].field_rollout_operator.field_normalization
        # Known synthetic truth uses one whole episode particle and a separate stream.
        target = residual_draws(normalizer.normalize(means),law,registry["grid"]["time_hours"],672.,
            torch.tensor([0 if point else bi%2]),episode_uid=sample,residual_seed=43,draw_ids=[0])[0]
        arrays["trajectory"][bi*2:bi*2+2] = normalizer.inverse(target).numpy()
    write_native_microenvironment_store(store,config,registry,arrays,fmt)
    development_registry["sources"] = copy.deepcopy(registry["sources"])
    return config,checkpoint,registry,arrays,store,fmt,law,loaded,development_registry,development_arrays,development_store


@pytest.mark.parametrize("point",[False,True])
def test_cli_fit_artifact_untouched_check_and_prediction(ten_expert_case,tmp_path,point):
    config,checkpoint,registry,arrays,store,fmt,law,loaded,dev_registry,dev_arrays,dev_store = calibration_fixture(ten_expert_case,tmp_path,point)
    artifact = tmp_path/"fit"/"residual.json"
    result,fit = cli(config,checkpoint,tmp_path/"fit","fit",artifact)
    assert result.returncode == 0,(result.stdout,result.stderr,fit)
    assert fit["status"] == "FIT_COMPLETE" and artifact.is_file()
    model_bytes = checkpoint.read_bytes()
    conditioner_path = Path(loaded["conditioning"]["identity"]["checkpoint_path"])
    conditioner_bytes = conditioner_path.read_bytes()
    normalizer_path = Path(loaded["model"].field_rollout_operator.field_normalization.spec["artifact_ref"])
    normalizer_bytes = normalizer_path.read_bytes()
    frozen_bytes = artifact.read_bytes()
    fitted = json.loads(frozen_bytes)
    assert fitted["counts"]["future_coordinate_count"] == 2*2*6*6*15
    assert fitted["counts"]["eligible_group_count"] == 1
    assert fitted["binding"]["direct_training_seed_binding"] != fitted["binding"]["conditioning_identity"]["training_seed_binding"]
    torch.testing.assert_close(torch.tensor(fitted["scales"]),law.float(),rtol=.3,atol=0)
    config["paper_grade_full_suite"]["native_transolver_residual"].update(
        frozen_artifact_path=str(artifact),frozen_artifact_sha256=hashlib.sha256(frozen_bytes).hexdigest())
    result,check = cli(config,checkpoint,tmp_path/"check","check",artifact)
    assert result.returncode == 0,(result.stdout,result.stderr,check)
    assert check["status"] == "CHECK_COMPLETE_DIAGNOSTICS_ONLY" and check["scientific_check_pass"] is None
    assert check["episodes"][0]["pit"]["t0_continuous_pit_count"] == 0
    assert check["predictive_scores"]["source_status"] == "native_calibration_check_prediction"
    assert check["predictive_scores"]["metrics"]["crps"]["count"] == 1
    assert check["predictive_scores"]["counts"]["realized_draw_count"] == 2
    arrays["trajectory"][2:,1:] += 10.
    write_native_microenvironment_store(store,config,registry,arrays,fmt)
    result,changed = cli(config,checkpoint,tmp_path/"changed-check","check",artifact)
    assert result.returncode == 0,(result.stdout,result.stderr,changed)
    assert changed["objective"] != check["objective"]
    assert changed["frozen_fit_cost"] == check["frozen_fit_cost"]
    assert changed["frozen_fit_counts"] == check["frozen_fit_counts"]
    assert artifact.read_bytes() == frozen_bytes
    assert checkpoint.read_bytes() == model_bytes
    assert conditioner_path.read_bytes() == conditioner_bytes
    assert normalizer_path.read_bytes() == normalizer_bytes
    # Restore the independently retained development fixture; calibration roles
    # and target stores remain unchanged throughout prediction.
    write_native_microenvironment_store(dev_store,config,dev_registry,dev_arrays,fmt)
    config["native_diffusion_data"]["registry_path"] = str(dev_store/(".zattrs" if point else "store_manifest.json"))
    config["paper_grade_full_suite"]["native_rollout_evaluation"]["predictor_kind"] = "transolver_fitted_residual"
    result,prediction = cli(config,checkpoint,tmp_path/"prediction","predict",artifact)
    assert result.returncode == 0,(result.stdout,result.stderr,prediction)
    assert prediction["status"] == "complete" and prediction["counts"]["realized_draw_count"] == 2
    assert prediction["metrics"]["crps"]["count"] == (1 if point else 2)
    assert prediction["provenance"]["residual_artifact_sha256"] == hashlib.sha256(frozen_bytes).hexdigest()
    assert all(row["prediction_provenance"]["predictive_law_draw_count"] == 2 for row in prediction["episodes"])
    config["paper_grade_full_suite"]["native_transolver_residual"]["frozen_artifact_sha256"] = "0"*64
    result,missing = cli(config,checkpoint,tmp_path/"wrong-law","predict",artifact)
    assert result.returncode == 2 and missing["status"] == "incomplete"


@pytest.mark.parametrize("mutation",["overlap","incomplete","role","grade","missingness","mask"])
def test_partition_admission_precedes_target_reads(ten_expert_case,tmp_path,monkeypatch,mutation):
    config,_,registry,arrays,store,fmt,_ = setup(ten_expert_case,tmp_path)
    p = config["paper_grade_full_suite"]["native_transolver_residual"]
    p.update(fit_groups=[{"group_uid":"fit","sample_uids":["sample-a"]}],
             check_groups=[{"group_uid":"check","sample_uids":["sample-b"]}])
    registry["sources"][0]["residual_readout"] = {"kind":p["readout_kind"],"missingness":p["missingness"],
        "independent_groups":True,"evidence_ref":"fixture"}
    registry["joint_sources"][0]["support"]["role"] = "calibration"
    for obs in registry["joint_sources"][0]["observations"]: obs["role"] = "calibration"
    for row in registry["teachers"]:
        row.update(split_name="calibration",source_holdout_role="calibration",
                   family_uid="fit" if row["sample_uid"] == "sample-a" else "check")
    if mutation == "overlap": p["check_groups"] = copy.deepcopy(p["fit_groups"])
    if mutation == "incomplete": p["fit_groups"][0]["sample_uids"].append("absent")
    if mutation == "role": registry["teachers"][0]["split_name"] = "development"
    if mutation == "grade":
        registry["teachers"][0].update(qc_grade="C",qc_use="weak_teacher_calibration",sample_weight=.1)
    if mutation == "missingness": registry["sources"][0]["residual_readout"]["missingness"] = "outcome_dependent"
    if mutation == "mask": registry["teachers"][0]["validity_mask"] = {"kind":"unknown"}
    def forbidden(*args,**kwargs): pytest.fail("payload read before whole partition admission")
    monkeypatch.setattr(native,"load_records",forbidden)
    with pytest.raises(ValueError):
        admit_partition(config,registry,residual_policy(config))


def test_cli_missing_inputs_status(ten_expert_case,tmp_path):
    config,_,_,_,_,_,checkpoint = setup(ten_expert_case,tmp_path)
    result,payload = cli(config,checkpoint,tmp_path/"missing","fit")
    assert result.returncode == 2 and payload["status"] == "NOT_FIT_MISSING_INPUTS"
    assert not (tmp_path/"missing"/"out"/"residual_parameters.json").exists()

def test_actual_particle_sampling_is_chunk_stable():
    means = torch.zeros(1,2,2,3,1,1,15,dtype=torch.float64)
    means[:,1] = 9.
    weights = torch.tensor([[.2,.8]])
    posterior = SimpleNamespace(weights=weights,posterior_uid="posterior",
                                support=SimpleNamespace(sample_uids=["sample"]))
    sampler = SimpleNamespace(inference_sample_count=4,sampler_seed=31)
    normalizer = SimpleNamespace(normalize=lambda x:x,inverse=lambda x:x)
    original = {"particle_conditional_means":means,"particle_weights":weights}
    def sample(ids):
        return predict_fitted_residual(copy.deepcopy(original),{"joint_posterior":posterior},
            sampler,normalizer,{"time_hours":[0,12,24]},
            {"scales":[1.]*15,"reference_time_hours":672.},policy(),
            episode_uid="sample",draw_ids=ids)
    full,left,right = sample([0,1,2,3]),sample([0,1]),sample([2,3])
    assert torch.equal(full["field_trajectory_samples"],
                       torch.cat([left["field_trajectory_samples"],right["field_trajectory_samples"]]))
    indices = full["particle_draw_indices"][0]
    assert torch.equal(full["field_trajectory_samples"][:,0,:,0],means[0,indices,:,0])
    assert full["predictive_law_draw_count"] == 4


def fitted_artifact_fixture(config, binding):
    p = config["paper_grade_full_suite"]["native_transolver_residual"]
    # A genuine numerical fit supplies the validator fixture.
    means = torch.zeros(1,1,3,2,2,15,dtype=torch.float64)
    y = torch.randn(means.shape[1:],generator=torch.Generator().manual_seed(41),dtype=torch.float64)
    stats = episode_statistics(means,[1.],y,torch.ones_like(y,dtype=torch.bool),[0,12,24],672,"e","g")
    fit = fit_scales([stats],["g"],p["numerical"])
    assert fit["status"] == "FIT_COMPLETE",fit
    return {**fit,"schema_version":config["schemas"]["native_transolver_residual"]["schema_version"],
        "law":LAW,"reference_time_hours":672.,"binding":binding,
        "numerical":p["numerical"],"partition":{"fit_groups":[],"check_groups":[]},
        "admission":{},"fit_target_records":[],"scientific_twin_status":"NOT_RUN"}


def test_artifact_loader_rejects_missing_identity_and_malformed_law(tmp_path):
    config = load_config("Libs/configs/config_stack.yaml")
    p = config["paper_grade_full_suite"]["native_transolver_residual"]
    original = fitted_artifact_fixture(config, {"normalization_id":"fixture"})
    path = tmp_path/"law.json"
    for mutation in ("none","scales","nan_gradient","identity","status","species","hessian","empty_hessian","short_hessian","matrix_hessian"):
        artifact = copy.deepcopy(original)
        if mutation == "scales": artifact["scales"] = [0.]*15
        if mutation == "nan_gradient": artifact["diagnostics"]["gradient_max_abs"] = float("nan")
        if mutation == "identity": artifact["binding"] = {}
        if mutation == "status": artifact["status"] = "FIT_FAILED"
        if mutation == "species": artifact["counts"]["per_species_future_coordinate_count"][-1] = 0
        if mutation == "hessian": artifact["diagnostics"]["hessian_eigenvalues"][-1] = float("inf")
        if mutation == "empty_hessian": artifact["diagnostics"]["hessian_eigenvalues"] = []
        if mutation == "short_hessian": artifact["diagnostics"]["hessian_eigenvalues"] = [1.]*14
        if mutation == "matrix_hessian": artifact["diagnostics"]["hessian_eigenvalues"] = [[1.]*15]
        path.write_text(json.dumps(artifact))
        p["frozen_artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        if mutation == "none":
            assert load_residual_artifact(path,config,original["binding"])["scales"] == original["scales"]
        else:
            with pytest.raises(ValueError):
                load_residual_artifact(path,config,original["binding"])
    p["frozen_artifact_sha256"] = ""
    with pytest.raises(FileNotFoundError):
        load_residual_artifact(path,config,original["binding"])


def test_fitted_prediction_cli_records_empty_hessian_failure(ten_expert_case,tmp_path):
    config,train_config,registry,_,_,_,_ = setup(ten_expert_case,tmp_path)
    _,checkpoint,_ = train_and_save(train_config,tmp_path/"trained")
    loaded = load_direct_predictor(config,checkpoint,["regimen-a","regimen-b"])
    binding = prediction_binding(config,checkpoint,loaded,registry["grid"])
    artifact = fitted_artifact_fixture(config,binding)
    path = tmp_path/"residual.json"
    path.write_text(json.dumps(artifact))
    policy = config["paper_grade_full_suite"]["native_transolver_residual"]
    policy.update(frozen_artifact_path=str(path),
                  frozen_artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert load_residual_artifact(path,config,binding)["status"] == "FIT_COMPLETE"
    # Change only Hessian evidence, then bind the changed bytes correctly.
    # A hash/identity rejection cannot substitute for this validation failure.
    artifact["diagnostics"]["hessian_eigenvalues"] = []
    path.write_text(json.dumps(artifact))
    malformed_bytes = path.read_bytes()
    policy["frozen_artifact_sha256"] = hashlib.sha256(malformed_bytes).hexdigest()
    config["paper_grade_full_suite"]["native_rollout_evaluation"]["predictor_kind"] = "transolver_fitted_residual"
    result,payload = cli(config,checkpoint,tmp_path/"prediction","predict",path)
    assert result.returncode == 2,(result.stdout,result.stderr,payload)
    assert payload["status"] == "incomplete"
    assert payload["source_status"] == "invalid_requested_residual_law"
    assert payload["reason"] == "Residual artifact requires exactly 15 finite Hessian eigenvalues"
    assert payload["metrics"] == {} and "scales" not in payload and "episodes" not in payload
    output = tmp_path/"prediction"/"out"
    assert json.loads((output/"artifact_manifest.json").read_text())["postrun_status"] == "incomplete"
    assert json.loads((output/"status.json").read_text())["status"] == "incomplete"
    assert path.read_bytes() == malformed_bytes
