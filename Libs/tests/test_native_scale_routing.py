"""Posterior-weighted native scale decisions and actual lazy route calls."""
import copy

import pytest
import torch

from Libs.utils.evaluation.native_intervention_decisions import summarize_posterior_action_loss
from Libs.utils.evaluation.native_scale_routing import execute_scale_route, scale_routing_policy


def policy():
    return {"schema_version": "native_scale_routing.v1",
        "methods": ["fixed_threshold", "fixed_route"], "stage_order": ["student", "large", "xl"],
        "fixed_route": ["student", "large", "xl"], "fixed_threshold": .95,
        "threshold_statistic": "selected_action_optimal_posterior_mass",
        "tie_break": "declared_action_order", "checkpoints": {}}


def test_complete_tuple_mass_and_declared_ties_include_deterministic_conditionals():
    draws = torch.tensor([[[0., 1.]], [[1., 0.]]], dtype=torch.float64)
    result = summarize_posterior_action_loss(draws, torch.tensor([.9, .1], dtype=torch.float64))
    assert result["selected_index"] == 0
    assert torch.allclose(result["predictive_risks"], torch.tensor([.1, .9], dtype=torch.float64))
    assert result["selected_optimal_mass"] == pytest.approx(.9)
    tied = summarize_posterior_action_loss(torch.ones(2, 1, 2), torch.tensor([.7, .3]))
    assert tied["selected_index"] == 0
    assert tied["selected_optimal_mass"] == pytest.approx(1.)
    assert tied["tied_optimal_mass"] == pytest.approx(1.)


def test_fixed_threshold_stops_real_calls_and_fixed_route_executes_every_stage():
    forecasts = {"student": [[[0., 1.]], [[1., 0.]]],
                 "large": [[[0., 1.]], [[0., 1.]]],
                 "xl": [[[1., 0.]], [[1., 0.]]]}
    calls = []
    def forecast(stage, event):
        calls.append(stage)
        event["sampling_calls"] = 1
        return {"loss": torch.tensor(forecasts[stage], dtype=torch.float64)}
    weights = torch.tensor([.9, .1], dtype=torch.float64)
    trace = []
    adaptive = execute_scale_route("fixed_threshold", policy(), weights, forecast, trace)
    assert calls == ["student", "large"]
    assert adaptive["stage"] == "large" and adaptive["decision"]["selected_index"] == 0
    assert adaptive["stop_reason"] == "fixed_posterior_mass_threshold"
    calls.clear()
    fixed = execute_scale_route("fixed_route", policy(), weights, forecast, [])
    assert calls == ["student", "large", "xl"]
    assert fixed["stage"] == "xl" and fixed["decision"]["selected_index"] == 1
    assert fixed["stop_reason"] == "declared_route_exhausted"
    assert all(event["stage_elapsed_seconds"] >= 0 for event in trace)


def test_route_failure_is_retained_and_cannot_silently_change_candidate_denominator():
    trace = []
    def forecast(stage, event):
        return {"loss": torch.ones(2, 1, 2 if stage == "student" else 3)}
    with pytest.raises(ValueError, match="candidate set"):
        execute_scale_route("fixed_route", policy(), torch.tensor([.5, .5]), forecast, trace)
    assert len(trace) == 2 and trace[-1]["status"] == "failed"
    assert "candidate set" in trace[-1]["failure"]
    assert trace[-1]["stage_elapsed_seconds"] >= 0


@pytest.mark.parametrize("weights", [torch.tensor([.7, .7]), torch.tensor([-.1, 1.1]),
                                     torch.tensor([float("nan"), 1.])])
def test_invalid_complete_posterior_mass_rejects(weights):
    with pytest.raises(ValueError, match="normalized particle weights"):
        summarize_posterior_action_loss(torch.zeros(2, 1, 2), weights)


def test_routing_policy_rejects_source_arm_and_unfrozen_rule():
    config = {"native_scale_routing": policy(), "native_scientific_training": {"source_scope": "all"}}
    assert scale_routing_policy(config)["fixed_threshold"] == .95
    broken = copy.deepcopy(config)
    broken["native_scientific_training"]["source_scope"] = "pure_shared"
    with pytest.raises(ValueError, match="full-mainline"):
        scale_routing_policy(broken)
    broken = copy.deepcopy(config)
    broken["native_scale_routing"]["fixed_route"] = ["xl"]
    with pytest.raises(ValueError, match="fixed full-mainline"):
        scale_routing_policy(broken)


def test_scale_route_profiles_reuse_selected_upstreams_without_new_fit_and_bind_public_reference():
    from Libs.utils.data.raw_data_registry import load_config
    from Libs.utils.evaluation.manuscript_native_protocol import training_profile, command_catalog
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    generated = {method: training_profile(config, method, 41)
                 for method in ("fixed_threshold", "fixed_route")}
    commands = command_catalog(config, [row[1] for row in generated.values()])
    assert any(row["command_uid"] == "public_cns_reference" for row in commands)
    assert not any(row["command_uid"].endswith(":fit") for row in commands)
    for method, (selected, spec) in generated.items():
        assert spec["training_manifest"] is None and spec["additional_optimizer_updates"] == 0
        assert selected["native_scientific_training"]["selection"]["max_steps"] == 12600
        stages = selected["native_scale_routing"]["checkpoints"]
        assert set(stages) == {"student", "large", "xl"}
        assert "/native_wam/" in stages["large"]["training_manifest"]
        assert "/student/" in stages["student"]["training_manifest"]
        row = next(row for row in commands if row["command_uid"] == spec["profile_id"]+":evaluate")
        assert set(row["dependencies"]) == {
            "native_wam_fraction_100_seed_41:fit", "student_fraction_100_seed_41:fit",
            "xl_fraction_100_seed_41:fit", "public_cns_reference"}
        assert "--training-manifest" not in row["argv"]
        assert "--decision-reference" in row["argv"]
        assert row["additional_optimizer_updates"] == 0


def test_float32_probability_roundoff_is_normalized_without_changing_tuple_weights():
    weights = torch.tensor([.2, .8], dtype=torch.float32)
    original = weights.clone()
    assert weights.double().sum() > 1
    result = summarize_posterior_action_loss(torch.ones(2, 1, 2), weights)
    assert result["selected_optimal_mass"] == 1.
    assert result["tied_optimal_mass"] == 1.
    assert torch.equal(result["predictive_risks"], torch.ones(2, dtype=torch.float64))
    assert torch.equal(weights, original)


def test_adaptive_threshold_search_matches_exhaustive_constrained_regret():
    import itertools
    import numpy as np
    from Libs.utils.evaluation.native_scale_routing import optimize_scale_thresholds
    masses=np.array([[.1,.1],[.2,.8],[.8,.1],[.9,.8]])
    regrets=np.array([[1.,.5,0.],[.4,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    costs=np.tile([1.,5.,6.],(4,1));shared=np.full(4,.5)
    result=optimize_scale_thresholds(masses,regrets,costs,shared)
    candidates=[np.unique(np.r_[0.,masses[:,i],2.]) for i in range(2)]
    possible=[]
    for student,large in itertools.product(*candidates):
        stages=np.where(masses[:,0]>=student,0,np.where(masses[:,1]>=large,1,2))
        cost=np.mean(shared+costs[:,0]+(stages>=1)*costs[:,1]+(stages==2)*costs[:,2])
        if cost<=np.mean(shared+costs[:,1]):
            possible.append((np.mean(regrets[np.arange(4),stages]),cost,student,large))
    assert tuple([result["mean_selected_controlled_oracle_regret"],result["mean_resident_cost_seconds"],
                  result["thresholds"]["student"],result["thresholds"]["large"]])==min(possible)
    assert result["thresholds"]=={"student":.8,"large":.8}
    assert result["selected_stage_counts"]=={"student":2,"large":1,"xl":1}
    assert result["mean_resident_cost_seconds"]==result["large_only_mean_resident_cost_seconds"]==5.5


def test_adaptive_endpoints_infeasibility_and_finite_json():
    import json
    import numpy as np
    from Libs.utils.evaluation.native_scale_routing import optimize_scale_thresholds
    masses=np.array([[0.,1.],[1.,0.]])
    costs=np.tile([1.,3.,6.],(2,1))
    result=optimize_scale_thresholds(masses,np.zeros((2,3)),costs,np.zeros(2))
    assert result["thresholds"]=={"student":0.,"large":0.}
    assert result["endpoint_thresholds"]=={"always_stop":0.,"always_escalate":2.}
    json.dumps(result,allow_nan=False)
    costs[:,0]=4.
    failed=optimize_scale_thresholds(masses,np.zeros((2,3)),costs,np.zeros(2))
    assert failed["status"]=="infeasible" and "thresholds" not in failed
    costs[0,0]=float("nan")
    with pytest.raises(ValueError,match="complete finite"):
        optimize_scale_thresholds(masses,np.zeros((2,3)),costs,np.zeros(2))


def test_adaptive_inference_uses_only_frozen_current_stage_mass():
    calls=[]
    rules=policy();rules["methods"].append("adaptive_scale")
    def forecast(stage,event):
        calls.append(stage)
        if stage=="xl":raise AssertionError("Unreached model or truth must not be queried")
        return {"loss":torch.tensor([[[0.,1.]],[[1.,0.]]] if stage=="student"
                                     else [[[0.,1.]],[[0.,1.]]])}
    thresholds={"student":1.,"large":.9}
    result=execute_scale_route("adaptive_scale",rules,torch.tensor([.9,.1]),forecast,[],
                               thresholds=thresholds)
    assert calls==["student","large"] and result["stage"]=="large"
    assert result["stop_reason"]=="development_fitted_mass_threshold"
    assert thresholds=={"student":1.,"large":.9}
    with pytest.raises(ValueError,match="frozen finite"):
        execute_scale_route("adaptive_scale",rules,torch.tensor([.9,.1]),forecast,[],
                            thresholds={"student":float("inf"),"large":.9})


def test_selected_loss_quantile_is_signed_order_statistic_with_explicit_infinity():
    import json
    from Libs.utils.evaluation.native_scale_routing import selected_loss_quantile,selected_loss_upper_bound
    finite=selected_loss_quantile([-.5,-.2,.1,.7],.8)
    assert finite["order_statistic_rank"]==4 and finite["value"]==.7
    assert selected_loss_upper_bound(.6,finite,1.1)==1.1
    negative=selected_loss_quantile([-.5,-.5,-.5,-.5],.8)
    assert selected_loss_upper_bound(.2,negative,1.1)==0.
    infinite=selected_loss_quantile([.1],.95)
    assert infinite["kind"]=="positive_infinity" and "value" not in infinite
    assert selected_loss_upper_bound(.2,infinite,1.1)==1.1
    json.dumps(infinite,allow_nan=False)


@pytest.fixture(scope="module")
def adaptive_full_record_fixture():
    from Libs.utils.data.raw_data_registry import load_config
    from Libs.utils.evaluation.native_scale_routing import scale_role_contract,fit_scale_policy_records
    config=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    denominator,declarations=scale_role_contract(config)
    assert denominator["status"]=="complete"
    assert denominator["roles"]["development"]["required_total"]==800
    assert all(denominator["roles"][name]["required_total"]==2048 for name in ("fit","check","threshold"))
    binding={"scope":"artificial_scalar_unit_records_no_model_or_source_execution"}
    rows=[]
    for index,(uid,declared) in enumerate(declarations["development"].items()):
        cell=index%4
        masses=[(.1,.1),(.2,.8),(.8,.1),(.9,.8)][cell]
        observed=[0.,[1.,.4,0.,0.][cell],[.5,0.,0.,0.][cell]]
        stages=[]
        for stage,choice,mass,cost in zip(("student","large","xl"),(1,2,0),(*masses,1.),(1.,5.,6.)):
            stages.append({"stage":stage,"selected_supported_index":choice,
                           "selected_optimal_mass":mass,"resident_query_seconds":cost})
        rows.append({"group_uid":uid,"family_uid":declared["family_uid"],"role":"development",
                     "calibration_reservation":None,"status":"defined","route":stages,
                     "observed_public_losses":observed,"shared_input_seconds":.5})
    report={"status":"complete","method":"fixed_route","frozen_scale_binding":binding,"groups":rows}
    policy_artifact=fit_scale_policy_records(report,declarations["development"],binding,denominator)
    assert policy_artifact["status"]=="complete" and policy_artifact["formal_qualification"] is False
    assert policy_artifact["thresholds"]=={"student":.8,"large":.8}
    return config,denominator,declarations,binding,policy_artifact,report


def _selected_scale_report(declarations,binding,policy_artifact,actual):
    rows=[]
    for uid,declared in declarations.items():
        rows.append({"group_uid":uid,"family_uid":declared["family_uid"],"role":"calibration",
            "calibration_reservation":declared["calibration_reservation"],"status":"defined",
            "supported_regimen_uids":["a","b"],"selected_regimen_uid":"a",
            "predictive_risks":[.3,.7],"observed_public_losses":[actual,1.],"maximum_supported_loss":1.1})
    return {"status":"complete","method":"adaptive_scale","groups":rows,
            "frozen_scale_binding":binding,"adaptive_policy_id":policy_artifact["policy_id"]}


def test_selected_calibration_is_disjoint_and_cannot_tune_policy_or_check_outcomes(adaptive_full_record_fixture):
    from Libs.utils.evaluation.native_scale_routing import (
        fit_selected_scale_calibration,check_selected_scale_calibration,validate_scale_artifact)
    config,_,ids,binding,policy_artifact,_=adaptive_full_record_fixture
    original=copy.deepcopy(policy_artifact)
    fit_report=_selected_scale_report(ids["fit"],binding,policy_artifact,.4)
    fitted=fit_selected_scale_calibration(fit_report,ids["fit"],policy_artifact,binding,[.8,.85,.9,.95])
    fixed_quantiles=copy.deepcopy(fitted["quantiles"])
    check_report=_selected_scale_report(ids["check"],binding,policy_artifact,.9)
    checked=check_selected_scale_calibration(check_report,ids["check"],fitted,policy_artifact,binding)
    assert checked["status"]=="complete" and checked["independent_check_complete"]
    assert all(row["coverage"]==0. for levels in checked["check_coverage"].values() for row in levels.values())
    assert fitted["quantiles"]==fixed_quantiles and policy_artifact==original
    assert not checked["formal_qualification"] and not checked["threshold_role_used"]
    validate_scale_artifact(checked,"calibration_id",binding)
    with pytest.raises(ValueError,match="identity/binding"):
        validate_scale_artifact(policy_artifact,"policy_id",{"scope":"changed_checkpoint_or_config"})
    bad=copy.deepcopy(policy_artifact);bad["thresholds"]["student"]=0.
    with pytest.raises(ValueError,match="identity/binding"):
        fit_selected_scale_calibration(fit_report,ids["fit"],bad,binding,[.8,.85,.9,.95])
    with pytest.raises(ValueError,match="role/reservation"):
        check_selected_scale_calibration(fit_report,ids["fit"],fitted,policy_artifact,binding)
    missing=copy.deepcopy(fit_report);missing["groups"].pop()
    with pytest.raises(ValueError,match="missing"):
        fit_selected_scale_calibration(missing,ids["fit"],policy_artifact,binding,[.8,.85,.9,.95])


def test_adaptive_consumers_reject_reduced_denominators_and_unavailable_inputs(adaptive_full_record_fixture):
    from Libs.utils.evaluation.native_scale_routing import scale_role_contract,fit_scale_policy_records
    config,denominator,ids,binding,_,report=adaptive_full_record_fixture
    small=copy.deepcopy(config)
    for partition in small["native_observable_bridge"]["group_partitions"]:partition["groups_per_family"]=1
    result,_=scale_role_contract(small)
    assert result["status"]=="insufficient_data"
    assert result["roles"]["development"]["declared_total"]==4
    assert result["roles"]["development"]["required_total"]==800
    with pytest.raises(ValueError,match="full independent"):
        fit_scale_policy_records(report,ids["development"],binding,result)
    clipped=copy.deepcopy(report);clipped["groups"][0]["status"]="defined_on_supported_actions"
    with pytest.raises(ValueError,match="role/reservation"):
        fit_scale_policy_records(clipped,ids["development"],binding,denominator)
    small["native_scale_routing"]["adaptive"]["development_groups_per_family"]=1
    with pytest.raises(ValueError,match="full group counts"):
        scale_role_contract(small)


def test_adaptive_catalog_freezes_policy_before_separate_calibration():
    from Libs.utils.data.raw_data_registry import load_config
    from Libs.utils.evaluation.manuscript_native_protocol import training_profile,command_catalog,validate_definition
    config=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    selected,spec=training_profile(config,"adaptive_scale",41)
    commands={row["command_uid"]:row for row in command_catalog(config,[spec])}
    uid=spec["profile_id"]
    assert commands[uid+":calibrate"]["dependencies"]==[uid+":fit_policy"]
    assert commands[uid+":evaluate"]["dependencies"]==[uid+":calibrate"]
    assert "--scale-policy" in commands[uid+":calibrate"]["argv"]
    assert "--scale-calibration" in commands[uid+":evaluate"]["argv"]
    assert all(row.get("additional_optimizer_updates",0)==0 for row in commands.values())
    assert selected["native_scientific_training"]["predictive_budget"]["categorical_reverse_steps"]==100
    assert spec["training_manifest"] is None
    validate_definition(config)


def test_actual_adaptive_cli_reuses_retained_bindings_and_withholds_tiny_data(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import yaml
    from Libs.utils.data.raw_data_registry import load_config
    from Libs.utils.model.native_direct_mean import file_sha256
    source=os.environ.get("NRNSIFT_RHS_FIXTURE")
    retained=os.environ.get("NRNSIFT_SCALE_ROUTES")
    if not source or not retained:pytest.skip("Requires retained frozen posterior and scale route bindings")
    source=Path(source);retained=Path(retained)
    config=load_config(str(retained))
    canonical=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    prior=copy.deepcopy(config["native_scale_routing"]["checkpoints"])
    config["native_scale_routing"]=copy.deepcopy(canonical["native_scale_routing"])
    config["native_scale_routing"]["checkpoints"]=prior
    for schema_name in ("native_adaptive_scale_policy","native_scale_selected_calibration","native_scale_evaluation"):
        config["schemas"][schema_name]=copy.deepcopy(canonical["schemas"][schema_name])
    original={str(path):file_sha256(path) for path in [retained,source/"config.yaml",source/"native_fit/manifest.json"]}
    configured=tmp_path/"adaptive_tiny.yaml";configured.write_text(yaml.safe_dump(config,sort_keys=False))
    root=Path(__file__).resolve().parents[2]
    reports=[]
    for command,name in (("fit_scale_policy","native_adaptive_scale_policy.json"),
                         ("calibrate_scale_policy","native_scale_selected_calibration.json"),
                         ("evaluate_baseline","native_scale_evaluation.json")):
        output=tmp_path/command
        completed=subprocess.run([sys.executable,str(root/"Libs/exp/baselines/run_external_baseline_bridge.py"),
            "--protocol-scope","manuscript-native","--config",str(configured),"--baseline-name","adaptive_scale",
            "--command",command,"--conditioning-config",str(source/"config.yaml"),
            "--conditioning-manifest",str(source/"native_fit/manifest.json"),
            "--decision-reference",str(source/"public_cns_reference.json"),"--device","cpu",
            "--output-dir",str(output)],cwd=root,text=True,capture_output=True)
        assert completed.returncode==2,completed.stdout+completed.stderr
        report=json.loads((output/name).read_text())
        assert report["schema_version"]==name.removesuffix(".json")+".v1"
        assert report["status"]=="insufficient_data" and report["model_calls"]==0
        assert report["additional_optimizer_updates"]==report["source_provider_calls"]==0
        assert report["denominator"]["roles"]["development"]["required_total"]==800
        assert report["denominator"]["roles"]["development"]["declared_total"]==4
        assert not report["calibration_outcome_access"]
        from Libs.utils.evaluation.native_scale_routing import validate_scale_artifact
        schema=config["schemas"][name.removesuffix(".json")]
        validate_scale_artifact(report,"",{},insufficient_schema=schema)
        key="policy_id" if command=="fit_scale_policy" else "calibration_id"
        with pytest.raises(ValueError,match="incomplete"):
            validate_scale_artifact(report,key,{})
        invalid=dict(report,model_calls=1)
        with pytest.raises(ValueError,match="zero-call"):
            validate_scale_artifact(invalid,"",{},insufficient_schema=schema)
        invalid=dict(report,policy_id="fabricated_success")
        with pytest.raises(ValueError,match="zero-call"):
            validate_scale_artifact(invalid,"",{},insufficient_schema=schema)
        reports.append({"command":command,"exit_code":completed.returncode,"report":report})
    assert all(file_sha256(Path(path))==digest for path,digest in original.items())
    receipt={"scope":"real_cli_tiny_fixture_role_rejection_no_predictions","commands":reports,
        "reused_checkpoint_bindings":prior,"original_file_hashes":original,
        "additional_optimizer_updates":0,"model_calls":0,"source_provider_calls":0,
        "scientific_twin":"NOT_RUN","formal_qualification":False}
    Path(os.environ.get("NRNSIFT_ADAPTIVE_SCALE_EVIDENCE",str(tmp_path/"result.json"))).write_text(json.dumps(receipt,indent=2))


@pytest.mark.parametrize("loaded",[False,True])
def test_scale_cost_keeps_cached_lookup_resident_and_separates_real_cold_load(monkeypatch,loaded):
    from Libs.utils.evaluation import native_scale_routing as routing
    clock=iter([10.,12.])
    monkeypatch.setattr(routing.time,"perf_counter",lambda:next(clock))
    def forecast(stage,event):
        event.update(model_load_seconds=.5,checkpoint_loaded=loaded)
        return {"loss":torch.zeros(2,1,2)}
    trace=[]
    execute_scale_route("fixed_threshold",policy(),torch.tensor([.5,.5]),forecast,trace)
    assert len(trace)==1
    assert trace[0]["cold_checkpoint_load_seconds"]==(.5 if loaded else 0.)
    assert trace[0]["resident_query_seconds"]==(1.5 if loaded else 2.)
    assert trace[0]["stage_elapsed_seconds"]==trace[0]["resident_query_seconds"]+trace[0]["cold_checkpoint_load_seconds"]
