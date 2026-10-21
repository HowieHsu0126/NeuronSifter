"""Disposable numerical checks with source IDs/seeds distinct from production."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data.native_observable_profile import quantity_support
from Libs.utils.data.native_stimulation import FORCING_ARRAY_NAMES
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.model import native_observable_source as source


def fixture(family):
    config = load_config("Libs/configs/exp/native_observable_bridge.yaml")
    config["native_observable_bridge"].update(seed=19371, source_uid="disposable_observable_fixture",
                                               snapshot_ref="disposable_observable_fixture_definition")
    group = next(g for g in source.group_definitions(config) if g["family_uid"] == family)
    generated = source.draw_group(config, group, 0)
    return config, group, generated


@pytest.mark.parametrize("family", [source.ELECTRICAL, source.SYNAPTIC, source.MOLECULAR])
def test_fresh_group_programs_precision_and_coupled_balance(family):
    config, group, generated = fixture(family)
    support = generated["support"]
    assert support.initial_field.shape == (1, 4, 6, 6, 26)
    assert set(support.factors) == {"z_tx", "z_pk", "z_env", "z_ephys", "z_morph"}
    assert len(set(generated["rng_streams"].values())) == 4
    actions = [source.action_definition(config, family, i) for i in range(2)]
    records = []
    for atom in range(4):
        row = []
        for action in actions:
            estimate = source.estimated_exposure(config, generated["parameters"][atom], action) if family == source.MOLECULAR else None
            row.append(source.compiled_program(config, group, support, atom,
                generated["parameters"][atom], action, "0"*64, estimate))
        records.append(row)
    raw = source.joint_source(config, group, generated, records)
    native.parse_joint_source(raw)
    posterior = SimpleNamespace(posterior_uid=group["posterior_uid"], support=support,
        weights=support.prior_weights, particle_context=torch.zeros(1, 4, 6, 6, 12))
    bound = native.bind_joint_programs(posterior, raw)
    contexts = native.joint_particle_contexts(posterior, bound, raw["particle_programs"]["regimen_uids"])
    assert len(contexts) == 4
    assert contexts[0]["quantity_support"].dtype == torch.bool
    assert all(name in context for name in FORCING_ARRAY_NAMES for context in contexts)
    truth = generated["hidden_index"]
    primary = source.physical_trajectory(config, family, generated["parameters"][truth], actions[0])
    precise = source.physical_trajectory(config, family, generated["parameters"][truth], actions[0], high_precision=True)
    numeric, serialization = source.physical_errors(primary, precise)
    assert np.isfinite(numeric).all() and numeric.max() < 1e-11
    assert np.isfinite(serialization).all() and serialization.max() > 0
    law = source.CoupledObservableLaw(config, group, support, truth, primary)
    coarse = source.integrate_coupled(law, "RK4", .25)
    fine = source.integrate_coupled(law, "RK4", .125)
    reference = source.integrate_coupled(law, "DOP853", .0625)
    assert fine["nfev"] > coarse["nfev"]
    assert reference["balance_max"] < 1e-9
    assert reference["drug_balance_max"] < 1e-9
    np.testing.assert_array_equal(reference["trajectory"][0].astype(np.float32), support.initial_field[0, truth].numpy())
    mask = quantity_support(law.grid)
    assert (reference["trajectory"][~mask] == 0).all()
    if family != source.MOLECULAR:
        assert (reference["trajectory"][..., :2] == 0).all()
    else:
        np.testing.assert_allclose(reference["trajectory"][:, 3, 4, 0], primary["values"][:, 0]*1e-9)
        assert np.allclose(reference["drug_mass"]-reference["drug_mass"][0],
            reference["boundary_mass_input"]-reference["boundary_mass_input"][0]
            - reference["eliminated_mass"]+reference["eliminated_mass"][0], atol=1e-15)
        estimated = source.estimated_exposure(config, generated["parameters"][truth], actions[0])
        changed = dict(generated["parameters"][truth], feedback_rate=1e6)
        np.testing.assert_array_equal(estimated["exposure_nmol_per_l"],
            source.estimated_exposure(config, changed, actions[0])["exposure_nmol_per_l"])


def test_future_physical_endpoint_does_not_drive_past_cns():
    config, group, generated = fixture(source.ELECTRICAL)
    truth = generated["hidden_index"]
    action = source.action_definition(config, source.ELECTRICAL, 0)
    physical = source.physical_trajectory(config, source.ELECTRICAL, generated["parameters"][truth], action)
    first = source.integrate_coupled(source.CoupledObservableLaw(config, group, generated["support"], truth, physical), "RK4", .25)
    changed = copy.deepcopy(physical)
    changed["values"][-1, 1:] += 100.
    second = source.integrate_coupled(source.CoupledObservableLaw(config, group, generated["support"], truth, changed), "RK4", .25)
    np.testing.assert_array_equal(first["trajectory"][..., :15], second["trajectory"][..., :15])


def test_declared_target_domain_rejects_missing_site_and_legacy_gap():
    from Libs.utils.data.physical_occupancy import compile_physical_record, digest
    config, group, generated = fixture(source.MOLECULAR)
    action = source.action_definition(config, source.MOLECULAR, 0)
    parameters = generated["parameters"][0]
    record = source.compiled_program(config, group, generated["support"], 0, parameters, action,
        "0"*64, source.estimated_exposure(config, parameters, action))
    inputs = record["physical_program"]["inputs"]
    support = np.asarray(record["physical_program"]["support"])
    assert support[0].sum() == 1
    assert (np.asarray(record["physical_program"]["occupancy"])[support == 0] == 0).all()
    for mutation in ("missing_declared_site", "legacy_gap", "ambiguous_zero"):
        bad = copy.deepcopy(inputs)
        if mutation == "missing_declared_site":
            bad["binding"]["target_site_domain"]["mask"][0][0] = True
            bad["run_snapshot"]["target_site_domain"] = copy.deepcopy(bad["binding"]["target_site_domain"])
        elif mutation == "legacy_gap":
            del bad["binding"]["target_site_domain"]
            del bad["run_snapshot"]["target_site_domain"]
        else:
            bad["binding"]["target_site_domain"]["excluded_site_meaning"] = "measured_zero_concentration"
            bad["run_snapshot"]["target_site_domain"] = copy.deepcopy(bad["binding"]["target_site_domain"])
        bad["provenance"]["run_snapshot_sha256"] = digest(bad["run_snapshot"])
        with pytest.raises(ValueError, match="coverage|semantics"):
            compile_physical_record(bad)


def build_disposable_observable_fixture(tmp_path, monkeypatch, *, scientific=False):
    import json
    from Libs.scripts.builders import build_controlled_native_source as builder
    from Libs.utils.data.physical_occupancy import digest
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml" if scientific
                         else "Libs/configs/exp/native_observable_bridge.yaml")
    settings = config["native_observable_bridge"]
    settings.update(seed=39117, source_uid="disposable_builder_fixture",
        snapshot_ref="disposable_builder_fixture_definition", output_root=str(tmp_path/"source"),
        group_roles=["train","development","calibration","calibration"], calibration_reservations={2:"fit",3:"check"})
    settings["budget"].update(groups=12, cases=24, physical=24, physical_precision=24,
        cns_integrations=72, programs=96, chemical_precision=32, operations=260)
    if scientific:
        from Libs.utils.model import native_scientific_source
        settings["source_contract_status"] = "frozen"
        settings["atoms"] = [[0,0,0,0,0],[0,1,1,0,1],[1,0,1,1,0],[1,1,0,1,1]]
        settings["prior_weights"] = [.1,.2,.3,.4]
        settings["source_variants"]["primary"].update(source_uid=settings["source_uid"],snapshot_ref=settings["snapshot_ref"])
        settings["source_variants"]["held"].update(source_uid="disposable_held_builder",snapshot_ref="disposable_held_builder_definition")
        settings["training_source_strata"].update(
            cycle=["shared_primary", "romani"], replicate_pair_size=2)
        config["native_romani_source"]["prior_particle_count"] = len(settings["atoms"])
        config["manuscript_native_protocol"]["source_comparison"].update(
            electrical_groups_per_arm=2,
            normalization_root=str(tmp_path/"source"/"physical_parameter_conditioning"/"source_scope_normalization"))
        settings["group_partitions"] = [
            {"partition_uid":"train","role":"train","groups_per_family":4,"action_indices":[0,1],"source_variant":"primary"},
            {"partition_uid":"development","role":"development","groups_per_family":1,"action_indices":[0,1],"source_variant":"primary"},
            {"partition_uid":"calibration_fit","role":"calibration","calibration_reservation":"fit","groups_per_family":1,"action_indices":[0,1],"source_variant":"primary"},
            {"partition_uid":"calibration_check","role":"calibration","calibration_reservation":"check","groups_per_family":1,"action_indices":[0,1],"source_variant":"primary"},
            {"partition_uid":"held","role":"validation","groups_per_family":1,"action_indices":[0,1],"source_variant":"held"}]
        settings["budget"].update(native_scientific_source.planned_operation_counts(config),seconds=1800)
    expected_cases = settings["budget"]["cases"]
    expected_groups = settings["budget"]["groups"]
    expected_families = 4 if scientific else 3
    identity = {"producer_commit":"disposable-software-fixture","source_code_sha256":"0"*64,
        "definition_sha256":digest({"profile":settings,"background":config["controlled_cns_source"]}),"bound_files":{}}
    monkeypatch.setattr(builder,"observable_identity",lambda cfg,path: identity)
    def verify(receipt, cfg):
        assert cfg["native_observable_bridge"]["source_uid"] == "disposable_builder_fixture"
        assert receipt["source_identity"] == identity
    monkeypatch.setattr(builder,"verify_observable_source",verify)
    path=tmp_path/"fixture.yaml"
    import yaml
    path.write_text(yaml.safe_dump(config))
    root=builder.execute_observable_source(config,path)
    original=(root/"source_receipt.json").read_bytes()
    receipt=json.loads(original)
    if receipt["source_status"] != "complete":
        incomplete_cases = [case for case in receipt["cases"] if case["status"] != "complete"]
        incomplete_operations = [
            (uid, operation) for uid, operation in receipt["operations"].items()
            if operation["status"] != "complete"
        ]
        diagnostics = {
            "reason_present": "reason" in receipt,
            "reason": receipt["reason"] if "reason" in receipt else None,
            "source_status": receipt["source_status"],
            "incomplete_case_count": len(incomplete_cases),
            "incomplete_operation_count": len(incomplete_operations),
        }
        if incomplete_cases:
            diagnostics["first_incomplete_case"] = {
                key: incomplete_cases[0][key]
                for key in ("case_index", "group_index", "case_uid", "family_uid", "status", "reason")
                if key in incomplete_cases[0]
            }
        if incomplete_operations:
            uid, operation = incomplete_operations[0]
            diagnostics["first_incomplete_operation"] = {
                "uid": uid, **{key: operation[key] for key in ("status", "reason") if key in operation}
            }
        pytest.fail("Disposable observable source did not complete: "
                    + json.dumps(diagnostics), pytrace=False)
    assert len(receipt["operations"]) == settings["budget"]["operations"]
    assert all(op["status"]=="complete" for op in receipt["operations"].values()), receipt
    assert all(not case["numerical"]["reasons"] for case in receipt["cases"]), receipt["cases"]
    if scientific:
        from Libs.utils.data.native_joint_source_storage import read_joint_source_reference
        for group in receipt["groups"]:
            reference = group["conditioning_reference"]
            assert Path(reference["path"]).is_relative_to(root)
            assert read_joint_source_reference(reference)[0]["support"]["sample_uids"] == [group["sample_uid"]]
        assert not Path(json.loads((root/"source_reservation.json").read_text())["node_local_path"]).exists()
    report=builder.consume_observable_source(config,path)
    assert report["status"]=="complete" and report["admitted_count"]==expected_cases, report
    artifact=json.loads((root/"train_normalization.json").read_text())
    assert artifact["trajectory_count"] == (32 if scientific else 2*expected_families)
    assert len(artifact["registered_grids"]) == expected_families
    # Every role remains visible; only TRAIN cases enter the shared statistics.
    effective_registry=load_config(str(root/"native_config.yaml"))
    for family,registry in native.load_family_registries(effective_registry).items():
        selected=copy.deepcopy(effective_registry)
        selected["native_diffusion_data"]["source_family_uid"]=family
        role_names = ("train","development","calibration","validation") if scientific else ("train","development","calibration")
        roles={role:native.select_records(registry,selected,role) for role in role_names}
        expected_roles = {"train":8 if scientific else 2,"development":2,"calibration":4}
        if scientific:
            expected_roles["validation"] = 2
        assert {role:len(rows) for role,rows in roles.items()}==expected_roles
        assert [row["teacher"]["calibration_reservation"] for row in roles["calibration"]].count("fit")==2
        assert [row["teacher"]["calibration_reservation"] for row in roles["calibration"]].count("check")==2
        samples={role:{row["teacher"]["sample_uid"] for row in rows} for role,rows in roles.items()}
        assert not (samples["train"] & (samples["development"] | samples["calibration"]))
        assert not (samples["development"] & samples["calibration"])
        with pytest.raises(ValueError):
            native.select_records(registry,selected,None)
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    effective=load_config(str(root/"native_config.yaml"))
    from Libs.utils.data.joint_physical_parameters import CONDITIONING_KIND
    effective["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"][
        "physical_parameter_conditioning"] = CONDITIONING_KIND
    effective["native_diffusion_data"]["registry_path"] = str(root/"physical_parameter_conditioning"/"model_registry.json")
    original_consumption = (root/"native_consumption.json").read_bytes()
    original_normalization = (root/"train_normalization.json").read_bytes()
    original_model_index = (root/"model_registry.json").read_bytes()
    for name in ("draw_group", "physical_trajectory", "compiled_program", "estimated_exposure", "integrate_coupled"):
        monkeypatch.setattr(source, name, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("source/program replay")))
    if scientific:
        for name in ("draw_group","physical_trajectory","compiled_program","estimated_exposure","integrate_coupled","long_cns_law","integrate_long_reference"):
            monkeypatch.setattr(native_scientific_source,name,lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError("scientific source/program replay")))
    repaired = builder.consume_observable_parameter_conditioning(effective, path)
    assert repaired["status"] == "complete" and repaired["admitted_count"] == expected_cases
    assert repaired["source_operations"] == repaired["model_initializations"] == 0
    assert repaired["array_values_unchanged"] and repaired["compiled_programs_unchanged"]
    assert len(repaired["groups"]) == expected_groups
    assert (root/"native_consumption.json").read_bytes() == original_consumption
    assert (root/"model_registry.json").read_bytes() == original_model_index
    assert (root/"train_normalization.json").read_bytes() == original_normalization
    return config, path, root, effective, original


def test_builder_source_admission_normalization_and_retained_consumption(tmp_path, monkeypatch):
    from Libs.scripts.builders import build_controlled_native_source as builder
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    config, path, root, effective, original = build_disposable_observable_fixture(tmp_path, monkeypatch)
    # Reduce dimensions only; preserve the complete configured ten-expert architecture.
    backbone=effective["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    backbone.update(operator_hidden_dim=12,operator_depth=1,attention_head_count=3,mechanism_state_token_count=6)
    effective["native_diffusion_data"]["posterior_context_width"]=12
    effective, diagnostic_binding = pretrain.observable_consumer_check_config(effective)
    assert diagnostic_binding["scientific_objective_qualification"] is False
    torch.manual_seed(3901)
    model=pretrain.MicroenvironmentPretrainingModel(effective,False)
    assert model.field_normalization.active
    for family in (source.ELECTRICAL,source.SYNAPTIC,source.MOLECULAR):
        model.zero_grad(set_to_none=True)
        consumed=pretrain.observable_consumer_family(model,effective,family)
        assert consumed["particle_count"]==4 and consumed["loss_backward_count"]==1
        assert len(consumed["regimen_uids"])==2
        readout=consumed["observable_prediction"]
        assert len(readout["trajectory_rows"])==2
        assert list(readout["trajectory_rows"][0][0])==readout["feature_names"]
        assert len(readout["trajectory_rows"][0])==len(readout["time_values"])
        assert consumed["observable_prediction"]["time_unit"]==("hour" if family==source.MOLECULAR else "ms")
        check_screening_serving_evaluation_cache(effective, family, tmp_path, monkeypatch)
    assert (root/"source_receipt.json").read_bytes()==original
    monkeypatch.setattr(source,"draw_group",lambda *args: (_ for _ in ()).throw(AssertionError("source replay")))
    reused=builder.consume_observable_source(config,path,reuse_completed=True)
    assert reused["status"]=="complete"
    assert (root/"source_receipt.json").read_bytes()==original


def test_observable_operation_ledger_consumes_failure_once(tmp_path):
    from Libs.scripts.builders.build_controlled_native_source import ObservableSourceLedger
    config=load_config("Libs/configs/exp/native_observable_bridge.yaml")
    ledger=ObservableSourceLedger(tmp_path/"receipt.json",config,{"fixture":True})
    assert len(ledger.data["groups"])==18 and len(ledger.data["cases"])==36
    assert len(ledger.data["operations"])==390
    calls=[]
    def fail():
        calls.append("once")
        raise ValueError("deliberate fixture failure")
    with pytest.raises(ValueError,match="deliberate fixture"):
        ledger.execute("group:0",fail)
    with pytest.raises(RuntimeError,match="already consumed"):
        ledger.execute("group:0",fail)
    assert calls==["once"] and ledger.data["operations"]["group:0"]["status"]=="failed"


def check_screening_serving_evaluation_cache(config, family, tmp_path, monkeypatch):
    import json
    import pandas as pd
    from Libs.exp.training import train_screening_system as training
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    from Libs.exp.evaluation import evaluate_screening_system as evaluation
    from Libs.models.networks.screening_system import ADScreeningSystem
    from Libs.scripts.services import serve_screening_api as service
    from Libs.utils.data.native_field_normalization import normalization_from_config
    from Libs.utils.model.screening_model_config import screening_model_kwargs_from_config
    from Libs.utils.model.screening_core import TRI_COMPONENT_NAMES
    from Libs.utils.evaluation.native_rollout_metrics import score_native_episode
    selected=copy.deepcopy(config)
    selected["native_diffusion_data"]["source_family_uid"]=family
    # A disposable structural prior supplies the ancillary screening input; it is
    # distinct from the independently generated complete native particle prior.
    from Libs.utils.data.structural_prior_registry import structural_prior_root
    selected["structural_prior_registry"]["local_root"]=str(tmp_path/"structural-prior-fixture")
    prior_path=structural_prior_root(selected)/selected["structural_prior_registry"]["relative_paths"]["region_prior_catalog"]
    prior_path.parent.mkdir(parents=True,exist_ok=True)
    regions=config["native_observable_bridge"]["quantity_profile"]["region_id"]
    pd.DataFrame([{"region_id":uid,
        "disease_program":selected["screening_pipeline"]["screening_benchmark"]["active_disease_program"],
        **{name:float(i+1)/len(regions) for name in selected["structural_prior_registry"]["region_feature_names"]}}
        for i,uid in enumerate(regions)]).to_csv(prior_path,sep="\t",index=False)
    registry=native.load_registry(selected)
    records=native.select_records(registry,selected,"train")
    sample=records[0]["teacher"]["sample_uid"]
    regimens=[row["teacher"]["regimen_uid"] for row in records]
    cached=native.screening_native_payload(selected,"train",[sample],regimens)
    payload={"selected_target_rows":[{"target_program_uid":"fixture-target"}],
        "selected_regimen_rows":[{"regimen_uid":uid} for uid in regimens],
        "modality_payload":{"fixture":{"train":{"sample_uids":[sample],"state_features":np.zeros((1,6),dtype=np.float32),
            "labels":[1],"native_diffusion":cached}}},
        "label_payload":{"fixture":{"train":{"target_labels":np.ones((1,1),dtype=np.float32),
            "regimen_labels":np.ones((1,2),dtype=np.float32)}}}}
    with monkeypatch.context() as patch:
        patch.setattr(training,"source_artifact_hashes",lambda *args:{})
        patch.setattr(training,"freeze_signature_sha256",lambda *args:"disposable-fixture")
        cache=tmp_path/("screening-cache-"+family)
        training.write_scope_training_cache(selected,tmp_path/"fixture.yaml",["disposable-fixture"],cache,"fixture","train",3902,payload,0.)
        restored,_,_=training.load_scope_training_cache(selected,tmp_path/"fixture.yaml",cache,"fixture","train",3902)
    item=restored["modality_payload"]["fixture"]["train"]
    assert item["native_diffusion"]["quantity_support"].dtype==np.bool_
    assert np.array_equal(item["native_diffusion"]["quantity_support"],cached["quantity_support"])
    assert item["native_diffusion"]["joint_source"]==cached["joint_source"]
    named = cached["joint_source"]["support"].get("physical_parameter_hypotheses")
    if named is not None:
        assert len(named["values"][0]) == 4
        changed = copy.deepcopy(item)
        changed["native_diffusion"]["joint_source"]["support"]["physical_parameter_hypotheses"]["values"][0][0][0] += .01
        with pytest.raises(ValueError, match="cache_stale"):
            native.validate_cached_native({"fixture":{"train":changed}}, regimens, selected)
    kwargs=screening_model_kwargs_from_config(selected)
    kwargs.update(training.rollout_model_kwargs(selected))
    groups=kwargs["rollout_update_policy"]["action_representation"]["feature_groups"]
    layout=kwargs["solver_config"]["regimen_feature_layout"]
    width=max([index for indices in groups.values() for index in indices]+list(layout.values()))+1
    model=ADScreeningSystem(patient_state_dim=6,target_feature_dim=3,regimen_feature_dim=width,
        hidden_dim=12,latent_dim=selected["screening_pipeline"]["screening_benchmark"]["training"]["latent_dim"],
        dropout_rate=0.,compartment_count=6,tri_weight_by_component={key:1. for key in TRI_COMPONENT_NAMES},
        adjacency_matrix=torch.as_tensor(pretrain.node_adjacency_matrix(selected)).tolist(),
        field_normalization_config=normalization_from_config(selected),**kwargs).eval()
    features=torch.arange(2*width,dtype=torch.float32).reshape(2,width)/10
    with torch.no_grad():
        native_kwargs=training.native_diffusion_training_kwargs(model,item,slice(None),slice(None),torch.device("cpu"))
        patient=native_kwargs["joint_posterior"].observation_features
        complete=model.score_regimens(patient,features,**native_kwargs)
        chunks=model.score_regimens(patient,features,**native_kwargs,regimen_chunk_size=1,include_auxiliary_outputs=False)
        torch.testing.assert_close(chunks["field_trajectory"],complete["field_trajectory"],rtol=2e-5,atol=2e-5)
        assert chunks["quantity_support"].dtype==torch.bool
        torch.testing.assert_close(chunks["observable_prediction"]["predictive_mean"],complete["observable_prediction"]["predictive_mean"],rtol=2e-5,atol=2e-5)
        assert (chunks["field_trajectory"][~chunks["quantity_support"]]==0).all()
        arrays,_=native.load_records(registry,records,selected,include_teacher=False)
        assert "trajectory" not in arrays and "quantity_support" not in arrays
        prepared=evaluation.prepare_native_episode(model,records,selected,arrays)
        prediction=evaluation.predict_native_episode(model,records,selected,arrays,features,prepared=prepared)
        mask=evaluation._native_validity_mask(records,registry["grid"])
        protocol={"draw_semantics":"emitted_whole_particle_mixture_draws","quantile_levels":[.1,.5,.9],"quantile_interpolation":"linear"}
        score=score_native_episode(prediction["field_trajectory_samples"][:,0],
            torch.from_numpy(cached["teacher_trajectory"][0]),mask,{"regimen_uids":regimens},
            model.field_rollout_operator.field_normalization,registry["grid"],protocol)
        assert score["status"]=="defined" and score["valid_cell_count"]==int(mask.sum())
        assert score["valid_cell_count"]<score["total_cell_count"]
        grid=registry["grid"]
        frame=pd.DataFrame([{"regimen_uid":uid,"target_symbol_set":"fixture","route":"intrathecal",
            "schedule":"daily","dose_level_id":"fixture"} for uid in regimens])
        bundle={"screening_model":model,"regimen_feature_matrix":features,"regimen_feature_frame":frame,
            "regimen_index_by_uid":{uid:i for i,uid in enumerate(regimens)},
            "region_catalog":pd.DataFrame({"region_id":grid["region_id"],"region_label":grid["region_id"]}),
            "species_catalog":pd.DataFrame({"species_id":grid["species_id"]}),
            "region_prior_catalog":pd.DataFrame([{"disease_program":"ad_control","region_id":uid,
                "program_support_score":.5,"disease_vulnerability_score":.5,"transport_centrality_score":.5}
                for uid in grid["region_id"]]),"preprocessing_manifest":{"active_disease_program":"ad_control"}}
        response=service.rollout_fields(bundle,{"joint_source":cached["joint_source"],"regimen_uids":regimens})
        json.dumps(response,allow_nan=False)
        readout=response["observable_prediction_sequence"]
        assert readout["time_values"]==grid["query_time"]["values"]
        assert readout["time_unit"]==grid["query_time"]["unit"]
        assert len(readout["predictive_mean"][0])==2
        # Use one complete fixed particle as a point-context admission fixture.
        contexts=native.joint_particle_contexts(native_kwargs["joint_posterior"],native_kwargs["particle_programs"],regimens)
        point={key:value.tolist() for key,value in contexts[0].items()}
        point.update(source_family_uid=family,sample_uids=[sample],regimen_uids=regimens,
                     programs=[cached["joint_source"]["particle_programs"]["records"][0][0]])
        stub=SimpleNamespace(field_rollout_operator=model.field_rollout_operator,
            world_state_posterior=SimpleNamespace(family="deterministic_compatibility"))
        point_bundle=dict(bundle,screening_model=stub)
        admitted=service.native_diffusion_serving_kwargs(point_bundle,{"native_diffusion_context":point},regimens)
        assert admitted["diffusion_context"]["quantity_support"].dtype==torch.bool
        wrong=copy.deepcopy(point);wrong["quantity_support"]=np.asarray(wrong["quantity_support"],dtype=float).tolist()
        with pytest.raises(ValueError,match="boolean"):
            service.native_diffusion_serving_kwargs(point_bundle,{"native_diffusion_context":wrong},regimens)
        wrong=copy.deepcopy(point);wrong[FORCING_ARRAY_NAMES[0]][0][0][0][0][0][0]+=1
        with pytest.raises(ValueError,match="numerical program mismatch"):
            service.native_diffusion_serving_kwargs(point_bundle,{"native_diffusion_context":wrong},regimens)
