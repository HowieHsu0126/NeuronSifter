"""Labeled OSP source-format fixtures; no biological PK or calibration evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from Libs.scripts.teacher_generation.build_osp_pbpk_teacher_rollouts import build_native_physical_source
from Libs.tests.test_joint_world_posterior import joint_case, joint_registry_case, raw_joint_source
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data.physical_occupancy import (
    compile_physical_record, digest, equilibrium_fraction, molar, physical_arrays, whole_particle_basis,
)
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.model.dynamics_fields import region_id_list, compartment_id_list
from Libs.utils.model.microenvironment_teacher import write_native_microenvironment_store

ROOT = Path(__file__).resolve().parents[2]


def make_packet(directory, config, source):
    directory.mkdir(parents=True, exist_ok=True)
    support = source["support"]
    regions, compartments = region_id_list(config), compartment_id_list(config)
    actions = [{"regimen_uid": uid, "action": copy.deepcopy(source["particle_programs"]["records"][0][0][ai]["action"])}
               for ai, uid in enumerate(source["particle_programs"]["regimen_uids"])]
    packet = {"design_id": "labeled-osp-source-format-fixture", "source_posterior_uid": source["source_posterior_uid"],
              "support": support, "observations": source["observations"], "region_ids": regions,
              "compartment_ids": compartments, "time_grid_hours": [0., 1., 2.],
              "graph_ref": "declared-fixture-grid", "organism_context": "fixture", "actions": actions, "exports": []}
    model = directory / "fixture.pkml"
    model.write_text("<Simulation name='source-format-fixture-only'/>")
    for bi, sample in enumerate(support["sample_uids"]):
        for ji, particle in enumerate(support["particle_uids"][bi]):
            for ai, action_entry in enumerate(actions):
                regimen, action = action_entry["regimen_uid"], action_entry["action"]
                prefix = f"{bi}-{ji}-{ai}"
                csv, snapshot_path = directory / (prefix+".csv"), directory / (prefix+".json")
                binding = {"source_uid": support["source_uid"], "source_snapshot_ref": support["source_snapshot_ref"],
                    "source_posterior_uid": source["source_posterior_uid"], "support_uid": support["support_uid"],
                    "sample_uid": sample, "particle_uid": particle, "regimen_uid": regimen,
                    "binding_law": "single_ligand_equilibrium", "requested_readout": "fractional_occupancy",
                    "interpolation": "exact", "particle_basis": whole_particle_basis(support, bi, ji),
                    "action_setting_paths": {key:["settings",key] for key in
                        ("dose","dose_unit","route","schedule","duration","agent_set")},
                    "particle_parameters": {"transport": {"source_path":["parameters","transport"],
                         "value":{"value":float(2**ji),"unit":"fixture_transport"}}}, "site_mappings":[]}
                snapshot = {key:binding[key] for key in ("source_uid","source_snapshot_ref","source_posterior_uid",
                    "support_uid","sample_uid","particle_uid","regimen_uid","particle_basis")}
                snapshot.update(settings_origin="labeled_source_format_fixture", source_model_id="fixture-osp",
                    source_run_id=prefix, model_path=model.name, model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                    settings={key:action[key] for key in binding["action_setting_paths"]},
                    parameters={key:value["value"] for key,value in binding["particle_parameters"].items()},
                    particle_parameter_paths={key:value["source_path"] for key,value in binding["particle_parameters"].items()},
                    quantities={}, affinities={})
                rows = []
                for ri, region in enumerate(regions):
                    for ci, compartment in enumerate(compartments):
                        site = f"{region}:{compartment}"
                        path = f"Fixture|{region}|{compartment}|Drug"
                        snapshot["quantities"][path] = {"site_uid":site, "ligand_uid":action["agent_set"][0],
                            "target_uid":action["target_set"][0], "quantity_kind":"free_concentration",
                            "concentration_unit":"nmol/L", "time_unit":"min"}
                        snapshot["affinities"][site] = {"value":10*1.5**ji, "unit":"nmol/L", "source_ref":"labeled-source-Kd",
                            "quantity_kind":"equilibrium_dissociation_constant", "binding_scope":"particle_parameter",
                            "source_uid":support["source_uid"], "source_snapshot_ref":support["source_snapshot_ref"],
                            "particle_uid":particle, "site_uid":site, "ligand_uid":action["agent_set"][0],
                            "target_uid":action["target_set"][0]}
                        parameter_name = "affinity:" + site
                        binding["particle_parameters"][parameter_name] = {"source_path":["affinities",site],
                            "value":copy.deepcopy(snapshot["affinities"][site])}
                        snapshot["particle_parameter_paths"][parameter_name] = ["affinities",site]
                        binding["site_mappings"].append({"region_id":region, "compartment_id":compartment,
                            "site_uid":site, "source_output_path":path, "kd_source_path":["affinities",site],
                            "kd_binding":{"kind":"particle_parameter","parameter_name":parameter_name}})
                        values = [0.,10.,1e8] if ai == 0 else [10.,20.,5.]
                        for time,value in zip([0.,60.,120.],values):
                            rows.append({"paths":path,"Time":time,"TimeUnit":"min",
                                "simulationValues":value*2**ji,"Unit":"nmol/L"})
                pd.DataFrame(rows).to_csv(csv,index=False)
                snapshot["export_sha256"] = hashlib.sha256(csv.read_bytes()).hexdigest()
                snapshot_path.write_text(json.dumps(snapshot))
                packet["exports"].append({"sample_uid":sample,"particle_uid":particle,"regimen_uid":regimen,
                    "csv_path":csv.name,"run_snapshot_path":snapshot_path.name,"binding":binding})
    return packet


@pytest.fixture(scope="module")
def physical_case(tmp_path_factory):
    directory = tmp_path_factory.mktemp("physical_source")
    model, support, production, regimens, _ = joint_case()
    config = load_config("Libs/configs/config_stack.yaml")
    config["spatial_atlas_registry"]["regions"] = config["spatial_atlas_registry"]["regions"][:3]
    config["screening_pipeline"]["pkpd_ready_registry"]["compartments"] = config["screening_pipeline"]["pkpd_ready_registry"]["compartments"][:2]
    packet = make_packet(directory, config, raw_joint_source(support, production))
    source = build_native_physical_source(config, packet, directory)
    return directory, model, support, regimens, config, packet, source


def input_copy(case):
    return copy.deepcopy(case[-1]["particle_programs"]["records"][0][0][0]["physical_program"]["inputs"])


def reseal(inputs):
    inputs["provenance"]["run_snapshot_sha256"] = digest(inputs["run_snapshot"])
    return inputs


def test_actual_r_export_and_numeric_reference(physical_case):
    source = physical_case[-1]
    record = source["particle_programs"]["records"][0][0][0]
    physical = record["physical_program"]
    rows = physical["inputs"]["source_rows"]
    assert {row["time_unit"] for row in rows} == {"min"}
    assert {row["concentration_unit"] for row in rows} == {"nmol/L"}
    assert {row["source_run_id"] for row in rows} == {"0-0-0"}
    assert {row["source_output_path"] for row in rows} == set(physical["inputs"]["run_snapshot"]["quantities"])
    values = np.asarray(physical["occupancy"])[:,0,0,0]
    np.testing.assert_allclose(values[:2], [0.,.5])
    assert values[2] > .999999
    assert physical["target_parameters"][0]["target_uid"] == record["action"]["target_set"][0]
    assert record["pkpd_context"]["parameter_provenance"]["calibrated"] is False
    assert not {"teacher_confidence","uncertainty"} & set(record["pkpd_context"])
    native.parse_joint_source(json.loads(json.dumps(source,allow_nan=False)))


def test_equivalent_units_absolute_scale_and_timing(physical_case):
    original = input_copy(physical_case)
    scaled = copy.deepcopy(original)
    for quantity in scaled["run_snapshot"]["quantities"].values():
        quantity["concentration_unit"] = "umol/L"
    for row in scaled["source_rows"]:
        row["concentration_unit"] = "umol/L"
        row["concentration"] /= 1000
    np.testing.assert_allclose(physical_arrays(reseal(scaled))["occupancy"], physical_arrays(original)["occupancy"])
    for row in scaled["source_rows"]:
        row["concentration"] *= 2
    assert np.asarray(physical_arrays(reseal(scaled))["occupancy"])[1,0,0,0] > .5
    curves = []
    for values in ([0.,20.,0.], [20.,0.,20.]):
        varied = copy.deepcopy(original)
        for row in varied["source_rows"]:
            row["concentration"] = values[[0.,60.,120.].index(row["source_time"])]
        curves.append(np.asarray(physical_arrays(varied)["occupancy"])[:,0,0,0])
        assert np.trapezoid(values,[0.,1.,2.]) == 20
    assert not np.array_equal(*curves)
    np.testing.assert_allclose(equilibrium_fraction([0.,1e308,1e308],[1.,1e308,1.]),[0.,.5,1.])


@pytest.mark.parametrize("bad", ["negative","nan","missing_unit","mass_without_mw","missing_kd","zero_kd",
    "missing_site","duplicate_time","uncovered_time","missing_exact_time","wrong_action","wrong_particle",
    "missing_parameter","multiple_targets","competition","bound_amount","total_without_fraction","broadcast"])
def test_reject_incomplete_or_unsupported_program(physical_case,bad):
    value = input_copy(physical_case)
    snapshot,binding = value["run_snapshot"],value["binding"]
    first = value["source_rows"][0]
    if bad == "negative": first["concentration"] = -1.
    if bad == "nan": first["concentration"] = float("nan")
    if bad in {"missing_unit","mass_without_mw"}:
        unit = "" if bad == "missing_unit" else "mg/L"
        for quantity in snapshot["quantities"].values(): quantity["concentration_unit"] = unit
        for row in value["source_rows"]: row["concentration_unit"] = unit
    if bad == "missing_kd": binding["site_mappings"][0]["kd_source_path"] = ["absent"]
    if bad == "zero_kd":
        site = next(iter(snapshot["affinities"]))
        snapshot["affinities"][site]["value"] = 0.
        binding["particle_parameters"]["affinity:"+site]["value"] = copy.deepcopy(snapshot["affinities"][site])
    if bad == "missing_site": binding["site_mappings"].pop()
    if bad == "duplicate_time": value["source_rows"].append(copy.deepcopy(first))
    if bad == "uncovered_time": value["time_grid_hours"][-1] = 3.
    if bad == "missing_exact_time": value["time_grid_hours"][1] = .5
    if bad == "wrong_action": value["action"]["dose"] *= 2
    if bad == "wrong_particle": snapshot["particle_uid"] = "unknown"
    if bad == "missing_parameter": binding["particle_parameters"].pop("transport")
    if bad == "multiple_targets": value["action"]["target_set"].append("other")
    if bad == "competition": binding["binding_law"] = "competitive"
    if bad == "bound_amount": binding["requested_readout"] = "bound_drug_concentration"
    if bad == "total_without_fraction":
        for quantity in snapshot["quantities"].values(): quantity["quantity_kind"] = "total_concentration"
        for row in value["source_rows"]: row["quantity_kind"] = "total_concentration"
    if bad == "broadcast": binding["site_mappings"][1]["source_output_path"] = binding["site_mappings"][0]["source_output_path"]
    # NaN cannot even enter strict JSON provenance.
    with pytest.raises(ValueError):
        physical_arrays(reseal(value))


def test_total_fraction_mass_and_interpolation(physical_case):
    inputs = input_copy(physical_case)
    for quantity in inputs["run_snapshot"]["quantities"].values(): quantity["quantity_kind"] = "total_concentration"
    for row in inputs["source_rows"]: row["quantity_kind"] = "total_concentration"
    inputs["run_snapshot"]["fractions"] = {}
    for mapping in inputs["binding"]["site_mappings"]:
        site = mapping["site_uid"]
        mapping["unbound_fraction_source_path"] = ["fractions",site]
        inputs["run_snapshot"]["fractions"][site] = {"value":.5,"site_uid":site,
            "particle_uid":inputs["binding"]["particle_uid"],"interpretation":"free_over_total_same_site","source_ref":"fixture"}
    inputs["binding"]["interpolation"] = "linear_inside_coverage"
    inputs["time_grid_hours"] = [0.,.5,2.]
    arrays = physical_arrays(reseal(inputs))
    assert np.asarray(arrays["occupancy"])[1,0,0,0] == pytest.approx(.2)
    np.testing.assert_allclose(molar([1.],"mg/L",{"value":1000.,"unit":"g/mol","source_ref":"fixture"}),[1e-6])
    inputs["run_snapshot"]["fractions"][site]["site_uid"] = "wrong"
    with pytest.raises(ValueError,match="site/particle"):
        physical_arrays(reseal(inputs))


def test_weights_particle_changes_reorder_and_numeric_tamper(physical_case):
    _,model,support,_,_,_,source = physical_case
    posterior = model.infer_world_state(joint_support=source["support"],observations=source["observations"])["joint_posterior"]
    programs = native.bind_joint_programs(posterior,source)
    changed = copy.deepcopy(source)
    changed["support"]["prior_weights"] = [[.9,.1]]
    updated = model.infer_world_state(joint_support=changed["support"],observations=changed["observations"])["joint_posterior"]
    rebound = native.bind_joint_programs(updated,changed)
    assert programs["records"] == rebound["records"]
    assert programs["occupancy"] == rebound["occupancy"]
    inputs = input_copy(physical_case)
    for site,kd in inputs["run_snapshot"]["affinities"].items():
        kd["value"] *= 2
        inputs["binding"]["particle_parameters"]["affinity:"+site]["value"] = copy.deepcopy(kd)
    record = compile_physical_record(reseal(inputs))
    assert record["physical_program"]["occupancy"] != programs["records"][0][0][0]["physical_program"]["occupancy"]
    altered = copy.deepcopy(source)
    altered["particle_programs"]["records"][0][0][0] = record
    altered["particle_programs"]["occupancy"][0][0][0] = record["physical_program"]["occupancy"]
    native.parse_joint_source(altered)
    assert altered["particle_programs"]["records"][0][1] == programs["records"][0][1]
    for row in inputs["source_rows"]: row["concentration"] *= 2
    assert compile_physical_record(inputs)["physical_program"]["occupancy"] != record["physical_program"]["occupancy"]
    selected = native.select_joint_source(source,list(support.sample_uids),["regimen-b","regimen-a"])
    assert selected["particle_programs"]["records"][0][0] == list(reversed(programs["records"][0][0]))
    tampered = copy.deepcopy(source)
    tampered["particle_programs"]["occupancy"][0][0][0][1][0][0][0] = .7
    with pytest.raises(ValueError,match="numeric array mismatch"):
        native.parse_joint_source(tampered)
    # Reorder all parts of each complete tuple, including retained physical programs.
    reordered = copy.deepcopy(source)
    raw = reordered["support"]
    for key in ("particle_uids","regime_ids","initial_field","physical_parameters","prior_weights"):
        raw[key] = [list(reversed(row)) for row in raw[key]]
    for key in ("factors","source_nuisance"):
        raw[key] = {name:[list(reversed(row)) for row in rows] for name,rows in raw[key].items()}
    for fields in raw["readout_parameters"].values():
        for key in set(fields)-{"law","units"}: fields[key] = [list(reversed(row)) for row in fields[key]]
    prog = reordered["particle_programs"]
    for key in ("particle_uids","records","occupancy","support","physical_time"):
        prog[key] = [list(reversed(row)) for row in prog[key]]
    native.parse_joint_source(reordered)
    assert prog["records"][0][0] == programs["records"][0][1]


def test_actual_screening_serving_and_sequential_continuity(physical_case,monkeypatch):
    from Libs.exp.training.train_screening_system import native_diffusion_training_kwargs
    from Libs.scripts.services.serve_screening_api import infer_world_state,native_diffusion_serving_kwargs
    _,model,support,regimens,_,_,source = physical_case
    batch = {"split_name":"train","sample_uids":list(support.sample_uids),"regimen_uids":["regimen-a","regimen-b"],
        "teacher_trajectory":torch.rand(1,2,3,3,2,7),"teacher_confidence":torch.ones(1,2),"joint_source":source}
    kwargs = native_diffusion_training_kwargs(model,{"labels":[1],"native_diffusion":batch},
        slice(None),[0,1],torch.device("cpu"),split_name="train")
    received, denoising = [], []
    operator = model.field_rollout_operator.quantized_mddm_rollout_operator
    original = operator.forward
    def capture(*args,**kw):
        received.append({key:kw["conditioning"][key].detach().clone()
                         for key in ("occupancy","support","physical_time")})
        return original(*args,**kw)
    monkeypatch.setattr(operator,"forward",capture)
    original_logits = operator.logits
    def capture_logits(coefficients,mask,u,context,*args):
        denoising.append({key:context[key].detach().clone() for key in ("occupancy","support","physical_time")})
        return original_logits(coefficients,mask,u,context,*args)
    monkeypatch.setattr(operator,"logits",capture_logits)
    result = model.score_regimens(torch.zeros(1,4),regimens,**kwargs)
    result["rollout_auxiliary_loss"].backward()
    assert len(received) == 2 and result["particle_uids"] == support.particle_uids
    programs = source["particle_programs"]
    for ji, context in enumerate(received):
        for key in context:
            assert torch.equal(context[key],torch.tensor(programs[key])[:,ji])
    assert len(denoising) >= 2 * 2 * operator.training_sample_count * operator.training_denoising_step_count
    for context in denoising:
        matches = [(ji,ai) for ji in range(2) for ai in range(2)
                   if torch.equal(context["occupancy"],torch.tensor(programs["occupancy"])[0,ji,ai])]
        assert len(matches) == 1
        ji,ai = matches[0]
        for key in context:
            assert torch.equal(context[key],torch.tensor(programs[key])[0,ji,ai])
    assert result["particle_trajectory_samples"].shape[2] == 2
    bundle = {"screening_model":model,"regimen_index_by_uid":{"regimen-a":0,"regimen-b":1}}
    json.dumps(infer_world_state(bundle,{"joint_source":source}),allow_nan=False)
    serving = native_diffusion_serving_kwargs(bundle,{"joint_source":source},["regimen-a","regimen-b"])
    assert serving["particle_programs"]["records"] == kwargs["particle_programs"]["records"]
    output = model.rollout(torch.zeros(1,4),regimens.unsqueeze(0),joint_posterior=serving["joint_posterior"],
        particle_programs=serving["particle_programs"],regimen_uids=["regimen-a","regimen-b"])
    fields = output["particle_field_trajectory_sequence"]
    assert torch.equal(fields[:,:,1,0],fields[:,:,0,-1])
    assert output["particle_uids"] == support.particle_uids


@pytest.mark.parametrize("store_format",["zarr_v2","neuronsifter_chunk_store"])
def test_physical_program_store_pretraining_and_cache(tmp_path,monkeypatch,store_format):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    from Libs.exp.training import train_screening_system as screening
    model,config,registry,arrays,legacy_source = joint_registry_case(tmp_path)
    packet = make_packet(tmp_path/"exports",config,legacy_source)
    source = build_native_physical_source(config,packet,tmp_path/"exports")
    registry["joint_sources"] = [source]
    store = tmp_path/"physical-store"
    for index,teacher in enumerate(registry["teachers"]):
        record = source["particle_programs"]["records"][index//2][0][index%2]
        teacher.update({name:record[name] for name in ("program_uid","posterior_uid","action_uid","target_program_uid")})
        teacher.update(trajectory_store_path=str(store),store_format=store_format)
    write_native_microenvironment_store(store,config,registry,arrays,store_format)
    Path(config["native_diffusion_data"]["registry_path"]).write_text(json.dumps(registry))
    restored,_ = native.load_records(registry,native.join_registry(registry,config),config)
    np.testing.assert_array_equal(restored["trajectory"],arrays["trajectory"])
    dataset = pretrain.MicroenvironmentTeacherFamilyDataset(config,"train")
    rows = dataset.__getitems__([3,0])
    batch = pretrain.collate_native_sources(rows)
    metrics,report = pretrain.run_pretraining_step(model,batch,config,torch.device("cpu"),
        {"gradient_accumulation_steps":1,"mixed_precision":{"enabled":False,"device_type":"cpu","dtype":"float32"}})
    assert np.isfinite(metrics["total_loss"])
    assert report["belief_evidence_status"]["calibration"] == "unverified"
    cached = native.screening_native_payload(config,"train",["sample-b","sample-a"],["regimen-b","regimen-a"])
    expected = source["particle_programs"]["records"][1][0][1]["physical_program"]
    assert cached["joint_source"]["particle_programs"]["records"][0][0][0]["physical_program"] == expected
    payload = {"selected_target_rows":[{"target_program_uid":"fixture-target"}],
        "selected_regimen_rows":[{"regimen_uid":uid} for uid in cached["regimen_uids"]],
        "modality_payload":{"fixture":{"train":{"sample_uids":cached["sample_uids"],
            "state_features":np.zeros((2,4),dtype=np.float32),"native_diffusion":cached}}},
        "label_payload":{"fixture":{"train":{"target_labels":np.ones((2,1),dtype=np.float32),
            "regimen_labels":np.ones((2,2),dtype=np.float32)}}}}
    monkeypatch.setattr(screening,"source_artifact_hashes",lambda *args:{})
    monkeypatch.setattr(screening,"freeze_signature_sha256",lambda *args:"controlled-fixture")
    config_path = tmp_path/"fixture-config.yaml"
    screening.write_scope_training_cache(config,config_path,["controlled-fixture"],tmp_path/"cache",
        "fixture","train",19,payload,0.)
    loaded,_,_ = screening.load_scope_training_cache(config,config_path,tmp_path/"cache","fixture","train",19)
    restored_native = loaded["modality_payload"]["fixture"]["train"]["native_diffusion"]
    assert restored_native["joint_source"] == cached["joint_source"]
    changed = copy.deepcopy(loaded["modality_payload"])
    changed["fixture"]["train"]["native_diffusion"]["joint_source"]["particle_programs"]["records"][0][0][0]["physical_program"]["inputs"]["provenance"]["export_sha256"] = "stale"
    with pytest.raises(ValueError,match="cache_stale"):
        native.validate_cached_native(changed,cached["regimen_uids"],config)


def test_legacy_standardizer_and_native_cli(physical_case,tmp_path):
    directory,_,_,_,config,packet,_ = physical_case
    source_csv = directory/packet["exports"][0]["csv_path"]
    output = tmp_path/"legacy.tsv"
    subprocess.run(["conda","run","--no-capture-output","-n","nrnsift_osp_pksim","Rscript",
        str(ROOT/"Libs/scripts/teacher_generation/run_osp_pksim_rollout.R"),"--standardize-csv-input",str(source_csv),
        "--output",str(output),"--csv-metadata-columns","IndividualId","--measure-column-keep-regex","Fixture",
        "--max-measure-columns-per-chunk","8","--standardization-policy-uid","fixture-legacy",
        "--plasma-regex","Fixture"],check=True)
    legacy = pd.read_csv(output,sep="\t")
    assert legacy["time_hours"].max() == 2.
    assert legacy["concentration"].max() == 1e8
    # Real producer CLI using an explicit complete YAML config, never current Input.
    import yaml
    configured = tmp_path/"config.yaml"
    configured.write_text(yaml.safe_dump(config))
    selected = directory/"selected.json"
    selected.write_text(json.dumps(packet))
    native_output = tmp_path/"native.json"
    subprocess.run([sys.executable,str(ROOT/"Libs/scripts/teacher_generation/build_osp_pbpk_teacher_rollouts.py"),
        "--config",str(configured),"--native-physical-input",str(selected),"--native-output",str(native_output)],check=True)
    native.parse_joint_source(json.loads(native_output.read_text()))


def test_r_wide_export_and_unit_rejection(physical_case,tmp_path):
    directory,_,_,_,_,packet,source = physical_case
    export = packet["exports"][0]
    original = pd.read_csv(directory/export["csv_path"])
    wide = original.pivot(index="Time",columns="paths",values="simulationValues").reset_index()
    wide_path = tmp_path/"wide.csv"
    wide.to_csv(wide_path,index=False)
    snapshot = json.loads((directory/export["run_snapshot_path"]).read_text())
    snapshot["csv_time_column"] = "Time"
    binding = tmp_path/"binding.json"
    binding.write_text(json.dumps(snapshot))
    output = tmp_path/"physical.tsv"
    command = ["conda","run","--no-capture-output","-n","nrnsift_osp_pksim","Rscript",
        str(ROOT/"Libs/scripts/teacher_generation/run_osp_pksim_rollout.R"),
        "--standardize-csv-input",str(wide_path),"--physical-binding",str(binding),"--output",str(output)]
    subprocess.run(command,check=True)
    inputs = input_copy(physical_case)
    inputs["source_rows"] = pd.read_csv(output,sep="\t").to_dict("records")
    np.testing.assert_array_equal(physical_arrays(inputs)["occupancy"],
        source["particle_programs"]["occupancy"][0][0][0])
    original["Unit"] = "mg/L"
    original.to_csv(wide_path,index=False)
    output.unlink()
    rejected = subprocess.run(command,text=True,capture_output=True)
    assert rejected.returncode != 0 and "Source concentration unit mismatch" in rejected.stderr
    assert not output.exists()


def test_source_bytes_and_required_physical_payload(physical_case,tmp_path,monkeypatch):
    directory,_,_,_,config,packet,source = physical_case
    from Libs.scripts.teacher_generation import build_osp_pbpk_teacher_rollouts as producer
    monkeypatch.setattr(producer,"discovered_osp_designs",lambda *args:pytest.fail("native discovery forbidden"))
    invalid = copy.deepcopy(packet)
    invalid["design_id"] = ""
    with pytest.raises(ValueError,match="explicit physical design"):
        build_native_physical_source(config,invalid,directory)
    changed = copy.deepcopy(source)
    changed["particle_programs"]["records"][0][0][0].pop("physical_program")
    with pytest.raises(ValueError,match="payload/provenance"):
        native.parse_joint_source(changed)
    snapshot = json.loads((directory/packet["exports"][0]["run_snapshot_path"]).read_text())
    snapshot["export_sha256"] = "stale"
    stale = tmp_path/"stale.json"
    stale.write_text(json.dumps(snapshot))
    invalid = copy.deepcopy(packet)
    invalid["exports"][0]["run_snapshot_path"] = str(stale)
    with pytest.raises(ValueError,match="export bytes changed"):
        build_native_physical_source(config,invalid,directory)


def affinity_violation(inputs, violation):
    binding,snapshot = inputs["binding"],inputs["run_snapshot"]
    mapping = binding["site_mappings"][0]
    site = mapping["site_uid"]
    parameter = binding["particle_parameters"][mapping["kd_binding"]["parameter_name"]]
    kd = snapshot["affinities"][site]
    if violation == "unbound_path":
        snapshot["detached_affinity"] = dict(kd,value=20.)
        mapping["kd_source_path"] = ["detached_affinity"]
    elif violation == "conflicting_value":
        kd["value"] = 20.
    elif violation == "fixed_conflicting_particle":
        fixed = dict(kd,binding_scope="fixed_source_affinity")
        fixed.pop("particle_uid")
        snapshot["fixed_affinities"] = {site:fixed}
        snapshot["fixed_affinity_paths"] = {site:["fixed_affinities",site]}
        mapping.update(kd_source_path=["fixed_affinities",site],
                       kd_binding={"kind":"fixed_source_affinity","affinity_name":site})
    else:
        key = {"wrong_target":"target_uid","wrong_ligand":"ligand_uid","wrong_site":"site_uid",
               "wrong_source":"source_uid","wrong_snapshot":"source_snapshot_ref",
               "wrong_particle":"particle_uid"}[violation]
        kd[key] = "contradictory-source-identity"
        parameter["value"] = copy.deepcopy(kd)
    return reseal(inputs)


@pytest.mark.parametrize("violation", ["unbound_path","conflicting_value","wrong_target","wrong_ligand",
    "wrong_site","wrong_source","wrong_snapshot","wrong_particle","fixed_conflicting_particle"])
def test_affinity_contradictions_reject_producer_and_admission(physical_case,tmp_path,violation):
    directory,_,_,_,config,packet,source = physical_case
    inputs = affinity_violation(input_copy(physical_case),violation)
    with pytest.raises(ValueError,match="affinity|Kd|particle parameter"):
        compile_physical_record(inputs)
    record = copy.deepcopy(source["particle_programs"]["records"][0][0][0])
    record["physical_program"]["inputs"] = inputs
    # Recompile metadata honestly to ensure rejection comes from source affinity,
    # not merely from a stale digest or an old canonical compiler record.
    record["pkpd_context"]["parameter_provenance"]["physical_input_sha256"] = digest(inputs)
    from Libs.utils.data.action_observation_contracts import compile_pharmacological_action
    record["compiled_program"] = compile_pharmacological_action(record["action"],
        pkpd_context=record["pkpd_context"],baseline_state_posterior=record["baseline_state_posterior"],
        graph_context=record["graph_context"])
    altered = copy.deepcopy(source)
    altered["particle_programs"]["records"][0][0][0] = record
    with pytest.raises(ValueError,match="affinity|Kd|particle parameter"):
        native.parse_joint_source(altered)
    selected = copy.deepcopy(packet)
    snapshot = inputs["run_snapshot"]
    snapshot["model_path"] = str(directory/"fixture.pkml")
    snapshot_path = tmp_path/"conflicting-readback.json"
    snapshot_path.write_text(json.dumps(snapshot))
    selected["exports"][0]["run_snapshot_path"] = str(snapshot_path)
    selected["exports"][0]["binding"] = inputs["binding"]
    with pytest.raises(ValueError,match="affinity|Kd|particle parameter"):
        build_native_physical_source(config,selected,directory)


def fixed_affinity_packet(packet,directory,output_directory):
    selected = copy.deepcopy(packet)
    for index,export in enumerate(selected["exports"]):
        snapshot = json.loads((directory/export["run_snapshot_path"]).read_text())
        snapshot["fixed_affinity_paths"] = {}
        snapshot["model_path"] = str(directory/"fixture.pkml")
        for mapping in export["binding"]["site_mappings"]:
            site = mapping["site_uid"]
            name = mapping["kd_binding"]["parameter_name"]
            export["binding"]["particle_parameters"].pop(name)
            snapshot["particle_parameter_paths"].pop(name)
            kd = snapshot["affinities"][site]
            kd.update(value=10.,binding_scope="fixed_source_affinity",source_ref="explicit-fixed-source-Kd")
            kd.pop("particle_uid")
            snapshot["fixed_affinity_paths"][site] = mapping["kd_source_path"]
            mapping["kd_binding"] = {"kind":"fixed_source_affinity","affinity_name":site}
        path = output_directory/f"fixed-readback-{index}.json"
        path.write_text(json.dumps(snapshot))
        export["run_snapshot_path"] = str(path)
    return selected


def test_fixed_source_affinity_shared_consistently_and_rejects_drift(physical_case,tmp_path):
    directory,model,_,_,config,packet,_ = physical_case
    selected = fixed_affinity_packet(packet,directory,tmp_path)
    source = build_native_physical_source(config,selected,directory)
    native.parse_joint_source(source)
    programs = source["particle_programs"]
    first = programs["records"][0][0][0]["physical_program"]
    second = programs["records"][0][1][0]["physical_program"]
    assert first["target_parameters"][0]["kd_source"] == second["target_parameters"][0]["kd_source"]
    assert first["target_parameters"][0]["kd_mol_per_l"] == pytest.approx(1e-8)
    assert np.asarray(programs["occupancy"])[0,0,0,1,0,0,0] == pytest.approx(.5)
    # The other particle retains its own transport exposure while sharing fixed Kd.
    assert np.asarray(programs["occupancy"])[0,1,0,1,0,0,0] == pytest.approx(2/3)
    posterior = model.infer_world_state(joint_support=source["support"],observations=source["observations"])["joint_posterior"]
    assert native.bind_joint_programs(posterior,source)["records"] == programs["records"]
    altered = copy.deepcopy(source)
    inputs = altered["particle_programs"]["records"][0][1][0]["physical_program"]["inputs"]
    for kd in inputs["run_snapshot"]["affinities"].values(): kd["value"] *= 2
    record = compile_physical_record(reseal(inputs))
    altered["particle_programs"]["records"][0][1][0] = record
    altered["particle_programs"]["occupancy"][0][1][0] = copy.deepcopy(record["physical_program"]["occupancy"])
    with pytest.raises(ValueError,match="Fixed source affinity differs"):
        native.parse_joint_source(altered)
    # Same contradiction must fail the actual producer before authoritative output.
    changed = json.loads(Path(selected["exports"][2]["run_snapshot_path"]).read_text())
    for kd in changed["affinities"].values(): kd["value"] *= 2
    Path(selected["exports"][2]["run_snapshot_path"]).write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="Fixed source affinity differs"):
        build_native_physical_source(config,selected,directory)


def test_local_membrane_physical_program_basis_rejects_swapped_coordinate(tmp_path):
    from Libs.tests.test_joint_world_posterior import local_membrane_case
    from Libs.scripts.builders.build_sample_observation_registry import produce_controlled_observations
    model, support, _, _, channels = joint_case()
    config = load_config("Libs/configs/config_stack.yaml")
    config["spatial_atlas_registry"]["regions"] = config["spatial_atlas_registry"]["regions"][:3]
    config["screening_pipeline"]["pkpd_ready_registry"]["compartments"] = config["screening_pipeline"]["pkpd_ready_registry"]["compartments"][:2]
    support, channels = local_membrane_case(support, channels,
        {"region_id":region_id_list(config), "compartment_id":compartment_id_list(config)})
    production = produce_controlled_observations(support, channels, latent_seed=7, observation_seed=13)
    packet = make_packet(tmp_path, config, raw_joint_source(support, production))
    source = build_native_physical_source(config, packet, tmp_path)
    native.parse_joint_source(source)
    posterior = model.infer_world_state(joint_support=source["support"], observations=source["observations"])["joint_posterior"]
    native.bind_joint_programs(posterior, source)
    # Keep observations consistent with each edited snapshot so rejection tests
    # the retained whole-program basis, not merely observation validation.
    for edit in ("voltage", "anatomy", "time", "reference"):
        changed = copy.deepcopy(source)
        snapshot = changed["support"]["local_membrane"]
        acquisition = changed["observations"][2]["physical_acquisition"]
        if edit == "voltage":
            snapshot["voltage"][0][0][0] += 1.
        elif edit == "anatomy":
            snapshot["sites"][0]["anatomy_ref"] = "another-soma"
        elif edit == "time":
            snapshot["current_time_ms"] = [.1]
            snapshot["acquisition_cutoff_ms"] = [.15]
            acquisition["measurement_time_ms"] = acquisition["acquired_time_ms"] = [.1]
        else:
            snapshot["sites"][0]["voltage_reference_uid"] = "another-reference"
            acquisition["voltage_reference_uids"] = ["another-reference"]
            changed["support"]["readout_parameters"]["ephys"]["physical_readout"]["voltage_reference_uids"] = ["another-reference"]
        with pytest.raises(ValueError, match="whole-particle binding mismatch"):
            native.parse_joint_source(changed)
    reordered = native.select_joint_source(source, list(support.sample_uids), ["regimen-b","regimen-a"])
    assert reordered["particle_programs"]["records"][0][0] == source["particle_programs"]["records"][0][0][::-1]
    assert reordered["support"]["local_membrane"] == support.local_membrane.to_payload()
