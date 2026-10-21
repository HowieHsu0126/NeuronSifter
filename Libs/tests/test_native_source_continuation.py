"""Bounded source continuation on disposable inputs, with no production-slot reuse."""
import copy
import json
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from Libs.scripts.builders import build_controlled_native_source as builder
from Libs.tests.test_native_scientific_corpus import corpus_fixture
from Libs.utils.data.physical_occupancy import digest
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.model import native_scientific_source as source


def continuation_fixture(tmp_path):
    config = corpus_fixture()
    config.pop("config_includes", None)
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["physical_parameter_conditioning"] = None
    settings = config["native_observable_bridge"]
    settings.update(seed=39581005, source_uid="disposable_group_continuation",
        snapshot_ref="disposable_group_continuation_definition",
        source_contract_status="frozen", output_root=str(tmp_path/"source"))
    settings["atoms"] = [[0,0,0,0,0],[0,1,1,0,1],[1,0,1,1,0],[1,1,0,1,1]]
    settings["prior_weights"] = [.1,.2,.3,.4]
    settings["group_partitions"] = [
        {"partition_uid":"train","role":"train","groups_per_family":2,"action_indices":[0,1]}]
    settings["budget"].update(source.planned_operation_counts(config),seconds=1800)
    storage = copy.deepcopy(load_config("Libs/configs/exp/manuscript_native_protocol.yaml")[
        "native_diffusion_data"]["group_storage"])
    storage["maximum_groups_per_invocation"] = 2
    config["native_diffusion_data"]["group_storage"] = storage
    path = tmp_path/"fixture.yaml"
    path.write_text(yaml.safe_dump(config,sort_keys=False))
    return config,path


def test_three_real_bounded_invocations_keep_every_operation_and_only_admit_at_end(tmp_path,monkeypatch,capsys):
    config,path = continuation_fixture(tmp_path)
    settings = config["native_observable_bridge"]
    identity = {"producer_commit":"disposable-continuation-fixture",
        "source_code_sha256":"0"*64,"bound_files":{},
        "definition_sha256":digest({"profile":settings,"background":config["controlled_cns_source"]})}
    monkeypatch.setattr(builder,"observable_identity",lambda *_:copy.deepcopy(identity))
    monkeypatch.setattr(builder,"verify_observable_source",
        lambda receipt,cfg: receipt["source_identity"] == identity or pytest.fail("Source identity changed"))
    events,groups,admissions = [],[],[]
    original_execute = builder.SourceLedger.execute
    original_draw = source.draw_group
    root = Path(settings["output_root"])
    def execute(self,key,function):
        events.append(key)
        return original_execute(self,key,function)
    def draw(cfg,group,index,**kwargs):
        if index:
            assert (root/f"group_{index-1}_conditioning.json.gz").is_file()
            local = Path(json.loads((root/"source_reservation.json").read_text())["node_local_path"])
            assert not (local/f"group_{index-1}_conditioning.json.gz").exists()
        groups.append(index)
        return original_draw(cfg,group,index,**kwargs)
    monkeypatch.setattr(builder.SourceLedger,"execute",execute)
    monkeypatch.setattr(source,"draw_group",draw)
    monkeypatch.setattr(builder,"consume_observable_source",
        lambda *a,**k: admissions.append("only_after_complete") or {"status":"complete"})
    monkeypatch.setattr(sys,"argv",["builder","--config",str(path)])
    assert builder.main() == 0
    first = json.loads((root/"source_receipt.json").read_text())
    assert first["source_status"] == "partial" and first["next_group_index"] == 2
    assert not admissions
    assert all(row["status"] == "planned" for row in first["cases"] if row["group_index"] >= 2)
    retained = {name:(root/name).read_bytes() for name in
                ("group_0_producer.json","group_0_conditioning.json.gz","case_0_reference.npz","group_0_receipt.json")}
    local = Path(json.loads((root/"source_reservation.json").read_text())["node_local_path"])
    assert local.is_dir()
    before = len(events)
    with pytest.raises(FileExistsError):
        builder.execute_observable_source(config,path)
    assert len(events) == before
    for stop in (4,6):
        monkeypatch.setattr(sys,"argv",["builder","--config",str(path),"--continue-source"])
        assert builder.main() == 0
        current = json.loads((root/"source_receipt.json").read_text())
        assert current["next_group_index"] == stop
        assert all((root/name).read_bytes() == raw for name,raw in retained.items())
    assert groups == list(range(6))
    assert len(events) == len(set(events)) == settings["budget"]["operations"]
    assert current["source_status"] == "complete"
    assert len(current["cases"]) == settings["budget"]["cases"]
    assert all(row["status"] == "complete" for row in current["operations"].values())
    assert admissions == ["only_after_complete"]
    assert current["source_clock_seconds"] >= first["source_clock_seconds"]
    assert [row["first_group_index"] for row in current["completed_chunks"]] == [0,2,4]
    assert not local.exists()
    assert "source_chunk_complete" in capsys.readouterr().out


def closed_prefix(ledger,start=1):
    previous = copy.deepcopy(ledger.data)
    previous.update(source_status="partial",next_group_index=start,source_clock_seconds=10.)
    for case in previous["cases"]:
        if case["group_index"] < start:
            case["status"] = "complete"
    for key,operation in previous["operations"].items():
        if ledger.operation_group_index(key) < start:
            operation["status"] = "complete"
    return previous


@pytest.mark.parametrize("mutation",["started","skipped","budget","expired"])
def test_source_continuation_rejects_unknown_skipped_or_changed_budget(tmp_path,mutation):
    config,_ = continuation_fixture(tmp_path)
    ledger = builder.ObservableSourceLedger(tmp_path/"receipt.json",config,{"fixture":True},
                                           clock=lambda:100.,persist_initial=False)
    previous = closed_prefix(ledger)
    if mutation == "started":
        previous["operations"]["group:1"]["status"] = "started"
    elif mutation == "skipped":
        previous["next_group_index"] = 2
    elif mutation == "budget":
        previous["resource_limit"]["source_seconds"] += 1
    else:
        previous["source_clock_seconds"] = previous["resource_limit"]["source_seconds"]
    with pytest.raises((ValueError,TimeoutError)):
        builder.restore_source_prefix(ledger,previous)
    assert not (tmp_path/"receipt.json").exists()


def test_catalog_declares_continuation_without_treating_chunk_as_corpus_completion():
    from Libs.utils.evaluation.manuscript_native_protocol import command_catalog
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    command = next(row for row in command_catalog(config,[]) if row["command_uid"] == "source_production_and_base_admission")
    assert command["maximum_groups_per_invocation"] == 32
    assert command["continuation_argv"] == command["argv"]+["--continue-source"]
    assert command["completion_predicate"] == "source_receipt.source_status=complete_and_native_consumption.status=complete"


def test_retained_capacity_is_reserved_before_any_source_operation(tmp_path,monkeypatch):
    config,path = continuation_fixture(tmp_path)
    storage = config["native_diffusion_data"]["group_storage"]
    storage["retained_max_bytes"] = storage["maximum_terminal_receipt_bytes"]+storage["maximum_group_write_bytes"]-1
    monkeypatch.setattr(builder,"observable_identity",lambda *_:{"fixture":"capacity"})
    monkeypatch.setattr(builder.SourceLedger,"execute",lambda *_:pytest.fail("Source ran without retained capacity"))
    root=builder.execute_observable_source(config,path)
    receipt=json.loads((root/"source_receipt.json").read_text())
    assert receipt["source_status"]=="failed"
    assert "Retained source budget cannot reserve" in receipt["reason"]
    assert all(row["status"]=="planned" for row in receipt["operations"].values())
    assert not any(root.glob("case_*"))
    # No source result was created; remove the task-owned empty execution workspace.
    local=Path(json.loads((root/"source_reservation.json").read_text())["node_local_path"])
    shutil.rmtree(local)


def test_exception_stageout_preserves_existing_raw_data_and_reports_local_result(tmp_path):
    local,output=tmp_path/"local",tmp_path/"output"
    local.mkdir();output.mkdir()
    (local/"raw.npz").write_bytes(b"new_source_result")
    (output/"raw.npz").write_bytes(b"existing_result")
    (local/"other.npz").write_bytes(b"new_other_result")
    (local/"source_receipt.json").write_bytes(b"new_checkpoint")
    (output/"source_receipt.json").write_bytes(b"old_checkpoint")
    conflicts=builder.stage_remaining_source_outputs(
        local,output,["raw.npz","other.npz","source_receipt.json"],set())
    assert (output/"raw.npz").read_bytes()==b"existing_result"
    assert (local/"raw.npz").read_bytes()==b"new_source_result"
    assert (output/"other.npz").read_bytes()==b"new_other_result"
    assert (output/"source_receipt.json").read_bytes()==b"new_checkpoint"
    assert conflicts==[{"local_path":str(local/"raw.npz"),"existing_path":str(output/"raw.npz")}]


@pytest.mark.skipif(
    os.environ.get("NRNSIFT_RUN_ROMANI_CONTINUATION") != "1",
    reason="Explicit real-source diagnostic; not part of routine software reruns",
)
def test_two_process_romani_continuation_preserves_full_quantity_admission_gate(tmp_path):
    """Use fresh disposable priors and the real identity, source and admission paths."""
    repo = Path(__file__).resolve().parents[2]
    reference = Path(os.environ["NRNSIFT_ROMANI_SOURCE_ROOT"]).resolve(strict=True)
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    config.pop("config_includes", None)
    settings = config["native_observable_bridge"]
    family = config["native_romani_source"]["family_uid"]
    settings["quantity_profile"]["families"] = {
        family: settings["quantity_profile"]["families"][family]}
    settings.update(seed=873908, source_uid="disposable_romani_process_continuation",
        snapshot_ref="disposable_romani_process_continuation_definition",
        source_contract_status="frozen", output_root=str(tmp_path/"source"))
    settings["group_partitions"] = [{
        "partition_uid": "train", "role": "train", "groups_per_family": 2,
        "action_indices": [0, 1, 2], "ood_class": "in_distribution",
        "source_variant": "romani", "conditions": ["control", "ad_positive"]}]
    settings["training_source_strata"]["cycle"] = ["romani"]
    settings["budget"].update(source.planned_operation_counts(config), seconds=1800, memory_gib=4)
    config["native_diffusion_data"]["group_storage"]["maximum_groups_per_invocation"] = 1
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["physical_parameter_conditioning"] = None
    modeldb = config["dynamics_ready_registry"]["neuron_rxd_teacher"]["model_sources"]["modeldb"]
    original_layout = Path(modeldb["extracted_root"])
    for name in ("mechanism_root", "simulation_hoc_path", "morphology_root"):
        modeldb[name] = str(reference/Path(modeldb[name]).relative_to(original_layout))
    modeldb["extracted_root"] = str(reference)

    # The fixture commit binds the actual bytes and effective diagnostic config;
    # source identity/verification algorithms are exercised without replacements.
    identity_root = tmp_path/"identity"
    relative_config = "Libs/configs/exp/romani_continuation_fixture.yaml"
    for name in settings["freeze_paths"]:
        destination = identity_root/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo/name, destination)
    settings["freeze_paths"].append(relative_config)
    config_path = identity_root/relative_config
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    git = ["git", "-c", "gc.auto=0", "-c", "maintenance.auto=false",
           "-c", "user.name=NeuronSifter disposable fixture",
           "-c", "user.email=neuronsifter-fixture@localhost"]
    for arguments in (["init", "-q"], ["add", "--", *settings["freeze_paths"]],
                      ["commit", "-q", "-m", "Bind disposable source continuation fixture"]):
        subprocess.run([*git, *arguments], cwd=identity_root, check=True, capture_output=True)

    child = r'''
import hashlib,json,os,sys
from pathlib import Path
from Libs.scripts.builders import build_controlled_native_source as builder
builder.ROOT=Path(sys.argv[1])
events=[]
original_stage=builder.romani_source.stage_source
original_cell=builder.romani_source.RomaniCell
def stage(*args,**kwargs):
    retained=os.environ.get("NRNSIFT_ROMANI_PREPARED_SOURCE")
    if retained:
        prepared=json.loads(Path(retained).read_text())
        library=Path(prepared["build"]["staged_root"])/prepared["build"]["library_relative_path"]
        if prepared["build"]["exit_code"]!=0 or hashlib.sha256(library.read_bytes()).hexdigest()!=prepared["build"]["identity"]["library_sha256"]:
            raise ValueError("Retained compilation is incomplete or changed")
        for name,expected in prepared["files"].items():
            if hashlib.sha256((Path(args[1])/name).read_bytes()).hexdigest()!=expected:
                raise ValueError("Retained build differs from the current source")
        events.append({"event":"reuse_verified_compilation","pid":os.getpid(),
                       "prepared_source":retained,"original_job":prepared["build"]["job_id"]})
        return prepared
    events.append({"event":"compile_source","pid":os.getpid()})
    return original_stage(*args,**kwargs)
def cell(config,prepared):
    library=Path(prepared["build"]["staged_root"])/prepared["build"]["library_relative_path"]
    result=original_cell(config,prepared)
    events.append({"event":"load_runtime","pid":os.getpid(),"library":str(library),
        "library_sha256":hashlib.sha256(library.read_bytes()).hexdigest(),
        "library_mtime_ns":library.stat().st_mtime_ns,
        "hoc_startup_initializations":result.initialization_count})
    return result
builder.romani_source.stage_source=stage
builder.romani_source.RomaniCell=cell
events_path=Path(sys.argv[3])
sys.argv=["source","--config",sys.argv[2],*sys.argv[4:]]
try:
    status=builder.main()
finally:
    events_path.write_text(json.dumps({"pid":os.getpid(),"events":events},indent=2))
raise SystemExit(status)
'''
    root = Path(settings["output_root"])
    first_hashes = {}
    results = []
    for index in range(2):
        events_path = tmp_path/f"process_{index}.json"
        log = tmp_path/f"process_{index}.log"
        command = [sys.executable, "-c", child, str(identity_root), str(config_path), str(events_path)]
        if index:
            command.append("--continue-source")
        with log.open("w") as output:
            process = subprocess.run(command, cwd=repo, stdout=output, stderr=subprocess.STDOUT)
        assert process.returncode == (0 if index == 0 else 2), log.read_text()[-6000:]
        results.append(json.loads(events_path.read_text()))
        receipt = json.loads((root/"source_receipt.json").read_text())
        if index == 0:
            assert receipt["source_status"] == "partial" and receipt["next_group_index"] == 1
            assert not (root/"native_consumption.json").exists()
            for name in ("group_0_producer.json", "group_0_conditioning.json.gz",
                         "case_0_reference.npz", "group_0_receipt.json"):
                first_hashes[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
            first_peak = receipt["source_peak_rss_kib"]
        else:
            assert receipt["source_status"] == "complete" and receipt["next_group_index"] == 2
            assert receipt["source_peak_rss_kib"] >= first_peak
            assert all(hashlib.sha256((root/name).read_bytes()).hexdigest() == value
                       for name, value in first_hashes.items())
    assert results[0]["pid"] != results[1]["pid"]
    compiled = [event for result in results for event in result["events"] if event["event"] == "compile_source"]
    loads = [event for result in results for event in result["events"] if event["event"] == "load_runtime"]
    reused = [event for result in results for event in result["events"]
              if event["event"] == "reuse_verified_compilation"]
    assert len(compiled) == (0 if os.environ.get("NRNSIFT_ROMANI_PREPARED_SOURCE") else 1)
    assert len(compiled)+len(reused) == 1 and len(loads) == 2
    assert all(loads[0][key] == loads[1][key]
               for key in ("library", "library_sha256", "library_mtime_ns"))
    assert all(row["hoc_startup_initializations"] == 1 for row in loads)
    assert len(receipt["reused_runtime_loads"]) == 1
    assert receipt["reused_runtime_loads"][0]["hoc_startup_initializations"] == 1
    assert all(row["status"] == "complete" for row in receipt["operations"].values())
    assert len(receipt["operations"]) == settings["budget"]["operations"]
    assert len(receipt["cases"]) == 6
    assert all(len(group["particle_uids"]) == 32 for group in receipt["groups"])
    admission = json.loads((root/"native_consumption.json").read_text())
    assert admission["status"] == "incomplete" and admission["admitted_count"] == 0
    assert "every declared quantity needs supported training observations" in admission["consumer_error"]
    prepared = json.loads((root/"romani_source_preparation.json").read_text())
    assert all(hashlib.sha256((reference/name).read_bytes()).hexdigest() == value
               for name, value in prepared["files"].items())
    local = Path(json.loads((root/"source_reservation.json").read_text())["node_local_path"])
    assert not local.exists()
    (tmp_path/"result.json").write_text(json.dumps({
        "scope": "disposable_real_romani_two_process_source_and_complete_quantity_gate",
        "status": "PASS", "groups": 2, "cases": 6, "complete_particles": 32,
        "compile_calls_current_run": len(compiled), "verified_retained_compilations": reused,
        "runtime_loads": len(loads),
        "source_operations": len(receipt["operations"]),
        "diagnostic_admitted_cases": admission["admitted_count"],
        "production_admitted_cases": 0, "source_peak_rss_kib": receipt["source_peak_rss_kib"],
        "source_seconds": receipt["source_clock_seconds"],
        "original_source_files": len(prepared["files"]),
        "optimizer_updates": 0, "scientific_twin": "NOT_RUN",
        "formal_qualification": False,
    }, indent=2))


def test_retained_romani_process_results_preserve_full_quantity_admission_gate(monkeypatch):
    location = os.environ.get("NRNSIFT_ROMANI_CONTINUATION_REUSE")
    if not location:
        pytest.skip("Requires retained real source results; never regenerates them")
    fixture = Path(location)
    identity_root = fixture/"identity"
    config_path = identity_root/"Libs/configs/exp/romani_continuation_fixture.yaml"
    config = load_config(str(config_path))
    root = Path(config["native_observable_bridge"]["output_root"])
    receipt = json.loads((root/"source_receipt.json").read_text())
    monkeypatch.setattr(builder, "ROOT", identity_root)
    builder.verify_observable_source(receipt, config)
    assert receipt["source_status"] == "complete" and receipt["next_group_index"] == 2
    assert len(receipt["cases"]) == 6 and all(case["status"] == "complete" for case in receipt["cases"])
    assert all(not case["numerical"]["reasons"] for case in receipt["cases"])
    assert len(receipt["operations"]) == config["native_observable_bridge"]["budget"]["operations"]
    assert all(row["status"] == "complete" for row in receipt["operations"].values())
    assert all(len(group["particle_uids"]) == 32 for group in receipt["groups"])
    markers = [json.loads((root/f"source_chunk_{index}.json").read_text()) for index in (0, 1)]
    assert [row["source_status"] for row in markers] == ["partial", "complete"]
    assert all(row["status"] == "complete" for row in markers)
    results = [json.loads((fixture/f"process_{index}.json").read_text()) for index in (0, 1)]
    assert results[0]["pid"] != results[1]["pid"]
    loads = [event for result in results for event in result["events"] if event["event"] == "load_runtime"]
    reused = [event for result in results for event in result["events"] if event["event"] == "reuse_verified_compilation"]
    assert len(loads) == 2 and len(reused) == 1
    assert not any(event["event"] == "compile_source" for result in results for event in result["events"])
    assert all(loads[0][key] == loads[1][key] for key in ("library", "library_sha256", "library_mtime_ns"))
    assert hashlib.sha256(Path(loads[0]["library"]).read_bytes()).hexdigest() == loads[0]["library_sha256"]
    assert receipt["reused_runtime_loads"] == [{
        "allocation": markers[1]["allocation"], "group_index": 1,
        "source_preparation_replayed": False, "hoc_startup_initializations": 1}]
    admission = json.loads((root/"native_consumption.json").read_text())
    assert admission["status"] == "incomplete" and admission["admitted_count"] == 0
    assert "every declared quantity needs supported training observations" in admission["consumer_error"]
    assert all(not case["reasons"] and case["native_status"] == "candidate" for case in admission["cases"])
    # The original first-group snapshot binds its compressed input before group1.
    snapshot = json.loads((root/"group_0_receipt.json").read_text())
    reference = snapshot["group"]["conditioning_reference"]
    assert reference == receipt["groups"][0]["conditioning_reference"]
    builder.read_joint_source_reference(reference)
    prepared = json.loads((root/"romani_source_preparation.json").read_text())
    original = Path(prepared["source_root"])
    assert all(hashlib.sha256((original/name).read_bytes()).hexdigest() == value
               for name, value in prepared["files"].items())
    local = Path(json.loads((root/"source_reservation.json").read_text())["node_local_path"])
    assert not local.exists()
    result = {
        "status": "PASS_SOURCE_CONTINUATION_AND_EXPECTED_ADMISSION_REJECTION",
        "source_job": markers[1]["allocation"], "retained_compilation_job": prepared["build"]["job_id"],
        "groups": 2, "cases": 6, "complete_prior_tuples": 32,
        "source_operations": len(receipt["operations"]), "new_source_operations": 0,
        "new_compilation_calls": 0, "runtime_loads_in_source_job": len(loads),
        "source_seconds": receipt["source_clock_seconds"],
        "source_peak_rss_kib": receipt["source_peak_rss_kib"],
        "numerical_cases_passed": 6, "native_admitted_cases": 0,
        "admission_reason": admission["consumer_error"],
        "source_input_boundary": "initial_observation_and_whole_prior_programs_separate_from_producer_truth",
        "original_source_files": len(prepared["files"]),
        "scientific_twin": "NOT_RUN", "formal_qualification": False,
    }
    output = os.environ.get("NRNSIFT_ROMANI_REUSE_REPORT")
    if output:
        Path(output).write_text(json.dumps(result, indent=2))
