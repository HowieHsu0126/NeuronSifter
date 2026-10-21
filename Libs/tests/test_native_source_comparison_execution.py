"""Disposable execution fixtures. No scientific tensors or source calls."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest
import yaml

from Libs.exp.submission import native_source_comparison_execution as execution
from Libs.utils.model import native_scientific_source


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    execution.write_json(path, value)


def scientific_profile(config: dict, normalization_id: str) -> dict:
    return {"training_data_scope": config["test_scope"], "normalization_id": normalization_id,
            "backbone": config["test_backbone"]}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    # Isolate pure metadata calculation. The runner still checks actual files,
    # receipt values, explicit transfers, immutable outputs and exit status.
    monkeypatch.setattr(native_scientific_source, "native_training_profile", scientific_profile)
    root, package = tmp_path / "original", tmp_path / "package"
    root.mkdir(); package.mkdir()
    groups = {"train": ["train_a", "train_b"], "development": ["dev_a"]}
    binding = {"selected_formal_seed": 40, "model_seed": 40}
    scope = {"source_scope": "pure_shared", "selected_training_group_sha256": "fixture_groups"}
    config = {"native_profile_identity": {"method": "state_single_source", "seed": 40, "percentage": 100},
              "native_scientific_training": {"source_scope": "pure_shared",
                "selection": {"minimum_steps": 10000, "minimum_epochs": 80}},
              "test_scope": scope, "test_backbone": {"kind": "neuronsifter_world_action"}}
    common_config = copy.deepcopy(config)
    common_config["native_profile_identity"]["method"] = "native_single_source_wam"
    profile_ref, common_ref = "profiles/state.yaml", "profiles/native.yaml"
    profile_path, common_path = str(root / profile_ref), str(root / common_ref)
    row = {"profile_id": "state_single_source_fraction_100_seed_40", "method": "state_single_source",
           "seed": 40, "training_subset_percentage": 100, "source_scope": "pure_shared",
           "config_path": profile_path, "joint_belief_fit": False,
           "output_root": "Output/results/state/primary", "training_manifest": "Output/results/state/primary/native_comparator_training_manifest.json",
           "effective_configuration_sha256": execution.effective_digest(config)}
    common = {**row, "method": "native_single_source_wam", "profile_id": "native_single_source_wam_fraction_100_seed_40",
              "config_path": common_path, "joint_belief_fit": True, "checkpoint_scale": "neuronsifter_large",
              "inference_profile": "large_main_screening",
              "training_manifest": "Output/results/native/primary/microenvironment_pretraining_manifest.json",
              "effective_configuration_sha256": execution.effective_digest(common_config)}
    profiles = {}
    for original, relative, payload in [(profile_path, profile_ref, config), (common_path, common_ref, common_config)]:
        source = Path(original); source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(yaml.safe_dump(payload))
        target = package / source.name; shutil.copy2(source, target)
        profiles[original] = {"reference": relative, "file": source.name, "sha256": execution.digest(source)}
    for name in execution.LAUNCHERS:
        (package / name).write_text("fixture launcher\n")
    normalizer_ref = "admitted/source_scope_normalization/pure_shared.json"
    normalizer = {"fit_role": "train", "training_data_scope": scope,
                  "training_records": [{"sample_uid": uid} for uid in groups["train"]]}
    write(root / normalizer_ref, normalizer)
    normalization_id = execution.digest(root / normalizer_ref)
    checkpoint = root / "Output/results/native/primary/checkpoints/selected.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"fixture selected checkpoint; no tensor payload")
    manifest = {"failure": None, "actual_total_parameter_count": 128, "executed_steps": 10000,
                "training_seed_binding": binding, "checkpoint_scale": "neuronsifter_large",
                "inference_profile": "large_main_screening", "train_group_count": 2, "development_group_count": 1,
                "native_scientific_profile": scientific_profile(common_config, normalization_id),
                "selection": {"terminal": "INCOMPLETE_BUDGET_CAP", "selection_role": "development",
                  "sufficient_training": False, "checkpoint_path_base": "manifest_directory",
                  "last": {"steps": 10000, "epoch": 80},
                  "best": {"steps": 9870, "epoch": 79, "checkpoint_path": "checkpoints/selected.pt"}}}
    write(root / common["training_manifest"], manifest)
    chain = {"schema_version": execution.SCHEMA, "source_worktree": str(root), "source_commit": "committed_fixture",
             "profile": row, "conditioning_profile": common, "profiles": profiles,
             "group_uids": groups, "groups": {role: len(uids) for role, uids in groups.items()},
             "training_seed_binding": binding, "training_data_scope": scope, "normalizer_reference": normalizer_ref,
             "checkpoint_storage": {"maximum_live_bytes": 1000},
             "launcher_sha256": {name: execution.digest(package / name) for name in execution.LAUNCHERS},
             "fit_argv": ["python", "Libs/exp/baselines/run_external_baseline_bridge.py", "--config", profile_path,
                          "--conditioning-config", common_path, "--conditioning-manifest", common["training_manifest"],
                          "--output-dir", row["output_root"], "--command", "train_baseline"],
             "stage_argv": ["python", execution.PRETRAINER, "--config", profile_path,
                            "--native-input-role", "train", "--native-input-role", "development"]}
    write(package / "execution_chain.json", chain)
    return {"root": root, "package": package, "chain": chain, "config": config, "common_config": common_config,
            "manifest": manifest, "manifest_path": root / common["training_manifest"], "checkpoint": checkpoint,
            "normalization_id": normalization_id, "tmp": tmp_path}


def test_state_stage_uses_pretrainer_and_actual_reuse_argument(prepared):
    chain = prepared["chain"]
    command = execution.stage_command(chain, Path("/node/repo"), Path("/node/inputs"), Path("/node/full/native_inputs.json"))
    assert command[1] == "/node/repo/" + execution.PRETRAINER
    assert "run_external_baseline_bridge.py" not in command[1]
    assert command[command.index("--config") + 1] == "/node/repo/profiles/state.yaml"
    assert command[command.index("--reuse-native-inputs") + 1] == "/node/full/native_inputs.json"
    assert command.count("--native-input-role") == 2
    assert "--device" not in command


@pytest.mark.parametrize("change", ["missing", "wrong_role", "duplicate", "not_in_reuse"])
def test_actual_group_identity_and_reuse_subset_rejection(prepared, change):
    groups = copy.deepcopy(prepared["chain"]["group_uids"])
    reused = {"groups": {"train": ["train_a", "train_b", "train_c"], "development": ["dev_a"]}}
    if change == "missing": groups["train"] = ["train_a"]
    elif change == "wrong_role": groups["development"] = ["train_a"]
    elif change == "duplicate": groups["train"] = ["train_a", "train_a"]
    else: reused["groups"]["train"] = ["train_a"]
    with pytest.raises(ValueError):
        execution.verify_groups({"groups": groups}, prepared["chain"], reused)


def test_actual_group_identity_accepts_declared_subset(prepared):
    chain = prepared["chain"]
    execution.verify_groups({"groups": chain["group_uids"]}, chain,
        {"groups": {"train": ["train_a", "train_b", "train_c"], "development": ["dev_a"]}})


@pytest.mark.parametrize("change,match", [("seed", "seed"), ("scope", "profile"),
    ("normalizer", "profile"), ("steps", "insufficient"), ("epochs", "insufficient"),
    ("failure", "failure-free"), ("unfinalized", "finalized"), ("absolute", "portable"),
    ("missing_file", "absent")])
def test_conditioner_rejects_wrong_identity_or_insufficient_actual_training(prepared, change, match):
    manifest = copy.deepcopy(prepared["manifest"])
    if change == "seed": manifest["training_seed_binding"] = {"selected_formal_seed": 41}
    elif change == "scope": manifest["native_scientific_profile"]["training_data_scope"]["source_scope"] = "mixed_shared"
    elif change == "normalizer": manifest["native_scientific_profile"]["normalization_id"] = "wrong"
    elif change == "steps": manifest["executed_steps"] = manifest["selection"]["last"]["steps"] = 9999
    elif change == "epochs": manifest["selection"]["last"]["epoch"] = 79
    elif change == "failure": manifest["failure"] = "ordinary fit failure"
    elif change == "unfinalized": manifest.pop("actual_total_parameter_count")
    elif change == "absolute": manifest["selection"]["best"]["checkpoint_path"] = str(prepared["checkpoint"])
    else: prepared["checkpoint"].unlink()
    write(prepared["manifest_path"], manifest)
    with pytest.raises(ValueError, match=match):
        execution.validate_conditioner(prepared["manifest_path"], prepared["chain"], prepared["common_config"], prepared["normalization_id"])


def test_conditioner_budget_attainment_preserves_false_sufficiency(prepared):
    selected = execution.validate_conditioner(prepared["manifest_path"], prepared["chain"], prepared["common_config"], prepared["normalization_id"])
    assert selected == prepared["checkpoint"]
    assert execution.read_json(prepared["manifest_path"])["selection"]["sufficient_training"] is False


def input_manifest(prepared, stage):
    chain = prepared["chain"]
    logical = str(prepared["root"] / chain["normalizer_reference"])
    target = execution.native_file(stage, logical)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(logical), target)
    manifest = {"groups": chain["group_uids"], "aliases": {chain["normalizer_reference"]: logical},
                "files": {logical: {"size_bytes": target.stat().st_size, "sha256": execution.digest(target)}},
                "file_count": 1, "size_bytes": target.stat().st_size}
    write(stage / "inputs/native_inputs.json", manifest)
    return manifest


def test_normalizer_rejects_scoped_training_identity_change(prepared):
    stage = prepared["tmp"] / "stage"
    manifest = input_manifest(prepared, stage)
    assert execution.verify_normalizer(stage, manifest, prepared["chain"]) == prepared["normalization_id"]
    chain = copy.deepcopy(prepared["chain"])
    chain["training_data_scope"]["source_scope"] = "mixed_shared"
    with pytest.raises(ValueError, match="normalizer"):
        execution.verify_normalizer(stage, manifest, chain)


def staging_archive(prepared):
    package, chain = prepared["package"], prepared["chain"]
    snapshot = prepared["tmp"] / "snapshot"
    code = snapshot / "entry.py"; code.parent.mkdir(); code.write_text("fixture committed code\n")
    chain["code_sha256"] = {"entry.py": execution.digest(code)}
    chain["code_archive_name"] = "code_snapshot.tar"
    archive = package / chain["code_archive_name"]
    with tarfile.open(archive, "w") as stream:
        stream.add(code, arcname="entry.py")
    chain["code_archive_sha256"] = execution.digest(archive)
    chain["registry_mapping"] = {"snapshot_relative_directory": "admitted", "original_directory": str(prepared["root"] / "admitted")}
    chain["admission"] = {"registry_reference": "admitted/model_registry.json"}
    write(package / "execution_chain.json", chain)


def test_state_stage_transfers_actual_manifest_and_selected_checkpoint(prepared, monkeypatch):
    staging_archive(prepared)
    chain = prepared["chain"]
    destination, evidence = prepared["tmp"] / "staged", prepared["tmp"] / "evidence"
    reuse = prepared["tmp"] / "full/native_inputs.json"
    write(reuse, {"groups": {"train": ["train_a", "train_b", "train_c"], "development": ["dev_a"]}})
    monkeypatch.setattr(execution.socket, "gethostname", lambda: "cpu03")
    monkeypatch.setenv("SLURM_JOB_ID", "fixture_stage")
    monkeypatch.setattr(execution, "admission_preflight", lambda base, chain: {"fixture_admission": True})
    def fake_stage(argv, **kwargs):
        assert argv[1].endswith(execution.PRETRAINER)
        assert argv[argv.index("--reuse-native-inputs") + 1] == str(reuse)
        input_manifest(prepared, destination)
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(execution.subprocess, "run", fake_stage)
    record = execution.stage_inputs(prepared["package"], destination, reuse, evidence)
    assert record["status"] == "COMPLETE"
    transfers = (destination / "transfer_files.txt").read_text().splitlines()
    for source in (prepared["manifest_path"], prepared["checkpoint"]):
        relative = str(source.relative_to(prepared["root"]))
        assert "repo/" + relative in transfers
        assert (destination / "repo" / relative).read_bytes() == source.read_bytes()
        assert record["conditioner_sha256"][relative] == execution.digest(source)
    assert execution.read_json(evidence / "staging_result.json")["status"] == "COMPLETE"
    command = execution.fit_command(chain, destination / "repo", destination / "inputs/native_inputs.json")
    actual_manifest = Path(command[command.index("--conditioning-manifest") + 1])
    assert actual_manifest == destination / "repo" / chain["conditioning_profile"]["training_manifest"]
    selected = execution.validate_conditioner(actual_manifest, chain, prepared["common_config"], prepared["normalization_id"])
    assert selected.is_relative_to(destination / "repo")


def test_stage_admission_failure_is_retained_before_any_pretrainer_call(prepared, monkeypatch):
    destination, evidence = prepared["tmp"] / "failed_stage", prepared["tmp"] / "failed_evidence"
    monkeypatch.setattr(execution.socket, "gethostname", lambda: "cpu03")
    monkeypatch.setenv("SLURM_JOB_ID", "fixture_stage")
    def reject(*args): raise ValueError("full admissions not complete")
    def no_process(*args, **kwargs): pytest.fail("Staging process must not start before admission completion")
    monkeypatch.setattr(execution, "admission_preflight", reject)
    monkeypatch.setattr(execution.subprocess, "run", no_process)
    with pytest.raises(ValueError, match="admissions"):
        execution.stage_inputs(prepared["package"], destination, prepared["tmp"] / "unused", evidence)
    actual = execution.read_json(evidence / "staging_result.json")
    assert actual["status"] == "FAILED" and "admissions" in actual["error"]
    assert actual["scientific_fit"] is False


def gpu_transfer_fixture(prepared):
    chain = prepared["chain"]
    source = prepared["tmp"] / "cpu_stage"
    source.mkdir()
    code = source / "repo/entry.py"; code.parent.mkdir(); code.write_text("fixture code")
    chain["code_sha256"] = {"entry.py": execution.digest(code)}
    for entry in chain["profiles"].values():
        target = source / "repo" / entry["reference"]; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prepared["package"] / entry["file"], target)
    files = ["repo/entry.py", *["repo/" + entry["reference"] for entry in chain["profiles"].values()]]
    conditioner = {}
    for original in (prepared["manifest_path"], prepared["checkpoint"]):
        relative = str(original.relative_to(prepared["root"]))
        target = source / "repo" / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        files.append("repo/" + relative); conditioner[relative] = execution.digest(target)
    manifest = input_manifest(prepared, source)
    files += [str(execution.native_file(source, name).relative_to(source)) for name in manifest["files"]]
    files += ["inputs/native_inputs.json", "execution_chain.json"]
    write(source / "execution_chain.json", chain); write(prepared["package"] / "execution_chain.json", chain)
    (source / "transfer_files.txt").write_text("\n".join(files) + "\n")
    record = {"status": "COMPLETE", "groups": chain["groups"], "profile_id": chain["profile"]["profile_id"],
              "source_commit": chain["source_commit"], "stage": "/tmp/NrnSift_fixture_inputs",
              "transfer_files_sha256": execution.digest(source / "transfer_files.txt"),
              "input_manifest_sha256": execution.digest(source / "inputs/native_inputs.json"),
              "transfer_bytes": sum((source / name).stat().st_size for name in files), "conditioner_sha256": conditioner}
    write(source / "staging_result.json", record)
    return source


def test_fit_preserves_nonzero_failure_outputs_and_explicit_conditioner_paths(prepared, monkeypatch):
    source = gpu_transfer_fixture(prepared)
    destination = prepared["tmp"] / "gpu_stage"
    monkeypatch.setattr(execution.socket, "gethostname", lambda: "gpu03")
    monkeypatch.setenv("SLURM_JOB_ID", "fixture_fit")
    seen = []
    def fake_process(argv, **kwargs):
        seen.append(argv)
        if argv[0] == "rsync":
            if any(arg.startswith("--files-from=") for arg in argv):
                for name in (destination / "transfer_files.txt").read_text().splitlines():
                    target = destination / name; target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source / name, target)
            else:
                for name in ("staging_result.json", "transfer_files.txt"):
                    shutil.copy2(source / name, destination / name)
            return subprocess.CompletedProcess(argv, 0)
        assert argv[0] == "/usr/bin/time"
        manifest = Path(argv[argv.index("--conditioning-manifest") + 1])
        assert manifest.is_relative_to(destination / "repo")
        assert manifest.is_file()
        assert Path(argv[argv.index("--conditioning-config") + 1]).is_file()
        write(destination / "repo" / prepared["chain"]["profile"]["training_manifest"], {"failure": "actual fixture fit failure"})
        return subprocess.CompletedProcess(argv, 7)
    monkeypatch.setattr(execution.subprocess, "run", fake_process)
    with pytest.raises(SystemExit) as result:
        execution.run_fit(prepared["package"], destination, "/tmp/NrnSift_fixture_inputs")
    assert result.value.code == 7
    durable = prepared["root"] / prepared["chain"]["profile"]["output_root"]
    record = execution.read_json(durable / "execution.json")
    assert record["status"] == "PROCESS_FAILED" and record["exit_code"] == 7
    assert record["source_commit"] == "committed_fixture" and record["attempt"] == "primary"
    assert execution.read_json(durable / "native_comparator_training_manifest.json")["failure"] == "actual fixture fit failure"
    assert sum(argv[0] == "/usr/bin/time" for argv in seen) == 1
    assert any(any(arg.startswith("--files-from=") for arg in argv) for argv in seen)


def test_fit_refuses_existing_primary_before_transfer(prepared, monkeypatch):
    durable = prepared["root"] / prepared["chain"]["profile"]["output_root"]
    durable.mkdir(parents=True); (durable / "original.txt").write_text("retained original")
    monkeypatch.setattr(execution.socket, "gethostname", lambda: "gpu03")
    monkeypatch.setattr(execution.subprocess, "run", lambda *args, **kwargs: pytest.fail("No transfer for an existing primary"))
    with pytest.raises(FileExistsError):
        execution.run_fit(prepared["package"], prepared["tmp"] / "new_gpu", "/tmp/NrnSift_fixture_inputs")
    assert (durable / "original.txt").read_text() == "retained original"


@pytest.mark.parametrize("raw", [b"../outside\n", b"/absolute\n", b"repo/a\nrepo/a\n", b""])
def test_transfer_list_rejects_escape_or_duplicate(raw):
    with pytest.raises(ValueError): execution.transfer_names(raw)


@pytest.mark.parametrize("shared_seed", [None, 41, 42, 43])
def test_prepare_uses_committed_snapshot_without_requiring_live_admission(prepared, monkeypatch, shared_seed):
    root, chain = prepared["root"], prepared["chain"]
    runner = root / execution.SUBMISSION / execution.RUNNER
    runner.parent.mkdir(parents=True)
    for name in execution.LAUNCHERS:
        (runner.parent / name).write_text("committed " + name)
    monkeypatch.setattr(execution, "__file__", str(runner))
    common = chain["conditioning_profile"]
    row = chain["profile"]
    row.update(conditioning_method=common["method"], conditioning_config=common["config_path"],
               conditioning_manifest=common["training_manifest"], training_seed_binding=chain["training_seed_binding"],
               budget={"selected_training_groups": 2})
    common.update(training_seed_binding=chain["training_seed_binding"], budget={"selected_training_groups": 2})
    comparison = {"scope_methods": {"state_single_source": "pure_shared"},
                  "conditioning_methods": {"pure_shared": "native_single_source_wam"}}
    reviews = {}
    if shared_seed is not None:
        row.update(profile_id=f"native_wam_fraction_100_seed_{shared_seed}", method="native_wam",
                   seed=shared_seed, source_scope="all", joint_belief_fit=True)
        prepared["config"]["native_profile_identity"].update(method="native_wam",seed=shared_seed)
        prepared["config"]["native_scientific_training"]["source_scope"]="all"
        from Libs.utils.data.raw_data_registry import load_config
        from Libs.utils.evaluation.manuscript_native_protocol import training_profile
        actual, actual_spec = training_profile(load_config("Libs/configs/exp/manuscript_native_protocol.yaml"),
                                               "native_wam", shared_seed)
        prepared["config"] = actual
        row.update(actual_spec)
        row["config_path"] = str(root / "profiles" / f"native_wam_{shared_seed}.yaml")
        chain["fit_argv"][chain["fit_argv"].index("--config")+1]=row["config_path"]
        chain["training_data_scope"] = native_scientific_source.training_data_scope(actual)
        chain["groups"]={"train":4000,"development":800}
        chain["training_seed_binding"]={"selected_formal_seed":shared_seed,"model_seed":shared_seed}
        reviews[row["profile_id"]]={"binding_kind":execution.SCHEMA,
            "execution_chain":"packages/"+row["profile_id"]+"/execution_chain.json",
            "component_review":"packages/"+row["profile_id"]+"/component_review.json"}
        chain["training_seed_binding"]=row["training_seed_binding"]
        row["budget"]["selected_training_groups"]=4000
    for profile, config in [(row, prepared["config"]), (common, prepared["common_config"])]:
        config.setdefault("manuscript_native_protocol",{}).update(source_comparison=comparison,
            seed_policy="manuscript_native_three",training_evidence={
                "approved_execution_profiles":{},"execution_review_profiles":reviews})
        config["native_scientific_training"]["checkpoint_storage"] = chain["checkpoint_storage"]
        Path(profile["config_path"]).write_text(yaml.safe_dump(config))
        profile["effective_configuration_sha256"] = execution.effective_digest(config)
    if shared_seed is None:
        monkeypatch.setattr(execution, "metadata_contract", lambda config, row: {
            key: chain[key] for key in ["group_uids", "groups", "training_seed_binding", "training_data_scope", "normalizer_reference"]})
    code_sha256 = {str(execution.SUBMISSION / name): execution.digest(runner.parent / name) for name in execution.LAUNCHERS}
    pinned = "Libs/scientific.py"; (root / pinned).write_text("pinned science")
    code_sha256[pinned] = execution.digest(root / pinned)
    archive = prepared["tmp"] / "committed.tar"
    with tarfile.open(archive, "w") as stream:
        for name in code_sha256: stream.add(root / name, arcname=name)
    binding = {"source_commit": "descendant_commit", "code_archive_name": archive.name,
               "code_archive_sha256": execution.digest(archive), "code_sha256": code_sha256}
    binding_path = prepared["tmp"] / "code_binding.json"; write(binding_path, binding)
    protocol = {"definition_validation": "PASS", "source_commit": "pinned_commit", "source_files": {pinned: code_sha256[pinned]},
                "profiles": [row, common], "commands": [{"command_uid": row["profile_id"] + ":fit", "profile_id": row["profile_id"],
                  "argv": chain["fit_argv"], "attempt_policy": {"primary_slots_per_fit": 1, "automatic_retries": False}}]}
    protocol_path = prepared["tmp"] / "protocol.json"; write(protocol_path, protocol)
    admission = prepared["tmp"] / "frozen_admission"; admission.mkdir()
    (admission / "stage_full_inputs.py").write_text("frozen admission checker")
    keys = ["admission_terminal_journal", "admission_job", "source_job", "frozen_source_commit", "admission_consumer_commit",
            "full_source_cases", "full_source_groups", "source_completion_receipt", "source_corpus", "registry_reference"]
    original = {key: "frozen_" + key for key in keys}
    original.update(groups=chain["groups"],source_worktree=str(root), registry_mapping={"snapshot_relative_directory": "admitted", "original_directory": str(root / "admitted")},
                    launcher_sha256={"stage_full_inputs.py": execution.digest(admission / "stage_full_inputs.py")})
    write(admission / "execution_chain.json", original)
    def git(argv, **kwargs):
        assert argv[:len(execution.GIT)] == execution.GIT
        argv = ["git", *argv[len(execution.GIT):]]
        if argv[1] == "show":
            data = (root / argv[2].split(":", 1)[1]).read_bytes()
        elif argv[1] == "ls-tree":
            data = b"".join(b"100644 blob " + hashlib.sha1(("blob " + str((root / name).stat().st_size) + "\0").encode()
                + (root / name).read_bytes()).hexdigest().encode() + b"\t" + name.encode() + b"\0" for name in code_sha256)
        else:
            assert argv[1:3] == ["merge-base", "--is-ancestor"]
            data = b""
        return subprocess.CompletedProcess(argv, 0, stdout=data)
    monkeypatch.setattr(execution.subprocess, "run", git)
    output = prepared["tmp"] / "comparison_execution"
    options={}
    if shared_seed is not None:
        options={"shared_input_stage":"/tmp/NrnSift_native_training_inputs_612799","shared_stage_job":"612799"}
    monkeypatch.setattr("sys.argv", ["fixture", "prepare", "--protocol",str(protocol_path),
        "--admission-package",str(admission),"--code-binding",str(binding_path),
        "--profile-id",row["profile_id"],"--destination",str(output),
        *(["--shared-input-stage",options["shared_input_stage"],"--shared-stage-job","612799"] if options else [])])
    execution.main()
    assert output.is_dir()
    actual = execution.read_json(output / "execution_chain.json")
    assert actual["source_commit"] == "descendant_commit" and actual["protocol_source_commit"] == "pinned_commit"
    assert actual["stage_argv"][1] == execution.PRETRAINER
    assert actual["fit_argv"][1].endswith("run_external_baseline_bridge.py")
    if shared_seed is not None:
        assert actual["component_review_status"] == "PENDING"
        assert actual["component_review"] == reviews[row["profile_id"]]["component_review"]
        assert actual["shared_input_stage"]["job_id"] == "612799"
        with pytest.raises(FileNotFoundError):execution.approved_shared_fit(output,actual)
        assert not (root/row["output_root"]).exists()
    assert actual["scientific_twin"] == "NOT_RUN"
    assert not (root / original["admission_terminal_journal"]).exists()
    binding["code_sha256"].pop(str(execution.SUBMISSION / execution.RUNNER)); write(binding_path, binding)
    with pytest.raises(ValueError, match="committed snapshot"):
        execution.prepare(protocol_path, admission, binding_path, row["profile_id"], prepared["tmp"] / "rejected", **options)
    assert not (prepared["tmp"] / "rejected").exists()


@pytest.mark.parametrize("name,mode,expected", [
    ("Libs/utils/model/native_scientific_source.py", b"100644", True),
    ("Libs/configs/exp/manuscript_native_protocol.yaml", b"100644", True),
    ("Libs/exp/submission/stage_native_source_comparison.sbatch", b"100644", True),
    ("Libs/external/reference/source.py", b"100644", False),
    ("Libs/external/reference", b"160000", False),
    ("Libs/tests/fixtures/checkpoint.pt", b"100644", False),
    ("Input/raw/source.py", b"100644", False),
    ("Output/results/checkpoint.pt", b"100644", False),
])
def test_committed_archive_membership_excludes_data_and_external_payloads(name, mode, expected):
    assert execution.executable_source(name, mode) is expected


@pytest.mark.parametrize("fault", [None, "unfinished", "wrong_job", "wrong_consumer", "wrong_source", "wrong_stage"])
def test_direct_shared_stage_requires_actual_completed_source_data_identity(prepared, fault):
    chain = prepared["chain"]
    chain["shared_input_stage"] = {"host":"cpu03", "job_id":"612799", "stage":"/tmp/NrnSift_native_training_inputs_612799"}
    chain["admission"] = {"admission_job":"612505", "source_job":"594138", "frozen_source_commit":"frozen",
        "admission_consumer_commit":"reviewed", "full_source_cases":63232, "full_source_groups":22144}
    data = {**chain["admission"], "source_commit":"c3104b7"}
    staged = {"status":"COMPLETE", "host":"cpu03", "job_id":"612799", "stage":chain["shared_input_stage"]["stage"],
        "groups":chain["groups"], "roles":["train","development"], "source_commit":"c3104b7",
        "source_generation":False, "scientific_fit":False,
        "admission_terminal":{"stage":"source_and_admissions_complete", "job_id":"612505",
            "source_generation_job":"594138", "source_producer_commit":"frozen", "consumer_commit":"reviewed",
            "source_generation":"NOT_REPLAYED", "admitted_count":63232},
        "source_terminal":{"status":"complete", "source_status":"complete", "stop_group_index":22144,
            "source_identity":{"producer_commit":"frozen"}}}
    if fault == "unfinished": staged["status"] = "STARTED"
    elif fault == "wrong_job": staged["job_id"] = "602569"
    elif fault == "wrong_consumer": data["admission_consumer_commit"] = "old"
    elif fault == "wrong_source": staged["source_terminal"]["source_identity"]["producer_commit"] = "other"
    elif fault == "wrong_stage": staged["stage"] = "/tmp/NrnSift_native_fit_612919"
    if fault:
        with pytest.raises(ValueError): execution.validate_shared_stage(chain, staged, data, chain["shared_input_stage"]["stage"])
    else:
        execution.validate_shared_stage(chain, staged, data, chain["shared_input_stage"]["stage"])


def test_direct_shared_mainline_fit_transfers_inputs_and_new_code_without_firstfit(prepared, monkeypatch):
    staging_archive(prepared)
    chain = prepared["chain"]; package = prepared["package"]
    chain["profile"].update(method="native_wam", joint_belief_fit=True)
    chain["input_mode"] = "completed_shared_stage"
    chain["component_review"] = "review.json"
    chain["profile_sha256"] = "fixture_profile_hash"
    chain["effective_configuration_sha256"] = "fixture_config_hash"
    stage = prepared["tmp"] / "shared_cpu"
    manifest = input_manifest(prepared, stage)
    data_chain = {"source_commit":"c3104b7"}
    write(stage / "execution_chain.json", data_chain)
    names = [str(execution.native_file(stage, logical).relative_to(stage)) for logical in manifest["files"]]
    (stage / "transfer_files.txt").write_text("\n".join([*names, "repo/old_unused.py"]) + "\n")
    shared = "/tmp/NrnSift_native_training_inputs_612799"
    chain["shared_input_stage"] = {"stage":shared, "job_id":"612799", "execution_chain_sha256":execution.digest(stage / "execution_chain.json")}
    staged = {"transfer_files_sha256":execution.digest(stage / "transfer_files.txt"),
        "input_manifest_sha256":execution.digest(stage / "inputs/native_inputs.json"),
        "input_file_count":manifest["file_count"], "input_bytes":manifest["size_bytes"]}
    write(stage / "staging_result.json", staged)
    write(package / "execution_chain.json", chain)
    write(prepared["root"] / "review.json", {"status":"APPROVED_COMPONENT", "profile_id":chain["profile"]["profile_id"],
        "source_commit":chain["source_commit"], "execution_chain_sha256":execution.digest(package / "execution_chain.json"),
        "profile_sha256":chain["profile_sha256"], "effective_configuration_sha256":chain["effective_configuration_sha256"]})
    monkeypatch.setattr(execution.socket, "gethostname", lambda:"gpu03")
    monkeypatch.setenv("SLURM_JOB_ID", "fixture_mainline")
    monkeypatch.setattr(execution, "require_serial_fit", lambda:{"native_fit_squeue":""})
    monkeypatch.setattr(execution, "validate_shared_stage", lambda *args:None)
    # Profile matching remains an existing separately covered metadata contract.
    monkeypatch.setattr(execution, "configuration", lambda *args:prepared["config"])
    destination = prepared["tmp"] / "mainline_gpu"
    calls=[]
    def process(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "rsync":
            filelist = next((a for a in argv if a.startswith("--files-from=")), None)
            if filelist:
                transferred=Path(filelist.split("=",1)[1]).read_text().splitlines()
                assert transferred == names and "repo/old_unused.py" not in transferred
                for name in transferred:
                    target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(stage/name,target)
            else:
                source=next(a for a in argv if a.startswith("cpu03:"));relative=source.split(shared+"/",1)[1]
                shutil.copy2(stage/relative,Path(argv[-1]))
            return subprocess.CompletedProcess(argv,0)
        assert argv[0] == "/usr/bin/time"
        assert Path(argv[argv.index("--native-input-manifest")+1]) == destination/"inputs/native_inputs.json"
        assert (destination/"repo/entry.py").read_text() == "fixture committed code\n"
        assert not (destination/"repo/old_unused.py").exists()
        assert not any("612919" in a for a in argv)
        return subprocess.CompletedProcess(argv,0)
    monkeypatch.setattr(execution.subprocess,"run",process)
    record=execution.run_fit(package,destination,shared)
    assert record["status"] == "PROCESS_COMPLETED"
    assert sum(argv[0] == "/usr/bin/time" for argv in calls) == 1
    assert execution.read_json(destination/"data_execution_chain.json") == data_chain
    assert execution.read_json(destination/"execution_chain.json") == chain
    # Removing independent approval rejects before consuming a new primary.
    (prepared["root"]/"review.json").unlink()
    with pytest.raises(FileNotFoundError): execution.run_fit(package,prepared["tmp"]/"rejected_mainline",shared)


def test_shared_review_reference_rejects_missing_malformed_and_historical_approval():
    profile="native_wam_fraction_100_seed_41"
    current={"binding_kind":execution.SCHEMA,"execution_chain":"package/execution_chain.json",
             "component_review":"package/component_review.json"}
    cfg={"manuscript_native_protocol":{"seed_policy":"manuscript_native_three",
        "training_evidence":{"approved_execution_profiles":{},"execution_review_profiles":{profile:current}}}}
    assert execution.shared_review_reference(cfg,profile)==current["component_review"]
    for change in (None,{**current,"component_review":"../outside"},{**current,"binding_kind":"unknown"}):
        mutated=copy.deepcopy(cfg)
        mutated["manuscript_native_protocol"]["training_evidence"]["execution_review_profiles"]={profile:change}
        with pytest.raises(ValueError):execution.shared_review_reference(mutated,profile)
    legacy={"execution_chain":"old/execution_chain.json","source_commit":"a"*40,
            "profile_sha256":"b"*64,"effective_configuration_sha256":"c"*64}
    cfg["manuscript_native_protocol"]["training_evidence"]["approved_execution_profiles"][profile]=legacy
    with pytest.raises(ValueError,match="Historical approval"):
        execution.shared_review_reference(cfg,profile)
    cfg["manuscript_native_protocol"]["seed_policy"]="manuscript_native_five"
    assert execution.shared_review_reference(cfg,profile)==current["component_review"]


def test_all_source_staged_normalizer_uses_actual_registry_and_rejects_identity_drift(tmp_path):
    import numpy as np
    from Libs.utils.data.raw_data_registry import load_config
    from Libs.utils.data import native_diffusion_data as native
    from Libs.utils.data.native_observable_profile import observable_profile,family_grid,quantity_support
    from Libs.utils.data.native_field_normalization import fit_multigrid,bind_artifact
    from Libs.utils.evaluation.manuscript_native_protocol import training_profile
    from Libs.utils.model.native_scientific_source import selected_group_definitions
    from Libs.scripts.builders.build_controlled_native_source import scientific_index
    base=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    base["native_observable_bridge"]["group_partitions"]=[
        dict(base["native_observable_bridge"]["group_partitions"][0],groups_per_family=1),
        dict(base["native_observable_bridge"]["group_partitions"][1],groups_per_family=1)]
    config,row=training_profile(base,"native_wam",41)
    profile=observable_profile(config)
    groups=selected_group_definitions(config,"train")
    batches=[]
    for family in profile["families"]:
        grid=family_grid(profile,family)
        group=next(g for g in groups if g["family_uid"]==family)
        support=quantity_support(grid,group["source_uid"])
        field=np.where(support,.125,0.).astype(np.float64)
        teacher={"trajectory_uid":family+":fixture", "source_uid":group["source_uid"],
            "sample_uid":group["sample_uid"],"regimen_uid":family+":action",
            "posterior_uid":group["posterior_uid"],"family_uid":family,
            "content_hash":hashlib.sha256(field.tobytes()).hexdigest(),
            "trajectory_store_path":family,"trajectory_index":0,
            "source_snapshot_ref":group["source_snapshot_ref"],
            "split_name":"train","source_holdout_role":"train","qc_grade":"A","sample_weight":1.}
        record={"teacher":teacher,"joint_source":{"source_posterior_uid":group["posterior_uid"],
            "support":{"sample_uids":[group["sample_uid"]],"support_uid":group["support_uid"],
                       "particle_uids":[group["particle_uids"]],
                       "initial_field":[np.broadcast_to(field[0],(32,*field.shape[1:])).tolist()]}}}
        batches.append((grid,[record],{"trajectory":field[None],"quantity_support":support[None]}))
    artifact_path=tmp_path/"source"/"shared.json";artifact_path.parent.mkdir()
    artifact_path.write_text(json.dumps(fit_multigrid(batches,config)))
    references={}
    for grid,_,_ in batches:
        bound=bind_artifact(grid,artifact_path,config)
        reference=artifact_path.parent/(grid["family_uid"]+".json")
        reference.write_text(json.dumps({"schema_version":native.SCHEMA_VERSION,"grid":bound}))
        references[grid["family_uid"]]=str(reference)
    registry_path=artifact_path.parent/"model_registry.json"
    registry_path.write_text(json.dumps(scientific_index(config,references)))
    config["native_diffusion_data"]["registry_path"]=str(registry_path)
    row["effective_configuration_sha256"]=execution.effective_digest(config)
    stage=tmp_path/"stage"
    profile_reference="profiles/actual.yaml"
    profile_path=stage/"repo"/profile_reference;profile_path.parent.mkdir(parents=True)
    profile_path.write_text(yaml.safe_dump(config))
    row["config_path"]=profile_reference
    chain={**execution.metadata_contract(config,row),"profile":row,
        "admission":{"registry_reference":str(registry_path)},
        "profiles":{profile_reference:{"reference":profile_reference}}}
    files={}
    for original in [artifact_path,registry_path,*map(Path,references.values())]:
        target=execution.native_file(stage,str(original));target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(original,target)
        files[str(original)]={"size_bytes":target.stat().st_size,"sha256":execution.digest(target)}
    manifest={"schema_version":"native_input_locations.v1","aliases":{},"files":files,
        "stores":[],"groups":chain["group_uids"],"file_count":len(files),
        "size_bytes":sum(v["size_bytes"] for v in files.values())}
    write(stage/"inputs/native_inputs.json",manifest)
    assert execution.verify_normalizer(stage,manifest,chain)==execution.digest(artifact_path)
    missing=copy.deepcopy(manifest);missing["files"].pop(str(artifact_path))
    write(stage/"inputs/native_inputs.json",missing)
    with pytest.raises(FileNotFoundError):execution.verify_normalizer(stage,missing,chain)
    write(stage/"inputs/native_inputs.json",manifest)
    drift=copy.deepcopy(chain);drift["admission"]["registry_reference"]="other_registry.json"
    with pytest.raises(ValueError,match="registry identity"):
        execution.verify_normalizer(stage,manifest,drift)
    staged_artifact=execution.native_file(stage,str(artifact_path))
    staged_artifact.write_bytes(staged_artifact.read_bytes()+b" ")
    with pytest.raises(ValueError,match="type/size changed|bytes changed"):
        execution.verify_normalizer(stage,manifest,chain)
