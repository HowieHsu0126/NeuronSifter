"""Prepare, stage, or run one matched-source primary fit. Never submit jobs.

Preparation accepts materialized protocol profiles and a committed code archive
binding with source_commit, code_archive_name, code_archive_sha256, code_sha256.
The archive binds all committed Python/YAML source under Libs plus this runner's
two sbatch wrappers. It excludes external repositories, data and result artifacts. Preparation does
not require completed source admission. Stage and fit require actual receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
import time

import yaml

RUNNER = "native_source_comparison_execution.py"
LAUNCHERS = (RUNNER, "stage_native_source_comparison.sbatch", "fit_native_source_comparison.sbatch")
SUBMISSION = Path("Libs/exp/submission")
PRETRAINER = "Libs/exp/training/pretrain_microenvironment_world_model.py"
SCHEMA = "native_source_comparison_execution.v1"
SSH = "ssh -o BatchMode=yes -o ConnectTimeout=10"
GIT = ["git", "-c", "gc.auto=0", "-c", "maintenance.auto=false"]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require_digest(path: Path, expected: str) -> None:
    if digest(path) != expected:
        raise ValueError("Execution bytes changed: " + str(path))


def local_reference(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Execution reference must stay inside its declared repository: " + value)
    return root / path


def option(argv: list[str], name: str) -> str:
    if argv.count(name) != 1:
        raise ValueError("Expected one explicit command option: " + name)
    return argv[argv.index(name) + 1]


def effective_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def configuration(path: Path, row: dict) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("config_includes"):
        raise ValueError("Comparison needs the standalone materialized profile")
    identity = payload["native_profile_identity"]
    if (identity["method"] != row["method"] or identity["seed"] != row["seed"]
            or identity["percentage"] != row["training_subset_percentage"]
            or payload["native_scientific_training"]["source_scope"] != row["source_scope"]
            or effective_digest(payload) != row["effective_configuration_sha256"]):
        raise ValueError("Comparison profile method, seed, scope or effective identity differs")
    return payload


def metadata_contract(config: dict, row: dict) -> dict:
    # These maintained helpers define group identity, seed and checkpoint profile.
    # They inspect declarations only; they do not read source arrays or models.
    from Libs.utils.exp.experiment_manifests import resolve_training_seed
    from Libs.utils.model.native_scientific_source import selected_group_definitions, training_data_scope

    groups = {role: sorted(item["sample_uid"] for item in selected_group_definitions(config, role))
              for role in ("train", "development")}
    if len(groups["train"]) != row["budget"]["selected_training_groups"]:
        raise ValueError("Selected comparison groups differ from the frozen profile budget")
    seed = resolve_training_seed(config, require_selected=True)
    if seed != row["training_seed_binding"]:
        raise ValueError("Comparison seed differs from the frozen profile binding")
    normalization = config["native_diffusion_data"]["field_normalization"]
    if normalization.get("fit_role") != "train":
        raise ValueError("Comparison normalizer is not training-only")
    scope = config["native_scientific_training"]["source_scope"]
    if scope == "all":
        if "model_artifact_ref" in normalization:
            raise ValueError("All-source fit must use its source-registry shared transform")
        registry_reference = config["native_diffusion_data"]["registry_path"]
        if not isinstance(registry_reference, str) or not registry_reference:
            raise ValueError("All-source normalization requires an explicit registry reference")
        normalizer = {"normalizer_binding": {
            "kind": "source_registry_shared", "registry_reference": registry_reference}}
    else:
        normalizer = {"normalizer_reference": normalization["model_artifact_ref"]}
    return {"group_uids": groups, "groups": {key: len(value) for key, value in groups.items()},
            "training_data_scope": training_data_scope(config), "training_seed_binding": seed,
            **normalizer}


def executable_source(name: str, mode: bytes) -> bool:
    """Retain the first-fit Python/YAML archive rule and the two owned wrappers."""
    path = Path(name)
    return (mode in {b"100644", b"100755"} and path.parts[0] == "Libs"
            and not path.is_relative_to("Libs/external")
            and (path.suffix in {".py", ".yaml"} or name in {str(SUBMISSION / value) for value in LAUNCHERS}))


def shared_review_reference(config: dict, profile_id: str) -> str:
    """Resolve a declared review output without granting shared execution."""
    evidence = config["manuscript_native_protocol"]["training_evidence"]
    approved = evidence.get("approved_execution_profiles", {}).get(profile_id)
    prepared = evidence.get("execution_review_profiles", {}).get(profile_id)
    current_fields = {"binding_kind", "execution_chain", "component_review"}
    legacy_fields = {"execution_chain", "source_commit", "profile_sha256", "effective_configuration_sha256"}
    if approved is not None:
        if not isinstance(approved, dict):
            raise ValueError("Shared execution authority must be a typed mapping")
        if set(approved) == current_fields:
            prepared = approved
        elif set(approved) == legacy_fields:
            if config["manuscript_native_protocol"].get("seed_policy", "manuscript_native_five") != "manuscript_native_five":
                raise ValueError("Historical approval cannot authorize an active three-seed profile")
            # Historical hashes do not approve the separately prepared shared fit.
        else:
            raise ValueError("Shared execution authority schema mismatch")
    if (not isinstance(prepared, dict) or set(prepared) != current_fields
            or prepared["binding_kind"] != SCHEMA):
        raise ValueError("Shared profile needs an explicit typed review-output reference")
    for key in ("execution_chain", "component_review"):
        value = prepared[key]
        if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError("Shared review output must be a repository-relative reference")
    return prepared["component_review"]


def prepare(protocol_path: Path, admission_package: Path, code_binding_path: Path,
            profile_id: str, destination: Path, *, shared_input_stage: str | None = None,
            shared_stage_job: str | None = None) -> Path:
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    protocol = read_json(protocol_path)
    if protocol.get("definition_validation") != "PASS":
        raise ValueError("Comparison protocol definition did not pass")
    rows = [row for row in protocol["profiles"] if row["profile_id"] == profile_id]
    if len(rows) != 1:
        raise ValueError("Comparison profile is missing or ambiguous")
    row = rows[0]
    config = configuration(Path(row["config_path"]), row)
    comparison = config["manuscript_native_protocol"]["source_comparison"]
    shared = shared_input_stage is not None or shared_stage_job is not None
    if shared:
        if (row["method"] != "native_wam" or row["source_scope"] != "all"
                or row["joint_belief_fit"] is not True or row["training_subset_percentage"] != 100
                or shared_input_stage != "/tmp/NrnSift_native_training_inputs_" + str(shared_stage_job)
                or not str(shared_stage_job).isdigit()):
            raise ValueError("Direct shared stage requires a full joint native_wam profile and exact CPU03 stage/job")
    elif (row["method"] not in comparison["scope_methods"]
            or comparison["scope_methods"][row["method"]] != row["source_scope"]
            or row["training_subset_percentage"] != 100):
        raise ValueError("Profile is not a declared full-fraction matched-source method")
    commands = [command for command in protocol["commands"] if command["command_uid"] == profile_id + ":fit"]
    if len(commands) != 1 or commands[0]["profile_id"] != profile_id:
        raise ValueError("Comparison fit command is missing or ambiguous")
    command = commands[0]
    if option(command["argv"], "--config") != row["config_path"]:
        raise ValueError("Fit argv refers to another scientific profile")
    binding = read_json(code_binding_path)
    # A launcher-only descendant commit keeps the materialized scientific profiles.
    for name, expected in protocol["source_files"].items():
        if binding["code_sha256"].get(name) != expected:
            raise ValueError("Committed snapshot changes a pinned scientific file: " + name)
    for name in LAUNCHERS:
        path = root / SUBMISSION / name
        expected = binding["code_sha256"].get(str(SUBMISSION / name))
        if expected is None:
            raise ValueError("Canonical comparison launcher is not in the committed snapshot")
        require_digest(path, expected)
        committed = subprocess.run([*GIT, "show", binding["source_commit"] + ":" + str(SUBMISSION / name)],
                                   cwd=root, check=True, capture_output=True).stdout
        if hashlib.sha256(committed).hexdigest() != expected:
            raise ValueError("Comparison launcher binding does not match committed code")
    subprocess.run([*GIT, "merge-base", "--is-ancestor", protocol["source_commit"], binding["source_commit"]],
                   cwd=root, check=True, capture_output=True)
    declared = subprocess.run([*GIT, "ls-tree", "-rz", binding["source_commit"]],
                              cwd=root, check=True, capture_output=True).stdout
    committed_blobs = {entry.split(b"\t", 1)[1].decode(): entry.split(b"\t", 1)[0].split()[2].decode()
                       for entry in declared.split(b"\0") if entry
                       and executable_source(entry.split(b"\t", 1)[1].decode(), entry.split(b" ", 1)[0])}
    if set(committed_blobs) != set(binding["code_sha256"]):
        raise ValueError("Comparison snapshot must declare the complete committed executable source file set")
    archive = code_binding_path.parent / binding["code_archive_name"]
    require_digest(archive, binding["code_archive_sha256"])
    with tarfile.open(archive) as snapshot:
        members = [member for member in snapshot.getmembers() if member.isfile()]
        if len(members) != len(committed_blobs) or {member.name for member in members} != set(committed_blobs):
            raise ValueError("Prepared archive differs from the complete committed executable source file set")
        for member in members:
            blob = hashlib.sha1(("blob " + str(member.size) + "\0").encode())
            sha256 = hashlib.sha256()
            with snapshot.extractfile(member) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    blob.update(chunk); sha256.update(chunk)
            if blob.hexdigest() != committed_blobs[member.name] or sha256.hexdigest() != binding["code_sha256"][member.name]:
                raise ValueError("Archive bytes differ from the actual committed code: " + member.name)
    original = read_json(admission_package / "execution_chain.json")
    admission_keys = ("admission_terminal_journal", "admission_job", "source_job", "frozen_source_commit",
                      "admission_consumer_commit", "full_source_cases", "full_source_groups",
                      "source_completion_receipt", "source_corpus", "registry_reference")
    if Path(original["source_worktree"]).resolve() != root:
        raise ValueError("Admission binding belongs to another worktree")
    chain = {"schema_version": SCHEMA, "status": "PREPARED", "scientific_twin": "NOT_RUN",
             "source_worktree": str(root), "source_commit": binding["source_commit"],
             "protocol_source_commit": protocol["source_commit"], "profile": row,
             "fit_argv": command["argv"], "attempt_policy": command["attempt_policy"],
             "admission": {key: original[key] for key in admission_keys},
             "registry_mapping": original["registry_mapping"],
             "checkpoint_storage": config["native_scientific_training"]["checkpoint_storage"],
             **{key: binding[key] for key in ("code_archive_name", "code_archive_sha256", "code_sha256")},
             **metadata_contract(config, row)}
    if shared:
        if chain["groups"] != original["groups"] or chain["groups"] != {"train": 4000, "development": 800}:
            raise ValueError("Shared mainline stage must retain TRAIN4000/DEV800")
        review_reference = shared_review_reference(config, profile_id)
        chain.update(input_mode="completed_shared_stage", shared_input_stage={
            "host": "cpu03", "job_id": str(shared_stage_job), "stage": shared_input_stage,
            "execution_chain_sha256": digest(admission_package / "execution_chain.json"),
            "execution_chain_reference": str(admission_package / "execution_chain.json")},
            component_review=review_reference, component_review_status="PENDING", frozen_package=str(destination.resolve()),
            profile_file=Path(row["config_path"]).name, profile_reference=row["config_path"],
            profile_sha256=digest(Path(row["config_path"])),
            effective_configuration_sha256=row["effective_configuration_sha256"],
            output_reference=row["output_root"])
    profile_paths = [Path(row["config_path"])]
    if not row["joint_belief_fit"]:
        common = [item for item in protocol["profiles"] if item["method"] == row["conditioning_method"]
                  and item["seed"] == row["seed"] and item["training_subset_percentage"] == 100]
        if len(common) != 1 or not common[0]["joint_belief_fit"] or common[0]["source_scope"] != row["source_scope"]:
            raise ValueError("State comparison has no exact same-seed source conditioner")
        chain["conditioning_profile"] = common[0]
        profile_paths.append(Path(row["conditioning_config"]))
        common_config = configuration(profile_paths[-1], common[0])
        if metadata_contract(common_config, common[0]) != metadata_contract(config, row):
            raise ValueError("State and conditioner group, seed or normalizer bindings differ")
    chain["profiles"] = {str(path): {"reference": str(path.resolve().relative_to(root)),
                                     "file": path.name, "sha256": digest(path)} for path in profile_paths}
    chain["stage_argv"] = ["python", PRETRAINER, "--config", row["config_path"],
                           "--native-input-role", "train", "--native-input-role", "development"]
    chain["launcher_sha256"] = {name: binding["code_sha256"][str(SUBMISSION / name)] for name in LAUNCHERS}
    admission_script = admission_package / "stage_full_inputs.py"
    chain["admission_preflight_sha256"] = original["launcher_sha256"]["stage_full_inputs.py"]
    require_digest(admission_script, chain["admission_preflight_sha256"])
    destination.mkdir(parents=True, exist_ok=False)
    for name in LAUNCHERS:
        shutil.copy2(root / SUBMISSION / name, destination / name)
    for path in profile_paths:
        shutil.copy2(path, destination / path.name)
    shutil.copy2(archive, destination / chain["code_archive_name"])
    shutil.copy2(admission_script, destination / "admission_preflight.py")
    write_json(destination / "execution_chain.json", chain)
    return destination


def package_chain(base: Path) -> dict:
    chain = read_json(base / "execution_chain.json")
    if chain.get("schema_version") != SCHEMA:
        raise ValueError("Comparison execution schema mismatch")
    for name in LAUNCHERS:
        require_digest(base / name, chain["launcher_sha256"][name])
    return chain


def admission_preflight(base: Path, chain: dict) -> dict:
    path = base / "admission_preflight.py"
    require_digest(path, chain["admission_preflight_sha256"])
    spec = importlib.util.spec_from_file_location("comparison_admission_preflight", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_complete_admissions(chain["admission"], Path(chain["source_worktree"]))


def verify_groups(manifest: dict, chain: dict, reused: dict | None = None) -> None:
    selected = manifest["groups"]
    if set(selected) != {"train", "development"}:
        raise ValueError("Comparison input roles must be TRAIN and DEV only")
    for role, expected in chain["group_uids"].items():
        actual = selected[role]
        if len(actual) != len(set(actual)) or sorted(actual) != expected:
            raise ValueError("Comparison input identities differ from the declared " + role + " groups")
        if reused is not None and not set(actual) <= set(reused["groups"].get(role, [])):
            raise ValueError("Comparison group is outside the actual reusable " + role + " manifest")
    if set(selected["train"]) & set(selected["development"]):
        raise ValueError("Comparison TRAIN and DEV group overlap")


def validate_conditioner(manifest_path: Path, chain: dict, config: dict, normalization_id: str) -> Path:
    """Check actual writer metadata before any checkpoint transfer or tensor load.

    Minimum budget attainment is required here. It does not change the writer's
    sufficient_training=False or establish scientific training sufficiency.
    """
    from Libs.utils.model.native_scientific_source import native_training_profile
    from Libs.utils.model.native_training_selection import training_checkpoint_path

    manifest = read_json(manifest_path)
    selection = manifest.get("selection", {})
    common = chain["conditioning_profile"]
    policy = config["native_scientific_training"]["selection"]
    last, best = selection.get("last"), selection.get("best")
    if (manifest.get("failure") is not None or type(manifest.get("actual_total_parameter_count")) is not int
            or manifest["actual_total_parameter_count"] <= 0
            or selection.get("terminal") not in {"INCOMPLETE_BUDGET_CAP", "INCOMPLETE_EPOCH_CAP",
                                                "DEVELOPMENT_PLATEAU_REQUIRES_SUFFICIENCY_REVIEW"}
            or selection.get("selection_role") != "development" or not last or not best):
        raise ValueError("Conditioner has no finalized failure-free development-selected fit")
    if (manifest.get("executed_steps", -1) < policy["minimum_steps"]
            or last.get("steps", -1) != manifest["executed_steps"]
            or last.get("epoch", -1) < policy["minimum_epochs"]):
        raise ValueError("Conditioner has insufficient executed optimizer steps or epochs")
    if (manifest.get("training_seed_binding") != chain["training_seed_binding"]
            or manifest.get("checkpoint_scale") != common["checkpoint_scale"]
            or manifest.get("inference_profile") != common["inference_profile"]
            or manifest.get("train_group_count") != chain["groups"]["train"]
            or manifest.get("development_group_count") != chain["groups"]["development"]):
        raise ValueError("Conditioner seed, scale, inference profile or groups differ")
    profile = manifest.get("native_scientific_profile", {})
    if profile != native_training_profile(config, normalization_id):
        raise ValueError("Conditioner scientific profile or source scope differs")
    # Current writer references move with the unmodified manifest. Repository or
    # absolute historical references require an explicit migration before use.
    reference = Path(best["checkpoint_path"])
    if (selection.get("checkpoint_path_base") != "manifest_directory" or reference.is_absolute()
            or ".." in reference.parts):
        raise ValueError("Conditioner checkpoint must use current portable manifest-directory references")
    checkpoint = training_checkpoint_path(manifest_path, selection, str(reference))
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise ValueError("Selected conditioner checkpoint is absent or not a regular file")
    return checkpoint


def copy_checked(source: Path, target: Path) -> str:
    expected = digest(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    require_digest(target, expected)
    return expected


def stage_command(chain: dict, code: Path, inputs: Path, reuse_manifest: Path) -> list[str]:
    argv = list(chain["stage_argv"])
    if argv[1] != PRETRAINER:
        raise ValueError("Comparison staging must use the explicit maintained pretrainer entry point")
    argv[0], argv[1] = sys.executable, str(code / argv[1])
    name = option(argv, "--config")
    argv[argv.index("--config") + 1] = str(code / chain["profiles"][name]["reference"])
    return [*argv, "--stage-native-inputs", str(inputs), "--reuse-native-inputs", str(reuse_manifest)]


def fit_command(chain: dict, code: Path, manifest: Path) -> list[str]:
    argv = list(chain["fit_argv"])
    argv[0], argv[1] = sys.executable, str(code / argv[1])
    for flag in ("--config", "--conditioning-config"):
        if flag in argv:
            name = option(argv, flag)
            argv[argv.index(flag) + 1] = str(code / chain["profiles"][name]["reference"])
    if "--conditioning-manifest" in argv:
        name = option(argv, "--conditioning-manifest")
        argv[argv.index("--conditioning-manifest") + 1] = str(local_reference(code, name))
    return [*argv, "--native-input-manifest", str(manifest)]


def native_file(stage: Path, logical: str) -> Path:
    path = Path(logical)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Invalid native logical path")
    return stage / "inputs/files" / path.relative_to("/")


def verify_shared_normalizer(stage: Path, manifest: dict, chain: dict) -> str:
    """Validate the exact shared transform through the maintained staged loader."""
    from Libs.utils.data.native_input_staging import NativeInputLocations
    from Libs.utils.data.native_diffusion_data import load_family_registries
    from Libs.utils.data.native_field_normalization import normalization_spec

    binding = chain["normalizer_binding"]
    if (not isinstance(binding, dict) or set(binding) != {"kind", "registry_reference"}
            or binding["kind"] != "source_registry_shared"):
        raise ValueError("Shared normalizer binding schema mismatch")
    profile = chain["profiles"][chain["profile"]["config_path"]]
    config = configuration(local_reference(stage / "repo", profile["reference"]), chain["profile"])
    if (config["native_scientific_training"]["source_scope"] != "all"
            or config["native_diffusion_data"]["registry_path"] != binding["registry_reference"]
            or chain["admission"]["registry_reference"] != binding["registry_reference"]):
        raise ValueError("Shared normalizer source/profile registry identity differs")
    locations = NativeInputLocations.load(stage / "inputs/native_inputs.json")
    if locations.files != manifest["files"] or locations.groups != manifest["groups"]:
        raise ValueError("Shared normalizer input manifest identity differs")
    registries = load_family_registries(config, input_locations=locations)
    if set(registries) != set(chain["profile"]["family_scope"]):
        raise ValueError("Shared normalizer must retain all declared family grids")
    specifications = [normalization_spec(registry["grid"], config, input_locations=locations)
                      for registry in registries.values()]
    identities = {(spec["kind"], spec.get("artifact_ref"), spec.get("sha256")) for spec in specifications}
    if len(identities) != 1 or specifications[0]["kind"] != "training_scaler":
        raise ValueError("Every family must use the same bound shared training transform")
    artifact = specifications[0]["artifact"]
    if {row["sample_uid"] for row in artifact["training_records"]} != set(chain["group_uids"]["train"]):
        raise ValueError("Shared normalizer must use exactly the admitted complete training groups")
    return specifications[0]["sha256"]


def verify_normalizer(stage: Path, manifest: dict, chain: dict) -> str:
    if "normalizer_binding" in chain:
        return verify_shared_normalizer(stage, manifest, chain)
    logical = manifest["aliases"][chain["normalizer_reference"]]
    metadata = manifest["files"][logical]
    path = native_file(stage, logical)
    require_digest(path, metadata["sha256"])
    normalizer = read_json(path)
    if (normalizer.get("fit_role") != "train"
            or normalizer.get("training_data_scope") != chain["training_data_scope"]
            or {row["sample_uid"] for row in normalizer["training_records"]} != set(chain["group_uids"]["train"])):
        raise ValueError("Actual scoped normalizer is not bound to exactly the admitted training groups")
    return metadata["sha256"]


def stage_inputs(base: Path, destination: Path, reuse_manifest: Path, evidence: Path) -> dict:
    if socket.gethostname() != "cpu03":
        raise RuntimeError("Comparison staging requires the admitted source host cpu03")
    chain = package_chain(base)
    destination.mkdir(parents=True, exist_ok=False)
    evidence.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {"status": "STARTED", "job_id": os.environ["SLURM_JOB_ID"], "host": "cpu03",
              "stage": str(destination), "source_commit": chain["source_commit"],
              "profile_id": chain["profile"]["profile_id"], "source_generation": False, "scientific_fit": False}
    try:
        record.update(admission_preflight(base, chain))
        reused = read_json(reuse_manifest)
        verify_groups({"groups": chain["group_uids"]}, chain, reused)
        archive = base / chain["code_archive_name"]
        require_digest(archive, chain["code_archive_sha256"])
        code = destination / "repo"
        with tarfile.open(archive) as package:
            if {member.name for member in package.getmembers() if member.isfile()} != set(chain["code_sha256"]):
                raise ValueError("Comparison archive differs from the declared committed file set")
            package.extractall(code, filter="data")
        for name, expected in chain["code_sha256"].items():
            require_digest(local_reference(code, name), expected)
        if str(code) not in sys.path:
            sys.path.insert(0, str(code))
        mapping = chain["registry_mapping"]
        target = local_reference(code, mapping["snapshot_relative_directory"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(mapping["original_directory"], target_is_directory=True)
        reference = chain["admission"]["registry_reference"]
        if local_reference(code, reference).resolve() != local_reference(Path(chain["source_worktree"]), reference).resolve():
            raise ValueError("Snapshot registry mapping changed admitted logical identity")
        transfer = ["repo/" + name for name in chain["code_sha256"]]
        for original, entry in chain["profiles"].items():
            require_digest(base / entry["file"], entry["sha256"])
            copy_checked(base / entry["file"], local_reference(code, entry["reference"]))
            transfer.append("repo/" + entry["reference"])
        configuration(local_reference(code, chain["profiles"][chain["profile"]["config_path"]]["reference"]), chain["profile"])
        conditioner = None
        if not chain["profile"]["joint_belief_fit"]:
            common = chain["conditioning_profile"]
            common_config = configuration(local_reference(code, chain["profiles"][common["config_path"]]["reference"]), common)
            source_manifest = local_reference(Path(chain["source_worktree"]), common["training_manifest"])
            normalization_id = digest(local_reference(Path(chain["source_worktree"]), chain["normalizer_reference"]))
            checkpoint = validate_conditioner(source_manifest, chain, common_config, normalization_id)
            conditioner = {}
            for source in (source_manifest, checkpoint):
                relative = str(source.relative_to(Path(chain["source_worktree"])))
                conditioner[relative] = copy_checked(source, local_reference(code, relative))
                transfer.append("repo/" + relative)
            record["conditioner_sha256"] = conditioner
        command = stage_command(chain, code, destination / "inputs", reuse_manifest)
        record["command"] = command
        with (destination / "staging.log").open("xb") as log:
            result = subprocess.run(command, cwd=code, env=dict(os.environ, PYTHONPATH=str(code) + ":" + str(code / "Libs")),
                                    stdout=log, stderr=subprocess.STDOUT)
        record["exit_code"] = result.returncode
        if result.returncode:
            raise RuntimeError("Maintained native subset staging failed; inspect staging.log")
        manifest_path = destination / "inputs/native_inputs.json"
        manifest = read_json(manifest_path)
        verify_groups(manifest, chain, reused)
        staged_normalization = verify_normalizer(destination, manifest, chain)
        if conditioner is not None and staged_normalization != normalization_id:
            raise ValueError("Conditioner and actual staged normalizer differ")
        for logical, metadata in manifest["files"].items():
            if native_file(destination, logical).stat().st_size != metadata["size_bytes"]:
                raise ValueError("Staged input size differs: " + logical)
        if (sum(row["size_bytes"] for row in manifest["files"].values()) != manifest["size_bytes"]
                or len(manifest["files"]) != manifest["file_count"]):
            raise ValueError("Native input file and byte accounting differs")
        shutil.copy2(base / "execution_chain.json", destination / "execution_chain.json")
        transfer += ["execution_chain.json", "inputs/native_inputs.json"]
        transfer += [str(native_file(destination, name).relative_to(destination)) for name in manifest["files"]]
        transfer = sorted(set(transfer))
        (destination / "transfer_files.txt").write_text("\n".join(transfer) + "\n")
        record.update(status="COMPLETE", groups=chain["groups"], input_manifest_sha256=digest(manifest_path),
                      transfer_files_sha256=digest(destination / "transfer_files.txt"),
                      transfer_bytes=sum(local_reference(destination, name).stat().st_size for name in transfer),
                      input_file_count=manifest["file_count"], input_bytes=manifest["size_bytes"])
    except BaseException as error:
        record.update(status="FAILED", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - started
        write_json(destination / "staging_result.json", record)
        for name in ("staging_result.json", "staging.log", "transfer_files.txt", "execution_chain.json"):
            if (destination / name).is_file():
                shutil.copy2(destination / name, evidence / name)
        print(json.dumps(record), flush=True)
    return record


def transfer_names(raw: bytes) -> list[str]:
    names = raw.decode("utf-8").splitlines()
    if not names or len(names) != len(set(names)):
        raise ValueError("Input transfer list is empty or duplicated")
    for name in names:
        local_reference(Path("."), name)
    return names


def approved_shared_fit(base: Path, chain: dict) -> dict:
    """Read the one independent component approval before consuming a primary."""
    review = read_json(local_reference(Path(chain["source_worktree"]), chain["component_review"]))
    expected = {"status": "APPROVED_COMPONENT", "profile_id": chain["profile"]["profile_id"],
                "source_commit": chain["source_commit"],
                "execution_chain_sha256": digest(base / "execution_chain.json"),
                "profile_sha256": chain["profile_sha256"],
                "effective_configuration_sha256": chain["effective_configuration_sha256"]}
    if any(review.get(key) != value for key, value in expected.items()):
        raise ValueError("Direct shared fit has no matching independent component approval")
    return review


def require_serial_fit() -> dict:
    active = subprocess.check_output(["squeue", "-h", "-u", str(os.getuid()), "-o", "%A|%j"], text=True)
    own = os.environ["SLURM_JOB_ID"]
    if any(line.split("|", 1)[0] != own and line.split("|", 1)[1].startswith(
            ("NrnSift_native_fit", "NrnSift_source_fit")) for line in active.splitlines() if "|" in line):
        raise RuntimeError("Another native fit is queued or active; concurrent_fits must remain1")
    return {"native_fit_squeue": active}


def validate_shared_stage(chain: dict, staged: dict, data_chain: dict, source_stage: str) -> None:
    """Bind new executable code to unchanged completed CPU03 data preparation."""
    shared = chain["shared_input_stage"]
    if (source_stage != shared["stage"] or staged.get("stage") != source_stage
            or staged.get("status") != "COMPLETE" or staged.get("host") != "cpu03"
            or str(staged.get("job_id")) != shared["job_id"]
            or staged.get("groups") != chain["groups"] or staged.get("roles") != ["train", "development"]
            or staged.get("source_commit") != data_chain["source_commit"]
            or staged.get("source_generation") is not False or staged.get("scientific_fit") is not False):
        raise ValueError("Shared CPU03 data stage has no matching complete identity")
    if any(data_chain.get(key) != value for key, value in chain["admission"].items()):
        raise ValueError("Shared data-stage chain changes source/admission authority")
    terminal = staged.get("admission_terminal", {})
    expected = {"stage": "source_and_admissions_complete", "job_id": chain["admission"]["admission_job"],
                "source_generation_job": chain["admission"]["source_job"],
                "source_producer_commit": chain["admission"]["frozen_source_commit"],
                "consumer_commit": chain["admission"]["admission_consumer_commit"],
                "source_generation": "NOT_REPLAYED", "admitted_count": chain["admission"]["full_source_cases"]}
    source = staged.get("source_terminal", {})
    if (any(terminal.get(key) != value for key, value in expected.items())
            or source.get("status") != "complete" or source.get("source_status") != "complete"
            or source.get("stop_group_index") != chain["admission"]["full_source_groups"]
            or source.get("source_identity", {}).get("producer_commit") != chain["admission"]["frozen_source_commit"]):
        raise ValueError("Shared staging lacks complete matching unreplayed source/admission receipts")


def transfer_shared_stage(base: Path, chain: dict, destination: Path, source_stage: str) -> dict:
    """Transfer declared input files and new committed executable bytes only."""
    metadata = ("staging_result.json", "transfer_files.txt", "execution_chain.json", "inputs/native_inputs.json")
    for name in metadata:
        target = local_reference(destination, name); target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "-e", SSH, "cpu03:" + source_stage + "/" + name, str(target)], check=True)
    staged = read_json(destination / "staging_result.json")
    require_digest(destination / "execution_chain.json", chain["shared_input_stage"]["execution_chain_sha256"])
    data_chain = read_json(destination / "execution_chain.json")
    validate_shared_stage(chain, staged, data_chain, source_stage)
    require_digest(destination / "transfer_files.txt", staged["transfer_files_sha256"])
    original_names = transfer_names((destination / "transfer_files.txt").read_bytes())
    manifest_path = destination / "inputs/native_inputs.json"
    require_digest(manifest_path, staged["input_manifest_sha256"])
    manifest = read_json(manifest_path); verify_groups(manifest, chain)
    if (manifest["file_count"] != len(manifest["files"])
            or manifest["size_bytes"] != sum(row["size_bytes"] for row in manifest["files"].values())
            or staged.get("input_file_count") != manifest["file_count"] or staged.get("input_bytes") != manifest["size_bytes"]):
        raise ValueError("Shared input file/byte accounting differs")
    names = [str(native_file(destination, logical).relative_to(destination)) for logical in manifest["files"]]
    if any(name not in original_names for name in names):
        raise ValueError("Shared input file is absent from the completed transfer list")
    archive = base / chain["code_archive_name"]; require_digest(archive, chain["code_archive_sha256"])
    free = shutil.disk_usage(destination).free
    with tarfile.open(archive) as package:
        members = [member for member in package.getmembers() if member.isfile()]
        if {member.name for member in members} != set(chain["code_sha256"]):
            raise ValueError("Shared fit archive differs from committed executable files")
        needed = manifest["size_bytes"] + sum(member.size for member in members) + chain["checkpoint_storage"]["maximum_live_bytes"]
        if free <= needed:
            raise RuntimeError("GPU capacity cannot hold shared inputs, new code and live checkpoints")
        package.extractall(destination / "repo", filter="data")
    (destination / "input_transfer_files.txt").write_text("\n".join(names) + "\n")
    subprocess.run(["rsync", "-a", "-e", SSH, "--files-from=" + str(destination / "input_transfer_files.txt"),
                    "cpu03:" + source_stage + "/", str(destination) + "/"], check=True)
    code = destination / "repo"
    for name, expected in chain["code_sha256"].items(): require_digest(local_reference(code, name), expected)
    for entry in chain["profiles"].values():
        require_digest(base / entry["file"], entry["sha256"])
        copy_checked(base / entry["file"], local_reference(code, entry["reference"]))
    configuration(local_reference(code, chain["profiles"][chain["profile"]["config_path"]]["reference"]), chain["profile"])
    for logical, row in manifest["files"].items():
        if native_file(destination, logical).stat().st_size != row["size_bytes"]:
            raise ValueError("Transferred shared input size differs: " + logical)
    verify_normalizer(destination, manifest, chain)
    shutil.copy2(destination / "execution_chain.json", destination / "data_execution_chain.json")
    shutil.copy2(base / "execution_chain.json", destination / "execution_chain.json")
    # NativeInputLocations verifies every content hash at the actual dataset read.
    return {"input_staging": staged, "disk_free_before_transfer": free,
            "known_transfer_plus_checkpoint_bytes": needed,
            "shared_data_chain_sha256": chain["shared_input_stage"]["execution_chain_sha256"]}


def run_fit(base: Path, destination: Path, source_stage: str) -> dict:
    if socket.gethostname() != "gpu03" or not source_stage.startswith("/tmp/NrnSift_") or ".." in Path(source_stage).parts:
        raise RuntimeError("Comparison fit requires gpu03 and an explicit CPU03 node-local stage")
    chain = package_chain(base)
    root = Path(chain["source_worktree"])
    shared = chain.get("input_mode") == "completed_shared_stage"
    if shared:
        approval = approved_shared_fit(base, chain)
        scheduler = require_serial_fit()
        if source_stage != chain["shared_input_stage"]["stage"]:
            raise ValueError("Shared fit source stage differs from its explicit binding")
    durable = local_reference(root, chain["profile"]["output_root"])
    # Even a failed start consumes the primary. Recovery remains separately authorized.
    if destination.exists():
        raise FileExistsError("Comparison GPU stage already exists")
    durable.mkdir(parents=True, exist_ok=False)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    local_output = local_reference(destination / "repo", chain["profile"]["output_root"])
    record = {"status": "STARTED", "job_id": os.environ["SLURM_JOB_ID"], "host": "gpu03",
              "source_commit": chain["source_commit"], "profile_id": chain["profile"]["profile_id"],
              "input_stage": source_stage, "node_local_stage": str(destination), "attempt": "primary",
              "automatic_retries": False, "scientific_twin": "NOT_RUN", "sufficient_training": False}
    returncode = None
    try:
        if shared:
            record.update(component_review=approval, scheduler_prelaunch=scheduler,
                          **transfer_shared_stage(base, chain, destination, source_stage))
            code = destination / "repo"
            manifest_path = destination / "inputs/native_inputs.json"
        else:
            subprocess.run(["rsync", "-a", "-e", SSH, "cpu03:" + source_stage + "/staging_result.json",
                            "cpu03:" + source_stage + "/transfer_files.txt", str(destination) + "/"], check=True)
            staged = read_json(destination / "staging_result.json")
            if (staged.get("status") != "COMPLETE" or staged.get("groups") != chain["groups"]
                    or staged.get("profile_id") != chain["profile"]["profile_id"]
                    or staged.get("source_commit") != chain["source_commit"] or staged.get("stage") != source_stage):
                raise ValueError("Comparison input staging receipt has no matching completed identity")
            require_digest(destination / "transfer_files.txt", staged["transfer_files_sha256"])
            names = transfer_names((destination / "transfer_files.txt").read_bytes())
            free = shutil.disk_usage(destination).free
            needed = staged["transfer_bytes"] + chain["checkpoint_storage"]["maximum_live_bytes"]
            record.update(input_staging=staged, disk_free_before_transfer=free, known_transfer_plus_checkpoint_bytes=needed)
            if free <= needed:
                raise RuntimeError("GPU local capacity cannot hold transfer and the unchanged live checkpoint ceiling")
            subprocess.run(["rsync", "-a", "-e", SSH, "--files-from=" + str(destination / "transfer_files.txt"),
                            "cpu03:" + source_stage + "/", str(destination) + "/"], check=True)
            if read_json(destination / "execution_chain.json") != chain:
                raise ValueError("CPU and GPU comparison execution bindings differ")
            if sum(local_reference(destination, name).stat().st_size for name in names) != staged["transfer_bytes"]:
                raise ValueError("Actual transferred bytes differ from completed staging accounting")
            code = destination / "repo"
            for name, expected in chain["code_sha256"].items():
                require_digest(local_reference(code, name), expected)
            for entry in chain["profiles"].values():
                require_digest(local_reference(code, entry["reference"]), entry["sha256"])
            manifest_path = destination / "inputs/native_inputs.json"
            require_digest(manifest_path, staged["input_manifest_sha256"])
            manifest = read_json(manifest_path)
            verify_groups(manifest, chain)
            staged_normalization = verify_normalizer(destination, manifest, chain)
            for logical, metadata in manifest["files"].items():
                if native_file(destination, logical).stat().st_size != metadata["size_bytes"]:
                    raise ValueError("Transferred native input size differs: " + logical)
            for name, expected in staged.get("conditioner_sha256", {}).items():
                if "repo/" + name not in names:
                    raise ValueError("Conditioner manifest or checkpoint is missing from transfer list")
                require_digest(local_reference(code, name), expected)
            if not chain["profile"]["joint_belief_fit"]:
                expected_manifest = chain["conditioning_profile"]["training_manifest"]
                if expected_manifest not in staged.get("conditioner_sha256", {}):
                    raise ValueError("Actual completed same-seed conditioner manifest was not transferred")
                if str(code) not in sys.path:
                    sys.path.insert(0, str(code))
                common = chain["conditioning_profile"]
                common_config = configuration(local_reference(code, chain["profiles"][common["config_path"]]["reference"]), common)
                checkpoint = validate_conditioner(local_reference(code, expected_manifest), chain, common_config, staged_normalization)
                if str(checkpoint.relative_to(code)) not in staged["conditioner_sha256"]:
                    raise ValueError("Actual DEV-selected conditioner file was not transferred")
        if shutil.disk_usage(destination).free <= chain["checkpoint_storage"]["maximum_live_bytes"]:
            raise RuntimeError("GPU checkpoint capacity was lost during transfer")
        command = fit_command(chain, code, manifest_path)
        record.update(command=command, training_manifest=str(local_reference(code, chain["profile"]["training_manifest"])))
        write_json(durable / "execution.json", record)
        with (destination / "training.log").open("xb") as log:
            process = subprocess.run(["/usr/bin/time", "-v", "-o", str(destination / "process_resources.log"), *command],
                                     cwd=code, env=dict(os.environ, PYTHONPATH=str(code) + ":" + str(code / "Libs")),
                                     stdout=log, stderr=subprocess.STDOUT)
        returncode = process.returncode
        record.update(status="PROCESS_COMPLETED" if returncode == 0 else "PROCESS_FAILED", exit_code=returncode)
    except BaseException as error:
        record.update(status="FAILED", error=type(error).__name__ + ": " + str(error), exit_code=returncode)
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - started
        write_json(destination / "execution.json", record)
        if local_output.is_dir():
            shutil.copytree(local_output, durable, dirs_exist_ok=True)
        for name in ("execution.json", "execution_chain.json", "staging_result.json", "transfer_files.txt",
                     "training.log", "process_resources.log", "data_execution_chain.json", "input_transfer_files.txt"):
            if (destination / name).is_file():
                shutil.copy2(destination / name, durable / name)
        print(json.dumps(record), flush=True)
    if returncode:
        raise SystemExit(returncode)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)
    prepare_parser = operations.add_parser("prepare")
    prepare_parser.add_argument("--protocol", type=Path, required=True)
    prepare_parser.add_argument("--admission-package", type=Path, required=True)
    prepare_parser.add_argument("--code-binding", type=Path, required=True)
    prepare_parser.add_argument("--profile-id", required=True)
    prepare_parser.add_argument("--destination", type=Path, required=True)
    prepare_parser.add_argument("--shared-input-stage")
    prepare_parser.add_argument("--shared-stage-job")
    stage_parser = operations.add_parser("stage")
    stage_parser.add_argument("--package", type=Path, required=True)
    stage_parser.add_argument("--destination", type=Path, required=True)
    stage_parser.add_argument("--reuse-manifest", type=Path, required=True)
    stage_parser.add_argument("--evidence", type=Path, required=True)
    fit_parser = operations.add_parser("fit")
    fit_parser.add_argument("--package", type=Path, required=True)
    fit_parser.add_argument("--destination", type=Path, required=True)
    fit_parser.add_argument("--source-stage", required=True)
    args = parser.parse_args()
    if args.operation == "prepare":
        print(prepare(args.protocol, args.admission_package, args.code_binding, args.profile_id, args.destination,
                      shared_input_stage=args.shared_input_stage, shared_stage_job=args.shared_stage_job))
    elif args.operation == "stage":
        stage_inputs(args.package, args.destination, args.reuse_manifest, args.evidence)
    else:
        run_fit(args.package, args.destination, args.source_stage)


if __name__ == "__main__":
    main()
