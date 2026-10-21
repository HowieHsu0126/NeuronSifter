"""Real tiny upstream fits and CLI scale routing; no scientific source production."""
from __future__ import annotations

import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from Libs.utils.model.native_training_selection import training_checkpoint_path
from torch.utils.data import DataLoader
import yaml

from Libs.exp.training import pretrain_microenvironment_world_model as training
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.exp.experiment_manifests import resolve_training_seed
from Libs.utils.exp.native_comparator_training import bind_frozen_training_belief
from Libs.utils.exp.native_learned_distillation import load_selected_native_teacher, validate_saved_learned_teacher
from Libs.utils.model.native_direct_mean import file_sha256


def select_saved_last(config_path: Path, manifest_path: Path, output_path: Path) -> tuple[dict, int]:
    """Finish a real development evaluation without replaying capped training."""
    if output_path.exists():
        raise FileExistsError("Development-selection receipt already exists")
    config = load_config(str(config_path))
    original = json.loads(manifest_path.read_text())
    last = original["selection"]["last"]
    if original["selection"]["best"] is not None or not last or last["steps"] != 1:
        raise ValueError("This recovery requires the retained single-update unselected checkpoint")
    checkpoint_path = training_checkpoint_path(manifest_path, original["selection"], last["checkpoint_path"])
    checkpoint_hash = file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    binding = resolve_training_seed(config, require_selected=True)
    training.validate_training_seed_checkpoint(checkpoint, binding)
    device = torch.device("cpu")
    model = training.MicroenvironmentPretrainingModel(config, False)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters >= 1_000_000:
        raise RuntimeError("Development fixture unexpectedly selected a full-capacity model")
    conditioning = bind_frozen_training_belief(model, config, device)
    profile = training.native_training_profile(config, model.field_normalization.identity)
    if (checkpoint.get("native_frozen_conditioning") != conditioning
            or checkpoint.get("native_scientific_profile") != profile
            or original.get("native_scientific_profile") != profile):
        raise ValueError("Saved fixture checkpoint differs from its frozen conditioning/profile")
    validate_saved_learned_teacher(checkpoint, original, config, conditioning)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False).eval()
    policy = config["native_scientific_training"]
    sampler = model.quantized_mddm
    sampler.sampler_seed = int(binding["model_seed"])+policy["validation_stream_offset"]
    sampler.inference_sample_count = policy["validation"]["sample_count"]
    sampler.inference_denoising_step_count = policy["validation"]["denoising_step_count"]
    dataset = training.NativeJointGroupDataset(config, "development")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, **training.data_loader_kwargs(config))
    losses = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            values, _ = training.scientific_group_batch(model, batch["joint_groups"], config, device,
                config["screening_pipeline"]["screening_benchmark"]["training"]["memory_policy"],
                backward=False)
            losses.append(values["total_loss"])
    if len(losses) != len(dataset) or len(losses) != 4:
        raise AssertionError("Incomplete actual four-domain development evaluation")
    selection = training.NativeTrainingSelection(policy["selection"])
    selection.update(role="development", epoch=last["epoch"], steps=last["steps"],
        training_loss=last["training_loss"], development_loss=sum(losses)/len(losses),
        checkpoint_path=str(checkpoint_path),
        elapsed_seconds=original["wall_time_seconds"]+time.perf_counter()-started)
    result = copy.deepcopy(original)
    result["selection"] = selection.payload()
    result["development_selection_completion"] = {
        "source_manifest": str(manifest_path), "checkpoint_sha256": checkpoint_hash,
        "development_group_uids": dataset.group_uids, "development_losses": losses,
        "additional_optimizer_updates": 0, "source_provider_calls": 0,
        "elapsed_seconds": time.perf_counter()-started,
        "scope": "software_fixture_saved_checkpoint_assessment_not_training_sufficiency"}
    if file_sha256(checkpoint_path) != checkpoint_hash:
        raise AssertionError("Development evaluation changed the retained checkpoint")
    output_path.write_text(json.dumps(result, indent=2))
    print("SCALE_DEVELOPMENT_SELECTED", json.dumps({
        "manifest": str(output_path), "losses": losses,
        "selected_loss": result["selection"]["best"]["development_loss"],
        "training_terminal": result["selection"]["terminal"],
        "additional_optimizer_updates": 0}), flush=True)
    return result, parameters


def run(origin: Path, output: Path, *, reused_student: Path | None = None) -> dict:
    """Reuse the selected Large; fit only the two missing small route fixtures."""
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Scale integration attempt must use a fresh output directory")
    config_path = origin/"config.yaml"
    manifest_path = origin/"native_fit"/"manifest.json"
    reference_path = origin/"public_cns_reference.json"
    config = load_config(str(config_path))
    current = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    for key in ("native_scale_routing", "manuscript_native_protocol"):
        config[key] = copy.deepcopy(current[key])
    config["schemas"]["native_scale_evaluation"] = copy.deepcopy(current["schemas"]["native_scale_evaluation"])
    config.pop("config_includes", None)
    config["selected_formal_seed"] = 40
    binding = resolve_training_seed(config, require_selected=True)
    device = torch.device("cpu")
    original_manifest = json.loads(manifest_path.read_text())
    original_best = training_checkpoint_path(manifest_path, original_manifest["selection"], original_manifest["selection"]["best"]["checkpoint_path"])
    original_hashes = {str(path): file_sha256(path) for path in
                       (config_path, manifest_path, reference_path, original_best)}
    checkpoints = {}
    reports = []
    # Large uses the identical selected checkpoint and native scientific profile.
    large = copy.deepcopy(config)
    large["native_profile_identity"] = {"method": "large", "percentage": 100, "seed": 40,
        "checkpoint_scale": "neuronsifter_large", "inference_profile": "large_main_screening",
        "protocol_uid": current["manuscript_native_protocol"]["protocol_uid"]}
    large_path = output/"large.yaml"
    large_path.write_text(yaml.safe_dump(large, sort_keys=False))
    checkpoints["large"] = {"config_path": str(large_path), "training_manifest": str(manifest_path)}

    for method, width, heads in (("student", 8, 2), ("xl", 16, 4)):
        if method == "student" and reused_student is not None:
            selected_path = reused_student/"student.yaml"
            original_training = reused_student/"student_training"/"microenvironment_pretraining_manifest.json"
            selected_manifest = output/"student_development_selection_manifest.json"
            result, parameters = select_saved_last(selected_path, original_training, selected_manifest)
            checkpoints[method] = {"config_path":str(selected_path),"training_manifest":str(selected_manifest)}
            reports.append({"method":method,"parameters":parameters,"optimizer_updates":1,
                "additional_optimizer_updates":0,"selected_updates":1,
                "visited_training_family":"electrophysiology_conductance_world",
                "declared_training_groups":result["train_group_count"],"training_sufficient":False,
                "learned_teacher_forward_calls":result["learned_teacher_forward_calls"],
                "original_training_manifest":str(original_training)})
            print("SCALE_REUSED_SELECTED",json.dumps(reports[-1]),flush=True)
            gc.collect()
            continue
        selected = copy.deepcopy(config)
        policy = selected["native_scientific_training"]
        policy["conditioning"] = {"mode": "frozen_selected_mainline",
            "config_path": str(config_path), "manifest_path": str(manifest_path)}
        policy["selection"].update(max_epochs=1, minimum_epochs=1, minimum_steps=1,
            patience=1, min_delta=0., max_steps=1, max_seconds=1800.)
        policy["validation"]["interval_optimizer_steps"] = 1
        policy["learning_rate_schedule"]["warmup_steps"] = 1
        benchmark = selected["screening_pipeline"]["screening_benchmark"]
        benchmark["training"]["batch_size"] = 1
        benchmark["training"]["memory_policy"].update(
            effective_batch_size=1, micro_batch_size=1, gradient_accumulation_steps=1)
        benchmark["model_architecture"]["rollout_operator"]["transition_backbone"].update(
            operator_hidden_dim=width, operator_depth=1, attention_head_count=heads,
            mechanism_state_token_count=6)
        selected["native_learned_distillation"].update(enabled=method == "student",
            teacher_config=str(config_path) if method == "student" else None,
            teacher_manifest=str(manifest_path) if method == "student" else None)
        declared = current["manuscript_native_protocol"]["native_method_profiles"][method]
        selected["native_profile_identity"] = {"method": method, "percentage": 100, "seed": 40,
            "checkpoint_scale": declared["checkpoint_scale"],
            "inference_profile": declared["inference_profile"],
            "protocol_uid": current["manuscript_native_protocol"]["protocol_uid"]}
        selected_path = output/(method+".yaml")
        selected_path.write_text(yaml.safe_dump(selected, sort_keys=False))
        torch.manual_seed(40)
        model = training.MicroenvironmentPretrainingModel(selected, False)
        parameters = sum(p.numel() for p in model.parameters())
        if not parameters < 1_000_000:
            raise RuntimeError("Routing software fixture unexpectedly selected full-capacity dimensions")
        conditioning = bind_frozen_training_belief(model, selected, device)
        teacher = load_selected_native_teacher(model, selected, device)
        dataset = training.NativeJointGroupDataset(selected, "train")
        first_family = next(key for key in dataset.family_group_counts
                            if "electrophysiology" in key)
        first_group = dataset[0]["native_joint_group"]
        if first_group["joint_source"]["support"]["source_family_uid"] != first_family:
            raise ValueError("Single-update route fixture must visit the original first electrical group")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, **training.data_loader_kwargs(selected))
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=1e-4, weight_decay=1e-4)
        before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        train_dir = output/(method+"_training")
        train_dir.mkdir()
        start = time.perf_counter()
        result = training.fit_scientific_native_groups(model, optimizer, loader, selected, device,
            benchmark["training"]["memory_policy"], train_dir,
            "tiny_scale_route_software_fixture", binding, learned_teacher=teacher)
        if result["executed_steps"] != 1:
            raise AssertionError("Unexpected tiny route optimizer update count")
        if not any(not torch.equal(before[name], p) for name, p in model.named_parameters()
                   if name in before):
            raise AssertionError("Scale fixture had no actual trainable parameter update")
        original_training = train_dir/"microenvironment_pretraining_manifest.json"
        declared_count = len(dataset)
        del model, optimizer, teacher, dataset, loader, before, first_group
        gc.collect()
        stage_manifest = original_training
        if result["selection"]["best"] is None:
            stage_manifest = output/(method+"_development_selection_manifest.json")
            result, checked_parameters = select_saved_last(selected_path, original_training, stage_manifest)
            assert checked_parameters == parameters
        checkpoints[method] = {"config_path": str(selected_path), "training_manifest": str(stage_manifest)}
        reports.append({"method": method, "parameters": parameters, "optimizer_updates": 1,
            "additional_optimizer_updates":1,"selected_updates": 1, "visited_training_family": first_family,
            "declared_training_groups": declared_count, "training_sufficient": False,
            "learned_teacher_forward_calls": result["learned_teacher_forward_calls"],
            "elapsed_seconds": time.perf_counter()-start})
        print("SCALE_UPSTREAM_FIT", json.dumps(reports[-1]), flush=True)
        gc.collect()

    routed = copy.deepcopy(large)
    routed["native_scale_routing"]["checkpoints"] = checkpoints
    route_path = output/"routes.yaml"
    route_path.write_text(yaml.safe_dump(routed, sort_keys=False))
    root = Path(__file__).resolve().parents[2]
    commands = []
    for method in routed["native_scale_routing"]["methods"]:
        destination = output/(method+"_evaluation")
        command = [sys.executable, str(root/"Libs/exp/baselines/run_external_baseline_bridge.py"),
            "--protocol-scope", "manuscript-native", "--config", str(route_path),
            "--baseline-name", method, "--command", "evaluate_baseline",
            "--conditioning-config", str(config_path), "--conditioning-manifest", str(manifest_path),
            "--decision-reference", str(reference_path), "--device", "cpu",
            "--output-dir", str(destination)]
        log = output/(method+"_cli.log")
        with log.open("w") as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        commands.append({"method": method, "argv": command, "exit_code": process.returncode})
        print("SCALE_CLI", json.dumps(commands[-1]), flush=True)
        if process.returncode != 0:
            raise RuntimeError(method+" actual CLI failed; preserved log: "+str(log))
        report = json.loads((destination/"native_scale_evaluation.json").read_text())
        if report["status"] != "complete" or report["defined_group_count"] != 4:
            raise AssertionError("Incomplete actual four-domain scale report")
        if method == "fixed_route":
            if any([stage["stage"] for stage in row["route"]] != ["student", "large", "xl"]
                   for row in report["groups"]):
                raise AssertionError("Fixed route did not actually execute all three stages")
        for row in report["groups"]:
            if len(row["posterior_weights"]) != 32 or row["declared_action_count"] != 3:
                raise AssertionError("Routing fixture lost original complete posterior/actions")
        commands[-1].update(sampling_calls=report["sampling_calls"],
            groups=report["defined_group_count"], escalation_rate=report["escalation_rate"])
    if any(file_sha256(Path(path)) != value for path, value in original_hashes.items()):
        raise AssertionError("The original Large/reference configuration or checkpoint changed")
    receipt = {"scope": "tiny_actual_three_scale_loader_and_cli_software_fixture",
        "reused_large": {"manifest": str(manifest_path), "checkpoint": str(original_best),
                         "checkpoint_sha256": original_hashes[str(original_best)]},
        "new_upstream_fits": reports, "commands": commands,
        "source_provider_calls": 0, "original_large_optimizer_updates": 0,
        "scientific_twin": "NOT_RUN", "formal_qualification": False,
        "training_sufficient": False}
    (output/"result.json").write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    if len(sys.argv) not in (3,4):
        raise SystemExit("usage: native_scale_routing_integration.py EXISTING_FIXTURE NEW_OUTPUT [RETAINED_STUDENT_ARTIFACTS]")
    run(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(),
        reused_student=Path(sys.argv[3]).resolve() if len(sys.argv)==4 else None)
