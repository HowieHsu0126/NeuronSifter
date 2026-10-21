"""Actual four-domain posterior CLI, using explicitly artificial future targets.

The full long-CNS backward fixture requires a CPU allocation with at least24GiB.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from Libs.exp.training import pretrain_microenvironment_world_model as training
from Libs.scripts.builders.build_controlled_native_source import merge_scientific_registry, scientific_index
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data.joint_physical_parameters import bind_candidate_parameters, parameter_schema_from_config
from Libs.utils.data.native_field_normalization import bind_artifact, fit_multigrid
from Libs.utils.data.native_joint_source_storage import write_joint_source_reference
from Libs.utils.data.native_observable_profile import family_grid, observable_profile, quantity_support
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.evaluation.native_comparator_evaluation import declared_dataset
from Libs.utils.exp.experiment_manifests import resolve_training_seed
from Libs.utils.model import controlled_cns_source as controlled
from Libs.utils.model import native_scientific_source as source
from Libs.utils.model.microenvironment_teacher import write_native_microenvironment_store


FIXTURE_SCOPE = "artificial_software_fixture_no_source_trajectory"


def fixture_config():
    cfg = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    cfg.pop("config_includes", None)
    partitions = cfg["native_observable_bridge"]["group_partitions"]
    cfg["native_observable_bridge"]["group_partitions"] = [
        dict(next(row for row in partitions if row["role"] == role), groups_per_family=1)
        for role in ("train", "development", "validation")]
    cfg["selected_formal_seed"] = 40
    fit = cfg["screening_pipeline"]["screening_benchmark"]["training"]
    fit.update(random_seed=40, seed_policy="manuscript_native_five", batch_size=1)
    fit["memory_policy"].update(effective_batch_size=1, micro_batch_size=1, gradient_accumulation_steps=1)
    settings = cfg["native_scientific_training"]
    settings.update(source_scope="all", training_subset_percentage=100, conditioning={"mode": "joint"})
    settings["selection"].update(max_epochs=1, minimum_epochs=1, minimum_steps=1, max_steps=2, max_seconds=1800)
    settings["validation"].update(sample_count=2, denoising_step_count=1, interval_optimizer_steps=1)
    settings["predictive_budget"].update(conditional_draws_per_particle=2,
        weighted_forecast_paths_per_action=64, categorical_reverse_steps=1)
    settings["learning_rate_schedule"]["warmup_steps"] = 1
    backbone = cfg["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    backbone.update(operator_hidden_dim=12, operator_depth=1, attention_head_count=3, mechanism_state_token_count=6)
    cfg["native_diffusion_data"].update(source_family_uid=None, posterior_context_width=12)
    return cfg


def fixture_program(cfg, group, generated, atom, action):
    """Compile supplied actions/PK; never advance a physical or CNS teacher."""
    if source.is_long_cns(cfg, group["family_uid"]):
        definition = source.long_cns_definition(cfg)
        definition["pk"]["total_dose_mg_per_kg"] *= action.get("total_dose_multiplier", 1.)
        law = source.long_cns_law(cfg, group, generated["support"], atom, action)
        pk = controlled.pk_values(definition["grid"]["time_hours"], law.pk, action, definition)
        return controlled.compiled_particle_program(definition, group, generated["support"],
            atom, action, pk, law.pk, FIXTURE_SCOPE)
    parameters = generated["parameters"][atom]
    estimate = source.estimated_exposure(cfg, parameters, action) if group["family_uid"] == source.MOLECULAR else None
    return source.compiled_program(cfg, group, generated["support"], atom, parameters, action, FIXTURE_SCOPE, estimate)


def write_fixture_sources(cfg, root):
    """Use real source-format stores and lossless whole-group gzip references."""
    profile = observable_profile(cfg)
    declarations = source.group_definitions(cfg)
    assert len(declarations) == 12
    assert all(group["source_variant"] == "primary" for group in declarations)
    receipt = root / "software_fixture.json"
    receipt.write_text(json.dumps({"scope": FIXTURE_SCOPE, "scientific_twin": "NOT_RUN",
        "formal_qualification": False, "physical_source_trajectory_calls": 0,
        "target_generation": "initial_complete_tuple_zero_plus_masked_linear_ramp",
        "uncertainty": "positive_artificial_software_channel_not_numerical_source_evidence"}))
    registries = {family: {} for family in profile["families"]}
    for gi, group in enumerate(declarations):
        local = root / "sources" / str(gi)
        local.mkdir(parents=True)
        selected = source.scoped_group_config(cfg, group)
        generated = source.draw_group(selected, group, gi)
        count = source.action_count(selected, group)
        records = [[fixture_program(selected, group, generated, atom,
                    source.action_definition(selected, group["family_uid"], ai, group))
                    for ai in range(count)] for atom in range(32)]
        raw = source.joint_source(selected, group, generated, records)
        producer = local / "candidate_parameters.json"
        producer.write_text(json.dumps({"parameters": generated["parameters"], "scope": FIXTURE_SCOPE}))
        raw = bind_candidate_parameters(raw, group, {"parameters": generated["parameters"]},
            parameter_schema_from_config(cfg), record_ref=str(producer), producer_commit=FIXTURE_SCOPE)
        reference = write_joint_source_reference(local / "conditioning.json.gz", raw,
            max_bytes=cfg["native_diffusion_data"]["group_storage"]["compressed_group_max_bytes"][group["family_uid"]])
        grid = family_grid(profile, group["family_uid"])
        mask = quantity_support(grid, group["source_uid"])
        initial = np.asarray(raw["support"]["initial_field"])[0, 0]
        ramp = np.linspace(0., 1., len(grid["time_hours"]))[:, None, None, None]
        scale = np.maximum(np.abs(initial), .1)
        store = local / "raw.zarr"
        targets, teachers = [], []
        for ai in range(count):
            target = np.where(mask, initial[None] + ramp * scale[None] * (.04 + .002*gi + .003*ai), 0.).astype(np.float32)
            program = records[0][ai]
            teachers.append({**{key: program[key] for key in
                ("regimen_uid", "action_uid", "target_program_uid", "program_uid", "posterior_uid", "source_uid")},
                "trajectory_uid": group["sample_uid"] + ":artificial_target:" + str(ai),
                "sample_uid": group["sample_uid"], "family_uid": group["family_uid"],
                "content_hash": hashlib.sha256(target.tobytes()).hexdigest(),
                "trajectory_store_path": str(store), "store_format": "zarr_v2", "trajectory_index": ai,
                "teacher_family": "full_microenvironment_teacher", "teacher_confidence": 1.,
                "qc_catalog_ref": str(receipt), "source_snapshot_ref": group["source_snapshot_ref"],
                "uncertainty_origin": "independent_supplied", "exact_zero_uncertainty": {},
                "validity_mask": {"kind": "all_valid", "evidence_ref": str(receipt)},
                "split_name": group["role"], "source_holdout_role": group["role"],
                "qc_grade": "A", "qc_use": "main_pretraining", "sample_weight": 1.,
                "calibration_reservation": group["calibration_reservation"], "software_fixture_only": True})
            targets.append(target)
        source_row = {"source_uid": group["source_uid"], "public": True, "protected": False,
            "license": "artificial_fixture_only_no_scientific_source_license_claim",
            "release": FIXTURE_SCOPE, "snapshot_ref": group["source_snapshot_ref"],
            "split_evidence_ref": str(receipt), "unit_evidence_ref": str(receipt),
            "source_kind": "controlled_simulated", "exact_zero_quantities": {},
            "predictive_evaluation_units": {"episode_field": "sample_uid", "group_field": "family_uid",
                "group_kind": "action_family", "independent_episode_units": True, "evidence_ref": str(receipt)}}
        registry = {"schema_version": native.SCHEMA_VERSION, "grid": grid, "sources": [source_row],
            "posterior_family": "finite_joint_particles", "joint_sources": [], "joint_source_references": [reference],
            "teachers": teachers, "programs": [], "observations": []}
        arrays = {"trajectory": np.stack(targets), "uncertainty": np.full((count, 26), .01, np.float32),
            "teacher_confidence": np.ones(count, np.float32), "quantity_support": np.stack([mask]*count)}
        write_native_microenvironment_store(store, cfg, registry, arrays, metadata={"software_fixture_only": True})
        merge_scientific_registry(registries[group["family_uid"]], registry)
        print("fixture_store", gi, group["role"], group["family_uid"], list(arrays["trajectory"].shape), flush=True)
        del generated, raw, records, registry, arrays, targets
    return registries


def bind_fixture_normalization(cfg, root, registries):
    def batches():
        for registry in registries.values():
            teachers = [row for row in registry["teachers"] if row["split_name"] == "train"]
            records = native.join_registry(registry, cfg, teachers)
            arrays, _ = native.load_records(registry, records, cfg)
            yield registry["grid"], records, arrays
    normalization = root / "training_normalization.json"
    normalization.write_text(json.dumps(fit_multigrid(batches(), cfg, streaming=True)))
    references = {}
    for family, registry in registries.items():
        combined = {}
        for role in ("train", "development", "validation"):
            teachers = [row for row in registry["teachers"] if row["split_name"] == role]
            records = native.join_registry(registry, cfg, teachers)
            arrays, _ = native.load_records(registry, records, cfg)
            store = Path(teachers[0]["trajectory_store_path"]).parent / "normalized.zarr"
            derived = native.reindex_registry(registry, records, store, "zarr_v2")
            derived["grid"] = bind_artifact(registry["grid"], normalization, cfg)
            write_native_microenvironment_store(store, cfg, derived, arrays, metadata={"software_fixture_only": True})
            merge_scientific_registry(combined, derived)
            del records, arrays, derived
        path = root / (family + ".json")
        path.write_text(json.dumps(combined))
        references[family] = str(path)
    index = root / "model_registry.json"
    index.write_text(json.dumps(scientific_index(cfg, references)))
    cfg["native_diffusion_data"]["registry_path"] = str(index)


def test_actual_native_posterior_cli_four_domains(tmp_path):
    started = time.monotonic()
    resume = os.environ.get("NRNSIFT_POSTERIOR_RESUME")
    if resume:
        tmp_path = Path(resume)
        config_path = tmp_path / "config.yaml"
        cfg = load_config(str(config_path))
    else:
        cfg = fixture_config()
        bind_fixture_normalization(cfg, tmp_path, write_fixture_sources(cfg, tmp_path))
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    train, train_ids = declared_dataset(cfg, "train")
    dev, dev_ids = declared_dataset(cfg, "development")
    assert len(train) == len(dev) == 4
    reference = tmp_path / "public_cns_reference.json"
    script = str(Path("Libs/exp/baselines/run_external_baseline_bridge.py").resolve())
    cli = [sys.executable, script, "--protocol-scope", "manuscript-native", "--config", str(config_path),
        "--baseline-name", "posterior_joint", "--decision-reference", str(reference),
        "--output-dir", str(tmp_path / "posterior_comparison"), "--device", "cpu"]
    commands = []
    def invoke(command, extra=()):
        result = subprocess.run([*cli, "--command", command, *extra], capture_output=True, text=True)
        log = tmp_path / (command + "_" + str(len(commands)) + ".log")
        log.write_text(result.stdout + result.stderr)
        commands.append({"command": command, "argv": [*cli, "--command", command, *extra],
                         "exit_code": result.returncode, "log": str(log)})
        assert result.returncode == 0, result.stdout + result.stderr
    if not reference.exists():
        invoke("fit_decision_reference")
    invoke("validate_decision_reference")
    # A validly hashed old reference must fail against current admitted metadata
    # before a missing checkpoint can cause model construction or sampling.
    index = json.loads((tmp_path / "model_registry.json").read_text())
    registry_path = Path(next(iter(index["family_registries"].values())))
    original = registry_path.read_text()
    changed = json.loads(original)
    next(row for row in changed["teachers"] if row["split_name"] == "train")["content_hash"] = "alien_content"
    registry_path.write_text(json.dumps(changed))
    rejected = subprocess.run([*cli, "--command", "evaluate_baseline",
        "--conditioning-config", str(config_path), "--conditioning-manifest", str(tmp_path / "missing.json"),
        "--training-manifest", str(tmp_path / "missing.json")], capture_output=True, text=True)
    (tmp_path / "rejected_training_identity.log").write_text(rejected.stdout + rejected.stderr)
    registry_path.write_text(original)
    assert rejected.returncode != 0 and "admitted training record identities/content" in rejected.stderr
    manifest_path = tmp_path / "native_fit" / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["executed_steps"] == 2 and manifest["software_fixture_only"]
        assert manifest["selection"]["sufficient_training"] is False
        parameter_count = manifest["native_parameter_count"]
        scores = manifest["selection"]["development_scores"]
        checkpoints = [Path(manifest["selection"]["best"]["checkpoint_path"])]
        best = 0
        del train, dev
    else:
        torch.manual_seed(40)
        model = training.MicroenvironmentPretrainingModel(cfg, False)
        model.quantized_mddm.training_sample_count = 2
        model.quantized_mddm.training_denoising_step_count = 1
        model.quantized_mddm.inference_sample_count = 2
        model.quantized_mddm.inference_denoising_step_count = 1
        model.native_scientific_profile = source.native_training_profile(cfg, model.field_normalization.identity)
        assert model.world_state_posterior.joint_candidate_parameter_projection.in_features == 442
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        (tmp_path / "native_fit").mkdir()
        optimizer = training.create_adamw_optimizer(model, learning_rate=1e-4, weight_decay=1e-4)
        seed = resolve_training_seed(cfg, require_selected=True)
        scores, checkpoints, used = [], [], []
        for step, index in enumerate((0, len(train)-1), 1):
            model.train()
            loss, _, status = training.scientific_joint_group_loss(model, train[index]["native_joint_group"], cfg, torch.device("cpu"))
            assert status["particle_count"] == 32 and status["paired_target_count"] == 3
            loss.backward()
            training.execute_optimizer_step(model, optimizer, cfg)
            used.append(train.group_uids[index])
            model.eval()
            values = []
            with torch.no_grad():
                for item in range(len(dev)):
                    score, _, _ = training.scientific_joint_group_loss(model, dev[item]["native_joint_group"], cfg, torch.device("cpu"))
                    values.append(float(score))
            scores.append(sum(values)/len(values))
            checkpoints.append(training.save_training_checkpoint(model, optimizer, tmp_path / "native_fit",
                "base_" + str(step), step, step, "single_process", training_seed_binding=seed, immutable=True))
            print("tiny_update", step, "development_scores", values, "parameter_count", parameter_count, flush=True)
        best = int(np.argmin(scores))
        manifest = {"training_seed_binding": seed, "executed_steps": 2,
            "native_scientific_profile": model.native_scientific_profile, "software_fixture_only": True,
            "native_parameter_count": parameter_count,
            "declared_training_groups": sorted(train_ids), "executed_training_group_uids": used,
            "development_group_uids": sorted(dev_ids),
            "selection": {"selection_role": "development",
                "best": {"checkpoint_path": str(checkpoints[best]), "steps": best+1},
                "last": {"checkpoint_path": str(checkpoints[-1]), "steps": 2}, "development_scores": scores,
                "terminal": "software_fixture_two_update_budget", "sufficient_training": False}}
        manifest_path = tmp_path / "native_fit" / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        del model, optimizer, train, dev
    invoke("evaluate_baseline", ["--conditioning-config", str(config_path),
        "--conditioning-manifest", str(manifest_path), "--training-manifest", str(manifest_path)])
    reports = {}
    for method in cfg["native_intervention_evaluation"]["posterior_methods"]:
        path = tmp_path / "posterior_comparison" / method / "native_intervention_evaluation.json"
        report = json.loads(path.read_text())
        assert report["status"] == "complete" and report["defined_group_count"] == 4
        assert report["shared_sampling_calls"] == 4 and report["fully_supported_group_count"] == 4
        assert {row["family_uid"] for row in report["groups"]} == set(observable_profile(cfg)["families"])
        assert all(row["declared_action_count"] == row["supported_action_count"] == 3 for row in report["groups"])
        assert all(row["particle_count"] == 32 and row["conditional_draws_per_particle"] == 2 for row in report["groups"])
        reports[method] = str(path)
    result = {"scope": FIXTURE_SCOPE, "status": "complete", "actual_cli": commands,
        "actual_sampler_calls": 4, "posterior_reports": reports, "domains": 4, "actions_per_group": 3,
        "typed_parameters": 221, "posterior_atoms": 32, "parameter_count": parameter_count,
        "optimizer_updates": 2, "training_sufficient": False, "development_scores": scores,
        "selected_checkpoint": str(checkpoints[best]), "normalization": str(tmp_path / "training_normalization.json"),
        "reference": str(reference), "config": str(config_path), "manifest": str(manifest_path),
        "admitted_training_content_mismatch_rejected_before_checkpoint_loading": True,
        "elapsed_seconds": time.monotonic()-started, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "child_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "scientific_twin": "NOT_RUN", "formal_qualification": False}
    (tmp_path / "result.json").write_text(json.dumps(result, indent=2))
    if os.environ.get("NRNSIFT_SCOPE_EVIDENCE"):
        Path(os.environ["NRNSIFT_SCOPE_EVIDENCE"]).write_text(json.dumps(result, indent=2))
