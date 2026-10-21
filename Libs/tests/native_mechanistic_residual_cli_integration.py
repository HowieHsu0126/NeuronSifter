"""Real tiny offline-hybrid fit/query CLIs on the retained artificial corpus."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from Libs.models.modules.native_frozen_rhs_solver import NativeFrozenRHSSolver
from Libs.utils.data.native_diffusion_data import load_family_registries
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.evaluation.native_comparator_evaluation import declared_dataset, load_native_comparator
from Libs.utils.exp.native_comparator_training import prepare_comparator_group
from Libs.utils.model.native_comparator_inputs import load_native_prediction_inputs


def test_actual_hybrid_fit_reload_and_query_cli(tmp_path):
    fixture = Path(os.environ["NRNSIFT_RHS_FIXTURE"])
    config = load_config(str(fixture/"config.yaml"))
    current = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    references = config["native_controlled_comparators"]["references"]
    references["mechanistic_residual"] = copy.deepcopy(current["native_controlled_comparators"]["references"]["mechanistic_residual"])
    references["same_compiler_solver"] = copy.deepcopy(current["native_controlled_comparators"]["references"]["same_compiler_solver"])
    config["native_controlled_comparators"]["conditional_mean_methods"].append("mechanistic_residual")
    for key in ("native_predose_query", "native_predose_evaluator", "native_arrived_observation_filter"):
        config[key] = copy.deepcopy(current[key])
    config["schemas"]["native_predose_episode"] = current["schemas"]["native_predose_episode"]
    config["native_scientific_training"]["selection"].update(max_epochs=1, minimum_epochs=1,
        minimum_steps=1, max_steps=2, max_seconds=1200)
    path = tmp_path/"hybrid_software_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    retained_fit = os.environ.get("NRNSIFT_HYBRID_FIT")
    fit = Path(retained_fit) if retained_fit else tmp_path/"fit"
    base = [sys.executable, "Libs/exp/baselines/run_external_baseline_bridge.py", "--protocol-scope", "manuscript-native",
        "--baseline-name", "mechanistic_residual", "--config", str(path), "--conditioning-config", str(fixture/"config.yaml"),
        "--conditioning-manifest", str(fixture/"native_fit/manifest.json"), "--device", "cpu"]
    command = [*base, "--command", "train_baseline", "--output-dir", str(fit)]
    if retained_fit:
        commands = [{"reuse_existing_fit": str(fit), "new_optimizer_steps": 0}]
    else:
        trained = subprocess.run(command, capture_output=True, text=True)
        (tmp_path/"train_cli.log").write_text(trained.stdout+trained.stderr)
        commands = [{"argv": command, "returncode": trained.returncode}]
        assert trained.returncode == 2, trained.stdout+trained.stderr
    (tmp_path/"commands.json").write_text(json.dumps(commands, indent=2))
    manifest_path = fit/"native_comparator_training_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # The epoch cap can stop at the first development validation. The two-step
    # limit is an upper bound, not evidence that two updates must have occurred.
    assert 0 < manifest["executed_steps"] <= 2 and manifest["selection"]["best"]
    assert manifest["failure"] is None and not manifest["selection"]["sufficient_training"]
    assert manifest["offline_residual_adaptation"]["online_adaptation"] == "NOT_IMPLEMENTED"
    development, declarations = declared_dataset(config, "development")
    arguments = (config, "mechanistic_residual", fixture/"config.yaml", fixture/"native_fit/manifest.json",
                 manifest_path, development, torch.device("cpu"))
    model, belief, binding = load_native_comparator(*arguments)
    inputs = load_native_prediction_inputs(config, declarations[development.group_uids[0]], belief,
                                          load_family_registries(config))
    prediction = model.predict_native(inputs, belief.field_normalization, query_count=3)
    restored, second_belief, second_binding = load_native_comparator(*arguments)
    again = restored.predict_native(inputs, second_belief.field_normalization, query_count=3)
    assert torch.equal(prediction, again) and binding["checkpoint_sha256"] == second_binding["checkpoint_sha256"]
    assert model.output.weight.count_nonzero() > 0
    mechanical = model.solver.forecast_native(inputs, belief.field_normalization, query_count=3)["particle_fields"].reshape_as(prediction)
    assert not torch.equal(prediction[:, 1:], mechanical[:, 1:])
    training, _ = declared_dataset(config, "train")
    batch = training[0]["native_joint_group"]
    training_inputs, target, mask, weights = prepare_comparator_group(batch, belief, config, torch.device("cpu"))
    engine = NativeFrozenRHSSolver(config, references["same_compiler_solver"]["hyperparameters"])
    state = engine.initialize_continuation(training_inputs, belief.field_normalization)
    observed = mask[:, None, :, 0].expand(1, 32, 3, 6, 6, 26)
    engine.teacher_force_independent_state(state, target[:, None, :, 0].expand_as(observed), observed=observed)
    next_field = engine.advance_continuation(state, query_count=2)["particle_fields"][:, :, :, -1]
    residual = (target[:, None, :, 1]-next_field)/belief.field_normalization.scale
    selected = engine.independent_state_mask(state)
    residual_rms = float(residual[selected].square().mean().sqrt())
    assert residual_rms > 0
    with torch.no_grad():
        post_loss = float(model.one_step_residual_loss(training_inputs, target, mask, weights, belief.field_normalization))
    assert post_loss > 0
    summary = {"scope": "artificial_software_fixture_not_scientific_training", "executed_optimizer_steps": manifest["executed_steps"],
        "trainable_parameter_count": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "one_step_target_normalized_residual_rms": residual_rms, "post_fit_one_step_loss": post_loss,
        "selected_checkpoint": binding, "reload_prediction_bitwise_equal": True,
        "recursive_prediction_differs_from_zero_residual_solver": True,
        "scientific_twin": "NOT_RUN", "formal_qualification": False}
    (tmp_path/"result.json").write_text(json.dumps(summary, indent=2))
    destination = tmp_path/"queries"
    command = [*base, "--command", "evaluate_predose_queries", "--training-manifest", str(manifest_path),
        "--decision-reference", str(fixture/"public_cns_reference.json"), "--output-dir", str(destination)]
    result = subprocess.run(command, capture_output=True, text=True)
    (tmp_path/"query_cli.log").write_text(result.stdout+result.stderr)
    commands = json.loads((tmp_path/"commands.json").read_text())
    commands.append({"argv": command, "returncode": result.returncode})
    (tmp_path/"commands.json").write_text(json.dumps(commands, indent=2))
    assert result.returncode == 0, result.stdout+result.stderr
    report = json.loads((destination/"native_predose_evaluation.json").read_text())
    assert report["completed_group_count"] == report["declared_group_count"] == 4
    assert all(row["conditional_law"] == "one_deterministic_path_per_tuple_no_residual_likelihood_qualification" for row in report["groups"])
