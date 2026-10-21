"""Fitted-predictor CLI consumer for the prospective pre-dose query component."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from Libs.scripts.builders.build_sample_observation_registry import NativePredoseMeasurementProvider
from Libs.utils.data.native_diffusion_data import load_family_registries, registry_path
from Libs.utils.data.native_input_staging import input_locations_from_config
from Libs.utils.data.raw_data_registry import write_json
from Libs.utils.evaluation.native_intervention_decisions import NativePublicCNSLoss, compiled_action_burden
from Libs.utils.model.controlled_cns_source import DYNAMIC
from Libs.utils.model.native_comparator_inputs import load_native_prediction_inputs
from Libs.utils.model.native_predose_query import (
    native_predose_channel, original_predose_readout, predose_query_policy, run_native_predose_episode,
)
from Libs.utils.exp.experiment_manifests import resolve_training_seed
from Libs.utils.evaluation.native_comparator_evaluation import (
    load_native_comparator_residual, draw_native_comparator_residual,
)


def predose_source_admission(config: dict, dataset) -> dict:
    """Require both admissions for scientific sources, without opening producer truth.

    The registered reference t0 preserves the six CNS dynamic coordinates:
    integrate_coupled explicitly writes law.initial at t0; controlled and held
    long-CNS integrators save x0 before integration and copy DYNAMIC unchanged.
    No claim is made that every auxiliary/PK quantity has pre-event semantics.
    """
    teachers = [child.native_records[position]["teacher"]
                for child, positions, _ in dataset.groups for position in positions]
    locations = input_locations_from_config(config)
    if locations is not None:
        locations.require_groups(dataset.split_name, dataset.group_uids)
    def read(path):
        return json.loads(locations.read_text(path) if locations else Path(path).read_text())
    if not teachers:
        raise ValueError("Pre-dose evaluation requires declared admitted targets")
    fixture = [row.get("software_fixture_only") is True for row in teachers]
    if all(fixture):
        for path in {row["qc_catalog_ref"] for row in teachers}:
            receipt = read(path)
            if (receipt.get("scope") != "artificial_software_fixture_no_source_trajectory"
                    or receipt.get("physical_source_trajectory_calls") != 0
                    or receipt.get("target_generation") != "initial_complete_tuple_zero_plus_masked_linear_ramp"):
                raise ValueError("Artificial pre-dose target construction lacks its retained receipt")
        return {"scope": "artificial_software_fixture", "pre_dose_binding": "retained_initial_field_plus_zero_t0_ramp"}
    if any(fixture):
        raise ValueError("Scientific pre-dose evaluation cannot mix artificial and admitted sources")
    named_path = registry_path(config, input_locations=locations).parent/"native_consumption.json"
    named = read(named_path)
    base_path = Path(named["original_consumption_ref"])
    base = read(base_path)
    required = config["native_observable_bridge"]["budget"]["cases"]
    if (named.get("status") != "complete" or base.get("status") != "complete"
            or named.get("operation") != "bind_retained_physical_candidates"
            or named.get("admitted_count") != required or base.get("admitted_count") != required
            or named.get("source_identity") != base.get("source_identity")
            or not named.get("array_values_unchanged") or not named.get("compiled_programs_unchanged")):
        raise ValueError("Scientific pre-dose evaluation requires complete unchanged base and named-parameter admissions")
    return {"scope": "admitted_controlled_source", "base_admission": str(base_path),
        "named_admission": str(named_path), "source_identity": named["source_identity"],
        "pre_dose_binding": "original_six_CNS_dynamic_readouts_preserved_at_registered_reference_t0"}


@torch.no_grad()
def evaluate_native_predose_queries(config, model, belief, binding, dataset, declarations,
                                    reference: dict, output: Path, *, residual_path: Path | None = None) -> dict:
    policy = predose_query_policy(config)
    measurement_seed = config["native_predose_evaluator"]["measurement_seed"]
    if type(measurement_seed) is not int:
        raise ValueError("Evaluator measurement seed must be explicit and private to the executor")
    path = Path(output)/"native_predose_evaluation.json"
    if path.exists():
        raise FileExistsError("Native pre-dose component report already exists")
    admission = predose_source_admission(config, dataset)
    hybrid = getattr(model, "prediction_kind", None) == "offline_one_step_mechanistic_residual_conditional_mean"
    if residual_path is not None and not hybrid:
        raise ValueError("The pre-dose residual artifact belongs only to its declared hybrid predictor")
    residual = load_native_comparator_residual(residual_path, config, binding) if residual_path is not None else None
    if hybrid and residual is None and admission["scope"] != "artificial_software_fixture":
        raise ValueError("Scientific hybrid query evaluation requires independently admitted residual calibration; this uncalibrated component is diagnostic only")
    loss_function = NativePublicCNSLoss(reference, config)
    registries = load_family_registries(config)
    seed = int(resolve_training_seed(config, require_selected=True)["model_seed"])
    report = {"schema_version": "native_predose_evaluation.v1", "status": "started",
        "binding": binding, "reference_id": reference["reference_id"], "source_admission": admission,
        "declared_group_count": len(declarations), "groups": [], "shared_sampling_calls": 0,
        "scope": "prospective_information_only_component_not_full_RQ4_or_residual_MPC",
        "scientific_twin": "NOT_RUN", "formal_qualification": False}
    if residual is not None:
        report["residual_calibration"] = {"artifact_path": str(residual_path), "law": residual["law"],
            "fit_status": residual["status"], "calibration_check_complete": True, "scientific_check_pass": None,
            "physical_state_certification": "not_certified_unclipped_teacher_readout"}
    for index, uid in enumerate(dataset.group_uids):
        row = {"group_uid": uid, "status": "started"}
        report["groups"].append(row)
        try:
            declaration = declarations[uid]
            inputs = load_native_prediction_inputs(config, declaration, belief, registries)
            channel = native_predose_channel(inputs, declaration, config)
            selected_names = {channel.readout.species_ids[q] for _, _, q in channel.readout.selector}
            if not selected_names <= set(DYNAMIC) or inputs.grid["time_hours"][0] != 0:
                raise ValueError("Registered target t0 does not establish this pre-dose readout")
            budget = config["native_scientific_training"]["predictive_budget"]
            if hybrid:
                fields = model.predict_native(inputs, belief.field_normalization)
                means = fields.reshape(*inputs.group_shape, *fields.shape[1:])[0]
                if residual is None:
                    samples = means[:, None]
                    row["conditional_law"] = "one_deterministic_path_per_tuple_no_residual_likelihood_qualification"
                else:
                    samples = draw_native_comparator_residual(config, inputs, means, belief.field_normalization, residual)
                    row["conditional_law"] = residual["law"]
                    row["physical_state_certification"] = "not_certified_unclipped_teacher_readout"
            else:
                law = model.sample_native(inputs, belief.field_normalization, budget["conditional_draws_per_particle"],
                    budget["categorical_reverse_steps"], seed+config["native_scientific_training"]["validation_stream_offset"]+index)
                samples = law["particle_trajectory_samples"][:, 0].transpose(0, 1)
            report["shared_sampling_calls"] += 1
            burdens = []
            for action, regimen in enumerate(inputs.regimen_uids):
                candidates = [compiled_action_burden(records[action], inputs.grid["family_uid"], config)
                              for records in inputs.programs["records"][0]]
                if any(value != candidates[0] for value in candidates[1:]):
                    raise ValueError("Native action burden differs across public candidate tuples")
                burdens.append(dict(candidates[0], regimen_uid=regimen))
            supported = [i for i, item in enumerate(burdens) if item["status"] == "defined"]
            if not supported:
                raise ValueError("No intervention has supported public burden")
            cost = samples.new_tensor([burdens[i]["normalized_burden"] for i in supported])
            # This public support mask and every predictive loss precede target access.
            mask = inputs.validity.reshape(1, 32, len(inputs.regimen_uids), *inputs.validity.shape[1:])[0, 0]
            draws = loss_function(samples[:, :, supported], mask[supported], inputs.grid, cost)["loss"]
            # Only the evaluator reads held targets, after the fitted predictions
            # and action-loss table are fixed. It never exports the private field.
            group = dataset[index]["native_joint_group"]
            target = torch.stack([item["trajectory"] for item in group["items"]]).to(samples.device)
            if target.shape[0] != len(inputs.regimen_uids):
                raise ValueError("Evaluator target actions differ from original support")
            private = target[0, 0].unsqueeze(0)
            readouts = original_predose_readout(target[:, 0], channel.readout)
            if not torch.equal(readouts, readouts[:1].expand_as(readouts)):
                raise ValueError("Paired programs do not share the declared pre-dose readout state")
            provider = NativePredoseMeasurementProvider(private, channel,
                torch.Generator(device=private.device).manual_seed(measurement_seed+index))
            decision_config = {"native_predose_query": policy, "schemas": config["schemas"]}
            updated, observations, episode = run_native_predose_episode(channel, inputs.posterior.log_weights,
                draws, tuple(inputs.regimen_uids[i] for i in supported), provider, decision_config,
                origin_posterior_uid=inputs.posterior_uid)
            posterior = inputs.posterior
            for observation in observations:
                posterior = posterior.assimilate(observation, parent_posterior_uid=posterior.posterior_uid, assays={})
            if (not torch.equal(updated, posterior.log_weights)
                    or posterior.support is not inputs.posterior.support
                    or posterior.particle_context is not inputs.posterior.particle_context
                    or posterior.observation_features is not inputs.posterior.observation_features):
                raise ValueError("Pre-dose component differs from shared posterior assimilation")
            row.update(status=episode["status"], family_uid=inputs.grid["family_uid"],
                source_uid=channel.readout.source_uid, episode=episode, burdens=burdens,
                supported_action_count=len(supported), declared_action_count=len(burdens),
                final_posterior_uid=posterior.posterior_uid, original_context_and_encoder_preserved=True)
            if episode["status"] == "component_complete":
                selected = list(inputs.regimen_uids).index(episode["selected_regimen_uid"])
                observed = loss_function(target[supported], mask[supported], inputs.grid, cost)["loss"]
                chosen = supported.index(selected)
                before = list(inputs.regimen_uids).index(episode["initial_selected_regimen_uid"])
                baseline = observed[supported.index(before)]
                row.update(selected_observed_public_loss=float(observed[chosen]),
                    initial_action_observed_public_loss=float(baseline),
                    realized_loss_change_after_queries=float(observed[chosen]-baseline),
                    supported_set_observed_regret=float(observed[chosen]-observed.min()),
                    observed_public_losses=observed.cpu().tolist(),
                    evaluation_origin="original_t0_supported_program_no_postzero_switch")
        except (ValueError, KeyError, RuntimeError, FloatingPointError, OSError) as error:
            row.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        finally:
            write_json(path, report)
    report["completed_group_count"] = sum(row["status"] == "component_complete" for row in report["groups"])
    complete_ids = [row["group_uid"] for row in report["groups"]]
    report["status"] = ("component_complete" if report["completed_group_count"] == len(declarations)
        and len(complete_ids) == len(set(complete_ids)) and set(complete_ids) == set(declarations) else "incomplete")
    write_json(path, report)
    return report
