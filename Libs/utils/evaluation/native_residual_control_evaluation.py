"""Prospective residual control on original public programs and evaluator-private worlds."""
from __future__ import annotations

from Libs.utils.data.native_actuation import actuation_payload

import copy
from dataclasses import replace
import json
import time
from pathlib import Path

import torch

from Libs.models.modules.native_mechanistic_residual import NativeResidualOnlineEpisode
from Libs.models.modules.world_state import JointParticleSupport
from Libs.utils.data.native_actuation import (
    NativeObservationIdentity, NativeActuation, NativeDoseEvent, native_actuation_library, compile_actuation_plan, executed_prefix,
    molecular_administration_policy, molecular_administration_bound, actuation_from_payload,
)
from Libs.utils.data.native_diffusion_data import load_family_registries, load_joint_sources
from Libs.utils.data.joint_physical_parameters import flatten_cns_pk_parameters, candidate_from_payload, validate_candidate_parameters
from Libs.utils.data.native_joint_source_storage import read_joint_source_reference
from Libs.utils.data.physical_occupancy import digest
from Libs.utils.data.raw_data_registry import write_json
from Libs.utils.evaluation.native_comparator_evaluation import (
    load_native_comparator_residual, admit_native_residual_calibration, declared_dataset,
)
from Libs.utils.evaluation.native_intervention_decisions import (
    NativePublicCNSLoss, composed_action_burden, summarize_posterior_action_loss,
)
from Libs.utils.evaluation.native_physical_plant import NativePhysicalPlant
from Libs.utils.evaluation.native_predose_evaluation import predose_source_admission
from Libs.utils.model.native_arrived_observation import native_cns_readout, NativeCNSReadout
from Libs.utils.model.native_conditional_readout import (
    AdaptiveReadoutReference, FixedReadoutReference, normalized_adaptive_initial_weights,
    AdaptiveCalibrationEpisode, reload_adaptive_readout_history, fit_adaptive_readout_scales,
    adaptive_history_log_likelihood, adaptive_readout_window, AdaptiveReadoutQuery, adaptive_observable_species, adaptive_calibration_objective, adaptive_scale_boundary_diagnostics,
)
from Libs.utils.data.native_observable_profile import original_query_time_hours
from Libs.utils.model.native_comparator_inputs import load_native_prediction_inputs
from Libs.utils.model.native_future_controls import public_suffix_candidates
from Libs.utils.model.native_temporal_query import (
    temporal_query_policy, compile_temporal_world_plans, evaluate_temporal_world_plans,
    plan_temporal_query, query_execution_reserve, verify_arrival_forecast, select_initial_public_plan,
)
from Libs.utils.model.native_observable_source import MOLECULAR
from Libs.utils.model.native_scientific_source import (
    is_long_cns, scoped_group_config, selected_group_definitions,
)
from Libs.utils.model.native_direct_mean import file_sha256
from Libs.utils.model.native_training_selection import training_checkpoint_path


def residual_control_policy(config):
    policy = config.get("native_residual_control")
    expected = {
        "schema_version": "native_residual_control.v1",
        "predictor": "mechanistic_residual",
        "decision_phase": "postzero_original_query_post_event",
        "selection": "posterior_expected_public_loss",
        "tie_break": "original_library_order",
        "loss_history": "recorded_model_past_original_full_grid",
        "burden": "composed_public_program_original_physical_support",
        "initial_execution": "explicit_prespecified_original_library_index",
        "forecast_law": "one_deterministic_path_per_complete_tuple",
    }
    required = config["schemas"]["native_residual_control"]["required_fields"]
    if (not isinstance(policy, dict) or set(required)-set(policy)
            or any(policy.get(key) != value for key, value in expected.items())):
        raise ValueError("Residual control requires its original-grid public selection contract")
    parsed = copy.deepcopy(policy)
    # Historical v1 fixed-index packages retain their paired schema and caller.
    # Only the old schema permits missing prospective fields. A current schema
    # still rejects their absence through required_fields above.
    parsed.setdefault("initial_selection_mode", "explicit_fixed_index")
    parsed.setdefault("readout_mode", "sensor_only")
    if parsed["readout_mode"] not in {"sensor_only", "original_action_adaptive_unqualified"}:
        raise ValueError("Residual control readout mode is unbound")
    parsed.setdefault("optimized_initial_selection",
                      "minimum_posterior_terminal_public_loss_complete_open_loop_plans")
    if (parsed["initial_selection_mode"] not in {"explicit_fixed_index", "public_open_loop_plan", "public_joint_query_control"}
            or parsed["optimized_initial_selection"]
                != "minimum_posterior_terminal_public_loss_complete_open_loop_plans"
            or parsed["initial_selection_mode"] == "public_joint_query_control"
                and parsed.get("joint_initial_selection") != "minimum_expected_terminal_query_cost_and_latency_complete_depth_two"):
        raise ValueError("Initial public-plan selection contract is unbound")
    return parsed


def residual_initial_mode(config, *, initial_index, optimize_initial_plan=False, optimize_initial_joint_plan=False):
    """Keep fixed-index comparison explicit and reject conflicting t0 requests."""
    policy = residual_control_policy(config)
    mode = policy["initial_selection_mode"]
    if mode not in {"explicit_fixed_index", "public_open_loop_plan", "public_joint_query_control"}:
        raise ValueError("Initial selection mode is unbound")
    if type(optimize_initial_plan) is not bool or type(optimize_initial_joint_plan) is not bool:
        raise ValueError("Initial optimization flag must be boolean")
    if optimize_initial_plan and optimize_initial_joint_plan:
        raise ValueError("Open-loop and joint initial optimization are ambiguous")
    if optimize_initial_plan:
        mode = "public_open_loop_plan"
    if optimize_initial_joint_plan:
        if policy.get("joint_initial_selection") != "minimum_expected_terminal_query_cost_and_latency_complete_depth_two":
            raise ValueError("Joint initial selection contract is unbound")
        mode = "public_joint_query_control"
    if mode in {"public_open_loop_plan", "public_joint_query_control"}:
        if initial_index is not None:
            raise ValueError("Optimized initial plan and explicit --initial-action-index are ambiguous")
    elif type(initial_index) is not int or initial_index < 0:
        raise ValueError("Fixed initial control requires explicit --initial-action-index >=0")
    return mode


def residual_readout_mode(config, *, adaptive_readout=False):
    """Explicit prospective mode; offline scales do not qualify adaptive errors."""
    mode = residual_control_policy(config)["readout_mode"]
    if type(adaptive_readout) is not bool:
        raise ValueError("Adaptive readout flag must be boolean")
    return "original_action_adaptive_unqualified" if adaptive_readout else mode


def adaptive_control_reference(inputs, normalization, readout, library, initial_index,
                               residual, residual_path, binding, adaptive_artifact=None):
    """Copy already validated scales into the existing original-action reference.

    This target-free boundary runs before episode initialization or private plant
    access. The offline artifact is only the inherited diagnostic scale model. Observable adaptive fit/check and final-policy artifacts may be supplied separately.
    Whole26 adaptive qualification remains a separate requirement.
    """
    if residual is None or residual_path is None:
        raise ValueError("Adaptive readout requires a validated offline residual artifact; supply the offline starting artifact before adaptive execution")
    support = inputs.posterior.support
    if (type(library) is not tuple or not library or type(initial_index) is not int
            or not 0 <= initial_index < len(library)
            or tuple(action.regimen_uid for action in library) != inputs.regimen_uids
            or inputs.normalization_id != normalization.identity
            or tuple(support.sample_uids) != inputs.sample_uids
            or tuple(tuple(row) for row in support.particle_uids) != inputs.particle_uids
            or readout.source_uid != support.source_uid
            or readout.source_snapshot_ref != support.source_snapshot_ref
            or readout.role != support.role or readout.family_uid != support.source_family_uid
            or readout.region_ids != tuple(inputs.grid["region_id"])
            or readout.compartment_ids != tuple(inputs.grid["compartment_id"])
            or readout.species_ids != tuple(inputs.grid["species_id"])):
        raise ValueError("Adaptive readout source/action/tuple/grid/normalizer binding differs")
    action = library[initial_index]
    hours = original_query_time_hours(action.source_times, action.source_time_unit)
    if tuple(inputs.grid["time_hours"]) != hours:
        raise ValueError("Adaptive readout source/hour clock binding differs")
    fixed = FixedReadoutReference.from_validated(readout=readout, normalization=normalization,
        residual_artifact=residual, residual_artifact_ref=str(residual_path), predictor_binding=binding,
        support_uid=support.support_uid, origin_posterior_uid=inputs.posterior_uid,
        sample_uids=inputs.sample_uids, particle_uids=inputs.particle_uids,
        executed_regimen_uid=action.regimen_uid, source_times=action.source_times,
        time_hours=hours, source_time_unit=action.source_time_unit)
    reference = AdaptiveReadoutReference(fixed, library, action.regimen_uid)
    if adaptive_artifact is not None:
        fixed = replace(fixed, residual_scales=tuple(adaptive_artifact["scales"]))
        reference = replace(reference, fixed=fixed, scale_authority="authenticated_observable_adaptive_fit",
            adaptive_artifact_id=adaptive_artifact["artifact_id"],
            fitted_species_indices=tuple(adaptive_artifact["fitted_species_indices"]))
    reference.validate()
    return reference


def residual_collection_policy(config: dict, reservation: str, phase: str | None = None) -> dict:
    if phase is not None:
        policy = config["native_adaptive_residual_calibration"]
        expected = {"adaptive_fit": "fit", "final_policy_check": "check", "final_selected_loss": "threshold"}
        if (policy["schema_version"] != "native_adaptive_residual_calibration.v1"
                or expected.get(phase) != reservation or policy["groups_per_family"] != 512
                or policy["fit_pool_use"] != "compound_offline_and_adaptive_fit_not_independent"
                or policy["check_pool_use"] != "independent_of_all_fitting_final_policy"
                or policy["threshold_pool_use"] != "untouched_until_final_policy_frozen"):
            raise ValueError("Adaptive calibration requires its explicit phase and immutable reservation")
        return dict(policy, purpose=phase)
    policy = config["native_residual_calibration_collection"]
    if (reservation not in {"fit", "check"}
            or policy["schema_version"] != "native_residual_calibration_collection.v1"
            or policy["purpose"] != "frozen_policy_actual_episode_history_collection"
            or policy["groups_per_family"] != 512
            or policy["roles"] != {"fit": "calibration", "check": "calibration"}
            or policy["threshold_use"] != "excluded"
            or policy["calibration_eligible"] is not False):
        raise ValueError("Residual collection requires its frozen purpose and fit/check calibration reservation")
    return policy


def _collection_public_config(config):
    # The evaluator measurement stream is private. Its seed never enters
    # public collection binding or history, including through a config snapshot.
    if isinstance(config, dict):
        return {key: _collection_public_config(value) for key, value in config.items()
                if key not in {"measurement_seed", "measurement_generator", "rng_state",
                               "private_fields", "private_initial_field", "hidden_index", "latent_index"}}
    if isinstance(config, (list, tuple)):
        return [_collection_public_config(value) for value in config]
    return config


def _collection_selected_predictor(binding: dict, *, artificial: bool, config=None, source_admission=None) -> dict:
    if artificial:
        return {"scope": "artificial_software_fixture", "scientific_checkpoint_sufficiency": False}
    conditioning = binding["model_identity"]["conditioning"]
    for name, identity, manifest_key in (("predictor", binding, "training_manifest"),
                                         ("conditioning", conditioning, "manifest_path")):
        manifest_path = Path(identity[manifest_key])
        manifest = json.loads(manifest_path.read_text())
        selection = manifest.get("selection", {})
        if selection.get("selection_role") != "development" or not selection.get("best"):
            raise ValueError("Production collection requires development-selected "+name)
        checkpoint = training_checkpoint_path(manifest_path, selection, selection["best"]["checkpoint_path"])
        if (str(checkpoint.resolve()) != str(Path(identity["checkpoint_path"]).resolve())
                or file_sha256(checkpoint) != identity["checkpoint_sha256"]):
            raise ValueError("Collection selected checkpoint differs from the loaded "+name)
        if selection.get("sufficient_training") is not False:
            raise ValueError("Native training sufficiency has no canonical reviewed verdict; a boolean cannot authorize collection")
    if config is None or source_admission is None:
        raise ValueError("Production collection requires canonical reviewed training sufficiency and source context")
    # Delayed import keeps historical audit/training imports outside fixture startup.
    from Libs.utils.evaluation.operational_twin_training_sufficiency import reviewed_native_predictor_training
    return reviewed_native_predictor_training(config, binding, source_admission)



def collect_native_residual_calibration(config, model, belief, binding, fit, check, reference, output,
                                        *, reservation, residual_path, initial_index=None,
                                        optimize_initial_plan=False, optimize_initial_joint_plan=False,
                                        world_factory=None, phase=None, adaptive_artifact=None,
                                        check_artifact=None, threshold=None):
    """Collect actual histories for the declared phase without fitting outcomes."""
    policy = residual_collection_policy(config, reservation, phase)
    partitions = admit_native_residual_calibration(config, fit, check)
    admissions = {name: predose_source_admission(config, dataset)
                  for name, dataset in (("fit", fit), ("check", check))}
    if admissions["fit"] != admissions["check"]:
        raise ValueError("Collection fit/check datasets have different original source admissions")
    artificial = admissions["fit"]["scope"] == "artificial_software_fixture"
    families = tuple(config["native_observable_bridge"]["quantity_profile"]["families"])
    thresholds = selected_group_definitions(config, "calibration", "threshold")
    counts = {name: {family: sum(row["family_uid"] == family for row in rows)
                    for family in families}
              for name, rows in (("fit", partitions["fit"].values()), ("check", partitions["check"].values()),
                                 ("threshold", thresholds))}
    threshold_ids = {row["sample_uid"] for row in thresholds}
    if (len(threshold_ids) != len(thresholds)
            or threshold_ids.intersection(set(partitions["fit"]) | set(partitions["check"]))):
        raise ValueError("Collection cannot use threshold reservations")
    complete = (len(families) == 4
                and all(value == policy["groups_per_family"] for row in counts.values() for value in row.values())
                and all(len(rows) == len(families)*policy["groups_per_family"]
                        for rows in (*partitions.values(), thresholds)))
    if not complete and not artificial:
        raise ValueError("Production collection requires all four families and512 fit/check/threshold groups each")
    selected = _collection_selected_predictor(binding, artificial=artificial, config=config, source_admission=admissions["fit"])
    context = {"purpose": policy["purpose"], "reservation": reservation,
               "source_admission": admissions["fit"], "selected_predictor": selected,
               "denominator": {"required_groups_per_family": policy["groups_per_family"],
                               "declared_by_family": counts, "production_complete": complete,
                               "software_fixture_only": artificial},
               "roles": {name: sorted(rows) for name, rows in partitions.items()},
               "threshold_group_uids": sorted(threshold_ids)}
    context["phase"] = phase
    if phase in {"final_policy_check", "final_selected_loss"} and adaptive_artifact is None:
        raise ValueError("Final-policy collection requires the authenticated adaptive fit artifact")
    if phase == "final_selected_loss" and check_artifact is None:
        raise ValueError("Threshold collection requires the observed independent final-policy check artifact")
    if reservation == "threshold":
        if threshold is None or set(threshold.group_uids) != threshold_ids:
            raise ValueError("Threshold collection requires the exact untouched declared threshold dataset")
        partitions["threshold"] = {row["sample_uid"]: row for row in thresholds}
        context["roles"]["threshold"] = sorted(threshold_ids)
        if predose_source_admission(config, threshold) != admissions["fit"]:
            raise ValueError("Threshold original source admission differs from fit/check")
    dataset = fit if reservation == "fit" else threshold if reservation == "threshold" else check
    return evaluate_native_residual_controls(config, model, belief, binding, dataset,
        partitions[reservation], reference, output, initial_index=initial_index,
        optimize_initial_plan=optimize_initial_plan, optimize_initial_joint_plan=optimize_initial_joint_plan,
        residual_path=residual_path, world_factory=world_factory, adaptive_readout=True,
        collection_context=context, adaptive_artifact=adaptive_artifact, check_artifact=check_artifact)



def final_residual_policy_id(config, binding, public_reference, residual, adaptive_artifact, initial_selection):
    """Freeze the fitted law and every public rule that affects final episodes."""
    return digest({"policy": _collection_public_config(config), "predictor": binding,
        "public_reference": public_reference, "offline_residual": residual,
        "adaptive_artifact_id": adaptive_artifact["artifact_id"], "initial_selection": initial_selection})


def _read_collection_report(path, config, reservation, phase):
    """Admit the collector's actual output before likelihood or selected loss use.

    Transport JSON is not standalone source authority. Production additionally
    needs current source receipts and selected-checkpoint sufficiency admission.
    """
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    frozen = report.get("collection_binding", {})
    if (report.get("schema_version") != "native_residual_calibration_collection.v1"
            or report.get("status") not in {"collection_complete", "software_collection_complete"}
            or report.get("reservation") != reservation or report.get("collection_binding_id") != digest(frozen)
            or frozen.get("reservation") != reservation or frozen.get("phase") != phase
            or frozen.get("configuration_sha256") != digest(_collection_public_config(config))
            or report.get("binding") != frozen.get("predictor")):
        raise ValueError("Calibration collection is incomplete or its phase/configuration/binding differs")
    dataset, metadata_declarations = declared_dataset(config,"calibration",reservation=reservation)
    actual_admission = predose_source_admission(config,dataset)
    if actual_admission != frozen["source_admission"]:
        raise ValueError("Collection source scope differs from canonical metadata and admission receipt")
    artificial = actual_admission["scope"] == "artificial_software_fixture"
    families = tuple(config["native_observable_bridge"]["quantity_profile"]["families"])
    if not artificial:
        counts = {family:sum(row["family_uid"] == family for row in metadata_declarations.values()) for family in families}
        required = config["native_adaptive_residual_calibration"]["groups_per_family"]
        if (len(families)!=4 or len(metadata_declarations)!=len(families)*required
                or any(value!=required for value in counts.values())
                or report["denominator"]["production_complete"] is not True):
            raise ValueError("Production artifact collection requires all four complete reserved family pools")
    selected = _collection_selected_predictor(report["binding"], artificial=artificial,
                                              config=config, source_admission=actual_admission)
    if selected != frozen["selected_predictor"] or artificial != report["denominator"]["software_fixture_only"]:
        raise ValueError("Collection source and selected-predictor authority differ")
    expected = selected_group_definitions(config, "calibration", reservation)
    declarations = {row["sample_uid"]: row for row in expected}
    if (len(declarations) != len(expected) or metadata_declarations != declarations
            or set(dataset.group_uids) != set(declarations)
            or set(frozen["roles"][reservation]) != set(declarations)):
        raise ValueError("Collection reservation differs from the immutable complete declared pool")
    if reservation == "check" and set(declarations).intersection(
            frozen["offline_starting_residual"].get("fit_group_uids", [])):
        raise ValueError("Check pool overlaps offline fitting")
    if reservation == "threshold" and set(declarations).intersection(
            set(frozen["roles"]["fit"]) | set(frozen["roles"]["check"])):
        raise ValueError("Threshold pool overlaps fitting or check")
    rows = report["groups"]
    if (len(rows) != len(declarations) or {row["group_uid"] for row in rows} != set(declarations)
            or any(row["status"] != "component_complete" for row in rows)):
        raise ValueError("Every declared group remains in the calibration denominator")
    episodes = []
    for row in rows:
        declaration = declarations[row["group_uid"]]
        episode_binding = row["episode_collection_binding"]
        if (episode_binding["collection_binding_id"] != report["collection_binding_id"]
                or episode_binding["declaration"] != _collection_public_config(declaration)
                or row["family_uid"] != declaration["family_uid"]):
            raise ValueError("Collection episode source or reservation binding differs")
        raw = copy.deepcopy(episode_binding["adaptive_reference"])
        fixed = raw.pop("fixed")
        readout = fixed.pop("readout")
        for key in ("region_ids", "compartment_ids", "species_ids", "scales", "observed"):
            readout[key] = tuple(readout[key])
        for key in ("selector", "matrix"):
            readout[key] = tuple(tuple(values) for values in readout[key])
        readout = NativeCNSReadout(**readout)
        if readout != native_cns_readout(config, declaration):
            raise ValueError("Collection readout differs from the original public source definition")
        for key in ("sample_uids", "residual_scales", "inverse_normalizer_scales", "source_times", "time_hours"):
            fixed[key] = tuple(fixed[key])
        fixed["particle_uids"] = tuple(tuple(values) for values in fixed["particle_uids"])
        fixed = FixedReadoutReference(readout=readout, **fixed)
        actions = []
        for action in raw.pop("original_actions"):
            actions.append(actuation_from_payload(action))
        if tuple(actions) != native_actuation_library(config, declaration):
            raise ValueError("Collection original action library differs from source authority")
        raw["fitted_species_indices"] = tuple(raw.get("fitted_species_indices", ()))
        reference = AdaptiveReadoutReference(fixed=fixed, original_actions=tuple(actions), **raw)
        if (fixed.predictor_binding != json.dumps(report["binding"], sort_keys=True, allow_nan=False)
                or fixed.normalization_id != report["binding"]["normalization_id"]
                or fixed.sample_uids != (row["group_uid"],)):
            raise ValueError("Collection fixed predictor/normalizer/sample binding differs")
        payload = row["actual_adaptive_readout_history"]
        context = {"group_uid": row["group_uid"], "family_uid": row["family_uid"],
            "binding": report["binding"], "reference_id": report["reference_id"],
            "pre_execution_policy_snapshot": report["pre_execution_policy_snapshot"]}
        restored = reload_adaptive_readout_history(payload, expected_reference=reference, expected_report_context=context)
        if list(payload["initial_log_weights"]) != episode_binding["initial_log_weights"] or restored.status != ("empty" if reservation == "threshold" and not restored.history else "complete"):
            raise ValueError("Collection initial immutable prior or complete accepted history differs")
        episodes.append(AdaptiveCalibrationEpisode(row["group_uid"], reference, restored.history, restored.initial_log_weights))
    return report, tuple(episodes), {"path": str(path), "sha256": file_sha256(path),
        "collection_binding_id": report["collection_binding_id"]}


def _artifact_write(output, filename, artifact):
    path = Path(output)/filename
    if path.exists():
        raise FileExistsError("Calibration artifact already exists")
    artifact["artifact_id"] = digest(artifact)
    write_json(path, artifact)
    return artifact


def validate_adaptive_fit_artifact(artifact, config, *, binding=None, residual=None):
    body = {key: value for key, value in artifact.items() if key != "artifact_id"}
    if (type(artifact) is not dict or artifact.get("schema_version") != "native_adaptive_residual_fit.v1"
            or artifact.get("artifact_id") != digest(body) or artifact.get("status") != "FIT_COMPLETE"
            or artifact.get("observable_fit_valid") is not True
            or artifact.get("whole_26_field_adaptive_qualification") != "UNQUALIFIED"
            or len(artifact.get("scales", [])) != 26
            or not torch.isfinite(torch.tensor(artifact["scales"], dtype=torch.float64)).all()
            or any(value <= 0 for value in artifact["scales"])):
        raise ValueError("Adaptive fit artifact has no valid observable fit")
    report, episodes, identity = _read_collection_report(artifact["fit_collection"]["path"], config, "fit", "adaptive_fit")
    if identity != artifact["fit_collection"] or report["binding"] != artifact["binding"]:
        raise ValueError("Adaptive fit source collection identity changed")
    if binding is not None and binding != artifact["binding"]:
        raise ValueError("Adaptive fit belongs to another selected predictor")
    offline = report["collection_binding"]["offline_starting_residual"]
    if residual is not None and residual != offline:
        raise ValueError("Adaptive fit offline starting residual differs")
    indices = artifact["fitted_species_indices"]
    if (tuple(indices) != adaptive_observable_species(episodes) or len(set(indices)) != len(indices) or not indices
            or artifact["inherited_species_indices"] != [i for i in range(26) if i not in indices]
            or any(artifact["scales"][i] != offline["scales"][i] for i in artifact["inherited_species_indices"])):
        raise ValueError("Adaptive fit must preserve every unobserved offline scale")
    numerical = config["native_controlled_comparators"]["residual_calibration"]["numerical"]
    eta = torch.tensor([artifact["scales"][index] for index in indices], dtype=torch.float64).log().requires_grad_()
    objective = lambda value: adaptive_calibration_objective(value, episodes, tuple(indices))
    loss = objective(eta)
    gradient, = torch.autograd.grad(loss, eta)
    hessian = torch.autograd.functional.hessian(objective, eta)
    eigenvalues = torch.linalg.eigvalsh(hessian)
    threshold = max(numerical["hessian_absolute_threshold"],
                    numerical["hessian_relative_threshold"]*float(eigenvalues[-1]))
    if (not all(torch.isfinite(value).all() for value in (loss,gradient,hessian,eigenvalues))
            or float(gradient.abs().max()) > numerical["gradient_tolerance"]
            or float(eigenvalues[0]) <= threshold):
        raise ValueError("Adaptive artifact scales fail the authenticated fit gradient or Hessian rank")
    boundary = adaptive_scale_boundary_diagnostics(episodes,indices,artifact["scales"],numerical["gradient_tolerance"])
    if not boundary["interior_separated"]:
        raise ValueError("Adaptive artifact scales are collapsed at a zero boundary")
    return artifact


def fit_native_adaptive_residual(config, collection_path, output):
    report, episodes, identity = _read_collection_report(collection_path, config, "fit", "adaptive_fit")
    numerical = config["native_controlled_comparators"]["residual_calibration"]["numerical"]
    result = fit_adaptive_readout_scales(episodes, tuple(row.group_uid for row in episodes), numerical)
    result.update(schema_version="native_adaptive_residual_fit.v1", binding=report["binding"],
        fit_collection=identity, fit_group_uids=[row.group_uid for row in episodes],
        compound_fit_pool_reuse=True, offline_adaptive_fit_independence_claimed=False,
        independent_final_policy_check_complete=False)
    return _artifact_write(output, "native_adaptive_residual_fit.json", result)


def validate_adaptive_check_artifact(check, fitted, final_policy_id, config):
    """Recompute required check results from the currently admitted collection."""
    if (type(check) is not dict or type(fitted) is not dict
            or check.get("schema_version") != "native_adaptive_residual_check.v1"
            or check.get("artifact_id") != digest({key: value for key, value in check.items() if key != "artifact_id"})
            or check.get("status") != "CHECK_OBSERVED" or check.get("observable_check_complete") is not True
            or check.get("adaptive_artifact_id") != fitted.get("artifact_id")
            or check.get("final_policy_id") != final_policy_id or "check_collection" not in check):
        raise ValueError("Final-policy independent check artifact differs or is incomplete")
    validate_adaptive_fit_artifact(fitted,config)
    report, episodes, identity = _read_collection_report(check["check_collection"]["path"],config,"check","final_policy_check")
    expected = _adaptive_check_result(config,report,episodes,identity,fitted)
    if {key:value for key,value in check.items() if key != "artifact_id"} != expected:
        raise ValueError("Independent check results or diagnostics differ from the admitted collection")
    return check


def adaptive_check_diagnostics(episodes):
    """Conditional marginal mixture PIT on actual pre-update histories and masks."""
    pit, coordinates = [], {}
    for episode in episodes:
        initial = torch.tensor(episode.initial_log_weights, dtype=torch.float64)
        for offset, row in enumerate(episode.history):
            mean = torch.zeros((32, 6), dtype=torch.float64)
            mean[:, list(row.coordinates)] = torch.tensor(row.mean, dtype=torch.float64)
            query = AdaptiveReadoutQuery(row.context, row.query_index, row.filtering_revision,
                tuple(i in row.coordinates for i in range(6)), mean)
            window = adaptive_readout_window(episode.reference, episode.history[:offset], (query,))
            weights = torch.softmax(initial+window.history_log_likelihood, 0)
            sigma = window.covariance.diag().sqrt()
            values = torch.tensor(row.values, dtype=torch.float64)
            cdf = (weights[:, None] * (.5*(1+torch.erf((values[None]-window.mean)/(sigma[None]*2**.5))))).sum(0)
            if not torch.isfinite(cdf).all():
                raise ValueError("Independent adaptive check PIT is nonfinite")
            pit.extend(cdf.tolist())
            for coordinate, value in zip(row.coordinates, cdf.tolist()):
                coordinates.setdefault(coordinate, []).append(value)
    return {"conditional_marginal_mixture_pit": {"count": len(pit), "mean": sum(pit)/len(pit),
                "by_coordinate": [{"coordinate": coordinate, "count": len(values), "mean": sum(values)/len(values)}
                                  for coordinate, values in sorted(coordinates.items())]},
        "scientific_acceptance_criterion": "NOT_DECLARED", "scientific_distribution_pass": None}


def _adaptive_check_result(config, report, episodes, identity, fitted):
    _validate_final_collection_law(report,episodes,fitted,config)
    if set(fitted["fit_group_uids"]).intersection(row.group_uid for row in episodes):
        raise ValueError("Independent check overlaps adaptive fitting")
    values = [float(adaptive_history_log_likelihood(row.reference,row.history,row.initial_log_weights)) for row in episodes]
    if not values or not torch.isfinite(torch.tensor(values,dtype=torch.float64)).all():
        raise ValueError("Independent check produced nonfinite or empty complete-history likelihood")
    return {"schema_version":"native_adaptive_residual_check.v1","status":"CHECK_OBSERVED",
        "observable_check_complete":True,"scientific_distribution_pass":None,
        "whole_26_field_adaptive_qualification":"UNQUALIFIED","check_collection":identity,
        "adaptive_artifact_id":fitted["artifact_id"],"final_policy_id":report["collection_binding"]["final_policy_id"],
        "group_count":len(values),"mean_complete_history_nll":-sum(values)/len(values),
        "diagnostics":adaptive_check_diagnostics(episodes),
        "group_log_likelihoods":[{"group_uid":row.group_uid,"log_likelihood":value} for row,value in zip(episodes,values)]}


def check_native_adaptive_residual(config, collection_path, fitted, output):
    validate_adaptive_fit_artifact(fitted,config)
    report,episodes,identity = _read_collection_report(collection_path,config,"check","final_policy_check")
    return _artifact_write(output,"native_adaptive_residual_check.json",
                           _adaptive_check_result(config,report,episodes,identity,fitted))


def _selected_loss_errors(report):
    errors = []
    for row in report["groups"]:
        pair = row["selected_loss_collection"]
        stop, realized = row["episode"]["selected_stop_record"], row["realized"]
        actual, risk = realized["realized_full_grid_public_loss"], stop["predicted_risk"]
        if (pair["status"] != "complete" or pair["stop"] != stop
                or realized["realized_loss_status"] != "complete_original_full_grid"
                or stop["reference_id"] != report["reference_id"] or realized["reference_id"] != report["reference_id"]
                or pair["signed_selected_loss_error"] != actual-risk):
            raise ValueError("Selected-loss calibration requires actual full-grid loss minus the frozen STOP risk")
        errors.append(actual-risk)
    return errors


def _validate_final_collection_law(report, episodes, fitted, config):
    frozen = report["collection_binding"]
    expected = final_residual_policy_id(config,report["binding"],frozen["public_loss_reference"],
        frozen["offline_starting_residual"],fitted,frozen["initial_selection"])
    if (frozen["final_policy_id"] != expected or frozen["adaptive_artifact_id"] != fitted["artifact_id"]
            or any(row.reference.adaptive_artifact_id != fitted["artifact_id"]
                   or list(row.reference.fixed.residual_scales) != fitted["scales"] for row in episodes)):
        raise ValueError("Final collection differs from the frozen fitted law and policy")


def calibrate_native_residual_selected_loss(config, collection_path, fitted, checked, output):
    from Libs.utils.evaluation.native_scale_routing import selected_loss_quantile
    validate_adaptive_fit_artifact(fitted, config)
    report, episodes, identity = _read_collection_report(collection_path, config, "threshold", "final_selected_loss")
    frozen = report["collection_binding"]
    validate_adaptive_check_artifact(checked, fitted, frozen["final_policy_id"], config)
    check_report, _, check_identity = _read_collection_report(checked["check_collection"]["path"], config, "check", "final_policy_check")
    if check_identity != checked["check_collection"] or check_report["collection_binding"]["final_policy_id"] != frozen["final_policy_id"]:
        raise ValueError("Selected-loss calibration requires the unchanged final-policy check")
    _validate_final_collection_law(report,episodes,fitted,config)
    errors = _selected_loss_errors(report)
    levels = config["native_adaptive_residual_calibration"]["selected_loss_nominal_levels"]
    return _artifact_write(output, "native_residual_selected_loss_calibration.json", {
        "schema_version": "native_residual_selected_loss_calibration.v1", "status": "CALIBRATION_COMPLETE",
        "threshold_collection": identity, "adaptive_artifact_id": fitted["artifact_id"],
        "check_artifact_id": checked["artifact_id"], "final_policy_id": frozen["final_policy_id"],
        "independent_group_count": len(episodes), "signed_selected_loss_errors": errors,
        "quantiles": [selected_loss_quantile(errors, level) for level in levels],
        "whole_26_field_adaptive_qualification": "UNQUALIFIED", "formal_twin_qualification": False})



def validate_selected_loss_artifact(artifact, fitted, checked, policy_id, config):
    if (fitted is None or checked is None or artifact.get("schema_version") != "native_residual_selected_loss_calibration.v1"
            or artifact.get("status") != "CALIBRATION_COMPLETE"
            or artifact.get("artifact_id") != digest({key: value for key, value in artifact.items() if key != "artifact_id"})
            or artifact.get("adaptive_artifact_id") != fitted["artifact_id"]
            or artifact.get("check_artifact_id") != checked["artifact_id"] or artifact.get("final_policy_id") != policy_id):
        raise ValueError("Selected-loss calibration artifact differs from the frozen final policy")
    validate_adaptive_check_artifact(checked,fitted,policy_id,config)
    report, episodes, identity = _read_collection_report(artifact["threshold_collection"]["path"], config, "threshold", "final_selected_loss")
    if identity != artifact["threshold_collection"] or report["collection_binding"]["final_policy_id"] != policy_id:
        raise ValueError("Selected-loss threshold collection identity changed")
    _validate_final_collection_law(report,episodes,fitted,config)
    from Libs.utils.evaluation.native_scale_routing import selected_loss_quantile
    errors = _selected_loss_errors(report)
    levels = config["native_adaptive_residual_calibration"]["selected_loss_nominal_levels"]
    expected = [selected_loss_quantile(errors,level) for level in levels]
    if (artifact["signed_selected_loss_errors"] != errors or artifact["independent_group_count"] != len(episodes)
            or artifact["quantiles"] != expected):
        raise ValueError("Selected-loss artifact errors, count or quantiles differ from actual threshold pairs")
    return artifact


def apply_selected_loss_bounds(row, calibration):
    from Libs.utils.evaluation.native_scale_routing import selected_loss_upper_bound
    stop = row["episode"]["selected_stop_record"]
    row["selected_loss_bounds"] = {"calibration_artifact_id": calibration["artifact_id"],
        "final_policy_id": calibration["final_policy_id"], "bounds": [{"nominal_level": quantile["nominal_level"],
            "upper_bound": selected_loss_upper_bound(stop["predicted_risk"], quantile, stop["maximum_supported_loss"])}
            for quantile in calibration["quantiles"]], "formal_twin_qualification": False}


def _collection_binding(config, binding, reference, residual, context):
    public = _collection_public_config(config)
    reuse = {name: sorted(set(context["roles"][context["reservation"]]).intersection(
        residual.get(name+"_group_uids", []))) for name in ("fit", "check")}
    result = {"purpose": context["purpose"], "reservation": context["reservation"],
            "configuration_sha256": digest(public), "predictor": copy.deepcopy(binding),
            "selected_predictor": copy.deepcopy(context["selected_predictor"]),
            "source_admission": copy.deepcopy(context["source_admission"]),
            "roles": copy.deepcopy(context["roles"]), "threshold_group_uids": context["threshold_group_uids"],
            "initial_selection": copy.deepcopy(context.get("initial_selection")),
            "public_loss_reference": copy.deepcopy(reference),
            "offline_starting_residual": copy.deepcopy(residual),
            "existing_residual_role_reuse": reuse, "statistical_independence_claimed": False,
            "policy": {key: copy.deepcopy(config[key]) for key in (
                "native_residual_control", "native_temporal_query", "native_mechanistic_residual_online",
                "native_arrived_observation_filter", "native_public_cns_loss")},
            "public_seed_provenance": copy.deepcopy(binding.get("model_identity", {}).get("seed_binding")),
            "calibration_eligible": False, "scientific_readiness": "NOT_ADMITTED",
            "phase": context.get("phase"), "adaptive_artifact_id": context.get("adaptive_artifact_id"),
            "final_policy_id": context.get("final_policy_id")}
    administration = molecular_administration_policy(config)
    if administration is not None:
        result["policy"]["native_molecular_administration"] = administration
    return result


def _collection_episode_binding(episode, library, declaration, frozen):
    initial = episode._adaptive_initial_log_weights
    normalized_adaptive_initial_weights(initial)
    if initial != episode.continuation._initial_readout_log_weights:
        raise ValueError("Collection initial prior differs from its immutable episode authority")
    reference = episode.adaptive_reference
    if (reference is None or reference.fixed.readout.role != "calibration"
            or declaration["role"] != "calibration"
            or declaration["calibration_reservation"] != frozen["reservation"]):
        raise ValueError("Collection episode has another role or reservation")
    return {"collection_binding_id": digest(frozen), "declaration": _collection_public_config(declaration),
            "adaptive_reference": actuation_payload(reference), "original_action_library": [actuation_payload(row) for row in library],
            "initial_log_weights": ["-inf" if value == -float("inf") else value for value in initial]}


def _pair_collection_stop(row, frozen):
    stop = row.get("episode", {}).get("selected_stop_record")
    realized = row.get("realized", {})
    result = {"status": "unavailable", "calibration_eligible": False, "scientific_readiness": "NOT_ADMITTED",
              "adaptive_scientific_qualification": "UNQUALIFIED"}
    if (row["status"] == "component_complete" and stop is not None
            and realized.get("realized_loss_status") == "complete_original_full_grid"):
        if (stop["reference_id"] != frozen["public_loss_reference"]["reference_id"]
                or realized.get("reference_id") != stop["reference_id"]):
            raise ValueError("Collection STOP and realized loss use different public references")
        actual, risk = realized["realized_full_grid_public_loss"], stop["predicted_risk"]
        if not torch.isfinite(torch.tensor([actual, risk], dtype=torch.float64)).all():
            raise ValueError("Collection selected loss and STOP risk must be finite")
        result.update(status="complete", stop=copy.deepcopy(stop), realized_full_grid_public_loss=actual,
                      signed_selected_loss_error=actual-risk, statistical_independence_claimed=False)
    row["selected_loss_collection"] = result


@torch.no_grad()
def select_native_residual_control(episode, loss_function, config):
    """Return scores, selected request and unadvanced branch without actuator/private access."""
    residual_control_policy(config)
    if episode.status not in {"ready", "active"}:
        raise ValueError("Failed residual episode cannot select another control")
    state = episode.continuation.rhs
    candidates = public_suffix_candidates(state)
    rows, eligible = [], []
    for candidate in candidates:
        row = {key: value for key, value in candidate.items() if key not in {"request", "public"}}
        if candidate["status"] == "legal":
            burden = composed_action_burden(candidate["public"], config)
            row.update(burden=burden, status=burden["status"])
            if burden["status"] == "defined":
                eligible.append(candidate)
        rows.append(row)
    result = {"query_index": state.query_index, "candidates": rows,
              "conditional_law": "one_deterministic_path_per_tuple_no_residual_likelihood_qualification",
              "particle_count": 32, "conditional_draw_count": 1,
              "independent_predictive_error_calibration": False,
              "loss_reference_id": loss_function.reference_id,
              "loss_time_grid": "original_full_grid_with_identical_recorded_model_past"}
    undefined = [row["regimen_uid"] for row in rows if "burden" in row and row["burden"]["status"] != "defined"]
    if undefined:
        return dict(result, status="blocked", reason="legal_candidate_burden_undefined", blocked_candidates=undefined), None, None
    if not eligible:
        return dict(result, status="blocked", reason="no_legal_candidate_with_defined_physical_burden"), None, None
    past = episode.recorded_model_history()
    branch = episode.plan_controls([item["regimen_uid"] for item in eligible])
    trajectories = []
    for candidate in eligible:
        forecast, fields, public = episode.model.forecast_public_control(
            episode.continuation, episode._execution_inputs, episode.normalization, branch, candidate["request"])
        if public != candidate["public"] or forecast.rhs._public_control is not None:
            raise ValueError("Residual candidate forecast changed its hypothetical public program")
        trajectories.append(torch.cat((past, fields), dim=1))
    # J x D(1) x A x original T x R x C x Q. Every candidate has the same
    # already recorded model prefix; online fitting cannot rewrite that prefix.
    fields = torch.stack(trajectories, 1)[:, None]
    costs = fields.new_tensor([rows[candidates.index(item)]["burden"]["normalized_burden"] for item in eligible])
    mask = state._validity[0, 0, 0]
    scores = loss_function(fields, mask, state._guide.grid, costs)
    weights = episode.model.solver.current_log_weights(state)[0].exp()
    summary = summarize_posterior_action_loss(scores["loss"], weights)
    for position, candidate in enumerate(eligible):
        row = rows[candidates.index(candidate)]
        row["predictive_risk"] = float(summary["predictive_risks"][position])
        row["status"] = "scored"
    chosen = eligible[summary["selected_index"]]
    result.update(status="selected", selected_regimen_uid=chosen["regimen_uid"],
                  selected_public_program=actuation_payload(chosen["public"]),
                  selected_optimal_mass=summary["selected_optimal_mass"],
                  tied_optimal_mass=summary["tied_optimal_mass"], tie_break=summary["tie_break"],
                  current_particle_weights=weights.cpu().tolist())
    return result, chosen["request"], branch


def run_native_residual_control_episode(episode, loss_function, provider, actuator, config):
    """Select, actuate, advance, acquire, filter and adapt on the original query clock."""
    policy = residual_control_policy(config)
    state = episode.continuation.rhs
    if state._public_control is None or state.query_index != 0:
        raise ValueError("Residual control consumer starts from a bound prespecified original t0 execution")
    report = {"status": "started", "family_uid": state._guide.family, "policy": policy,
              "initial_regimen_uid": state._public_control.regimen_uid,
              "initial_original_library_index": tuple(item.regimen_uid for item in state._execution_library).index(state._public_control.regimen_uid),
              "initial_execution": "prespecified_common_context_not_model_optimized_t0",
              "decisions": [], "arrivals": [], "scientific_twin": "NOT_RUN",
              "formal_qualification": False, "full_residual_mpc_eligible": False,
              "scope": "postzero_model_selected_control_component",
              "conditional_law": "one_deterministic_path_per_tuple_no_residual_likelihood_qualification"}
    if state._guide.family == MOLECULAR and not molecular_administration_bound(state._public_control,config):
        return dict(report, status="blocked", reason="molecular_delivered_dose_mapping_unbound")
    try:
        while episode.continuation.rhs.query_index < len(state._times)-1:
            if episode.continuation.rhs.query_index:
                decision, request, branch = select_native_residual_control(episode, loss_function, config)
                report["decisions"].append(decision)
                if request is None:
                    report.update(status="blocked", reason=decision["reason"])
                    break
                # Candidate forecasts advance selected copies. The original full
                # branch stays at the live origin and is validated again on activation.
                previous = episode.continuation.rhs._public_control
                decision["activation"] = episode.activate_selected_control(branch, request, actuator)
                current = episode.continuation.rhs._public_control
                decision["actuator_called"] = True
                decision["public_control_changed"] = ((previous.interval_values, previous.dose_events)
                                                      != (current.interval_values, current.dose_events))
            update = episode.step(provider)
            report["arrivals"].append(update["adaptation"])
        else:
            report["status"] = "component_complete"
    except (ValueError, RuntimeError, FloatingPointError, OSError) as error:
        episode.status = "failed"
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
    report.update(observations_seen=episode.observations_seen, optimizer_updates=episode.optimizer_updates,
                  actual_model_rhs_tuple_action_work=episode.continuation.rhs._work_ledger.spent,
                  final_query_index=episode.continuation.rhs.query_index,
                  decision_count=len(report["decisions"]),
                  actuator_calls=sum(bool(row.get("actuator_called")) for row in report["decisions"]),
                  control_change_count=sum(bool(row.get("public_control_changed")) for row in report["decisions"]))
    return report


def run_native_temporal_residual_episode(episode, loss_function, provider, actuator, stop_executor, config,
                                         *, initial_selection=None, initial_node=None, collection_validate=None):
    """Execute depth-two queries and absorbing STOP on consecutive original clocks."""
    policy = temporal_query_policy(config)
    state = episode.continuation.rhs
    if state._public_control is None or state.query_index != 0:
        raise ValueError("Temporal control starts at a bound pristine original t0 program")
    report = {"status": "started", "family_uid": state._guide.family, "policy": policy,
        "initial_regimen_uid": state._public_control.regimen_uid,
        "initial_execution": "prespecified_common_context_not_model_optimized_t0" if initial_selection is None
            else initial_selection["initial_execution"],
        "initial_selection": initial_selection,
        "decisions": [], "arrivals": [], "scientific_twin": "NOT_RUN",
        "formal_qualification": False, "full_residual_mpc_eligible": False,
        "scope": "temporal_query_control_component",
        "conditional_law": "one_deterministic_path_per_tuple_no_residual_likelihood_qualification",
        "maximum_measurement_units": len(state._times)-1, "measurement_units_spent": 0,
        "physical_latency_hours": 0., "measurement_wall_latency_seconds": 0.,
        "planner_anticipates_residual_retraining": False}
    if state._guide.family == MOLECULAR and not molecular_administration_bound(state._public_control,config):
        return dict(report, status="blocked", reason="molecular_delivered_dose_mapping_unbound")
    fallback = None
    if (initial_selection is None) != (initial_node is None):
        raise ValueError("Initial public selection requires its admitted complete forest")
    try:
        while True:
            if collection_validate is not None:
                collection_validate()
            state = episode.continuation.rhs
            ledger = state._work_ledger
            if initial_node is not None:
                if (state.query_index != 0 or initial_selection["status"] != "selected"
                        or initial_node.public != state._public_control
                        or initial_node.losses is None
                        or initial_selection["actual_rhs_work_after_planning"] != ledger.spent):
                    raise ValueError("Initial selected forest differs from the current pristine public carry")
                node, initial_node = initial_node, None
                force_stop, stop_reason = False, "no_positive_admissible_query_value"
                fresh = None
            else:
                fresh = compile_temporal_world_plans(episode, config, depth=policy["lookahead_depth"])
            capacity = ledger.maximum-ledger.spent
            if fresh is not None and fresh.required_rhs_work > capacity:
                if fallback is None:
                    report.update(status="blocked", reason="complete_initial_temporal_comparison_exceeds_rhs_budget",
                        required_planning_rhs_work=fresh.required_rhs_work, remaining_rhs_work=capacity)
                    break
                # This family was fully admitted before the last actual query,
                # and refreshed after H3 when necessary. The uncomputed extended
                # family is never treated as an optimized STOP comparison.
                node = fallback
                stop_reason = "remaining_admitted_family_no_budget_for_complete_receding_horizon"
                force_stop = True
            elif fresh is not None:
                node = evaluate_temporal_world_plans(episode, fresh, loss_function)
                stop_reason = "no_positive_admissible_query_value"
                force_stop = False
            reserve = [query_execution_reserve(episode, edge) if edge.child is not None else None
                       for edge in node.edges]
            admissible = [not force_stop and item is not None
                          and item["total_rhs_work"] <= ledger.maximum-ledger.spent for item in reserve]
            logs = episode.model.solver.current_log_weights(state)
            adaptive = episode if episode.adaptive_reference is not None else None
            joint_initial = initial_selection is not None and initial_selection.get("initial_joint_future_query_value_optimized") is True
            if joint_initial and state.query_index == 0:
                if (initial_selection["temporal_policy"] != policy
                        or initial_selection["current_particle_log_weights"] != logs.tolist()
                        or initial_selection.get("adaptive_initial_reference") !=
                            (None if adaptive is None else actuation_payload(episode.adaptive_reference))):
                    raise ValueError("Initial joint selection changed its posterior, law or policy")
            decision = plan_temporal_query(node, logs, episode.readout, policy, admissible=admissible,
                adaptive_episode=adaptive,
                rhs_budget=ledger.maximum-ledger.spent if joint_initial else None,
                query_reserve=(lambda edge, preceding: query_execution_reserve(
                    episode, edge, preceding_edges=preceding)) if joint_initial else None)
            if joint_initial and state.query_index == 0:
                original_decision = initial_selection["candidates"][
                    initial_selection["selected_initial_original_library_index"]]["joint_decision"]
                if decision != original_decision:
                    raise ValueError("Installed initial joint decision differs from its all-origin comparison")
            if adaptive is not None:
                report.update(conditional_law=episode.adaptive_reference.law,
                              adaptive_scientific_qualification="UNQUALIFIED", live_policy_uses_adaptive_law=True)
            decision.update(query_index=state.query_index, query_reserves=reserve,
                query_admissible=admissible, model_rhs_work_after_planning=ledger.spent,
                comparison_scope="remaining_previously_admitted_family" if force_stop else "complete_depth_two_family",
                terminal_loss_reference_id=loss_function.reference_id)
            report["decisions"].append(decision)
            if decision["decision"] == "STOP":
                plan = compile_actuation_plan(state._public_control, executed_prefix(state._public_control, state.query_index),
                    node.plans[decision["stop_plan_index"]], state._execution_library)
                decision.update(stop_reason=stop_reason, committed_plan=actuation_payload(plan))
                if collection_validate is not None:
                    collection_validate()
                    report["selected_stop_record"] = {
                        "predicted_risk": decision["stop_risk"], "committed_plan": actuation_payload(plan),
                        "query_index": state.query_index, "reference_id": loss_function.reference_id,
                        "maximum_supported_loss": 1.+config["native_public_cns_loss"]["intervention_weight"]}
                    stop_id = digest(report["selected_stop_record"])
                stop_executor(plan)
                if collection_validate is not None:
                    collection_validate()
                    if digest(report["selected_stop_record"]) != stop_id:
                        raise ValueError("Collection STOP record changed during terminal execution")
                report.update(status="component_complete", stop_reason=stop_reason)
                break
            edge = node.edges[decision["selected_edge_index"]]
            issued = reserve[decision["selected_edge_index"]]
            report["measurement_units_spent"] += policy["measurement_cost"]
            report["physical_latency_hours"] += node.latency_hours
            decision.update(incurred_measurement_units=policy["measurement_cost"],
                            incurred_physical_latency_hours=node.latency_hours)
            before_work, before_updates = ledger.spent, episode.optimizer_updates
            if edge.request is not None:
                decision["activation"] = episode.activate_selected_control(edge.branch, edge.request, actuator)
                decision["actuator_called"] = True
            measurement_started = None
            def acquire(request):
                nonlocal measurement_started
                measurement_started = time.monotonic()
                try:
                    return provider(request)
                finally:
                    elapsed = time.monotonic()-measurement_started
                    decision["measurement_wall_latency_seconds"] = elapsed
                    report["measurement_wall_latency_seconds"] += elapsed
            result = episode.step(acquire, selected_coordinate=decision["selected_coordinate"])
            report["arrivals"].append(result["adaptation"])
            if ledger.spent-before_work != issued["actual_advance_rhs_work"]+issued["actual_H3_rhs_work"]:
                raise RuntimeError("Actual query/H3 work differs from the reserved recorded transitions")
            if episode.optimizer_updates != before_updates:
                fallback = compile_temporal_world_plans(episode, config, depth=edge.child.depth)
                if fallback.required_rhs_work != issued["post_H3_complete_terminal_family_rhs_work"]:
                    raise ValueError("Post-H3 terminal family changed its reserved compiled work")
                evaluate_temporal_world_plans(episode, fallback, loss_function)
                decision["terminal_family_reused"] = False
            else:
                verify_arrival_forecast(episode, edge.arrival)
                fallback = edge.child
                decision["terminal_family_reused"] = True
            decision["model_rhs_work_after_actual_query_and_refresh"] = ledger.spent
    except (ValueError, RuntimeError, FloatingPointError, OSError) as error:
        episode.status = "failed"
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
    report.update(observations_seen=episode.observations_seen, optimizer_updates=episode.optimizer_updates,
        actual_model_rhs_tuple_action_work=episode.continuation.rhs._work_ledger.spent,
        final_query_index=episode.continuation.rhs.query_index, decision_count=len(report["decisions"]),
        actuator_calls=sum(bool(row.get("actuator_called")) for row in report["decisions"]),
        adaptation_trace=copy.deepcopy(episode.trace),
        total_measurement_loss_penalty=report["measurement_units_spent"]*policy["cost_penalty"],
        total_physical_latency_loss_penalty=report["physical_latency_hours"]*policy["latency_penalty_per_hour"])
    return report


def record_actual_adaptive_readout_history(row: dict, episode, report: dict) -> None:
    """Persist the accepted episode prefix; transport errors retain prior failures."""
    if report["readout_mode"] == "sensor_only":
        row["actual_adaptive_readout_history"] = {"status": "not_applicable", "reason": "sensor_only",
            "scientific_readiness": "NOT_ADMITTED", "calibration_eligible": False}
        return
    if episode is None:
        row["actual_adaptive_readout_history"] = {"status": "unavailable", "reason": "episode_not_initialized",
            "scientific_readiness": "NOT_ADMITTED", "calibration_eligible": False}
        return
    try:
        context = {"group_uid": row["group_uid"], "family_uid": row["family_uid"],
            "binding": report["binding"], "reference_id": report["reference_id"],
            "pre_execution_policy_snapshot": report["pre_execution_policy_snapshot"]}
        payload = episode.actual_adaptive_readout_history(context, episode_status=row["status"])
        if payload.get("status") == "not_applicable":
            raise ValueError("Adaptive report has no bound actual adaptive episode reference")
        row["actual_adaptive_readout_history"] = payload
    except (ValueError, TypeError, KeyError, RuntimeError, FloatingPointError) as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        row["status_before_history_export"] = row["status"]
        row["history_export_failure"] = failure
        row["actual_adaptive_readout_history"] = {"status": "error", "failure": failure,
            "scientific_readiness": "NOT_ADMITTED", "calibration_eligible": False}
        row["status"] = "failed"


def realized_native_residual_control(plant, episode_status, loss_function, config):
    """Score actual complete source history only after control execution has ended."""
    snapshot = plant._evaluation_history()
    public = snapshot.pop("final_public_program")
    fields, mask, grid = (snapshot.pop(name) for name in ("private_fields", "support", "grid"))
    snapshot["consumed_dose_events"] = [actuation_payload(event) for event in snapshot["consumed_dose_events"]]
    snapshot.update(realized_full_grid_public_loss=None, realized_loss_status="incomplete_episode",
                    actual_composed_final_burden=None)
    if episode_status != "component_complete" or not snapshot["source_horizon_complete"]:
        return snapshot
    burden = composed_action_burden(public, config)
    snapshot["actual_composed_final_burden"] = burden
    if burden["status"] != "defined":
        snapshot["realized_loss_status"] = burden["status"]
        return snapshot
    values = torch.from_numpy(fields)
    result = loss_function(values, torch.from_numpy(mask), grid, values.new_tensor(burden["normalized_burden"]))
    snapshot.update(realized_full_grid_public_loss=float(result["loss"]), realized_loss_status="complete_original_full_grid",
                    realized_endpoint_loss=result["endpoint_loss"].cpu().tolist(),
                    realized_clipped_fraction=result["clipped_fraction"].cpu().tolist(),
                    reference_id=result["reference_id"])
    return snapshot


class NativeRetainedSourceWorlds:
    """Read original source truth only inside the evaluator; never replay generation."""
    def __init__(self, config, admission, *, collection_reservation=None, collection_phase=None):
        if admission.get("scope") != "admitted_controlled_source":
            raise ValueError("Retained private worlds require original complete source admissions")
        if collection_reservation is not None:
            residual_collection_policy(config, collection_reservation, collection_phase)
        self.collection_reservation = collection_reservation
        self.config = config
        self._parameter_registries = load_family_registries(config)
        self.root = Path(admission["base_admission"]).resolve().parent
        receipt = json.loads((self.root/"source_receipt.json").read_text())
        self.identity = receipt["source_identity"]
        if (receipt.get("source_status") != "complete" or self.identity != admission["source_identity"]
                or self.identity["definition_sha256"] != digest({"profile": config["native_observable_bridge"],
                                                               "background": config["controlled_cns_source"]})):
            raise ValueError("Private source receipt differs from the admitted original definition")
        self.groups = {group["sample_uid"]: (index, group) for index, group in enumerate(receipt["groups"])}
        if len(self.groups) != len(receipt["groups"]):
            raise ValueError("Private source receipt repeats a declared sample")

    def __call__(self, declaration, library, identity, initial_index, measurement_seed):
        scoped = scoped_group_config(self.config, declaration)
        if scoped["native_observable_bridge"].get("active_source_law", "original_discrete_source") != "original_discrete_source":
            raise ValueError("Private continuation supports only the original primary source law")
        if self.collection_reservation is None:
            if declaration["role"] not in {"development", "validation"}:
                raise ValueError("Private control evaluation requires its declared evaluation role")
        elif (declaration["role"] != "calibration"
                or declaration.get("calibration_reservation") != self.collection_reservation):
            raise ValueError("Private collection requires its exact declared calibration reservation")
        index, group = self.groups[declaration["sample_uid"]]
        keys = ("source_uid", "source_snapshot_ref", "family_uid", "role", "sample_uid", "support_uid",
                "posterior_uid", "particle_uids", "group_uid", "source_law")
        if any(group.get(key) != declaration.get(key) for key in keys):
            raise ValueError("Private source group differs from the declared original evaluation identity")
        if library != native_actuation_library(self.config, group):
            raise ValueError("Private source and controller do not share the original public library")
        group_receipt = json.loads((self.root/f"group_{index}_receipt.json").read_text())
        if (group_receipt.get("source_identity") != self.identity or group_receipt.get("group_index") != index
                or group_receipt.get("group") != group
                or not group_receipt.get("operations")
                or any(item.get("status") != "complete" for item in group_receipt["operations"].values())):
            raise ValueError("Private group operations or original source identity are incomplete")
        reference = group.get("conditioning_reference")
        if not reference:
            raise ValueError("Private continuation requires the retained original complete-group reference")
        if Path(reference["path"]).resolve() != self.root/f"group_{index}_conditioning.json.gz":
            raise ValueError("Private initialization reference leaves its original source group")
        raw, _ = read_joint_source_reference(reference)
        support = JointParticleSupport.from_payload(raw["support"], "cpu")
        if (raw["source_posterior_uid"] != group["posterior_uid"] or support.support_uid != group["support_uid"]
                or support.source_uid != group["source_uid"] or support.source_snapshot_ref != group["source_snapshot_ref"]
                or support.role != group["role"] or support.source_family_uid != group["family_uid"]
                or support.sample_uids != (group["sample_uid"],)
                or support.particle_uids != (tuple(group["particle_uids"]),) or support.particle_count != 32):
            raise ValueError("Original private initialization support differs from the source group")
        producer = json.loads((self.root/f"group_{index}_producer.json").read_text())
        hidden = producer["hidden_index"]
        if type(hidden) is not int or not 0 <= hidden < 32 or len(producer["parameters"]) != 32:
            raise ValueError("Original producer truth selection lacks its complete source support")
        named = load_joint_sources(self._parameter_registries[group["family_uid"]], [group["posterior_uid"]])[group["posterior_uid"]]
        named_support = copy.deepcopy(named["support"])
        parameter_binding = candidate_from_payload(named_support.pop("physical_parameter_hypotheses"))
        original_support = copy.deepcopy(raw["support"])
        original_support.pop("physical_parameter_hypotheses", None)
        if named_support != original_support:
            raise ValueError("Admitted named conditioning changed the original private source support")
        validate_candidate_parameters(parameter_binding, support)
        names = [item["name"] for item in parameter_binding["declarations"]]
        expected = [dict(zip(names, row)) for row in parameter_binding["values"][0].tolist()]
        if (parameter_binding["producer_commit"] != self.identity["producer_commit"]
                or parameter_binding["source_record_refs"] != [str(self.root/f"group_{index}_producer.json")]
                or producer["parameters"] != expected
                or any(type(value) not in (int, float) for row in producer["parameters"] for value in row.values())):
            raise ValueError("Private producer parameters differ from admitted named source conditioning")
        # The original writer did not bind hidden_index to an independent content
        # digest. Its retained group filename/receipt remains its index provenance;
        # do not invent a historical checksum or infer a replacement index.
        # This is the writer's actual generating index into its original source
        # support. No posterior, inferred parameters or seed replay enter here.
        plant = NativePhysicalPlant(self.config, group,
            private_initial_field=support.initial_field[0, hidden].cpu().numpy(),
            private_factors={key: value[0, hidden].cpu().numpy() for key, value in support.factors.items()},
            private_physical_parameters=None if is_long_cns(self.config, group["family_uid"]) else producer["parameters"][hidden],
            initial_actuation=library[initial_index], observation_identity=identity,
            measurement_generator=torch.Generator().manual_seed(measurement_seed))
        if (is_long_cns(self.config, group["family_uid"])
                and flatten_cns_pk_parameters(plant._law.pk) != producer["parameters"][hidden]):
            raise ValueError("Original private CNS parameters differ from the retained source factors/definition")
        return plant


def evaluate_native_residual_controls(config, model, belief, binding, dataset, declarations, reference, output,
                                      *, initial_index=None, optimize_initial_plan=False,
                                      optimize_initial_joint_plan=False, residual_path=None, world_factory=None,
                                      adaptive_readout=False, collection_context=None, adaptive_artifact=None,
                                      check_artifact=None, selected_loss_calibration=None):
    """Existing baseline consumer, with original admissions and a private-world callback boundary."""
    if check_artifact is not None and adaptive_artifact is None:
        raise ValueError("Independent adaptive check requires the adaptive fit artifact")
    if selected_loss_calibration is not None and (adaptive_artifact is None or check_artifact is None):
        raise ValueError("Selected-loss bounds require adaptive fit and independent check artifacts")
    if adaptive_artifact is not None:
        phase = collection_context.get("phase") if collection_context is not None else None
        if collection_context is None and check_artifact is None:
            raise ValueError("Fitted observable evaluation requires the independent final-policy check artifact")
        if collection_context is not None and phase not in {"final_policy_check","final_selected_loss"}:
            raise ValueError("Fitted-law collection requires an explicit final-policy phase")
        if phase == "final_policy_check" and check_artifact is not None:
            raise ValueError("Independent check collection cannot consume a prior check artifact")
        if phase == "final_selected_loss" and check_artifact is None:
            raise ValueError("Threshold collection requires the independent final-policy check artifact")
    if selected_loss_calibration is not None and collection_context is not None:
        raise ValueError("Selected-loss bounds apply only to evaluation after calibration")
    residual_control_policy(config)
    temporal_query_policy(config)
    readout_mode = residual_readout_mode(config, adaptive_readout=adaptive_readout)
    if adaptive_artifact is not None and readout_mode != "original_action_adaptive_unqualified":
        raise ValueError("Fitted adaptive artifact requires the adaptive readout mode")
    if readout_mode == "original_action_adaptive_unqualified" and residual_path is None:
        raise ValueError("Adaptive readout requires a validated offline residual artifact; supply the offline starting artifact before adaptive execution")
    started = time.monotonic()
    resource_kind = "cpu" if next(model.parameters()).device.type == "cpu" else "evaluation"
    deadline = started+config["manuscript_native_protocol"]["resources"][resource_kind]["wall_seconds"]
    initial_mode = residual_initial_mode(config, initial_index=initial_index,
                                         optimize_initial_plan=optimize_initial_plan,
                                         optimize_initial_joint_plan=optimize_initial_joint_plan)
    if getattr(model, "prediction_kind", None) != "offline_one_step_mechanistic_residual_conditional_mean":
        raise ValueError("Actual residual control requires the selected mechanistic_residual predictor")
    if collection_context is not None:
        policy = residual_collection_policy(config, collection_context["reservation"], collection_context.get("phase"))
        if (collection_context["purpose"] != policy["purpose"]
                or readout_mode != "original_action_adaptive_unqualified"
                or set(dataset.group_uids) != set(declarations)
                or any(row["role"] != "calibration" or row.get("calibration_reservation") != collection_context["reservation"]
                       for row in declarations.values())):
            raise ValueError("Collection purpose, adaptive law or calibration dataset changed")
    path = Path(output)/("native_residual_calibration_collection.json" if collection_context is not None
                        else "native_residual_control_evaluation.json")
    if path.exists():
        raise FileExistsError("Native residual control evaluation already exists")
    admission = predose_source_admission(config, dataset)
    residual = load_native_comparator_residual(residual_path, config, binding) if residual_path is not None else None
    if adaptive_artifact is not None:
        validate_adaptive_fit_artifact(adaptive_artifact, config, binding=binding, residual=residual)
        if collection_context is None and check_artifact is None:
            raise ValueError("Fitted observable evaluation requires the independent final-policy check artifact")
        if readout_mode != "original_action_adaptive_unqualified":
            raise ValueError("Fitted adaptive artifact requires the adaptive readout mode")
    if admission["scope"] != "artificial_software_fixture" and residual is None:
        raise ValueError("Scientific hybrid control requires independently admitted residual calibration; uncalibrated execution is artificial only")
    loss_function = NativePublicCNSLoss(reference, config)
    registries = load_family_registries(config)
    measurement_seed = config["native_predose_evaluator"]["measurement_seed"]
    if type(measurement_seed) is not int:
        raise ValueError("Evaluator measurement seed must be explicit and private")
    report = {"schema_version": "native_residual_control_evaluation.v1", "status": "started",
              "method_id": "residual_mpc", "predictor_identity": "mechanistic_residual", "binding": binding,
              "initial_original_library_index": initial_index, "initial_selection_mode": initial_mode,
              "source_admission": admission,
              "reference_id": reference["reference_id"], "groups": [], "declared_group_count": len(declarations),
              "formal_qualification": False, "full_residual_mpc_eligible": False, "scientific_twin": "NOT_RUN",
              "conditional_law": "one_deterministic_path_per_tuple_no_residual_likelihood_qualification",
              "independent_online_predictive_error_calibration": False,
              "readout_mode": readout_mode, "adaptive_scientific_qualification": "UNQUALIFIED",
              "independent_adaptive_fit": False, "independent_adaptive_check": False}
    if residual is not None:
        report["offline_residual_artifact"] = {"path": str(residual_path), "law": residual["law"],
            "use": "existing_offline_admission_only_not_online_predictive_error_calibration"}
    if readout_mode == "original_action_adaptive_unqualified":
        report.update(conditional_law="native_original_action_adaptive_readout_brownian.v1",
            scale_authority="inherited_fixed_action_starting_model_not_adaptive_calibration",
            pre_execution_policy_snapshot={key: copy.deepcopy(config[key]) for key in
                ("native_residual_control", "native_temporal_query", "native_mechanistic_residual_online",
                 "native_arrived_observation_filter")})
    frozen = None
    if adaptive_artifact is not None:
        initial_selection_binding = {"mode": initial_mode, "initial_index": initial_index,
            "optimize_initial_plan": optimize_initial_plan, "optimize_initial_joint_plan": optimize_initial_joint_plan}
        policy_id = final_residual_policy_id(config, binding, reference, residual, adaptive_artifact, initial_selection_binding)
        if check_artifact is not None:
            validate_adaptive_check_artifact(check_artifact, adaptive_artifact, policy_id, config)
        report.update(adaptive_artifact_id=adaptive_artifact["artifact_id"], final_policy_id=policy_id,
            observable_fit_valid=True, observable_check_complete=check_artifact is not None,
            scale_authority="authenticated_observable_adaptive_fit",
            inherited_species_indices=adaptive_artifact["inherited_species_indices"])
    if selected_loss_calibration is not None:
        validate_selected_loss_artifact(selected_loss_calibration, adaptive_artifact, check_artifact, policy_id, config)
        report["selected_loss_calibration_id"] = selected_loss_calibration["artifact_id"]
    if collection_context is not None:
        collection_context = dict(collection_context, initial_selection={
            "mode": initial_mode, "initial_index": initial_index,
            "optimize_initial_plan": optimize_initial_plan,
            "optimize_initial_joint_plan": optimize_initial_joint_plan})
        if admission != collection_context["source_admission"]:
            raise ValueError("Collection original source admission changed")
        artificial = admission["scope"] == "artificial_software_fixture"
        if (collection_context["denominator"]["software_fixture_only"] != artificial
                or collection_context["selected_predictor"] != _collection_selected_predictor(binding, artificial=artificial, config=config, source_admission=admission)):
            raise ValueError("Collection fixture/selected-predictor authority differs from original admission")
        if adaptive_artifact is not None:
            collection_context["adaptive_artifact_id"] = adaptive_artifact["artifact_id"]
            collection_context["final_policy_id"] = final_residual_policy_id(config, binding, reference, residual,
                adaptive_artifact, collection_context["initial_selection"])
        if check_artifact is not None:
            validate_adaptive_check_artifact(check_artifact, adaptive_artifact,
                                            collection_context.get("final_policy_id"), config)
        frozen = _collection_binding(config, binding, reference, residual, collection_context)
        report.update(schema_version="native_residual_calibration_collection.v1",
            purpose=collection_context["purpose"], reservation=collection_context["reservation"],
            collection_binding=frozen, collection_binding_id=digest(frozen),
            denominator=collection_context["denominator"], calibration_eligible=False,
            selected_decision_calibration_complete=False, statistical_independence_claimed=False)
        original_versions = tuple(value._version for value in (*model.parameters(), *model.buffers()))
        private_measurement_seed = measurement_seed
        def validate_global():
            if (config["native_predose_evaluator"]["measurement_seed"] != private_measurement_seed
                    or _collection_binding(config, binding, reference, residual, collection_context) != frozen
                    or tuple(value._version for value in (*model.parameters(), *model.buffers())) != original_versions
                    or belief.field_normalization.identity != binding["normalization_id"]
                    or any(model.config[key] != frozen["policy"][key] for key in (
                        "native_residual_control", "native_temporal_query",
                        "native_mechanistic_residual_online", "native_arrived_observation_filter"))):
                raise ValueError("Frozen collection policy, predictor or reference changed before private access")
    for index, uid in enumerate(dataset.group_uids):
        declaration = declarations[uid]
        row = {"group_uid": uid, "family_uid": declaration["family_uid"], "status": "started"}
        report["groups"].append(row)
        plant, episode = None, None
        try:
            if declaration["family_uid"] == MOLECULAR and molecular_administration_policy(config) is None:
                row.update(status="blocked", reason="molecular_delivered_dose_mapping_unbound")
                continue
            scoped = scoped_group_config(config, declaration)
            if scoped["native_observable_bridge"].get("active_source_law", "original_discrete_source") != "original_discrete_source":
                row.update(status="blocked", reason="private_continuation_for_nonprimary_source_law_unimplemented")
                continue
            inputs = load_native_prediction_inputs(config, declaration, belief, registries)
            library = native_actuation_library(config, declaration)
            if initial_mode == "explicit_fixed_index" and initial_index >= len(library):
                raise ValueError("Prespecified initial index is outside the original action library")
            readout = native_cns_readout(config, declaration)
            adaptive_reference = None
            if readout_mode == "original_action_adaptive_unqualified":
                if any(model.config[key] != report["pre_execution_policy_snapshot"][key]
                       or config[key] != report["pre_execution_policy_snapshot"][key] for key in report["pre_execution_policy_snapshot"]):
                    raise ValueError("Adaptive control policy differs from its pre-execution snapshot")
                adaptive_reference = adaptive_control_reference(inputs, belief.field_normalization,
                    readout, library, initial_index if initial_mode == "explicit_fixed_index" else 0,
                    residual, residual_path, binding, adaptive_artifact=adaptive_artifact)
                row["adaptive_readout_reference"] = actuation_payload(adaptive_reference)
            episode = NativeResidualOnlineEpisode(model, inputs, belief.field_normalization,
                readout, executed_regimen_uid=None
                    if initial_mode in {"public_open_loop_plan", "public_joint_query_control"} else library[initial_index].regimen_uid,
                adaptive_reference=adaptive_reference)
            episode.model._execution_deadline = deadline
            initial_selection, initial_node = None, None
            selected_index = initial_index
            if initial_mode in {"public_open_loop_plan", "public_joint_query_control"}:
                initial_selection, initial_node = select_initial_public_plan(episode, library, loss_function, config,
                    **({"selection_mode": initial_mode} if initial_mode == "public_joint_query_control" else {}))
                row["initial_selection"] = initial_selection
                if initial_selection["status"] != "selected":
                    row.update(status="blocked", reason=initial_selection["reason"])
                    continue
                selected_index = initial_selection["selected_initial_original_library_index"]
                if adaptive_reference is not None:
                    row["adaptive_readout_reference"] = actuation_payload(episode.adaptive_reference)
            identity = NativeObservationIdentity.from_context(episode.model.solver.filter_context(episode.continuation.rhs))
            # The optimized public t0 decision precedes the first private world,
            # outcome and measurement RNG access in this callback.
            if frozen is not None:
                validate_global()
                episode_binding = _collection_episode_binding(episode, library, declaration, frozen)
                row["episode_collection_binding"] = copy.deepcopy(episode_binding)
                def validate_episode():
                    validate_global()
                    if episode.continuation.rhs._public_control is not None:
                        episode._adaptive_history()
                    if _collection_episode_binding(episode, library, declaration, frozen) != episode_binding:
                        raise ValueError("Frozen collection prior, support, readout or action library changed")
            if world_factory is None:
                world_factory = (NativeRetainedSourceWorlds(config, admission,
                    collection_reservation=collection_context["reservation"], collection_phase=collection_context.get("phase")) if frozen is not None
                    else NativeRetainedSourceWorlds(config, admission))
            plant = world_factory(declaration, library, identity, selected_index, measurement_seed+index)
            if frozen is not None:
                validate_episode()
            if initial_mode == "explicit_fixed_index":
                episode.bind_execution(library, library[selected_index], plant.observation_identity)
            elif plant.observation_identity != identity:
                raise ValueError("Selected initial private plant changed the public observation identity")
            def private_call(callback, *args, **kwargs):
                if frozen is not None:
                    validate_episode()
                return callback(*args, **kwargs)
            result = run_native_temporal_residual_episode(episode, loss_function,
                lambda request: private_call(plant.acquire, request, deadline=deadline),
                lambda request: private_call(plant.switch, request),
                lambda plan: private_call(plant._complete_planned_tail, plan, deadline=deadline), config,
                initial_selection=initial_selection, initial_node=initial_node,
                **({"collection_validate": validate_episode} if frozen is not None else {}))
            row.update(status=result["status"], episode=result)
        except (ValueError, KeyError, RuntimeError, FloatingPointError, OSError) as error:
            row.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        finally:
            record_actual_adaptive_readout_history(row, episode, report)
            if plant is not None:
                try:
                    if frozen is not None:
                        validate_episode()
                    row["realized"] = realized_native_residual_control(plant, row["status"], loss_function, config)
                    if frozen is not None:
                        _pair_collection_stop(row, frozen)
                    if selected_loss_calibration is not None and row["status"] == "component_complete":
                        apply_selected_loss_bounds(row, selected_loss_calibration)
                except (ValueError, RuntimeError, FloatingPointError) as error:
                    row.update(status="failed", realized_failure={"type": type(error).__name__, "message": str(error)})
            write_json(path, report)
    completed = sum(row["status"] == "component_complete" for row in report["groups"])
    observed_ids = [row["group_uid"] for row in report["groups"]]
    report.update(evaluation_wall_seconds=time.monotonic()-started, completed_group_count=completed,
                  status="component_complete" if completed == len(declarations)
                  and len(observed_ids) == len(set(observed_ids)) and set(observed_ids) == set(declarations) else "incomplete")
    if frozen is not None:
        report["status"] = ("software_collection_complete" if collection_context["denominator"]["software_fixture_only"]
                            else "collection_complete") if report["status"] == "component_complete" else "incomplete"
        report["production_collection_complete"] = report["status"] == "collection_complete"
    write_json(path, report)
    return report
