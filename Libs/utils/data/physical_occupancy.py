"""Source-bound single-ligand equilibrium programs; no PK solver or calibration."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

from Libs.utils.data.local_membrane import READOUT_METADATA, membrane_particle_basis
from Libs.utils.data.action_observation_contracts import compile_pharmacological_action

PHYSICAL_MODE = "osp_source_bound_equilibrium.v1"
CONTROLLED_PHYSICAL_MODE = "controlled_source_bound_equilibrium.v1"
MOLAR_UNITS = {"mol/L": 1., "mmol/L": 1e-3, "umol/L": 1e-6, "µmol/L": 1e-6,
               "nmol/L": 1e-9, "pmol/L": 1e-12, "M": 1., "mM": 1e-3, "uM": 1e-6, "nM": 1e-9}
MASS_UNITS = {"g/L": 1., "mg/L": 1e-3, "ug/L": 1e-6, "µg/L": 1e-6,
              "ng/L": 1e-9, "mg/mL": 1., "ug/mL": 1e-3, "ng/mL": 1e-6}
TIME_UNITS = {"h": 1., "hour": 1., "hours": 1., "min": 1/60, "s": 1/3600, "day": 24.}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("physical occupancy: " + message)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_value(snapshot: dict, path: object) -> object:
    require(isinstance(path, list) and bool(path), "explicit source setting path required")
    result = snapshot
    try:
        for key in path:
            result = result[key]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("physical occupancy: missing source setting path " + repr(path)) from error
    return result


def finite_values(value: object, label: str, positive: bool = False) -> np.ndarray:
    require(not isinstance(value, (bool, str)), label + " must be numeric")
    raw = np.asarray(value)
    require(raw.dtype.kind in "ifu", label + " must contain numeric values")
    array = raw.astype(np.float64)
    require(np.isfinite(array).all() and (array > 0 if positive else array >= 0).all(),
            label + " must be finite and " + ("positive" if positive else "nonnegative"))
    return array


def molar(value: object, unit: str, molecular_weight: dict | None = None) -> np.ndarray:
    values = finite_values(value, "concentration/affinity")
    if unit in MOLAR_UNITS:
        result = values * MOLAR_UNITS[unit]
    elif unit in MASS_UNITS:
        require(isinstance(molecular_weight, dict) and molecular_weight.get("unit") == "g/mol"
                and bool(molecular_weight.get("source_ref")), "mass concentration requires source molecular weight in g/mol")
        mw = finite_values(molecular_weight["value"], "molecular weight", positive=True)
        require(mw.ndim == 0, "molecular weight must be scalar")
        result = values * MASS_UNITS[unit] / mw
    else:
        raise ValueError("physical occupancy: unsupported concentration unit " + str(unit))
    require(np.isfinite(result).all(), "unit conversion overflow")
    return result


def equilibrium_fraction(free_concentration: object, kd: object) -> np.ndarray:
    free = finite_values(free_concentration, "free concentration")
    affinity = finite_values(kd, "Kd", positive=True)
    # Scale by the larger operand so that finite C and Kd cannot overflow C+Kd.
    scale = np.maximum(free, affinity)
    return (free / scale) / (free / scale + affinity / scale)


def whole_particle_basis(support: dict, bi: int, ji: int) -> dict:
    """Source particle tuple, excluding its replaceable posterior/prior mass."""
    result = {name: support[name][bi][ji] for name in ("initial_field", "physical_parameters", "regime_ids")} | {
        "factors": {name: value[bi][ji] for name, value in support["factors"].items()},
        "source_nuisance": {name: value[bi][ji] for name, value in support["source_nuisance"].items()},
        "readout_parameters": {name: {key: value if key in READOUT_METADATA else value[bi][ji]
            for key, value in fields.items()} for name, fields in support["readout_parameters"].items()}}
    if support.get("local_membrane") is not None:
        result["local_membrane"] = membrane_particle_basis(support["local_membrane"], bi, ji)
    return result


def validate_bindings(inputs: dict) -> None:
    snapshot, binding, action = inputs["run_snapshot"], inputs["binding"], inputs["action"]
    for key in ("source_model_id", "source_run_id", "source_snapshot_ref", "model_sha256", "export_sha256"):
        require(bool(snapshot.get(key)), "missing source run " + key)
    require(snapshot.get("settings_origin") in {"osp_run_parameter_readback", "labeled_source_format_fixture", "controlled_source_parameter_definition"},
            "source settings must be recorded run readbacks")
    for key in ("source_uid", "source_snapshot_ref", "source_posterior_uid", "support_uid",
                "sample_uid", "particle_uid", "regimen_uid"):
        require(snapshot.get(key) == binding.get(key) and bool(binding.get(key)), key + " source binding mismatch")
    require(snapshot.get("particle_basis") == binding.get("particle_basis")
            and bool(binding.get("particle_basis")), "whole-particle source tuple mismatch")
    require(action.get("intervention_type") == "pharmacological"
            and len(action.get("agent_set", [])) == len(action.get("target_set", [])) == 1,
            "only single-ligand single-target equilibrium actions are represented")
    require(binding.get("binding_law") == "single_ligand_equilibrium"
            and binding.get("requested_readout") == "fractional_occupancy",
            "unsupported binding law/readout; bound amount or depletion needs a separate density model")
    for key in ("dose", "dose_unit", "route", "schedule", "duration", "agent_set"):
        require(source_value(snapshot, binding["action_setting_paths"].get(key)) == action.get(key),
                "action setting does not bind " + key)
    parameters = binding.get("particle_parameters")
    require(isinstance(parameters, dict) and bool(parameters), "named whole-particle parameters required")
    require(set(parameters) == set(snapshot.get("particle_parameter_paths", {})),
            "complete source particle parameter basis required")
    for name, parameter in parameters.items():
        require(parameter.get("source_path") == snapshot["particle_parameter_paths"][name]
                and source_value(snapshot, parameter["source_path"]) == parameter.get("value"),
                "particle parameter mismatch: " + name)
    for key in ("model_sha256", "export_sha256"):
        require(inputs["provenance"].get(key) == snapshot[key], key + " provenance mismatch")
    require(inputs["provenance"].get("run_snapshot_sha256") == digest(snapshot),
            "run snapshot provenance mismatch")


def bound_affinity(inputs: dict, mapping: dict) -> dict:
    """Resolve the consumed Kd from its declared particle or fixed source basis."""
    snapshot, binding, action = inputs["run_snapshot"], inputs["binding"], inputs["action"]
    association = mapping.get("kd_binding", {})
    kind = association.get("kind")
    path = mapping["kd_source_path"]
    require(kind in {"particle_parameter", "fixed_source_affinity"}, "explicit Kd basis binding required")
    if kind == "particle_parameter":
        name = association.get("parameter_name")
        parameter = binding["particle_parameters"].get(name, {})
        require(parameter.get("source_path") == path
                and snapshot["particle_parameter_paths"].get(name) == path,
                "consumed Kd path is outside the named particle affinity basis")
    else:
        name = association.get("affinity_name")
        require(snapshot.get("fixed_affinity_paths", {}).get(name) == path,
                "consumed Kd path is outside the declared fixed affinity basis")
        require(all(parameter["source_path"] != path for parameter in binding["particle_parameters"].values()),
                "fixed affinity cannot also be a particle parameter")
    record = source_value(snapshot, path)
    require(isinstance(record, dict) and bool(record.get("source_ref")),
            "Kd source reference required")
    require(record.get("quantity_kind") == "equilibrium_dissociation_constant"
            and record.get("binding_scope") == kind, "Kd quantity/scope mismatch")
    for key, expected in {"source_uid": binding["source_uid"],
            "source_snapshot_ref": binding["source_snapshot_ref"], "site_uid": mapping["site_uid"],
            "ligand_uid": action["agent_set"][0], "target_uid": action["target_set"][0]}.items():
        require(record.get(key) == expected, "Kd " + key + " identity mismatch")
    if kind == "particle_parameter":
        require(record.get("particle_uid") == binding["particle_uid"],
                "Kd whole-particle identity mismatch")
    else:
        require("particle_uid" not in record, "fixed affinity cannot declare one particle")
        for parameter in binding["particle_parameters"].values():
            candidate = parameter["value"]
            if isinstance(candidate, dict) and candidate.get("quantity_kind") == "equilibrium_dissociation_constant":
                require(any(candidate.get(key) != record[key] for key in ("site_uid", "ligand_uid", "target_uid")),
                        "fixed affinity conflicts with a declared particle affinity")
    return record


def physical_arrays(inputs: dict) -> dict:
    """Reconstruct retained exposure/occupancy, verifying the complete source binding."""
    validate_bindings(inputs)
    regions, compartments = inputs["region_ids"], inputs["compartment_ids"]
    require(regions and compartments and len(set(regions)) == len(regions)
            and len(set(compartments)) == len(compartments), "unique declared grid required")
    times = finite_values(inputs["time_grid_hours"], "requested time")
    require(times.ndim == 1 and len(times) >= 2 and np.all(np.diff(times) > 0), "increasing requested times required")
    binding, snapshot, action = inputs["binding"], inputs["run_snapshot"], inputs["action"]
    mappings = binding.get("site_mappings", [])
    domain = binding.get("target_site_domain")
    if domain is None:
        require(len(mappings) == len(regions) * len(compartments), "missing site coverage; no anatomical broadcast")
        site_support = np.ones((len(regions), len(compartments)), dtype=bool)
    else:
        require(domain == snapshot.get("target_site_domain")
                and domain.get("semantics") == "target_program_support_not_concentration_absence"
                and domain.get("excluded_site_meaning") == "outside_declared_target_program"
                and bool(domain.get("evidence_ref")), "target-site domain requires explicit source semantics/provenance")
        site_support = np.asarray(domain.get("mask"))
        require(site_support.dtype == np.bool_ and site_support.shape == (len(regions), len(compartments))
                and site_support.any(), "target-site domain requires a nonempty boolean grid")
        expected = {(regions[ri], compartments[ci]) for ri, ci in np.argwhere(site_support)}
        actual = {(mapping["region_id"], mapping["compartment_id"]) for mapping in mappings}
        require(len(mappings) == len(expected) and actual == expected,
                "missing or extra declared target-site coverage; no anatomical broadcast")
    shape = (len(times), len(regions), len(compartments), 1)
    # Outside the declared program, zeros are placeholders, not measured absence.
    exposure, occupancy = np.zeros(shape), np.zeros(shape)
    seen, seen_paths, targets = set(), set(), []
    for mapping in mappings:
        region, compartment = mapping["region_id"], mapping["compartment_id"]
        require(region in regions and compartment in compartments and (region, compartment) not in seen,
                "unknown or duplicate site mapping")
        seen.add((region, compartment))
        path = mapping["source_output_path"]
        require(path not in seen_paths, "source quantity cannot be broadcast across anatomical sites")
        seen_paths.add(path)
        require(path in snapshot.get("quantities", {}), "missing source quantity declaration")
        quantity = snapshot["quantities"][path]
        require(quantity["site_uid"] == mapping["site_uid"]
                and quantity["ligand_uid"] == action["agent_set"][0]
                and quantity["target_uid"] == action["target_set"][0], "quantity site/ligand/target mismatch")
        rows = [row for row in inputs["source_rows"] if row["source_output_path"] == path]
        require(bool(rows), "missing source quantity " + path)
        for row in rows:
            for key in ("concentration_unit", "time_unit", "quantity_kind", "site_uid", "ligand_uid", "target_uid"):
                require(row.get(key) == quantity.get(key) and bool(row.get(key)), "source quantity metadata mismatch: " + key)
            for key in ("source_model_id", "source_run_id", "source_snapshot_ref"):
                require(row.get(key) == snapshot[key], "export identity mismatch: " + key)
        require(quantity["time_unit"] in TIME_UNITS, "unsupported time unit")
        source_times = finite_values([row["source_time"] for row in rows], "source time") * TIME_UNITS[quantity["time_unit"]]
        require(len(np.unique(source_times)) == len(source_times), "duplicate source quantity/time identity")
        order = np.argsort(source_times)
        source_times = source_times[order]
        source_concentration = molar([row["concentration"] for row in rows], quantity["concentration_unit"],
                                     quantity.get("molecular_weight"))[order]
        interpolation = binding["interpolation"]
        require(times[0] >= source_times[0] and times[-1] <= source_times[-1], "uncovered time; extrapolation rejected")
        if interpolation == "exact":
            require(all(value in source_times for value in times), "requested time absent from source")
            concentration = source_concentration[[int(np.flatnonzero(source_times == value)[0]) for value in times]]
        elif interpolation == "linear_inside_coverage":
            concentration = np.interp(times, source_times, source_concentration)
        else:
            raise ValueError("physical occupancy: undeclared interpolation convention")
        kd_record = bound_affinity(inputs, mapping)
        kd = molar(kd_record["value"], kd_record.get("unit"), quantity.get("molecular_weight"))
        require(kd.ndim == 0 and kd > 0, "finite positive scalar Kd required")
        fraction = None
        if quantity["quantity_kind"] == "total_concentration":
            fraction = source_value(snapshot, mapping.get("unbound_fraction_source_path"))
            require(fraction.get("site_uid") == mapping["site_uid"]
                    and fraction.get("particle_uid") == binding["particle_uid"]
                    and fraction.get("interpretation") == "free_over_total_same_site"
                    and bool(fraction.get("source_ref")), "site/particle-specific unbound fraction required")
            fu = finite_values(fraction["value"], "unbound fraction")
            require(fu.ndim == 0 and fu <= 1, "unbound fraction outside [0,1]")
            concentration = concentration * fu
        else:
            require(quantity["quantity_kind"] == "free_concentration", "unsupported concentration quantity")
        ri, ci = regions.index(region), compartments.index(compartment)
        exposure[:, ri, ci, 0] = concentration
        occupancy[:, ri, ci, 0] = equilibrium_fraction(concentration, kd)
        targets.append({"region_id": region, "compartment_id": compartment, "site_uid": mapping["site_uid"],
                        "target_uid": quantity["target_uid"], "ligand_uid": quantity["ligand_uid"],
                        "kd_mol_per_l": float(kd), "kd_source": kd_record, "affinity_binding": mapping["kd_binding"],
                        "unbound_fraction": fraction})
    return {"free_concentration_mol_per_l": exposure.tolist(), "target_occupancy_fraction": occupancy.tolist(),
            "target_parameters": targets, "occupancy": occupancy.tolist(),
            "support": np.broadcast_to(site_support[None, :, :, None], shape).astype(float).tolist(),
            "physical_time": np.broadcast_to(times[:, None, None, None], shape).tolist()}


def compile_physical_record(inputs: dict) -> dict:
    arrays = physical_arrays(inputs)
    binding = inputs["binding"]
    mode = (CONTROLLED_PHYSICAL_MODE if inputs["run_snapshot"]["settings_origin"] == "controlled_source_parameter_definition" else PHYSICAL_MODE)
    physical = {"mode": mode, "inputs": copy.deepcopy(inputs), **arrays}
    provenance = {"status": "verified", "artifact_ref": binding["source_snapshot_ref"],
                  "verification_scope": "source_bindings_and_numerical_mapping_only",
                  "physical_input_sha256": digest(inputs), "calibrated": False}
    context = {"time_grid": inputs["time_grid_hours"], "parameter_provenance": provenance,
               "target_program": {"targets": inputs["action"]["target_set"], "binding_law": binding["binding_law"]},
               "exposure_fields": {"array_name": "free_concentration_mol_per_l", "unit": "mol/L"},
               "occupancy_fields": {"array_name": "occupancy", "support_array": "support", "unit": "fraction"},
               "organism_context": inputs["organism_context"], "supported_organisms": [inputs["organism_context"]]}
    baseline, graph = {"state_ref": binding["source_posterior_uid"]}, {"graph_ref": inputs["graph_ref"]}
    compiled = compile_pharmacological_action(inputs["action"], pkpd_context=context,
        baseline_state_posterior=baseline, graph_context=graph)
    require(not compiled["blockers"], "canonical action compiler rejected: " + repr(compiled["blockers"]))
    return {"program_uid": compiled["program_uid"], "posterior_uid": binding["source_posterior_uid"],
            "particle_uid": binding["particle_uid"], "sample_uid": binding["sample_uid"],
            "source_uid": binding["source_uid"], "action_uid": inputs["action"]["action_uid"],
            "regimen_uid": binding["regimen_uid"], "target_program_uid": inputs["action"]["target_set"][0],
            "representation": "explicit_grid", "action": copy.deepcopy(inputs["action"]),
            "pkpd_context": context, "baseline_state_posterior": baseline, "graph_context": graph,
            "compiled_program": compiled, "physical_program": physical}


def validate_physical_record(record: dict) -> dict:
    physical = record["physical_program"]
    require(physical.get("mode") in {PHYSICAL_MODE, CONTROLLED_PHYSICAL_MODE}, "physical program mode mismatch")
    expected = compile_physical_record(physical["inputs"])
    require(expected == record, "retained physical program or compiler binding mismatch")
    return physical


# This prospective comparison definition is a protocol constant, not a fitted scaler.
SCALAR_EXPOSURE_DEFINITION = {
    "rule": "source_site_free_concentration_trapezoidal_auc.v1",
    "unit": "nmol*h/L",
    "interval": "original_declared_program",
    "reduction_axes": ["time"],
}


def validate_scalar_exposure_definition(definition: dict) -> dict:
    require(definition == SCALAR_EXPOSURE_DEFINITION, "scalar exposure definition mismatch")
    return copy.deepcopy(definition)


def source_site_exposure(record: dict) -> dict:
    """One source-validated temporal integral per site and whole particle."""
    # These are storage coordinates, not part of the physical compiler record.
    physical_record = {key: value for key, value in record.items()
                       if key not in {"store_path", "store_format", "program_index"}}
    require("physical_program" in physical_record, "scalar exposure requires genuine physical evidence")
    physical = validate_physical_record(physical_record)
    inputs = physical["inputs"]
    times = np.asarray(inputs["time_grid_hours"], dtype=np.float64)
    concentration = np.asarray(physical["free_concentration_mol_per_l"], dtype=np.float64)
    # Scale each trapezoid after reduction; neither occupancy inversion nor peak normalization.
    auc = np.sum((concentration[:-1] / 2 + concentration[1:] / 2)
                 * np.diff(times)[:, None, None, None], axis=0, dtype=np.float64) * 1e9
    require(np.isfinite(auc).all() and (auc >= 0).all(), "scalar exposure integration overflow")
    regions, compartments = inputs["region_ids"], inputs["compartment_ids"]
    sites = {(row["region_id"], row["compartment_id"]): row["site_uid"]
             for row in physical["target_parameters"]}
    result = {
        "values": auc.tolist(), "interval_hours": [float(times[0]), float(times[-1])],
        "region_ids": regions, "compartment_ids": compartments,
        "site_uids": [[sites.get((region, compartment)) for compartment in compartments] for region in regions],
        "program_uid": record["program_uid"], "source_binding": copy.deepcopy(inputs["binding"]),
        "physical_input_sha256": digest(inputs),
    }
    domain = inputs["binding"].get("target_site_domain")
    if domain is not None:
        # physical_arrays validated exact mapping coverage. Preserve its domain;
        # excluded zero placeholders are not a source concentration measurement.
        result["target_site_domain"] = copy.deepcopy(domain)
        result["site_support"] = np.asarray(physical["support"], dtype=bool)[0].tolist()
    return result
