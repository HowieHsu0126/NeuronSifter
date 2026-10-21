"""Prospective scientific source design; the frozen observable producer stays intact."""
from __future__ import annotations

import hashlib
import json

import copy
import math

import numpy as np
import torch

from Libs.models.modules.world_state import FACTOR_NAMES, JointParticleSupport
from Libs.scripts.builders.build_sample_observation_registry import produce_controlled_observations
from Libs.utils.data.native_observable_profile import family_grid, observable_profile, quantity_support
from Libs.utils.data.native_stimulation import FORCING_ARRAY_NAMES
from Libs.utils.model import controlled_cns_source as cns
from Libs.utils.model import native_held_source as held
from Libs.utils.model import native_romani_source as romani
from Libs.utils.model.controlled_cns_source import ControlledLaw, DYNAMIC
from Libs.utils.data.joint_physical_parameters import flatten_cns_pk_parameters
from Libs.utils.model.native_observable_source import (
    ELECTRICAL, SYNAPTIC, MOLECULAR, CoupledObservableLaw as OriginalCoupledObservableLaw, compiled_program as original_compiled_program, action_levels,
    estimated_exposure, integrate_coupled, numerical_summary, physical_trajectory as original_physical_trajectory,
)

def scoped_group_config(config: dict, group: dict) -> dict:
    """Apply prospective source/domain declarations before any latent/target draw."""
    result = copy.deepcopy(config)
    settings = result["native_observable_bridge"]
    variant = settings.get("source_variants", {}).get(group.get("source_variant", "primary"))
    if settings.get("source_variants") and variant is None:
        raise ValueError("Source group names an undeclared source variant")
    if variant is not None:
        settings["source_uid"], settings["snapshot_ref"] = variant["source_uid"], variant["snapshot_ref"]
        settings["active_source_law"] = variant["law"]
        result["controlled_cns_source"]["observation"]["standard_deviation"] = variant["observation_standard_deviation"]
        settings["observation_matrix"] = variant.get("observation_matrix", np.eye(len(result["controlled_cns_source"]["observation"]["selector"])).tolist())
        if variant.get("family_uids") and group["family_uid"] not in variant["family_uids"]:
            raise ValueError("Source variant is not applicable to this physical family")
    for key in ("condition","coefficient_levels","physical_parameter_jitter_multiplier"):
        if key in group:
            settings[key] = copy.deepcopy(group[key])
    observation = result["controlled_cns_source"]["observation"]
    if "observation_mask" in group:
        observation["mask"] = list(group["observation_mask"])
    if "observation_noise_multiplier" in group:
        observation["standard_deviation"] *= group["observation_noise_multiplier"]
    removed = set(group.get("removed_edge_indices", []))
    edges = result["controlled_cns_source"]["edges"]
    if any(isinstance(i,bool) or not isinstance(i,int) or not 0 <= i < len(edges) for i in removed):
        raise ValueError("Structural OOD requires declared existing symmetric edge indices")
    result["controlled_cns_source"]["edges"] = [row for index,row in enumerate(edges) if index not in removed]
    result["controlled_cns_source"]["held_source_law"] = copy.deepcopy(settings.get("held_source_law", {}))
    return result


def active_law_class(config):
    name = config["native_observable_bridge"].get("active_source_law","original_discrete_source")
    if name not in {"original_discrete_source","cooperative_exponential_source","romani_vm_source"}:
        raise ValueError("Unknown prospective source law")
    return held.HeldCNSLaw if name == "cooperative_exponential_source" else ControlledLaw


def physical_trajectory(config, family, parameters, action, *, high_precision=False,
                        refinement=None, source_runtime=None):
    active_law_class(config)
    law = config["native_observable_bridge"].get("active_source_law")
    if law == "romani_vm_source":
        if family != ELECTRICAL or source_runtime is None:
            raise ValueError("Romani Vm requires its actual source runtime and electrical domain")
        grid = family_grid(observable_profile(config), family)
        label = refinement or ("reference" if high_precision else "coarse")
        dt = romani.definition(config)["solver"]["dt_ms"][("coarse","fine","reference").index(label)]
        result = source_runtime.trajectory(parameters, action, grid, dt)
        result["refinement"] = label
        return result
    if refinement is not None:
        raise ValueError("Physical refinement level is only defined for the Romani source")
    physical_action = action
    if action.get("physical_update_clock") == "left_interval_current":
        if family != ELECTRICAL or law == "cooperative_exponential_source":
            raise ValueError("Shared current adaptation is limited to the primary electrical source")
        # The unchanged primary simulator uses right-index current for a left
        # interval update. Shift its private execution index on this uniform
        # clock; the canonical IClamp/compiler program retains the actual onset.
        times = np.asarray(family_grid(observable_profile(config), family)["query_time"]["values"])
        if not np.allclose(np.diff(times), times[1]-times[0], rtol=0, atol=0):
            raise ValueError("Shared primary current requires its registered uniform clock")
        physical_action = copy.deepcopy(action)
        physical_action["response_delay_steps"] += 1
        if physical_action["response_delay_steps"] >= len(times):
            raise ValueError("Shared current starts outside the primary integration clock")
    function = held.physical_trajectory if law == "cooperative_exponential_source" else original_physical_trajectory
    return function(config, family, parameters, physical_action, high_precision=high_precision)


def source_interval_action(config: dict, family: str, action: dict) -> dict:
    """Translate the source's actual update intervals into its canonical program."""
    if family == MOLECULAR or not config["native_observable_bridge"].get("bind_source_update_intervals"):
        return action
    grid = family_grid(observable_profile(config), family)
    if config["native_observable_bridge"].get("active_source_law") == "romani_vm_source":
        return action
    execution_action = copy.deepcopy(action)
    if action.get("physical_update_clock") == "left_interval_current":
        execution_action["response_delay_steps"] += 1
    endpoint_levels = np.asarray(action_levels(grid, execution_action), dtype=float)
    left_levels = np.r_[endpoint_levels[1:], 0.]
    indices = np.flatnonzero(left_levels)
    if not len(indices) or not np.all(left_levels[indices] == left_levels[indices[0]]):
        raise ValueError("Discrete source requires one nonempty constant-amplitude interval program")
    times = np.asarray(grid["query_time"]["values"])
    physical_action = copy.deepcopy(action)
    physical_action["response_delay_steps"] = int(indices[0])
    physical_action["extension"]["duration"]["value"] = float(times[indices[-1]+1]-times[indices[0]])
    if not np.array_equal(np.asarray(action_levels(grid, physical_action)), left_levels):
        raise ValueError("Compiled program differs from the actually executed source intervals")
    return physical_action


def source_action_timing(grid: dict, action: dict, physical_action: dict) -> dict:
    times = np.asarray(grid["query_time"]["values"], dtype=float)
    levels = np.asarray(action_levels(grid, physical_action), dtype=float)
    active = np.flatnonzero(levels[:-1])
    if not len(active):
        raise ValueError("Source action has no nonzero executed stimulation interval")
    start, stop = float(times[active[0]]), float(times[active[-1]+1])
    requested_duration = float(action["extension"]["duration"]["value"])
    return {"schema_version":"native_source_action_timing.v1",
        "source_requested_endpoint_index":action["response_delay_steps"],
        "source_requested_duration":requested_duration,
        "source_request_semantics":action.get("physical_update_clock","right_endpoint_update"),
        "actual_onset":start, "actual_stop":stop, "effective_duration":stop-start,
        "time_unit":grid["query_time"]["unit"],
        "cumulative_stimulation":float(np.dot(levels[:-1], np.diff(times))),
        "cumulative_stimulation_unit":action["extension"]["dose_or_intensity"]["unit"]+"*"+grid["query_time"]["unit"],
        "requested_duration_exact":requested_duration == stop-start,
        "forcing_semantics":"left_hold_on_actual_source_update_intervals"}


def compiled_program(config, group, support, atom, parameters, action, code_hash, estimate=None):
    physical_action = source_interval_action(config, group["family_uid"], action)
    record = original_compiled_program(config, group, support, atom, parameters, physical_action, code_hash, estimate)
    if group["family_uid"] != MOLECULAR and config["native_observable_bridge"].get("bind_source_update_intervals"):
        grid = family_grid(observable_profile(config), group["family_uid"])
        record["source_action_timing"] = source_action_timing(grid, action, physical_action)
    return record


def effective_stimulation_actions(config: dict, groups: list[dict]) -> list[dict]:
    """Report actual nominal-action equivalence using the production clock compiler."""
    representatives = {}
    for group in groups:
        if group["family_uid"] not in {ELECTRICAL, SYNAPTIC}:
            continue
        key = (group["family_uid"], group["source_variant"], group.get("action_stratum"))
        representatives.setdefault(key, group)
    reports = []
    for (family, variant, stratum), representative in representatives.items():
        group = copy.deepcopy(representative)
        group.pop("action_indices", None)
        scoped = scoped_group_config(config, group)
        grid = family_grid(observable_profile(scoped), family)
        actions = (romani.definition(scoped)["common_current"]["action_variants"]
                   if stratum == "shared_current" else scoped["native_observable_bridge"]["action_variants"])
        classes = {}
        for index in range(len(actions)):
            action = action_definition(scoped, family, index, group)
            effective = source_interval_action(scoped, family, action)
            waveform = tuple(action_levels(grid, effective))
            entry = classes.setdefault(waveform, {"action_indices": [], "source_requests": [],
                "time_values": grid["query_time"]["values"], "values": list(waveform)})
            entry["action_indices"].append(index)
            entry["source_requests"].append({"action_index":index, **source_action_timing(grid, action, effective)})
        rows = list(classes.values())
        reports.append({"family_uid":family, "source_variant":variant, "action_stratum":stratum,
            "declared_action_count":len(actions), "effective_program_count":len(rows),
            "collapsed_action_indices":[row["action_indices"] for row in rows if len(row["action_indices"]) > 1],
            "effective_programs":rows})
    return reports


class CoupledObservableLaw(OriginalCoupledObservableLaw):
    def __init__(self,config,group,support,truth,physical):
        super().__init__(config,group,support,truth,physical)
        self.law = active_law_class(config)(config["controlled_cns_source"],
            {name:value[0,truth].detach().cpu().numpy() for name,value in support.factors.items()},self.initial[...,:15])


def integrate_long_reference(law,action,method,step,deadline):
    if isinstance(law,held.HeldCNSLaw) and method == "DOP853":
        return held.integrate_reference(law,action,step,deadline)
    return cns.integrate(law,action,method,step,deadline)



def is_long_cns(config: dict, family: str) -> bool:
    return family == config["native_observable_bridge"].get("long_cns_family", {}).get("family_uid")


def long_cns_definition(config: dict) -> dict:
    settings = config["native_observable_bridge"]
    definition = copy.deepcopy(config["controlled_cns_source"])
    family = settings["long_cns_family"]["family_uid"]
    grid = family_grid(observable_profile(config), family)
    if (grid["time_hours"] != definition["grid"]["time_hours"] or len(grid["time_hours"]) != 28
            or grid["time_hours"][-1] != 672. or grid["species_id"][:15] != definition["grid"]["species_id"]):
        raise ValueError("Long-CNS adapter must retain actual canonical source coordinates")
    definition["source_uid"], definition["snapshot_ref"] = settings["source_uid"], settings["snapshot_ref"]
    definition["actions"] = []
    for index, row in enumerate(settings["long_cns_family"]["actions"]):
        action = copy.deepcopy(row)
        amount = action.get("total_dose_multiplier",1.)
        if isinstance(amount,bool) or not isinstance(amount,(int,float)) or not math.isfinite(amount) or amount <= 0:
            raise ValueError("Long-CNS dose multiplier must be finite and positive")
        times, fractions = np.asarray(action["dose_times_hours"]), np.asarray(action["fractions"])
        if (times.ndim != 1 or not len(times) or times.shape != fractions.shape
                or not np.isfinite(times).all() or not np.isfinite(fractions).all()
                or times[0] < 0 or times[-1] > 672 or not (np.diff(times)>0).all()
                or (fractions<=0).any() or not np.isclose(fractions.sum(),1.,rtol=0,atol=1e-12)):
            raise ValueError("Long-CNS action requires ordered real dose events and normalized positive fractions")
        action["regimen_uid"] = settings["source_uid"]+":"+family+":action:"+str(index)
        definition["actions"].append(action)
    return definition


def long_cns_law(config, group, support, atom, action=None):
    definition = long_cns_definition(config)
    if action is not None:
        definition["pk"]["total_dose_mg_per_kg"] *= action.get("total_dose_multiplier",1.)
    return active_law_class(config)(definition,
        {key:value[0,atom].detach().cpu().numpy() for key,value in support.factors.items()},
        support.initial_field[0,atom,...,:15].detach().cpu().numpy())


def long_cns_numerical_summary(config, results, pk_error, pk_serialization):
    definition = long_cns_definition(config)
    scales = [definition["reference_scales"][name] for name in definition["grid"]["species_id"]]
    summary = cns.numerical_summary(results["coarse"],results["fine"],results["reference"],
        scales,pk_error,pk_serialization,definition["numerics"])
    summary["uncertainty"] += [0.]*11
    summary["exact_zero_uncertainty"] = {}
    summary["unsupported_quantities"] = [q["quantity_id"] for q in observable_profile(config)["quantities"][15:]]
    summary["unsupported_meaning"] = "missing_masked_coordinates_not_exact_measurements"
    return summary



def prior_particle_count(config: dict) -> int:
    settings = config["native_observable_bridge"]
    atoms, weights = settings["atoms"], np.asarray(settings["prior_weights"], dtype=float)
    if (not atoms or any(len(row) != len(FACTOR_NAMES) or any(value not in (0, 1) for value in row) for row in atoms)
            or len({tuple(row) for row in atoms}) != len(atoms)
            or weights.shape != (len(atoms),) or not np.isfinite(weights).all()
            or (weights <= 0).any() or not np.isclose(weights.sum(), 1., rtol=0., atol=1e-12)):
        raise ValueError("Source requires unique complete five-factor atoms and normalized positive prior masses")
    if len(settings["coefficient_levels"]) != 2:
        raise ValueError("Binary source atoms require two declared coefficient levels")
    return len(atoms)


def action_count(config: dict, group: dict | None = None) -> int:
    settings = config["native_observable_bridge"]
    count = (len(settings["long_cns_family"]["actions"]) if group is not None and is_long_cns(config, group["family_uid"])
             else len(settings.get("action_variants", settings["action_multipliers"])))
    indices = list(range(count)) if group is None else group.get("action_indices", list(range(count)))
    if (len(indices) < 2 or len(set(indices)) != len(indices)
            or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < count for i in indices)):
        raise ValueError("Each source group requires distinct supported paired action indices")
    return len(indices)


def group_definitions(config: dict, *, role_filter: str | None = None) -> list[dict]:
    settings = config["native_observable_bridge"]
    particles = prior_particle_count(config)
    groups = []
    for family in observable_profile(config)["families"]:
        partitions = settings.get("group_partitions")
        if partitions is None:
            specifications = [(str(replicate), replicate, role,
                settings["calibration_reservations"].get(replicate), {})
                for replicate, role in enumerate(settings["group_roles"])]
        else:
            specifications = []
            if len({row["partition_uid"] for row in partitions}) != len(partitions):
                raise ValueError("Source group partition identities must be unique")
            for partition in partitions:
                if role_filter is not None and partition["role"] != role_filter:
                    continue
                count = partition["groups_per_family"]
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    raise ValueError("Source group counts must be positive integers")
                role, reservation = partition["role"], partition.get("calibration_reservation")
                if role not in {"train", "development", "calibration", "validation"}:
                    raise ValueError("Source group role is not supported")
                if (role == "calibration") != (reservation in {"fit", "check", "threshold"}):
                    raise ValueError("Calibration groups need disjoint fit/check/threshold reservations")
                for replicate in range(count):
                    key = partition["partition_uid"] + ":" + str(replicate)
                    metadata = {k:v for k,v in partition.items()
                                if k not in {"groups_per_family", "role", "calibration_reservation", "conditions"}}
                    if "conditions" in partition:
                        metadata["condition"] = partition["conditions"][replicate % len(partition["conditions"])]
                    strata = settings.get("training_source_strata")
                    if role == "train" and strata and family == strata["family_uid"]:
                        name = strata["cycle"][(replicate // strata["replicate_pair_size"]) % len(strata["cycle"])]
                        metadata.update(copy.deepcopy(strata["strata"][name]), source_stratum=name)
                    variant = settings.get("source_variants", {}).get(metadata.get("source_variant","primary"))
                    if variant:
                        if role not in variant["roles"] or (variant.get("family_uids") and family not in variant["family_uids"]):
                            raise ValueError("Source variant is not admitted for the requested split role")
                        metadata.update(source_uid=variant["source_uid"],source_snapshot_ref=variant["snapshot_ref"],
                                        source_law=variant["law"])
                    if role == "train":
                        metadata["training_subset_percentages"] = [fraction for fraction in settings.get("nested_training_percentages", [100])
                                                                  if replicate < count*fraction//100]
                    specifications.append((key, replicate, role, reservation, metadata))
        for key, replicate, role, reservation, metadata in specifications:
            uid = settings["source_uid"] + ":" + family + ":" + key
            group = {"group_uid":uid, "sample_uid":uid, "support_uid":uid+":support",
                "posterior_uid":uid+":initial-prior",
                "particle_uids":[uid+":atom:"+str(i) for i in range(particles)],
                "family_uid":family, "replicate":replicate, "role":role,
                "calibration_reservation":reservation, **copy.deepcopy(metadata)}
            action_count(config, group)
            groups.append(group)
    return groups




def scoped_training_groups(config: dict, groups: list[dict]) -> list[dict]:
    """Outcome-blind paired source contrast; all other families keep every group."""
    scope = config["native_scientific_training"].get("source_scope", "all")
    if scope == "all":
        return groups
    policy = config["manuscript_native_protocol"]["source_comparison"]
    if scope not in {"pure_shared", "mixed_shared"}:
        raise ValueError("Unknown matched training source scope")
    if config["native_scientific_training"].get("training_subset_percentage",100) != 100:
        raise ValueError("Matched source controls have no extra data-fraction sweep")
    count = policy["electrical_groups_per_arm"]
    if type(count) is not int or count < 2 or count % 2:
        raise ValueError("Matched source controls require an even positive electrical group count")
    primary = [row for row in groups if row.get("source_stratum") == "shared_primary"]
    romani_rows = [row for row in groups if row.get("source_stratum") == "romani"]
    if len(primary) != count or len(romani_rows) != count:
        raise ValueError("Matched source strata do not contain the declared complete groups")
    electrical = primary if scope == "pure_shared" else primary[:count//2] + romani_rows[count//2:]
    selected = {row["sample_uid"] for row in electrical}
    return [row for row in groups if row["family_uid"] != ELECTRICAL or row["sample_uid"] in selected]


def selected_group_definitions(config: dict, role: str, reservation: str | None = None) -> list[dict]:
    """Outcome-independent role, reservation and nested-fraction selection."""
    selected=[row for row in group_definitions(config,role_filter=role) if row["role"]==role
              and (reservation is None or row["calibration_reservation"]==reservation)]
    families=config["native_scientific_training"].get("family_scope")
    if families is not None:
        known=set(observable_profile(config)["families"])
        if not families or len(set(families))!=len(families) or not set(families)<=known:
            raise ValueError("Native family scope must be a declared nonempty unique domain set")
        selected=[row for row in selected if row["family_uid"] in families]
    if role=="train":
        selected=scoped_training_groups(config,selected)
        fraction=config["native_scientific_training"].get("training_subset_percentage",100)
        if fraction not in config["native_observable_bridge"].get("nested_training_percentages",[100]):
            raise ValueError("Training fraction was not declared before source generation")
        selected=[row for row in selected if fraction in row.get("training_subset_percentages",[100])]
    ids=[row["sample_uid"] for row in selected]
    if not ids or len(set(ids))!=len(ids):
        raise ValueError("Declared native group selection is empty or duplicated")
    return selected


def planned_operation_counts(config: dict) -> dict[str, int]:
    groups = group_definitions(config)
    cases = sum(action_count(config, group) for group in groups)
    particles = prior_particle_count(config)
    chemical = sum(action_count(config, group) for group in groups
                   if group["family_uid"] == MOLECULAR or is_long_cns(config, group["family_uid"]))
    original_cases = sum(action_count(config, group) for group in groups if not is_long_cns(config, group["family_uid"]))
    result = {"groups":len(groups), "cases":cases, "physical":original_cases, "physical_precision":original_cases,
              "cns_integrations":3*cases, "programs":particles*cases, "chemical_precision":particles*chemical}
    romani_groups = [row for row in groups if row.get("source_law") == "romani_vm_source"]
    if romani_groups:
        result.update(source_prepare=1, parameter_binding=particles*len(romani_groups),
                      physical_refinement=sum(action_count(config, row) for row in romani_groups))
    result["operations"] = sum(result[key] for key in result if key != "cases")
    return result


def action_definition(config: dict, family: str, action_index: int, group: dict | None = None) -> dict:
    settings = config["native_observable_bridge"]
    action_count(config, group)
    if group is not None and "action_indices" in group:
        action_index = group["action_indices"][action_index]
    if is_long_cns(config, family):
        return long_cns_definition(config)["actions"][action_index]
    definition = copy.deepcopy(settings["physical_families"][family]["action"])
    if "action_variants" in settings:
        variant = settings["action_variants"][action_index]
        multiplier = variant["intensity_multiplier"]
        duration = variant["duration_multiplier"]
        delay = variant["response_delay_steps"]
        if (isinstance(multiplier, bool) or not isinstance(multiplier, (int,float))
                or not math.isfinite(multiplier) or multiplier < 0
                or isinstance(duration, bool) or not isinstance(duration, (int,float))
                or not math.isfinite(duration) or duration <= 0
                or isinstance(delay, bool) or not isinstance(delay, int)
                or not 0 <= delay < len(observable_profile(config)["families"][family]["time_values"])):
            raise ValueError("Action variants require finite nonnegative intensity, positive duration and a registered delay")
        definition["extension"]["duration"]["value"] *= duration
        definition["response_delay_steps"] = delay
    else:
        multiplier = settings["action_multipliers"][action_index]
    definition["extension"]["dose_or_intensity"]["value"] *= multiplier
    if group is not None and group.get("action_stratum") == "shared_current":
        if family != ELECTRICAL:
            raise ValueError("Shared IClamp stratum is electrical only")
        common = romani.definition(config)["common_current"]
        variant = common["action_variants"][action_index]
        times = observable_profile(config)["families"][family]["time_values"]
        onset, duration = variant["delay_ms"], variant["duration_ms"]
        if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in (onset, duration)) or duration <= 0
                or onset not in times or onset + duration not in times):
            raise ValueError("Shared current onset and stop must be exact registered physical times with positive duration")
        definition["response_delay_steps"] = times.index(variant["delay_ms"])
        definition["extension"]["dose_or_intensity"] = {"value": common["base_amplitude_uA"]*variant["intensity_multiplier"], "unit":"uA"}
        definition["extension"]["duration"] = {"value":variant["duration_ms"], "unit":"ms"}
        definition["physical_update_clock"] = "left_interval_current"
    stratum = ":" + group["action_stratum"] if group is not None and group.get("action_stratum") else ""
    definition["regimen_uid"] = settings["source_uid"]+":"+family+stratum+":action:"+str(action_index)
    return definition


def draw_group(config: dict, group: dict, group_index: int, *, physical_initializer=None) -> dict:
    settings = config["native_observable_bridge"]
    particles = prior_particle_count(config)
    long_scope = is_long_cns(config, group["family_uid"])
    background = long_cns_definition(config) if long_scope else config["controlled_cns_source"]
    profile = observable_profile(config)
    grid = family_grid(profile, group["family_uid"])
    seed = int(settings["seed"]) + group_index * 100
    streams = {name: seed+int(offset) for name, offset in settings["rng_stream_offsets"].items()}
    rng = np.random.default_rng(streams["background"])
    factors = {}
    for fi, name in enumerate(FACTOR_NAMES):
        width = len(background["factor_basis"][name])
        factors[name] = np.asarray([[settings["coefficient_levels"][(atom[fi]+coordinate)%2]
                                    for coordinate in range(width)] for atom in settings["atoms"]], dtype=np.float32)
    is_romani = settings.get("active_source_law") == "romani_vm_source"
    declarations = {} if long_scope or is_romani else settings["physical_families"][group["family_uid"]]["parameters"]
    jitter_rng = np.random.default_rng(streams["physical_parameters"])
    jitter = {name: jitter_rng.uniform(0., 1.)*row.get("seed_jitter", 0.)*settings.get("physical_parameter_jitter_multiplier",1.) for name, row in declarations.items()}
    parameters, initial = [], np.zeros((particles, 6, 6, 26), dtype=np.float32)
    romani_parameters = romani.parameter_candidates(config, factors, jitter_rng) if is_romani else None
    if is_romani and physical_initializer is None:
        raise ValueError("Romani complete prior requires actual t0 parameter application/readbacks")
    readbacks = physical_initializer(romani_parameters) if is_romani else None
    if is_romani and (len(readbacks) != particles or any(row["time_ms"] != 0 for row in readbacks)):
        raise ValueError("Romani complete prior readbacks must precede action time and hidden selection")
    source_mask = quantity_support(grid, settings["source_uid"])[0]
    names = background["grid"]["species_id"]
    family_quantities = [(si, q) for si, q in enumerate(profile["quantities"]) if q.get("family_uid") == group["family_uid"]]
    for atom in range(particles):
        current = {}
        for index, (name, declaration) in enumerate(declarations.items()):
            factor_name, coordinate = settings["parameter_factor_binding"][group["family_uid"]][name]
            factor = factors[factor_name][atom, coordinate]
            sign = -1. if factor < 1. else 1.
            current[name] = float(declaration["value"]) + sign*jitter[name]
        if is_romani:
            current = romani_parameters[atom]
        parameters.append(current)
        variations = rng.uniform(*background["initial_jitter"], size=(36, len(DYNAMIC)))
        for si, name in enumerate(DYNAMIC):
            initial[atom, :, :, names.index(name)] = (background["initial_conditions"][settings["condition"]][name]
                * variations[:, si] * factors["z_env"][atom, np.arange(36)%factors["z_env"].shape[1]]).reshape(6, 6)
        law = active_law_class(config)(background, {key: value[atom] for key, value in factors.items()}, initial[atom, ..., :15])
        dynamic = initial[atom, ..., :15].reshape(1, 36, 15)[:, :, [names.index(name) for name in DYNAMIC]]
        initial[atom, ..., :15] = law.quantities([0.], dynamic, background["actions"][0])[0]
        if long_scope:
            parameters[atom] = flatten_cns_pk_parameters(law.pk)
        for si, quantity in family_quantities:
            site = quantity["incidence"]
            ri, ci = profile["region_id"].index(site["anchor_region_id"]), profile["compartment_id"].index(site["anchor_compartment_id"])
            if source_mask[ri, ci, si]:
                initial[atom, ri, ci, si] = (readbacks[atom]["Vm_mV"] if is_romani
                                            else current.get("initial_"+quantity["observable_name"], 0.))
    # Every complete prior atom and its physical parameters precede hidden selection.
    selector = background["observation"]["selector"]
    means = np.asarray([[initial[atom, row["region_index"], row["compartment_index"], names.index(row["species"])]
                         / background["reference_scales"][row["species"]] for row in selector] for atom in range(particles)], dtype=np.float32)
    matrix = np.asarray(settings.get("observation_matrix",np.eye(len(selector))),dtype=np.float64)
    if matrix.shape != (len(selector),len(selector)) or not np.isfinite(matrix).all():
        raise ValueError("Source measurement matrix must align to all declared normalized readouts")
    means = (means @ matrix.T).astype(np.float32)
    scales = [background["reference_scales"].get(q["quantity_id"], settings["auxiliary_reference_scales"].get(q["quantity_id"])) for q in profile["quantities"]]
    channel = "controlled_predose_selector"
    support = JointParticleSupport(group["support_uid"], settings["source_uid"], settings["snapshot_ref"],
        "controlled_simulated", group["role"], (group["sample_uid"],), (tuple(group["particle_uids"]),),
        {key: torch.from_numpy(value[None]) for key, value in factors.items()},
        (tuple("complete_atom_"+str(i) for i in range(particles)),), torch.from_numpy(initial[None]),
        torch.tensor([[scales]*particles], dtype=torch.float32), torch.tensor([settings["prior_weights"]], dtype=torch.float32),
        {channel: {"law": "gaussian", "units": "source_reference_standardized", "mean": torch.from_numpy(means[None]),
                   "variance": torch.full((1, particles, len(selector)), background["observation"]["standard_deviation"]**2)}},
        {channel: torch.zeros(1, particles, len(selector))}, tuple(q["unit"] for q in profile["quantities"]),
        quantity_profile=profile, source_family_uid=group["family_uid"]).validate()
    production = produce_controlled_observations(support, [{"channel_uid": channel,
        "observation_uid": group["sample_uid"]+":initial", "observed": [background["observation"]["mask"]],
        "encoder_indices": list(range(len(selector))), "normalization_ref": settings["snapshot_ref"]+":fixed_source_scales",
        "modality": "state", "assay_uid": "controlled_predose_gaussian"}],
        latent_seed=streams["latent_selection"], observation_seed=streams["observations"])
    return {"support": support, "production": production, "hidden_index": int(production.generating_indices[0]),
            "parameters": parameters, "rng_streams": streams, **({"parameter_readbacks":readbacks} if is_romani else {})}


def joint_source(config: dict, group: dict, generated: dict, records: list) -> dict:
    from Libs.utils.data.native_diffusion_data import JOINT_SOURCE_SCHEMA
    support=generated["support"]
    programs={"source_posterior_uid":group["posterior_uid"],"source_snapshot_ref":support.source_snapshot_ref,
        "support_uid":support.support_uid,"source_uid":support.source_uid,"sample_uids":[group["sample_uid"]],
        "particle_uids":[group["particle_uids"]],"regimen_uids":[records[0][ai]["regimen_uid"] for ai in range(len(records[0]))],"records":[records]}
    for name in ("occupancy","support","physical_time",*FORCING_ARRAY_NAMES):
        fields=[]
        for row in records:
            actions=[]
            for record in row:
                if record["representation"]=="explicit_stimulation_grid":
                    value=record["compiled_program"]["arrays"][name]
                elif name in FORCING_ARRAY_NAMES:
                    value=np.zeros_like(record["physical_program"]["occupancy"]).tolist()
                else:
                    value=record["physical_program"][name]
                actions.append(value)
            fields.append(actions)
        programs[name]=[fields]
    return {"schema_version":JOINT_SOURCE_SCHEMA,"source_posterior_uid":group["posterior_uid"],
        "support":support.to_payload(),"observations":generated["production"].inference_payload()["observations"],
        "particle_programs":programs,
        "producer_seeds":{"latent_seed":generated["production"].latent_seed,"observation_seed":generated["production"].observation_seed}}


def training_data_scope(config:dict) -> dict:
    """Checkpoint provenance for predeclared domains and within-domain data fractions."""
    settings=config["native_observable_bridge"]
    policy=config["native_scientific_training"]
    families=policy.get("family_scope",list(observable_profile(config)["families"]))
    scope = policy.get("source_scope","all")
    source_scope = {} if scope == "all" else {
        "source_scope":scope,
        "selected_training_group_sha256":hashlib.sha256(json.dumps(
            sorted(row["sample_uid"] for row in selected_group_definitions(config,"train"))).encode()).hexdigest()}
    return {**source_scope, "family_scope":list(families),"training_subset_percentage":policy.get("training_subset_percentage",100),
            "source_uid":settings["source_uid"],"source_snapshot_ref":settings["snapshot_ref"],
            "source_definition_sha256":hashlib.sha256(json.dumps(settings,sort_keys=True,allow_nan=False).encode()).hexdigest()}


def native_posterior_context_width(config:dict,operator_width:int) -> int:
    """Frozen scales share the existing declared inference width before operator projection."""
    conditioning=config.get("native_scientific_training",{}).get("conditioning",{"mode":"joint"})
    width=(config["native_diffusion_data"]["posterior_context_width"]
           if conditioning["mode"]=="frozen_selected_mainline" else operator_width)
    if type(width) is not int or width<1:
        raise ValueError("Native posterior context width must be a positive declared integer")
    return width


def native_training_profile(config:dict,normalization_id:str) -> dict:
    architecture=config["screening_pipeline"]["screening_benchmark"]["model_architecture"]
    backbone=architecture["rollout_operator"]["transition_backbone"]
    width=native_posterior_context_width(config,int(backbone["operator_hidden_dim"]))
    distillation=config.get("native_learned_distillation",{})
    return {**({"posterior_context_width":width} if width!=backbone["operator_hidden_dim"] else {}),
            **({"learned_distillation":copy.deepcopy(distillation)} if distillation.get("enabled") else {}),
            "backbone":copy.deepcopy(backbone),
            "action_representation":copy.deepcopy(architecture["rollout_operator"]["action_representation"]),
            "world_state_posterior":copy.deepcopy(architecture["world_state_posterior"]),
            "normalization_id":normalization_id,"training_data_scope":training_data_scope(config)}


def native_pretrainer_method(config: dict, method: str) -> bool:
    """Canonical profile registry also covers native Student and XL scales."""
    return method in config.get("manuscript_native_protocol", {}).get("native_method_profiles", {}) or (
        method.startswith("native_") and method not in config["native_controlled_comparators"]["conditional_mean_methods"])
