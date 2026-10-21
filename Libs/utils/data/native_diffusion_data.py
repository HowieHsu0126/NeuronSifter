"""Registry joins shared by native teacher writers, datasets and screening caches.

Only source metadata is read during admission. Array reads follow validation of
the entire selected set; failed rows are never silently dropped.
"""
from __future__ import annotations

import csv
import copy
import hashlib
import json
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from Libs.utils.data.local_membrane import MembraneAcquisition, select_readouts
from Libs.utils.data.native_observable_profile import observable_profile, validate_profile_grid, quantity_support, family_grid
from Libs.models.modules.world_state import JointParticleSupport
from Libs.utils.data.joint_physical_parameters import (
    candidate_conditioning, parameter_schema_from_config, select_candidate_samples,
    validate_program_parameter_subset,
)
from Libs.utils.data.native_field_normalization import normalization_spec, NativeNormalizationContext
from Libs.utils.data.native_joint_source_storage import read_joint_source_reference
from Libs.utils.data.physical_occupancy import (validate_physical_record, whole_particle_basis,
    source_site_exposure, SCALAR_EXPOSURE_DEFINITION, validate_scalar_exposure_definition)
from Libs.utils.data.joint_observation import observation_from_payload
from Libs.utils.data.native_stimulation import FORCING_ARRAY_NAMES, validate_stimulation_record
from Libs.utils.data.action_observation_contracts import compile_pharmacological_action
from Libs.models.modules.world_action_encoder import native_action_identity_spec
from Libs.utils.data.raw_data_registry import resolve_repo_path
from Libs.utils.data.native_input_staging import NativeInputLocations, input_locations_from_config
from Libs.utils.model import dynamics_store
from Libs.utils.model.screening_model_config import quantized_mddm_rollout_from_config, native_direct_mean_config
from Libs.utils.model.dynamics_fields import region_id_list, compartment_id_list, species_id_list

SCHEMA_VERSION = "native_diffusion_registry.v1"
JOINT_SOURCE_SCHEMA = "native_joint_observation_source.v1"
TEACHER_ARRAY_NAMES = {key: value for key, value in {
    "trajectory": "trajectories", "uncertainty": "teacher_uncertainty",
    "teacher_confidence": "teacher_confidence"}.items()}
ARRAY_NAMES = {
    "trajectory": "trajectories", "uncertainty": "teacher_uncertainty",
    "initial_state": "initial_state", "posterior_context": "posterior_context",
    "occupancy": "occupancy", "support": "support", "physical_time": "physical_time",
    "teacher_confidence": "teacher_confidence",
}
IDENTITY_FIELDS = (
    "trajectory_uid", "source_uid", "sample_uid", "regimen_uid", "action_uid",
    "target_program_uid", "program_uid", "posterior_uid", "family_uid", "content_hash",
)
SOURCE_FIELDS = ("license", "release", "snapshot_ref", "split_evidence_ref", "unit_evidence_ref")
ROLES = {"train", "development", "validation", "calibration"}


class NativeDataError(ValueError):
    """A rejected source binding, before model construction."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise NativeDataError(reason)


def _number(value: object, label: str) -> float:
    _require(not isinstance(value, (bool, str)) and isinstance(value, (int, float, np.number)),
             f"{label}: numeric value required (categorical confidence is not uncertainty)")
    value = float(value)
    _require(np.isfinite(value) and value >= 0, f"{label}: finite nonnegative value required")
    return value


def policy(config: dict) -> dict:
    return config.get("native_diffusion_data", {})


def native_action_policy(config: dict) -> str:
    return config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["action_representation"]["active_policy"]


def validate_compiled_program(program: dict) -> dict:
    """Reuse canonical native program admission for registry and serving callers."""
    _require(isinstance(program, dict), "program: metadata required")
    if program.get("representation") == "explicit_stimulation_grid":
        return validate_stimulation_record(program)
    for key in ("action", "compiled_program", "pkpd_context", "baseline_state_posterior", "graph_context"):
        _require(isinstance(program.get(key), dict), f"program: {key} metadata required")
    _require(program.get("representation") == "explicit_grid",
             "program: explicit CNS grid support required; scalar broadcast unsupported")
    _require(bool(program.get("posterior_uid"))
             and program["baseline_state_posterior"].get("state_ref") == program["posterior_uid"],
             "program posterior lineage mismatch")
    compiled = compile_pharmacological_action(
        program["action"], pkpd_context=program["pkpd_context"],
        baseline_state_posterior=program["baseline_state_posterior"], graph_context=program["graph_context"],
    )
    _require(compiled["status"] in {"compiled", "compiled_with_explicit_uncertainty"}
             and not compiled["blockers"] and compiled == program["compiled_program"],
             "program: compiler output/provenance mismatch")
    _require(compiled["program_uid"] == program.get("program_uid")
             and compiled["action_uid"] == program.get("action_uid"),
             "program: compiled program/action binding mismatch")
    _require(compiled["occupancy_fields"].get("array_name") == "occupancy"
             and compiled["occupancy_fields"].get("support_array") == "support",
             "program occupancy/support array binding mismatch")
    physical_digest = program["pkpd_context"].get("parameter_provenance", {}).get("physical_input_sha256")
    _require(bool(physical_digest) == ("physical_program" in program),
             "Physical program payload/provenance required together")
    if "physical_program" in program:
        validate_physical_record({key: value for key, value in program.items()
                                  if key not in {"store_path", "store_format", "program_index"}})
    return compiled


def program_action_identity(program: dict, regimen_uid: str | None = None, *, action_policy: str = "drug_id") -> list[float]:
    """Derive identity only after the canonical program has passed admission."""
    _require(isinstance(program, dict), "canonical action identity: program metadata required")
    _require(bool(program.get("regimen_uid")) and (regimen_uid is None or program["regimen_uid"] == regimen_uid),
             "canonical action identity: program/regimen binding mismatch")
    validate_compiled_program(program)
    if program.get("representation") == "explicit_stimulation_grid":
        _require(action_policy == "occupancy_program", "Chemical identity/exposure ablations do not admit stimulation aliases")
        # Metadata identity is not a pharmacological target or drug embedding.
        return [int(v)/255. for v in hashlib.sha256(json.dumps(program["action"], sort_keys=True).encode()).digest()[:4]]
    encoder, _ = native_action_identity_spec(action_policy)
    return encoder(program["action"])


def native_enabled(config: dict) -> bool:
    backbone = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    mddm = quantized_mddm_rollout_from_config(config, backbone) if backbone["kind"] == "neuronsifter_world_action" else None
    enabled = bool(native_direct_mean_config(config)) or bool(mddm and mddm["enabled"])
    allowed = {"occupancy_program", "drug_id", "scalar_exposure"}
    if mddm and mddm["enabled"]:
        allowed.add("target_id")
    if enabled and native_action_policy(config) not in allowed:
        raise NativeDataError(f"Native action policy {native_action_policy(config)!r} is unimplemented.")
    if enabled and native_action_policy(config) == "scalar_exposure":
        scalar_exposure_definition(config)
    return enabled


def scalar_exposure_definition(config: dict) -> dict:
    action = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["action_representation"]
    return validate_scalar_exposure_definition(action["policies"]["scalar_exposure"].get("native_definition"))


def scalar_exposure_view(records: list) -> dict:
    """Derived metadata in exactly the supplied sample/particle/action order."""
    def reduce(rows):
        return [reduce(row) if isinstance(row, list) else source_site_exposure(row) for row in rows]
    return {"definition": copy.deepcopy(SCALAR_EXPOSURE_DEFINITION), "records": reduce(records)}


def scalar_exposure_arrays(view: dict) -> dict:
    def collect(rows, key):
        return [collect(row, key) if isinstance(row, list) else row[key] for row in rows]
    result = {"scalar_exposure_auc": np.asarray(collect(view["records"], "values"), dtype=np.float64),
              "scalar_exposure_interval": np.asarray(collect(view["records"], "interval_hours"), dtype=np.float64)}
    def sparse_flags(rows):
        return [flag for row in rows for flag in (sparse_flags(row) if isinstance(row, list) else ["site_support" in row])]
    flags = sparse_flags(view["records"])
    if any(flags):
        _require(all(flags), "scalar exposure requires a consistent declared site-domain representation")
        result["scalar_exposure_site_support"] = np.asarray(collect(view["records"], "site_support"), dtype=bool)
    return result


def verify_scalar_exposure(supplied: dict, expected: dict) -> None:
    _require(supplied == expected, "scalar exposure source/value/definition mismatch")


def verify_scalar_arrays(payload: dict, view: dict) -> None:
    for key, expected in scalar_exposure_arrays(view).items():
        if key in payload:
            _require(np.array_equal(np.asarray(payload[key], dtype=np.float32), expected.astype(np.float32)),
                     "scalar exposure supplied numeric value mismatch")


def validate_scalar_cache(native: dict) -> None:
    joint = native.get("posterior_family") == "finite_joint_particles"
    records = (native["joint_source"]["particle_programs"]["records"] if joint
               else [record["program"] for record in native["records"]])
    expected = scalar_exposure_view(records)
    verify_scalar_exposure(native.get("scalar_exposure"), expected)
    if not joint:
        shape = (len(native["sample_uids"]), len(native["regimen_uids"]))
        for key, values in scalar_exposure_arrays(expected).items():
            values = values.astype(np.float32).reshape(*shape, *values.shape[1:])
            _require(key in native and np.array_equal(np.asarray(native[key]), values),
                     "scalar exposure cached numeric value mismatch")


def registry_path(config: dict, *, input_locations: NativeInputLocations | None = None) -> Path:
    value = policy(config).get("registry_path")
    _require(isinstance(value, str) and bool(value),
             "native_diffusion_data.registry_path: missing teacher/program/observation registry binding; native_conditioning_array_names alone cannot admit data")
    return input_locations.logical_path(value) if input_locations else resolve_repo_path(value)


def _unique(rows: list[dict], fields: tuple[str, ...], label: str) -> dict:
    output = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in fields)
        _require(all(key), f"{label}: missing identity {fields}")
        _require(key not in output, f"{label}: duplicate identity {key}")
        output[key] = row
    return output


def validate_grid(grid: dict, config: dict, *, input_locations: NativeInputLocations | None = None,
                 normalization_context: NativeNormalizationContext | None = None) -> None:
    profile = observable_profile(config)
    if profile is not None:
        validate_profile_grid(grid, profile)
    elif grid.get("quantity_profile") is not None:
        raise NativeDataError("Observable quantity profile requires its explicit configuration")
    for name, expected in (("region_id", region_id_list(config)),
                           ("compartment_id", compartment_id_list(config)),
                           ("species_id", species_id_list(config))):
        _require(grid.get(name) == expected, f"grid.{name}: ordering mismatch")
    times = np.asarray(grid.get("time_hours", []), dtype=float)
    _require(times.ndim == 1 and len(times) >= 2 and np.isfinite(times).all()
             and times[0] == 0 and np.all(np.diff(times) > 0),
             "grid.time_hours: require increasing physical times including initial time zero")
    _require(grid.get("physical_time_unit") == "hour", "grid.physical_time_unit: expected hour")
    units = policy(config).get("species_units")
    _require(isinstance(units, dict) and set(units) == set(grid["species_id"]) and all(units.values()),
             "native_diffusion_data.species_units: explicit source/training units required")
    _require(grid.get("species_units") == units, "grid.species_units: source/training units mismatch")
    normalization = grid.get("normalization", {})
    _require(normalization.get("kind") in {"physical", "training_scaler"}, "grid.normalization: missing explicit convention")
    normalization_spec(grid, config, input_locations=input_locations, normalization_context=normalization_context)


def validate_teacher_metadata(row: dict) -> list[str]:
    """Used also for bounded inspection of current, potentially non-native QC rows."""
    reasons = [f"teacher.{name}: missing identity/provenance" for name in IDENTITY_FIELDS if not row.get(name)]
    for name in ("trajectory_store_path", "store_format", "qc_catalog_ref", "source_snapshot_ref"):
        if not row.get(name):
            reasons.append(f"teacher.{name}: missing binding")
    try:
        _number(row.get("teacher_confidence"), "teacher.teacher_confidence")
    except NativeDataError as error:
        reasons.append(str(error))
    if row.get("uncertainty_origin") != "independent_supplied":
        reasons.append("teacher.uncertainty_origin: independent numerical uncertainty required; legacy catalog fallback is non-native")
    role = row.get("split_name")
    if role not in ROLES or row.get("source_holdout_role") != role:
        reasons.append("teacher.split_name/source_holdout_role: training/development/calibration role required")
    grade = row.get("qc_grade")
    use = row.get("qc_use")
    expected = {"A": ("main_pretraining", 1.0), "B": ("main_pretraining", .5), "C": ("weak_teacher_calibration", .1)}
    if grade not in expected or (grade == "C" and role != "calibration"):
        reasons.append("teacher.qc_grade: A/B training or C calibration-only required; D/S rejected")
    elif use != expected[grade][0]:
        reasons.append("teacher.qc_use: inconsistent QC grade")
    else:
        try:
            if float(row.get("sample_weight", -1)) != expected[grade][1]:
                reasons.append("teacher.sample_weight: inconsistent QC policy")
        except (TypeError, ValueError):
            reasons.append("teacher.sample_weight: numeric QC weight required")
    return reasons


def _joint_source_indexes(registry: dict) -> tuple[dict, dict]:
    inline = _unique(registry.get("joint_sources", []), ("source_posterior_uid",), "joint source")
    references = _unique(registry.get("joint_source_references", []), ("source_posterior_uid",), "joint source reference")
    _require(not set(inline) & set(references), "joint source has both inline and referenced authority")
    return inline, references


def _load_indexed_joint_sources(indexes: tuple[dict, dict], posterior_uids=None, *,
                                bindings: dict | None = None,
                                input_locations: NativeInputLocations | None = None,
                                copy_inline: bool = False) -> dict[str, dict]:
    inline, references = indexes
    uids = list(dict.fromkeys(posterior_uids)) if posterior_uids is not None else [key[0] for key in (*inline, *references)]
    result = {}
    for uid in uids:
        key = (uid,)
        _require(key in inline or key in references, "Joint source posterior is missing")
        if key in inline:
            result[uid] = copy.deepcopy(inline[key]) if copy_inline else inline[key]
        else:
            result[uid], hashes = read_joint_source_reference(references[key], **(
                {"input_locations": input_locations} if input_locations else {}))
            if bindings is not None:
                bindings.update(hashes)
    return result


def load_joint_sources(registry: dict, posterior_uids=None, *, bindings: dict | None = None,
                       input_locations: NativeInputLocations | None = None) -> dict[str, dict]:
    """Resolve selected payloads after checking authority across the complete registry."""
    return _load_indexed_joint_sources(_joint_source_indexes(registry), posterior_uids,
                                      bindings=bindings, input_locations=input_locations)


class NativeRegistryAdmission:
    """Own one metadata snapshot for ordered complete-group admission.

    Complete indexes are checked once. Groups contain snapshot row indices,
    never caller-supplied teacher metadata. Returned records cannot mutate the
    snapshot. Joint payload files and array evidence are checked on every load.
    Create a new admission for a changed registry or configuration.
    """
    def __init__(self, registry: dict, config: dict, *,
                 input_locations: NativeInputLocations | None = None,
                 normalization_context: NativeNormalizationContext | None = None):
        self.__normalization = normalization_context
        self.__registry = copy.deepcopy(registry)
        self.__config = copy.deepcopy(config)
        self.__locations = input_locations or input_locations_from_config(self.__config)
        _require(self.__registry.get("schema_version") == SCHEMA_VERSION, "native registry schema_version mismatch")
        validate_grid(self.__registry.get("grid", {}), self.__config, input_locations=self.__locations,
                      normalization_context=self.__normalization)
        self.__sources = _unique(self.__registry.get("sources", []), ("source_uid",), "source")
        self.__joint = self.__registry.get("posterior_family") == "finite_joint_particles"
        self.__joint_indexes = _joint_source_indexes(self.__registry) if self.__joint else None
        rows = self.__registry.get("teachers", [])
        self.__teachers = _unique(rows, ("trajectory_uid",), "teacher")
        _unique(rows, ("trajectory_store_path", "trajectory_index"), "teacher source row")
        _unique(rows, ("sample_uid", "regimen_uid"), "teacher sample/action")
        groups = {}
        for index, row in enumerate(rows):
            groups.setdefault(row["sample_uid"], []).append(index)
        self.__groups = tuple(tuple(group) for group in groups.values())
        self.__groups_set = frozenset(self.__groups)

    def groups(self, role: str | None = None, *, sample_uids: set[str] | None = None):
        """Return complete groups in the original teacher order."""
        for group in self.__groups:
            roles = {self.__registry["teachers"][index].get("split_name") for index in group}
            _require(len(roles) == 1 and roles <= ROLES, "Complete scientific group crosses source roles")
            if (role is None or role in roles) and (sample_uids is None or
                    self.__registry["teachers"][group[0]]["sample_uid"] in sample_uids):
                yield group

    def join(self, group: tuple[int, ...]) -> list[dict]:
        _require(isinstance(group, tuple) and group in self.__groups_set,
                 "admission requires a complete snapshot-owned group")
        selected = [copy.deepcopy(self.__registry["teachers"][index]) for index in group]
        _validate_selected_teachers(selected)
        if self.__joint:
            joint_sources = _load_indexed_joint_sources(self.__joint_indexes,
                [row["posterior_uid"] for row in selected], input_locations=self.__locations, copy_inline=True)
            return _join_joint_registry(self.__registry, self.__config, selected, self.__sources,
                                        input_locations=self.__locations, joint_sources=joint_sources)
        return copy.deepcopy(join_registry(self.__registry, self.__config, selected, input_locations=self.__locations,
                                          normalization_context=self.__normalization))

    def load(self, group: tuple[int, ...]) -> tuple[list[dict], dict, dict]:
        records = self.join(group)
        # Descriptor and array checks need only this group's joint authority.
        # Keep complete source declarations; stored provenance stays unchanged.
        view = dict(self.__registry)
        if self.__joint:
            uids = list(dict.fromkeys(row["teacher"]["posterior_uid"] for row in records))
            inline, references = self.__joint_indexes
            view.update(teachers=[row["teacher"] for row in records],
                        joint_sources=[row["joint_source"] for row in records
                                       if (row["teacher"]["posterior_uid"],) in inline],
                        joint_source_references=[references[(uid,)] for uid in uids if (uid,) in references])
            view["joint_sources"] = list({row["source_posterior_uid"]: row for row in view["joint_sources"]}.values())
        arrays, bindings = load_records(view, records, self.__config, input_locations=self.__locations,
                                        normalization_context=self.__normalization)
        return records, arrays, bindings

    def reindex(self, records: list[dict], store_path: Path, store_format: str) -> dict:
        for record in records:
            teacher = record["teacher"]
            _require(self.__teachers.get((str(teacher.get("trajectory_uid", "")),)) == teacher,
                     "admission record teacher differs from its snapshot")
            if self.__joint:
                key = (teacher["posterior_uid"],)
                inline, _ = self.__joint_indexes
                if key in inline:
                    _require(record.get("joint_source") == inline[key],
                             "admission inline joint source differs from its snapshot")
        return _reindex_registry(self.__registry, records, store_path, store_format,
                                joint_references=self.__joint_indexes[1] if self.__joint else None)


def _validate_selected_teachers(selected: list[dict]) -> None:
    _require(bool(selected), "native registry contains no selected teacher rows")
    all_reasons = [f"{row.get('trajectory_uid', index)}: {reason}" for index, row in enumerate(selected)
                   for reason in validate_teacher_metadata(row)]
    _require(not all_reasons, "; ".join(all_reasons))
    _unique(selected, ("trajectory_uid",), "teacher")
    _unique(selected, ("trajectory_store_path", "trajectory_index"), "teacher source row")
    _unique(selected, ("sample_uid", "regimen_uid"), "teacher sample/action")


def join_registry(registry: dict, config: dict, teachers: list[dict] | None = None, *,
                  input_locations: NativeInputLocations | None = None,
                 normalization_context: NativeNormalizationContext | None = None) -> list[dict]:
    input_locations = input_locations or input_locations_from_config(config)
    _require(registry.get("schema_version") == SCHEMA_VERSION, "native registry schema_version mismatch")
    selected = registry.get("teachers", []) if teachers is None else teachers
    _validate_selected_teachers(selected)
    validate_grid(registry.get("grid", {}), config, input_locations=input_locations, normalization_context=normalization_context)
    sources = _unique(registry.get("sources", []), ("source_uid",), "source")
    if registry.get("posterior_family") == "finite_joint_particles":
        return _join_joint_registry(registry, config, selected, sources, input_locations=input_locations)
    _require(posterior_family(config) != "finite_joint_particles",
             "Joint family requires raw joint_sources; point posterior stores are not observation sources")
    programs = _unique(registry.get("programs", []), ("program_uid", "posterior_uid"), "program/posterior")
    observations = _unique(registry.get("observations", []), ("posterior_uid",), "observation posterior")
    posterior_by_sample = {}
    for teacher in selected:
        previous = posterior_by_sample.setdefault(teacher["sample_uid"], teacher["posterior_uid"])
        _require(previous == teacher["posterior_uid"], "same sample action comparison requires one posterior lineage")
    result = []
    for teacher in selected:
        source = sources.get((str(teacher["source_uid"]),), {})
        _require(source.get("public") is True and source.get("protected") is False
                 and all(source.get(key) for key in SOURCE_FIELDS), "source: public/license/release/snapshot/split/unit evidence required")
        _require(source["snapshot_ref"] == teacher["source_snapshot_ref"], "teacher source snapshot mismatch")
        observation = observations.get((str(teacher["posterior_uid"]),), {})
        program = programs.get((str(teacher["program_uid"]), str(teacher["posterior_uid"])), {})
        _require(observation and program, "missing observation-derived posterior or compiled program registry join")
        _require(observation.get("sample_uid") == teacher["sample_uid"]
                 and observation.get("source_uid") == teacher["source_uid"], "observation sample/source identity mismatch")
        _require(observation.get("origin") == "observation_derived" and observation.get("observation_ref")
                 and observation.get("source_snapshot_ref") == source["snapshot_ref"],
                 "observation: independent observation/posterior provenance required")
        _require(observation.get("initial_time_hours") == 0, "observation initial time mismatch")
        _require(program.get("source_uid") == teacher["source_uid"]
                 and program.get("action_uid") == teacher["action_uid"]
                 and program.get("regimen_uid") == teacher["regimen_uid"]
                 and program.get("target_program_uid") == teacher["target_program_uid"],
                 "program action/regimen/target/source identity mismatch")
        compiled = validate_compiled_program(program)
        if native_action_policy(config) == "scalar_exposure":
            scalar_exposure_definition(config)
            source_site_exposure(program)
            inputs = program["physical_program"]["inputs"]
            _require(inputs["binding"]["sample_uid"] == teacher["sample_uid"]
                     and inputs["region_ids"] == registry["grid"]["region_id"]
                     and inputs["compartment_ids"] == registry["grid"]["compartment_id"],
                     "scalar exposure source sample/site order mismatch")
        _require(compiled["program_uid"] == teacher["program_uid"]
                 and compiled["action_uid"] == teacher["action_uid"]
                 and compiled["time_grid"] == registry["grid"]["time_hours"], "program identity/time mismatch")
        for owner, index_name, path_name in ((teacher, "trajectory_index", "trajectory_store_path"),
                                             (observation, "observation_index", "store_path"),
                                             (program, "program_index", "store_path")):
            _require(isinstance(owner.get(index_name), int) and owner[index_name] >= 0
                     and bool(owner.get(path_name)), f"{index_name}/{path_name}: explicit nonnegative source index required")
            _require(owner.get("store_format") in {"zarr_v2", "neuronsifter_chunk_store"}, "unsupported native store format")
        if native_action_policy(config) in {"drug_id", "target_id"}:
            encoder, _ = native_action_identity_spec(native_action_policy(config))
            encoder(program["action"])
        result.append({"teacher": teacher, "observation": observation, "program": program})
    return result


def _owner(record: dict, key: str) -> tuple[dict, str, str]:
    if key in {"trajectory", "uncertainty", "teacher_confidence", "quantity_support"}:
        return record["teacher"], "trajectory_store_path", "trajectory_index"
    if key in {"initial_state", "posterior_context"}:
        return record["observation"], "store_path", "observation_index"
    return record["program"], "store_path", "program_index"


def _shapes(grid: dict, context_width: int) -> dict[str, tuple[int, ...]]:
    times, regions, compartments, species = (len(grid[name]) for name in ("time_hours", "region_id", "compartment_id", "species_id"))
    return {
        "trajectory": (times, regions, compartments, species), "quantity_support": (times, regions, compartments, species), "uncertainty": (species,),
        "initial_state": (regions, compartments, species),
        "posterior_context": (regions, compartments, context_width),
        "occupancy": (times, regions, compartments, 1), "support": (times, regions, compartments, 1),
        "physical_time": (times, regions, compartments, 1), "teacher_confidence": (),
        **{name: (times, regions, compartments, 1) for name in FORCING_ARRAY_NAMES},
    }


def descriptor_bindings(records: list[dict], registry: dict, config: dict, *,
                        input_locations: NativeInputLocations | None = None,
                 normalization_context: NativeNormalizationContext | None = None) -> dict[str, str]:
    """Validate descriptors for all selected records before opening any array payload."""
    input_locations = input_locations or input_locations_from_config(config)
    expected = _shapes(registry["grid"], int(policy(config)["posterior_context_width"]))
    bindings = {}
    # These snapshots live for this admission call only. Every row/array binding
    # is still compared; a later call rereads files and detects changed evidence.
    descriptors: dict[Path, dict] = {}
    source_digests: dict[int, str] = {}
    stored_joint_sources: dict[tuple[int, str], dict] = {}

    def read_descriptor(path: Path) -> dict:
        if path not in descriptors:
            text = input_locations.read_text(path) if input_locations else path.read_text(encoding="utf-8")
            descriptors[path] = json.loads(text)
            bindings[str(path)] = hashlib.sha256(text.encode()).hexdigest()
        return descriptors[path]

    def joint_digest(source: dict) -> str:
        identity = id(source)
        if identity not in source_digests:
            source_digests[identity] = _source_digest(source)
        return source_digests[identity]

    transform = normalization_spec(registry["grid"], config, input_locations=input_locations, normalization_context=normalization_context)
    if transform["kind"] == "training_scaler":
        bindings[transform["artifact_ref"]] = transform["sha256"]
    for record in records:
        for key, array_name in consumed_array_names(registry).items():
            owner, path_name, index_name = _owner(record, key)
            path = input_locations.logical_path(owner[path_name]) if input_locations else resolve_repo_path(owner[path_name])
            descriptor = path / ("store_manifest.json" if owner["store_format"] == "neuronsifter_chunk_store" else ".zattrs")
            metadata = read_descriptor(descriptor)
            stored_registry = metadata.get("metadata", metadata).get("native_diffusion", {})
            store_grid = stored_registry.get("grid")
            declared_source = next(row for row in registry["sources"] if row["source_uid"] == record["teacher"]["source_uid"])
            stored_source = next((row for row in stored_registry.get("sources", []) if row.get("source_uid") == declared_source["source_uid"]), {})
            _require(stored_source.get("predictive_evaluation_units") == declared_source.get("predictive_evaluation_units"),
                     "predictive episode/group source declaration mismatch")
            _require(stored_source.get("exact_zero_quantities") == declared_source.get("exact_zero_quantities"),
                     "exact-zero source declaration mismatch")
            collection, identities = ("teachers", (*IDENTITY_FIELDS, "split_name", "source_holdout_role", "qc_grade", "qc_use", "sample_weight", "validity_mask", "exact_zero_uncertainty")) if key in {"trajectory", "uncertainty", "teacher_confidence", "quantity_support"} else (
                ("observations", ("posterior_uid", "sample_uid", "source_uid", "screening_features")) if key in {"initial_state", "posterior_context"} else (
                    "programs", ("program_uid", "posterior_uid", "action_uid", "regimen_uid", "source_uid", "action", "compiled_program", "physical_program")))
            stored_rows = [row for row in stored_registry.get(collection, []) if row.get(index_name) == owner[index_name]]
            _require(len(stored_rows) == 1 and all(stored_rows[0].get(field) == owner.get(field) for field in identities),
                     f"{key}: store row identity mismatch")
            if key == "uncertainty":
                _require(stored_rows[0].get("uncertainty_origin") == "independent_supplied", "uncertainty: legacy fallback is non-native")
            _require(store_grid == registry["grid"], f"{path}: native source grid metadata mismatch")
            if "joint_source" in record:
                uid = record["teacher"]["posterior_uid"]
                cache_key = (id(stored_registry), uid)
                if cache_key not in stored_joint_sources:
                    stored_joint_sources[cache_key] = load_joint_sources(stored_registry, [uid], bindings=bindings, input_locations=input_locations)[uid]
                _require(joint_digest(stored_joint_sources[cache_key]) == joint_digest(record["joint_source"]),
                         "joint source: store observation/support binding mismatch")
            if owner["store_format"] == "neuronsifter_chunk_store":
                array = metadata.get("arrays", {}).get(array_name, {})
            else:
                array_path = path / array_name / ".zarray"
                array = read_descriptor(array_path)
            if key == "quantity_support":
                _require(np.dtype(array.get("dtype")) == np.dtype(bool),
                         "quantity_support: storage descriptor must declare boolean dtype")
            shape = array.get("shape", [])
            _require(len(shape) >= 1 and tuple(shape[1:]) == expected[key]
                     and owner[index_name] < shape[0], f"{key}: array axes/cardinality/source index mismatch")
    return bindings


def validate_arrays(arrays: dict, registry: dict, records: list[dict], config: dict, *, include_teacher: bool = True) -> None:
    expected = _shapes(registry["grid"], int(policy(config)["posterior_context_width"]))
    for key in consumed_array_names(registry):
        if not include_teacher and key in {*TEACHER_ARRAY_NAMES, "quantity_support"}:
            continue
        tail = expected[key]
        value = np.asarray(arrays[key])
        _require(value.shape == (len(records), *tail), f"{key}: array axes mismatch {value.shape}")
        _require((np.issubdtype(value.dtype, np.number) or (key == "quantity_support" and value.dtype == np.bool_))
                 and np.isfinite(value).all(), f"{key}: finite numeric array required")
    if include_teacher:
        if registry["grid"].get("quantity_profile") is not None:
            expected_support = np.stack([quantity_support(registry["grid"], row["teacher"]["source_uid"]) for row in records])
            _require(arrays["quantity_support"].dtype == np.bool_
                     and np.array_equal(arrays["quantity_support"], expected_support),
                     "quantity_support: source/profile mask mismatch")
            _require((arrays["trajectory"][~expected_support] == 0).all(),
                     "missing quantities require masked storage placeholders")
        uncertainty = arrays["uncertainty"]
        _require((uncertainty >= 0).all(), "uncertainty: negative numerical uncertainty")
        for row_index, record in enumerate(records):
            for species_index in np.flatnonzero(uncertainty[row_index] == 0):
                species = registry["grid"]["species_id"][species_index]
                if "quantity_support" in arrays and not arrays["quantity_support"][row_index,...,species_index].any():
                    continue
                evidence = record["teacher"].get("exact_zero_uncertainty", {}).get(species, {})
                source = next(row for row in registry["sources"] if row["source_uid"] == record["teacher"]["source_uid"])
                declared = source.get("exact_zero_quantities", {}).get(species, {})
                required = config["schemas"]["native_exact_zero_uncertainty"]["required_fields"]
                _require(set(required) <= set(evidence)
                         and evidence.get("status") == "exact_by_definition"
                         and evidence.get("derivation_kind") == declared.get("kind") == "identically_zero"
                         and bool(evidence.get("derivation_ref"))
                         and evidence.get("derivation_ref") == declared.get("derivation_ref")
                         and type(evidence.get("serialization_error")) in (int, float)
                         and evidence.get("serialization_error") == 0.0
                         and evidence.get("source_snapshot_ref") == record["teacher"]["source_snapshot_ref"]
                         and evidence.get("quantity") == species
                         and (arrays["trajectory"][row_index,...,species_index] == 0).all(),
                         "uncertainty: genuine positive numerical uncertainty or bound exact-zero derivation required")
        _require((arrays["teacher_confidence"] >= 0).all(), "teacher_confidence: negative value")
        _require(np.array_equal(arrays["teacher_confidence"], np.asarray([r["teacher"]["teacher_confidence"] for r in records], dtype=np.float32)),
                 "teacher_confidence: numeric source value mismatch")
    if registry.get("posterior_family") == "finite_joint_particles":
        return
    for name in ("occupancy", "support"):
        _require(((arrays[name] >= 0) & (arrays[name] <= 1)).all(), f"{name}: values outside [0,1]")
    expected_time = np.asarray(registry["grid"]["time_hours"], dtype=np.float32).reshape(1, -1, 1, 1, 1)
    _require(np.all(arrays["physical_time"] == expected_time), "physical_time: hour grid mismatch")
    if native_action_policy(config) == "scalar_exposure":
        for index, record in enumerate(records):
            for name in ("occupancy", "support", "physical_time"):
                expected_value = np.asarray(record["program"]["physical_program"][name], dtype=np.float32)
                _require(np.array_equal(arrays[name][index], expected_value),
                         "scalar exposure physical program array mismatch: " + name)
    if registry["grid"].get("quantity_profile"):
        for index, record in enumerate(records):
            compiled = validate_compiled_program(record["program"])
            stimulation = record["program"].get("representation") == "explicit_stimulation_grid"
            for name in FORCING_ARRAY_NAMES:
                expected_value = np.asarray(compiled["arrays"][name], dtype=np.float32) if stimulation else np.zeros_like(arrays[name][index])
                _require(np.array_equal(arrays[name][index], expected_value), "typed numerical stimulation mismatch: " + name)
            if stimulation:
                for name in ("occupancy", "support"):
                    _require((arrays[name][index] == 0).all(), "Stimulation cannot carry chemical occupancy aliases")
    if include_teacher:
        for name in ("initial_state", "posterior_context", "occupancy"):
            _require(not np.shares_memory(arrays["trajectory"], arrays[name]), f"{name}: teacher-derived view is not independent conditioning")


def load_records(registry: dict, records: list[dict], config: dict, *, include_teacher: bool = True,
                 input_locations: NativeInputLocations | None = None,
                 normalization_context: NativeNormalizationContext | None = None) -> tuple[dict, dict]:
    input_locations = input_locations or input_locations_from_config(config)
    bindings = descriptor_bindings(records, registry, config, input_locations=input_locations, normalization_context=normalization_context)
    verified_arrays: set[Path] = set()
    arrays = {}
    for key, array_name in consumed_array_names(registry).items():
        if not include_teacher and key in {*TEACHER_ARRAY_NAMES, "quantity_support"}:
            continue
        rows: list[np.ndarray | None] = [None] * len(records)
        by_store: dict[tuple[Path, str], list[tuple[int, int]]] = {}
        for position, record in enumerate(records):
            owner, path_name, index_name = _owner(record, key)
            path = input_locations.logical_path(owner[path_name]) if input_locations else resolve_repo_path(owner[path_name])
            store = (path, owner["store_format"])
            by_store.setdefault(store, []).append((position, owner[index_name]))
        for (path, store_format), positions in by_store.items():
            loader = dynamics_store.load_chunked_rows if store_format == "neuronsifter_chunk_store" else dynamics_store.load_zarr_rows
            indices = [index for _, index in positions]
            if input_locations:
                descriptor = path / ("store_manifest.json" if store_format == "neuronsifter_chunk_store" else array_name + "/.zarray")
                metadata = json.loads(input_locations.read_text(descriptor))
                array = metadata["arrays"][array_name] if store_format == "neuronsifter_chunk_store" else metadata
                input_locations.check_array_files(path, array_name, array, store_format, indices, verified_arrays)
                physical_path = input_locations.store_path(path)
            else:
                physical_path = path
            loaded = loader(physical_path, array_name, np.asarray(indices, dtype=np.int64))
            for (position, _), value in zip(positions, loaded):
                rows[position] = value
        values = np.stack(rows)
        if key == "quantity_support":
            _require(values.dtype == np.bool_, "quantity_support: storage values must retain boolean dtype")
            arrays[key] = values
        else:
            arrays[key] = values.astype(np.float32, copy=False)
    validate_arrays(arrays, registry, records, config, include_teacher=include_teacher)
    if native_action_policy(config) in {"drug_id", "target_id"} and registry.get("posterior_family") != "finite_joint_particles":
        arrays["action_identity"] = np.asarray([program_action_identity(record["program"], record["teacher"]["regimen_uid"],
                                                                          action_policy=native_action_policy(config))
                                                for record in records], dtype=np.float32)
    if native_action_policy(config) == "scalar_exposure" and registry.get("posterior_family") != "finite_joint_particles":
        view = scalar_exposure_view([record["program"] for record in records])
        arrays.update({key: value.astype(np.float32) for key, value in scalar_exposure_arrays(view).items()})
    return arrays, bindings


FAMILY_INDEX_SCHEMA = "native_diffusion_family_index.v1"


def load_family_registries(config: dict, *, input_locations: NativeInputLocations | None = None,
                 normalization_context: NativeNormalizationContext | None = None) -> dict[str, dict]:
    """Admit explicit family registry references without selecting a default clock."""
    input_locations = input_locations or input_locations_from_config(config)
    path = registry_path(config, input_locations=input_locations)
    metadata = json.loads(input_locations.read_text(path) if input_locations else path.read_text(encoding="utf-8"))
    registry = metadata.get("metadata", metadata).get("native_diffusion", metadata)
    if registry.get("schema_version") != FAMILY_INDEX_SCHEMA:
        return {registry["grid"].get("family_uid", "legacy"): registry}
    profile = observable_profile(config)
    _require(profile is not None, "family index requires an explicit observable profile")
    references = registry.get("family_registries", {})
    _require(isinstance(references, dict) and set(references) == set(profile["families"]),
             "family registry index must contain every registered clock")
    result = {}
    for family, path in references.items():
        payload = json.loads(input_locations.read_text(path) if input_locations else resolve_repo_path(path).read_text(encoding="utf-8"))
        child = payload.get("metadata", payload).get("native_diffusion", payload)
        _require(child.get("schema_version") == SCHEMA_VERSION
                 and child.get("grid", {}).get("family_uid") == family,
                 "family registry index points to a different source family")
        validate_grid(child["grid"], config, input_locations=input_locations, normalization_context=normalization_context)
        _require(child["grid"]["quantity_profile_sha256"] == registry.get("quantity_profile_sha256"),
                 "family index quantity profile identity mismatch")
        result[family] = child
    return result


def load_registry(config: dict, *, normalization_context: NativeNormalizationContext | None = None) -> dict:
    registries = load_family_registries(config, normalization_context=normalization_context)
    selected = policy(config).get("source_family_uid")
    if selected is not None:
        _require(selected in registries, "requested source family is not in the registry")
        return registries[selected]
    _require(len(registries) == 1, "multiple family clocks require explicit source_family_uid selection")
    return next(iter(registries.values()))


def select_records(registry: dict, config: dict, split_name: str,
                   sample_uids: list[str] | None = None, regimen_uids: list[str] | None = None, *, normalization_context: NativeNormalizationContext | None = None) -> list[dict]:
    _require(split_name in ROLES, "native teacher requested role must be training/development/calibration")
    selected = [row for row in registry["teachers"] if row.get("split_name") == split_name]
    if sample_uids is not None:
        _require(len(set(sample_uids)) == len(sample_uids) and len(set(regimen_uids or [])) == len(regimen_uids or []),
                 "duplicate selected sample/regimen identity")
        lookup = _unique(selected, ("sample_uid", "regimen_uid"), "teacher selection")
        missing = [(sample, regimen) for sample in sample_uids for regimen in regimen_uids or [] if (sample, regimen) not in lookup]
        _require(not missing, f"missing native sample/regimen joins: {missing}")
        selected = [lookup[(sample, regimen)] for sample in sample_uids for regimen in regimen_uids or []]
    return join_registry(registry, config, selected, normalization_context=normalization_context)


def select_store_records(registry: dict, config: dict, store_path: Path, split_name: str | None,
                         trajectory_uids: list[str] | None = None,
                         row_indices: list[int] | None = None, *, normalization_context: NativeNormalizationContext | None = None) -> list[dict]:
    """Bind a direct native Zarr selection to its requested role and registry UIDs."""
    _require(split_name in ROLES, "native dataset requires an explicit training/development/calibration role")
    store_rows = [row for row in registry["teachers"]
                  if resolve_repo_path(row["trajectory_store_path"]) == store_path.resolve()
                  and row["store_format"] == "zarr_v2"]
    if trajectory_uids is None:
        _require(row_indices is None, "native positional selection requires trajectory identity binding")
        selected = [row for row in store_rows if row.get("split_name") == split_name]
    else:
        _require(len(set(trajectory_uids)) == len(trajectory_uids), "duplicate native trajectory identity")
        lookup = _unique(store_rows, ("trajectory_uid",), "native store trajectory")
        _require(all((uid,) in lookup for uid in trajectory_uids), "native trajectory identity missing from requested store")
        selected = [lookup[(uid,)] for uid in trajectory_uids]
    _require(bool(selected) and all(row.get("split_name") == split_name for row in selected),
             "native trajectory selection does not match requested role")
    if row_indices is not None:
        _require(row_indices == [row["trajectory_index"] for row in selected],
                 "native positional selection disagrees with trajectory identity binding")
    return join_registry(registry, config, selected, normalization_context=normalization_context)


def stable_numeric_id(uid: str) -> int:
    return int.from_bytes(hashlib.sha256(uid.encode()).digest()[:7], "big")


def screening_native_payload(config: dict, split_name: str, sample_uids: list[str], regimen_uids: list[str]) -> dict:
    _require(split_name in ROLES, "native teacher requested role must be training/development/calibration")
    registry = load_registry(config)
    records = select_records(registry, config, split_name, sample_uids, regimen_uids)
    arrays, bindings = load_records(registry, records, config)
    bindings[str(registry_path(config))] = hashlib.sha256(registry_path(config).read_bytes()).hexdigest()
    shape = (len(sample_uids), len(regimen_uids))
    result = {("teacher_trajectory" if key == "trajectory" else key): value.reshape(*shape, *value.shape[1:]) for key, value in arrays.items()}
    result.update(action_identity_schema=native_action_identity_spec(native_action_policy(config))[1], action_policy=native_action_policy(config),
                  split_name=split_name, sample_uids=sample_uids, regimen_uids=regimen_uids, source_bindings=bindings,
                  trajectory_uids=[r["teacher"]["trajectory_uid"] for r in records],
                  posterior_uids=[r["teacher"]["posterior_uid"] for r in records],
                  records=records, grid=registry["grid"])
    result["posterior_family"] = registry.get("posterior_family", "deterministic_compatibility")
    if result["posterior_family"] == "finite_joint_particles":
        sources = {record["joint_source"]["source_posterior_uid"] for record in records}
        _require(len(sources) == 1, "Screening batch requires a common declared joint source; partition source roles before batching")
        result["joint_source"] = select_joint_source(records[0]["joint_source"], sample_uids, regimen_uids,
                                                     action_policy=native_action_policy(config))
        result["joint_source_digest"] = _source_digest(result["joint_source"])
    if native_action_policy(config) == "scalar_exposure":
        result["scalar_exposure"] = scalar_exposure_view(
            result["joint_source"]["particle_programs"]["records"] if result["posterior_family"] == "finite_joint_particles"
            else [record["program"] for record in records])
    return result


def validate_cached_native(payload: dict, regimen_uids: list[str], config: dict | None = None) -> None:
    for splits in payload.values():
        for role, item in splits.items():
            if role not in ROLES:
                _require("native_diffusion" not in item, "cache native teachers are forbidden for this role")
                continue
            native = item.get("native_diffusion")
            _require(isinstance(native, dict), "cache native_diffusion is missing")
            _require(native.get("split_name") == role, "cache native requested role mismatch")
            action_policy = native.get("action_policy")
            _require(action_policy in {"occupancy_program", "drug_id", "target_id", "scalar_exposure"}
                     and (config is None or action_policy == native_action_policy(config)),
                     "cache_stale: native action policy mismatch")
            _require(native.get("action_identity_schema") == native_action_identity_spec(action_policy)[1],
                     "cache_stale: native action identity schema mismatch")
            _require(native.get("sample_uids") == item.get("sample_uids")
                     and native.get("regimen_uids") == regimen_uids, "cache native sample/regimen identity mismatch")
            count = len(item["sample_uids"]) * len(regimen_uids)
            if config is not None:
                validate_grid(native.get("grid", {}), config)
            records = native.get("records", [])
            _require(len(records) == count and all(
                record["teacher"].get("split_name") == role
                and record["teacher"].get("source_holdout_role") == role for record in records),
                "cache native teacher role binding mismatch")
            _require(all(record["teacher"].get("sample_uid") == item["sample_uids"][index // len(regimen_uids)]
                         and record["teacher"].get("regimen_uid") == regimen_uids[index % len(regimen_uids)]
                         for index, record in enumerate(records)), "cache native record order mismatch")
            joint = native.get("posterior_family") == "finite_joint_particles"
            if config is not None:
                _require(native.get("posterior_family", "deterministic_compatibility") == posterior_family(config),
                         "cache_stale: posterior family mismatch")
            if joint:
                source = native.get("joint_source", {})
                _require(native.get("joint_source_digest") == _source_digest(source), "cache_stale: joint source changed")
                support, _ = parse_joint_source(source, action_policy=action_policy)
                parameter_schema = parameter_schema_from_config(config) if config is not None else None
                if parameter_schema is not None:
                    candidate_conditioning(support, parameter_schema)
                _require(list(support.sample_uids) == item["sample_uids"] and support.role == role,
                         "cache joint sample/role mismatch")
                _require(source["particle_programs"]["regimen_uids"] == regimen_uids, "cache joint action order mismatch")
            if action_policy == "scalar_exposure":
                validate_scalar_cache(native)
            if action_policy in {"drug_id", "target_id"} and not joint:
                expected_identity = np.asarray([
                    program_action_identity(record["program"], regimen_uids[index % len(regimen_uids)], action_policy=action_policy)
                    for index, record in enumerate(records)
                ], dtype=np.float32).reshape(len(item["sample_uids"]), len(regimen_uids), -1)
                _require("action_identity" in native and np.array_equal(native["action_identity"], expected_identity),
                         "cache_stale: canonical action identity mismatch")
            _require(len(native.get("trajectory_uids", [])) == count and len(native.get("posterior_uids", [])) == count,
                     "cache native identity cardinality mismatch")
            keys = dict(TEACHER_ARRAY_NAMES if joint else ARRAY_NAMES)
            if native.get("grid", {}).get("quantity_profile"):
                keys["quantity_support"] = "quantity_support"
                actual = np.asarray(native.get("quantity_support"))
                expected_support = np.stack([quantity_support(native["grid"], row["teacher"]["source_uid"]) for row in records]).reshape(np.asarray(native["teacher_trajectory"]).shape)
                _require(actual.dtype == np.bool_ and np.array_equal(actual, expected_support),
                         "cache quantity support/profile mismatch")
                _require((np.asarray(native["teacher_trajectory"])[~actual] == 0).all(),
                         "cache missing quantities require zero storage placeholders")
            for key in ("teacher_trajectory", *[name for name in keys if name != "trajectory"]):
                _require(tuple(np.asarray(native[key]).shape[:2]) == (len(item["sample_uids"]), len(regimen_uids)),
                         f"cache {key}: sample/action axes mismatch")
            for path, expected in native.get("source_bindings", {}).items():
                _require(Path(path).is_file() and hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected,
                         f"cache_stale: native source binding changed: {path}")
            _require(bool(native.get("source_bindings")), "cache native source bindings missing")


def bounded_metadata_report(catalog: Path, max_rows: int, config: dict | None = None) -> dict:
    _require(0 < max_rows <= 256, "metadata row bound must be in 1..256")
    with catalog.open(encoding="utf-8", newline="") as handle:
        rows = list(islice(csv.DictReader(handle, delimiter="\t"), max_rows))
    for row in rows:
        for key, cast in (("teacher_confidence", float), ("trajectory_index", int)):
            try:
                row[key] = cast(row[key])
            except (ValueError, KeyError, TypeError):
                pass  # Preserve nonnumeric source text for explicit rejection below.
    rejected = [{"row_index": index, "trajectory_uid": row.get("trajectory_uid", ""),
                 "reasons": validate_teacher_metadata(row)} for index, row in enumerate(rows)]
    for item, row in zip(rejected, rows):
        if not item["reasons"]:
            try:
                _require(config is not None, "metadata-only: full configured registry joins required")
                registry = load_registry(config)
                records = join_registry(registry, config, [row])
                descriptor_bindings(records, registry, config)
            except (ValueError, OSError) as error:
                item["reasons"].append(str(error))
    return {"catalog": str(catalog), "inspected_rows": len(rows), "admitted_rows": 0, "metadata_qualified_rows": sum(not item["reasons"] for item in rejected),
            "candidate_train_development_rows": sum(row.get("split_name") in {"train", "development"} for row in rows),
            "rejected_rows": sum(bool(item["reasons"]) for item in rejected), "rows": rejected,
            "array_payload_reads": 0, "scope": "metadata only; unseen corpus and scientific evidence not assessed"}


def reindex_registry(registry: dict, records: list[dict], store_path: Path, store_format: str) -> dict:
    """Preserve source indices in provenance while assigning derived output indices."""
    return _reindex_registry(registry, records, store_path, store_format)


def _reindex_registry(registry: dict, records: list[dict], store_path: Path, store_format: str, *,
                      joint_references: dict | None = None) -> dict:
    output = {key: copy.deepcopy(value) for key, value in registry.items() if key not in {"teachers", "programs", "observations", "joint_sources", "joint_source_references"}}
    output.update(teachers=[], programs=[], observations=[])
    # An observation is shared by all actions for its posterior; preserve that join.
    observations = {}
    joint = registry.get("posterior_family") == "finite_joint_particles"
    if joint:
        selected = {record["teacher"]["posterior_uid"]: record["joint_source"] for record in records}
        references = ({row["source_posterior_uid"]: row for row in registry.get("joint_source_references", [])}
                      if joint_references is None else {uid: joint_references[(uid,)] for uid in selected if (uid,) in joint_references})
        output["joint_sources"] = [copy.deepcopy(raw) for uid, raw in selected.items() if uid not in references]
        if references:
            output["joint_source_references"] = [copy.deepcopy(references[uid]) for uid in selected if uid in references]
    for index, record in enumerate(records):
        teacher, program, observation = (copy.deepcopy(record[key]) for key in ("teacher", "program", "observation"))
        if joint:
            teacher = copy.deepcopy(record["teacher"])
            teacher["source_trajectory_index"] = teacher["trajectory_index"]
            teacher["source_trajectory_store_path"] = teacher["trajectory_store_path"]
            teacher.update(trajectory_store_path=str(store_path.resolve()), store_format=store_format, trajectory_index=index)
            output["teachers"].append(teacher)
            continue
        for owner, path_key, index_key in ((teacher, "trajectory_store_path", "trajectory_index"),
                                           (program, "store_path", "program_index"),
                                           (observation, "store_path", "observation_index")):
            owner["source_binding"] = {"store_path": owner[path_key], "store_format": owner["store_format"], "row_index": owner[index_key]}
            owner[path_key], owner["store_format"], owner[index_key] = str(store_path.resolve()), store_format, index
        if observation["posterior_uid"] not in observations:
            observations[observation["posterior_uid"]] = observation
        teacher["source_trajectory_index"] = record["teacher"]["trajectory_index"]
        teacher["source_trajectory_store_path"] = record["teacher"]["trajectory_store_path"]
        output["teachers"].append(teacher)
        output["programs"].append(program)
    output["observations"] = list(observations.values())
    return output


def joint_particle_contexts(posterior, programs: dict, regimen_uids: list[str], *,
                            action_policy: str = "occupancy_program") -> list[dict[str, object]]:
    """Bind supplied numeric programs to whole posterior particles, without broadcasting."""
    if hasattr(posterior, "assert_current_binding"):
        posterior.assert_current_binding()
    support = posterior.support
    b, j = posterior.weights.shape
    _require(isinstance(programs, dict), "Missing particle-aligned programs; numeric PKPD compiler remains required")
    declared = programs.get("regimen_uids", [])
    _require(bool(regimen_uids) and len(set(declared)) == len(declared)
             and all(uid in declared for uid in regimen_uids), "Joint program regimen identity mismatch")
    _require(programs.get("posterior_uid") == posterior.posterior_uid
             and programs.get("support_uid") == support.support_uid
             and programs.get("source_uid") == support.source_uid
             and programs.get("sample_uids") == list(support.sample_uids)
             and programs.get("particle_uids") == [list(row) for row in support.particle_uids],
             "Joint program posterior/particle/source lineage mismatch")
    records = programs.get("records")
    _require(isinstance(records, list) and len(records) == b
             and all(len(row) == j and all(len(actions) == len(declared) for actions in row) for row in records),
             "Joint programs require a complete sample/particle/action record grid")
    # Admit every canonical record before converting any program array.
    identities = posterior.weights.new_tensor(_joint_action_identities(programs, action_policy))
    for bi, row in enumerate(records):
        for ji, actions in enumerate(row):
            for ai, record in enumerate(actions):
                _require(record.get("particle_uid") == support.particle_uids[bi][ji]
                         and record.get("posterior_uid") == programs.get("source_posterior_uid", posterior.posterior_uid)
                         and record.get("source_uid") == support.source_uid,
                         "Joint program record particle/posterior mismatch")
                _require(record["action"] == records[bi][0][ai]["action"],
                         "One candidate action must have identical semantics across particles")
    arrays = {}
    r, c = support.initial_field.shape[2:4]
    program_names = ("occupancy", "support", "physical_time") + (FORCING_ARRAY_NAMES if support.quantity_profile is not None else ())
    for name in program_names:
        _require(programs.get(name) is not None,
                 f"Missing particle-aligned {name}; numeric PKPD compiler remains required")
        value = torch.as_tensor(programs[name], dtype=posterior.weights.dtype, device=posterior.weights.device)
        _require(value.ndim == 7 and value.shape[:3] == (b, j, len(declared))
                 and value.shape[4:] == (r, c, 1) and value.shape[3] >= 2 and torch.isfinite(value).all(),
                 f"Joint {name} requires explicit sample/particle/action/time/grid axes")
        if name in {"occupancy", "support", "electrical_applicability", "optical_applicability"}:
            _require(((value >= 0) & (value <= 1)).all(), f"Joint {name} outside [0,1]")
        if name == "optical_power_mW":
            _require((value >= 0).all(), "Optical power must be nonnegative")
        arrays[name] = value
    indices = [declared.index(uid) for uid in regimen_uids]
    fixed_affinities = {}
    for bi in range(b):
        for ji in range(j):
            for ai, uid in enumerate(declared):
                record = records[bi][ji][ai]
                if "physical_program" in record:
                    physical = record["physical_program"]
                    _require(physical["inputs"]["binding"]["sample_uid"] == support.sample_uids[bi],
                             "Physical program sample binding mismatch")
                    _require(physical["inputs"]["binding"]["particle_basis"]
                             == whole_particle_basis(support.to_payload(), bi, ji),
                             "Physical program whole-particle binding mismatch")
                    for target in physical["target_parameters"]:
                        affinity = target["affinity_binding"]
                        if affinity["kind"] == "fixed_source_affinity":
                            kd = target["kd_source"]
                            key = tuple(kd[name] for name in ("source_uid", "source_snapshot_ref", "site_uid",
                                                             "ligand_uid", "target_uid")) + (affinity["affinity_name"],)
                            _require(key not in fixed_affinities or fixed_affinities[key] == kd,
                                     "Fixed source affinity differs across whole particles")
                            fixed_affinities[key] = kd
                    for name in ("occupancy", "support", "physical_time"):
                        expected = posterior.weights.new_tensor(physical[name])
                        _require(torch.equal(arrays[name][bi, ji, ai], expected),
                                 "Physical program numeric array mismatch: " + name)
                if support.quantity_profile is not None:
                    compiled = validate_compiled_program(record)
                    stimulation = record.get("representation") == "explicit_stimulation_grid"
                    for name in FORCING_ARRAY_NAMES:
                        expected = posterior.weights.new_tensor(compiled["arrays"][name]) if stimulation else torch.zeros_like(arrays[name][bi, ji, ai])
                        _require(torch.equal(arrays[name][bi, ji, ai], expected), "Joint typed forcing mismatch: " + name)
                    if stimulation:
                        _require(record["stimulation_binding"]["particle_basis"] == whole_particle_basis(support.to_payload(), bi, ji),
                                 "Stimulation whole-particle binding mismatch")
                        _require((arrays["occupancy"][bi, ji, ai] == 0).all() and (arrays["support"][bi, ji, ai] == 0).all(),
                                 "Stimulation cannot supply chemical occupancy aliases")
                times = posterior.weights.new_tensor(record["compiled_program"]["time_grid"])
                supplied = arrays["physical_time"][bi, ji, ai]
                _require(times.shape == (supplied.shape[0],)
                         and torch.equal(supplied, times[:, None, None, None].expand_as(supplied)),
                         "Joint program physical time mismatch")
    if action_policy == "scalar_exposure" or any(key in programs for key in ("scalar_exposure", "scalar_exposure_auc", "scalar_exposure_interval")):
        view = scalar_exposure_view(records)
        if "scalar_exposure" in programs:
            verify_scalar_exposure(programs["scalar_exposure"], view)
        verify_scalar_arrays(programs, view)
        if action_policy == "scalar_exposure":
            arrays.update({key: posterior.weights.new_tensor(value) for key, value in scalar_exposure_arrays(view).items()})
    output = []
    for ji in range(j):
        context = {key: value[:, ji, indices] for key, value in arrays.items()}
        context["initial_state"] = support.initial_field[:, ji, None].expand(-1, len(indices), -1, -1, -1)
        context["posterior_context"] = posterior.particle_context[:, ji, None].expand(-1, len(indices), -1, -1, -1)
        context["action_identity"] = identities[:, ji, indices]
        if support.quantity_profile is not None:
            grid = family_grid(support.quantity_profile, support.source_family_uid)
            declared_support = torch.as_tensor(quantity_support(grid, support.source_uid), device=posterior.weights.device)
            context["quantity_support"] = declared_support[None, None].expand(b, len(indices), *declared_support.shape)
        output.append(context)
    return output


def posterior_family(config: dict) -> str:
    architecture = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]
    return architecture.get("world_state_posterior", {}).get("family", "deterministic_compatibility")


def consumed_array_names(registry: dict) -> dict:
    names = dict(TEACHER_ARRAY_NAMES if registry.get("posterior_family") == "finite_joint_particles" else ARRAY_NAMES)
    if registry.get("grid", {}).get("quantity_profile") is not None:
        names["quantity_support"] = "quantity_support"
        if registry.get("posterior_family") != "finite_joint_particles":
            names.update({name: name for name in FORCING_ARRAY_NAMES})
    return names


def _source_digest(source: dict) -> str:
    return hashlib.sha256(json.dumps(source, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _joint_action_identities(programs: dict, action_policy: str) -> list:
    """Validate canonical program metadata before posterior or program tensor reads."""
    _require(isinstance(programs, dict), "Joint source requires particle-aligned program metadata")
    declared, records = programs.get("regimen_uids"), programs.get("records")
    _require(isinstance(declared, list) and bool(declared) and len(set(declared)) == len(declared),
             "Joint program regimen identity mismatch")
    _require(isinstance(records, list) and bool(records)
             and all(isinstance(row, list) and bool(row)
                     and all(isinstance(actions, list) and len(actions) == len(declared) for actions in row)
                     for row in records), "Joint programs require a complete sample/particle/action record grid")
    return [[[program_action_identity(record, uid, action_policy=action_policy)
              for record, uid in zip(actions, declared)] for actions in row] for row in records]


def parse_joint_source(source: dict, device="cpu", *, action_policy: str = "occupancy_program"):
    _require(isinstance(source, dict) and source.get("schema_version") == JOINT_SOURCE_SCHEMA,
             "Raw joint source schema mismatch")
    _require(not {"joint_posterior", "particle_context", "weights", "log_weights", "generating_indices", "generating_fields", "hidden_index", "selected_truth", "future_targets"} & set(source),
             "Model-derived outputs and producer-only latent draws cannot be posterior source inputs")
    _joint_action_identities(source.get("particle_programs"), action_policy)
    support = JointParticleSupport.from_payload(source["support"], device)
    validate_program_parameter_subset(support, source["particle_programs"])
    observations = tuple(observation_from_payload(item, device) for item in source["observations"])
    _require(bool(observations), "Joint source requires declared retained observation records")
    programs = source.get("particle_programs", {})
    _require(isinstance(programs, dict), "Joint source requires particle-aligned program metadata")
    _require(source.get("source_posterior_uid") == programs.get("source_posterior_uid")
             and bool(source.get("source_posterior_uid"))
             and programs.get("support_uid") == support.support_uid
             and programs.get("source_uid") == support.source_uid
             and programs.get("source_snapshot_ref") == support.source_snapshot_ref
             and programs.get("sample_uids") == list(support.sample_uids)
             and programs.get("particle_uids") == [list(row) for row in support.particle_uids],
             "Joint raw source/program basis mismatch")
    for index, observation in enumerate(observations):
        _require(observation.sample_uids == support.sample_uids and observation.source_uid == support.source_uid
                 and observation.source_snapshot_ref == support.source_snapshot_ref and observation.role == support.role,
                 "Joint observation sample/source/role mismatch")
        _require(observation.sequence_number == index, "Joint source observations must retain assimilation order")
        _require(observation.channel_uid in support.readout_parameters, "Joint observation channel binding mismatch")
        # Source parsing can lack the configured occupancy-assay table. Physical
        # acquisition must still be admitted before any downstream encoder.
        if "physical_readout" in support.readout_parameters[observation.channel_uid] or observation.physical_acquisition is not None:
            support.validate_observation(observation, {})
    view = SimpleNamespace(support=support, weights=support.prior_weights,
                           posterior_uid=source["source_posterior_uid"], particle_context=support.initial_field)
    joint_particle_contexts(view, dict(programs, posterior_uid=view.posterior_uid), programs["regimen_uids"], action_policy=action_policy)
    return support, observations


def bind_joint_programs(posterior, source: dict, *, action_policy: str = "occupancy_program") -> dict:
    """Bind fixed conditional programs to new model weights without rewriting compiler records."""
    support, _ = parse_joint_source(source, posterior.weights.device, action_policy=action_policy)
    _require(support.to_payload() == posterior.support.to_payload(), "Live posterior/source support mismatch")
    result = dict(source["particle_programs"], posterior_uid=posterior.posterior_uid)
    if action_policy == "scalar_exposure":
        result["scalar_exposure"] = scalar_exposure_view(result["records"])
    joint_particle_contexts(posterior, result, result["regimen_uids"], action_policy=action_policy)
    return result


def select_joint_source(source: dict, sample_uids: list[str], regimen_uids: list[str] | None = None, *,
                        action_policy: str = "occupancy_program") -> dict:
    """Select/reorder only complete tuples; model outputs are never cached as sources."""
    support, _ = parse_joint_source(source, action_policy=action_policy)
    _require(len(set(sample_uids)) == len(sample_uids)
             and all(uid in support.sample_uids for uid in sample_uids), "Joint source sample selection mismatch")
    selected = copy.deepcopy(source)
    indices = [support.sample_uids.index(uid) for uid in sample_uids]
    raw = selected["support"]
    for key in ("sample_uids", "particle_uids", "regime_ids", "initial_field", "physical_parameters", "prior_weights"):
        raw[key] = [raw[key][index] for index in indices]
    for key in ("factors", "source_nuisance"):
        raw[key] = {name: [value[index] for index in indices] for name, value in raw[key].items()}
    raw["readout_parameters"] = select_readouts(raw["readout_parameters"], indices)
    if raw.get("physical_parameter_hypotheses") is not None:
        raw["physical_parameter_hypotheses"] = select_candidate_samples(support.physical_parameter_hypotheses, indices)
    if raw.get("local_membrane") is not None:
        raw["local_membrane"] = support.local_membrane.select_samples(indices).to_payload()
    for observation in selected["observations"]:
        if observation.get("physical_acquisition") is not None:
            observation["physical_acquisition"] = MembraneAcquisition.from_payload(
                observation["physical_acquisition"]).select_samples(indices).to_payload()
        observation["sample_uids"] = sample_uids
        for key in ("values", "observed", "left_censored", "library_offset", "detection_limit", "admission_log_probability"):
            if observation.get(key) is not None:
                observation[key] = [observation[key][index] for index in indices]
    programs = selected["particle_programs"]
    declared = programs["regimen_uids"]
    chosen = declared if regimen_uids is None else regimen_uids
    _require(chosen and len(set(chosen)) == len(chosen) and all(uid in declared for uid in chosen),
             "Joint source action selection mismatch")
    actions = [declared.index(uid) for uid in chosen]
    programs["sample_uids"] = sample_uids
    programs["particle_uids"] = raw["particle_uids"]
    programs["records"] = [[[row[ai] for ai in actions] for row in programs["records"][bi]] for bi in indices]
    for key in ("occupancy", "support", "physical_time", "scalar_exposure_auc", "scalar_exposure_interval", *FORCING_ARRAY_NAMES):
        if key in programs:
            programs[key] = np.asarray(programs[key])[indices][:, :, actions].tolist()
    if "scalar_exposure" in programs:
        programs["scalar_exposure"] = scalar_exposure_view(programs["records"])
    programs["regimen_uids"] = list(chosen)
    parse_joint_source(selected, action_policy=action_policy)
    return selected


def _join_joint_registry(registry: dict, config: dict, selected: list[dict], sources: dict, *,
                         input_locations: NativeInputLocations | None = None,
                         joint_sources: dict | None = None) -> list[dict]:
    _require(posterior_family(config) == "finite_joint_particles", "Joint registry requires the configured joint family")
    if joint_sources is None:
        joint_sources = load_joint_sources(registry, [row["posterior_uid"] for row in selected], input_locations=input_locations)
    parsed_sources = {}
    parameter_schema = parameter_schema_from_config(config)
    result = []
    for teacher in selected:
        source = sources.get((teacher["source_uid"],), {})
        _require(source.get("public") is True and source.get("protected") is False
                 and all(source.get(key) for key in SOURCE_FIELDS)
                 and source["snapshot_ref"] == teacher["source_snapshot_ref"],
                 "source: public/license/release/snapshot/split/unit evidence required")
        raw = joint_sources[teacher["posterior_uid"]]
        uid = teacher["posterior_uid"]
        if uid not in parsed_sources:
            support, observations = parse_joint_source(raw, action_policy=native_action_policy(config))
            if parameter_schema is not None:
                candidate_conditioning(support, parameter_schema)
            parsed_sources[uid] = (support, observations)
        support, observations = parsed_sources[uid]
        _require(support.source_uid == teacher["source_uid"] and support.source_snapshot_ref == source["snapshot_ref"]
                 and support.role == teacher["split_name"] and teacher["sample_uid"] in support.sample_uids,
                 "Joint teacher/source/sample/role mismatch")
        _require(tuple(support.initial_field.shape[2:]) == tuple(len(registry["grid"][key]) for key in
                 ("region_id", "compartment_id", "species_id"))
                 and list(support.species_units) == [registry["grid"]["species_units"][name] for name in registry["grid"]["species_id"]],
                 "Joint source grid/units mismatch")
        if support.local_membrane is not None:
            _require(list(support.local_membrane.region_ids) == registry["grid"]["region_id"]
                     and list(support.local_membrane.compartment_ids) == registry["grid"]["compartment_id"],
                     "Local membrane anatomical incidence grid identity mismatch")
        for observation in observations:
            support.validate_observation(observation,
                config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"].get("registered_occupancy_assays", {}))
        programs = raw["particle_programs"]
        if native_action_policy(config) == "scalar_exposure":
            scalar_exposure_definition(config)
            scalar_exposure_view(programs["records"])
            for sample in programs["records"]:
                for particle in sample:
                    for record in particle:
                        inputs = record["physical_program"]["inputs"]
                        _require(inputs["region_ids"] == registry["grid"]["region_id"]
                                 and inputs["compartment_ids"] == registry["grid"]["compartment_id"],
                                 "scalar exposure source site order mismatch")
        bi = support.sample_uids.index(teacher["sample_uid"])
        _require(teacher["regimen_uid"] in programs["regimen_uids"], "Joint teacher action is missing")
        ai = programs["regimen_uids"].index(teacher["regimen_uid"])
        program = programs["records"][bi][0][ai]
        _require(all(program.get(key) == teacher[key] for key in
                 ("program_uid", "posterior_uid", "source_uid", "action_uid", "regimen_uid", "target_program_uid")),
                 "Joint teacher compiled program identity mismatch")
        _require(isinstance(teacher.get("trajectory_index"), int) and teacher["trajectory_index"] >= 0
                 and teacher["store_format"] in {"zarr_v2", "neuronsifter_chunk_store"}, "Joint teacher source index/format mismatch")
        # Every particle/action shares the admitted teacher's physical-time grid.
        times = np.asarray(registry["grid"]["time_hours"], dtype=np.float32).reshape(1,1,1,-1,1,1,1)
        supplied_times = np.asarray(programs["physical_time"], dtype=np.float32)
        _require(supplied_times.shape[3] == times.shape[3] and np.all(supplied_times == times),
                 "Joint program/teacher time grid mismatch")
        result.append({"teacher": teacher, "program": program, "observation": {"posterior_uid": teacher["posterior_uid"]},
                       "joint_source": raw})
    return result
