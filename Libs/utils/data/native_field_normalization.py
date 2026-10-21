"""Training-only affine native field coordinates, bound to source and Haar support."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from Libs.models.modules.neuronsifter_world_action import GraphSpectralTemporalWaveletTokenizer
from Libs.utils.data.native_observable_profile import coefficient_support, observable_profile, quantity_support
from Libs.utils.data.raw_data_registry import resolve_repo_path, load_config
from Libs.utils.data.native_input_staging import NativeInputLocations, input_locations_from_config
from Libs.utils.model.screening_model_config import quantized_mddm_rollout_from_config, native_direct_mean_config

SCHEMA = "native_field_normalization.v1"
RULE = "qc_weighted_population_sd_with_complete_particle_haar_support"
MULTIGRID_SCHEMA = "native_field_normalization.v2"
MULTIGRID_RULE = "qc_weighted_supported_cells_with_complete_particle_haar_support"
DOMAIN_KEYS = ("region_id", "compartment_id", "species_id", "species_units", "time_hours", "physical_time_unit")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("native field normalization: " + message)


def source_domain(grid: dict) -> dict:
    keys = DOMAIN_KEYS
    if "quantity_profile" in grid:
        keys += ("quantity_profile", "quantity_profile_sha256", "family_uid", "query_time")
    return {key: copy.deepcopy(grid[key]) for key in keys}


def representation(config: dict, *, input_locations: NativeInputLocations | None = None) -> dict:
    input_locations = input_locations or input_locations_from_config(config)
    direct = native_direct_mean_config(config)
    if direct:
        # The direct mean shares the already fitted native coordinates; its own
        # operator has no Haar/codebook module and cannot redefine that transform.
        common = (input_locations.conditioning_config(direct["conditioning_config_path"]) if input_locations
                  else load_config(resolve_repo_path(direct["conditioning_config_path"])))
        common_backbone = common["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
        require(common_backbone["kind"] == "neuronsifter_world_action"
                and not common_backbone.get("native_direct_mean", {}).get("enabled", False),
                "direct mean requires the original native conditioning representation")
        return representation(common, input_locations=input_locations)
    backbone = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    token = quantized_mddm_rollout_from_config(config, backbone)["residual_tokenization"]
    low, high = float(token["residual_min_value"]), float(token["residual_max_value"])
    require(low < 0 < high and np.isfinite([low, high]).all(), "codebook must straddle zero")
    levels = int(backbone["temporal_wavelet_tokenizer"]["wavelet_level_count"])
    require(levels >= 1 and int(token["codebook_size"]) >= 4, "invalid Haar/codebook cardinality")
    return {"wavelet_levels": levels,
            "boundary": "repeat_last_then_crop", "coefficient_order": "approximation_reversed_details",
            "codebook_size": int(token["codebook_size"]), "codebook_min": low, "codebook_max": high}


def pack_field(field: torch.Tensor, tokenizer) -> tuple[torch.Tensor, dict]:
    payload = tokenizer.temporal_wavelet_analysis(field.movedim(-4, -2))
    pieces = [payload["approximation"], *reversed(payload["details"])]
    return torch.cat(pieces, -2).movedim(-2, -4), {
        "lengths": payload["lengths"], "sizes": [part.shape[-2] for part in pieces]}


def fit_haar(field: torch.Tensor, levels: int) -> torch.Tensor:
    # Reuse the maintained analysis implementation, including odd-length padding.
    tokenizer = SimpleNamespace(wavelet_level_count=levels)
    tokenizer.temporal_wavelet_analysis = lambda value: GraphSpectralTemporalWaveletTokenizer.temporal_wavelet_analysis(tokenizer, value)
    return pack_field(field, tokenizer)[0]


def _fit_inputs(grid: dict, records: list[dict], arrays: dict, config: dict) -> tuple:
    """Input records have passed registry/descriptor admission; fields remain raw."""
    require(records and all(row["teacher"]["split_name"] == "train"
            and row["teacher"]["source_holdout_role"] == "train"
            and row["teacher"]["qc_grade"] in {"A", "B"} for row in records), "fit accepts training A/B records only")
    require(grid["normalization"]["kind"] == "physical"
            and grid["normalization"].get("stored_field_coordinates", "source") == "source",
            "fit requires explicitly raw physical/source coordinates")
    # Delayed import avoids the shared registry -> normalization import cycle.
    from Libs.utils.data.native_diffusion_data import validate_grid
    validate_grid(grid, config)
    identities, seen = [], set()
    weights = []
    initials = []
    for index, row in enumerate(records):
        teacher = row["teacher"]
        key = (teacher["source_uid"], teacher["trajectory_store_path"], teacher["trajectory_index"])
        require(key not in seen, "duplicate source training row")
        seen.add(key)
        weight = float(teacher["sample_weight"])
        require(weight == {"A": 1., "B": .5}[teacher["qc_grade"]], "invalid QC fit weight")
        weights.append(weight)
        identity = {name: teacher[name] for name in (
            "trajectory_uid", "source_uid", "sample_uid", "regimen_uid", "posterior_uid", "family_uid",
            "content_hash", "trajectory_store_path", "trajectory_index", "source_snapshot_ref",
            "split_name", "source_holdout_role", "qc_grade", "sample_weight")}
        if "joint_source" in row:
            support = row["joint_source"]["support"]
            bi = support["sample_uids"].index(teacher["sample_uid"])
            initial = np.asarray(support["initial_field"][bi], dtype=np.float64)
            identity["initial_binding"] = {"support_uid": support["support_uid"],
                "particle_uids": support["particle_uids"][bi],
                "source_posterior_uid": row["joint_source"]["source_posterior_uid"],
                "initial_sha256": hashlib.sha256(initial.tobytes()).hexdigest()}
        else:
            initial = np.asarray(arrays["initial_state"][index:index+1], dtype=np.float64)
            identity["initial_binding"] = {name: row["observation"][name] for name in
                ("posterior_uid", "source_uid", "observation_ref", "store_path", "observation_index")}
        require(initial.ndim == 4 and np.isfinite(initial).all(), "finite complete initial field required")
        initials.append(initial)
        identities.append(identity)
    require(len({row["trajectory_uid"] for row in identities}) == len(records)
            and len({row["content_hash"] for row in identities}) == len(records), "duplicate training identity/content")
    x = np.asarray(arrays["trajectory"], dtype=np.float64)
    shape = tuple(len(grid[key]) for key in ("time_hours", "region_id", "compartment_id", "species_id"))
    require(x.shape == (len(records), *shape) and np.isfinite(x).all(), "finite dense source fields required")
    w = np.asarray(weights)
    weight_sum = float(w.sum())
    require(np.isfinite(w).all() and weight_sum > 0, "positive total training weight required")
    return x, w, initials, identities, shape


def fit_transform(grid: dict, records: list[dict], arrays: dict, config: dict) -> dict:
    """Fit the established single-grid dense transform."""
    require(not grid.get("quantity_profile"), "observable profiles require fit_multigrid over all registered families")
    require(config["native_diffusion_data"]["field_normalization"]["fit_rule"] == RULE,
            "unsupported configured fit rule")
    x, w, initials, identities, shape = _fit_inputs(grid, records, arrays, config)
    weight_sum = float(w.sum())
    row_means = x.mean(axis=(1, 2, 3))
    mean = (w[:, None] * row_means).sum(0) / weight_sum
    variance = (w[:, None] * ((x - mean) ** 2).mean(axis=(1, 2, 3))).sum(0) / weight_sum
    std = np.sqrt(variance)
    rep = representation(config)
    positive, negative = np.zeros(shape[-1]), np.zeros(shape[-1])
    for index, initial in enumerate(initials):
        require(initial.shape[1:] == shape[1:], "initial field grid mismatch")
        coefficients = fit_haar(torch.from_numpy(x[index][None] - initial[:, None]), rep["wavelet_levels"]).numpy()
        positive = np.maximum(positive, coefficients.reshape(-1, shape[-1]).max(0))
        negative = np.maximum(negative, -coefficients.reshape(-1, shape[-1]).min(0))
    support_scale = np.maximum(positive / rep["codebook_max"], negative / -rep["codebook_min"])
    scale = np.maximum(std, support_scale)
    constant = scale == 0
    scale[constant] = 1.
    return {"schema_version": SCHEMA, "fit_role": "train", "rule": RULE, "domain": source_domain(grid),
        "representation": rep, "training_records": identities, "trajectory_count": len(records),
        "weight_sum": weight_sum, "entries_per_trajectory": int(np.prod(shape[:-1])),
        "weighted_sum": (mean * weight_sum).tolist(), "weighted_m2": (variance * weight_sum).tolist(),
        "mean": mean.tolist(), "population_sd": std.tolist(), "scale": scale.tolist(), "constant_channels": constant.tolist(),
        "source_min": x.reshape(-1, shape[-1]).min(0).tolist(), "source_max": x.reshape(-1, shape[-1]).max(0).tolist(),
        "haar_positive_max": positive.tolist(), "haar_negative_max": negative.tolist(),
        "support_scale": support_scale.tolist(), "stored_field_coordinates": "source",
        "model_field_coordinates": "affine_native", "output_field_coordinates": "source"}



def fit_multigrid(batches, config: dict, *, streaming: bool = False) -> dict:
    """One training-only transform across registered clocks and supported quantities."""
    profile = observable_profile(config)
    require(profile is not None and batches, "multigrid fitting requires the explicit observable profile")
    require(config["native_diffusion_data"]["field_normalization"]["fit_rule"] == MULTIGRID_RULE,
            "unsupported configured multigrid fit rule")
    if not streaming:
        family_ids = [grid.get("family_uid") for grid, _, _ in batches]
        require(len(set(family_ids)) == len(family_ids) and set(family_ids) == set(profile["families"]),
                "fit requires every registered family exactly once")
    rep = representation(config)
    width = len(profile["quantities"])
    prepared, identities, domains = [], [], []
    grids = {}
    running_mean = np.zeros(width, dtype=np.float64)
    running_m2 = np.zeros(width, dtype=np.float64)
    count, weighted_count, total = (np.zeros(width, dtype=np.float64) for _ in range(3))
    low, high = np.full(width, np.inf), np.full(width, -np.inf)
    positive, negative = np.zeros(width), np.zeros(width)
    for grid, records, arrays in batches:
        x, w, initials, rows, shape = _fit_inputs(grid, records, arrays, config)
        supports = np.stack([quantity_support(grid, row["source_uid"]) for row in rows])
        actual = np.asarray(arrays.get("quantity_support"))
        require(actual.dtype == np.bool_ and actual.shape == x.shape
                and np.array_equal(actual, supports),
                "fit needs the exact declared quantity support")
        require((x[~actual] == 0).all(), "unsupported values must be zero storage placeholders")
        cell_counts = supports.sum(axis=(1, 2, 3))
        for row, cell_count in zip(rows, cell_counts):
            row["grid_family_uid"] = grid["family_uid"]
            row["supported_cell_count"] = cell_count.tolist()
        identities.extend(rows)
        domain = source_domain(grid)
        family = grid["family_uid"]
        require(family not in grids or source_domain(grids[family]) == domain, "family grid changed during fitting")
        if family not in grids:
            grids[family] = grid
            domains.append(domain)
        count += cell_counts.sum(0)
        batch_count = (cell_counts * w[:, None]).sum(0)
        batch_total = (x * actual * w[:, None, None, None, None]).sum(axis=(0, 1, 2, 3))
        if streaming:
            # Parallel weighted central-moment merge is one-pass and needs only
            # this complete group's fields, including every prior initial state.
            batch_mean = np.divide(batch_total, batch_count, out=np.zeros(width), where=batch_count > 0)
            batch_m2 = (((x - batch_mean)**2) * actual * w[:, None, None, None, None]).sum(axis=(0, 1, 2, 3))
            combined = weighted_count + batch_count
            delta = batch_mean - running_mean
            ratio = np.divide(batch_count, combined, out=np.zeros(width), where=combined > 0)
            running_m2 += batch_m2 + delta**2 * weighted_count * ratio
            running_mean += delta * ratio
        weighted_count += batch_count
        total += batch_total
        low = np.minimum(low, np.where(actual, x, np.inf).min(axis=(0, 1, 2, 3)))
        high = np.maximum(high, np.where(actual, x, -np.inf).max(axis=(0, 1, 2, 3)))
        for index, initial in enumerate(initials):
            support = supports[index]
            packed_support = coefficient_support(torch.from_numpy(support), rep["wavelet_levels"]).numpy()
            require(initial.shape[1:] == shape[1:] and (initial[:, ~support[0]] == 0).all(),
                    "complete prior initial fields must share the declared support")
            coefficients = fit_haar(torch.from_numpy(x[index][None] - initial[:, None]),
                                    rep["wavelet_levels"]).numpy()
            coefficients = np.where(packed_support[None], coefficients, 0.)
            positive = np.maximum(positive, coefficients.reshape(-1, width).max(0))
            negative = np.maximum(negative, -coefficients.reshape(-1, width).min(0))
        if not streaming:
            prepared.append((x, w, actual))
    require(set(grids) == set(profile["families"]), "fit requires every registered family")
    require((weighted_count > 0).all(), "every declared quantity needs supported training observations")
    mean = total / weighted_count
    m2 = running_m2 if streaming else np.zeros(width)
    for x, w, support in prepared:
        m2 += (((x - mean)**2) * support * w[:, None, None, None, None]).sum(axis=(0, 1, 2, 3))
    std = np.sqrt(m2 / weighted_count)
    support_scale = np.maximum(positive / rep["codebook_max"], negative / -rep["codebook_min"])
    scale = np.maximum(std, support_scale)
    constant = scale == 0
    scale[constant] = 1.
    common = {key: value for key, value in domains[0].items()
              if key not in {"time_hours", "family_uid", "query_time"}}
    artifact = {"schema_version": MULTIGRID_SCHEMA, "fit_role": "train", "rule": MULTIGRID_RULE,
        "domain": common, "registered_grids": domains, "representation": rep,
        "training_records": identities, "trajectory_count": len(identities),
        "weight_sum": sum(row["sample_weight"] for row in identities),
        "supported_cell_count": count.tolist(), "weighted_supported_cell_count": weighted_count.tolist(),
        "weighted_sum": total.tolist(), "weighted_m2": m2.tolist(),
        "mean": mean.tolist(), "population_sd": std.tolist(), "scale": scale.tolist(),
        "constant_channels": constant.tolist(), "source_min": low.tolist(), "source_max": high.tolist(),
        "haar_positive_max": positive.tolist(), "haar_negative_max": negative.tolist(),
        "support_scale": support_scale.tolist(), "stored_field_coordinates": "source",
        "model_field_coordinates": "affine_native", "output_field_coordinates": "source"}
    for grid in grids.values():
        validate_artifact(artifact, grid, rep)
    return artifact

def validate_artifact(artifact: dict, grid: dict, rep: dict) -> None:
    multigrid = artifact.get("schema_version") == MULTIGRID_SCHEMA
    require((artifact.get("schema_version"), artifact.get("rule")) in {
                (SCHEMA, RULE), (MULTIGRID_SCHEMA, MULTIGRID_RULE)}
            and artifact.get("fit_role") == "train", "unsupported fit artifact/rule/role")
    if multigrid:
        domains = artifact.get("registered_grids", [])
        require(source_domain(grid) in domains and domains, "unregistered normalization grid")
        common = artifact.get("domain", {})
        require(all({key: value for key, value in domain.items()
                     if key not in {"time_hours", "family_uid", "query_time"}} == common for domain in domains),
                "incompatible registered quantity domains")
        families = common.get("quantity_profile", {}).get("families", {})
        require(len(domains) == len(families)
                and {domain.get("family_uid") for domain in domains} == set(families),
                "normalization must register all family clocks")
        for domain in domains:
            quantity_support(domain)
    else:
        require(artifact.get("domain") == source_domain(grid), "source domain binding mismatch")
    require(artifact.get("representation") == rep, "Haar/codebook binding mismatch")
    require(artifact.get("stored_field_coordinates") == artifact.get("output_field_coordinates") == "source"
            and artifact.get("model_field_coordinates") == "affine_native", "ambiguous field coordinates")
    s = len(grid["species_id"])
    values = {}
    for key in ("mean", "population_sd", "scale", "source_min", "source_max", "weighted_sum", "weighted_m2",
                "haar_positive_max", "haar_negative_max", "support_scale"):
        value = np.asarray(artifact.get(key), dtype=np.float64)
        require(value.shape == (s,) and np.isfinite(value).all(), "missing/invalid numerical " + key)
        values[key] = value
    constant = np.asarray(artifact.get("constant_channels"))
    require(constant.shape == (s,) and constant.dtype == np.bool_, "constant-channel flags required")
    weight = artifact.get("weight_sum", 0)
    rows = artifact.get("training_records", [])
    require(isinstance(weight, (int, float)) and np.isfinite(weight) and weight > 0 and rows
            and artifact.get("trajectory_count") == len(rows)
            and all(row.get("split_name") == row.get("source_holdout_role") == "train"
                    and row.get("qc_grade") in {"A", "B"} and row.get("initial_binding") for row in rows),
            "invalid training-only fit identity")
    require(len({row["trajectory_uid"] for row in rows}) == len(rows)
            and len({row["content_hash"] for row in rows}) == len(rows), "duplicate fit identity")
    expected_weight = sum({"A": 1., "B": .5}[row["qc_grade"]] for row in rows)
    require(weight == expected_weight and all(row["sample_weight"] == {"A": 1., "B": .5}[row["qc_grade"]] for row in rows),
            "QC fit weights mismatch")
    denominator = weight
    if multigrid:
        family_domains = {domain["family_uid"]: domain for domain in domains}
        count, denominator = np.zeros(s), np.zeros(s)
        support_counts = {}
        for row in rows:
            family = row.get("grid_family_uid")
            require(family in family_domains, "fit row family is unregistered")
            key = (family, row["source_uid"])
            if key not in support_counts:
                support_counts[key] = quantity_support(family_domains[family], row["source_uid"]).sum(axis=(0, 1, 2))
            cells = support_counts[key]
            require(row.get("supported_cell_count") == cells.tolist(), "fit row support denominator mismatch")
            count += cells
            denominator += cells * row["sample_weight"]
        require(np.array_equal(np.asarray(artifact.get("supported_cell_count")), count)
                and np.array_equal(np.asarray(artifact.get("weighted_supported_cell_count")), denominator)
                and (denominator > 0).all(), "fit supported-cell denominator mismatch")
    else:
        require(artifact.get("entries_per_trajectory") == len(grid["time_hours"])*len(grid["region_id"])*len(grid["compartment_id"]),
                "fit denominator mismatch")
    for key in ("population_sd", "weighted_m2", "haar_positive_max", "haar_negative_max", "support_scale"):
        require((values[key] >= 0).all(), "negative " + key)
    eps = 32*np.finfo(np.float64).eps
    require(np.allclose(values["mean"], values["weighted_sum"]/denominator, rtol=eps, atol=0)
            and np.allclose(values["population_sd"]**2, values["weighted_m2"]/denominator, rtol=eps, atol=0),
            "numerical sufficient-statistic mismatch")
    support_scale = np.maximum(values["haar_positive_max"]/rep["codebook_max"], values["haar_negative_max"]/-rep["codebook_min"])
    expected_scale = np.maximum(values["population_sd"], support_scale)
    require(np.array_equal(values["support_scale"], support_scale) and np.array_equal(constant, expected_scale == 0),
            "residual support/constant-channel mismatch")
    expected_scale[constant] = 1.
    require(np.array_equal(values["scale"], expected_scale) and (values["scale"] > 0).all()
            and (values["source_min"] <= values["mean"]).all() and (values["mean"] <= values["source_max"]).all(),
            "merged scale/range mismatch")



class NativeNormalizationContext:
    """One admission-call snapshot; selectors may change, scientific config may not.

    Hash, parse and complete TRAIN support validation happen once per artifact.
    Each use checks the exact file metadata and registered grid. verify() hashes
    bytes again before a successful call closes, including unchanged-stat edits.
    No process/global cache is retained.
    """
    def __init__(self, config: dict, *, input_locations: NativeInputLocations | None = None):
        self._config = self._scientific_config(config)
        self._locations = input_locations or input_locations_from_config(config)
        self._representation = representation(config, input_locations=self._locations)
        self._specs = {}
        self._files = {}
        self._training = None
        self._artifact_digests = {}

    @staticmethod
    def _scientific_config(config):
        value = copy.deepcopy(config)
        # The two dataset adapters select the same admitted registry by family
        # and descriptor. They cannot change source roles or normalization policy.
        for key in ("registry_path", "source_family_uid"):
            value.get("native_diffusion_data", {}).pop(key, None)
        return value

    @staticmethod
    def _stat(path):
        value = path.stat()
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    def _check_config(self, config, locations):
        require(self._scientific_config(config) == self._config,
                "normalization context scientific configuration changed")
        require(locations is self._locations or (locations is None and self._locations is None),
                "normalization context input locations changed")

    def _read(self, path, digest=None):
        physical = self._locations.file_path(path) if self._locations else path
        before = self._stat(physical)
        data = self._locations.read_bytes(path) if self._locations else physical.read_bytes()
        require(self._stat(physical) == before, "artifact changed while reading")
        actual = hashlib.sha256(data).hexdigest()
        require(digest is None or actual == digest, "artifact bytes changed")
        self._files[str(path)] = (physical, before, actual)
        return data, actual

    def assert_unchanged(self, *, verify_bytes=False):
        for physical, expected, digest in self._files.values():
            require(self._stat(physical) == expected, "artifact bytes changed during normalization context")
            if verify_bytes:
                require(hashlib.sha256(physical.read_bytes()).hexdigest() == digest,
                        "artifact bytes changed during normalization context")
                require(self._stat(physical) == expected, "artifact changed while verifying")

    def verify(self):
        self.assert_unchanged(verify_bytes=True)
        for key, spec in self._specs.items():
            require(hashlib.sha256(json.dumps(spec["artifact"], sort_keys=True, allow_nan=False).encode()).hexdigest()
                    == self._artifact_digests[key], "normalization context artifact was mutated")

    def spec(self, grid, config, *, input_locations=None, artifact_data=None):
        locations = input_locations if input_locations is not None else self._locations
        self._check_config(config, locations)
        self.assert_unchanged()
        convention = grid.get("normalization", {})
        if convention.get("kind") == "physical":
            return normalization_spec(grid, config, input_locations=locations)
        require(convention.get("kind") == "training_scaler" and convention.get("fit_role") == "train"
                and convention.get("stored_field_coordinates") == "source"
                and convention.get("artifact_ref") and convention.get("sha256"),
                "missing numerical training_scaler binding or stored source coordinates")
        path = locations.logical_path(convention["artifact_ref"]) if locations else resolve_repo_path(convention["artifact_ref"])
        key = (str(path), convention["sha256"])
        if key not in self._specs:
            data = artifact_data if artifact_data is not None else self._read(path, convention["sha256"])[0]
            artifact = json.loads(data)
            validate_artifact(artifact, grid, self._representation)
            selected = config["native_diffusion_data"]["field_normalization"].get("model_artifact_ref")
            spec = {"kind": "training_scaler", "artifact_ref": str(path),
                    "sha256": convention["sha256"], "artifact": artifact}
            if selected is not None:
                from Libs.utils.model.native_scientific_source import training_data_scope, selected_group_definitions
                require(config.get("native_scientific_training", {}).get("source_scope", "all") != "all",
                        "a model normalizer override requires an explicit isolated source scope")
                selected_path = locations.logical_path(selected) if locations else resolve_repo_path(selected)
                selected_data, selected_digest = self._read(selected_path)
                fitted = json.loads(selected_data)
                require(fitted.get("training_data_scope") == training_data_scope(config),
                        "model normalizer training source/group scope differs")
                require({row["sample_uid"] for row in fitted["training_records"]} ==
                        {row["sample_uid"] for row in selected_group_definitions(config, "train")},
                        "model normalizer must use exactly the selected complete training groups")
                require(fitted.get("source_grid_normalization_sha256") == convention["sha256"],
                        "model normalizer is bound to another source-store grid identity")
                validate_artifact(fitted, grid, self._representation)
                spec = {"kind": "training_scaler", "artifact_ref": str(selected_path),
                        "sha256": selected_digest, "artifact": fitted,
                        "source_grid_normalization_sha256": convention["sha256"]}
            self._specs[key] = spec
            self._artifact_digests[key] = hashlib.sha256(json.dumps(
                spec["artifact"], sort_keys=True, allow_nan=False).encode()).hexdigest()
        spec = self._specs[key]
        artifact = spec["artifact"]
        require(source_domain(grid) in artifact.get("registered_grids", [artifact["domain"]]),
                "unregistered normalization grid")
        return spec

    def bind(self, grid, path, config):
        self._check_config(config, self._locations)
        logical = self._locations.logical_path(path) if self._locations else path.resolve()
        data = None
        if str(logical) not in self._files:
            data, digest = self._read(logical)
        else:
            self.assert_unchanged()
            digest = self._files[str(logical)][2]
        bound = copy.deepcopy(grid)
        bound["normalization"] = {"kind": "training_scaler", "fit_role": "train",
            "stored_field_coordinates": "source", "artifact_ref": str(logical), "sha256": digest}
        self.spec(bound, config, input_locations=self._locations, artifact_data=data)
        return bound

    def bind_training_registries(self, registries):
        require(len(self._specs) == 1, "retained intake requires one fitted source normalizer")
        artifact = next(iter(self._specs.values()))["artifact"]
        fitted = {row["trajectory_uid"]: row for row in artifact["training_records"]}
        expected = {}
        for registry in registries.values():
            for row in registry["teachers"]:
                if row["split_name"] == "train":
                    require(row["trajectory_uid"] not in expected, "duplicate retained training row")
                    expected[row["trajectory_uid"]] = row
        require(set(fitted) == set(expected), "retained normalizer complete TRAIN rows differ")
        for uid, row in expected.items():
            fields = ("trajectory_uid", "source_uid", "sample_uid", "regimen_uid", "posterior_uid", "family_uid",
                      "content_hash", "trajectory_store_path", "trajectory_index", "source_snapshot_ref",
                      "split_name", "source_holdout_role", "qc_grade", "sample_weight")
            require(all(fitted[uid].get(key) == row.get(key) for key in fields),
                    "retained normalizer training source/role identity differs")
        self._training = fitted

    def assert_training_records(self, records):
        if self._training is None:
            return
        for record in records:
            teacher = record["teacher"]
            if teacher["split_name"] != "train":
                continue
            row = self._training[teacher["trajectory_uid"]]
            support = record["joint_source"]["support"]
            bi = support["sample_uids"].index(teacher["sample_uid"])
            initial = np.asarray(support["initial_field"][bi], dtype=np.float64)
            binding = {"support_uid": support["support_uid"], "particle_uids": support["particle_uids"][bi],
                "source_posterior_uid": record["joint_source"]["source_posterior_uid"],
                "initial_sha256": hashlib.sha256(initial.tobytes()).hexdigest()}
            require(row["initial_binding"] == binding, "retained normalizer initial support differs")


def normalization_spec(grid: dict, config: dict, *, input_locations: NativeInputLocations | None = None,
                       normalization_context: NativeNormalizationContext | None = None) -> dict:
    if normalization_context is not None:
        return normalization_context.spec(grid, config, input_locations=input_locations)
    input_locations = input_locations or input_locations_from_config(config)
    convention = grid.get("normalization", {})
    if convention.get("kind") == "physical":
        require(convention.get("stored_field_coordinates", "source") == "source", "physical fields must be source coordinates")
        return {"kind": "physical", **({"quantity_profile": copy.deepcopy(grid["quantity_profile"])}
                                     if grid.get("quantity_profile") else {})}
    require(convention.get("kind") == "training_scaler" and convention.get("fit_role") == "train"
            and convention.get("stored_field_coordinates") == "source"
            and convention.get("artifact_ref") and convention.get("sha256"),
            "missing numerical training_scaler binding or stored source coordinates; rebuild from explicit raw source")
    path = input_locations.logical_path(convention["artifact_ref"]) if input_locations else resolve_repo_path(convention["artifact_ref"])
    physical_path = input_locations.file_path(path) if input_locations else path
    require(physical_path.is_file(), "missing fitted artifact")
    data = physical_path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == convention["sha256"], "artifact bytes changed")
    artifact = json.loads(data)
    validate_artifact(artifact, grid, representation(config, input_locations=input_locations))
    selected = config["native_diffusion_data"]["field_normalization"].get("model_artifact_ref")
    if selected is not None:
        from Libs.utils.model.native_scientific_source import training_data_scope, selected_group_definitions
        require(config.get("native_scientific_training",{}).get("source_scope","all") != "all",
                "a model normalizer override requires an explicit isolated source scope")
        selected_path = input_locations.logical_path(selected) if input_locations else resolve_repo_path(selected)
        selected_data = input_locations.read_bytes(selected_path) if input_locations else selected_path.read_bytes()
        fitted = json.loads(selected_data)
        require(fitted.get("training_data_scope") == training_data_scope(config),
                "model normalizer training source/group scope differs")
        expected_groups = {row["sample_uid"] for row in selected_group_definitions(config,"train")}
        require({row["sample_uid"] for row in fitted["training_records"]} == expected_groups,
                "model normalizer must use exactly the selected complete training groups")
        require(fitted.get("source_grid_normalization_sha256") == convention["sha256"],
                "model normalizer is bound to another source-store grid identity")
        validate_artifact(fitted, grid, representation(config, input_locations=input_locations))
        return {"kind":"training_scaler", "artifact_ref":str(selected_path),
                "sha256":hashlib.sha256(selected_data).hexdigest(), "artifact":fitted,
                "source_grid_normalization_sha256":convention["sha256"]}
    return {"kind": "training_scaler", "artifact_ref": str(path), "sha256": convention["sha256"],
            "artifact": artifact}


def normalization_from_config(config: dict) -> dict:
    path = config.get("native_diffusion_data", {}).get("registry_path")
    if not path:
        return {"kind": "physical"}
    # Delayed import avoids the registry -> normalization dependency cycle.
    from Libs.utils.data.native_diffusion_data import load_family_registries
    input_locations = input_locations_from_config(config)
    registries = load_family_registries(config, input_locations=input_locations)
    specs = [normalization_spec(registry["grid"], config, input_locations=input_locations) for registry in registries.values()]
    identities = {(spec["kind"], spec.get("sha256")) for spec in specs}
    require(len(identities) == 1, "every family must use the same shared training transform")
    return specs[0]


def bind_artifact(grid: dict, path: Path, config: dict, *, normalization_context: NativeNormalizationContext | None = None) -> dict:
    if normalization_context is not None:
        return normalization_context.bind(grid, path, config)
    bound = copy.deepcopy(grid)
    bound["normalization"] = {"kind": "training_scaler", "fit_role": "train", "stored_field_coordinates": "source",
        "artifact_ref": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    normalization_spec(bound, config)
    return bound


class NativeFieldNormalization(nn.Module):
    """Fixed constants saved with the model; loading a different transform rejects."""
    def __init__(self, spec: dict | None, species_count: int, *,
                 input_locations: NativeInputLocations | None = None):
        super().__init__()
        self.spec = copy.deepcopy(spec or {"kind": "physical"})
        require(self.spec.get("kind") in {"physical","training_scaler"}, "explicit model coordinate convention required")
        self.active = self.spec["kind"] == "training_scaler"
        self.identity = self.spec.get("sha256", "physical")
        self.artifact = self.spec.get("artifact", {})
        self.quantity_profile = copy.deepcopy(self.spec.get("quantity_profile",
            self.artifact.get("domain", {}).get("quantity_profile")))
        if self.active:
            data = (input_locations.read_bytes(self.spec["artifact_ref"]) if input_locations
                    else resolve_repo_path(self.spec["artifact_ref"]).read_bytes())
            require(hashlib.sha256(data).hexdigest() == self.identity and json.loads(data) == self.artifact,
                    "model numerical artifact mismatch")
            validate_artifact(self.artifact, self.artifact.get("registered_grids", [self.artifact["domain"]])[0],
                              self.artifact["representation"])
            require(len(self.artifact["mean"]) == species_count, "model species width mismatch")
        for name, default in (("mean", 0.), ("scale", 1.), ("source_min", 0.), ("source_max", 0.)):
            self.register_buffer(name, torch.tensor(self.artifact[name] if self.active else [default]*species_count,
                                                   dtype=torch.float64), persistent=self.active)
        self.register_buffer("constant", torch.tensor(self.artifact["constant_channels"] if self.active else [False]*species_count),
                             persistent=self.active)
        self.register_buffer("binding_digest", torch.tensor(list(bytes.fromhex(self.identity)) if self.active else [], dtype=torch.uint8),
                             persistent=self.active)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for name in ("mean", "scale", "source_min", "source_max", "constant", "binding_digest"):
            key = prefix + name
            if self.active:
                require(key in state_dict and torch.equal(state_dict[key].cpu(), getattr(self, name).detach().cpu()),
                        "checkpoint numerical transform mismatch: " + name)
            else:
                require(key not in state_dict, "normalized checkpoint cannot load in physical mode")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _apply_support(self, value: torch.Tensor, support: torch.Tensor | None) -> torch.Tensor:
        require(support is not None or self.quantity_profile is None,
                "observable normalization requires explicit quantity support")
        if support is None:
            return value
        require(support.dtype == torch.bool, "normalization support must be boolean")
        return value.masked_fill(~torch.broadcast_to(support.to(value.device), value.shape), 0.)

    def normalize(self, value: torch.Tensor, support: torch.Tensor | None = None) -> torch.Tensor:
        result = (value-self.mean.to(value))/self.scale.to(value) if self.active else value
        return self._apply_support(result, support)

    def inverse(self, value: torch.Tensor, support: torch.Tensor | None = None) -> torch.Tensor:
        result = self.mean.to(value)+self.scale.to(value)*value if self.active else value
        return self._apply_support(result, support)

    def assert_ids(self, values) -> None:
        identities = values if isinstance(values, (list, tuple)) else [values or "physical"]
        require(bool(identities) and all(value == self.identity for value in identities),
                "batch/model transform mismatch")

    def assert_support(self, support) -> None:
        require(getattr(support, "quantity_profile", None) == self.quantity_profile,
                "whole-particle quantity profile mismatch")
        if self.active:
            domain = self.artifact["domain"]
            require(tuple(support.initial_field.shape[-3:]) == tuple(len(domain[key]) for key in
                    ("region_id","compartment_id","species_id"))
                    and list(support.species_units) == [domain["species_units"][key] for key in domain["species_id"]],
                    "raw whole-particle field domain mismatch")

    def assert_binding(self, grid: dict | None) -> None:
        normalization = (grid or {}).get("normalization", {"kind":"physical"})
        expected = normalization.get("sha256") if normalization.get("kind") == "training_scaler" else "physical"
        require(expected == self.spec.get("source_grid_normalization_sha256", self.identity), "batch/cache/model transform mismatch")

    def assert_domain(self, grid: dict) -> None:
        require(self.active, "fixed training normalization required")
        domains = self.artifact.get("registered_grids", [self.artifact["domain"]])
        require(source_domain(grid) in [source_domain(domain) for domain in domains],
                "grid/species/units/time differs from the fitted registered domains")

    def assert_physical_time(self, times: torch.Tensor) -> None:
        if self.active:
            domains = self.artifact.get("registered_grids", [self.artifact["domain"]])
            for domain in domains:
                shape = (len(domain["time_hours"]), len(domain["region_id"]), len(domain["compartment_id"]), 1)
                if times.ndim != 6 or tuple(times.shape[2:]) != shape:
                    continue
                expected = times.new_tensor(domain["time_hours"]).reshape(1, 1, shape[0], 1, 1, 1)
                if torch.equal(times, expected.expand_as(times)):
                    return
            require(False, "conditioning physical_time values differ from fitted source hours or registered grids")

    def assert_representation(self, native, sampler) -> None:
        sampler.assert_codebook()
        if self.active:
            rep = self.artifact["representation"]
            require(native.tokenizer.wavelet_level_count == rep["wavelet_levels"]
                    and sampler.codebook_size == rep["codebook_size"]
                    and sampler.residual_min_value == rep["codebook_min"] and sampler.residual_max_value == rep["codebook_max"],
                    "model Haar/codebook differs from fitted transform")

    def excursions(self, value: torch.Tensor, support: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # B,A,T,R,C,S -> B,A,S. Unsupported cells do not enter any count.
        if not self.active:
            return {"field_reference_available": value.new_zeros(value.shape[:2])}
        require(support is not None or self.quantity_profile is None,
                "observable profile excursion diagnostics require quantity support")
        if support is None:
            support = torch.ones_like(value, dtype=torch.bool)
        require(support.dtype == torch.bool, "quantity support must be boolean")
        support = torch.broadcast_to(support.to(value.device), value.shape)
        lower, upper = self.source_min.to(value), self.source_max.to(value)
        outside = ((value < lower) | (value > upper)) & support
        return {"field_reference_available": value.new_ones(value.shape[:2]),
            "source_range_excursion_count": outside.sum((2,3,4)),
            "constant_channel_excursion_count": (outside & self.constant).sum((2,3,4)),
            "source_reference_denominator": support.sum((2,3,4))}
