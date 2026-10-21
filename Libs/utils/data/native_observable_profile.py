"""Typed support, registered clocks and fixed readout for the native observable profile."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import torch

PROFILE_SCHEMA = "native_observable_quantity_profile.v1"


def profile_identity(profile: dict) -> str:
    return hashlib.sha256(json.dumps(profile, sort_keys=True, allow_nan=False).encode()).hexdigest()


def observable_profile(config: dict) -> dict | None:
    settings = config.get("native_observable_bridge", {})
    if not settings.get("enabled", False):
        return None
    profile = settings["quantity_profile"]
    if settings.get("source_support_overrides"):
        profile = copy.deepcopy(profile)
        profile["source_support"] = copy.deepcopy(settings["source_support_overrides"])
    if profile.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError("Unsupported native observable quantity profile")
    quantities = profile["quantities"]
    ids = [q["quantity_id"] for q in quantities]
    core = config["controlled_cns_source"]["grid"]["species_id"]
    if len(core) != 15 or len(quantities) != 26 or ids[:15] != core or len(set(ids)) != 26:
        raise ValueError("Observable profile must preserve the ordered15 CNS quantities and11 auxiliaries")
    if len(profile["region_id"]) != 6 or len(profile["compartment_id"]) != 6:
        raise ValueError("Observable profile requires the complete CNS graph")
    for quantity in quantities:
        if not quantity.get("unit") or not isinstance(quantity.get("signed"), bool):
            raise ValueError("Every quantity needs explicit units and signedness")
    for family_uid, family in profile["families"].items():
        values = np.asarray(family["time_values"], dtype=float)
        if (family["time_unit"] not in {"ms", "hour"} or values.ndim != 1
                or len(values) < 2 or values[0] != 0 or not np.isfinite(values).all()
                or not (np.diff(values) > 0).all()):
            raise ValueError("Invalid registered physical query clock")
        selected = [q for q in quantities[15:] if q["family_uid"] == family_uid]
        if family.get("readout_kind") == "cns_field":
            if selected or family.get("readout_quantity_ids") != core or family["observable_names"] != []:
                raise ValueError("Long-CNS readout must select the complete15 core fields and no auxiliary values")
            if family["time_unit"] != "hour" or values[-1] != 672. or len(values) != 28:
                raise ValueError("Long-CNS scope requires its actual28-time672hour source clock")
        elif [q["observable_name"] for q in selected] != family["observable_names"]:
            raise ValueError("Original observable denominator/order mismatch")
        for quantity in selected:
            site = quantity["incidence"]
            if (site["anchor_region_id"] not in profile["region_id"]
                    or site["anchor_compartment_id"] not in profile["compartment_id"]
                    or not site.get("physical_site_uid") or not site.get("physical_compartment")):
                raise ValueError("Auxiliary quantities require explicit physical incidence and tensor attachment")
    for source_uid, definition in profile.get("source_support", {}).items():
        family = definition["family_uid"]
        declared = {q["quantity_id"] for q in quantities[15:] if q["family_uid"] == family}
        selected = definition["auxiliary_quantity_ids"]
        if (not source_uid or family not in profile["families"] or not selected
                or len(set(selected)) != len(selected) or not set(selected) <= declared
                or set(definition.get("incidence", {})) != set(selected)):
            raise ValueError("Source support must select declared family auxiliary quantities with exact incidence")
        for site in definition["incidence"].values():
            if (site["anchor_region_id"] not in profile["region_id"]
                    or site["anchor_compartment_id"] not in profile["compartment_id"]
                    or not site["physical_site_uid"] or not site["physical_compartment"]):
                raise ValueError("Source auxiliary incidence is incomplete")
    return profile


def original_query_time_hours(time_values, time_unit: str) -> tuple[float, ...]:
    """Use the registered source-clock conversion without changing its float law."""
    if time_unit not in {"ms", "hour"}:
        raise ValueError("Original query time unit must be ms or hour")
    multiplier = 1.0 / 3_600_000.0 if time_unit == "ms" else 1.0
    return tuple(float(t) * multiplier for t in time_values)


def family_grid(profile: dict, family_uid: str) -> dict:
    family = profile["families"][family_uid]
    return {
        "time_hours": list(original_query_time_hours(family["time_values"], family["time_unit"])),
        "physical_time_unit": "hour",
        "region_id": list(profile["region_id"]),
        "compartment_id": list(profile["compartment_id"]),
        "species_id": [q["quantity_id"] for q in profile["quantities"]],
        "species_units": {q["quantity_id"]: q["unit"] for q in profile["quantities"]},
        "normalization": {"kind": "physical"},
        "quantity_profile": copy.deepcopy(profile),
        "quantity_profile_sha256": profile_identity(profile),
        "family_uid": family_uid,
        "query_time": {"values": list(family["time_values"]), "unit": family["time_unit"]},
    }


def validate_profile_grid(grid: dict, profile: dict) -> None:
    if grid.get("family_uid") not in profile["families"]:
        raise ValueError("Unregistered original-family query grid")
    expected = family_grid(profile, grid["family_uid"])
    for name in expected:
        if name != "normalization" and grid.get(name) != expected[name]:
            raise ValueError("Observable profile/grid identity mismatch: " + name)


def source_support_definition(profile: dict, family: str, source_uid: str | None) -> dict | None:
    selected = profile.get("source_support", {}).get(source_uid)
    if selected is not None and selected["family_uid"] != family:
        raise ValueError("Physical source is not supported in this observable family")
    return selected


def quantity_incidence(profile: dict, quantity: dict, source_uid: str | None = None) -> dict:
    source = source_support_definition(profile, quantity["family_uid"], source_uid)
    return copy.deepcopy(source["incidence"].get(quantity["quantity_id"], quantity["incidence"])
                         if source is not None else quantity["incidence"])


def quantity_support(grid: dict, source_uid: str | None = None) -> np.ndarray:
    profile = grid.get("quantity_profile")
    shape = tuple(len(grid[k]) for k in ("time_hours", "region_id", "compartment_id", "species_id"))
    if profile is None:
        return np.ones(shape, dtype=bool)
    validate_profile_grid(grid, profile)
    source = source_support_definition(profile, grid["family_uid"], source_uid)
    selected = None if source is None else set(source["auxiliary_quantity_ids"])
    mask = np.zeros(shape, dtype=bool)
    mask[..., :15] = True
    for index, q in enumerate(profile["quantities"][15:], 15):
        if q["family_uid"] == grid["family_uid"] and (selected is None or q["quantity_id"] in selected):
            site = quantity_incidence(profile, q, source_uid)
            ri = grid["region_id"].index(site["anchor_region_id"])
            ci = grid["compartment_id"].index(site["anchor_compartment_id"])
            mask[:, ri, ci, index] = True
    return mask


def validate_quantity_support(mask: torch.Tensor, grid: dict) -> None:
    """Allow only source-declared masks; source identity is checked by the caller."""
    shape = tuple(len(grid[k]) for k in ("time_hours", "region_id", "compartment_id", "species_id"))
    if mask.dtype != torch.bool or tuple(mask.shape[-4:]) != shape:
        raise ValueError("Source quantity support shape/dtype mismatch")
    rows = mask.reshape(-1, *shape)
    accepted = torch.zeros(len(rows), dtype=torch.bool, device=mask.device)
    sources = [None, *[key for key, value in grid["quantity_profile"].get("source_support", {}).items()
                       if value["family_uid"] == grid["family_uid"]]]
    for source_uid in sources:
        candidate = torch.as_tensor(quantity_support(grid, source_uid), device=mask.device)
        accepted |= (rows == candidate).flatten(1).all(1)
    if not accepted.all():
        raise ValueError("Observable quantity support differs from registered source incidence")


def coefficient_support(mask: torch.Tensor, levels: int) -> torch.Tensor:
    """Haar support on real times; odd repeat padding adds no measured detail."""
    if mask.dtype != torch.bool or mask.ndim != 4 or not torch.equal(mask, mask[:1].expand_as(mask)):
        raise ValueError("Observable support must be boolean and constant on the real family clock")
    temporal = torch.ones(mask.shape[0], dtype=torch.bool, device=mask.device)
    details = []
    for _ in range(levels):
        odd = len(temporal) % 2
        if odd:
            temporal = torch.cat([temporal, temporal[-1:]])
        left, right = temporal[::2], temporal[1::2]
        detail = left & right
        if odd:
            detail[-1] = False
        details.append(detail)
        temporal = left | right
    packed = torch.cat([temporal, *reversed(details)])
    return packed[:, None, None, None] & mask[0]


def constrain_quantities(value: torch.Tensor, profile: dict | None) -> torch.Tensor:
    if profile is None:
        return value
    columns = []
    for index, q in enumerate(profile["quantities"]):
        column = value[..., index]
        if not q["signed"]:
            column = column.clamp_min(0)
        if q.get("maximum") is not None:
            column = column.clamp_max(float(q["maximum"]))
        columns.append(column)
    return torch.stack(columns, -1)


def observable_readout(mean: torch.Tensor, scale: torch.Tensor, grid: dict, field_support: torch.Tensor | None = None) -> dict:
    """Select already source-unit model outputs; contains no transition dynamics."""
    profile = grid["quantity_profile"]
    validate_profile_grid(grid, profile)
    if mean.shape != scale.shape or mean.ndim != 5:
        raise ValueError("Readout requires aligned sample/time/region/compartment/quantity tensors")
    expected = tuple(len(grid[k]) for k in ("time_hours", "region_id", "compartment_id", "species_id"))
    if tuple(mean.shape[1:]) != expected or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale < 0).any():
        raise ValueError("Invalid source-unit prediction/readout")
    if field_support is None:
        field_support = torch.as_tensor(quantity_support(grid), device=mean.device).expand_as(mean)
    if field_support.shape != mean.shape:
        raise ValueError("Readout source support must align to prediction coordinates")
    validate_quantity_support(field_support, grid)
    values, deviations, names, units, masks = [], [], [], [], []
    for si, q in enumerate(profile["quantities"]):
        if q.get("family_uid") != grid["family_uid"]:
            continue
        site = q["incidence"]
        ri = grid["region_id"].index(site["anchor_region_id"])
        ci = grid["compartment_id"].index(site["anchor_compartment_id"])
        mask = field_support[:, :, ri, ci, si]
        values.append(torch.where(mask, mean[:, :, ri, ci, si], 0.))
        deviations.append(torch.where(mask, scale[:, :, ri, ci, si], 0.))
        masks.append(mask[:, 0])
        names.append(q["observable_name"])
        units.append(q["unit"])
    family = profile["families"][grid["family_uid"]]
    if family.get("readout_kind") == "cns_field":
        for ri, region in enumerate(grid["region_id"]):
            for ci, compartment in enumerate(grid["compartment_id"]):
                for si, quantity in enumerate(profile["quantities"][:15]):
                    values.append(mean[:,:,ri,ci,si])
                    deviations.append(scale[:,:,ri,ci,si])
                    names.append(region+"|"+compartment+"|"+quantity["quantity_id"])
                    units.append(quantity["unit"])
                    masks.append(field_support[:, 0, ri, ci, si])
    predicted = torch.stack(values, -1)
    return {"predictive_mean": predicted, "predictive_scale": torch.stack(deviations, -1),
            "feature_names": names, "feature_units": units,
            "feature_mask": torch.stack(masks, -1),
            "time_mask": torch.ones(mean.shape[:2], dtype=torch.bool, device=mean.device),
            "time_values": list(grid["query_time"]["values"]), "time_unit": grid["query_time"]["unit"],
            "quantity_profile_sha256": grid["quantity_profile_sha256"],
            "uncertainty_status": "empirical_draw_moments_uncalibrated"}


def merge_observable_readouts(readouts: list[dict], action_axis: int = 1) -> dict:
    """Join candidate/sequence axes while preserving one original clock and units."""
    if not readouts:
        raise ValueError("Observable readout merge requires at least one output")
    tensor_keys = {"predictive_mean", "predictive_scale", "feature_mask", "time_mask"}
    first = readouts[0]
    for row in readouts:
        if row.keys() != first.keys():
            raise ValueError("Observable readout schema differs across action chunks")
        for key in first.keys() - tensor_keys:
            if row[key] != first[key]:
                raise ValueError("Observable readout metadata differs across action chunks: " + key)
        if any(row[key].dtype != torch.bool for key in ("feature_mask", "time_mask")):
            raise ValueError("Observable readout masks must remain boolean")
    return {key: torch.cat([row[key] for row in readouts], dim=action_axis)
            if key in tensor_keys else value for key, value in first.items()}
