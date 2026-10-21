"""Current local membrane H=V binding, independent of electrical evolution."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import copy
import math

import torch

SNAPSHOT_SCHEMA = "local_membrane_snapshot.v1"
VOLTAGE_CONVENTION = "intracellular_minus_extracellular"
READOUT_METADATA = frozenset({"law", "units", "physical_readout", "error_model"})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("local membrane: " + message)


def payload(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, "to_payload"):
        return value.to_payload()
    if isinstance(value, dict):
        return {key: payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [payload(item) for item in value]
    return value


def _exact_fields(value: dict, kind) -> dict:
    require(isinstance(value, dict) and set(value) == {item.name for item in fields(kind)},
            kind.__name__ + " requires exactly its declared fields; history/future targets are unsupported")
    return dict(value)


def _clock(value, device) -> torch.Tensor:
    require(not torch.is_tensor(value) or value.dtype == torch.float64,
            "local-ms clocks must retain float64 precision")
    raw = torch.as_tensor(value, device=device)
    require(raw.dtype != torch.bool and not raw.is_complex(), "clock values must be real numbers")
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def _identities(value, label: str) -> None:
    require(bool(value) and all(isinstance(item, str) and item.strip() for item in value)
            and len(set(value)) == len(value), label + " requires unique nonempty identities")


@dataclass(frozen=True)
class MembraneSite:
    site_uid: str
    membrane_ref: str
    anatomy_ref: str
    region_id: str
    adjacent_compartment_id: str
    voltage_reference_uid: str
    intracellular_ref: str
    extracellular_ref: str

    def validate(self, regions, compartments) -> None:
        require(all(isinstance(value, str) and value.strip() for value in vars(self).values()),
                "site/anatomy/reference identities must be explicit")
        require(self.region_id in regions and self.adjacent_compartment_id in compartments,
                "site anatomical incidence is outside the declared CNS grid")
        require(self.intracellular_ref != self.extracellular_ref,
                "transmembrane voltage requires distinct intracellular/extracellular references")
        require(self.membrane_ref not in (*regions, *compartments)
                and self.site_uid not in (*regions, *compartments),
                "a membrane coordinate cannot alias a CNS transport compartment")

    def to_payload(self) -> dict:
        return dict(vars(self))


@dataclass(frozen=True)
class LocalMembraneSnapshot:
    schema_version: str
    coordinate_uid: str
    source_uid: str
    source_snapshot_ref: str
    support_uid: str
    role: str
    sample_uids: tuple[str, ...]
    particle_uids: tuple[tuple[str, ...], ...]
    region_ids: tuple[str, ...]
    compartment_ids: tuple[str, ...]
    sites: tuple[MembraneSite, ...]
    voltage_convention: str
    voltage_unit: str
    time_unit: str
    clock_uid: str
    time_origin_uid: str
    hour_origin: float
    current_time_ms: torch.Tensor
    acquisition_cutoff_ms: torch.Tensor
    voltage: torch.Tensor
    axes: tuple[str, ...] = ("sample", "particle", "site")

    @classmethod
    def from_payload(cls, value: dict, device="cpu") -> "LocalMembraneSnapshot":
        raw = _exact_fields(value, cls)
        for name in ("sample_uids", "region_ids", "compartment_ids", "axes"):
            raw[name] = tuple(raw[name])
        raw["particle_uids"] = tuple(tuple(row) for row in raw["particle_uids"])
        raw["sites"] = tuple(MembraneSite(**_exact_fields(site, MembraneSite)) for site in raw["sites"])
        for name in ("current_time_ms", "acquisition_cutoff_ms"):
            raw[name] = _clock(raw[name], device)
        voltage = torch.as_tensor(raw["voltage"], device=device)
        require(voltage.dtype != torch.bool and not voltage.is_complex(), "voltage must be real numeric data")
        raw["voltage"] = torch.as_tensor(raw["voltage"], dtype=torch.float64, device=device)
        return cls(**raw)

    def validate(self, support) -> "LocalMembraneSnapshot":
        require(self.schema_version == SNAPSHOT_SCHEMA and bool(self.coordinate_uid), "snapshot schema/coordinate mismatch")
        for name in ("source_uid", "source_snapshot_ref", "support_uid", "role", "sample_uids", "particle_uids"):
            require(getattr(self, name) == getattr(support, name), "snapshot " + name + " binding mismatch")
        b, j = support.prior_weights.shape
        _identities(self.region_ids, "region")
        _identities(self.compartment_ids, "compartment")
        require((len(self.region_ids), len(self.compartment_ids)) == tuple(support.initial_field.shape[2:4]),
                "snapshot incidence grid shape mismatch")
        _identities(tuple(site.site_uid for site in self.sites), "site")
        for site in self.sites:
            site.validate(self.region_ids, self.compartment_ids)
        require(self.voltage_convention == VOLTAGE_CONVENTION and self.voltage_unit == "mV" and self.time_unit == "ms",
                "transmembrane voltage reference/units mismatch")
        require(all(isinstance(value, str) and value.strip() for value in (self.clock_uid, self.time_origin_uid)),
                "clock and hour-origin identities required")
        require(isinstance(self.hour_origin, (int, float)) and not isinstance(self.hour_origin, bool)
                and math.isfinite(self.hour_origin), "hour origin must be a finite scalar")
        for clock in (self.current_time_ms, self.acquisition_cutoff_ms):
            require(clock.dtype == torch.float64 and clock.shape == (b,) and torch.isfinite(clock).all(),
                    "local-ms clocks require finite float64 sample axes")
        require((self.current_time_ms <= self.acquisition_cutoff_ms).all(),
                "current state is beyond the acquisition cutoff")
        require(self.axes == ("sample", "particle", "site") and self.voltage.shape == (b, j, len(self.sites))
                and self.voltage.is_floating_point() and torch.isfinite(self.voltage).all(),
                "voltage requires finite sample/particle/site axes")
        return self

    def to_payload(self) -> dict:
        return {key: payload(value) for key, value in vars(self).items()}

    def select_samples(self, indices: list[int]) -> "LocalMembraneSnapshot":
        return replace(self, sample_uids=tuple(self.sample_uids[i] for i in indices),
                       particle_uids=tuple(self.particle_uids[i] for i in indices),
                       current_time_ms=self.current_time_ms[indices],
                       acquisition_cutoff_ms=self.acquisition_cutoff_ms[indices], voltage=self.voltage[indices])

    def resampled_payload(self, indices: torch.Tensor) -> dict:
        selected = self.to_payload()
        selected["particle_uids"] = [[self.particle_uids[b][i] for i in row.tolist()] for b, row in enumerate(indices)]
        selected["voltage"] = payload(self.voltage.gather(1, indices.unsqueeze(-1).expand(-1, -1, len(self.sites))))
        return selected

    def readout_indices(self, binding: dict) -> list[int]:
        require(isinstance(binding, dict) and set(binding) == {"kind", "coordinate_uid", "site_uids", "voltage_reference_uids"},
                "physical readout requires explicit coordinate/site/reference fields")
        require(binding["kind"] == "local_membrane_voltage" and binding["coordinate_uid"] == self.coordinate_uid,
                "physical readout coordinate mismatch")
        _identities(binding["site_uids"], "readout site")
        lookup = {site.site_uid: site for site in self.sites}
        require(all(uid in lookup for uid in binding["site_uids"]), "unknown readout site")
        require(binding["voltage_reference_uids"] == [lookup[uid].voltage_reference_uid for uid in binding["site_uids"]],
                "physical readout voltage reference mismatch")
        return [list(lookup).index(uid) for uid in binding["site_uids"]]


@dataclass(frozen=True)
class MembraneAcquisition:
    coordinate_uid: str
    support_uid: str
    site_uids: tuple[str, ...]
    voltage_reference_uids: tuple[str, ...]
    voltage_convention: str
    clock_uid: str
    time_origin_uid: str
    hour_origin: float
    time_unit: str
    mode: str
    measurement_time_ms: torch.Tensor
    acquired_time_ms: torch.Tensor

    @classmethod
    def from_payload(cls, value: dict, device="cpu") -> "MembraneAcquisition":
        raw = _exact_fields(value, cls)
        for name in ("site_uids", "voltage_reference_uids"):
            raw[name] = tuple(raw[name])
        for name in ("measurement_time_ms", "acquired_time_ms"):
            raw[name] = _clock(raw[name], device)
        return cls(**raw)

    def validate(self, snapshot: LocalMembraneSnapshot, binding: dict) -> None:
        require(isinstance(self.hour_origin, (int, float)) and not isinstance(self.hour_origin, bool)
                and math.isfinite(self.hour_origin), "acquisition hour origin must be a finite scalar")
        for name in ("coordinate_uid", "support_uid", "voltage_convention", "clock_uid", "time_origin_uid", "hour_origin", "time_unit"):
            require(getattr(self, name) == getattr(snapshot, name), "acquisition " + name + " binding mismatch")
        require(self.mode == "exact_snapshot", "electrical evolution/interpolation binding is missing")
        require(list(self.site_uids) == binding["site_uids"]
                and list(self.voltage_reference_uids) == binding["voltage_reference_uids"],
                "acquisition site/voltage reference mismatch")
        for clock in (self.measurement_time_ms, self.acquired_time_ms):
            require(clock.dtype == torch.float64 and clock.shape == snapshot.current_time_ms.shape
                    and torch.isfinite(clock).all(), "acquisition clocks require finite float64 sample axes")
        # Compare original local-ms clocks, never rounded absolute-hour differences.
        require(torch.equal(self.measurement_time_ms, snapshot.current_time_ms),
                "electrical evolution binding is missing for a different state time")
        require(((self.measurement_time_ms <= self.acquired_time_ms)
                 & (self.acquired_time_ms <= snapshot.acquisition_cutoff_ms)).all(),
                "future acquisition or measurement after acquisition cutoff")

    def to_payload(self) -> dict:
        return {key: payload(value) for key, value in vars(self).items()}

    def select_samples(self, indices: list[int]) -> "MembraneAcquisition":
        return replace(self, measurement_time_ms=self.measurement_time_ms[indices], acquired_time_ms=self.acquired_time_ms[indices])


def physical_readout_bound(parameters: dict) -> bool:
    return "physical_readout" in parameters and parameters.get("error_model") is not None


def validate_physical_readout(support, channel: str) -> bool:
    parameters = support.readout_parameters[channel]
    snapshot = support.local_membrane
    require(snapshot is not None, "physical readout has no local coordinate")
    snapshot.validate(support)
    snapshot.readout_indices(parameters["physical_readout"])
    require(parameters.get("units") == "mV", "physical readout must use mV")
    require("mean" not in parameters and "probabilities" not in parameters,
            "physical H=V cannot be overridden by a free readout mean")
    if not physical_readout_bound(parameters):
        require(set(parameters) == {"physical_readout", "units"}, "unbound error channel contains partial error parameters")
        return False
    error = parameters["error_model"]
    required = {"error_model_uid", "source_ref", "evidence_kind", "source_uid", "source_snapshot_ref",
                "support_uid", "channel_uid", "calibration_status"}
    require(isinstance(error, dict) and set(error) == required
            and all(isinstance(value, str) and value.strip() for value in error.values()),
            "source-declared error model identity/provenance required")
    for name in ("source_uid", "source_snapshot_ref", "support_uid"):
        require(error[name] == getattr(support, name), "error model " + name + " mismatch")
    require(error["channel_uid"] == channel and error["calibration_status"] == "unverified",
            "error model channel/calibration binding mismatch")
    require(error["evidence_kind"] in {"synthetic_fixture", "source_declared"}
            and (support.source_kind != "controlled_simulated" or error["evidence_kind"] == "synthetic_fixture"),
            "controlled error laws must be labelled synthetic fixtures")
    require(parameters.get("law") in {"gaussian", "log_normal"} and "variance" in parameters,
            "an explicit supported continuous error law and numerical variance are required")
    require(set(parameters) == {"physical_readout", "units", "law", "variance", "error_model"},
            "physical error parameters contain unsupported fields")
    return True


def parse_readouts(readouts: dict, device="cpu") -> dict:
    return {channel: {name: copy.deepcopy(value) if name in READOUT_METADATA
                      else torch.as_tensor(value, dtype=torch.float32, device=device)
                      for name, value in parameters.items()} for channel, parameters in readouts.items()}


def select_readouts(readouts: dict, indices: list[int]) -> dict:
    return {channel: {name: copy.deepcopy(value) if name in READOUT_METADATA else [value[i] for i in indices]
                      for name, value in parameters.items()} for channel, parameters in readouts.items()}


def membrane_particle_basis(snapshot: dict, bi: int, ji: int) -> dict:
    result = copy.deepcopy(snapshot)
    result["sample_uids"] = [snapshot["sample_uids"][bi]]
    result["particle_uids"] = [[snapshot["particle_uids"][bi][ji]]]
    for name in ("current_time_ms", "acquisition_cutoff_ms"):
        result[name] = [snapshot[name][bi]]
    result["voltage"] = [[snapshot["voltage"][bi][ji]]]
    return result


class ResolvedMembraneReadout(dict):
    """Internal resolved parameters; serialized free means are never accepted."""

    def __init__(self, parameters: dict, snapshot: LocalMembraneSnapshot):
        indices = snapshot.readout_indices(parameters["physical_readout"])
        super().__init__(parameters, mean=snapshot.voltage[:, :, indices])
        self.snapshot = snapshot


def validate_physical_observation(observation, parameters) -> None:
    if "physical_readout" not in parameters:
        require(observation.physical_acquisition is None, "acquisition metadata attached to a nonphysical channel")
        return
    require(isinstance(parameters, ResolvedMembraneReadout), "physical readout must resolve from its joint coordinate before likelihood")
    require(isinstance(observation.physical_acquisition, MembraneAcquisition), "physical acquisition binding is required")
    observation.physical_acquisition.validate(parameters.snapshot, parameters["physical_readout"])
    indices = parameters.snapshot.readout_indices(parameters["physical_readout"])
    require(torch.equal(parameters["mean"], parameters.snapshot.voltage[:, :, indices]),
            "resolved physical H=V mean no longer matches its current coordinate")
    require(observation.values.ndim == 2
            and observation.values.shape[-1] == len(parameters["physical_readout"]["site_uids"]),
            "observation feature/site shape mismatch")
