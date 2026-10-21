from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.local_membrane import MembraneAcquisition
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.models.modules.world_state import JointParticleSupport
from Libs.utils.data.joint_observation import RetainedObservation, observation_distribution, observation_to_payload
from Libs.utils.model.native_arrived_observation import (
    NativeAcquisition, NativeArrivedObservation, native_acquisition_mask, project_cns_readout,
)
from Libs.utils.data.platform_contracts import sample_observation_rows
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, resolve_repo_path, write_tsv

if TYPE_CHECKING:
    from Libs.utils.model.native_predose_query import NativePredoseChannel, NativePredoseRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical sample-observation contract catalog.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    root = resolve_repo_path(config["data"]["sample_observation_registry"]["local_root"])
    ensure_directory(root)
    relative_paths = config["data"]["sample_observation_registry"]["relative_paths"]
    rows = sample_observation_rows(config)
    output_path = root / relative_paths["sample_observation_catalog"]
    write_tsv(output_path, config["schemas"]["sample_observation_fields"], rows)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=output_path,
        sample_observation_row_count=len(rows),
    )


@dataclass(frozen=True)
class ControlledObservationProduction:
    observations: tuple[RetainedObservation, ...]
    latent_seed: int
    observation_seed: int
    # Producer-only diagnostics; never included in the inference request.
    generating_indices: torch.Tensor
    generating_fields: torch.Tensor

    def inference_payload(self) -> dict:
        return {"observations": [observation_to_payload(item) for item in self.observations],
                "source_kind": "controlled_simulated", "latent_seed": self.latent_seed,
                "observation_seed": self.observation_seed}


def produce_controlled_observations(support: JointParticleSupport, channels: list[dict], *,
                                    latent_seed: int, observation_seed: int) -> ControlledObservationProduction:
    """Draw declared readouts; the latent draw remains confined to this producer."""
    support.validate()
    if support.source_kind != "controlled_simulated":
        raise ValueError("Controlled observation production cannot manufacture public evidence.")
    # Validate the requested physical readout/acquisition batch before either
    # stochastic boundary, including any invalid acquisition in a later channel.
    prepared_channels = []
    for spec in channels:
        parameters, nuisance = support.resolve_readout(spec["channel_uid"])
        acquisition = (MembraneAcquisition.from_payload(spec["physical_acquisition"], support.prior_weights.device)
                       if spec.get("physical_acquisition") is not None else None)
        if "physical_readout" in parameters:
            if acquisition is None:
                raise ValueError("local membrane: physical acquisition binding is required before production")
            acquisition.validate(support.local_membrane, parameters["physical_readout"])
        elif acquisition is not None:
            raise ValueError("local membrane: acquisition metadata attached to a nonphysical channel")
        prepared_channels.append((spec, parameters, nuisance, acquisition))
    latent_generator = torch.Generator().manual_seed(latent_seed)
    indices = torch.multinomial(support.prior_weights, 1, generator=latent_generator).squeeze(1)
    observations = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(observation_seed)
        for sequence, (spec, parameters, nuisance, acquisition) in enumerate(prepared_channels):
            channel_uid = spec["channel_uid"]
            b, _, f = next(value for value in parameters.values() if torch.is_tensor(value)).shape[:3]
            sampling_parameters = dict(parameters)
            library = torch.as_tensor(spec["library_offset"], dtype=torch.float32) if "library_offset" in spec else None
            if parameters["law"] == "negative_binomial":
                sampling_parameters["mean"] = parameters["mean"] * library.unsqueeze(1)
            distribution = observation_distribution(sampling_parameters, nuisance)
            generated = distribution.sample()
            values = generated[torch.arange(b), indices].float()
            observed = torch.as_tensor(spec["observed"])
            if observed.dtype != torch.bool:
                raise ValueError("Controlled observation masks must be declared booleans.")
            if observed.shape != (b, f):
                raise ValueError("Controlled observation mask does not align to the source readout.")
            limit = torch.as_tensor(spec["detection_limit"], dtype=torch.float32) if "detection_limit" in spec else None
            censored = observed & (values < limit) if parameters["law"] == "log_normal" else torch.zeros_like(observed)
            observed = observed & ~censored
            values = torch.where(observed, values, torch.full_like(values, float("nan")))
            record = RetainedObservation(
                observation_uid=spec["observation_uid"], source_uid=support.source_uid,
                source_snapshot_ref=support.source_snapshot_ref, sample_uids=support.sample_uids,
                role=support.role, channel_uid=channel_uid, law=parameters["law"], units=parameters["units"],
                values=values, observed=observed, left_censored=censored,
                encoder_indices=tuple(spec["encoder_indices"]), sequence_number=sequence,
                library_offset=library, detection_limit=limit, normalization_ref=spec.get("normalization_ref"),
                modality=spec.get("modality", "state"), assay_uid=spec.get("assay_uid"),
                informative_admission=bool(spec.get("informative_admission", False)),
                admission_log_probability=spec.get("admission_log_probability"), physical_acquisition=acquisition)
            support.validate_observation(record, spec.get("registered_occupancy_assays", {}))
            observations.append(record)
    fields = support.initial_field[torch.arange(len(indices)), indices].clone()
    return ControlledObservationProduction(tuple(observations), latent_seed, observation_seed, indices, fields)


@torch.no_grad()
def produce_native_arrived_observation(
        private_field: torch.Tensor, acquisition: NativeAcquisition, *,
        episode_source_uid: str, episode_source_snapshot_ref: str,
        episode_sample_uids: tuple[str, ...], episode_query_index: int,
        episode_regimen_uid: str, measurement_generator: torch.Generator,
        observation_uid: str) -> NativeArrivedObservation:
    """Draw one monitoring measurement from an already-carried evaluator field.

    The caller retains episode state and RNG. This function never chooses an
    episode/particle, advances a source, resets RNG, or exports either private input.
    """
    acquisition.validate()
    readout = acquisition.readout
    if (episode_source_uid != readout.source_uid
            or episode_source_snapshot_ref != readout.source_snapshot_ref
            or episode_sample_uids != acquisition.sample_uids
            or type(episode_query_index) is not int or episode_query_index != acquisition.query_index
            or episode_regimen_uid != acquisition.executed_regimen_uid
            or private_field.ndim != 4 or private_field.shape[0] != len(episode_sample_uids)
            or not isinstance(observation_uid, str) or not observation_uid):
        raise ValueError("Evaluator episode source/sample/query/program differs from the acquisition")
    mean = project_cns_readout(private_field, readout)
    noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=measurement_generator)
    values = mean+readout.standard_deviation*noise
    # Draw the unchanged full channel before selecting an output coordinate, so
    # paired worlds retain the same conditional noise irrespective of selection.
    observed = torch.tensor(native_acquisition_mask(readout, acquisition.selected_coordinate),
                            device=mean.device, dtype=torch.bool).expand_as(values).clone()
    missing = (torch.full_like(values, float("nan")) if acquisition.selected_coordinate is None
               else torch.zeros_like(values))
    observation = RetainedObservation(
        observation_uid=observation_uid, source_uid=readout.source_uid,
        source_snapshot_ref=readout.source_snapshot_ref, sample_uids=episode_sample_uids,
        role=readout.role, channel_uid=readout.channel_uid, law="gaussian", units=readout.units,
        values=torch.where(observed, values, missing),
        observed=observed, left_censored=torch.zeros_like(observed),
        encoder_indices=tuple(range(len(readout.selector))), sequence_number=acquisition.filtering_revision+1,
        normalization_ref=readout.normalization_ref, modality="state", assay_uid=readout.assay_uid)
    observation.validate({"law": "gaussian", "units": readout.units}, {})
    return NativeArrivedObservation(acquisition, observation)


class NativePredoseMeasurementProvider:
    """Evaluator-owned repeated readouts of one explicitly supplied pre-dose field.

    No trajectory, hidden index, latent draw or generator state enters its output.
    The caller is responsible for an evidence-backed pre-dose field binding.
    """
    def __init__(self, private_predose_field: torch.Tensor, channel: NativePredoseChannel,
                 measurement_generator: torch.Generator):
        channel.validate()
        if private_predose_field.ndim != 4 or private_predose_field.shape[0] != 1:
            raise ValueError("Evaluator provider needs one supplied pre-dose field")
        self._field = private_predose_field.detach().clone()
        self._channel = channel
        self._generator = measurement_generator
        self._issued: tuple[int, ...] = ()

    @torch.no_grad()
    def __call__(self, request: NativePredoseRequest) -> RetainedObservation:
        # The native source imports this producer; defer this reverse edge.
        from Libs.utils.model.native_predose_query import original_predose_readout
        expected = self._channel.request(request.coordinate, len(self._issued)+1,
                                         request.origin_posterior_uid, self._issued)
        if (request.readout != expected.readout or request.support_uid != expected.support_uid
                or request.variance != expected.variance
                or request.observation.sample_uids != expected.observation.sample_uids
                or request.observation.source_uid != expected.observation.source_uid
                or request.observation.sequence_number != expected.observation.sequence_number
                or request.observation.observation_uid != expected.observation.observation_uid):
            raise ValueError("Evaluator pre-dose source/sample/sequence differs from the acquisition")
        # Preserve both producer float32 roundings and the original Gaussian
        # scale (sqrt of its retained float32 variance), with fresh independent noise.
        mean = original_predose_readout(self._field, self._channel.readout)[0, request.coordinate].double()
        scale = self._channel.parameters["variance"][0, 0, request.coordinate].sqrt().double()
        noise = torch.randn((), dtype=torch.float64, device=mean.device, generator=self._generator)
        values = expected.observation.values.clone()
        values[0, request.coordinate] = mean+scale*noise
        self._issued = (*self._issued, request.coordinate)
        return replace(expected.observation, values=values)


if __name__ == "__main__":
    main()
