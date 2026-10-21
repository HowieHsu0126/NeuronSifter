from __future__ import annotations

import torch
from torch import nn

from Libs.models.layers.feed_forward import FeedForwardBlock, required_policy_vector
from Libs.models.layers.transolver_attention import PhysicsStateSliceAttention, TransolverPlusMicroenvironmentBlock
from Libs.models.modules.evidence import RegionPriorEncoder, SampleEvidenceEncoder
from Libs.models.modules.field_rollout import ActionConditionedOperatorBlock, FieldRolloutOperator
from Libs.models.modules.moe import SemanticMechanismMixtureOfExperts
from Libs.models.modules.transolver_operator import TransolverMicroenvironmentOperator
from Libs.models.modules.world_state import ObservationAndScreeningHeads, WorldStatePosterior, JointParticleSupport, JointWorldStatePosterior, FACTOR_NAMES
from Libs.utils.data.joint_observation import retained_encoder_inputs, observation_from_payload
from Libs.utils.model.screening_core import TRI_COMPONENT_NAMES
from Libs.utils.data.native_diffusion_data import joint_particle_contexts
from Libs.utils.data.native_observable_profile import merge_observable_readouts
from Libs.utils.model.joint_rollout import aggregate_joint_rollouts, aggregate_joint_sequences, operator_splitting_record


class ADScreeningSystem(nn.Module):
    def __init__(
        self,
        patient_state_dim: int,
        target_feature_dim: int,
        regimen_feature_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout_rate: float,
        compartment_count: int,
        tri_weight_by_component: dict[str, float],
        region_count: int,
        species_count: int,
        rollout_step_count: int,
        adjacency_matrix: list[list[float]],
        region_prior_feature_dim: int,
        default_region_prior: list[list[float]],
        fast_steps_per_slow_step: int = 1,
        fast_state_indices: list[int] | None = None,
        slow_state_indices: list[int] | None = None,
        slow_update_scale: float = 0.25,
        baseline_field_policy: dict[str, object] | None = None,
        rollout_update_policy: dict[str, object] | None = None,
        solver_state_layout: dict[str, int] | None = None,
        solver_config: dict[str, object] | None = None,
        world_state_posterior_config: dict[str, object] | None = None,
        multimodal_moe_config: dict[str, object] | None = None,
        field_normalization_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.patient_state_dim = patient_state_dim
        self.target_feature_dim = target_feature_dim
        self.regimen_feature_dim = regimen_feature_dim
        self.compartment_count = compartment_count
        self.region_count = region_count
        self.species_count = species_count
        self.rollout_step_count = rollout_step_count
        self.fast_steps_per_slow_step = max(int(fast_steps_per_slow_step), 1)
        self.region_prior_feature_dim = region_prior_feature_dim
        self.register_buffer(
            "default_region_prior",
            torch.tensor(default_region_prior, dtype=torch.float32),
        )
        self.sample_evidence_encoder = SampleEvidenceEncoder(
            patient_state_dim,
            hidden_dim,
            latent_dim,
            region_count,
            compartment_count,
            species_count,
            region_prior_feature_dim,
            dropout_rate,
            baseline_field_policy=baseline_field_policy,
            multimodal_moe_config=multimodal_moe_config,
            joint_observations=(world_state_posterior_config or {}).get('family') == 'finite_joint_particles',
        )
        self.field_rollout_operator = FieldRolloutOperator(
            latent_dim,
            regimen_feature_dim,
            hidden_dim,
            region_count,
            compartment_count,
            species_count,
            rollout_step_count,
            adjacency_matrix,
            fast_steps_per_slow_step=fast_steps_per_slow_step,
            fast_state_indices=fast_state_indices,
            slow_state_indices=slow_state_indices,
            slow_update_scale=slow_update_scale,
            rollout_update_policy=rollout_update_policy,
            solver_state_layout=solver_state_layout,
            solver_config=solver_config,
            field_normalization_config=field_normalization_config,
        )
        if world_state_posterior_config is None:
            raise ValueError("world_state_posterior_config is required for ADScreeningSystem.")
        self.world_state_posterior = WorldStatePosterior(
            latent_dim,
            species_count,
            int(world_state_posterior_config["posterior_hidden_dim"]),
            float(world_state_posterior_config["posterior_dropout_rate"]),
            int(self.field_rollout_operator.pkpd_indices.numel()),
            int(self.field_rollout_operator.transport_indices.numel()),
            int(self.field_rollout_operator.electrophysiology_indices.numel()),
            joint_config=world_state_posterior_config,
            native_context_width=self.field_rollout_operator.operator_hidden_dim,
        )
        self.observation_and_screening_heads = ObservationAndScreeningHeads(
            patient_state_dim,
            target_feature_dim,
            latent_dim,
            hidden_dim,
            compartment_count,
            species_count,
            tri_weight_by_component,
        )

    def default_observation_mask(self, patient_latent: torch.Tensor) -> torch.Tensor:
        return torch.ones((patient_latent.shape[0], self.region_count), dtype=patient_latent.dtype, device=patient_latent.device)

    def encode_patient(
        self,
        patient_latent: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
        region_prior: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        effective_mask = self.default_observation_mask(patient_latent) if observation_mask is None else observation_mask.to(patient_latent.device, patient_latent.dtype)
        effective_region_prior = self.default_region_prior.to(patient_latent.device, patient_latent.dtype) if region_prior is None else region_prior.to(patient_latent.device, patient_latent.dtype)
        return self.sample_evidence_encoder(patient_latent, effective_mask, effective_region_prior)

    def infer_state(
        self,
        patient_latent: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
        region_prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_payload = self.encode_patient(patient_latent, observation_mask, region_prior)
        disease_logit = self.observation_and_screening_heads.disease_logit(
            encoded_payload["encoded_patient"],
            encoded_payload["baseline_field"],
        )
        return disease_logit, encoded_payload["encoded_patient"]

    def infer_world_state(
        self,
        patient_latent: torch.Tensor | None = None,
        observation_mask: torch.Tensor | None = None,
        region_prior: torch.Tensor | None = None,
        *, joint_support: JointParticleSupport | dict | None = None,
        observations: tuple | list | None = None,
    ) -> dict:
        if self.world_state_posterior.family == "finite_joint_particles":
            if joint_support is None or observations is None:
                raise ValueError("Joint posterior requires source-bound prior support and retained observations.")
            device = next(self.parameters()).device
            support = JointParticleSupport.from_payload(joint_support, device) if isinstance(joint_support, dict) else joint_support
            retained = tuple(observation_from_payload(item, device) if isinstance(item, dict) else item for item in observations)
            values, masks = retained_encoder_inputs(retained, self.patient_state_dim, support=support,
                assays=self.world_state_posterior.joint_config.get("registered_occupancy_assays", {}))
            encoded = self.sample_evidence_encoder.encode_retained(values, masks)
            posterior = self.world_state_posterior.infer_joint(encoded, values, support, retained,
                self.field_rollout_operator.field_normalization)
            weight = posterior.weights
            result = {name: (weight.unsqueeze(-1) * support.factors[name]).sum(1) for name in FACTOR_NAMES}
            result.update(joint_posterior=posterior, posterior_family="finite_joint_particles",
                          baseline_field=(weight[:, :, None, None, None] * support.initial_field).sum(1),
                          hidden_parameter_posterior=(weight.unsqueeze(-1) * support.physical_parameters).sum(1))
            if support.local_membrane is not None:
                result.update(local_membrane=support.local_binding_payload(), local_membrane_status=support.local_membrane_status())
            return result
        if joint_support is not None or observations is not None:
            raise ValueError("Joint inputs cannot enter the deterministic compatibility posterior.")
        effective_mask = self.default_observation_mask(patient_latent) if observation_mask is None else observation_mask.to(patient_latent.device, patient_latent.dtype)
        encoded_payload = self.encode_patient(patient_latent, effective_mask, region_prior)
        baseline_field = encoded_payload["baseline_field"]
        posterior_payload = self.world_state_posterior(
            encoded_payload["encoded_patient"],
            baseline_field,
            encoded_payload["parameter_posterior"],
            encoded_payload["region_prior_context"],
            effective_mask,
            self.field_rollout_operator.pkpd_indices.to(baseline_field.device),
            self.field_rollout_operator.transport_indices.to(baseline_field.device),
            self.field_rollout_operator.electrophysiology_indices.to(baseline_field.device),
        )
        return {
            "posterior_family": "deterministic_compatibility",
            "z_tx": posterior_payload["z_tx"],
            "z_pk": posterior_payload["z_pk"],
            "z_env": posterior_payload["z_env"],
            "z_ephys": posterior_payload["z_ephys"],
            "z_morph": posterior_payload["z_morph"],
            "hidden_parameter_posterior": encoded_payload["parameter_posterior"],
            "baseline_field": baseline_field,
            "region_prior_field": encoded_payload["region_prior_field"],
            "region_prior_context": encoded_payload["region_prior_context"],
        }

    def score_targets(
        self,
        patient_latent: torch.Tensor,
        target_features: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
        region_prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_payload = self.encode_patient(patient_latent, observation_mask, region_prior)
        return self.observation_and_screening_heads.target_scores(
            encoded_payload["encoded_patient"],
            encoded_payload["baseline_field"],
            target_features,
        )

    def _slice_tri_context(
        self,
        tri_context: dict[str, torch.Tensor] | None,
        start_index: int,
        end_index: int,
        regimen_count: int,
    ) -> dict[str, torch.Tensor] | None:
        if tri_context is None:
            return None
        sliced_context: dict[str, torch.Tensor] = {}
        for key, value in tri_context.items():
            if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == regimen_count:
                sliced_context[key] = value[:, start_index:end_index]
            else:
                sliced_context[key] = value
        return sliced_context

    def _score_regimens_chunk(
        self,
        patient_latent: torch.Tensor,
        encoded_payload: dict[str, torch.Tensor],
        regimen_features: torch.Tensor,
        tri_context: dict[str, torch.Tensor] | None,
        diffusion_kwargs: dict[str, object] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        disease_logit = self.observation_and_screening_heads.disease_logit(
            encoded_payload["encoded_patient"],
            encoded_payload["baseline_field"],
        )
        rollout_payload = self.field_rollout_operator(
            encoded_payload["encoded_patient"],
            encoded_payload["baseline_field"],
            encoded_payload["parameter_posterior"],
            encoded_payload["region_prior_field"],
            encoded_payload["region_prior_context"],
            regimen_features,
            **(diffusion_kwargs or {}),
        )
        pooled_field, exposure, occupancy = self.observation_and_screening_heads.regimen_summaries(rollout_payload["final_field"])
        repeated_patient = encoded_payload["encoded_patient"].unsqueeze(1).expand(-1, regimen_features.shape[0], -1)
        regimen_scores = self.observation_and_screening_heads.regimen_score_head(
            torch.cat(
                [
                    repeated_patient,
                    rollout_payload["encoded_regimens"].unsqueeze(0).expand(patient_latent.shape[0], -1, -1),
                    pooled_field,
                ],
                dim=-1,
            )
        ).squeeze(-1)
        repeated_pkpd = torch.cat(
            [
                exposure.unsqueeze(0).expand(patient_latent.shape[0], -1, -1),
                occupancy.unsqueeze(0).expand(patient_latent.shape[0], -1, -1),
                pooled_field,
            ],
            dim=-1,
        )
        pkpd_consistency = torch.sigmoid(self.observation_and_screening_heads.pkpd_consistency_head(repeated_pkpd).squeeze(-1))
        effective_tri_context = dict(tri_context) if tri_context is not None else {}
        if "ranking_stability" not in effective_tri_context and "ranking_stability" in rollout_payload:
            effective_tri_context["ranking_stability"] = rollout_payload["ranking_stability"]
        tri_components, tri_score = self.observation_and_screening_heads.tri_outputs(
            encoded_payload["encoded_patient"],
            rollout_payload["encoded_regimens"],
            pooled_field,
            regimen_scores,
            pkpd_consistency,
            effective_tri_context,
        )
        return {
            **{key: rollout_payload[key] for key in ("quantity_support", "observable_prediction")
               if key in rollout_payload},
            "disease_logit": disease_logit,
            "encoded_patient": encoded_payload["encoded_patient"],
            "encoded_regimens": rollout_payload["encoded_regimens"],
            "baseline_field": encoded_payload["baseline_field"],
            "parameter_posterior": encoded_payload["parameter_posterior"],
            "region_prior_field": encoded_payload["region_prior_field"],
            "multimodal_router_diagnostics": encoded_payload["multimodal_router_diagnostics"],
            "field_trajectory": rollout_payload["field_trajectory"],
            "field_training_reconstruction": rollout_payload["field_training_reconstruction"],
            "fast_field_trajectory": rollout_payload["fast_field_trajectory"],
            "slow_field_trajectory": rollout_payload["slow_field_trajectory"],
            "field_trajectory_std": rollout_payload["field_trajectory_std"],
            "field_trajectory_quantiles": rollout_payload["field_trajectory_quantiles"],
            "field_trajectory_samples": rollout_payload["field_trajectory_samples"],
            "rollout_auxiliary_loss": rollout_payload["rollout_auxiliary_loss"],
            "rollout_auxiliary_loss_per_action": rollout_payload["rollout_auxiliary_loss_per_action"],
            "ranking_stability": rollout_payload["ranking_stability"],
            "router_diagnostics": rollout_payload["router_diagnostics"],
            "field_normalization_diagnostics": rollout_payload["field_normalization_diagnostics"],
            "action_conditioning": rollout_payload["action_conditioning"],
            "final_field": rollout_payload["final_field"],
            "transitioned_state": pooled_field,
            "transition_delta": rollout_payload["slow_field_trajectory"][:, :, -1] - encoded_payload["baseline_field"].unsqueeze(1),
            "transition_gate": torch.sigmoid(rollout_payload["slow_field_trajectory"][:, :, -1]),
            "reconstructed_patient": encoded_payload["reconstructed_patient"],
            "transitioned_observation": pooled_field.mean(dim=2),
            "exposure": exposure,
            "occupancy": occupancy,
            "regimen_scores": regimen_scores,
            "pkpd_consistency": pkpd_consistency,
            "tri_components": tri_components,
            "tri_score": tri_score,
        }

    def score_regimens(
        self,
        patient_latent: torch.Tensor,
        regimen_features: torch.Tensor,
        tri_context: dict[str, torch.Tensor] | None = None,
        observation_mask: torch.Tensor | None = None,
        region_prior: torch.Tensor | None = None,
        regimen_chunk_size: int | None = None,
        include_auxiliary_outputs: bool = True,
        *, diffusion_context: dict[str, torch.Tensor] | None = None,
        teacher_trajectory: torch.Tensor | None = None, teacher_confidence: torch.Tensor | None = None,
        sample_ids: torch.Tensor | None = None, regimen_ids: torch.Tensor | None = None,
        joint_posterior: JointWorldStatePosterior | None = None, particle_programs: dict | None = None,
        regimen_uids: list[str] | None = None, _joint_particle_call: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if joint_posterior is not None:
            if diffusion_context is not None or not regimen_uids:
                raise ValueError("Joint native calls require particle programs and regimen identity, without point context.")
            contexts = joint_particle_contexts(joint_posterior, particle_programs, regimen_uids,
                                               action_policy=self.field_rollout_operator.action_policy_name)
            outputs = [self.score_regimens(joint_posterior.observation_features, regimen_features,
                tri_context=tri_context, observation_mask=observation_mask, region_prior=region_prior,
                regimen_chunk_size=regimen_chunk_size, include_auxiliary_outputs=include_auxiliary_outputs,
                diffusion_context=context, teacher_trajectory=teacher_trajectory, teacher_confidence=teacher_confidence,
                sample_ids=sample_ids, regimen_ids=regimen_ids, _joint_particle_call=True) for context in contexts]
            result = aggregate_joint_rollouts(outputs, joint_posterior, self.field_rollout_operator.quantized_mddm_rollout_operator)
            if joint_posterior.support.local_membrane is not None:
                result.update(local_membrane=joint_posterior.support.local_binding_payload(),
                              local_membrane_status=joint_posterior.support.local_membrane_status())
            return result
        if self.world_state_posterior.family == "finite_joint_particles" and not _joint_particle_call:
            raise ValueError("The selected joint family requires a joint posterior and particle-aligned programs.")
        encoded_payload = self.encode_patient(patient_latent, observation_mask, region_prior)
        total_regimen_count = regimen_features.shape[0]
        regimen_ids = torch.arange(total_regimen_count) if regimen_ids is None else regimen_ids
        diffusion_kwargs = {"diffusion_context": diffusion_context, "teacher_trajectory": teacher_trajectory,
                            "teacher_confidence": teacher_confidence, "sample_ids": sample_ids, "regimen_ids": regimen_ids}
        effective_chunk_size = total_regimen_count if regimen_chunk_size is None or int(regimen_chunk_size) <= 0 else int(regimen_chunk_size)
        if effective_chunk_size >= total_regimen_count:
            chunk_output = self._score_regimens_chunk(patient_latent, encoded_payload, regimen_features, tri_context, diffusion_kwargs)
            if include_auxiliary_outputs:
                return chunk_output
            return {
                **{key: chunk_output[key] for key in ("quantity_support", "observable_prediction",
                    "router_diagnostics", "field_normalization_diagnostics", "action_conditioning")
                   if key in chunk_output},
                "disease_logit": chunk_output["disease_logit"],
                "reconstructed_patient": chunk_output["reconstructed_patient"],
                "field_trajectory": chunk_output["field_trajectory"],
                "field_training_reconstruction": chunk_output["field_training_reconstruction"],
                "field_trajectory_std": chunk_output["field_trajectory_std"],
                "field_trajectory_quantiles": chunk_output["field_trajectory_quantiles"],
                "field_trajectory_samples": chunk_output["field_trajectory_samples"],
                "rollout_auxiliary_loss": chunk_output["rollout_auxiliary_loss"],
                "rollout_auxiliary_loss_per_action": chunk_output["rollout_auxiliary_loss_per_action"],
                "ranking_stability": chunk_output["ranking_stability"],
                "exposure": chunk_output["exposure"],
                "occupancy": chunk_output["occupancy"],
                "regimen_scores": chunk_output["regimen_scores"],
                "pkpd_consistency": chunk_output["pkpd_consistency"],
                "tri_components": chunk_output["tri_components"],
                "tri_score": chunk_output["tri_score"],
            }

        chunk_outputs: list[dict[str, torch.Tensor | dict[str, torch.Tensor]]] = []
        for start_index in range(0, total_regimen_count, effective_chunk_size):
            end_index = min(start_index + effective_chunk_size, total_regimen_count)
            chunk_outputs.append(
                self._score_regimens_chunk(
                    patient_latent,
                    encoded_payload,
                    regimen_features[start_index:end_index],
                    self._slice_tri_context(tri_context, start_index, end_index, total_regimen_count),
                    {"diffusion_context": None if diffusion_context is None else {key: value[:, start_index:end_index] for key, value in diffusion_context.items()},
                     "teacher_trajectory": None if teacher_trajectory is None else teacher_trajectory[:, start_index:end_index],
                     "teacher_confidence": None if teacher_confidence is None else teacher_confidence[:, start_index:end_index],
                     "sample_ids": sample_ids, "regimen_ids": regimen_ids[start_index:end_index]},
                )
            )

        merged_output = dict(chunk_outputs[0])
        if "quantity_support" in merged_output:
            merged_output["quantity_support"] = torch.cat([row["quantity_support"] for row in chunk_outputs], dim=1)
            merged_output["observable_prediction"] = merge_observable_readouts(
                [row["observable_prediction"] for row in chunk_outputs])
        merged_output["encoded_regimens"] = torch.cat([chunk_output["encoded_regimens"] for chunk_output in chunk_outputs], dim=0)
        for key in [
            "field_trajectory",
            "field_training_reconstruction",
            "fast_field_trajectory",
            "slow_field_trajectory",
            "field_trajectory_std",
            "final_field",
            "transitioned_state",
            "transition_delta",
            "transition_gate",
            "transitioned_observation",
            "regimen_scores",
            "pkpd_consistency",
            "tri_score",
        ]:
            merged_output[key] = torch.cat([chunk_output[key] for chunk_output in chunk_outputs], dim=1)
        merged_output["field_trajectory_quantiles"] = torch.cat([chunk_output["field_trajectory_quantiles"] for chunk_output in chunk_outputs], dim=2)
        merged_output["field_trajectory_samples"] = torch.cat([chunk_output["field_trajectory_samples"] for chunk_output in chunk_outputs], dim=2)
        merged_output["rollout_auxiliary_loss_per_action"] = torch.cat(
            [chunk_output["rollout_auxiliary_loss_per_action"] for chunk_output in chunk_outputs], dim=1)
        merged_output["rollout_auxiliary_loss"] = sum(chunk_output["rollout_auxiliary_loss"] * chunk_output["field_trajectory"].shape[1] for chunk_output in chunk_outputs) / total_regimen_count
        merged_output["ranking_stability"] = torch.cat([chunk_output["ranking_stability"] for chunk_output in chunk_outputs], dim=1)
        merged_output["exposure"] = torch.cat([chunk_output["exposure"] for chunk_output in chunk_outputs], dim=0)
        merged_output["occupancy"] = torch.cat([chunk_output["occupancy"] for chunk_output in chunk_outputs], dim=0)
        router_outputs = [chunk_output["router_diagnostics"] for chunk_output in chunk_outputs if chunk_output["router_diagnostics"]]
        if router_outputs:
            merged_output["router_diagnostics"] = {
                key: (torch.cat([router_output[key] for router_output in router_outputs], dim=-2)
                      if key == "operator_splitting_case_activation" and router_outputs[0][key].ndim > 2
                      else sum(row[key] * out["field_trajectory"].shape[1] for row, out in zip(router_outputs, chunk_outputs)) / total_regimen_count)
                for key in router_outputs[0] if key != "operator_splitting_identity"
            }
            if "operator_splitting_identity" in router_outputs[0]:
                identity = router_outputs[0]["operator_splitting_identity"]
                if any(row["operator_splitting_identity"] != identity for row in router_outputs):
                    raise ValueError("Conflicting executed expert identities across screening chunks.")
                merged_output["router_diagnostics"]["operator_splitting_identity"] = identity
        merged_output["action_conditioning"] = [
            dict(record, chunk_index=index) for index, row in enumerate(chunk_outputs)
            for record in row["action_conditioning"]]
        field_rows = [row.get("field_normalization_diagnostics", {}) for row in chunk_outputs]
        merged_output["field_normalization_diagnostics"] = {
            key:torch.cat([row[key] for row in field_rows],dim=1) for key in field_rows[0]}
        effective_tri_context = dict(tri_context) if tri_context is not None else {}
        if "ranking_stability" not in effective_tri_context and "ranking_stability" in merged_output:
            effective_tri_context["ranking_stability"] = merged_output["ranking_stability"]
        tri_components, tri_score = self.observation_and_screening_heads.tri_outputs(
            encoded_payload["encoded_patient"],
            merged_output["encoded_regimens"],
            merged_output["transitioned_state"],
            merged_output["regimen_scores"],
            merged_output["pkpd_consistency"],
            effective_tri_context,
        )
        merged_output["tri_components"] = tri_components
        merged_output["tri_score"] = tri_score
        if not include_auxiliary_outputs:
            return {
                **{key: merged_output[key] for key in ("quantity_support", "observable_prediction",
                    "router_diagnostics", "field_normalization_diagnostics", "action_conditioning")
                   if key in merged_output},
                "disease_logit": merged_output["disease_logit"],
                "reconstructed_patient": merged_output["reconstructed_patient"],
                "field_trajectory": merged_output["field_trajectory"],
                "action_conditioning": merged_output["action_conditioning"],
                "field_training_reconstruction": merged_output["field_training_reconstruction"],
                "field_trajectory_std": merged_output["field_trajectory_std"],
                "field_trajectory_quantiles": merged_output["field_trajectory_quantiles"],
                "field_trajectory_samples": merged_output["field_trajectory_samples"],
                "rollout_auxiliary_loss": merged_output["rollout_auxiliary_loss"],
                "rollout_auxiliary_loss_per_action": merged_output["rollout_auxiliary_loss_per_action"],
                "ranking_stability": merged_output["ranking_stability"],
                "exposure": merged_output["exposure"],
                "occupancy": merged_output["occupancy"],
                "regimen_scores": merged_output["regimen_scores"],
                "pkpd_consistency": merged_output["pkpd_consistency"],
                "tri_components": merged_output["tri_components"],
                "tri_score": merged_output["tri_score"],
            }
        return merged_output

    def transition_once(
        self,
        encoded_patient: torch.Tensor,
        regimen_features: torch.Tensor,
        tri_context: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if encoded_patient.ndim != 2 or regimen_features.ndim != 2:
            raise ValueError("transition_once expects [batch, latent_dim] encoded_patient and [batch, regimen_feature_dim] regimen_features.")
        rollout_payload = self.score_regimens(
            encoded_patient,
            regimen_features,
            tri_context=tri_context,
            observation_mask=torch.ones((encoded_patient.shape[0], self.region_count), dtype=encoded_patient.dtype, device=encoded_patient.device),
        )
        return {
            "encoded_patient": rollout_payload["encoded_patient"],
            "encoded_regimens": rollout_payload["encoded_regimens"],
            "transitioned_state": rollout_payload["transitioned_state"][:, 0],
            "transition_delta": rollout_payload["transition_delta"][:, 0].mean(dim=(1, 2)),
            "transition_gate": rollout_payload["transition_gate"][:, 0].mean(dim=(1, 2)),
            "reconstructed_patient": rollout_payload["reconstructed_patient"],
            "transitioned_observation": rollout_payload["transitioned_observation"][:, 0],
            "disease_logit": rollout_payload["disease_logit"],
            "next_disease_logit": rollout_payload["regimen_scores"][:, 0],
            "field_trajectory": rollout_payload["field_trajectory"][:, 0],
            "fast_field_trajectory": rollout_payload["fast_field_trajectory"][:, 0],
            "slow_field_trajectory": rollout_payload["slow_field_trajectory"][:, 0],
            "field_trajectory_std": rollout_payload["field_trajectory_std"][:, 0],
            "rollout_auxiliary_loss": rollout_payload["rollout_auxiliary_loss"],
            "rollout_auxiliary_loss_per_action": rollout_payload["rollout_auxiliary_loss_per_action"],
            "ranking_stability": rollout_payload["ranking_stability"][:, 0],
            "exposure": rollout_payload["exposure"][0],
            "occupancy": rollout_payload["occupancy"][0],
            "regimen_scores": rollout_payload["regimen_scores"][:, 0],
            "pkpd_consistency": rollout_payload["pkpd_consistency"][:, 0],
            "tri_components": {name: value[:, 0] for name, value in rollout_payload["tri_components"].items()},
            "tri_score": rollout_payload["tri_score"][:, 0],
        }

    def rollout(
        self,
        patient_latent: torch.Tensor,
        regimen_sequence: torch.Tensor,
        tri_context_sequence: list[dict[str, torch.Tensor] | None] | None = None,
        observation_mask: torch.Tensor | None = None,
        region_prior: torch.Tensor | None = None,
        *, diffusion_context_sequence: list[dict[str, torch.Tensor]] | None = None,
        regimen_ids_sequence: list[torch.Tensor] | None = None,
        joint_posterior: JointWorldStatePosterior | None = None, particle_programs: dict | None = None,
        regimen_uids: list[str] | None = None, _joint_particle_call: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if regimen_sequence.ndim != 3:
            raise ValueError("rollout expects regimen_sequence shaped [batch, step_count, regimen_feature_dim].")
        step_count = regimen_sequence.shape[1]
        if joint_posterior is not None:
            if regimen_uids is None or len(regimen_uids) != step_count:
                raise ValueError("Joint sequential rollout requires one action identity per step.")
            # Repeated actions share their source program; their state is propagated.
            distinct = list(dict.fromkeys(regimen_uids))
            particle_contexts = joint_particle_contexts(joint_posterior, particle_programs, distinct,
                                                       action_policy=self.field_rollout_operator.action_policy_name)
            outputs = []
            for context in particle_contexts:
                sequence = [{key: value[:, distinct.index(uid):distinct.index(uid)+1] for key, value in context.items()}
                            for uid in regimen_uids]
                outputs.append(self.rollout(joint_posterior.observation_features, regimen_sequence,
                    tri_context_sequence, observation_mask, region_prior, diffusion_context_sequence=sequence,
                    regimen_ids_sequence=regimen_ids_sequence, _joint_particle_call=True))
            result = aggregate_joint_sequences(outputs, joint_posterior, self.field_rollout_operator.quantized_mddm_rollout_operator)
            if joint_posterior.support.local_membrane is not None:
                result.update(local_membrane=joint_posterior.support.local_binding_payload(),
                              local_membrane_status=joint_posterior.support.local_membrane_status())
            return result
        if self.world_state_posterior.family == "finite_joint_particles" and not _joint_particle_call:
            raise ValueError("Joint sequential rollout requires source-bound posterior and particle programs.")
        if tri_context_sequence is None:
            tri_context_sequence = [None] * step_count
        if len(tri_context_sequence) != step_count:
            raise ValueError("rollout requires one TRI context entry per step.")
        if diffusion_context_sequence is not None and len(diffusion_context_sequence) != step_count:
            raise ValueError("rollout requires one compiled diffusion context per action step.")
        if regimen_ids_sequence is not None and len(regimen_ids_sequence) != step_count:
            raise ValueError("rollout requires one regimen RNG identity per action step.")
        encoded_payload = self.encode_patient(patient_latent, observation_mask, region_prior)
        current_field = encoded_payload["baseline_field"]
        initial_field = current_field
        native_sequence = self.field_rollout_operator.quantized_mddm_rollout_operator is not None
        observation_context = diffusion_context_sequence[0] if diffusion_context_sequence else {}
        step_outputs = []
        for step_index in range(step_count):
            regimen_features = regimen_sequence[:, step_index, :]
            diffusion_context = None if diffusion_context_sequence is None else diffusion_context_sequence[step_index]
            if native_sequence and diffusion_context is not None:
                initial = diffusion_context.get("initial_state")
                posterior = diffusion_context.get("posterior_context")
                if (initial is None) != (posterior is None):
                    raise ValueError("Native initial state and posterior context must share one explicit lineage.")
                anchor = observation_context.get("initial_state")
                if (initial is None) != (anchor is None):
                    raise ValueError("Native sequential contexts must share one observation-derived initial/posterior lineage.")
                if initial is not None:
                    if step_index == 0:
                        initial_field = initial.squeeze(1)
                    else:
                        if (not torch.equal(initial, anchor)
                                or not torch.equal(posterior, observation_context["posterior_context"])):
                            raise ValueError("Native sequential contexts must share one observation-derived initial/posterior lineage.")
                        # The supplied pair is the common observation anchor.
                        # A later action starts from the propagated field, while
                        # retaining that posterior; it is not a new observation.
                        diffusion_context = dict(diffusion_context, initial_state=current_field.unsqueeze(1))
            rollout_payload = self.field_rollout_operator(
                encoded_payload["encoded_patient"],
                current_field,
                encoded_payload["parameter_posterior"],
                encoded_payload["region_prior_field"],
                encoded_payload["region_prior_context"],
                regimen_features,
                diffusion_context=diffusion_context,
                regimen_ids=None if regimen_ids_sequence is None else regimen_ids_sequence[step_index],
            )
            pooled_field = rollout_payload["final_field"].mean(dim=(2, 3)).squeeze(1)
            regimen_scores = self.observation_and_screening_heads.regimen_score_head(
                torch.cat([encoded_payload["encoded_patient"], rollout_payload["encoded_regimens"], pooled_field], dim=-1)
            ).squeeze(-1)
            exposure = rollout_payload["final_field"][..., 0].mean(dim=2).squeeze(1)
            occupancy = rollout_payload["final_field"][..., 1].mean(dim=2).squeeze(1)
            pkpd_consistency = torch.sigmoid(
                self.observation_and_screening_heads.pkpd_consistency_head(
                    torch.cat(
                        [
                            exposure,
                            occupancy,
                            pooled_field,
                        ],
                        dim=-1,
                    )
                ).squeeze(-1)
            )
            tri_components, tri_score = self.observation_and_screening_heads.tri_outputs(
                encoded_payload["encoded_patient"],
                rollout_payload["encoded_regimens"],
                pooled_field.unsqueeze(1),
                regimen_scores.unsqueeze(1),
                pkpd_consistency.unsqueeze(1),
                tri_context_sequence[step_index],
            )
            step_outputs.append(
                {
                    **({"quantity_support": rollout_payload["quantity_support"].squeeze(1),
                        "observable_prediction": rollout_payload["observable_prediction"]}
                       if "quantity_support" in rollout_payload else {}),
                    "field_trajectory": rollout_payload["field_trajectory"].squeeze(1),
                    "operator_splitting_diagnostics": operator_splitting_record(rollout_payload["router_diagnostics"]),
                    "field_normalization_diagnostics": rollout_payload["field_normalization_diagnostics"],
                    "action_conditioning": rollout_payload["action_conditioning"],
                    **({"field_trajectory_std": rollout_payload["field_trajectory_std"].squeeze(1),
                        "field_trajectory_samples": rollout_payload["field_trajectory_samples"].squeeze(2)} if native_sequence else {}),
                    "fast_field_trajectory": rollout_payload["fast_field_trajectory"].squeeze(1),
                    "slow_field_trajectory": rollout_payload["slow_field_trajectory"].squeeze(1),
                    "final_field": rollout_payload["final_field"].squeeze(1),
                    "next_disease_logit": regimen_scores,
                    "tri_score": tri_score.squeeze(1),
                    "tri_components": {name: value.squeeze(1) for name, value in tri_components.items()},
                    "pkpd_consistency": pkpd_consistency,
                }
            )
            current_field = rollout_payload["final_field"].squeeze(1)
        return {
            **({"quantity_support_sequence": torch.stack([step["quantity_support"] for step in step_outputs], dim=1),
                "observable_prediction_sequence": merge_observable_readouts(
                    [step["observable_prediction"] for step in step_outputs])}
               if "quantity_support" in step_outputs[0] else {}),
            "initial_state": initial_field,
            "action_conditioning": [dict(record, sequence_index=index)
                                   for index, step in enumerate(step_outputs) for record in step["action_conditioning"]],
            "operator_splitting_diagnostics": [
                dict(step["operator_splitting_diagnostics"], sequence_index=index)
                for index, step in enumerate(step_outputs)],
            "field_normalization_diagnostics": {key:torch.cat([step["field_normalization_diagnostics"][key]
                for step in step_outputs],dim=1) for key in step_outputs[0]["field_normalization_diagnostics"]},
            **({"field_trajectory_std_sequence": torch.stack([step["field_trajectory_std"] for step in step_outputs], 1),
                "field_trajectory_samples_sequence": torch.stack([step["field_trajectory_samples"] for step in step_outputs], 2)}
               if native_sequence else {}),
            "transitioned_state_sequence": torch.stack([step_output["final_field"] for step_output in step_outputs], dim=1),
            "field_trajectory_sequence": torch.stack([step_output["field_trajectory"] for step_output in step_outputs], dim=1),
            "fast_field_trajectory_sequence": torch.stack([step_output["fast_field_trajectory"] for step_output in step_outputs], dim=1),
            "slow_field_trajectory_sequence": torch.stack([step_output["slow_field_trajectory"] for step_output in step_outputs], dim=1),
            "next_disease_logit_sequence": torch.stack([step_output["next_disease_logit"] for step_output in step_outputs], dim=1),
            "pkpd_consistency_sequence": torch.stack([step_output["pkpd_consistency"] for step_output in step_outputs], dim=1),
            "tri_components": {
                component_name: torch.stack([step_output["tri_components"][component_name] for step_output in step_outputs], dim=1)
                for component_name in TRI_COMPONENT_NAMES
            },
            "tri_score_sequence": torch.stack([step_output["tri_score"] for step_output in step_outputs], dim=1),
        }
