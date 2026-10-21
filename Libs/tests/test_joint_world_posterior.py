from __future__ import annotations

from dataclasses import replace
import copy
import json

import pytest
import torch

from Libs.models.modules.world_state import JointParticleSupport, FACTOR_NAMES
from Libs.scripts.builders.build_sample_observation_registry import produce_controlled_observations
from Libs.tests.test_native_masked_diffusion import _screening_action_case
from Libs.utils.data.joint_observation import RetainedObservation, retained_log_likelihood


def joint_case(action_policy="occupancy_program"):
    config = {"family": "finite_joint_particles", "posterior_hidden_dim": 8, "posterior_dropout_rate": 0.,
              "registered_occupancy_assays": {}, "joint_objectives": {
                  "observation_likelihood_weight": .7, "categorical_kl_weight": .2}}
    model, _, regimens, _ = _screening_action_case(action_policy, config)
    dimensions = model.world_state_posterior.factor_dimensions
    means = torch.tensor([[[1.], [3.]]])
    parameters = {
        "counts": {"law": "negative_binomial", "units": "counts", "mean": means, "dispersion": torch.ones_like(means) * 2},
        "concentration": {"law": "log_normal", "units": "relative_concentration", "mean": means, "variance": torch.ones_like(means) * .3},
        "ephys": {"law": "gaussian", "units": "standardized", "mean": means, "variance": torch.tensor([[[.2], [.8]]])},
        "state": {"law": "categorical", "units": "state_label", "probabilities": torch.tensor([[[[.8, .2]], [[.2, .8]]]])},
    }
    nuisance = {name: torch.zeros_like(fields.get("mean", fields.get("probabilities"))) for name, fields in parameters.items()}
    support = JointParticleSupport(
        "support", "controlled-source", "controlled-source-v1", "controlled_simulated", "train",
        ("sample",), (("particle-a", "particle-b"),),
        {name: torch.tensor([[[.1] * width, [.8] * width]]) for name, width in dimensions.items()},
        (("low", "high"),), torch.stack([torch.full((3, 2, 7), .2), torch.full((3, 2, 7), .6)]).unsqueeze(0),
        torch.stack([torch.full((7,), .1), torch.full((7,), .5)]).unsqueeze(0),
        torch.tensor([[.4, .6]]), parameters, nuisance, ("relative",) * 7).validate()
    channels = [
        {"channel_uid": "counts", "observation_uid": "count-readout", "observed": [[True]], "encoder_indices": [0], "library_offset": [[2.]]},
        {"channel_uid": "concentration", "observation_uid": "concentration-readout", "observed": [[True]], "encoder_indices": [1], "detection_limit": [[.5]]},
        {"channel_uid": "ephys", "observation_uid": "ephys-readout", "observed": [[True]], "encoder_indices": [2], "normalization_ref": "controlled-scaler"},
        {"channel_uid": "state", "observation_uid": "state-readout", "observed": [[True]], "encoder_indices": [3]},
    ]
    production = produce_controlled_observations(support, channels, latent_seed=41, observation_seed=42)
    return model, support, production, regimens, channels


def test_joint_normalization_kl_ess_and_whole_tuple_assimilation():
    model, support, production, _, _ = joint_case()
    posterior = model.infer_world_state(joint_support=support, observations=production.observations)["joint_posterior"]
    assert torch.allclose(posterior.weights.sum(-1), torch.ones(1))
    assert posterior.ess.item() == pytest.approx(1 / posterior.weights.square().sum().item())
    prior = support.prior_weights / support.prior_weights.sum(-1, keepdim=True)
    expected_kl = (posterior.weights * (posterior.weights.log() - prior.log())).sum(-1)
    assert torch.allclose(posterior.categorical_kl(), expected_kl)
    next_observation = replace(production.observations[2], observation_uid="next-ephys", sequence_number=4,
                               values=torch.tensor([[2.]]))
    likelihood = retained_log_likelihood(next_observation, support.readout_parameters["ephys"], support.source_nuisance["ephys"], {})
    updated = posterior.assimilate(next_observation, parent_posterior_uid=posterior.posterior_uid, assays={})
    expected = (posterior.log_weights + likelihood.double()).softmax(-1).to(posterior.weights.dtype)
    assert torch.allclose(updated.weights, expected)
    assert updated.support is posterior.support and updated.particle_context is posterior.particle_context
    with pytest.raises(ValueError, match="Duplicate|out-of-order"):
        updated.assimilate(next_observation, parent_posterior_uid=updated.posterior_uid, assays={})
    with pytest.raises(ValueError, match="lineage"):
        posterior.assimilate(next_observation, parent_posterior_uid="other", assays={})
    samples = updated.resample(20, torch.Generator().manual_seed(3))
    for draw, index in enumerate(samples["indices"][0]):
        for factor in FACTOR_NAMES:
            assert torch.equal(samples["factors"][factor][0, draw], support.factors[factor][0, index])


def test_likelihood_laws_censoring_offsets_missingness_and_rejection():
    _, support, production, _, _ = joint_case()
    count = replace(production.observations[0], values=torch.zeros(1, 1))
    mean = support.readout_parameters["counts"]["mean"] * 2
    expected = torch.distributions.NegativeBinomial(total_count=torch.ones_like(mean)*2, logits=mean.log()-torch.log(torch.tensor(2.))).log_prob(torch.zeros(1,1,1)).sum(-1)
    assert torch.allclose(retained_log_likelihood(count, support.readout_parameters["counts"], support.source_nuisance["counts"], {}), expected)
    concentration = replace(production.observations[1], values=torch.full((1,1), float("nan")),
                            observed=torch.zeros(1,1,dtype=torch.bool), left_censored=torch.ones(1,1,dtype=torch.bool))
    parameters = support.readout_parameters["concentration"]
    expected = torch.special.log_ndtr((torch.log(torch.tensor(.5))-parameters["mean"].log()) / parameters["variance"].sqrt()).sum(-1)
    assert torch.allclose(retained_log_likelihood(concentration, parameters, support.source_nuisance["concentration"], {}), expected)
    ephys = replace(production.observations[2], values=torch.tensor([[1.5]]))
    parameters = support.readout_parameters["ephys"]
    expected = torch.distributions.Normal(parameters["mean"], parameters["variance"].sqrt()).log_prob(ephys.values.unsqueeze(1)).sum(-1)
    assert torch.allclose(retained_log_likelihood(ephys, parameters, support.source_nuisance["ephys"], {}), expected)
    categorical = replace(production.observations[3], values=torch.tensor([[1.]]))
    assert torch.allclose(retained_log_likelihood(categorical, support.readout_parameters["state"], support.source_nuisance["state"], {}), torch.tensor([[.2,.8]]).log())
    missing = replace(ephys, values=torch.full((1,1),float("nan")), observed=torch.zeros(1,1,dtype=torch.bool))
    assert torch.equal(retained_log_likelihood(missing, parameters, support.source_nuisance["ephys"], {}), torch.zeros(1,2))
    with pytest.raises(ValueError, match="withheld"):
        retained_log_likelihood(replace(ephys, informative_admission=True), parameters, support.source_nuisance["ephys"], {})
    with pytest.raises(ValueError, match="registered assay"):
        retained_log_likelihood(replace(ephys, modality="occupancy"), parameters, support.source_nuisance["ephys"], {})


def test_producer_hidden_draw_and_future_targets_are_not_inference_inputs():
    model, support, production, _, _ = joint_case()
    first = model.infer_world_state(joint_support=support, observations=production.observations)["joint_posterior"]
    production.generating_indices.fill_(1)
    production.generating_fields.fill_(99)
    ignored_future_teacher = torch.rand(2, 10) * 100
    assert ignored_future_teacher.shape == (2, 10)
    second = model.infer_world_state(torch.full((1,4),999.), joint_support=support, observations=production.observations)["joint_posterior"]
    assert torch.equal(first.weights, second.weights) and torch.equal(first.particle_context, second.particle_context)
    changed = list(production.observations)
    changed[2] = replace(changed[2], values=changed[2].values + 2.)
    third = model.infer_world_state(joint_support=support, observations=changed)["joint_posterior"]
    assert (first.log_weights - third.log_weights).abs().max() > 1
    first.weights[0,0].backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.sample_evidence_encoder.observation_encoder.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.world_state_posterior.joint_score.parameters())


def test_source_validation_and_support_serialization():
    model, support, production, _, _ = joint_case()
    restored = JointParticleSupport.from_payload(json.loads(json.dumps(support.to_payload())))
    assert torch.equal(restored.initial_field, support.initial_field)
    with pytest.raises(ValueError, match="source/snapshot"):
        replace(support, source_snapshot_ref="").validate()
    with pytest.raises(ValueError, match="zero total mass"):
        replace(support, prior_weights=torch.zeros_like(support.prior_weights)).validate()
    with pytest.raises(ValueError, match="all five"):
        replace(support, factors={}).validate()
    with pytest.raises(ValueError, match="source-bound"):
        model.infer_world_state(torch.zeros(1,4))
    parameters = copy.deepcopy(support.readout_parameters["ephys"])
    parameters["variance"].zero_()
    with pytest.raises(ValueError, match="variance"):
        retained_log_likelihood(production.observations[2], parameters, support.source_nuisance["ephys"], {})


def particle_program_case(posterior):
    from Libs.utils.data.action_observation_contracts import SCHEMA_VERSIONS, compile_pharmacological_action
    support = posterior.support
    b, j = posterior.weights.shape
    r, c = support.initial_field.shape[2:4]
    regimens = ["regimen-a", "regimen-b"]
    shape = (b, j, 2, 3, r, c, 1)
    occupancy = torch.rand(shape, generator=torch.Generator().manual_seed(43))
    records = []
    for bi in range(b):
        particles = []
        for ji in range(j):
            actions = []
            for uid in regimens:
                action = {"schema_version": SCHEMA_VERSIONS["world_action"], "action_uid": "action-"+uid,
                    "intervention_type": "pharmacological", "target_set": ["target"], "agent_set": [uid],
                    "dose": 1., "dose_unit": "mg/kg", "schedule": {"kind": "interval", "interval": 1, "unit": "day"},
                    "route": "intravenous", "duration": {"value": 2, "unit": "day"},
                    "state_scope": {"regions": ["declared-fixture-grid"]}, "support_domain_ref": "controlled-fixture",
                    "source_evidence_refs": ["explicit-synthetic-program-fixture"], "action_provenance_ref": "fixture-v1",
                    "conflicts": [], "status": "contract_valid"}
                pkpd = {"time_grid": [0.,1.,2.], "parameter_provenance": {"status": "verified", "artifact_ref": "explicit-fixture"},
                    "target_program": {"targets": ["target"]}, "exposure_fields": {"array_name": "explicit_fixture_exposure"},
                    "occupancy_fields": {"array_name": "occupancy", "support_array": "support"},
                    "organism_context": "fixture", "supported_organisms": ["fixture"], "uncertainty": {"kind": "supplied_fixture"}}
                baseline, graph = {"state_ref": posterior.posterior_uid}, {"graph_ref": "declared-fixture-grid"}
                compiled = compile_pharmacological_action(action, pkpd_context=pkpd, baseline_state_posterior=baseline, graph_context=graph)
                actions.append({"program_uid": compiled["program_uid"], "posterior_uid": posterior.posterior_uid,
                    "particle_uid": support.particle_uids[bi][ji], "source_uid": support.source_uid,
                    "action_uid": action["action_uid"], "regimen_uid": uid, "target_program_uid": "target",
                    "representation": "explicit_grid", "action": action, "pkpd_context": pkpd,
                    "baseline_state_posterior": baseline, "graph_context": graph, "compiled_program": compiled})
            particles.append(actions)
        records.append(particles)
    return {"posterior_uid": posterior.posterior_uid, "support_uid": support.support_uid,
            "source_uid": support.source_uid, "sample_uids": list(support.sample_uids),
            "particle_uids": [list(row) for row in support.particle_uids], "regimen_uids": regimens, "records": records,
            "occupancy": occupancy, "support": torch.ones(shape),
            "physical_time": torch.arange(3.).reshape(1,1,1,3,1,1,1).expand(shape).clone()}


def test_actual_joint_native_loss_uncertainty_and_shared_particle_ids():
    from Libs.utils.data.native_diffusion_data import joint_particle_contexts
    model, support, production, regimens, _ = joint_case()
    posterior = model.infer_world_state(joint_support=support, observations=production.observations)["joint_posterior"]
    posterior.weights.retain_grad()
    programs = particle_program_case(posterior)
    teacher = torch.rand(1,2,3,3,2,7)
    kwargs = {"joint_posterior": posterior, "particle_programs": programs, "regimen_uids": programs["regimen_uids"],
              "teacher_trajectory": teacher, "teacher_confidence": torch.ones(1,2), "regimen_ids": torch.tensor([0,1])}
    result = model.score_regimens(torch.zeros(1,4), regimens, **kwargs)
    contexts = joint_particle_contexts(posterior, programs, programs["regimen_uids"])
    single = [model.score_regimens(posterior.observation_features, regimens, diffusion_context=context,
        teacher_trajectory=teacher, teacher_confidence=torch.ones(1,2), _joint_particle_call=True) for context in contexts]
    per_particle = torch.stack([value["rollout_auxiliary_loss_per_action"] for value in single], 1)
    expected_loss = (per_particle * posterior.weights.unsqueeze(-1)).sum(1).mean()
    assert torch.allclose(result["rollout_auxiliary_loss"], expected_loss)
    assert torch.allclose(result["field_trajectory_std"].square(), result["within_diffusion_variance"] + result["between_particle_variance"])
    assert result["particle_trajectory_samples"].shape[2] == 2
    assert result["particle_uids"] == support.particle_uids
    result["rollout_auxiliary_loss"].backward()
    assert posterior.weights.grad is not None and torch.isfinite(posterior.weights.grad).all()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.world_state_posterior.joint_factor_projection.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.sample_evidence_encoder.observation_encoder.parameters())
    reordered = model.score_regimens(torch.zeros(1,4), regimens.flip(0),
        **dict(kwargs, regimen_uids=list(reversed(programs["regimen_uids"])), regimen_ids=torch.tensor([1,0]),
               teacher_trajectory=teacher.flip(1)), regimen_chunk_size=1)
    assert torch.equal(reordered["field_trajectory_samples"], result["field_trajectory_samples"].flip(2))
    assert torch.equal(reordered["particle_draw_indices"], result["particle_draw_indices"])
    with pytest.raises(ValueError, match="particle-aligned programs"):
        model.score_regimens(torch.zeros(1,4), regimens, joint_posterior=posterior, regimen_uids=programs["regimen_uids"])
    mismatch = dict(programs, particle_uids=[["wrong", "particle-b"]])
    with pytest.raises(ValueError, match="lineage"):
        model.score_regimens(torch.zeros(1,4), regimens, **dict(kwargs, particle_programs=mismatch))


def test_splitting_an_identical_particle_preserves_native_weighted_loss():
    model, support, production, regimens, _ = joint_case()
    original = model.infer_world_state(joint_support=support, observations=production.observations)["joint_posterior"]
    programs = particle_program_case(original)
    indices = torch.tensor([0,0,1])
    split = replace(support, support_uid="split-support",
        particle_uids=(("a-copy-1","a-copy-2","particle-b"),), regime_ids=(("low","low","high"),),
        factors={name: value[:, indices] for name, value in support.factors.items()},
        initial_field=support.initial_field[:, indices], physical_parameters=support.physical_parameters[:, indices],
        prior_weights=torch.tensor([[.2,.2,.6]]),
        readout_parameters={key: {name: value[:,indices] if torch.is_tensor(value) else value for name,value in fields.items()}
                            for key,fields in support.readout_parameters.items()},
        source_nuisance={key:value[:,indices] for key,value in support.source_nuisance.items()}).validate()
    duplicate = model.infer_world_state(joint_support=split, observations=production.observations)["joint_posterior"]
    split_programs = particle_program_case(duplicate)
    for key in ("occupancy","support","physical_time"):
        split_programs[key] = programs[key][:, indices]
    teacher = torch.rand(1,2,3,3,2,7)
    def evaluate(posterior, program):
        return model.score_regimens(torch.zeros(1,4), regimens, joint_posterior=posterior,
            particle_programs=program, regimen_uids=program["regimen_uids"],
            teacher_trajectory=teacher, teacher_confidence=torch.ones(1,2))["rollout_auxiliary_loss"]
    assert torch.allclose(evaluate(original, programs), evaluate(duplicate, split_programs), atol=2e-6)
    assert torch.allclose(original.weights[:,0], duplicate.weights[:,:2].sum(1), atol=1e-6)


def test_equal_marginals_do_not_erase_whole_particle_dependence():
    _, support, _, _, _ = joint_case()
    coupled = dict(support.factors)
    coupled["z_tx"] = torch.tensor([[[-1.]*6,[1.]*6]])
    coupled["z_pk"] = torch.tensor([[[-1.]*3,[1.]*3]])
    alternate = dict(coupled, z_pk=coupled["z_pk"].flip(1))
    for name in ("z_tx","z_pk"):
        assert torch.equal(coupled[name].mean(1), alternate[name].mean(1))
    first = (coupled["z_tx"][:,:,0]*coupled["z_pk"][:,:,0]).mean(1)
    second = (alternate["z_tx"][:,:,0]*alternate["z_pk"][:,:,0]).mean(1)
    assert first.item() == 1 and second.item() == -1


def raw_joint_source(support, production):
    from types import SimpleNamespace
    from Libs.utils.data.native_diffusion_data import JOINT_SOURCE_SCHEMA
    programs = particle_program_case(SimpleNamespace(support=support, weights=support.prior_weights,
                                                    posterior_uid="source-conditional-programs"))
    programs["source_posterior_uid"] = programs.pop("posterior_uid")
    programs["source_snapshot_ref"] = support.source_snapshot_ref
    def json_value(value):
        if torch.is_tensor(value):
            return value.tolist()
        if isinstance(value, dict):
            return {key: json_value(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [json_value(item) for item in value]
        return value
    payload = production.inference_payload()
    return {"schema_version": JOINT_SOURCE_SCHEMA, "source_posterior_uid": programs["source_posterior_uid"],
            "support": support.to_payload(), "observations": payload["observations"],
            "particle_programs": json_value(programs), "producer_seeds": {"latent_seed": payload["latent_seed"], "observation_seed": payload["observation_seed"]}}


@pytest.mark.parametrize("action_policy", ["occupancy_program", "target_id"])
def test_raw_source_training_serving_and_action_independent_lineage(action_policy):
    from Libs.exp.training.train_screening_system import native_diffusion_training_kwargs
    from Libs.scripts.services.serve_screening_api import infer_world_state, native_diffusion_serving_kwargs, patient_latent_from_payload
    from Libs.utils.data.native_diffusion_data import bind_joint_programs, select_joint_source
    from Libs.models.modules.world_action_encoder import native_action_identity_spec
    model, support, production, regimens, _ = joint_case(action_policy)
    source = raw_joint_source(support, production)
    teacher = torch.rand(1,2,3,3,2,7)
    native = {"split_name": "train", "sample_uids": ["sample"], "regimen_uids": ["regimen-a", "regimen-b"],
              "teacher_trajectory": teacher, "teacher_confidence": torch.ones(1,2), "joint_source": source, "action_policy": action_policy,
              "action_identity_schema": native_action_identity_spec(action_policy)[1]}
    kwargs = native_diffusion_training_kwargs(model, {"labels": [1], "native_diffusion": native},
                                              slice(None), [1,0], torch.device("cpu"), split_name="train")
    posterior = kwargs["joint_posterior"]
    assert kwargs["regimen_uids"] == ["regimen-b", "regimen-a"]
    assert kwargs["particle_programs"]["records"][0][0][0]["posterior_uid"] == source["source_posterior_uid"]
    assert kwargs["particle_programs"]["posterior_uid"] == posterior.posterior_uid
    assert posterior.posterior_uid != source["source_posterior_uid"]
    result = model.score_regimens(torch.full((1,4),99.), regimens.flip(0), **kwargs)
    from Libs.utils.data.raw_data_registry import load_config
    from Libs.exp.training.train_screening_system import screening_teacher_losses, screening_belief_loss
    config = load_config("Libs/configs/config_stack.yaml")
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"] = model.world_state_posterior.joint_config
    masks = {name: torch.tensor([1.,0.]) for name in ("z_pk","z_env","z_ephys")}
    rollout_loss, factor_loss = screening_teacher_losses(result, kwargs["teacher_trajectory"], masks, config, kwargs, training=True)
    belief_loss, status = screening_belief_loss(model, kwargs, result, None, None, None, None, None, None, config)
    assert all(torch.isfinite(value) for value in (rollout_loss,factor_loss,belief_loss))
    assert status["held_modality"] == "not_provided"
    result["rollout_auxiliary_loss"].backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.world_state_posterior.joint_score.parameters())
    bundle = {"screening_model": model, "regimen_index_by_uid": {"regimen-a": 0, "regimen-b": 1}}
    response = infer_world_state(bundle, {"joint_source": source, "action_policy": action_policy,
              "action_identity_schema": native_action_identity_spec(action_policy)[1]})
    json.dumps(response, allow_nan=False)
    serving = native_diffusion_serving_kwargs(bundle, {"joint_source": source, "action_policy": action_policy,
              "action_identity_schema": native_action_identity_spec(action_policy)[1]}, ["regimen-a"])
    assert torch.equal(serving["joint_posterior"].weights, posterior.weights)
    assert patient_latent_from_payload(bundle, {"joint_source": source, "action_policy": action_policy,
              "action_identity_schema": native_action_identity_spec(action_policy)[1]})[0] == "retained_observations"
    changed = select_joint_source(source, ["sample"], ["regimen-a"])
    assert model.infer_world_state(joint_support=changed["support"], observations=changed["observations"])["joint_posterior"].posterior_uid == posterior.posterior_uid
    bad = copy.deepcopy(source)
    bad["support"]["initial_field"][0][0][0][0][0] += 1
    with pytest.raises(ValueError, match="support mismatch"):
        bind_joint_programs(posterior, bad)


def joint_registry_case(tmp_path, admission_event=False, *, action_policy="occupancy_program", store_format="zarr_v2", local_membrane=False):
    from Libs.tests.test_native_diffusion_data import producer_case
    from Libs.exp.training.pretrain_microenvironment_world_model import MicroenvironmentPretrainingModel
    from Libs.utils.model.microenvironment_teacher import write_native_microenvironment_store
    config, registry, arrays, _ = producer_case(tmp_path, store_format, action_policy)
    _, small, _, _, channels = joint_case()
    joint = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]
    joint.update(family="finite_joint_particles", observation_feature_width=4,
                 joint_objectives={"observation_likelihood_weight": .7, "categorical_kl_weight": .2})
    config["screening_pipeline"]["screening_benchmark"]["training"].update(latent_dim=8, hidden_dim=12, dropout_rate=0.)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        model = MicroenvironmentPretrainingModel(config, False)
        expand = lambda value: value.expand(2, *value.shape[1:]).clone()
        support = replace(small, source_uid="source", source_snapshot_ref=registry["sources"][0]["snapshot_ref"],
            sample_uids=("sample-a","sample-b"), particle_uids=(("a-low","a-high"),("b-low","b-high")),
            regime_ids=(("low","high"),("low","high")),
            factors={name: torch.rand(2,2,width) for name,width in model.world_state_posterior.factor_dimensions.items()},
            initial_field=torch.rand(2,2,6,6,15), physical_parameters=torch.rand(2,2,15),
            prior_weights=expand(small.prior_weights), species_units=("fixture_unit",)*15,
            readout_parameters={name:{key:expand(value) if torch.is_tensor(value) else value for key,value in fields.items()}
                                for name,fields in small.readout_parameters.items()},
            source_nuisance={name:expand(value) for name,value in small.source_nuisance.items()}).validate()
    channels = copy.deepcopy(channels)
    for channel in channels:
        for name in ("observed", "library_offset", "detection_limit"):
            if name in channel:
                channel[name] = channel[name] * 2
    if local_membrane:
        support, channels = local_membrane_case(support, channels, registry["grid"])
    production = produce_controlled_observations(support, channels, latent_seed=41, observation_seed=42)
    if admission_event:
        observations = list(production.observations)
        observations[2] = replace(observations[2], informative_admission=True,
            values=torch.full((2,1), float("nan")), observed=torch.zeros(2,1,dtype=torch.bool),
            admission_log_probability=torch.tensor([[0., -float("inf")], [-1e300, 0.]], dtype=torch.float64))
        production = replace(production, observations=tuple(observations))
    source = raw_joint_source(support, production)
    registry.update(posterior_family="finite_joint_particles", joint_sources=[source], programs=[], observations=[])
    store = tmp_path / "joint-source"
    for index, teacher in enumerate(registry["teachers"]):
        program = source["particle_programs"]["records"][index//2][0][index%2]
        teacher.update({name:program[name] for name in ("program_uid","posterior_uid","action_uid","target_program_uid")})
        teacher["trajectory_store_path"] = str(store)
    teacher_arrays = {name:arrays[name] for name in ("trajectory","uncertainty","teacher_confidence")}
    write_native_microenvironment_store(store, config, registry, teacher_arrays, store_format)
    from pathlib import Path
    Path(config["native_diffusion_data"]["registry_path"]).write_text(json.dumps(registry))
    return model, config, registry, teacher_arrays, source


@pytest.mark.parametrize("action_policy", ["occupancy_program", "target_id"])
@pytest.mark.parametrize("store_format", ["zarr_v2", "neuronsifter_chunk_store"])
def test_raw_registry_cache_reorder_role_and_actual_pretraining_backward(tmp_path, action_policy, store_format):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    from Libs.utils.data import native_diffusion_data as native
    model, config, registry, arrays, source = joint_registry_case(tmp_path, action_policy=action_policy, store_format=store_format)
    dataset = pretrain.MicroenvironmentTeacherFamilyDataset(config, "train")
    rows = dataset.__getitems__([3,0])
    assert "initial_state" not in rows[0] and "posterior_context" not in rows[0]
    assert rows[0]["joint_source"]["support"]["sample_uids"] == ["sample-b"]
    assert rows[0]["joint_source"]["particle_programs"]["regimen_uids"] == ["regimen-b"]
    batch = pretrain.collate_native_sources(rows)
    metrics, report = pretrain.run_pretraining_step(model, batch, config, torch.device("cpu"),
        {"gradient_accumulation_steps": 1, "mixed_precision": {"enabled": False, "device_type": "cpu", "dtype": "float32"}})
    assert torch.isfinite(torch.tensor(metrics["total_loss"]))
    assert report["belief_evidence_status"]["calibration"] == "unverified"
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.world_state_posterior.joint_score.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.observation_encoder.parameters())
    cached = native.screening_native_payload(config, "train", ["sample-b","sample-a"], ["regimen-b","regimen-a"])
    assert cached["joint_source"]["support"]["particle_uids"] == [["b-low","b-high"],["a-low","a-high"]]
    assert not {"initial_state","posterior_context","occupancy"} & set(cached)
    import pickle
    restored = pickle.loads(pickle.dumps({"fixture": {"train": {"sample_uids": ["sample-b","sample-a"], "native_diffusion": cached}}}))
    native.validate_cached_native(restored, ["regimen-b","regimen-a"], config)
    changed = copy.deepcopy(restored)
    changed["fixture"]["train"]["native_diffusion"]["joint_source"]["observations"][0]["values"][0][0] += 1
    with pytest.raises(ValueError, match="cache_stale"):
        native.validate_cached_native(changed, ["regimen-b","regimen-a"], config)
    wrong = copy.deepcopy(registry)
    wrong["joint_sources"][0]["support"]["role"] = "validation"
    with pytest.raises(ValueError, match="role"):
        native.join_registry(wrong, config)


def test_joint_sequence_preserves_complete_particle_and_propagated_state():
    from Libs.utils.data.native_diffusion_data import bind_joint_programs
    model, support, production, regimens, _ = joint_case()
    source = raw_joint_source(support, production)
    posterior = model.infer_world_state(joint_support=support, observations=production.observations)["joint_posterior"]
    output = model.rollout(torch.zeros(1,4), regimens.unsqueeze(0), joint_posterior=posterior,
                           particle_programs=bind_joint_programs(posterior, source),
                           regimen_uids=["regimen-a","regimen-b"])
    fields = output["particle_field_trajectory_sequence"]
    assert torch.equal(fields[:,:,1,0], fields[:,:,0,-1])
    assert output["particle_draw_indices"].shape[0] == 1
    assert output["particle_uids"] == support.particle_uids
    assert torch.allclose(output["field_trajectory_std_sequence"].square(),
                          output["within_diffusion_variance"] + output["between_particle_variance"], atol=1e-7)


def test_zero_prior_extreme_evidence_registered_assay_and_bad_masks():
    from Libs.models.modules.world_state import _normalized_log_weights
    from Libs.utils.data.joint_observation import observation_from_payload
    model, support, production, _, _ = joint_case()
    zero = replace(support, prior_weights=torch.tensor([[0.,1.]]))
    posterior = model.infer_world_state(joint_support=zero, observations=production.observations)["joint_posterior"]
    assert torch.equal(posterior.weights, torch.tensor([[0.,1.]]))
    assert torch.isfinite(posterior.categorical_kl()).all()
    extreme_log = _normalized_log_weights(torch.tensor([[-1e30,-1e20]]))
    assert torch.isfinite(extreme_log).all()
    assert torch.equal(extreme_log.exp(), torch.tensor([[0.,1.]], dtype=torch.float64))
    with pytest.raises(ValueError, match="impossible"):
        _normalized_log_weights(torch.tensor([[-float("inf"),-float("inf")]]))
    ephys = replace(production.observations[2], modality="occupancy", assay_uid="declared-assay")
    value = retained_log_likelihood(ephys, support.readout_parameters["ephys"], support.source_nuisance["ephys"],
                                    {"declared-assay":"gaussian"})
    assert torch.isfinite(value).all()
    with pytest.raises(ValueError, match="mask"):
        retained_log_likelihood(replace(ephys, observed=torch.ones(1,1)), support.readout_parameters["ephys"],
                                support.source_nuisance["ephys"], {"declared-assay":"gaussian"})
    payload = production.inference_payload()["observations"][0]
    payload["observed"] = [[1]]
    with pytest.raises(ValueError, match="boolean"):
        observation_from_payload(payload)


@pytest.mark.parametrize("admission_event", [False, True])
def test_joint_source_transfer_and_normal_tensor_cache_roundtrip(tmp_path, monkeypatch, admission_event,
                                                               store_format="zarr_v2", local_membrane=False):
    import numpy as np
    from Libs.scripts.builders.build_full_microenvironment_teacher_registry import rebuild_primary_trajectory_store
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    from Libs.exp.training import train_screening_system as screening
    from Libs.utils.data import native_diffusion_data as native
    model, config, registry, arrays, source = joint_registry_case(
        tmp_path, admission_event, store_format=store_format, local_membrane=local_membrane)
    rows = rebuild_primary_trajectory_store(config, tmp_path, {"trajectory_store": "derived-joint"},
                                             [registry["teachers"][3],registry["teachers"][0]])
    assert [row["source_trajectory_index"] for row in rows] == [3,0]
    derived_path = tmp_path/"derived-joint"
    derived = json.loads((derived_path/".zattrs").read_text())["native_diffusion"]
    assert derived["joint_sources"] == registry["joint_sources"]
    restored, _ = native.load_records(derived, native.join_registry(derived, config), config)
    assert set(restored) == set(native.TEACHER_ARRAY_NAMES)
    assert np.array_equal(restored["trajectory"], arrays["trajectory"][[3,0]])
    source_registry = config["native_diffusion_data"]["registry_path"]
    config["native_diffusion_data"]["registry_path"] = str(derived_path/".zattrs")
    items = pretrain.MicroenvironmentZarrDataset(derived_path, config, split_name="train").__getitems__([1,0])
    assert [item["joint_source"]["support"]["sample_uids"] for item in items] == [["sample-a"],["sample-b"]]
    assert items[0]["joint_source"]["support"]["particle_uids"] == [["a-low","a-high"]]
    config["native_diffusion_data"]["registry_path"] = source_registry
    cached = native.screening_native_payload(config, "train", ["sample-b","sample-a"], ["regimen-b","regimen-a"])
    payload = {
        "selected_target_rows": [{"target_program_uid":"fixture-target"}],
        "selected_regimen_rows": [{"regimen_uid":uid} for uid in cached["regimen_uids"]],
        "modality_payload": {"fixture": {"train": {"sample_uids": cached["sample_uids"],
            "state_features":np.zeros((2,4),dtype=np.float32), "native_diffusion":cached}}},
        "label_payload": {"fixture":{"train":{"target_labels":np.ones((2,1),dtype=np.float32),
                                               "regimen_labels":np.ones((2,2),dtype=np.float32)}}},
    }
    monkeypatch.setattr(screening, "source_artifact_hashes", lambda *args: {})
    monkeypatch.setattr(screening, "freeze_signature_sha256", lambda *args: "controlled-fixture")
    cache_root = tmp_path/"cache"
    config_path = tmp_path/"fixture-config.yaml"
    manifest = screening.write_scope_training_cache(config, config_path, ["controlled-fixture"], cache_root,
                                                     "fixture", "train", 19, payload, 0.)
    assert manifest["native_posterior_family"] == "finite_joint_particles"
    assert manifest["native_joint_source_schema"] == native.JOINT_SOURCE_SCHEMA
    loaded, _, _ = screening.load_scope_training_cache(config, config_path, cache_root, "fixture","train",19)
    loaded_native = loaded["modality_payload"]["fixture"]["train"]["native_diffusion"]
    assert loaded_native["joint_source"] == cached["joint_source"]
    if local_membrane:
        support, observations = native.parse_joint_source(loaded_native["joint_source"])
        assert support.local_membrane.sample_uids == ("sample-b", "sample-a")
        assert support.local_membrane.current_time_ms.tolist() == [.1, .05]
        assert support.local_membrane.current_time_ms.dtype == torch.float64
        assert observations[2].physical_acquisition.measurement_time_ms.tolist() == [.1, .05]
        original = model.infer_joint_source(source)
        posterior = model.infer_joint_source(loaded_native["joint_source"])
        assert torch.allclose(posterior.weights, original.weights.flip(0))
        assert posterior.support.local_membrane.voltage.tolist() == source["support"]["local_membrane"]["voltage"][::-1]
        rows = pretrain.MicroenvironmentTeacherFamilyDataset(config, "train").__getitems__([3,0])
        metrics, report = pretrain.run_pretraining_step(model, pretrain.collate_native_sources(rows), config, torch.device("cpu"),
            {"gradient_accumulation_steps":1, "mixed_precision":{"enabled":False,"device_type":"cpu","dtype":"float32"}})
        assert np.isfinite(metrics["total_loss"])
        assert report["belief_evidence_status"]["calibration"] == "unverified"
        stale = copy.deepcopy(loaded_native)
        stale["joint_source"]["support"]["local_membrane"]["voltage"][0][0][0] += 1.
        with pytest.raises(ValueError, match="cache_stale"):
            native.validate_cached_native({"fixture":{"train":{"sample_uids":stale["sample_uids"],"native_diffusion":stale}}},
                                         stale["regimen_uids"], config)
        invalid = copy.deepcopy(source)
        invalid["observations"][2]["physical_acquisition"]["measurement_time_ms"][0] += .05
        with monkeypatch.context() as context:
            context.setattr(model.observation_encoder, "forward", lambda *a, **k: pytest.fail("future acquisition reached pretraining encoder"))
            with pytest.raises(ValueError, match="different state time"):
                model.infer_joint_source(invalid)
    if admission_event:
        _, restored_observations = native.parse_joint_source(loaded_native["joint_source"])
        event = restored_observations[2]
        assert torch.isnan(event.values).all() and not event.observed.any()
        assert event.admission_log_probability[0,0].item() == -1e300  # sample-b first
        assert torch.isneginf(event.admission_log_probability[1,1])
        json.dumps(loaded_native["joint_source"], allow_nan=False)
    assert np.array_equal(loaded_native["teacher_trajectory"], cached["teacher_trajectory"])
    missing_program = copy.deepcopy(source)
    missing_program["particle_programs"].pop("occupancy")
    with pytest.raises(ValueError, match="Missing particle-aligned occupancy"):
        native.parse_joint_source(missing_program)
    invalid = copy.deepcopy(source)
    invalid["particle_programs"]["records"][0][0][0]["compiled_program"]["status"] = "failed"
    with pytest.raises(ValueError, match="compiler"):
        native.parse_joint_source(invalid)
    for key in ("native_posterior_family","native_joint_source_schema"):
        wrong = dict(manifest, **{key:"stale"})
        assert key in screening.cache_manifest_mismatches(config, config_path, wrong, "fixture","train",19)
    manifest_path = screening.training_cache_manifest_path(cache_root, config)
    manifest_path.write_text(json.dumps(dict(manifest, native_joint_source_schema="stale")))
    with monkeypatch.context() as context:
        context.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Stale joint cache reached tensor deserialization"))
        with pytest.raises(RuntimeError, match="cache_stale.*native_joint_source_schema"):
            screening.load_scope_training_cache(config, config_path, cache_root, "fixture","train",19)


def test_joint_module_parameter_accounting_and_total_budget(tmp_path):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    model, config, _, _, _ = joint_registry_case(tmp_path)
    legacy = copy.deepcopy(config)
    legacy["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["family"] = "deterministic_compatibility"
    point_model = pretrain.MicroenvironmentPretrainingModel(legacy, False)
    total = sum(parameter.numel() for parameter in model.parameters())
    point_total = sum(parameter.numel() for parameter in point_model.parameters())
    assert total-point_total == pretrain.joint_inference_parameter_count(config)
    budget = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["parameter_budget"]
    budget.update(active_parameter_lower_bound=total, active_parameter_upper_bound=total)
    assert pretrain.validate_actual_parameter_budget(model, config) == total
    budget["active_parameter_upper_bound"] = total-1
    with pytest.raises(ValueError):
        pretrain.validate_actual_parameter_budget(model, config)


def test_contradictory_gaussian_evidence_recovers_underflowed_log_mass():
    model, support, production, _, _ = joint_case()
    with torch.no_grad():
        for parameter in model.world_state_posterior.joint_score.parameters():
            parameter.zero_()
    parameters = copy.deepcopy(support.readout_parameters)
    parameters["ephys"].update(mean=torch.tensor([[[-20.],[20.]]]), variance=torch.ones(1,2,1))
    support = replace(support, prior_weights=torch.tensor([[.5,.5]]), readout_parameters=parameters).validate()
    first = replace(production.observations[2], observation_uid="first-contradictory-readout",
                    sequence_number=0, values=torch.tensor([[-20.]], requires_grad=True))
    second = replace(first, observation_uid="second-contradictory-readout",
                     sequence_number=1, values=torch.tensor([[20.]]))
    posterior = model.infer_world_state(joint_support=support, observations=(first,))["joint_posterior"]
    assert posterior.weights[0,1].item() == 0.
    assert torch.isfinite(posterior.log_weights).all()
    likelihood1 = retained_log_likelihood(first, parameters["ephys"], support.source_nuisance["ephys"], {})
    likelihood2 = retained_log_likelihood(second, parameters["ephys"], support.source_nuisance["ephys"], {})
    reference = (support.prior_weights.double().log()+likelihood1.double()+likelihood2.double()).softmax(-1)
    updated = posterior.assimilate(second, parent_posterior_uid=posterior.posterior_uid, assays={})
    assert torch.allclose(updated.weights.double(), reference, atol=1e-7, rtol=0.)
    assert torch.equal(updated.weights, torch.tensor([[.5,.5]]))
    assert updated.categorical_kl().item() == pytest.approx(0., abs=1e-7)
    serialized = json.loads(json.dumps(posterior.to_payload(), allow_nan=False))
    assert serialized["weights"][0][1] == 0.
    assert serialized["log_weights"] == posterior.log_weights.detach().tolist()
    altered = copy.deepcopy(parameters)
    altered["ephys"]["mean"][0,1,0] = 21.
    other = model.infer_world_state(joint_support=replace(support,readout_parameters=altered), observations=(first,))["joint_posterior"]
    assert torch.equal(posterior.weights, other.weights)
    assert not torch.equal(posterior.log_weights, other.log_weights)
    assert posterior.posterior_uid != other.posterior_uid
    updated.weights[0,0].backward()
    assert torch.isfinite(first.values.grad).all() and torch.count_nonzero(first.values.grad)
    print("R1 contradictory Gaussian update:", updated.weights.detach().tolist())


def test_log_weight_kl_gradients_preserve_true_zero_support():
    from Libs.models.modules.world_state import _normalized_log_weights
    model, support, production, _, _ = joint_case()
    base = model.infer_world_state(joint_support=support, observations=production.observations)["joint_posterior"]
    for raw in ([[0.,-800.]], [[0.,-float("inf")]]):
        logits = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
        posterior = replace(base, log_weights=_normalized_log_weights(logits))
        loss = posterior.categorical_kl().sum()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    prior_zero = replace(support, prior_weights=torch.tensor([[1.,0.]]))
    missing = replace(production.observations[2], observation_uid="no-readout", sequence_number=0,
                      values=torch.full((1,1),float("nan")), observed=torch.zeros(1,1,dtype=torch.bool))
    posterior = model.infer_world_state(joint_support=prior_zero, observations=(missing,))["joint_posterior"]
    later = replace(missing, observation_uid="finite-later-readout", sequence_number=1,
                    values=torch.tensor([[3.]]), observed=torch.ones(1,1,dtype=torch.bool))
    updated = posterior.assimilate(later, parent_posterior_uid=posterior.posterior_uid, assays={})
    assert torch.isneginf(updated.log_weights[0,1]) and updated.weights[0,1].item() == 0.
    assert updated.categorical_kl().item() == 0.
    assert json.loads(json.dumps(updated.to_payload(), allow_nan=False))["log_weights"][0][1] == "-inf"


@pytest.mark.parametrize("event_log", [-float("inf"), -1e300])
def test_admission_event_strict_json_preserves_likelihood_and_update(event_log):
    from Libs.utils.data.joint_observation import observation_to_payload, observation_from_payload
    model, support, production, _, _ = joint_case()
    missing = replace(production.observations[2], observation_uid="initial-missing", sequence_number=0,
        values=torch.full((1,1),float("nan")), observed=torch.zeros(1,1,dtype=torch.bool))
    prior = model.infer_world_state(joint_support=support, observations=(missing,))["joint_posterior"]
    event = replace(missing, observation_uid="admission-event", sequence_number=1, informative_admission=True,
                    admission_log_probability=torch.tensor([[0.,event_log]],dtype=torch.float64))
    direct = retained_log_likelihood(event, support.readout_parameters["ephys"], support.source_nuisance["ephys"], {})
    payload = json.loads(json.dumps(observation_to_payload(event), allow_nan=False))
    assert payload["values"] == [[None]]
    assert payload["admission_log_probability"][0][1] == ("-inf" if event_log == -float("inf") else event_log)
    restored = observation_from_payload(payload)
    assert torch.equal(restored.admission_log_probability, event.admission_log_probability)
    assert torch.isnan(restored.values).all() and not restored.observed.any()
    after = retained_log_likelihood(restored, support.readout_parameters["ephys"], support.source_nuisance["ephys"], {})
    assert torch.equal(direct,after)
    updated = prior.assimilate(event,parent_posterior_uid=prior.posterior_uid,assays={})
    restored_update = prior.assimilate(restored,parent_posterior_uid=prior.posterior_uid,assays={})
    assert torch.equal(updated.log_weights,restored_update.log_weights)
    assert updated.posterior_uid == restored_update.posterior_uid
    if event_log == -float("inf"):
        assert torch.isneginf(updated.log_weights[0,1])
    else:
        assert torch.isfinite(updated.log_weights).all() and updated.log_weights[0,1].item() == event_log
        opposite = replace(event,observation_uid="opposite-event",sequence_number=2,
                           admission_log_probability=torch.tensor([[event_log,0.]],dtype=torch.float64))
        recovered = updated.assimilate(opposite,parent_posterior_uid=updated.posterior_uid,assays={})
        reference = (support.prior_weights.double().log()+direct+opposite.admission_log_probability).softmax(-1)
        assert torch.equal(recovered.weights.double(),reference)
    impossible = replace(event,admission_log_probability=torch.full((1,2),-float("inf"),dtype=torch.float64))
    impossible = observation_from_payload(json.loads(json.dumps(observation_to_payload(impossible),allow_nan=False)))
    with pytest.raises(ValueError,match="impossible"):
        prior.assimilate(impossible,parent_posterior_uid=prior.posterior_uid,assays={})


@pytest.mark.parametrize("bad", [float("nan"),float("inf"),.1])
def test_invalid_admission_log_probabilities_reject_serialization(bad):
    from Libs.utils.data.joint_observation import observation_to_payload, observation_from_payload
    _, _, production, _, _ = joint_case()
    event = replace(production.observations[2],informative_admission=True,
                    admission_log_probability=torch.tensor([[0.,bad]],dtype=torch.float64))
    with pytest.raises(ValueError,match="Admission-event"):
        observation_to_payload(event)
    payload = observation_to_payload(production.observations[2])
    payload.update(informative_admission=True,admission_log_probability=[[0.,bad]])
    with pytest.raises(ValueError,match="Admission-event"):
        observation_from_payload(payload)
    payload["admission_log_probability"] = [[0.,None]]
    with pytest.raises(ValueError,match="never null"):
        observation_from_payload(payload)



def test_target_joint_admission_precedes_all_tensor_reads(monkeypatch):
    from Libs.utils.data import native_diffusion_data as native
    model, support, production, _, _ = joint_case("target_id")
    source = raw_joint_source(support, production)
    source["particle_programs"]["records"][0][0][0]["action"].pop("target_set")
    monkeypatch.setattr(torch, "as_tensor", lambda *args, **kwargs: pytest.fail("Invalid target metadata reached tensor conversion"))
    with pytest.raises(ValueError, match="target_set|compiler"):
        native.parse_joint_source(source, action_policy="target_id")


def test_target_joint_cache_rejects_agent_schema_before_teacher_access():
    from Libs.exp.training.train_screening_system import native_diffusion_training_kwargs
    from Libs.models.modules.world_action_encoder import NATIVE_ACTION_IDENTITY_SCHEMA
    model, support, production, _, _ = joint_case("target_id")
    source = raw_joint_source(support, production)
    payload = {"split_name": "train", "sample_uids": ["sample"], "regimen_uids": ["regimen-a", "regimen-b"],
               "teacher_trajectory": object(), "teacher_confidence": object(), "joint_source": source,
               "action_policy": "target_id", "action_identity_schema": NATIVE_ACTION_IDENTITY_SCHEMA}
    with pytest.raises(ValueError, match="identity cache"):
        native_diffusion_training_kwargs(model, {"labels": [1], "native_diffusion": payload}, [0], [0], torch.device("cpu"))



def recompiled_noncanonical_targets(source):
    from Libs.utils.data.action_observation_contracts import compile_pharmacological_action
    invalid = copy.deepcopy(source)
    for sample in invalid["particle_programs"]["records"]:
        for particle in sample:
            for record in particle:
                record["action"]["target_set"] = [" target-with-leading-space"]
                record["pkpd_context"]["target_program"]["targets"] = record["action"]["target_set"]
                compiled = compile_pharmacological_action(record["action"], pkpd_context=record["pkpd_context"],
                    baseline_state_posterior=record["baseline_state_posterior"], graph_context=record["graph_context"])
                assert compiled["status"] in {"compiled", "compiled_with_explicit_uncertainty"} and not compiled["blockers"]
                record.update(compiled_program=compiled, program_uid=compiled["program_uid"])
    return invalid


@pytest.fixture(scope="module")
def target_admission_fixture(tmp_path_factory):
    from Libs.utils.data import native_diffusion_data as native
    model, support, production, regimens, _ = joint_case("target_id")
    source = raw_joint_source(support, production)
    posterior = model.infer_world_state(joint_support=source["support"], observations=source["observations"])["joint_posterior"]
    pretrainer, config, registry, _, full_source = joint_registry_case(tmp_path_factory.mktemp("target_admission"), action_policy="target_id")
    records = native.join_registry(registry, config)
    return model, regimens, source, posterior, pretrainer, config, registry, records, full_source


@pytest.mark.parametrize("consumer", ["parser", "selector", "screening", "screening_tensor_indices", "pretraining_inference",
    "pretraining_forward", "dataset", "evaluation", "serving", "serving_state", "serving_latent", "serving_tri",
    "typed_context", "typed_screening", "typed_pretraining"])
def test_recompiled_target_admission_precedes_consumption(target_admission_fixture, monkeypatch, consumer):
    from Libs.utils.data import native_diffusion_data as native
    from Libs.models.modules.world_action_encoder import NATIVE_TARGET_IDENTITY_SCHEMA
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    from Libs.exp.training.train_screening_system import native_diffusion_training_kwargs
    from Libs.exp.evaluation.evaluate_screening_system import prepare_native_episode
    from Libs.scripts.services import serve_screening_api as serving
    model, regimens, valid, posterior, pretrainer, config, registry, records, full_source = target_admission_fixture
    invalid = recompiled_noncanonical_targets(valid)
    invalid_full = recompiled_noncanonical_targets(full_source)
    broken_records = [dict(row, joint_source=invalid_full) for row in records[:2]]
    programs = dict(invalid["particle_programs"], posterior_uid=posterior.posterior_uid)
    class UnreadTeacher:
        def __array__(self, *args, **kwargs):
            pytest.fail("Noncanonical target reached a teacher array")
        def __getitem__(self, key):
            pytest.fail("Noncanonical target reached teacher selection")
    teacher = UnreadTeacher()
    payload = {"split_name": "train", "sample_uids": ["sample"], "regimen_uids": ["regimen-a", "regimen-b"],
               "teacher_trajectory": teacher, "teacher_confidence": teacher, "joint_source": invalid,
               "action_policy": "target_id", "action_identity_schema": NATIVE_TARGET_IDENTITY_SCHEMA}
    tensor_indices = torch.tensor([0])
    bundle = {"screening_model": model, "regimen_feature_matrix": regimens,
              "regimen_index_by_uid": {"regimen-a": 0, "regimen-b": 1}}
    def denied(*args, **kwargs):
        pytest.fail("Noncanonical target reached tensor conversion, store read or inference")
    monkeypatch.setattr(torch, "as_tensor", denied)
    monkeypatch.setattr(native.dynamics_store, "load_zarr_rows", denied)
    monkeypatch.setattr(native.dynamics_store, "load_chunked_rows", denied)
    monkeypatch.setattr(pretrain, "load_records", denied)
    monkeypatch.setattr(model, "infer_world_state", denied)
    monkeypatch.setattr(pretrainer.world_state_posterior, "infer_joint", denied)
    monkeypatch.setattr(model.field_rollout_operator.quantized_mddm_rollout_operator, "forward", denied)
    monkeypatch.setattr(pretrainer.quantized_mddm, "forward", denied)
    calls = {
        "parser": lambda: native.parse_joint_source(invalid, action_policy="target_id"),
        "selector": lambda: native.select_joint_source(invalid, ["sample"], ["regimen-a"], action_policy="target_id"),
        "screening": lambda: native_diffusion_training_kwargs(model, {"labels": [1], "native_diffusion": payload}, [0], [0], torch.device("cpu")),
        "screening_tensor_indices": lambda: native_diffusion_training_kwargs(model, {"labels": [1], "native_diffusion": payload}, tensor_indices, tensor_indices, torch.device("cpu")),
        "pretraining_inference": lambda: pretrainer.infer_joint_source(invalid_full),
        "pretraining_forward": lambda: pretrainer(None, None, joint_source=invalid_full),
        "dataset": lambda: pretrain.native_dataset_items(registry, broken_records, config, [1., 1.]),
        "evaluation": lambda: prepare_native_episode(model, broken_records, config, {}),
        "serving": lambda: serving.native_diffusion_serving_kwargs(bundle, {"joint_source": invalid}, ["regimen-a"]),
        "serving_state": lambda: serving.infer_world_state(bundle, {"joint_source": invalid}),
        "serving_latent": lambda: serving.patient_latent_from_payload(bundle, {"joint_source": invalid}),
        "serving_tri": lambda: serving.regimen_tri_context(bundle, {"joint_source": invalid}, 1),
        "typed_context": lambda: native.joint_particle_contexts(posterior, programs, ["regimen-a"], action_policy="target_id"),
        "typed_screening": lambda: model.score_regimens(posterior.observation_features, regimens,
            joint_posterior=posterior, particle_programs=programs, regimen_uids=["regimen-a", "regimen-b"], teacher_trajectory=teacher),
        "typed_pretraining": lambda: pretrain.run_joint_pretraining_step(pretrainer,
            {"joint_posterior": posterior, "particle_programs": programs, "regimen_uids": ["regimen-a"], "trajectory": teacher},
            config, torch.device("cpu"), {}),
    }
    with pytest.raises(ValueError, match="target_set"):
        calls[consumer]()


@pytest.mark.parametrize("policy", ["occupancy_program", "drug_id", "target_id"])
def test_joint_selection_retains_valid_selected_policy(target_admission_fixture, policy):
    from Libs.utils.data import native_diffusion_data as native
    from Libs.models.modules.world_action_encoder import native_action_identity_spec
    _, _, source, posterior, _, _, _, _, _ = target_admission_fixture
    selected = native.select_joint_source(source, ["sample"], ["regimen-b", "regimen-a"], action_policy=policy)
    assert selected["particle_programs"]["regimen_uids"] == ["regimen-b", "regimen-a"]
    programs = native.bind_joint_programs(posterior, source, action_policy=policy)
    contexts = native.joint_particle_contexts(posterior, programs, ["regimen-b"], action_policy=policy)
    encoder, _ = native_action_identity_spec(policy)
    expected = torch.tensor(encoder(source["particle_programs"]["records"][0][0][1]["action"]))
    assert all(torch.equal(context["action_identity"][0, 0], expected) for context in contexts)
    if policy == "occupancy_program":
        assert selected == native.select_joint_source(source, ["sample"], ["regimen-b", "regimen-a"])


def test_target_serving_observation_only_inference_remains_supported(target_admission_fixture):
    from Libs.scripts.services import serve_screening_api as serving
    model, _, source, _, _, _, _, _, _ = target_admission_fixture
    bundle = {"screening_model": model}
    observations_only = {key: source[key] for key in ("support", "observations")}
    assert serving.infer_world_state(bundle, {"joint_source": observations_only}) == serving.infer_world_state(bundle, {"joint_source": source})
    assert torch.equal(serving.patient_latent_from_payload(bundle, {"joint_source": observations_only})[1],
                       serving.patient_latent_from_payload(bundle, {"joint_source": source})[1])


def local_membrane_case(support, channels, grid=None):
    """Synthetic H=V/error fixture only; no measured noise/calibration evidence."""
    from Libs.utils.data.local_membrane import LocalMembraneSnapshot, SNAPSHOT_SCHEMA, VOLTAGE_CONVENTION
    b, j = support.prior_weights.shape
    r, c = support.initial_field.shape[2:4]
    regions = grid["region_id"] if grid else [f"region-{i}" for i in range(r)]
    compartments = grid["compartment_id"] if grid else [f"compartment-{i}" for i in range(c)]
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA, "coordinate_uid": "synthetic-local-voltage",
        **{key: getattr(support, key) for key in ("source_uid", "source_snapshot_ref", "support_uid", "role", "sample_uids", "particle_uids")},
        "region_ids": regions, "compartment_ids": compartments,
        "sites": [{"site_uid": "cell-1:soma", "membrane_ref": "cell-1:soma-membrane",
                   "anatomy_ref": "synthetic-soma-incidence", "region_id": regions[0],
                   "adjacent_compartment_id": compartments[-1], "voltage_reference_uid": "local-in-minus-out",
                   "intracellular_ref": "cell-1:cytosol", "extracellular_ref": "cell-1:extracellular-space"}],
        "voltage_convention": VOLTAGE_CONVENTION, "voltage_unit": "mV", "time_unit": "ms",
        "clock_uid": "local-electrical-clock", "time_origin_uid": "synthetic-one-hour-origin", "hour_origin": 1.,
        "current_time_ms": [.05 * (i + 1) for i in range(b)],
        "acquisition_cutoff_ms": [.05 * (i + 2) for i in range(b)],
        "voltage": [[[-65. + 10 * k - i] for k in range(j)] for i in range(b)],
        "axes": ["sample", "particle", "site"],
    }
    binding = {"kind": "local_membrane_voltage", "coordinate_uid": snapshot["coordinate_uid"],
               "site_uids": ["cell-1:soma"], "voltage_reference_uids": ["local-in-minus-out"]}
    error = {"error_model_uid": "mathematical-gaussian-fixture", "source_ref": "synthetic-variance-4-mV2",
             "evidence_kind": "synthetic_fixture", "source_uid": support.source_uid,
             "source_snapshot_ref": support.source_snapshot_ref, "support_uid": support.support_uid,
             "channel_uid": "ephys", "calibration_status": "unverified"}
    parameters = dict(support.readout_parameters, ephys={"law": "gaussian", "units": "mV",
        "physical_readout": binding, "error_model": error, "variance": torch.full((b,j,1), 4.)})
    support = replace(support, local_membrane=LocalMembraneSnapshot.from_payload(snapshot),
                      readout_parameters=parameters,
                      source_nuisance=dict(support.source_nuisance, ephys=torch.zeros(b,j,1))).validate()
    channels = copy.deepcopy(channels)
    for spec in channels:
        if spec["channel_uid"] == "ephys":
            spec["normalization_ref"] = "identity-mV-synthetic-fixture"
            spec["physical_acquisition"] = {
                **{key: snapshot[key] for key in ("coordinate_uid", "support_uid", "voltage_convention",
                                                  "clock_uid", "time_origin_uid", "hour_origin", "time_unit")},
                "site_uids": binding["site_uids"], "voltage_reference_uids": binding["voltage_reference_uids"],
                "mode": "exact_snapshot", "measurement_time_ms": snapshot["current_time_ms"],
                "acquired_time_ms": snapshot["current_time_ms"],
            }
    return support, channels


def test_local_membrane_producer_likelihood_posterior_and_paired_native():
    from Libs.scripts.services.serve_screening_api import infer_world_state
    from Libs.utils.data.joint_observation import observation_to_payload, observation_from_payload
    from Libs.utils.data.native_diffusion_data import bind_joint_programs, joint_particle_contexts
    model, support, _, regimens, channels = joint_case()
    support, channels = local_membrane_case(support, channels)
    model.eval()
    for parameter in model.world_state_posterior.joint_score.parameters():
        parameter.data.zero_()
    production = produce_controlled_observations(support, [channels[2]], latent_seed=41, observation_seed=42)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        expected_draw = torch.distributions.Normal(support.local_membrane.voltage, 2.).sample()
    assert torch.equal(production.observations[0].values,
                       expected_draw[torch.arange(1), production.generating_indices].float())
    assert support.readout_parameters["ephys"].get("mean") is None
    inference_payload = json.loads(json.dumps(production.inference_payload(), allow_nan=False))
    assert not {"generating_indices", "generating_fields", "voltage", "future_values"} & set(inference_payload)
    observation = replace(production.observations[0], values=torch.tensor([[-61.]]))
    restored = observation_from_payload(json.loads(json.dumps(observation_to_payload(observation), allow_nan=False)))
    assert restored.physical_acquisition.measurement_time_ms.dtype == torch.float64
    assert restored.physical_acquisition.hour_origin == 1.
    assert restored.physical_acquisition.measurement_time_ms.item() == .05
    result = model.infer_world_state(joint_support=support, observations=[restored])
    posterior = result["joint_posterior"]
    # Log-normalization constants cancel: both synthetic particles have variance4.
    expected = (support.prior_weights.double().log()
                - .5 * ((restored.values.double().unsqueeze(1) - support.local_membrane.voltage).square() / 4).sum(-1)).softmax(-1)
    assert torch.allclose(posterior.weights.double(), expected, atol=1e-7)
    assert not torch.allclose(posterior.weights, support.prior_weights)
    for name in FACTOR_NAMES:
        assert torch.allclose(result[name], (posterior.weights.unsqueeze(-1) * support.factors[name]).sum(1))
    assert torch.allclose(result["baseline_field"], (posterior.weights[:,:,None,None,None] * support.initial_field).sum(1))
    assert torch.allclose(result["hidden_parameter_posterior"], (posterior.weights.unsqueeze(-1) * support.physical_parameters).sum(1))
    draws = posterior.resample(20, torch.Generator().manual_seed(43))
    for draw, index in enumerate(draws["indices"][0]):
        assert draws["local_membrane"]["voltage"][0][draw] == support.local_membrane.voltage[0,index].tolist()
        assert draws["particle_uids"][0][draw] == draws["local_membrane"]["particle_uids"][0][draw]
        assert torch.equal(draws["initial_field"][0,draw], support.initial_field[0,index])
        assert torch.equal(draws["readout_parameters"]["ephys"]["variance"][0,draw],
                           support.readout_parameters["ephys"]["variance"][0,index])
        assert draws["readout_parameters"]["ephys"]["error_model"] == support.readout_parameters["ephys"]["error_model"]
    next_observation = replace(restored, observation_uid="same-time-next", sequence_number=1)
    updated = posterior.assimilate(next_observation, parent_posterior_uid=posterior.posterior_uid, assays={})
    assert updated.weights[0,0] > posterior.weights[0,0]
    later = replace(next_observation, physical_acquisition=replace(next_observation.physical_acquisition,
                    measurement_time_ms=torch.tensor([.1], dtype=torch.float64)))
    with pytest.raises(ValueError, match="different state time"):
        posterior.assimilate(later, parent_posterior_uid=posterior.posterior_uid, assays={})
    parameters, nuisance = support.resolve_readout("ephys")
    parameters["mean"] = parameters["mean"] + 1.
    with pytest.raises(ValueError, match="H=V mean"):
        retained_log_likelihood(restored, parameters, nuisance, {})
    source = raw_joint_source(support, replace(production, observations=(restored,)))
    served = infer_world_state({"screening_model":model}, {"joint_source":source})
    assert served["joint_posterior"]["weights"] == posterior.to_payload()["weights"]
    assert served["local_membrane_status"]["electrical_evolution_status"] == "unbound"
    assert served["local_membrane_status"]["calibration_status"] == "unverified"
    assert served["joint_posterior"]["local_membrane"]["snapshot"] == support.local_membrane.to_payload()
    programs = bind_joint_programs(posterior, source)
    teacher = torch.rand(1,2,3,3,2,7, generator=torch.Generator().manual_seed(41))
    kwargs = {"joint_posterior":posterior, "particle_programs":programs,
              "regimen_uids":programs["regimen_uids"], "teacher_trajectory":teacher,
              "teacher_confidence":torch.ones(1,2), "regimen_ids":torch.tensor([0,1])}
    output = model.score_regimens(torch.zeros(1,4), regimens, **kwargs)
    contexts = joint_particle_contexts(posterior, programs, programs["regimen_uids"])
    single = [model.score_regimens(posterior.observation_features, regimens, diffusion_context=context,
        teacher_trajectory=teacher, teacher_confidence=torch.ones(1,2), _joint_particle_call=True) for context in contexts]
    losses = torch.stack([value["rollout_auxiliary_loss_per_action"] for value in single], 1)
    assert torch.allclose(output["rollout_auxiliary_loss"], (losses * posterior.weights.unsqueeze(-1)).sum(1).mean())
    assert output["particle_uids"] == support.particle_uids
    assert output["local_membrane"]["snapshot"] == support.local_membrane.to_payload()
    reordered = model.score_regimens(torch.zeros(1,4), regimens.flip(0), **dict(kwargs,
        regimen_uids=list(reversed(programs["regimen_uids"])), regimen_ids=torch.tensor([1,0]),
        teacher_trajectory=teacher.flip(1)), regimen_chunk_size=1)
    assert torch.equal(reordered["field_trajectory_samples"], output["field_trajectory_samples"].flip(2))
    assert torch.equal(reordered["particle_draw_indices"], output["particle_draw_indices"])
    sequence = model.rollout(torch.zeros(1,4), regimens.unsqueeze(0), joint_posterior=posterior,
                             particle_programs=programs, regimen_uids=programs["regimen_uids"])
    assert sequence["local_membrane_status"]["electrical_evolution_status"] == "unbound"
    assert sequence["local_membrane"]["snapshot"] == support.local_membrane.to_payload()
    fields = sequence["particle_field_trajectory_sequence"]
    assert torch.equal(fields[:,:,1,0], fields[:,:,0,-1])


@pytest.mark.parametrize("bad", [
    "free_mean", "unit", "site", "reference", "convention", "clock", "origin", "time_shape",
    "measurement_time", "future_acquisition", "before_measurement", "interpolation",
    "sample", "particle", "source", "role", "voltage_shape", "voltage_nan", "clock_inf",
    "site_alias", "grid_incidence", "future_target", "variance_zero", "error_source", "unlabelled_error",
    "unbound_error", "missing_acquisition", "observed_nan", "feature_shape",
])
def test_local_membrane_rejection_precedes_encoder_and_likelihood(monkeypatch, bad):
    from Libs.models.modules import world_state
    from Libs.utils.data.joint_observation import observation_to_payload
    model, support, _, _, channels = joint_case()
    support, channels = local_membrane_case(support, channels)
    production = produce_controlled_observations(support, [channels[2]], latent_seed=7, observation_seed=13)
    raw = support.to_payload()
    record = observation_to_payload(production.observations[0])
    snapshot, readout, acquisition = raw["local_membrane"], raw["readout_parameters"]["ephys"], record["physical_acquisition"]
    if bad == "free_mean": readout["mean"] = [[[-65.],[-55.]]]
    elif bad == "unit": record["units"] = "V"
    elif bad == "site": acquisition["site_uids"] = ["wrong"]
    elif bad == "reference": acquisition["voltage_reference_uids"] = ["wrong"]
    elif bad == "convention": acquisition["voltage_convention"] = "extracellular_minus_intracellular"
    elif bad == "clock": acquisition["clock_uid"] = "another-clock"
    elif bad == "origin": acquisition["hour_origin"] = 0.
    elif bad == "time_shape": acquisition["measurement_time_ms"] = [[.05]]
    elif bad == "measurement_time": acquisition["measurement_time_ms"] = [.1]
    elif bad == "future_acquisition": acquisition["acquired_time_ms"] = [.15]
    elif bad == "before_measurement": acquisition["acquired_time_ms"] = [0.]
    elif bad == "interpolation": acquisition["mode"] = "sparse_history_interpolation"
    elif bad == "sample": snapshot["sample_uids"] = ["wrong"]
    elif bad == "particle": snapshot["particle_uids"] = [["particle-b","particle-a"]]
    elif bad == "source": snapshot["source_uid"] = "wrong"
    elif bad == "role": record["role"] = "validation"
    elif bad == "voltage_shape": snapshot["voltage"] = [[[-65.]]]
    elif bad == "voltage_nan": snapshot["voltage"][0][0][0] = float("nan")
    elif bad == "clock_inf": snapshot["current_time_ms"] = [float("inf")]
    elif bad == "site_alias": snapshot["sites"][0]["membrane_ref"] = snapshot["compartment_ids"][0]
    elif bad == "grid_incidence": snapshot["sites"][0]["region_id"] = "outside-grid"
    elif bad == "future_target": snapshot["future_voltage"] = [[-60.]]
    elif bad == "variance_zero": readout["variance"][0][0][0] = 0.
    elif bad == "error_source": readout["error_model"]["source_uid"] = "wrong"
    elif bad == "unlabelled_error": readout["error_model"]["evidence_kind"] = "source_declared"
    elif bad == "unbound_error":
        for key in ("error_model", "law", "variance"): readout.pop(key)
        raw["source_nuisance"].pop("ephys")
    elif bad == "missing_acquisition": record.pop("physical_acquisition")
    elif bad == "observed_nan": record["values"] = [[None]]
    elif bad == "feature_shape":
        record.update(values=[[-60.,-59.]], observed=[[True,True]], left_censored=[[False,False]], encoder_indices=[1,2])
    monkeypatch.setattr(model.sample_evidence_encoder, "encode_retained", lambda *a, **k: pytest.fail("invalid acquisition reached encoder"))
    monkeypatch.setattr(world_state, "retained_log_likelihood", lambda *a, **k: pytest.fail("invalid acquisition reached likelihood"))
    with pytest.raises(ValueError):
        model.infer_world_state(joint_support=raw, observations=[record])


def test_local_membrane_unbound_channel_coexists_and_identity_tracks_support():
    from Libs.utils.data.joint_observation import retained_encoder_inputs
    from Libs.utils.data.native_diffusion_data import joint_particle_contexts
    model, legacy, _, _, channels = joint_case()
    support, channels = local_membrane_case(legacy, channels)
    assert "local_membrane" not in legacy.to_payload()
    parameters = copy.deepcopy(support.readout_parameters)
    parameters["ephys"] = {key:value for key,value in parameters["ephys"].items() if key in {"physical_readout","units"}}
    unbound = replace(support, readout_parameters=parameters,
                      source_nuisance={key:value for key,value in support.source_nuisance.items() if key != "ephys"}).validate()
    other = produce_controlled_observations(unbound, [channels[0]], latent_seed=7, observation_seed=13)
    posterior = model.infer_world_state(joint_support=unbound, observations=other.observations)["joint_posterior"]
    assert posterior.to_payload()["local_membrane_status"]["likelihood_status_by_channel"]["ephys"] == "unbound"
    programs = particle_program_case(posterior)
    changed = copy.deepcopy(unbound)
    changed.local_membrane.voltage[0,0,0] += 1.
    fresh = model.infer_world_state(joint_support=changed, observations=other.observations)["joint_posterior"]
    assert torch.equal(posterior.weights, fresh.weights)
    assert fresh.posterior_uid != posterior.posterior_uid
    with pytest.raises(ValueError, match="lineage"):
        joint_particle_contexts(fresh, programs, programs["regimen_uids"])
    # In-place changes also cannot travel under the existing consumed UID.
    unbound.local_membrane.voltage[0,0,0] += 2.
    for consume in (posterior.to_payload,
                    lambda: posterior.resample(2, torch.Generator().manual_seed(1)),
                    lambda: joint_particle_contexts(posterior, programs, programs["regimen_uids"])):
        with pytest.raises(ValueError, match="consumed posterior identity"):
            consume()
    physical = produce_controlled_observations(support, [channels[2]], latent_seed=7, observation_seed=13).observations
    with pytest.raises(ValueError, match="before the evidence encoder"):
        retained_encoder_inputs(physical, 4)
    with pytest.raises(ValueError, match="joint coordinate before likelihood"):
        retained_log_likelihood(physical[0], support.readout_parameters["ephys"], support.source_nuisance["ephys"], {})
    public = replace(legacy, source_kind="public_observation")
    with pytest.raises(ValueError, match="manufacture public"):
        produce_controlled_observations(public, [channels[0]], latent_seed=7, observation_seed=13)


@pytest.mark.parametrize("store_format", ["zarr_v2", "neuronsifter_chunk_store"])
def test_local_membrane_source_transfer_and_normal_tensor_cache(tmp_path, monkeypatch, store_format):
    test_joint_source_transfer_and_normal_tensor_cache_roundtrip(
        tmp_path, monkeypatch, False, store_format=store_format, local_membrane=True)

@pytest.mark.parametrize("valid_prefix", [False, True], ids=["invalid_first", "invalid_later"])
@pytest.mark.parametrize("fault", ["interpolation", "later_time", "future_acquisition", "unbound_error"])
def test_local_membrane_invalid_producer_acquisition_precedes_sampling(monkeypatch, valid_prefix, fault):
    from Libs.scripts.builders import build_sample_observation_registry as producer
    _, support, _, _, channels = joint_case()
    support, channels = local_membrane_case(support, channels)
    bad = copy.deepcopy(channels[2])
    bad["observation_uid"] = "invalid-acquisition"
    if fault == "unbound_error":
        readouts = copy.deepcopy(support.readout_parameters)
        readouts["ephys"] = {key:value for key,value in readouts["ephys"].items() if key in {"physical_readout","units"}}
        support = replace(support, readout_parameters=readouts,
                          source_nuisance={key:value for key,value in support.source_nuisance.items() if key != "ephys"}).validate()
    else:
        changes = {"interpolation":{"mode":"interpolation"}, "later_time":{"measurement_time_ms":[.1]},
                   "future_acquisition":{"acquired_time_ms":[.15]}}
        bad["physical_acquisition"].update(changes[fault])
    # A selected unbound channel can follow a legitimate nonphysical channel;
    # the other later-invalid cases follow a valid physical acquisition.
    prefix = [channels[0] if fault == "unbound_error" else channels[2]] if valid_prefix else []
    draws = {"latent":0, "observation":0}
    latent_sampler = torch.multinomial
    distribution_builder = producer.observation_distribution
    def latent_spy(*args, **kwargs):
        draws["latent"] += 1
        return latent_sampler(*args, **kwargs)
    def distribution_spy(*args, **kwargs):
        distribution = distribution_builder(*args, **kwargs)
        sample = distribution.sample
        def sample_spy(*sample_args, **sample_kwargs):
            draws["observation"] += 1
            return sample(*sample_args, **sample_kwargs)
        distribution.sample = sample_spy
        return distribution
    monkeypatch.setattr(torch, "multinomial", latent_spy)
    monkeypatch.setattr(producer, "observation_distribution", distribution_spy)
    with pytest.raises(ValueError, match="local membrane"):
        producer.produce_controlled_observations(support, [*prefix, bad], latent_seed=7, observation_seed=13)
    assert draws == {"latent":0, "observation":0}


# Exact CPU outputs captured from the pre-repair producer at8e5865c; these
# fixtures test seeded compatibility rather than recomputing the new algorithm.
@pytest.mark.parametrize("case,seeds,index,values", [
    ("legacy", (7,13), 1, [1., 1.268505334854126, 4.672172546386719, 0.]),
    ("legacy", (17,23), 0, [1., 1.0429784059524536, 1.3109278678894043, 0.]),
    ("physical", (7,13), 1, [1., 1.268505334854126, -51.26090621948242, 0.]),
    ("physical", (17,23), 0, [1., 1.0429784059524536, -63.60948944091797, 0.]),
    ("unused_unbound", (7,13), 1, [1., 1.268505334854126, 1.]),
    ("unused_unbound", (17,23), 0, [1., 1.0429784059524536, 0.]),
])
def test_local_membrane_producer_preserves_seeded_values(case, seeds, index, values):
    _, support, _, _, channels = joint_case()
    if case != "legacy":
        support, channels = local_membrane_case(support, channels)
    if case == "unused_unbound":
        readouts = copy.deepcopy(support.readout_parameters)
        readouts["ephys"] = {key:value for key,value in readouts["ephys"].items() if key in {"physical_readout","units"}}
        support = replace(support, readout_parameters=readouts,
                          source_nuisance={key:value for key,value in support.source_nuisance.items() if key != "ephys"}).validate()
        channels = [channels[0], channels[1], channels[3]]
    production = produce_controlled_observations(support, channels, latent_seed=seeds[0], observation_seed=seeds[1])
    assert torch.equal(production.generating_indices, torch.tensor([index]))
    assert torch.equal(production.generating_fields, support.initial_field[:,index])
    assert torch.equal(torch.stack([observation.values for observation in production.observations]).flatten(),
                       torch.tensor(values))
    assert all(observation.observed.all() and not observation.left_censored.any() for observation in production.observations)
    if case == "physical":
        assert production.observations[2].physical_acquisition.to_payload() == channels[2]["physical_acquisition"]


def test_shared_native_fixture_construction_restores_rng_and_repeats(tmp_path):
    from Libs.utils.model.screening_model_config import quantized_mddm_rollout_from_config

    states = []
    with torch.random.fork_rng(devices=[]):
        for caller_seed in (41, 43):
            torch.manual_seed(caller_seed)
            before = torch.random.get_rng_state().clone()
            action_model, patient, regimens, context = _screening_action_case("occupancy_program")
            assert torch.equal(torch.random.get_rng_state(), before)
            assert action_model.field_rollout_operator.quantized_mddm_rollout_operator.sampler_seed == 41
            model, config, _, arrays, source = joint_registry_case(tmp_path / str(caller_seed))
            assert torch.equal(torch.random.get_rng_state(), before)
            backbone = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
            resolved = quantized_mddm_rollout_from_config(config, backbone)
            assert resolved["partial_mask_schedule"]["sampler_seed"] == model.quantized_mddm.sampler_seed == 41
            states.append((
                {key: value.detach().clone() for key, value in action_model.state_dict().items()},
                patient, regimens, context,
                {key: value.detach().clone() for key, value in model.state_dict().items()},
                source["support"], source["observations"], arrays,
            ))
    left, right = states
    for index in (0, 3, 4):
        assert left[index].keys() == right[index].keys()
        assert all(torch.equal(left[index][key], right[index][key]) for key in left[index])
    assert torch.equal(left[1], right[1]) and torch.equal(left[2], right[2])
    assert left[5] == right[5] and left[6] == right[6]
    assert left[7].keys() == right[7].keys()
    assert all((left[7][key] == right[7][key]).all() for key in left[7])
