"""Prospective protocol metadata checks; no source generation or model construction."""
import copy
import shlex

import pytest

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.evaluation.manuscript_native_protocol import (
    validate_definition,method_roster,training_profile,training_budget,command_catalog)
from Libs.utils.model.native_scientific_source import selected_group_definitions
from Libs.utils.model.screening_model_config import quantized_mddm_rollout_from_config


def config():
    return load_config("Libs/configs/exp/manuscript_native_protocol.yaml")


def test_full_roster_source_counts_and_three_calibration_reservations():
    cfg=config();result=validate_definition(cfg)
    assert result["counts"]=={"groups":22144,"cases":63232,"physical":47424,"physical_precision":47424,
        "cns_integrations":189696,"programs":2023424,"chemical_precision":1011712,"source_prepare":1,"parameter_binding":8000,"physical_refinement":750,"operations":3350575}
    assert result["defined_training_source_contract_count"]==2
    assert result["multiple_training_source_contracts_defined"] is True
    assert result["source_admission_evaluated"] is False
    assert "admitted_training_source_contract_count" not in result
    assert result["calibration_group_counts"]=={"fit":2048,"check":2048,"threshold":2048}
    sets={role:{row["sample_uid"] for row in selected_group_definitions(cfg,"calibration",role)}
          for role in ("fit","check","threshold")}
    assert not sets["fit"]&sets["threshold"] and not sets["check"]&sets["threshold"]
    rows={row["method_id"]:row for row in method_roster(cfg)}
    assert rows["native_wam"]["software_interface_bound"]
    for name in ("planet","dreamer","dreamerv3"):
        assert rows[name]["software_interface_bound"]
        assert rows[name]["implementation_scope"]=="RQ2_predictive_world_model_component"
        assert rows[name]["later_gate_obligations"]["status"]=="NOT_IMPLEMENTED"
        assert rows[name]["prerequisite_gate"]=="ScientificTwin"
    for name in ("gru","native_deterministic","flattened_parameter_matched","fno"):
        assert rows[name]["software_interface_bound"]
    assert rows["scvi"]["implementation_kind"]=="empirical_prerequisite"
    assert rows["numerical_decision_bed"]["implementation_kind"]=="downstream_gate"
    assert rows["native_scalar_exposure"]["domain_scope"]=="chemical_program_domains_only"
    missing=copy.deepcopy(cfg);missing["manuscript_native_protocol"]["methods"].pop()
    with pytest.raises(ValueError,match="roster"):validate_definition(missing)
    missing=copy.deepcopy(cfg);missing["native_scientific_training"]["objectives"]["crps"]=None
    with pytest.raises(ValueError,match="objective"):validate_definition(missing)


def test_effective_profiles_bind_seed_steps_capacity_conditioning_and_real_commands():
    cfg=config();budget=training_budget(cfg,25)
    full,full_spec=training_profile(cfg,"native_wam",42)
    reduced,reduced_spec=training_profile(cfg,"native_wam",42,25,budget=budget)
    ablation,ablation_spec=training_profile(cfg,"native_target_id",42)
    for current in (full,reduced,ablation):
        train=current["screening_pipeline"]["screening_benchmark"]["training"]
        native=current["native_scientific_training"]
        backbone=current["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
        sampler=quantized_mddm_rollout_from_config(current,backbone)["partial_mask_schedule"]
        assert current["selected_formal_seed"]==train["random_seed"]==sampler["sampler_seed"]==42
        assert (backbone["operator_hidden_dim"],backbone["operator_depth"],backbone["attention_head_count"],backbone["mechanism_state_token_count"])==(696,8,12,96)
        assert native["selection"]["max_steps"]==12600 and native["validation"]["interval_optimizer_steps"]==63
        assert train["memory_policy"]["effective_batch_size"]==64
        assert sampler["inference_sample_count"]==2 and sampler["inference_denoising_step_count"]==100
        assert train["zarr_loader"]["formal_artifact_subdir_template"]==""
    assert reduced["native_scientific_training"]["selection"]["max_epochs"]==788
    assert full["native_scientific_training"]["conditioning"]=={"mode":"joint"}
    assert ablation["native_scientific_training"]["conditioning"]["manifest_path"]==full_spec["training_manifest"]
    native=ablation["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]
    assert native["action_representation"]["active_policy"]=="target_id"
    commands=command_catalog(cfg,[full_spec,reduced_spec,ablation_spec])
    for row in commands:assert shlex.split(row["shell_command"])==row["argv"] and row["dispatch_authorized"] is False
    fits=[row for row in commands if row["command_uid"].endswith(":fit")]
    assert all("--seed" not in row["argv"] and "--output-dir" not in row["argv"] for row in fits)
    evaluation=next(row for row in commands if row["command_uid"]==reduced_spec["profile_id"]+":evaluate")
    assert evaluation["argv"][evaluation["argv"].index("--conditioning-manifest")+1]==reduced_spec["training_manifest"]
    assert full_spec["profile_id"]+":fit" in fits[-1]["dependencies"]
    _,solver_spec=training_profile(cfg,"same_compiler_solver",42)
    assert solver_spec["training_manifest"] is None and solver_spec["additional_optimizer_updates"]==0
    assert solver_spec["transition_fit"]=="not_applicable_frozen_primary_rhs"
    solver_commands=command_catalog(cfg,[solver_spec])
    assert not any(row["command_uid"].endswith(":fit") for row in solver_commands)
    solver_evaluation=next(row for row in solver_commands if row["command_uid"]==solver_spec["profile_id"]+":evaluate")
    assert solver_evaluation["conditional_paths_per_particle"]==1
    assert "--training-manifest" not in solver_evaluation["argv"]
    with pytest.raises(ValueError,match="native seeds"):training_profile(cfg,"native_wam",3407)


def test_chemical_scope_changes_pool_not_fraction_and_keeps_one_matched_reference(monkeypatch):
    from Libs.exp.training import pretrain_microenvironment_world_model as trainer
    from Libs.utils.model.native_scientific_source import training_data_scope
    cfg=config()
    specifications=[]
    for name in ("native_chemical_reference","native_action_id","native_target_id","native_scalar_exposure"):
        selected,row=training_profile(cfg,name,41)
        specifications.append(row)
        assert row["budget"]["selected_training_groups"]==2000
        assert row["budget"]["training_subset_percentage"]==100
        assert row["budget"]["max_epochs"]==394 and row["budget"]["max_optimizer_updates"]==12600
        assert row["budget"]["planned_group_visits_at_update_cap_without_early_stop"]==787536
        assert len(selected_group_definitions(selected,"development"))==400
        assert len(selected_group_definitions(selected,"calibration","fit"))==1024
        assert len(selected_group_definitions(selected,"calibration","check"))==1024
        assert len(selected_group_definitions(selected,"calibration","threshold"))==1024
    assert len({row["conditioning_manifest"] for row in specifications})==1
    assert training_data_scope(selected)!=training_data_scope(cfg)
    teachers={name:{"teachers":[{"sample_uid":name,"split_name":"train","regimen_uid":a,"posterior_uid":name+":p"}
              for a in ("a","b")]} for name in ("chemical","stimulation")}
    monkeypatch.setattr(trainer,"load_family_registries",lambda config, *, input_locations=None:teachers)
    fixture={"native_diffusion_data":{},"native_scientific_training":{"family_scope":["chemical"]},
        "screening_pipeline":{"screening_benchmark":{"training":{"teacher_family_training":{
            "families":[{"family_key":name,"teacher_loss_weight":1.} for name in teachers]}}}}}
    dataset=trainer.NativeJointGroupDataset(fixture,"train",group_uids={"chemical"})
    assert dataset.family_group_counts=={"chemical":1}
    with pytest.raises(ValueError,match="nonempty|coverage|selection|admitted|groups"):
        trainer.NativeJointGroupDataset(fixture,"train",group_uids={"stimulation"})


@pytest.mark.parametrize("field,value", [
    ("horizon_transitions", 4), ("warmup_transitions", 2), ("buffer_transitions", 4),
    ("updates_per_observation", 0), ("updates_per_observation", True),
    ("prior", "current_log_weights"), ("objective", "full_state_MSE"),
    ("update_scope", "live_state_correction"), ("network_mode", "train"),
    ("failure_policy", "retry_best_loss"), ("optimizer", None),
])
def test_online_policy_rejects_invalid_budgets_or_changed_boundaries(field, value):
    from Libs.models.modules.native_mechanistic_residual import native_residual_online_policy
    cfg = config()
    cfg["native_mechanistic_residual_online"][field] = value
    with pytest.raises(ValueError, match="online"):
        native_residual_online_policy(cfg)


@pytest.mark.parametrize("field,value", [("learning_rate", float("nan")), ("epsilon", 0),
    ("betas", [.9, 1]), ("name", "SGD"), ("weight_decay", .1)])
def test_online_policy_rejects_invalid_optimizer(field, value):
    from Libs.models.modules.native_mechanistic_residual import native_residual_online_policy
    cfg = config()
    cfg["native_mechanistic_residual_online"]["optimizer"][field] = value
    with pytest.raises(ValueError, match="optimizer"):
        native_residual_online_policy(cfg)


def test_protocol_requires_online_policy_but_residual_MPC_stays_open():
    cfg = config()
    row = next(row for row in method_roster(cfg) if row["method_id"] == "residual_mpc")
    assert row["software_interface_bound"] is False and row["implementation_blocker"]
    cfg.pop("native_mechanistic_residual_online")
    with pytest.raises(ValueError, match="online policy"):
        validate_definition(cfg)


@pytest.mark.parametrize("field,value", [("prefix", "each_old_hypothetical_action"),
    ("molecular_budget", "delivered_dose"), ("rhs_budget", "reset_per_branch"),
    ("activation", "physical_plant")])
def test_future_control_protocol_rejects_changed_boundaries(field, value):
    from Libs.utils.model.native_future_controls import future_control_policy
    cfg = config()
    cfg["native_future_controls"][field] = value
    with pytest.raises(ValueError, match="future-control"):
        future_control_policy(cfg)


def test_future_activation_definition_requires_real_public_execution_binding():
    from Libs.utils.data.raw_data_registry import load_config
    from Libs.utils.model.native_future_controls import future_control_policy
    config=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    assert future_control_policy(config)["activation"] == "requires_verified_public_execution_binding"
    config["native_future_controls"]["activation"]="unsupported_model_planning_only"
    assert future_control_policy(config)["activation"] == "unsupported_model_planning_only"



def test_native_training_evidence_command_uses_existing_full_fit_dependencies():
    cfg = config()
    profiles = [training_profile(cfg, name, 41, fraction)[1] for name, fraction in
                [("native_wam", 100), ("native_wam", 25), ("large", 100), ("planet", 100), ("dreamer", 100)]]
    commands = command_catalog(cfg, profiles)
    audit = next(row for row in commands if row["command_uid"] == "manuscript_native_training_evidence")
    assert audit["dependencies"] == ["native_wam_fraction_100_seed_41:fit", "planet_fraction_100_seed_41:fit"]
    assert audit["dispatch_authorized"] is False and audit["sufficient_training"] is False
    assert audit["scientific_twin"] == "NOT_RUN"
    assert audit["argv"][audit["argv"].index("--config") + 1] == cfg["manuscript_native_protocol"]["config_path"]
    assert "--protocol-scope" in audit["argv"] and "manuscript-native" in audit["argv"]


@pytest.mark.parametrize("seed", [39,45,True])
def test_active_protocol_rejects_undeclared_seed_without_profile(seed):
    with pytest.raises(ValueError, match="native seeds"):
        training_profile(config(), "native_wam", seed)


@pytest.mark.parametrize("seed_policy,seeds", [
    ("manuscript_native_five", [40,41,42,43,44]),
    ("manuscript_native_three", [41,42,43]),
], ids=["historical_five", "canonical_three"])
def test_declared_seed_every_method_profile_budget_and_statistics_identity(seed_policy, seeds):
    cfg = config(); protocol = cfg["manuscript_native_protocol"]
    assert protocol["training_seeds"] == [41,42,43]
    assert protocol["seed_policy"] == "manuscript_native_three"
    assert cfg["manuscript_native_statistics"]["training_seeds"] == [41,42,43]
    if seed_policy == "manuscript_native_five":
        # Explicit historical compatibility fixture; no new execution approval.
        protocol.update(seed_policy=seed_policy, training_seeds=seeds,
            protocol_uid="fixture_historical_five_seed", profile_root="fixture_historical_five_seed/profiles",
            execution_root="fixture_historical_five_seed/training")
        protocol["training_evidence"]["approved_execution_profiles"] = {}
        cfg["manuscript_native_statistics"].update(seed_policy=seed_policy, training_seeds=seeds)
    for row in method_roster(cfg):
        if not row["software_interface_bound"]: continue
        for seed in seeds:
            actual, spec = training_profile(cfg, row["method_id"], seed)
            assert spec["training_seed_binding"]["allowed_training_seeds"] == seeds
            assert spec["training_seed_binding"]["seed_policy"] == seed_policy
            assert spec["output_root"].startswith(protocol["execution_root"])
            assert spec["config_path"].startswith(protocol["profile_root"])
    assert training_budget(cfg,100)["minimum_optimizer_updates"] == 10000
    changed=copy.deepcopy(cfg)
    changed["manuscript_native_statistics"]["training_seeds"] = [41,42,43] if len(seeds) == 5 else [40,41,42,43,44]
    with pytest.raises(ValueError, match="seed"):
        validate_definition(changed)
