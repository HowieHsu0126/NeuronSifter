"""Prospective group/prior/action arithmetic on tiny disposable source inputs."""
import copy
import itertools
import json
from pathlib import Path

import pytest

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.model import native_scientific_source as source


def corpus_fixture():
    config = load_config("Libs/configs/exp/native_observable_bridge.yaml")
    settings = config["native_observable_bridge"]
    settings["scientific_profile"] = True
    settings["source_uid"] = "disposable_scientific_source"
    settings["snapshot_ref"] = "disposable_scientific_definition"
    settings["atoms"] = [list(row) for row in itertools.product((0, 1), repeat=5)]
    settings["prior_weights"] = [1/32]*32
    settings["action_variants"] = [
        {"intensity_multiplier":.5, "duration_multiplier":1., "response_delay_steps":1},
        {"intensity_multiplier":1., "duration_multiplier":1., "response_delay_steps":1},
        {"intensity_multiplier":1.5, "duration_multiplier":.5, "response_delay_steps":2},
    ]
    settings["group_partitions"] = [
        {"partition_uid":"train", "role":"train", "groups_per_family":1, "action_indices":[0,1,2]},
        {"partition_uid":"development", "role":"development", "groups_per_family":1, "action_indices":[0,2]},
        {"partition_uid":"calibration_fit", "role":"calibration", "calibration_reservation":"fit",
         "groups_per_family":1, "action_indices":[0,1,2]},
        {"partition_uid":"calibration_check", "role":"calibration", "calibration_reservation":"check",
         "groups_per_family":1, "action_indices":[0,1,2]},
    ]
    settings["budget"].update(source.planned_operation_counts(config))
    return config


def test_whole_prior_group_action_and_operation_denominators(tmp_path):
    from Libs.scripts.builders.build_controlled_native_source import ObservableSourceLedger
    config = corpus_fixture()
    groups = source.group_definitions(config)
    counts = source.planned_operation_counts(config)
    assert len(groups) == 12 and counts["cases"] == 33
    assert counts["programs"] == 1056 and counts["chemical_precision"] == 352
    assert counts["operations"] == 1585
    assert len({group["sample_uid"] for group in groups}) == 12
    assert all(len(group["particle_uids"]) == 32 for group in groups)
    ledger = ObservableSourceLedger(tmp_path/"ledger.json", config, {"fixture":True})
    assert len(ledger.data["cases"]) == 33 and len(ledger.data["operations"]) == 1585
    assert all(row["status"] == "planned" for row in ledger.data["operations"].values())
    invalid = copy.deepcopy(config)
    invalid["native_observable_bridge"]["budget"]["programs"] -= 1
    with pytest.raises(ValueError, match="operation counts"):
        ObservableSourceLedger(tmp_path/"invalid.json", invalid, {"fixture":True})


def test_original_definition_keeps_its_exact_case_and_operation_counts():
    config = load_config("Libs/configs/exp/native_observable_bridge.yaml")
    groups = source.group_definitions(config)
    assert source.planned_operation_counts(config) == {
        "groups":18, "cases":36, "physical":36, "physical_precision":36,
        "cns_integrations":108, "programs":144, "chemical_precision":48, "operations":390}
    assert groups[0]["group_uid"].endswith(":electrophysiology_conductance_world:0")
    assert set(groups[0]) == {"group_uid", "sample_uid", "support_uid", "posterior_uid",
                             "particle_uids", "family_uid", "replicate", "role", "calibration_reservation"}


def test_complete_prior_draw_and_three_action_source_are_admissible():
    from Libs.utils.data.native_diffusion_data import parse_joint_source
    config = corpus_fixture()
    group = source.group_definitions(config)[0]
    generated = source.draw_group(config, group, 0)
    assert generated["support"].prior_weights.shape == (1, 32)
    assert len(generated["parameters"]) == 32
    assert 0 <= generated["hidden_index"] < 32
    actions = [source.action_definition(config, group["family_uid"], i, group) for i in range(3)]
    records = [[source.compiled_program(config, group, generated["support"], atom,
        generated["parameters"][atom], action, "disposable_code_identity") for action in actions]
        for atom in range(32)]
    raw = source.joint_source(config, group, generated, records)
    support, observations = parse_joint_source(raw)
    assert support.prior_weights.shape == (1, 32) and len(observations) == 1
    assert len(raw["particle_programs"]["records"][0]) == 32
    assert len(raw["particle_programs"]["regimen_uids"]) == 3
    assert "hidden_index" not in raw and "parameters" not in raw


@pytest.mark.parametrize("change", ["fractional_groups", "duplicate_atoms", "missing_calibration_role", "duplicate_actions"])
def test_invalid_prospective_design_rejects_before_drawing(change):
    config = corpus_fixture()
    settings = config["native_observable_bridge"]
    if change == "fractional_groups":
        settings["group_partitions"][0]["groups_per_family"] = 1.5
    elif change == "duplicate_atoms":
        settings["atoms"][1] = settings["atoms"][0]
    elif change == "missing_calibration_role":
        settings["group_partitions"][-1].pop("calibration_reservation")
    else:
        settings["group_partitions"][0]["action_indices"] = [0,0]
    with pytest.raises(ValueError):
        source.group_definitions(config)


def long_cns_fixture():
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    settings = config["native_observable_bridge"]
    settings["source_uid"], settings["snapshot_ref"] = "disposable_long_cns", "disposable_long_cns_definition"
    settings["source_variants"] = {}
    settings["atoms"] = [[0,0,0,0,0],[0,1,1,0,1],[1,0,1,1,0],[1,1,0,1,1]]
    settings["prior_weights"] = [.1,.2,.3,.4]
    settings["long_cns_family"]["actions"] = [
        {"dose_times_hours":[0.],"fractions":[1.]},
        {"dose_times_hours":[0.,336.],"fractions":[.5,.5]}]
    settings["group_partitions"] = [{"partition_uid":"train","role":"train","groups_per_family":1,"action_indices":[0,1]}]
    settings["budget"].update(source.planned_operation_counts(config))
    group = next(row for row in source.group_definitions(config) if source.is_long_cns(config,row["family_uid"]))
    return config, group


def test_long_cns_profile_and_real_site_parameter_block():
    import torch
    from Libs.utils.data.joint_physical_parameters import (
        parameter_schema_from_config, cns_pk_parameter_declarations, flatten_cns_pk_parameters)
    from Libs.utils.data.native_observable_profile import observable_profile, family_grid, quantity_support, observable_readout
    config, group = long_cns_fixture()
    old = load_config("Libs/configs/exp/native_observable_bridge.yaml")
    profile = observable_profile(config)
    for family, definition in observable_profile(old)["families"].items():
        assert profile["families"][family] == definition
    schema = parameter_schema_from_config(config)
    assert sum(len(rows) for rows in schema["families"].values()) == 214
    assert len(schema["families"][group["family_uid"]]) == 180
    grid = family_grid(profile,group["family_uid"])
    mask = quantity_support(grid)
    assert len(grid["time_hours"]) == 28 and grid["time_hours"][-1] == 672
    assert mask[...,:15].all() and not mask[...,15:].any()
    # Readout is a pure selection of source-unit fields, without fabricated auxiliaries.
    field = torch.arange(28*6*6*26,dtype=torch.float32).reshape(1,28,6,6,26)
    readout = observable_readout(field,torch.ones_like(field),grid)
    assert readout["predictive_mean"].shape == (1,28,540)
    assert torch.equal(readout["predictive_mean"],field[...,:15].flatten(2))
    changed = copy.deepcopy(config)
    changed["controlled_cns_source"]["sites"][0], changed["controlled_cns_source"]["sites"][1] = (
        changed["controlled_cns_source"]["sites"][1],changed["controlled_cns_source"]["sites"][0])
    with pytest.raises(ValueError,match="site order"):
        cns_pk_parameter_declarations(changed)
    generated = source.draw_group(config,group,0)
    law = source.long_cns_law(config,group,generated["support"],0)
    values = copy.deepcopy(law.pk)
    values["delivery_fraction"][0] = 0.
    assert flatten_cns_pk_parameters(values)["delivery_fraction[0]"] == 0.


def test_long_cns_complete_same_atom_program_and_present_zero(tmp_path):
    import torch
    from Libs.utils.model import controlled_cns_source as cns
    from Libs.utils.data.native_diffusion_data import parse_joint_source
    from Libs.utils.data.joint_physical_parameters import bind_candidate_parameters, candidate_conditioning, parameter_schema_from_config
    config, group = long_cns_fixture()
    generated = source.draw_group(config,group,0)
    definition = source.long_cns_definition(config)
    actions = [source.action_definition(config,group["family_uid"],i,group) for i in range(2)]
    records = []
    for atom in range(source.prior_particle_count(config)):
        law = source.long_cns_law(config,group,generated["support"],atom)
        row = []
        for action in actions:
            pk = cns.pk_values(definition["grid"]["time_hours"],law.pk,action,definition)
            row.append(cns.compiled_particle_program(definition,group,generated["support"],atom,
                action,pk,law.pk,"1"*64))
        records.append(row)
    raw = source.joint_source(config,group,generated,records)
    schema = parameter_schema_from_config(config)
    bound = bind_candidate_parameters(raw,group,{"parameters":generated["parameters"]},schema,
        record_ref=str(tmp_path/"all_candidates.json"),producer_commit="disposable")
    support, observations = parse_joint_source(bound)
    encoded = candidate_conditioning(support,schema)
    assert encoded.shape == (1,4,428)
    assert encoded[...,:34].eq(0).all() and encoded[...,214:248].eq(0).all()
    assert encoded[...,248:].eq(1).all()
    assert len(observations) == 1 and not {"hidden_index","future_targets"} & bound.keys()
    mutated = copy.deepcopy(bound)
    mutated["support"]["physical_parameter_hypotheses"]["values"][0][0][0] *= 1.1
    with pytest.raises(ValueError,match="Named candidate parameter disagrees"):
        parse_joint_source(mutated)
    # Zero and missing are distinct at the typed candidate interface. This
    # separate table is not admitted as a changed numerical program.
    support.physical_parameter_hypotheses["values"][0,0,36] = 0.
    encoded = candidate_conditioning(support,schema)
    assert encoded[0,0,34+36] == 0 and encoded[0,0,214+34+36] == 1


def test_actual_long_cns_solver_keeps_clock_and_dose_events():
    import numpy as np
    from Libs.utils.model import controlled_cns_source as cns
    config, group = long_cns_fixture()
    generated = source.draw_group(config,group,0)
    law = source.long_cns_law(config,group,generated["support"],0)
    definition = source.long_cns_definition(config)
    bolus, split = definition["actions"][:2]
    reference = cns.integrate(law,split,"DOP853",definition["numerics"]["reference_max_step_hours"])
    assert reference["trajectory"].shape == (28,6,6,15)
    assert np.isfinite(reference["trajectory"]).all()
    assert (reference["trajectory"][0,...,0] == 0).all()
    assert max(reference[key] for key in ("balance_max","paired_balance_max","transport_balance_max")) < definition["numerics"]["balance_tolerance"]
    split_c = cns.pk_values([0.,336.],law.pk,split,definition)[0]
    bolus_c = cns.pk_values([0.,336.],law.pk,bolus,definition)[0]
    assert np.all(split_c[0] == 0) and np.all(split_c[1] > bolus_c[1])


def test_scientific_corpus_counts_ood_roles_and_nested_training_subsets():
    from collections import Counter
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    groups = source.group_definitions(config)
    counts = source.planned_operation_counts(config)
    assert counts == {"groups":22144,"cases":63232,"physical":47424,"physical_precision":47424,
        "cns_integrations":189696,"programs":2023424,"chemical_precision":1011712,
        "source_prepare":1,"parameter_binding":8000,"physical_refinement":750,"operations":3350575}
    assert all(config["native_observable_bridge"]["budget"][key] == value for key,value in counts.items())
    assert len({row["group_uid"] for row in groups}) == counts["groups"]
    held = [row for row in groups if row["source_variant"] == "held"]
    assert len(held) == 1600 and all(row["role"] == "validation" for row in held)
    assert set(row["ood_class"] for row in groups) == {
        "in_distribution","parametric","intervention","combination","observation_shift","structural","source_holdout"}
    for family in {row["family_uid"] for row in groups}:
        training = [row for row in groups if row["family_uid"] == family and row["role"] == "train"]
        assert [sum(fraction in row["training_subset_percentages"] for row in training) for fraction in (10,25,50,100)] == [100,250,500,1000]
        assert Counter(row["condition"] for row in training) == {"control":500,"ad_positive":500}
    bad = copy.deepcopy(config)
    bad["native_observable_bridge"]["group_partitions"][-1]["role"] = "train"
    with pytest.raises(ValueError,match="Source variant"):
        source.group_definitions(bad)


def held_law_fixture(family):
    config, _ = long_cns_fixture()
    settings = config["native_observable_bridge"]
    settings["source_variants"] = {
        "held":{"source_uid":"disposable_held_law","snapshot_ref":"disposable_held_law_definition",
                "law":"cooperative_exponential_source","roles":["validation"],
                "observation_standard_deviation":.06,
                "observation_matrix":[[.9 if i==j else .1 if j==(i+1)%6 else 0. for j in range(6)] for i in range(6)]}}
    settings["group_partitions"] = [{"partition_uid":"held","role":"validation","groups_per_family":1,
                                      "action_indices":[0,1],"source_variant":"held"}]
    group = next(row for row in source.group_definitions(config) if row["family_uid"] == family)
    scoped = source.scoped_group_config(config,group)
    generated = source.draw_group(scoped,group,0)
    return config,scoped,group,generated


@pytest.mark.parametrize("family",[source.ELECTRICAL,source.SYNAPTIC,source.MOLECULAR])
def test_held_source_changes_law_and_measurement_with_same_clock(family):
    import numpy as np
    from Libs.utils.model.native_observable_source import physical_errors
    _,config,group,generated = held_law_fixture(family)
    action = source.action_definition(config,family,0,group)
    parameters = generated["parameters"][0]
    primary = source.physical_trajectory(config,family,parameters,action)
    precise = source.physical_trajectory(config,family,parameters,action,high_precision=True)
    original = source.original_physical_trajectory(config,family,parameters,action)
    assert primary["values"].shape == original["values"].shape
    assert np.max(np.abs(primary["values"]-original["values"])) > 1e-7
    assert physical_errors(primary,precise)[0].max() < 1e-10
    support = generated["support"]
    assert support.source_uid == "disposable_held_law" and support.role == "validation"
    observation = config["controlled_cns_source"]["observation"]
    selectors = observation["selector"]
    names = config["controlled_cns_source"]["grid"]["species_id"]
    direct = np.array([[support.initial_field[0,atom,row["region_index"],row["compartment_index"],names.index(row["species"])].item()
                        /config["controlled_cns_source"]["reference_scales"][row["species"]]
                        for row in selectors] for atom in range(4)])
    expected = direct @ np.array(config["native_observable_bridge"]["observation_matrix"]).T
    np.testing.assert_allclose(support.readout_parameters["controlled_predose_selector"]["mean"][0],expected,rtol=1e-6)
    assert not np.allclose(direct,expected)


def test_independent_held_cns_reference_uses_different_equations_and_solver():
    import numpy as np
    from Libs.utils.model import controlled_cns_source as cns
    from Libs.utils.model import native_held_source as held
    _,config,group,generated = held_law_fixture("cns_long_horizon_world")
    action = source.action_definition(config,group["family_uid"],1,group)
    law = source.long_cns_law(config,group,generated["support"],0,action)
    assert isinstance(law,held.HeldCNSLaw)
    reference = source.integrate_long_reference(law,action,"DOP853",.25,float("inf"))
    ordinary = cns.ControlledLaw(law.source,law.factors,law.initial)
    indices = [law.source["grid"]["species_id"].index(name) for name in cns.DYNAMIC]
    state = law.initial.reshape(36,15)[:,indices]
    assert any(not np.allclose(a,b) for a,b in zip(law.reaction(state,np.zeros(36)),ordinary.reaction(state,np.zeros(36))))
    assert reference["trajectory"].shape == (28,6,6,15)
    assert np.isfinite(reference["trajectory"]).all()
    assert max(reference[key] for key in ("balance_max","paired_balance_max","transport_balance_max")) < 1e-9


def test_scientific_source_to_admission_preserves_all_four_families_and_held_source(tmp_path, monkeypatch):
    from Libs.tests.test_native_observable_source import build_disposable_observable_fixture
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    from Libs.utils.data.joint_physical_parameters import parameter_schema_from_config
    config, _, root, effective, _ = build_disposable_observable_fixture(tmp_path,monkeypatch,scientific=True)
    check_streaming_fit_parity_and_isolation(effective, root)
    declared = source.group_definitions(config)
    train = [row for row in declared if row["role"] == "train"]
    primary = [row["sample_uid"] for row in train if row.get("source_stratum") == "shared_primary"]
    romani = [row["sample_uid"] for row in train if row.get("source_stratum") == "romani"]
    other = {row["sample_uid"] for row in train if row["family_uid"] != source.ELECTRICAL}
    excluded = {row["sample_uid"] for row in declared if row["role"] != "train"}
    assert len(primary) == len(romani) == 2 and len(other) == 12
    expected_scopes = {
        "pure_shared": other | set(primary),
        "mixed_shared": other | {primary[0], romani[1]},
    }
    normalization_root = Path(config["manuscript_native_protocol"]["source_comparison"]["normalization_root"])
    assert normalization_root.is_relative_to(tmp_path)
    for scope, expected in expected_scopes.items():
        artifact = json.loads((normalization_root/(scope+".json")).read_text())
        rows = artifact["training_records"]
        actual = {row["sample_uid"] for row in rows}
        assert len(expected) == 14 and artifact["trajectory_count"] == len(rows) == 28
        assert actual == expected and not actual & excluded
        assert all(row["split_name"] == row["source_holdout_role"] == "train" for row in rows)
    training = pretrain.NativeJointGroupDataset(effective,"train")
    development = pretrain.NativeJointGroupDataset(effective,"development")
    evaluation = pretrain.NativeJointGroupDataset(effective,"validation")
    assert len(training) == 16
    assert len(development) == len(evaluation) == 4
    assert set(training.family_group_counts.values()) == {4}
    schema = parameter_schema_from_config(effective)
    assert sum(len(rows) for rows in schema["families"].values()) == 214
    assert [len(row["declarations"]) for row in schema["source_parameter_blocks"]] == [7]
    assert all(evaluation[index]["native_joint_group"]["joint_source"]["support"]["source_uid"] == "disposable_held_builder"
               for index in range(len(evaluation)))
    long_index = next(index for index, (child, _, _) in enumerate(training.groups)
                      if child.native_registry["grid"]["family_uid"] == "cns_long_horizon_world")
    check_long_cns_training_and_sampling(effective, training[long_index]["native_joint_group"])


def check_long_cns_training_and_sampling(config, group):
    """Tiny real paired loss and target-free sampling on the admitted 672 h field."""
    import torch
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    config = copy.deepcopy(config)
    backbone = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    backbone.update(operator_hidden_dim=12, operator_depth=1, attention_head_count=3, mechanism_state_token_count=6)
    config["native_diffusion_data"]["posterior_context_width"] = 12
    torch.manual_seed(973)
    model = pretrain.MicroenvironmentPretrainingModel(config, False)
    model.quantized_mddm.training_sample_count = 2
    model.quantized_mddm.training_denoising_step_count = 1
    model.quantized_mddm.inference_sample_count = 2
    model.quantized_mddm.inference_denoising_step_count = 1
    model.quantized_mddm.sampler_seed = 973
    assert model.field_normalization.active
    assert all(item["trajectory"].shape == (28, 6, 6, 26) for item in group["items"])
    model.train()
    loss, metrics, status = pretrain.scientific_joint_group_loss(model, group, config, torch.device("cpu"))
    loss.backward()
    assert status["family_uid"] == "cns_long_horizon_world"
    assert status["particle_count"] == 4 and status["paired_target_count"] == 2
    assert len(status["expert_utilization"]) == 10
    projection = model.world_state_posterior.joint_candidate_parameter_projection
    assert projection.in_features == 2*(214+7)  # Family and Romani values plus their masks.
    assert projection.weight.grad[:, 34:214].abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    captured = []

    def capture(_module, _args, output):
        draws = output["sample_trajectories"]
        assert draws.shape == (2, 2, 1, 28, 6, 6, 26)
        assert torch.isfinite(draws).all() and draws.grad_fn is None
        captured.append(draws.clone())

    hook = model.quantized_mddm.register_forward_hook(capture)
    model.eval()
    try:
        with torch.no_grad():
            _, first, _ = pretrain.scientific_joint_group_loss(model, group, config, torch.device("cpu"))
            reference = captured[:]
            captured.clear()
            changed = copy.deepcopy(group)
            for item in changed["items"]:
                item["trajectory"][1:] += .01 * item["quantity_support"][1:]
            _, second, _ = pretrain.scientific_joint_group_loss(model, changed, config, torch.device("cpu"))
    finally:
        hook.remove()
    assert len(reference) == len(captured) == 4
    assert all(torch.equal(a, b) for a, b in zip(reference, captured))
    assert first["crps"] != second["crps"]


def test_scientific_ledger_persists_bounded_group_starts_and_unknown_interruption(tmp_path):
    import json
    from Libs.scripts.builders.build_controlled_native_source import ObservableSourceLedger
    config, _ = long_cns_fixture()
    ledger = ObservableSourceLedger(tmp_path/"source_receipt.json", config, {"fixture": True})
    root = json.loads(ledger.path.read_text())
    assert root["source_status"] == "running" and root["operations"] == {}
    assert root["planned_operation_count"] == len(ledger.data["operations"])
    def interrupted():
        snapshot = json.loads((tmp_path/"group_0_receipt.json").read_text())
        assert snapshot["operations"]["group:0"]["status"] == "started"
        assert len(snapshot["operations"]) < len(ledger.data["operations"])
        assert snapshot["interrupted_started_operation"] == "consumed_outcome_unknown_no_replay"
        raise RuntimeError("disposable interruption")
    with pytest.raises(RuntimeError, match="disposable interruption"):
        ledger.execute("group:0", interrupted)
    assert json.loads((tmp_path/"group_0_receipt.json").read_text())["operations"]["group:0"]["status"] == "failed"
    with pytest.raises(RuntimeError, match="already consumed"):
        ledger.execute("group:0", lambda: None)
    ledger.data["source_status"] = "failed"
    ledger.save()
    terminal = json.loads(ledger.path.read_text())
    assert len(terminal["operations"]) == len(ledger.data["operations"])
    assert terminal["operations"]["group:0"]["status"] == "failed"


def check_streaming_fit_parity_and_isolation(effective, root):
    import json
    import numpy as np
    from Libs.utils.data import native_diffusion_data as native
    from Libs.utils.data.native_field_normalization import fit_multigrid
    config = copy.deepcopy(effective)
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"].pop("physical_parameter_conditioning")
    config["native_diffusion_data"]["registry_path"] = str(root/"native_registry.json")
    batches = []
    for registry in native.load_family_registries(config).values():
        records = native.select_records(registry, config, "train")
        arrays, _ = native.load_records(registry, records, config)
        batches.append((registry["grid"], records, arrays))
    dense = fit_multigrid(batches, config)
    # A one-shot iterable cannot be revisited or retained as a second-pass source.
    class Once:
        def __init__(self, values):
            self.values, self.used = values, False
        def __iter__(self):
            assert not self.used
            self.used = True
            yield from self.values
    streamed = fit_multigrid(Once(batches), config, streaming=True)
    saved = json.loads((root/"train_normalization.json").read_text())
    for value in (streamed, saved):
        for key in ("mean", "population_sd", "scale", "weighted_sum", "weighted_m2", "haar_positive_max", "haar_negative_max"):
            np.testing.assert_allclose(value[key], dense[key], rtol=1e-10, atol=1e-25, err_msg=key)
        assert value["trajectory_count"] == 32
        assert all(row["split_name"] == row["source_holdout_role"] == "train" for row in value["training_records"])
    grid, records, arrays = batches[0]
    changed = copy.deepcopy(records)
    changed[0]["teacher"]["split_name"] = "validation"
    with pytest.raises(ValueError, match="training A/B"):
        fit_multigrid(Once([(grid, changed, arrays), *batches[1:]]), config, streaming=True)


def test_scientific_source_verifier_binds_declared_groups_and_retained_paths(tmp_path):
    from Libs.scripts.builders.build_controlled_native_source import ObservableSourceLedger, observable_code_identity, verify_observable_source
    from Libs.utils.data.physical_occupancy import digest
    config, _ = long_cns_fixture()
    settings = config["native_observable_bridge"]
    settings["output_root"] = str(tmp_path)
    settings["producer_freeze_paths"] = []  # This unit tests metadata, not a production Git freeze.
    source_file, code_hash = observable_code_identity(settings, {})
    identity = {
        "definition_sha256": digest({"profile": settings, "background": config["controlled_cns_source"]}),
        "bound_files": {}, "source_file": source_file, "source_code_sha256": code_hash,
    }
    ledger = ObservableSourceLedger(tmp_path/"receipt.json", config, identity)
    receipt = copy.deepcopy(ledger.data)
    receipt["source_status"] = "complete"
    for operation in receipt["operations"].values():
        operation["status"] = "complete"
    for gi, group in enumerate(receipt["groups"]):
        group["conditioning_reference"] = {"source_posterior_uid": group["posterior_uid"],
                                           "path": str(tmp_path/f"group_{gi}_conditioning.json.gz")}
    verify_observable_source(receipt, config)
    changed = copy.deepcopy(receipt)
    changed["source_identity"]["source_code_sha256"] = "0"*64
    with pytest.raises(ValueError, match="source code identity"):
        verify_observable_source(changed, config)
    changed = copy.deepcopy(receipt)
    changed["groups"][0]["conditioning_reference"]["path"] = str(tmp_path/"temporary-staging.json.gz")
    with pytest.raises(ValueError, match="retained output path"):
        verify_observable_source(changed, config)
    changed = copy.deepcopy(receipt)
    changed["groups"][0]["group_uid"] += ":changed"
    with pytest.raises(ValueError, match="declared group identity"):
        verify_observable_source(changed, config)
