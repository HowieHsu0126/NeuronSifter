from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from Libs.models.modules.neuronsifter_world_action import MechanismOperatorExpertBank
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.exp.experiment_manifests import deep_merge, load_experiment_family
from Libs.utils.model.screening_model_config import rollout_update_policy_from_config
from Libs.tests.test_native_field_normalization import joint_normalized_case, screening_kwargs, screening_fixture_bundle


CONTROL_PROFILES = ["rollout_operator_reversed_experts", "rollout_operator_interleaved_experts"]


@pytest.fixture(scope="module")
def ten_expert_case(joint_normalized_case):
    # The reused R60 fixture narrows the roster to four. Only numerical dimensions
    # and runtime may shrink for the named Round64 controls; restore the full bank.
    case = list(joint_normalized_case)
    config = copy.deepcopy(case[1])
    canonical = load_config("Libs/configs/config_stack.yaml")
    def backbone(payload):
        return payload["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    source, target = backbone(canonical), backbone(config)
    target["mechanism_moe"]["expert_names"] = copy.deepcopy(source["mechanism_moe"]["expert_names"])
    for key in ("composition_order", "coupling_masks"):
        target["operator_splitting"][key] = copy.deepcopy(source["operator_splitting"][key])
    case[1] = config
    return tuple(case)


def expert_bank(mode="serial",order=None,names=None,masks=None):
    return MechanismOperatorExpertBank(2,0.,names or ["a","b"],order or ["a","b"],
        {"a":["s0"],"b":["s0","s1"]} if masks is None else masks,.8,.3,1,
        composition_mode=mode).double()


class LinearExpert(nn.Module):
    def __init__(self,matrix,condition):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor(matrix,dtype=torch.float64))
        self.condition=nn.Parameter(torch.tensor(condition,dtype=torch.float64))
        self.inputs=[]

    def forward(self,x,c):
        self.inputs.append((x.detach().clone(),c.detach().clone()))
        return x @ self.weight.T+c @ self.condition.T


@pytest.mark.parametrize("mode",["serial","parallel"])
def test_noncommuting_updates_weights_diagnostics_and_gradients(mode):
    bank=expert_bank(mode,order=["b","a"] if mode=="parallel" else ["a","b"])
    bank.experts=nn.ModuleDict({
        "a":LinearExpert([[0.,1.],[0.,0.]],[[.3,0.],[0.,.2]]),
        "b":LinearExpert([[0.,0.],[1.,0.]],[[.1,.2],[.2,0.]])})
    shared=nn.Linear(2,2,bias=False,dtype=torch.float64)
    with torch.no_grad():
        bank.router.weight.zero_()
        bank.router.bias.copy_(torch.log(torch.tensor([2.,3.],dtype=torch.float64)))
        shared.weight.copy_(torch.tensor([[.4,.2],[-.1,.3]],dtype=torch.float64))
    bank.shared_coupling_operator=shared
    shared_inputs=[]
    handle=shared.register_forward_pre_hook(lambda module,args:shared_inputs.append(args[0].detach().clone()))
    x=torch.tensor([[[1.,2.],[3.,4.]]],dtype=torch.float64,requires_grad=True)
    c=torch.tensor([[[.2,.3],[.5,-.25]]],dtype=torch.float64,requires_grad=True)
    # Router prior/cardinality give probabilities[1/4,3/4] and final
    # residual weights[0.8*0.5/4,0.8*3/4]=[0.1,0.6].
    matrices=[bank.experts[name].weight.detach() for name in ("a","b")]
    conditions=[bank.experts[name].condition.detach() for name in ("a","b")]
    first=.1*(x @ matrices[0].T+c @ conditions[0].T)
    parallel_second=.6*(x @ matrices[1].T+c @ conditions[1].T)
    serial_second=.6*((x+first) @ matrices[1].T+c @ conditions[1].T)
    second=parallel_second if mode=="parallel" else serial_second
    updated=x+first+second
    expected=updated-x+.3*((updated+c) @ shared.weight.detach().T)
    actual,diagnostics=bank(x,c)
    handle.remove()
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    assert not torch.allclose(parallel_second,serial_second)
    assert len(shared_inputs)==1
    torch.testing.assert_close(shared_inputs[0],updated+c)
    for name in ("a","b"):
        assert len(bank.experts[name].inputs)==1
        torch.testing.assert_close(bank.experts[name].inputs[0][1],c)
    torch.testing.assert_close(bank.experts["a"].inputs[0][0],x)
    torch.testing.assert_close(bank.experts["b"].inputs[0][0],x if mode=="parallel" else x+first)
    torch.testing.assert_close(diagnostics["operator_splitting_expert_utilization"],x.new_tensor([.25,.75]))
    torch.testing.assert_close(diagnostics["operator_splitting_case_activation"],x.new_tensor([[.25,.75]]))
    torch.testing.assert_close(diagnostics["operator_splitting_score_norm_by_expert"],
                               torch.stack([r.norm(dim=-1).mean() for r in (first,second)]))
    torch.testing.assert_close(diagnostics["operator_splitting_uncertainty_by_expert"],
                               torch.stack([r.std(unbiased=False) for r in (first,second)]))
    attribution=x.new_tensor([[1.,0.],[.5,.5]])
    torch.testing.assert_close(diagnostics["operator_splitting_species_attribution"],attribution)
    torch.testing.assert_close(diagnostics["operator_splitting_coupling_matrix"],attribution @ attribution.T)
    assert diagnostics["operator_splitting_composition_count"]==2
    assert diagnostics["operator_splitting_parallel_update"]==int(mode=="parallel")
    assert diagnostics["operator_splitting_update_depth"]==(1 if mode=="parallel" else 2)
    actual.square().sum().backward()
    for module in (*bank.experts.values(),bank.router,shared):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                   for p in module.parameters())
    assert x.grad.abs().sum()>0 and c.grad.abs().sum()>0


@pytest.mark.parametrize("kwargs,reason",[
    ({"mode":"unknown"},"composition_mode"),
    ({"mode":"parallel","order":["a"]},"complete unique"),
    ({"mode":"parallel","order":["a","a"]},"complete unique"),
    ({"mode":"parallel","order":["a","unknown"]},"complete unique"),
    ({"mode":"parallel","names":["a","a"]},"complete unique"),
    ({"mode":"parallel","masks":{"a":[],"b":["s0"]}},"must not be empty"),
])
def test_composition_mode_and_matched_roster_rejection(kwargs,reason):
    with pytest.raises(ValueError,match=reason):
        expert_bank(**kwargs)


def test_checkpoint_mode_binding_and_legacy_serial_loading():
    serial,parallel=expert_bank(),expert_bank("parallel")
    assert {n:p.shape for n,p in serial.named_parameters()}=={n:p.shape for n,p in parallel.named_parameters()}
    assert {n:t.shape for n,t in serial.state_dict().items() if not n.endswith("composition_identity")}=={n:t.shape for n,t in parallel.state_dict().items() if not n.endswith("composition_identity")}
    assert sum(p.numel() for p in serial.parameters())==sum(p.numel() for p in parallel.parameters())
    # Parameter-only copying is an explicit unit comparison, not run reuse.
    with torch.no_grad():
        for name,p in parallel.named_parameters():
            p.copy_(dict(serial.named_parameters())[name])
    with pytest.raises(ValueError,match="checkpoint composition_mode"):
        parallel.load_state_dict(serial.state_dict(),strict=False)
    legacy=copy.deepcopy(serial.state_dict())
    legacy.pop("composition_mode_code")
    expert_bank().load_state_dict(legacy)
    with pytest.raises(ValueError,match="requires explicit composition_mode"):
        parallel.load_state_dict(legacy,strict=False)
    parallel.load_state_dict(copy.deepcopy(parallel.state_dict()))


def resolved_profile(name,base):
    family=load_experiment_family(name,base)
    return family,deep_merge(base,family["config_overrides"])


def test_real_matched_profile_and_blocked_baseline_contract():
    base=load_config("Libs/configs/config_stack.yaml")
    reference,serial=resolved_profile("dynamics_mainline_ad",base)
    family,parallel=resolved_profile("rollout_operator_parallel_experts",base)
    serial_policy=rollout_update_policy_from_config(serial)
    parallel_policy=rollout_update_policy_from_config(parallel)
    assert serial_policy["transition_backbone"]["operator_splitting"]["composition_mode"]=="serial"
    assert parallel_policy["transition_backbone"]["operator_splitting"]["composition_mode"]=="parallel"
    parallel["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["operator_splitting"]["composition_mode"]="serial"
    assert parallel==serial
    assert family["execution"]==reference["execution"]
    for key in ("cluster_prebuild","cluster_training"):
        assert family[key]==reference[key]
    assert family["runtime"]["seed_values"]==reference["runtime"]["seed_values"]==base["screening_pipeline"]["screening_benchmark"]["formal_seed_values"]
    reference_runtime=copy.deepcopy(reference["runtime"])
    parallel_runtime=copy.deepcopy(family["runtime"])
    for runtime in (reference_runtime,parallel_runtime):
        runtime.pop("benchmark_name_template")
        runtime.pop("artifact_root")
    assert reference_runtime==parallel_runtime
    entry=next(row for row in base["paper_grade_baseline_registry"]["baselines"] if row["baseline_id"]=="neuronsifter_parallel_experts")
    assert entry["profile_name"]=="rollout_operator_parallel_experts"
    assert not entry["main_table_runnable"] and not entry["appendix_runnable"] and entry["reference_only"]
    assert "pending_admitted_native_data" in entry["reason_if_not_runnable"]
    print("Effective shared seeds:",family["runtime"]["seed_values"])


def observe_composition_inputs(native):
    handles,states=[],[]
    for block in native.blocks:
        bank=block.operator_splitting_expert_bank
        assert bank.composition_mode in {"serial", "parallel"}
        state={"completed":0}
        states.append(state)
        def enter(module,args,state=state):
            state.update(field=args[0].detach().clone(),condition=args[1].detach().clone(),seen=[],shared=0)
        handles.append(bank.register_forward_pre_hook(enter))
        for name,expert in bank.experts.items():
            def expert_enter(module,args,name=name,state=state):
                assert name not in state["seen"]
                state["seen"].append(name)
                if bank.composition_mode == "parallel":
                    assert torch.equal(args[0],state["field"])
                assert torch.equal(args[1],state["condition"])
            handles.append(expert.register_forward_pre_hook(expert_enter))
        def shared_enter(module,args,state=state):
            state["shared"]+=1
        handles.append(bank.shared_coupling_operator.register_forward_pre_hook(shared_enter))
        def leave(module,args,output,state=state):
            expected = module.expert_names if module.composition_mode == "parallel" else module.composition_order
            assert state["seen"] == expected and state["shared"] == 1
            assert output[1]["operator_splitting_identity"] == module.execution_identity()
            state["completed"]+=1
        handles.append(bank.register_forward_hook(leave))
    return handles,states


def assert_composition_record(record,expert_count,mode="parallel",names=None,order=None):
    assert record["diagnostic_status"]=="available" and record["composition_mode"]==mode
    if names is not None:
        assert record["expert_names"] == names
        assert record["composition_order"] == (names if mode == "parallel" else order)
    values=record["diagnostics"]
    assert values["operator_splitting_composition_count"]==expert_count
    assert values["operator_splitting_update_depth"]==(1 if mode=="parallel" else expert_count)
    for key in ("score_norm_by_expert","uncertainty_by_expert","expert_utilization"):
        tensor=torch.tensor(values["operator_splitting_"+key])
        assert tensor.shape==(expert_count,) and torch.isfinite(tensor).all()
    coupling=torch.tensor(values["operator_splitting_coupling_matrix"])
    assert coupling.shape==(expert_count,expert_count) and coupling.abs().sum()>0


@pytest.mark.parametrize("profile", ["rollout_operator_parallel_experts", *CONTROL_PROFILES])
def test_native_pretrainer_composition_consumer(ten_expert_case, profile):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    _,base,_,source,original,_=ten_expert_case
    config=control_config(base,profile)
    names, order = control_order(config)
    mode = "parallel" if "parallel" in profile else "serial"
    assert len(names) == len(order) == 10
    model=pretrain.MicroenvironmentPretrainingModel(config,False)
    serial=pretrain.MicroenvironmentPretrainingModel(base,False)
    assert {n:p.shape for n,p in model.named_parameters()}=={n:p.shape for n,p in serial.named_parameters()}
    assert {n:t.shape for n,t in model.state_dict().items() if not n.endswith("composition_identity")}=={n:t.shape for n,t in serial.state_dict().items() if not n.endswith("composition_identity")}
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in serial.parameters())
    assert model.field_normalization.identity==serial.field_normalization.identity
    handles,states=observe_composition_inputs(model.operator)
    dataset=pretrain.MicroenvironmentTeacherFamilyDataset(config,"train")
    try:
        metrics,report=pretrain.run_pretraining_step(model,pretrain.collate_native_sources(dataset.__getitems__([0,3])),
            config,torch.device("cpu"),{"gradient_accumulation_steps":1,
            "mixed_precision":{"enabled":False,"device_type":"cpu","dtype":"float32"}})
    finally:
        for handle in handles:
            handle.remove()
    assert torch.isfinite(torch.tensor(metrics["total_loss"])) and all(state["completed"]>0 for state in states)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.operator.parameters())
    count=len(model.operator.blocks[0].operator_splitting_expert_bank.expert_names)
    assert len(report["source_rows"])==2
    for row in report["source_rows"]:
        assert len(row["operator_splitting_diagnostics"])==2
        for record in row["operator_splitting_diagnostics"]:
            assert_composition_record(record,count,mode,names,order)
        assert row["field_normalization_diagnostics"]
    assert json.dumps(source,sort_keys=True)==original


@pytest.mark.parametrize("profile", ["rollout_operator_parallel_experts", *CONTROL_PROFILES])
def test_screening_serving_sequence_composition_consumer(ten_expert_case,tmp_path,profile):
    from Libs.models.networks.screening_system import ADScreeningSystem
    from Libs.scripts.services import serve_screening_api as serving
    from Libs.exp.training.train_screening_system import native_diffusion_training_kwargs
    from Libs.utils.data import native_diffusion_data as native
    _,base,_,source,original,_=ten_expert_case
    config=control_config(base,profile)
    names, order = control_order(config)
    mode = "parallel" if "parallel" in profile else "serial"
    assert len(names) == len(order) == 10
    kwargs=screening_kwargs(config)
    model=ADScreeningSystem(**kwargs).eval()
    selected=native.select_joint_source(source,["sample-a"],["regimen-a","regimen-b"])
    regimens=torch.rand(2,model.regimen_feature_dim)
    cached=native.screening_native_payload(config,"train",["sample-a"],["regimen-a","regimen-b"])
    training=native_diffusion_training_kwargs(model,{"labels":[0],"native_diffusion":cached},
        slice(None),[0,1],torch.device("cpu"),split_name="train")
    handles,states=observe_composition_inputs(model.field_rollout_operator.neuronsifter_world_action_operator)
    try:
        scored=model.score_regimens(torch.zeros(1,4),regimens,**training)
    finally:
        for handle in handles:
            handle.remove()
    assert all(state["completed"]>0 for state in states)
    assert training["particle_programs"]["records"]==selected["particle_programs"]["records"]
    count=len(model.field_rollout_operator.neuronsifter_world_action_operator.blocks[0].operator_splitting_expert_bank.expert_names)
    assert len(scored["particle_operator_splitting_diagnostics"])==2
    for record in scored["particle_operator_splitting_diagnostics"]:
        assert_composition_record(record,count,mode,names,order)
    torch.testing.assert_close(scored["field_trajectory_std"].square(),
        scored["within_diffusion_variance"]+scored["between_particle_variance"],rtol=1e-5,atol=1e-7)
    checkpoint=tmp_path/"composition.pt"
    torch.save({"model_kwargs":kwargs,"model_state":model.state_dict()},checkpoint)
    restored=serving.load_screening_model(checkpoint,config)
    assert restored.field_rollout_operator.field_normalization.identity==model.field_rollout_operator.field_normalization.identity
    with pytest.raises(ValueError,match="Checkpoint expert"):
        serving.load_screening_model(checkpoint,base)
    bundle=screening_fixture_bundle(restored,config,regimens)
    payload={"joint_source":selected,"regimen_uids":["regimen-a","regimen-b"]}
    joint=serving.native_diffusion_serving_kwargs(bundle,payload,payload["regimen_uids"])
    handles,states=observe_composition_inputs(restored.field_rollout_operator.neuronsifter_world_action_operator)
    try:
        sequence=restored.rollout(torch.zeros(1,4),regimens.unsqueeze(0),
            joint_posterior=joint["joint_posterior"],particle_programs=joint["particle_programs"],
            regimen_uids=payload["regimen_uids"])
        served=serving.rollout_fields(bundle,payload)
    finally:
        for handle in handles:
            handle.remove()
    assert all(state["completed"]>0 for state in states)
    assert torch.equal(sequence["particle_field_trajectory_sequence"][:,:,1,0],
                       sequence["particle_field_trajectory_sequence"][:,:,0,-1])
    assert torch.equal(scored["particle_field_trajectory"][:,:,:,0],
                       torch.tensor(selected["support"]["initial_field"])[:,:,None].expand(-1,-1,2,-1,-1,-1))
    assert torch.isfinite(sequence["field_trajectory_std_sequence"]).all()
    assert len(served["operator_splitting_diagnostics"])==4
    assert {(r["particle_index"],r["sequence_index"]) for r in served["operator_splitting_diagnostics"]}=={(0,0),(0,1),(1,0),(1,1)}
    for record in served["operator_splitting_diagnostics"]:
        assert_composition_record(record,count,mode,names,order)
    json.dumps(served,allow_nan=False)
    assert served["field_normalization_diagnostics"]
    assert json.dumps(source,sort_keys=True)==original


def test_invalid_config_and_missing_mode_diagnostics():
    from Libs.utils.model.joint_rollout import operator_splitting_record
    config=load_config("Libs/configs/config_stack.yaml")
    splitting=config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["operator_splitting"]
    splitting["composition_mode"]="unknown"
    with pytest.raises(ValueError,match="composition_mode"):
        rollout_update_policy_from_config(config)
    splitting["composition_mode"]="parallel"
    splitting["composition_order"].pop()
    with pytest.raises(ValueError,match="complete unique"):
        rollout_update_policy_from_config(config)
    for diagnostics in ({},{"operator_splitting_parallel_update":torch.tensor(.5)}):
        record=operator_splitting_record(diagnostics)
        assert record["diagnostic_status"]=="composition_mode_unavailable" and "composition_mode" not in record
    missing = operator_splitting_record({"operator_splitting_parallel_update": torch.tensor(0.)})
    assert missing["diagnostic_status"] == "composition_identity_unavailable"
    assert "expert_names" not in missing and "composition_order" not in missing




def control_config(base, profile):
    _, config = resolved_profile(profile, base)
    config["selected_experiment_family"] = profile
    return config


def control_order(config):
    backbone = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    return backbone["mechanism_moe"]["expert_names"], backbone["operator_splitting"]["composition_order"]


@pytest.mark.parametrize("profile", CONTROL_PROFILES)
def test_depth_matched_noncommuting_order_and_canonical_diagnostics(profile):
    from Libs.utils.model.joint_rollout import operator_splitting_record
    config = control_config(load_config("Libs/configs/config_stack.yaml"), profile)
    names, order = control_order(config)
    assert len(names) == len(order) == 10
    masks = {name: ["s0"] if i % 2 == 0 else ["s0", "s1"] for i, name in enumerate(names)}
    bank = MechanismOperatorExpertBank(2, 0., names, order, masks, .8, .3, 1).double()
    events = []
    for i, name in enumerate(names):
        matrix = [[0., (i + 1) / 10], [0., 0.]] if i % 2 == 0 else [[0., 0.], [(i + 1) / 10, 0.]]
        expert = LinearExpert(matrix, [[.02 * (i + 1), 0.], [0., .03]])
        expert.register_forward_pre_hook(lambda module, args, name=name: events.append(name))
        bank.experts[name] = expert
    shared = nn.Linear(2, 2, bias=False, dtype=torch.float64)
    shared.register_forward_pre_hook(lambda module, args: events.append("shared"))
    bank.shared_coupling_operator = shared
    with torch.no_grad():
        bank.router.weight.zero_()
        bank.router.bias.copy_(torch.log(torch.arange(1, 11, dtype=torch.float64)))
        shared.weight.copy_(torch.tensor([[.4, .2], [-.1, .3]], dtype=torch.float64))
    x = torch.tensor([[[1., 2.], [3., 4.]]], dtype=torch.float64)
    c = torch.tensor([[[.2, .3], [.5, -.25]]], dtype=torch.float64)
    cardinality = torch.tensor([len(masks[name]) for name in names], dtype=torch.float64)
    probabilities = cardinality * torch.arange(1, 11, dtype=torch.float64)
    probabilities /= probabilities.sum()
    def reference(execution):
        state = x
        residuals = {}
        for name in execution:
            i = names.index(name)
            expert = bank.experts[name]
            residual = .8 * cardinality[i] / cardinality.max() * probabilities[i] * (
                state @ expert.weight.T + c @ expert.condition.T)
            residuals[name] = residual
            state = state + residual
        return state - x + .3 * ((state + c) @ shared.weight.T), residuals
    expected, residuals = reference(order)
    canonical, _ = reference(names)
    actual, diagnostics = bank(x, c)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert not torch.allclose(actual, canonical)
    assert events == order + ["shared"]
    assert list(bank.experts) == names and bank.expert_index_lookup == dict(zip(names, range(10)))
    torch.testing.assert_close(diagnostics["operator_splitting_expert_utilization"], probabilities)
    torch.testing.assert_close(diagnostics["operator_splitting_case_activation"], probabilities.unsqueeze(0))
    torch.testing.assert_close(diagnostics["operator_splitting_score_norm_by_expert"],
                              torch.stack([residuals[name].norm(dim=-1).mean() for name in names]))
    torch.testing.assert_close(diagnostics["operator_splitting_uncertainty_by_expert"],
                              torch.stack([residuals[name].std(unbiased=False) for name in names]))
    attribution = torch.tensor([[1., 0.] if i % 2 == 0 else [.5, .5] for i in range(10)], dtype=torch.float64)
    torch.testing.assert_close(diagnostics["operator_splitting_species_attribution"], attribution)
    torch.testing.assert_close(diagnostics["operator_splitting_coupling_matrix"], attribution @ attribution.T)
    record = operator_splitting_record(diagnostics)
    assert record["expert_names"] == names and record["composition_order"] == order
    assert record["composition_mode"] == "serial" and record["diagnostic_status"] == "available"
    assert record["diagnostics"]["operator_splitting_update_depth"] == 10
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("profile", CONTROL_PROFILES)
def test_control_profile_exact_matching_and_closed_baseline(profile):
    base = load_config("Libs/configs/config_stack.yaml")
    reference, mainline = resolved_profile("dynamics_mainline_ad", base)
    family, config = resolved_profile(profile, base)
    names, order = control_order(config)
    expected = list(reversed(names)) if "reversed" in profile else [names[i] for i in [2,3,0,4,1,5,6,7,8,9]]
    assert order == expected and len(names) == 10
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["operator_splitting"]["composition_order"] = names
    assert config == mainline
    assert family["execution"] == reference["execution"]
    assert family["cluster_prebuild"] == reference["cluster_prebuild"]
    assert family["cluster_training"] == reference["cluster_training"]
    for key in ("device", "training_scope", "seed_values"):
        assert family["runtime"][key] == reference["runtime"][key]
    row = next(row for row in base["paper_grade_baseline_registry"]["baselines"] if row.get("profile_name") == profile)
    assert not any(row[key] for key in ("main_table_runnable", "appendix_runnable"))
    assert row["reference_only"] is True


@pytest.mark.parametrize("profile", CONTROL_PROFILES)
@pytest.mark.parametrize("mutation", ["duplicate", "missing", "unknown", "other_order", "names", "mode"])
def test_named_control_config_rejects_roster_or_order_change(profile, mutation):
    config = control_config(load_config("Libs/configs/config_stack.yaml"), profile)
    names, order = control_order(config)
    splitting = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["operator_splitting"]
    if mutation == "duplicate":
        order[0] = order[1]
    elif mutation == "missing":
        order.pop()
    elif mutation == "unknown":
        order[0] = "unknown"
    elif mutation == "other_order":
        order.reverse()
    elif mutation == "names":
        names.reverse()
    else:
        splitting["composition_mode"] = "parallel"
    with pytest.raises(ValueError, match="roster|order"):
        rollout_update_policy_from_config(config)


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_mode_only_legacy_canonical_checkpoint_compatibility(mode, strict):
    model = expert_bank(mode)
    legacy = {key: value for key, value in model.state_dict().items() if key != "composition_identity"}
    result = model.load_state_dict(legacy, strict=strict)
    assert not result.missing_keys and not result.unexpected_keys
    if mode == "serial":
        legacy.pop("composition_mode_code")
        result = model.load_state_dict(legacy, strict=strict)
        assert not result.missing_keys and not result.unexpected_keys


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("mutation", ["names", "order", "mode_only", "untagged"])
def test_noncanonical_checkpoint_rejection_independent_of_strict(strict, mutation):
    model = expert_bank(order=["b", "a"])
    payload = copy.deepcopy(model.state_dict())
    if mutation in {"mode_only", "untagged"}:
        payload.pop("composition_identity")
        if mutation == "untagged":
            payload.pop("composition_mode_code")
    else:
        identity = json.loads(bytes(payload["composition_identity"].tolist()).decode())
        identity["expert_names" if mutation == "names" else "composition_order"].reverse()
        payload["composition_identity"] = torch.tensor(list(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()), dtype=torch.uint8)
    with pytest.raises(ValueError, match="composition identity"):
        model.load_state_dict(payload, strict=strict)


@pytest.mark.parametrize("profile", CONTROL_PROFILES)
@pytest.mark.parametrize("consumer", ["pretraining", "screening"])
@pytest.mark.parametrize("strict", [True, False])
def test_actual_ten_expert_checkpoint_roundtrip_and_cross_order(
        ten_expert_case, tmp_path, profile, consumer, strict):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain, train_screening_system as screening
    from Libs.models.networks.screening_system import ADScreeningSystem
    from Libs.utils.exp.experiment_manifests import resolve_training_seed
    from Libs.tests.test_native_training_seeds import selected
    config = selected(control_config(ten_expert_case[1], profile), 41)
    config["screening_pipeline"]["screening_benchmark"]["training"]["seed_policy"] = "manuscript_native_five"
    model = pretrain.MicroenvironmentPretrainingModel(config, False) if consumer == "pretraining" else ADScreeningSystem(**screening_kwargs(config))
    binding = resolve_training_seed(config, require_selected=True) if consumer == "pretraining" else screening.screening_training_seed_binding(config, "folds", "fixture")
    optimizer = SimpleNamespace(state_dict=lambda: {})
    if consumer == "pretraining":
        path = pretrain.save_training_checkpoint(model, optimizer, tmp_path, "fixture", 0, 0, "single_process", training_seed_binding=binding)
        payload = torch.load(path, map_location="cpu")
        state = payload["model_state_dict"]
    else:
        import numpy as np
        path = tmp_path / "fixture.pt"
        screening.save_checkpoint_state(path, model, optimizer, copy.deepcopy(model.state_dict()), 1., 0, [], np.random.default_rng(0), 0, 0, training_seed_binding=binding)
        payload = torch.load(path, map_location="cpu")
        screening.validate_screening_training_checkpoint(model, payload, binding)
        state = payload["model_state"]
    result = model.load_state_dict(state, strict=strict)
    assert not result.missing_keys and not result.unexpected_keys
    other_profile = CONTROL_PROFILES[1 - CONTROL_PROFILES.index(profile)]
    other_config = control_config(ten_expert_case[1], other_profile)
    other = pretrain.MicroenvironmentPretrainingModel(other_config, False) if consumer == "pretraining" else ADScreeningSystem(**screening_kwargs(other_config))
    assert {n:p.shape for n,p in model.named_parameters()} == {n:p.shape for n,p in other.named_parameters()}
    with pytest.raises(ValueError, match="composition identity"):
        other.load_state_dict(state, strict=strict)


@pytest.mark.parametrize("profile", CONTROL_PROFILES)
@pytest.mark.parametrize("slot", ["current", "best", "completed"])
@pytest.mark.parametrize("mutation", ["other_order", "mode_only"])
def test_actual_screening_reuse_guard_covers_current_best_completed(
        ten_expert_case, tmp_path, profile, slot, mutation):
    from Libs.exp.training import train_screening_system as screening
    from Libs.models.networks.screening_system import ADScreeningSystem
    config = control_config(ten_expert_case[1], profile)
    model = ADScreeningSystem(**screening_kwargs(config))
    binding = screening.screening_training_seed_binding(config, "folds", "fixture")
    payload = {"model_state": copy.deepcopy(model.state_dict()), "training_seed_binding": binding}
    if slot != "completed":
        payload.update(best_state=copy.deepcopy(model.state_dict()), best_training_seed_binding=binding)
    screening.validate_screening_training_checkpoint(model, payload, binding)
    target = payload["best_state" if slot == "best" else "model_state"]
    keys = [key for key in target if key.endswith("composition_identity")]
    assert keys
    for key in keys:
        if mutation == "mode_only":
            target.pop(key)
        else:
            identity = json.loads(bytes(target[key].tolist()).decode())
            identity["composition_order"].reverse()
            target[key] = torch.tensor(list(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()), dtype=torch.uint8)
    with pytest.raises(ValueError, match="composition identity"):
        screening.validate_screening_training_checkpoint(model, payload, binding)


@pytest.mark.parametrize("profile", CONTROL_PROFILES)
def test_rank_local_writer_carries_actual_ten_expert_identity(ten_expert_case,tmp_path,profile):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    config = control_config(ten_expert_case[1],profile)
    model = pretrain.MicroenvironmentPretrainingModel(config,False)
    path = pretrain.save_training_checkpoint(model, SimpleNamespace(state_dict=lambda:{}),
        tmp_path,"fixture",0,0,"native_pytorch_fsdp")
    state = torch.load(path,map_location="cpu")["model_state_dict"]
    for name,module in model.named_modules():
        if isinstance(module,MechanismOperatorExpertBank):
            assert len(module.expert_names)==10
            for field in ("composition_mode_code","composition_identity"):
                assert torch.equal(state[name+"."+field],getattr(module,field))


@pytest.mark.parametrize("profile", CONTROL_PROFILES)
def test_actual_control_chunked_scoring_preserves_order_metadata(ten_expert_case,profile):
    from Libs.models.networks.screening_system import ADScreeningSystem
    from Libs.exp.training.train_screening_system import native_diffusion_training_kwargs
    from Libs.utils.data import native_diffusion_data as native
    config = control_config(ten_expert_case[1],profile)
    model = ADScreeningSystem(**screening_kwargs(config)).eval()
    names, order = control_order(config)
    regimens = torch.rand(2,model.regimen_feature_dim)
    cached = native.screening_native_payload(config,"train",["sample-a"],["regimen-a","regimen-b"])
    kwargs = native_diffusion_training_kwargs(model,{"labels":[0],"native_diffusion":cached},
        slice(None),[0,1],torch.device("cpu"),split_name="train")
    with torch.no_grad():
        full = model.score_regimens(torch.zeros(1,4),regimens,**kwargs)
        chunks = model.score_regimens(torch.zeros(1,4),regimens,regimen_chunk_size=1,**kwargs)
    torch.testing.assert_close(chunks["field_trajectory_samples"],full["field_trajectory_samples"])
    assert len(chunks["particle_operator_splitting_diagnostics"]) == 2
    for record in chunks["particle_operator_splitting_diagnostics"]:
        assert_composition_record(record,10,"serial",names,order)
        assert torch.tensor(record["diagnostics"]["operator_splitting_case_activation"]).shape == (1,2,10)
