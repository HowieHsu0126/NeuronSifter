"""Native sequence mean invariants and the upstream FNO checkpoint boundary."""
import copy
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
import yaml

from Libs.models.modules import native_sequence_comparators as sequence
from Libs.models.modules.native_diffusion_comparators import NativeFNO
from Libs.utils.data.native_stimulation import FORCING_ARRAY_NAMES


@pytest.mark.parametrize("method",["gru","flattened_parameter_matched"])
def test_native_sequence_uses_past_forcing_preserves_initial_and_masks(method,monkeypatch):
    torch.manual_seed(905)
    times=torch.arange(4.).reshape(1,1,4,1,1,1).expand(1,2,4,1,1,1)
    context={key:torch.zeros_like(times) for key in ("occupancy","support",*FORCING_ARRAY_NAMES)}
    context["physical_time"]=times
    monkeypatch.setattr(sequence,"joint_particle_contexts",lambda *args:[context])
    initial=torch.tensor([1.,0.]).reshape(1,1,1,1,2)
    support=torch.ones(2,4,1,1,2,dtype=torch.bool);support[...,-1]=False
    inputs=SimpleNamespace(initial=initial.expand(2,1,1,1,2).reshape(2,1,1,2),validity=support,
        condition=torch.randn(2,4),blocks={"source_nuisance":(0,3),"physical_domain":(3,4)},
        normalization_id="fixture",group_shape=(1,1,2),programs={},regimen_uids=("a","b"),
        posterior=SimpleNamespace(support=SimpleNamespace(initial_field=initial)))
    normalizer=SimpleNamespace(identity="fixture",inverse=lambda value,mask:torch.where(mask,value,0))
    config={"native_controlled_comparators":{"conditioning":{"program_reference_scales":{
        name:1. for name in ("occupancy","support","physical_time",*FORCING_ARRAY_NAMES)}}}}
    settings={"hidden_dim":8,"depth":1,"head_count":2,"dropout_rate":0.,"parameter_match_relative_tolerance":.001}
    model=sequence.NativeSequenceMean(method,settings,inputs,config,
        {"reference_total_parameter_count":4000,"shared_inference_parameter_count":200}).eval()
    first=model.predict_native(inputs,normalizer)
    context["electrical_current_uA"][:,:,2:]=3.
    second=model.predict_native(inputs,normalizer)
    assert torch.equal(first[:,:2],second[:,:2])
    assert not torch.equal(first[:,3:],second[:,3:])
    assert torch.equal(first[:,0],inputs.initial)
    assert not first[~support].any()
    second.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    if method=="flattened_parameter_matched":
        assert model.parameter_match["actual_dynamics_parameters"]==sum(p.numel() for p in model.parameters())
        assert model.parameter_match["relative_error"]<=.001
        with pytest.raises(ValueError,match="measured parameter count"):
            sequence.NativeSequenceMean(method,settings,inputs,config)
    changed=copy.copy(inputs);changed.validity=support.clone();changed.validity[0,1,0,0,1]=True
    with pytest.raises(ValueError,match="time-invariant"):
        model.predict_native(changed,normalizer)


def test_native_fno_preserves_and_checks_upstream_constructor_metadata():
    root=Path(__file__).resolve().parents[2]
    entry=yaml.safe_load((root/"Libs/configs/exp/manuscript_native_protocol.yaml").read_text())[
        "native_controlled_comparators"]["references"]["fno"]
    entry["hyperparameters"].update(hidden_channels=4,n_layers=1,n_modes=[2,2],conditioning_channels=2)
    model=NativeFNO(entry,(3,1,1,26),30)
    state=model.state_dict()
    assert "_metadata" in state
    loaded=NativeFNO(entry,(3,1,1,26),30)
    loaded.load_state_dict(state,strict=True)
    assert "_metadata" in state
    changed=copy.deepcopy(state);changed["_metadata"]["hidden_channels"]+=1
    with pytest.raises(ValueError,match="constructor metadata"):
        loaded.load_state_dict(changed,strict=False)
    missing=state.copy();missing.pop("_metadata")
    with pytest.raises(ValueError,match="constructor metadata"):
        loaded.load_state_dict(missing)


@pytest.mark.parametrize("method",["native_deterministic","flattened_parameter_matched","gru","fno"])
def test_added_mean_methods_dispatch_calibration_and_predictive_law(method,tmp_path,monkeypatch):
    from Libs.utils.evaluation import native_comparator_evaluation as evaluation
    from Libs.utils.data.raw_data_registry import load_config
    cfg=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    cfg["native_scientific_training"]["predictive_budget"]["conditional_draws_per_particle"]=2
    normalizer=SimpleNamespace(normalize=lambda value,mask:value,inverse=lambda value,mask:value,
                               scale=torch.ones(26),identity="fixture")
    belief=torch.nn.Linear(1,1);belief.field_normalization=normalizer
    def prepare(group,*args):
        inputs=SimpleNamespace(sample_uids=[group],regimen_uids=["a"],weights=torch.full((1,32),1/32),
            group_shape=(1,32,1),normalization_id="fixture",
            validity=torch.ones(32,2,1,1,26,dtype=torch.bool),
            posterior=SimpleNamespace(support=SimpleNamespace(initial_field=torch.zeros(1,32,1,1,26))),
            grid={"family_uid":"fixture","time_hours":[0.,1.],"species_id":[row["quantity_id"]
                for row in cfg["native_observable_bridge"]["quantity_profile"]["quantities"]]})
        target=torch.zeros(1,1,2,1,1,26);mask=torch.ones_like(target,dtype=torch.bool)
        return inputs,target,mask,None
    monkeypatch.setattr(evaluation,"prepare_comparator_group",prepare)
    monkeypatch.setattr(evaluation,"comparator_mean",lambda *args,**kwargs:(torch.zeros(1,1,2,1,1,26),torch.zeros(1,32,1,2,1,1,26)))
    monkeypatch.setattr(evaluation,"episode_statistics",lambda *args,**kwargs:{"group":args[-1]})
    calls=[]
    def fit(rows,groups,numerical,*,species_count):
        calls.append((groups,species_count))
        return {"status":"FIT_COMPLETE","scales":[1.]*26,
            "counts":{"groups":[{"group_uid":"fit","declared_episode_count":1,"eligible_episode_count":1}],
                      "episodes":[{"group_uid":"fit","future_coordinate_count":26,"eligible":True}],
                      "declared_group_count":1,"declared_episode_count":1,"eligible_episode_count":1,
                      "eligible_group_count":1,"future_coordinate_count":26,"per_species_future_coordinate_count":[1]*26},
            "diagnostics":{"hessian_eigenvalues":[1.]*26,"gradient_max_abs":0.,
                           "hessian_required_minimum":1e-8,"objective":0.},"cost":{}}
    monkeypatch.setattr(evaluation,"fit_scales",fit)
    monkeypatch.setattr(evaluation,"node_adjacency_matrix",lambda config:torch.ones(1,1))
    quantities=[row["quantity_id"] for row in cfg["native_observable_bridge"]["quantity_profile"]["quantities"]]
    monkeypatch.setattr(evaluation,"check_diagnostics",lambda *args,**kwargs:{
        "pit":{"kind":"exact_marginal_mixture_cdf","t0_continuous_pit_count":0,
               "future":{"count":26,"mean":.5},
               "by_species":[{"species_id":q,"count":1,"mean":.5} for q in quantities]},
        "covariance":{kind:[{"count":1,"centered_product_mean":0.,"total_covariance_mean":1.}]
                      for kind in ("adjacent_retained_times","graph_edges","cross_species_pairs")},
        "interpretation":"mocked_numerical_fit_for_dispatch_only"})
    declared = {reservation:{reservation:{"family_uid":"fixture","action_indices":[0]}}
                for reservation in ("fit", "check")}
    monkeypatch.setattr(evaluation, "native_residual_partitions", lambda config: declared)
    class Groups:
        def __init__(self,name):
            self.group_uids=[name]
            self.split_name="calibration" if name in declared else "validation"
            teacher={"sample_uid":name,"split_name":self.split_name,
                     "source_holdout_role":self.split_name,"calibration_reservation":name,
                     "family_uid":"fixture","regimen_uid":"a"}
            self.groups=[(SimpleNamespace(native_records=[{"teacher":teacher}]),[0],None)]
        def __len__(self):return 1
        def __getitem__(self,index):return {"native_joint_group":self.group_uids[index]}
    wrong_role=Groups("fit")
    wrong_role.split_name="train"
    with pytest.raises(ValueError, match="complete declared fit role"):
        evaluation.admit_native_residual_calibration(cfg,wrong_role,Groups("check"))
    binding={"model_identity":{"method":method},"normalization_id":"fixture",
             "fixture":"dispatch_only_not_calibration_evidence"}
    fit_root=tmp_path/"calibration"
    result=evaluation.calibrate_native_comparator(cfg,method,None,belief,binding,Groups("fit"),Groups("check"),fit_root)
    assert calls==[(["fit"],26)] and result["check_group_uids"]==["check"]
    assert result["calibration_check_complete"]
    scored=evaluation.evaluate_native_comparator(cfg,method,None,belief,binding,Groups("evaluation"),
        {"evaluation":{"partition_uid":"id","family_uid":"fixture"}},tmp_path/"evaluation",
        residual_path=fit_root/"native_comparator_residual.json")
    assert scored["status"]=="complete" and scored["defined_group_count"]==1
    assert scored["groups"][0]["metrics"]["crps"] is not None
    assert scored["scientific_twin"]=="NOT_RUN" and not scored["formal_qualification"]
