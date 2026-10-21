"""Core distribution, block-update and checkpoint-reuse contracts."""
import copy
from types import SimpleNamespace

import pytest
import torch

from Libs.models.modules.native_rssm import NativeDiscreteRSSM,gaussian_kl,weighted_free_nats,categorical_kl
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.evaluation.manuscript_native_protocol import training_profile,command_catalog,validate_definition
from Libs.utils.data.native_stimulation import FORCING_ARRAY_NAMES


def test_gaussian_free_nats_mean_order_and_discrete_gradient_split():
    terms=torch.tensor([0.,4.],requires_grad=True)
    floor=weighted_free_nats(terms,torch.tensor([.5,.5]),3.)
    assert floor==3
    floor.backward();assert terms.grad.abs().sum()==0
    zero=torch.zeros(2,3);one=torch.ones(2,3)
    assert torch.equal(gaussian_kl((zero,one),(zero,one)),torch.zeros(2))
    q=torch.tensor([[[.8,.2]]],requires_grad=True);p=torch.tensor([[[.3,.7]]],requires_grad=True)
    categorical_kl(q.detach(),p).sum().backward()
    assert q.grad is None and p.grad.abs().sum()>0
    p.grad=None
    categorical_kl(q,p.detach()).sum().backward()
    assert p.grad is None and q.grad.abs().sum()>0


def test_discrete_block_gru_uses_reset_candidate_and_shifted_update():
    config=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    settings=copy.deepcopy(config["native_controlled_comparators"]["references"]["dreamerv3"]["hyperparameters"])
    settings.update(deterministic_size=8,hidden_size=4,stochastic_size=2,classes=3,blocks=2,vector_units=4)
    inputs=SimpleNamespace(initial=torch.zeros(1,1,1,1,1,2),blocks={"source_nuisance":(0,2),"physical_domain":(2,3)})
    model=NativeDiscreteRSSM(settings,inputs,config)
    with torch.no_grad():
        for parameter in model.parameters():parameter.zero_()
        model.gates.bias[:,4:8]=1
    previous=torch.full((2,8),.3)
    latent=torch.zeros(2,2,3);latent[:,:,0]=1
    program=torch.zeros(2,3+len(FORCING_ARRAY_NAMES))
    hidden,prior=model.transition(previous,latent,program)
    update=torch.tensor(-1.).sigmoid()
    expected=update*torch.tensor(.5).tanh()+(1-update)*previous
    torch.testing.assert_close(hidden,expected)
    torch.testing.assert_close(prior,torch.full_like(prior,1/3))


def test_prediction_profiles_reuse_gaussian_and_preserve_behavior_obligations():
    config=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    validate_definition(config)
    profiles={name:training_profile(config,name,41) for name in ("planet","dreamer","dreamerv3")}
    assert profiles["dreamer"][1]["training_manifest"]==profiles["planet"][1]["training_manifest"]
    commands=command_catalog(config,[row[1] for row in profiles.values()])
    assert not any(row["command_uid"]==profiles["dreamer"][1]["profile_id"]+":fit" for row in commands)
    for row in config["manuscript_native_protocol"]["methods"]:
        if row["method_id"] in profiles:
            assert row["later_gate_obligations"]["required_for_full_method_comparison"] is True
            assert row["later_gate_obligations"]["status"]=="NOT_IMPLEMENTED"
    invalid=copy.deepcopy(config);invalid["native_controlled_comparators"]["references"]["dreamerv3"]["hyperparameters"]["dynamics_weight"]=.5
    with pytest.raises(ValueError,match="core/loss"):validate_definition(invalid)
