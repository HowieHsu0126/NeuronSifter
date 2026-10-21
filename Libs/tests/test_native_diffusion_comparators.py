"""Software-only checks for native reference diffusion adapters."""
import copy
from pathlib import Path

import pytest
import torch
import numpy as np
import pywt
import yaml

from Libs.models.modules.native_diffusion_comparators import (
    NativeDiffusionPDE, NativeFieldPacking, NativeWDNO,
    PeriodizedBiorthogonalWavelet, local_reference,
)

ROOT = Path(__file__).resolve().parents[2]


def reference(name):
    config = yaml.safe_load((ROOT/"Libs/configs/exp/manuscript_native_protocol.yaml").read_text())
    result = copy.deepcopy(config["native_controlled_comparators"]["references"][name])
    if name == "wdno":
        result["hyperparameters"].update(dim=8, dim_mults=[1], resnet_block_groups=1,
            conditioning_channels=4, diffusion_steps=8)
    else:
        result["hyperparameters"].update(model_channels=8, channel_mult=[1],
            num_blocks=1, attention_resolutions=[], sigma_max=1.)
    return result


def test_complete_native_field_and_wavelet_roundtrip():
    packing = NativeFieldPacking((28, 6, 6, 26), 16)
    value = torch.randn(1, 28, 6, 6, 26, requires_grad=True)
    assert torch.equal(packing.unpack(packing.pack(value)), value)
    wavelet = PeriodizedBiorthogonalWavelet(packing.side)
    packed = packing.pack(value)
    coefficients = wavelet(packed)
    low, (horizontal, vertical, diagonal) = pywt.dwt2(
        packed.detach().numpy(), "bior2.4", mode="periodization", axes=(-2,-1))
    expected = np.stack((low,horizontal,vertical,diagonal), axis=2).reshape(coefficients.shape)
    torch.testing.assert_close(coefficients, torch.from_numpy(expected), atol=2e-6, rtol=2e-6)
    actual = packing.unpack(wavelet.inverse(coefficients))
    torch.testing.assert_close(actual, value, atol=2e-6, rtol=2e-6)
    actual.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    support = torch.zeros_like(value, dtype=torch.bool)
    support[:, :, 2, 4, 15] = True
    coefficient_support = wavelet.support(packing.pack(support)).reshape(1, 26, 4, 24, 24)
    assert coefficient_support[:, 15].any()
    assert not coefficient_support[:, :15].any() and not coefficient_support[:, 16:].any()
    assert coefficient_support[:, 15, 1:].any(), "Signed detail cancellation must not hide support"
    with pytest.raises(ValueError, match="axes mismatch"):
        packing.pack(value[:, :-1])


def test_reference_identity_rejects_changed_algorithm(tmp_path):
    entry = reference("wdno")
    entry["source_repo_path"] = str(tmp_path)
    path = tmp_path/"network.py"; path.write_text("changed algorithm")
    entry["source_files"] = {"network.py": "0"*64}
    with pytest.raises(ValueError, match="identity changed"):
        local_reference(entry)
    entry["license_status"] = "missing"
    with pytest.raises(ValueError, match="license"):
        local_reference(entry)


def test_wdno_uses_wavelet_noise_objective_and_target_free_samples():
    torch.manual_seed(41)
    model = NativeWDNO(reference("wdno"), (3,2,2,26), 7)
    target = torch.randn(2,3,2,2,26)
    support = torch.ones_like(target, dtype=torch.bool)
    support[..., 20:] = False
    condition = torch.randn(2,7)
    weights = torch.tensor([.4,.6])
    generator = torch.Generator().manual_seed(42)
    calls = []
    hook = model.network.register_forward_hook(lambda module,args,output:calls.append((args,output)))
    loss = model.loss(target, support, condition, weights, generator)
    hook.remove()
    replay = torch.Generator().manual_seed(42)
    clean = model.wavelet(model.packing.pack(torch.where(support,target,0)))
    mask = model.wavelet.support(model.packing.pack(support))
    times = torch.randint(8,(2,),generator=replay)
    noise = torch.randn(clean.shape,generator=replay)
    alpha = model.alpha_bar[times,None,None,None]
    torch.testing.assert_close(calls[0][0][0][:,:104],
        torch.where(mask,alpha.sqrt()*clean+(1-alpha).sqrt()*noise,0))
    assert torch.equal(calls[0][0][1],times)
    expected = torch.where(mask,(calls[0][1]-noise).square(),0).flatten(1).sum(1)/mask.flatten(1).sum(1)
    torch.testing.assert_close(loss,(expected*weights).sum())
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert model.condition_projection.weight.grad.abs().sum() > 0
    assert sum(p.grad.abs().sum() for p in model.network.parameters() if p.grad is not None) > 0
    model.eval()
    one = model.sample(support, condition, 3, torch.Generator().manual_seed(43))
    same = model.sample(support, condition, 3, torch.Generator().manual_seed(43))
    other = model.sample(support, condition, 3, torch.Generator().manual_seed(41))
    assert torch.equal(one, same) and not torch.equal(one[support], other[support])
    assert one.shape == target.shape and not one[~support].any()
    assert not torch.equal(model.alpha_bar[:-1], model.alpha_bar[1:])
    with pytest.raises(ValueError, match="schedule"):
        model.sample(support, condition, 9, torch.Generator().manual_seed(43))


def test_diffusionpde_log_sigma_loss_and_guided_heun_sampling():
    torch.manual_seed(41)
    model = NativeDiffusionPDE(reference("diffusion_pde"), (3,2,2,26), 7)
    target = torch.randn(2,3,2,2,26)
    support = torch.ones_like(target, dtype=torch.bool)
    support[...,20:] = False
    condition = torch.randn(2,7)
    calls = []
    hook = model.network.register_forward_hook(lambda module,args,output:calls.append((args,output)))
    loss = model.loss(target, support, condition, torch.tensor([.4,.6]),
                      torch.Generator().manual_seed(42))
    hook.remove()
    replay = torch.Generator().manual_seed(42)
    sigma = (torch.randn((2,1,1,1),generator=replay)*1.2-1.2).exp()
    clean = model.packing.pack(torch.where(support,target,0)); mask = model.packing.pack(support)
    noise = torch.randn(clean.shape,generator=replay)
    torch.testing.assert_close(calls[0][0][0],torch.where(mask,clean+sigma*noise,0))
    torch.testing.assert_close(calls[0][0][1],sigma)
    variance_weight = (sigma.square()+.25)/(sigma*.5).square()
    squared = variance_weight*(calls[0][1]-clean).square()
    expected = torch.where(mask,squared,0).flatten(1).sum(1)/mask.flatten(1).sum(1)
    torch.testing.assert_close(loss,(expected*torch.tensor([.4,.6])).sum())
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert sum(p.grad.abs().sum() for p in model.network.parameters() if p.grad is not None) > 0
    model.eval()
    visited = []
    def diagnostic_guidance(value):
        visited.append(tuple(value.shape))
        observation = value[:,0].square().flatten(1).mean(1)
        equation = (value[:,1:]-value[:,:-1]).square().flatten(1).mean(1)
        return observation, equation
    one = model.sample(support, condition, 3, torch.Generator().manual_seed(43), diagnostic_guidance)
    same = model.sample(support, condition, 3, torch.Generator().manual_seed(43), diagnostic_guidance)
    assert torch.equal(one, same) and not one[~support].any()
    assert visited == [tuple(target.shape)]*6
    model.settings["observation_guidance"] = 0.
    model.settings["equation_guidance"] = 0.
    unguided = model.sample(support, condition, 3, torch.Generator().manual_seed(43), diagnostic_guidance)
    assert not torch.equal(one[support], unguided[support])
    with pytest.raises(ValueError, match="guidance"):
        model.sample(support, condition, 3, torch.Generator().manual_seed(43), None)
