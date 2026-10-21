"""Native conditional adapters for the local WDNO and DiffusionPDE references.

The external networks are imported unchanged from configured, fingerprinted local
sources. Packing only permutes and zero-pads axes; it never interpolates a clock
or invents a biological target. WDNO uses bior2.4 periodized coefficients and
cosine noise-prediction diffusion. DiffusionPDE uses its lognormal EDM weighting
and second-order sampler, including caller-bound observation/equation guidance.
"""
from __future__ import annotations

import hashlib
import importlib
import math
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from Libs.utils.data.raw_data_registry import resolve_repo_path
from Libs.utils.exp.external_baseline_bridge import prepend_python_path


def local_reference(entry: dict) -> Path:
    """Fail before import when the declared local algorithm identity changed."""
    root = resolve_repo_path(entry["source_repo_path"]).resolve()
    if not entry.get("source_files") or entry.get("license_status") != "declared":
        raise ValueError("Comparator requires declared source files and license")
    for relative, expected in entry["source_files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing declared comparator source: " + relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Comparator source identity changed: " + relative)
    return root


def _import_reference(root: Path, module: str, *extra: Path):
    # External projects use generic package names. A cached module from another
    # reference must never silently supply an algorithm to this adapter.
    with prepend_python_path(root, *extra):
        imported = importlib.import_module(module)
    if not Path(imported.__file__).resolve().is_relative_to(root):
        raise ValueError("External module collision: " + module)
    return imported


class NativeFieldPacking:
    """Invertible [N,T,R,C,S] <-> [N,S,H,W], padding outside the real grid."""

    def __init__(self, shape: tuple[int, int, int, int], multiple: int):
        if len(shape) != 4 or min(shape) < 1 or multiple < 1:
            raise ValueError("Native comparator requires positive T/R/C/S axes")
        self.shape = tuple(shape)
        t, r, c, _ = shape
        self.side = math.ceil(max(t, r*c) / multiple) * multiple

    def pack(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5 or tuple(value.shape[1:]) != self.shape:
            raise ValueError("Native comparator field axes mismatch")
        t, r, c, s = self.shape
        packed = value.reshape(value.shape[0], t, r*c, s).permute(0, 3, 1, 2)
        return F.pad(packed, (0, self.side-r*c, 0, self.side-t))

    def unpack(self, value: torch.Tensor) -> torch.Tensor:
        t, r, c, s = self.shape
        if value.ndim != 4 or tuple(value.shape[1:]) != (s, self.side, self.side):
            raise ValueError("Packed comparator axes mismatch")
        return value[:, :, :t, :r*c].permute(0, 2, 3, 1).reshape(-1, t, r, c, s)


class PeriodizedBiorthogonalWavelet(nn.Module):
    def __init__(self, side: int):
        super().__init__()
        # The local reference declares PyWavelets' bior2.4/periodization.
        # Fixed analysis/synthesis matrices retain that exact transform and
        # autograd, without importing pytorch_wavelets' unrelated DTCWT loader.
        import numpy as np
        import pywt
        if side < 2 or side % 2:
            raise ValueError("Periodized wavelet packing requires an even side")
        low, high = pywt.dwt(np.eye(side), "bior2.4", "periodization", axis=0)
        eye = np.eye(side//2)
        synth_low = pywt.idwt(eye, np.zeros_like(eye), "bior2.4", "periodization", axis=0)
        synth_high = pywt.idwt(np.zeros_like(eye), eye, "bior2.4", "periodization", axis=0)
        for name, value in (("low",low),("high",high),("synth_low",synth_low),("synth_high",synth_high)):
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

    @staticmethod
    def _bands(value: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        # Exact PyWavelets band order: LL, HL, LH, HH.
        bands = [left @ value @ right.T for left, right in (
            (low,low),(high,low),(low,high),(high,high))]
        return torch.stack(bands, dim=2).flatten(1, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self._bands(value, self.low.to(value), self.high.to(value))

    def inverse(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.shape[1] % 4:
            raise ValueError("Wavelet coefficients require four bands per quantity")
        bands = coefficients.reshape(coefficients.shape[0], -1, 4, *coefficients.shape[-2:])
        low, high = self.synth_low.to(coefficients), self.synth_high.to(coefficients)
        return sum(left @ bands[:, :, index] @ right.T for index, (left, right) in enumerate(
            ((low,low),(high,low),(low,high),(high,high))))

    def support(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype != torch.bool:
            raise ValueError("Wavelet support requires boolean validity")
        # Propagate absolute filters to retain signed detail dependencies.
        return self._bands(mask.float(), self.low.abs(), self.high.abs()) > 0


def _validate_inputs(target: torch.Tensor, support: torch.Tensor,
                     condition: torch.Tensor, width: int) -> None:
    if (support.dtype != torch.bool or support.shape != target.shape
            or condition.ndim != 2 or condition.shape != (target.shape[0], width)
            or not torch.isfinite(condition).all()
            or not torch.isfinite(target[support]).all()
            or not support.flatten(1).any(1).all()):
        raise ValueError("Comparator needs finite conditioning and supported native fields")


def _weighted_loss(values: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if (weight.shape != (values.shape[0],) or not torch.isfinite(weight).all()
            or (weight < 0).any() or weight.sum() <= 0):
        raise ValueError("Comparator loss requires explicit positive total row weight")
    count = mask.flatten(1).sum(1)
    if (count == 0).any() or not torch.isfinite(values[mask]).all():
        raise ValueError("Comparator loss is undefined on missing/nonfinite targets")
    means = torch.where(mask, values, 0).flatten(1).sum(1) / count
    return (means * weight).sum() / weight.sum()


class NativeWDNO(nn.Module):
    def __init__(self, entry: dict, shape: tuple[int, int, int, int], condition_width: int):
        super().__init__()
        root = local_reference(entry)
        network = _import_reference(root, "ddpm_burgers.unet", root / "burgers")
        settings = entry["hyperparameters"]
        if settings["wavelet"] != "bior2.4" or settings["wavelet_mode"] != "periodization":
            raise ValueError("WDNO reference requires declared bior2.4 periodization")
        self.condition_width = condition_width
        self.steps = int(settings["diffusion_steps"])
        self.eta = float(settings["ddim_eta"])
        self.shape = tuple(shape)
        self.packing = NativeFieldPacking(shape, 2 ** len(settings["dim_mults"]))
        self.wavelet = PeriodizedBiorthogonalWavelet(self.packing.side)
        channels = shape[-1] * 4
        conditioning_channels = int(settings["conditioning_channels"])
        self.condition_projection = nn.Linear(condition_width, conditioning_channels)
        self.network = network.Unet2D(
            dim=int(settings["dim"]), out_dim=channels,
            dim_mults=tuple(settings["dim_mults"]), channels=channels+conditioning_channels,
            resnet_block_groups=int(settings["resnet_block_groups"]))
        if self.steps < 2 or not 0 <= self.eta <= 1:
            raise ValueError("Invalid WDNO diffusion schedule")
        t = torch.linspace(0, self.steps, self.steps+1, dtype=torch.float64) / self.steps
        alpha = torch.cos((t + .008) / 1.008 * math.pi / 2).square()
        alpha = alpha / alpha[0]
        beta = (1-alpha[1:]/alpha[:-1]).clamp(0, .999)
        self.register_buffer("alpha_bar", (1-beta).cumprod(0).float())

    def predict_noise(self, noisy: torch.Tensor, time: torch.Tensor,
                      condition: torch.Tensor) -> torch.Tensor:
        context = self.condition_projection(condition)[:, :, None, None]
        context = context.expand(-1, -1, *noisy.shape[-2:])
        return self.network(torch.cat((noisy, context), dim=1), time)

    def loss(self, target: torch.Tensor, support: torch.Tensor, condition: torch.Tensor,
             row_weight: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        _validate_inputs(target, support, condition, self.condition_width)
        clean = self.wavelet(self.packing.pack(torch.where(support, target, 0)))
        mask = self.wavelet.support(self.packing.pack(support))
        time = torch.randint(self.steps, (len(target),), device=target.device, generator=generator)
        alpha = self.alpha_bar[time, None, None, None]
        noise = torch.randn(clean.shape, dtype=clean.dtype, device=clean.device, generator=generator)
        noisy = torch.where(mask, alpha.sqrt()*clean+(1-alpha).sqrt()*noise, 0)
        predicted = self.predict_noise(noisy, time, condition)
        return _weighted_loss((predicted-noise).square(), mask, row_weight)

    @torch.no_grad()
    def sample(self, support: torch.Tensor, condition: torch.Tensor, steps: int,
               generator: torch.Generator) -> torch.Tensor:
        _validate_inputs(torch.zeros_like(support, dtype=condition.dtype), support,
                         condition, self.condition_width)
        if not 1 <= steps <= self.steps:
            raise ValueError("WDNO sampling steps exceed the trained schedule")
        mask = self.wavelet.support(self.packing.pack(support))
        value = torch.randn(mask.shape, device=condition.device, dtype=condition.dtype,
                            generator=generator) * mask
        schedule = torch.linspace(-1, self.steps-1, steps+1, device=value.device).long().flip(0)
        for current, following in zip(schedule[:-1], schedule[1:]):
            time = current.expand(len(value))
            noise = self.predict_noise(value, time, condition)
            alpha = self.alpha_bar[current]
            clean = (value-(1-alpha).sqrt()*noise)/alpha.sqrt()
            if following < 0:
                value = clean
            else:
                next_alpha = self.alpha_bar[following]
                sigma = self.eta * ((1-alpha/next_alpha)*(1-next_alpha)/(1-alpha)).sqrt()
                coefficient = (1-next_alpha-sigma.square()).clamp_min(0).sqrt()
                fresh = torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator)
                value = next_alpha.sqrt()*clean+coefficient*noise+sigma*fresh
            value = torch.where(mask, value, 0)
        physical = self.packing.unpack(self.wavelet.inverse(value))
        if not torch.isfinite(physical[support]).all():
            raise ValueError("Nonfinite WDNO sample")
        return torch.where(support, physical, 0)


class NativeDiffusionPDE(nn.Module):
    def __init__(self, entry: dict, shape: tuple[int, int, int, int], condition_width: int):
        super().__init__()
        root = local_reference(entry)
        network = _import_reference(root, "training.networks")
        self.settings = dict(entry["hyperparameters"])
        settings = self.settings
        self.condition_width = condition_width
        self.packing = NativeFieldPacking(shape, 2 ** (len(settings["channel_mult"])-1))
        self.network = network.EDMPrecond(
            img_resolution=self.packing.side, img_channels=shape[-1],
            label_dim=condition_width, model_type="SongUNet",
            model_channels=int(settings["model_channels"]),
            channel_mult=list(settings["channel_mult"]),
            num_blocks=int(settings["num_blocks"]),
            sigma_data=float(settings["sigma_data"]),
            attn_resolutions=list(settings["attention_resolutions"]))
        if (settings["P_std"] <= 0 or settings["sigma_data"] <= 0
                or not 0 < settings["sigma_min"] < settings["sigma_max"]
                or settings["rho"] <= 0):
            raise ValueError("Invalid EDM noise schedule")

    def loss(self, target: torch.Tensor, support: torch.Tensor, condition: torch.Tensor,
             row_weight: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        _validate_inputs(target, support, condition, self.condition_width)
        clean = self.packing.pack(torch.where(support, target, 0))
        mask = self.packing.pack(support)
        rnd = torch.randn((len(target),1,1,1), dtype=target.dtype, device=target.device, generator=generator)
        sigma = (rnd*self.settings["P_std"]+self.settings["P_mean"]).exp()
        data = self.settings["sigma_data"]
        weight = (sigma.square()+data**2)/(sigma*data).square()
        noise = torch.randn(clean.shape, dtype=target.dtype, device=target.device, generator=generator)
        denoised = self.network(torch.where(mask, clean+sigma*noise, 0), sigma, condition)
        return _weighted_loss(weight*(denoised-clean).square(), mask, row_weight)

    def sample(self, support: torch.Tensor, condition: torch.Tensor, steps: int,
               generator: torch.Generator,
               guidance: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Guidance receives only predicted native fields; closures bind retained
        readouts and compiled physical laws, never a future teacher or oracle."""
        _validate_inputs(torch.zeros_like(support, dtype=condition.dtype), support,
                         condition, self.condition_width)
        if steps < 2 or not callable(guidance):
            raise ValueError("DiffusionPDE requires observation and equation guidance")
        cfg = self.settings
        mask = self.packing.pack(support)
        ramp = torch.linspace(0, 1, steps, device=condition.device, dtype=torch.float64)
        times = (cfg["sigma_max"]**(1/cfg["rho"])+ramp*(
            cfg["sigma_min"]**(1/cfg["rho"])-cfg["sigma_max"]**(1/cfg["rho"])))**cfg["rho"]
        times = torch.cat((times, times.new_zeros(1)))
        value = torch.randn(mask.shape, dtype=condition.dtype, device=condition.device,
                            generator=generator).double()*times[0]*mask
        for index, (current, following) in enumerate(zip(times[:-1], times[1:])):
            with torch.enable_grad():
                x = value.detach().requires_grad_(True)
                denoised = self.network(x, current, condition).double()
                derivative = (x-denoised)/current
                next_value = x+(following-current)*derivative
                if index < steps-1:
                    corrected = self.network(next_value, following, condition).double()
                    next_value = x+(following-current)*.5*(derivative+(next_value-corrected)/following)
                    denoised = corrected
                native = self.packing.unpack(denoised)
                observation, equation = guidance(native)
                if (observation.shape != (len(value),) or equation.shape != observation.shape
                        or not torch.isfinite(observation).all() or not torch.isfinite(equation).all()
                        or (observation < 0).any() or (equation < 0).any()):
                    raise ValueError("DiffusionPDE guidance must return finite per-row losses")
                obs_gradient = torch.autograd.grad(observation.sum(), x, retain_graph=True)[0]
                equation_gradient = torch.autograd.grad(equation.sum(), x)[0]
                early = index <= cfg["guidance_switch_fraction"]*steps
                observation_scale = cfg["observation_guidance"] if early else (
                    cfg["observation_guidance"]*cfg["late_observation_ratio"])
                equation_scale = 0 if early else cfg["equation_guidance"]
                value = torch.where(mask, next_value-observation_scale*obs_gradient
                                    -equation_scale*equation_gradient, 0).detach()
        native = self.packing.unpack(value).to(condition.dtype)
        if not torch.isfinite(native[support]).all():
            raise ValueError("Nonfinite DiffusionPDE sample")
        return torch.where(support, native, 0)


class NativeCODANO(nn.Module):
    """Maintained NeuralOperator CODANO: a conditional mean until residual fit."""

    def __init__(self, entry: dict, shape: tuple[int,int,int,int], condition_width: int,
                 quantity_ids: list[str]):
        super().__init__()
        import importlib.metadata
        root = local_reference(entry)
        if importlib.metadata.version("neuraloperator") != entry["package_version"]:
            raise ValueError("CODANO installed version differs from the protocol")
        implementation = _import_reference(root, "neuralop.models.codano")
        settings = entry["hyperparameters"]
        if len(quantity_ids) != shape[-1] or len(set(quantity_ids)) != shape[-1]:
            raise ValueError("CODANO requires each native quantity identity")
        self.quantity_ids = list(quantity_ids)
        self.condition_width = condition_width
        self.packing = NativeFieldPacking(shape,1)
        self.condition_projection = nn.Linear(condition_width,settings["conditioning_channels"])
        layers = settings["n_layers"]
        self.network = implementation.CODANO(
            output_variable_codimension=1,
            lifting_channels=settings["lifting_channels"],
            hidden_variable_codimension=settings["hidden_variable_codimension"],
            projection_channels=settings["projection_channels"],
            static_channel_dim=settings["conditioning_channels"],
            use_positional_encoding=True, positional_encoding_dim=settings["positional_encoding_dim"],
            positional_encoding_modes=list(settings["positional_encoding_modes"]),
            variable_ids=list(quantity_ids), n_layers=layers,
            n_modes=[list(settings["n_modes"]) for _ in range(layers)],
            n_heads=[settings["n_heads"]]*layers,
            per_layer_scaling_factors=[[1,1] for _ in range(layers)],
            attention_scaling_factors=[1.]*layers, domain_padding=0.)

    def predict(self, support: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        _validate_inputs(torch.zeros_like(support,dtype=condition.dtype),support,
                         condition,self.condition_width)
        t,r,c,s = self.packing.shape
        # The caller contract fixes the full normalized initial field as block0.
        initial=condition[:,:r*c*s].reshape(-1,1,r,c,s).expand(-1,t,-1,-1,-1)
        features=self.packing.pack(torch.where(support,initial,0))
        static=self.condition_projection(condition)[:,:,None,None].expand(
            -1,-1,self.packing.side,self.packing.side)
        prediction=self.network(features,static_channel=static,input_variable_ids=self.quantity_ids)
        output=self.packing.unpack(prediction)
        if not torch.isfinite(output[support]).all():
            raise ValueError("Nonfinite CODANO conditional mean")
        return torch.where(support,output,0)

    def loss(self,target,support,condition,row_weight,generator):
        _validate_inputs(target,support,condition,self.condition_width)
        predicted=self.predict(support,condition)
        return _weighted_loss((predicted-target).square(),support,row_weight)


class NativeFNO(nn.Module):
    """Maintained dense FNO on the reversible native lattice; mean-only until calibration."""
    prediction_kind="mean_only"

    def __init__(self,entry:dict,shape:tuple[int,int,int,int],condition_width:int):
        super().__init__()
        import importlib.metadata
        root=local_reference(entry)
        if importlib.metadata.version(entry["package"])!=entry["package_version"]:
            raise ValueError("FNO installed version differs from the bound reference")
        implementation=_import_reference(root,"neuralop.models.fno")
        settings=entry["hyperparameters"]
        self.packing=NativeFieldPacking(shape,1)
        self.condition_width=condition_width
        self.condition_projection=nn.Linear(condition_width,settings["conditioning_channels"])
        self.network=implementation.FNO(n_modes=tuple(settings["n_modes"]),
            in_channels=2*shape[-1]+settings["conditioning_channels"],out_channels=shape[-1],
            hidden_channels=settings["hidden_channels"],n_layers=settings["n_layers"],
            positional_embedding="grid",domain_padding=0.,factorization=None,
            fno_block_precision="full",channel_mlp_dropout=settings["dropout_rate"])

    def load_state_dict(self,state_dict,strict=True,assign=False):
        # The maintained FNO BaseModel writes its constructor metadata at the
        # enclosing dictionary root. Validate it before PyTorch loads this wrapper.
        supplied=state_dict.copy()
        metadata=supplied.pop("_metadata",None)
        if metadata!=self.network._init_kwargs:
            raise ValueError("FNO checkpoint constructor metadata differs from its declared reference")
        if hasattr(state_dict,"_metadata"):supplied._metadata=state_dict._metadata
        return super().load_state_dict(supplied,strict=strict,assign=assign)

    def predict(self,support,condition):
        _validate_inputs(torch.zeros_like(support,dtype=condition.dtype),support,condition,self.condition_width)
        t,r,c,s=self.packing.shape
        initial=condition[:,:r*c*s].reshape(-1,1,r,c,s).expand(-1,t,-1,-1,-1)
        static=self.condition_projection(condition)[:,:,None,None].expand(-1,-1,self.packing.side,self.packing.side)
        features=torch.cat((self.packing.pack(torch.where(support,initial,0)),
                            self.packing.pack(support.to(condition)),static),1)
        predicted=self.packing.unpack(self.network(features))
        predicted=torch.cat((initial[:,:1],predicted[:,1:]),1)
        if not torch.isfinite(predicted[support]).all():raise FloatingPointError("Nonfinite native FNO mean")
        return torch.where(support,predicted,0)

    def loss(self,target,support,condition,row_weight,generator):
        _validate_inputs(target,support,condition,self.condition_width)
        return _weighted_loss((self.predict(support,condition)-target).square(),support,row_weight)
