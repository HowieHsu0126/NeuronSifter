"""Numerical forecast-boundary checks used by the retained native CLI fixture."""
from pathlib import Path

import torch

from Libs.utils.exp.native_comparator_training import prepare_comparator_group
from Libs.utils.model.native_comparator_inputs import native_comparator_group_loss,sample_native_comparator


def verify_forecast_contract(model,belief,group,config):
    inputs,target,mask,weights=prepare_comparator_group(group,belief,config,torch.device("cpu"))
    normalizer=belief.field_normalization
    def sample():
        return sample_native_comparator(model,inputs,normalizer,draws_per_particle=2,steps=1,
            seed=45)["particle_trajectory_samples"]
    original_filter=model.filter_step
    original_read=Path.read_text
    def forbidden(*args,**kwargs):raise AssertionError("Forecast used future-target posterior filtering")
    def read(path,*args,**kwargs):
        if str(path).endswith("_producer.json"):raise AssertionError("RSSM opened a producer oracle")
        return original_read(path,*args,**kwargs)
    Path.read_text=read
    try:
        model.eval();model.filter_step=forbidden
        before=sample()
        assert torch.isfinite(before).all() and (before[0]-before[1]).abs().max()>0
        original=inputs.posterior.support.initial_field[:,:,None].expand(
            *inputs.group_shape,*inputs.initial.shape[-3:])
        torch.testing.assert_close(before[:,:,:,:,0],original[None].expand_as(before[:,:,:,:,0]),rtol=0,atol=0)
        model.filter_step=original_filter
        model.train();model.zero_grad(set_to_none=True)
        loss=native_comparator_group_loss(model,inputs,target,mask,weights,normalizer,
            torch.Generator().manual_seed(46))
        loss.backward()
        gradient=float(sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None))
        assert gradient>0 and torch.isfinite(loss)
        model.zero_grad(set_to_none=True)
        changed=target.clone();changed[:,1:]+=normalizer.scale.to(target)
        second=native_comparator_group_loss(model,inputs,changed,mask,weights,normalizer,
            torch.Generator().manual_seed(46))
        assert not torch.equal(loss.detach(),second.detach())
        model.eval();model.filter_step=forbidden
        after=sample()
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        # A tensor-level causal-prefix check; no modified program is admitted.
        original_sequence=model.sequence_inputs
        def later_inputs(current):
            initial,support,static,program=original_sequence(current)
            program=program.clone();program[:,3:]+=1
            return initial,support,static,program
        model.sequence_inputs=later_inputs
        try:later=sample()
        finally:model.sequence_inputs=original_sequence
        torch.testing.assert_close(before[:,:,:,:,:3],later[:,:,:,:,:3],rtol=0,atol=0)
        assert not torch.equal(before[:,:,:,:,3:],later[:,:,:,:,3:])
        return {"nonzero_training_gradient_sum":gradient,"future_filtering_blocked_during_prediction":True,
            "no_fitting_state_retained_by_forecast":True,"tensor_causal_prefix":True,
            "source_oracle_reads_blocked":True,"stochastic_draws_distinct":True,"initial_field_preserved":True}
    finally:
        model.filter_step=original_filter
        Path.read_text=original_read
