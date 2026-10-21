"""State observation boundaries, oracle-only recovery and checkpoint reuse."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from Libs.models.modules.native_state_probe import observation_view
from Libs.models.modules.world_state import FACTOR_NAMES
from Libs.utils.data.joint_observation import RetainedObservation
from Libs.utils.evaluation import native_state_evaluation as state_evaluation
from Libs.utils.evaluation.manuscript_native_protocol import (
    validate_definition, training_profile, command_catalog)
from Libs.utils.data.raw_data_registry import load_config


def test_state_view_removes_hidden_values_observed_and_censored_flags():
    observation = RetainedObservation("o","source","snapshot",("sample",),"validation",
        "readout","gaussian","source_reference_standardized",
        torch.tensor([[2.,float("nan"),7.]]),torch.tensor([[True,False,True]]),
        torch.tensor([[False,True,False]]),(0,1,2),0,normalization_ref="fixed")
    view, = observation_view((observation,),{0})
    assert torch.equal(view.values,torch.tensor([[2.,0.,0.]]))
    assert torch.equal(view.observed,torch.tensor([[True,False,False]]))
    assert not view.left_censored.any()
    assert torch.isnan(observation.values[0,1])


def test_factor_oracle_keeps_admitted_precision_and_rejects_changed_table(monkeypatch):
    number=1.234567890123
    record={"hidden_index":1,"parameters":[{"gain":number},{"gain":2.345678901234}]}
    support=SimpleNamespace(particle_count=2,
        physical_parameter_hypotheses={"source_record_refs":["oracle"],"declarations":[{"name":"gain"}],
            "values":torch.tensor([[[number],[2.345678901234]]],dtype=torch.float64)},
        factors={name:torch.tensor([[[1.],[2.]]]) for name in FACTOR_NAMES})
    monkeypatch.setattr(state_evaluation,"resolve_repo_path",
        lambda path:SimpleNamespace(read_text=lambda:json.dumps(record)))
    index,truth=state_evaluation.controlled_factor_truth(support)
    assert index==1 and all(value.item()==2. for value in truth.values())
    record["parameters"][0]["gain"]=float(torch.tensor(number,dtype=torch.float32))
    with pytest.raises(ValueError,match="differ"):
        state_evaluation.controlled_factor_truth(support)
    record["parameters"][0]["gain"]=number
    record["hidden_index"]=2
    with pytest.raises(ValueError,match="whole-particle index"):
        state_evaluation.controlled_factor_truth(support)


def test_state_profiles_bind_independent_source_fits_and_equal_budgets():
    config=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    definition=validate_definition(config)
    assert len(definition["defined_training_source_contracts"])==2
    assert definition["multiple_training_source_contracts_defined"] is True
    joint,joint_spec=training_profile(config,"state_joint",42)
    alias,alias_spec=training_profile(config,"state_single_source",42)
    mixed,mixed_spec=training_profile(config,"state_mixed_source",42)
    assert alias_spec["training_manifest"]!=joint_spec["training_manifest"]
    assert alias_spec["conditioning_manifest"]!=mixed_spec["conditioning_manifest"]
    assert alias_spec["budget"]==mixed_spec["budget"]
    assert alias_spec["budget"]["selected_training_groups"]==3250
    assert joint_spec["budget"]["selected_training_groups"]==4000
    assert "checkpoint_reuse_profile" not in alias_spec
    assert joint_spec["budget"]["max_optimizer_updates"]==12600
    commands=command_catalog(config,[joint_spec,alias_spec])
    assert any(row["command_uid"]==alias_spec["profile_id"]+":fit" for row in commands)
    evaluate=next(row for row in commands if row["command_uid"]==alias_spec["profile_id"]+":evaluate")
    assert evaluate["dependencies"]==[alias_spec["profile_id"]+":fit"]
    assert evaluate["outputs"][0].endswith("native_state_evaluation.json")
    assert "--training-manifest" in evaluate["argv"]
    changed=copy.deepcopy(config)
    changed["native_state_comparison"]["readout_blocks"]=[[0,1],[1,2],[3,4,5]]
    with pytest.raises(ValueError,match="partition"):
        validate_definition(changed)
    changed=copy.deepcopy(config)
    changed["native_state_comparison"]["objectives"].pop("held_readout")
    with pytest.raises(ValueError,match="objectives"):
        validate_definition(changed)


def test_single_source_cli_loads_its_own_state_checkpoint(tmp_path,monkeypatch):
    from argparse import Namespace
    from Libs.exp.baselines import run_external_baseline_bridge as cli
    from Libs.utils.evaluation import native_comparator_evaluation as comparison
    config,_=training_profile(load_config("Libs/configs/exp/manuscript_native_protocol.yaml"),
                              "state_single_source",42)
    args=Namespace(command="train_baseline",baseline_name="state_single_source",
        output_dir=str(tmp_path/"alias"),conditioning_config="common.yaml",
        conditioning_manifest="selected.json",training_manifest="joint_probe.json",
        device="cpu",evaluation_role="validation",residual_artifact=None)
    calls=[]
    monkeypatch.setattr(comparison,"declared_dataset",lambda *a,**k:([],{}))
    monkeypatch.setattr(comparison,"load_native_comparator",
        lambda cfg,method,*a:(calls.append(method) or object(),None,{}))
    monkeypatch.setattr(state_evaluation,"evaluate_native_state",
        lambda cfg,method,*a:(calls.append(method) or {"status":"complete"}))
    args.command="evaluate_baseline"
    assert cli.run_native_command(args,config)==0
    assert calls==["state_single_source","state_single_source"]
