"""Mathematical KL behavior and prospective native scale dispatch."""
import copy

import pytest
import torch

from Libs.models.modules.quantized_mddm_rollout_operator import masked_categorical_distillation
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.evaluation.manuscript_native_protocol import training_profile,command_catalog,validate_definition
from Libs.utils.model.native_scientific_source import native_training_profile


def test_masked_kl_has_teacher_direction_and_only_student_gradients():
    student=torch.tensor([[1.,-1.],[.3,.8]],requires_grad=True)
    teacher=torch.tensor([[.2,.7],[.4,-.2]],requires_grad=True)
    mask=torch.tensor([True,False]);support=torch.ones(2,dtype=torch.bool)
    u=torch.tensor(.25);confidence=torch.tensor(.5)
    loss=masked_categorical_distillation(student,teacher,mask,u,confidence,support)
    target=teacher.detach().softmax(-1)
    kl=(target*(teacher.detach().log_softmax(-1)-student.log_softmax(-1))).sum(-1)
    expected=confidence*(1-(1-u)**2)*kl[0]/(2*u)
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert teacher.grad is None
    assert student.grad[0].abs().sum()>0 and torch.equal(student.grad[1],torch.zeros(2))
    with pytest.raises(ValueError,match="aligned"):
        masked_categorical_distillation(student,teacher,~support,u,confidence,support)
    with pytest.raises(ValueError,match="aligned"):
        masked_categorical_distillation(student,teacher,mask,u,confidence,~support)


def test_scale_profiles_bind_actual_teacher_keep_budgets_and_reuse_large():
    config=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    validate_definition(config)
    generated={name:training_profile(config,name,41) for name in ("native_wam","student","student_no_distillation","large","xl")}
    specs=[entry[1] for entry in generated.values()]
    full=generated["native_wam"][0]
    for name,(selected,spec) in generated.items():
        assert selected["native_scientific_training"]["selection"]["max_steps"]==12600
        assert selected["screening_pipeline"]["screening_benchmark"]["training"]["memory_policy"]["effective_batch_size"]==64
        assert selected["native_learned_distillation"]["enabled"]==(name=="student")
    student=generated["student"][0];plain=generated["student_no_distillation"][0]
    assert student["native_learned_distillation"]["teacher_manifest"]==generated["native_wam"][1]["training_manifest"]
    assert student["native_scientific_training"]["conditioning"]==plain["native_scientific_training"]["conditioning"]
    assert generated["large"][1]["training_manifest"]==generated["native_wam"][1]["training_manifest"]
    assert native_training_profile(full,"fixture")==native_training_profile(generated["large"][0],"fixture")
    commands=command_catalog(config,specs)
    assert not any(row["command_uid"]==generated["large"][1]["profile_id"]+":fit" for row in commands)
    for name in ("student","student_no_distillation","xl"):
        row=next(row for row in commands if row["command_uid"]==generated[name][1]["profile_id"]+":fit")
        assert "Libs/exp/training/pretrain_microenvironment_world_model.py" in row["argv"]
        assert generated["native_wam"][1]["profile_id"]+":fit" in row["dependencies"]
    invalid=copy.deepcopy(config);invalid["native_learned_distillation"]["weight"]=0
    with pytest.raises(ValueError,match="positive"):validate_definition(invalid)


def test_large_alias_uses_mainline_loader_and_rejects_duplicate_fit(tmp_path,monkeypatch):
    from argparse import Namespace
    from Libs.exp.baselines import run_external_baseline_bridge as cli
    from Libs.utils.evaluation import native_comparator_evaluation as evaluation
    config,_=training_profile(load_config("Libs/configs/exp/manuscript_native_protocol.yaml"),"large",41)
    args=Namespace(command="train_baseline",baseline_name="large",output_dir=str(tmp_path/"large"),
        conditioning_config="common.yaml",conditioning_manifest="selected.json",training_manifest="selected.json",
        device="cpu",evaluation_role="validation",residual_artifact=None)
    with pytest.raises(ValueError,match="must not refit"):cli.run_native_command(args,config)
    assert not (tmp_path/"large").exists()
    calls=[]
    monkeypatch.setattr(evaluation,"declared_dataset",lambda *a,**k:([],{}))
    monkeypatch.setattr(evaluation,"load_native_comparator",
        lambda cfg,method,*a:(calls.append(method) or object(),None,{}))
    monkeypatch.setattr(evaluation,"evaluate_native_comparator",
        lambda cfg,method,*a,decision_reference=None:(calls.append(method) or {"status":"complete"}))
    args.command="evaluate_baseline"
    assert cli.run_native_command(args,config)==0
    assert calls==["native_wam","native_wam"]
