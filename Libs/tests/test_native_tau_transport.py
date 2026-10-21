from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from Libs.utils.model.native_training_selection import training_checkpoint_path

from Libs.models.modules.native_tau_transport import (
    FourierDifferentialLayer, TauFunctionOperator, directed_proximities, NativeTauTransport,
)


def test_zero_sum_difference_annihilates_constants_and_uses_only_available_modes():
    torch.manual_seed(19)
    layer=FourierDifferentialLayer(4,6,16,1/5)
    constant=torch.randn(3,1,4).expand(-1,6,-1)
    assert torch.equal(layer.differential(constant),torch.zeros_like(constant))
    assert layer.requested_modes==16 and layer.actual_modes==4
    values=torch.randn(3,6,4,requires_grad=True)
    predicted=layer(values)
    predicted.square().mean().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert layer.spectral.grad is not None and layer.spectral.grad.abs().sum()>0
    assert layer.stencil.grad is not None and layer.stencil.grad.abs().sum()>0


def test_directed_surrogates_preserve_inbound_outbound_proximity_and_isolated_nodes():
    adjacency=torch.tensor([[0.,2.,1.],[0.,0.,3.],[4.,0.,0.]],dtype=torch.float64)
    graphs=directed_proximities(adjacency)
    assert torch.isfinite(graphs).all()
    assert torch.allclose(graphs,graphs.transpose(1,2),atol=1e-14)
    # Source/sink second-order proximities are distinct; transpose exchanges them.
    assert not torch.allclose(graphs[1],graphs[2])
    assert graphs[1,1,2] == pytest.approx((2/3)/(15**.5),abs=1e-14)
    assert graphs[2,0,1] == pytest.approx(.1875,abs=1e-14)
    transposed=directed_proximities(adjacency.T)
    assert torch.equal(graphs[0],transposed[0])
    assert torch.equal(graphs[1],transposed[2])
    assert torch.equal(graphs[2],transposed[1])
    isolated=directed_proximities(torch.zeros(3,3,dtype=torch.float64))
    assert torch.equal(isolated,torch.eye(3,dtype=torch.float64).expand(3,-1,-1))
    with pytest.raises(ValueError,match="nonnegative"):
        directed_proximities(-torch.ones(2,2))


def test_query_path_uses_only_initial_field_and_function_path_responds_to_parameters():
    torch.manual_seed(21)
    initial=torch.randn(2,6,3)
    parameters=torch.randn(2,6,2)
    qo=TauFunctionOperator(3,8,6,16,.2)
    fo=TauFunctionOperator(5,8,6,16,.2)
    original_query=qo(initial).detach().clone()
    original_function=fo(torch.cat((initial,parameters),-1)).detach().clone()
    changed=parameters+.3
    assert torch.equal(qo(initial),original_query)
    assert not torch.equal(fo(torch.cat((initial,changed),-1)),original_function)
    assert len(qo.layers)==len(fo.layers)==2


def test_tiny_tau_native_four_training_domains_actual_fields_and_backward(tmp_path):
    from Libs.utils.data.raw_data_registry import load_config
    from Libs.utils.evaluation.native_comparator_evaluation import declared_dataset
    from Libs.utils.exp.native_comparator_training import load_frozen_native_belief,prepare_comparator_group
    from Libs.utils.model.native_comparator_inputs import native_comparator_group_loss

    location=os.environ.get("NRNSIFT_RHS_FIXTURE")
    if not location:pytest.skip("Requires retained artificial training arrays and selected belief")
    fixture=Path(location)
    config=load_config(str(fixture/"config.yaml"))
    belief,binding=load_frozen_native_belief(config,fixture/"config.yaml",fixture/"native_fit/manifest.json","cpu")
    dataset,declarations=declared_dataset(config,"train")
    settings={"operator_layers":[2,2,2],"fourier_modes":16,"activation":"gelu",
        "spatial_axis":"declared_region_index","loss":"relative_l2_train_normalized_native_fields",
        "hidden_dim":8}
    model=None;rows=[]
    for index in range(len(dataset)):
        group=dataset[index]["native_joint_group"]
        inputs,target,mask,weights=prepare_comparator_group(group,belief,config,"cpu")
        assert declarations[inputs.sample_uids[0]]["role"]=="train"
        if model is None:model=NativeTauTransport(settings,inputs,config)
        model.zero_grad(set_to_none=True)
        loss=native_comparator_group_loss(model,inputs,target,mask,weights,belief.field_normalization,
                                          torch.Generator().manual_seed(19))
        assert loss.ndim==0 and torch.isfinite(loss) and loss>0
        loss.backward()
        gradients={name:float(sum((parameter.grad.abs().sum() for parameter in module.parameters()
                          if parameter.grad is not None),torch.tensor(0.)))
                   for name,module in (("function",model.function_operator),("query",model.query_operator),
                                       ("directed_graph",model.graph_operator),("trajectory",model.trajectory_projection))}
        assert all(value>0 for value in gradients.values()),gradients
        with torch.no_grad():predicted=model.predict_native(inputs,belief.field_normalization)
        assert predicted.shape==inputs.validity.shape and torch.isfinite(predicted).all()
        assert inputs.group_shape==(1,32,3) and predicted.shape[-3:]==(6,6,26)
        rows.append({"family_uid":inputs.grid["family_uid"],"role":"train",
                     "original_time_count":predicted.shape[1],"loss":float(loss.detach()),
                     "gradient_l1_by_module":gradients})
    assert len(rows)==4
    assert model.method_mapping["actual_fourier_modes"]==4
    receipt={"scope":"tiny_tau_native_source_format_training_fields_software_backward_only",
        "hidden_dim":8,"full_paper_hidden_dim":64,"optimizer_updates":0,"new_source_provider_calls":0,
        "selected_belief":binding,"method_mapping":model.method_mapping,
        "actual_parameter_count":sum(parameter.numel() for parameter in model.parameters()),"groups":rows,
        "scientific_twin":"NOT_RUN","formal_qualification":False}
    Path(os.environ.get("NRNSIFT_TAU_EVIDENCE",str(tmp_path/"result.json"))).write_text(json.dumps(receipt,indent=2))


def test_tau_catalog_retains_paper_recipe_and_independent_residual_calibration():
    from Libs.utils.data.raw_data_registry import load_config
    from Libs.utils.evaluation.manuscript_native_protocol import training_profile,command_catalog
    config=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    selected,profile=training_profile(config,"tau_transport",41)
    reference=selected["native_controlled_comparators"]["references"]["tau_transport"]
    assert reference["hyperparameters"]["hidden_dim"]==64
    assert reference["hyperparameters"]["operator_layers"]==[2,2,2]
    assert reference["hyperparameters"]["fourier_modes"]==16
    rows={row["command_uid"]:row for row in command_catalog(config,[profile])}
    uid=profile["profile_id"]
    assert uid+":fit" in rows and uid+":calibrate" in rows and uid+":evaluate" in rows
    assert rows[uid+":calibrate"]["dependencies"]==[uid+":fit"]
    assert rows[uid+":evaluate"]["dependencies"]==[uid+":calibrate"]
    assert "--residual-artifact" in rows[uid+":evaluate"]["argv"]


def test_actual_tiny_tau_cli_fit_strict_reload_and_uncalibrated_withholding(tmp_path):
    import copy
    import subprocess
    import sys
    import yaml
    from Libs.utils.data.raw_data_registry import load_config

    location=os.environ.get("NRNSIFT_RHS_FIXTURE")
    if not location:pytest.skip("Requires retained artificial arrays and selected belief")
    fixture=Path(location)
    root=Path(__file__).resolve().parents[2]
    artifacts=Path(os.environ.get("NRNSIFT_TAU_ARTIFACTS",str(tmp_path/"artifacts")))
    artifacts.mkdir(parents=True,exist_ok=False)
    reused=os.environ.get("NRNSIFT_TAU_REUSE_ARTIFACTS")
    if reused:
        prior=Path(reused)
        config_path=prior/"tau_evaluation_config.yaml"
        train_output=prior/"training"
    else:
        config=load_config(str(fixture/"config.yaml"))
        current=load_config(str(root/"Libs/configs/exp/manuscript_native_protocol.yaml"))
        reference=copy.deepcopy(current["native_controlled_comparators"]["references"]["tau_transport"])
        reference["hyperparameters"]["hidden_dim"]=8
        config["native_controlled_comparators"]["references"]["tau_transport"]=reference
        if "native_profile_identity" in config:
            config["native_profile_identity"]["method"]="tau_transport"
        config["native_controlled_comparators"]["conditional_mean_methods"].append("tau_transport")
        config["native_scientific_training"]["selection"].update(max_steps=2,max_epochs=1,minimum_steps=1,minimum_epochs=1)
        config["native_scientific_training"]["validation"]["interval_optimizer_steps"]=1
        config_path=artifacts/"tau_evaluation_config.yaml"
        config_path.write_text(yaml.safe_dump(config,sort_keys=False))
        train_output=artifacts/"training"
    common=[sys.executable,str(root/"Libs/exp/baselines/run_external_baseline_bridge.py"),
        "--protocol-scope","manuscript-native","--config",str(config_path),
        "--baseline-name","tau_transport","--conditioning-config",str(fixture/"config.yaml"),
        "--conditioning-manifest",str(fixture/"native_fit/manifest.json"),"--device","cpu"]
    train_exit=None
    if not reused:
        completed=subprocess.run(common+["--command","train_baseline","--output-dir",str(train_output)],
                                 cwd=root,text=True,capture_output=True)
        (artifacts/"train_cli.log").write_text(completed.stdout+completed.stderr)
        assert completed.returncode==2,completed.stdout+completed.stderr
        train_exit=completed.returncode
    manifest=train_output/"native_comparator_training_manifest.json"
    trained=json.loads(manifest.read_text())
    assert trained["executed_steps"]==1 and trained["failure"] is None
    assert trained["development_group_count"]==4
    assert trained["development_metric"]=="posterior_weighted_relative_l2_train_normalized_fields"
    assert trained["selection"]["best"]["steps"]==1
    assert trained["selection"]["terminal"]=="INCOMPLETE_EPOCH_CAP"
    assert trained["actual_predictor_parameter_count"]==338168
    output=artifacts/"evaluation"
    evaluated=subprocess.run(common+["--command","evaluate_baseline","--output-dir",str(output),
        "--training-manifest",str(manifest),"--evaluation-role","validation"],cwd=root,text=True,capture_output=True)
    (artifacts/"evaluate_cli.log").write_text(evaluated.stdout+evaluated.stderr)
    assert evaluated.returncode==2,evaluated.stdout+evaluated.stderr
    report=json.loads((output/"native_comparator_evaluation.json").read_text())
    assert report["status"]=="incomplete" and report["declared_group_count"]==4
    assert report["binding"]["checkpoint_path"]==str(training_checkpoint_path(
        manifest,trained["selection"],trained["selection"]["best"]["checkpoint_path"]))
    assert all(row["status"]=="residual_unbound" and row["metrics"]["crps"] is None for row in report["groups"])
    assert all(torch.isfinite(torch.tensor(row["metrics"]["normalized_rmse"])) for row in report["groups"])
    receipt={"scope":"actual_tiny_tau_fit_and_strict_reload_without_residual_calibration",
        "training_cli_executed_this_attempt":not bool(reused),"train_cli_exit_this_attempt":train_exit,
        "reused_training_artifacts":reused,"evaluate_cli_exit":evaluated.returncode,
        "training_manifest":str(manifest),"evaluation_report":report,
        "prior_optimizer_updates":trained["executed_steps"] if reused else 0,
        "additional_optimizer_updates":0 if reused else trained["executed_steps"],"source_provider_calls":0,
        "scientific_twin":"NOT_RUN","formal_qualification":False}
    Path(os.environ.get("NRNSIFT_TAU_EVIDENCE",str(tmp_path/"result.json"))).write_text(json.dumps(receipt,indent=2))
