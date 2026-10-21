"""Controlled26 calibration contracts using real numeric fits, never scientific data."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from Libs.tests.native_comparator_cli_integration import retained_group_metadata
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.evaluation import native_comparator_evaluation as evaluation
from Libs.utils.evaluation import native_predose_evaluation as predose
from Libs.utils.model.native_transolver_residual import CONTROLLED_LAW, residual_draws


class AffineFixtureNormalization:
    identity = "independent_statistical_law_fixture_not_scientific_normalization"
    center = torch.linspace(.1, .4, 26, dtype=torch.float64)
    scale = torch.linspace(.7, 2., 26, dtype=torch.float64)

    def normalize(self, values, mask):
        return torch.where(mask, (values-self.center.to(values))/self.scale.to(values), 0.)

    def inverse(self, values, mask):
        return torch.where(mask, values*self.scale.to(values)+self.center.to(values), 0.)


class StatisticalDataset:
    def __init__(self, declarations, payloads):
        self.split_name = "calibration"
        self.group_uids = [row["sample_uid"] for row in declarations]
        self.groups = []
        self.payloads = payloads
        self.target_reads = 0
        for row in declarations:
            teachers = [{"sample_uid": row["sample_uid"], "family_uid": row["family_uid"],
                "regimen_uid": "action:"+str(i), "split_name": "calibration", "source_holdout_role": "calibration",
                "calibration_reservation": row["calibration_reservation"]} for i in range(3)]
            self.groups.append((SimpleNamespace(native_records=[{"teacher": t} for t in teachers]), [0, 1, 2], None))

    def __len__(self):
        return len(self.group_uids)

    def __getitem__(self, index):
        self.target_reads += 1
        return {"native_joint_group": self.payloads[self.group_uids[index]]}


def selected_rows(case, role, reservation=None):
    assert role == "calibration"
    return copy.deepcopy(case.rows[reservation])


def bind_declarations(case, monkeypatch):
    monkeypatch.setattr(evaluation, "selected_group_definitions",
        lambda config, role, reservation=None: selected_rows(case, role, reservation))
    monkeypatch.setattr(evaluation, "group_definitions",
        lambda config, role_filter=None: copy.deepcopy(sum(case.rows.values(), [])))


@pytest.fixture(scope="module")
def fitted_law(tmp_path_factory):
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    normalizer = AffineFixtureNormalization()
    binding = {"model_identity": {"scope": "independent_statistical_law_fixture"},
               "checkpoint_sha256": "statistical_fixture_has_no_neural_checkpoint", "normalization_id": normalizer.identity}
    rows = {r: [{"sample_uid": f"statistical:{r}:{i}", "role": "calibration", "calibration_reservation": r,
                "family_uid": "statistical_readout_law", "action_indices": [0, 1, 2]} for i in range(2 if r != "threshold" else 1)]
            for r in ("fit", "check", "threshold")}
    case = SimpleNamespace(config=config, binding=binding, rows=rows, normalizer=normalizer)
    q = torch.full((1, 32), 1/32, dtype=torch.float64)
    means = torch.zeros(32, 3, 4, 6, 6, 26, dtype=torch.float64)
    mask = torch.ones_like(means[0], dtype=torch.bool)
    profile = config["native_observable_bridge"]["quantity_profile"]
    grid = {"time_hours": [0., 1., 2., 3.], "family_uid": "statistical_readout_law",
            "species_id": [row["quantity_id"] for row in profile["quantities"]]}
    payloads = {}
    generating_scales = torch.linspace(.3, 1.2, 26, dtype=torch.float64)
    for reservation in ("fit", "check"):
        for declaration in rows[reservation]:
            uid = declaration["sample_uid"]
            # Independent complete synthetic readouts, not copied train/eval data.
            target = residual_draws(normalizer.normalize(means, mask), generating_scales, grid["time_hours"], 672.,
                torch.tensor([0]), episode_uid=uid, residual_seed=8041, draw_ids=[0], law=CONTROLLED_LAW)[0]
            target = normalizer.inverse(target, mask)
            target[:, 0] = means[0, :, 0]
            inputs = SimpleNamespace(group_shape=(1, 32, 3), weights=q, sample_uids=(uid,),
                normalization_id=normalizer.identity, validity=mask[None].expand(32, *mask.shape).reshape(96, *mask.shape[1:]), grid=grid,
                posterior=SimpleNamespace(support=SimpleNamespace(initial_field=means[:, 0, 0][None].clone())))
            payloads[uid] = {"inputs": inputs, "target": target[None], "mask": mask[None], "means": means[None]}
    case.fit, case.check = StatisticalDataset(rows["fit"], payloads), StatisticalDataset(rows["check"], payloads)
    case.inputs = payloads[rows["fit"][0]["sample_uid"]]["inputs"]
    case.means = means
    retained = os.environ.get("NRNSIFT_STATISTICAL_RESIDUAL_ARTIFACT")
    if retained:
        case.path = Path(retained)
        case.artifact = json.loads(case.path.read_text())
        return case
    parameter = torch.nn.Parameter(torch.zeros(()))
    belief = SimpleNamespace(field_normalization=normalizer, parameters=lambda: iter([parameter]))
    destination = tmp_path_factory.mktemp("real_numeric_native26_fit")
    with patch.object(evaluation, "selected_group_definitions", lambda config, role, reservation=None: selected_rows(case, role, reservation)), \
            patch.object(evaluation, "group_definitions", lambda config, role_filter=None: copy.deepcopy(sum(rows.values(), []))), \
            patch.object(evaluation, "prepare_comparator_group", lambda group, *args: (group["inputs"], group["target"], group["mask"], torch.ones(1, 3))), \
            patch.object(evaluation, "comparator_mean", lambda model, method, inputs, *args: (means.mean(0)[None], means[None])):
        case.artifact = evaluation.calibrate_native_comparator(config, "mechanistic_residual", None,
            belief, binding, case.fit, case.check, destination)
    assert case.artifact["status"] == "FIT_COMPLETE", case.artifact
    assert case.artifact["calibration_check_complete"]
    case.path = destination/"native_comparator_residual.json"
    return case


def test_real_numeric_fit_and_independent_diagnostics(fitted_law, monkeypatch):
    case = fitted_law; bind_declarations(case, monkeypatch)
    artifact = evaluation.load_native_comparator_residual(case.path, case.config, case.binding)
    assert case.fit.target_reads == case.check.target_reads == 2
    assert artifact["cost"]["optimizer_closure_evaluations"] > 0
    assert artifact["counts"]["eligible_group_count"] == 2
    assert all(row["scientific_check_pass"] is None for row in artifact["checks"])
    assert artifact["scientific_twin"] == "NOT_RUN" and artifact["formal_qualification"] is False


@pytest.mark.parametrize("mutation", ["dataset_role", "teacher_role", "reservation", "incomplete", "overlap", "threshold", "actions"])
def test_complete_partition_admission_precedes_any_target_read(fitted_law, monkeypatch, tmp_path, mutation):
    case = copy.deepcopy(fitted_law); bind_declarations(case, monkeypatch)
    case.fit.target_reads = case.check.target_reads = 0
    if mutation == "dataset_role": case.check.split_name = "validation"
    if mutation == "teacher_role": case.check.groups[0][0].native_records[0]["teacher"]["source_holdout_role"] = "development"
    if mutation == "reservation": case.check.groups[0][0].native_records[0]["teacher"]["calibration_reservation"] = "fit"
    if mutation == "incomplete": case.fit.group_uids.pop()
    if mutation == "overlap": case.rows["check"][0]["sample_uid"] = case.rows["fit"][0]["sample_uid"]
    if mutation == "threshold": case.rows["fit"][0]["sample_uid"] = case.rows["threshold"][0]["sample_uid"]
    if mutation == "actions": case.check.groups[0][1].pop()
    with pytest.raises(ValueError):
        evaluation.calibrate_native_comparator(case.config, "mechanistic_residual", None,
            None, case.binding, case.fit, case.check, tmp_path)
    assert case.fit.target_reads == case.check.target_reads == 0
    assert not (tmp_path/"native_comparator_residual.json").exists()


@pytest.mark.parametrize("mutation", ["binding", "normalizer", "policy", "clock", "fit_group", "check_group", "threshold", "scale_shape", "nonpositive_scale", "nan_scale", "hessian_shape", "hessian_finite", "hessian_threshold", "gradient", "information", "objective", "check_incomplete", "scientific_pass", "check_diagnostics", "information_groups", "pit_none", "covariance_none", "pit_nonfinite", "pit_zero_nonnull", "pit_identity", "covariance_kind", "covariance_nonfinite", "covariance_zero_nonnull", "covariance_counts", "information_totals", "episode_eligibility", "group_episode_counts"])
def test_loader_rejects_mutated_real_fit(fitted_law, monkeypatch, tmp_path, mutation):
    case = fitted_law; bind_declarations(case, monkeypatch)
    value = copy.deepcopy(case.artifact)
    if mutation == "binding": value["binding"]["checkpoint_sha256"] = "different"
    if mutation == "normalizer": value["binding"]["normalization_id"] = "different"
    if mutation == "policy": value["numerical"]["max_iterations"] += 1
    if mutation == "clock": value["reference_time_hours"] = 24.
    if mutation == "fit_group": value["fit_group_uids"].pop()
    if mutation == "check_group": value["check_group_uids"].append(value["check_group_uids"][0])
    if mutation == "threshold": value["fit_group_uids"][0] = case.rows["threshold"][0]["sample_uid"]
    if mutation == "scale_shape": value["scales"].pop()
    if mutation == "nonpositive_scale": value["scales"][0] = 0.
    if mutation == "nan_scale": value["scales"][0] = float("nan")
    if mutation == "hessian_shape": value["diagnostics"]["hessian_eigenvalues"] = []
    if mutation == "hessian_finite": value["diagnostics"]["hessian_eigenvalues"][0] = float("inf")
    if mutation == "hessian_threshold": value["diagnostics"]["hessian_eigenvalues"][0] = 0.
    if mutation == "gradient": value["diagnostics"]["gradient_max_abs"] = 1.
    if mutation == "information": value["counts"]["per_species_future_coordinate_count"][0] = 0
    if mutation == "objective": value["diagnostics"]["objective"] = float("nan")
    if mutation == "check_incomplete": value["checks"].pop()
    if mutation == "scientific_pass": value["checks"][0]["scientific_check_pass"] = True
    if mutation == "check_diagnostics": del value["checks"][0]["pit"]
    if mutation == "information_groups": value["counts"]["groups"][0]["group_uid"] = case.rows["check"][0]["sample_uid"]
    if mutation == "pit_none": value["checks"][0]["pit"] = None
    if mutation == "covariance_none": value["checks"][0]["covariance"] = None
    if mutation == "pit_nonfinite": value["checks"][0]["pit"]["future"]["mean"] = float("nan")
    if mutation == "pit_zero_nonnull": value["checks"][0]["pit"]["future"]["count"] = 0
    if mutation == "pit_identity": value["checks"][0]["pit"]["by_species"][0]["species_id"] = "undeclared_quantity"
    if mutation == "covariance_kind": del value["checks"][0]["covariance"]["graph_edges"]
    if mutation == "covariance_nonfinite": value["checks"][0]["covariance"]["graph_edges"]["centered_product_mean"] = float("inf")
    if mutation == "covariance_zero_nonnull": value["checks"][0]["covariance"]["graph_edges"]["count"] = 0
    if mutation == "covariance_counts": value["checks"][0]["covariance"]["graph_edges"]["missing_stratum_count"] += 1
    if mutation == "information_totals": value["counts"]["future_coordinate_count"] += 1
    if mutation == "episode_eligibility": value["counts"]["episodes"][0]["eligible"] = False
    if mutation == "group_episode_counts": value["counts"]["groups"][0]["declared_episode_count"] += 1
    path = tmp_path/"mutated.json"; path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        evaluation.load_native_comparator_residual(path, case.config, case.binding)


def test_shared_draws_keep_full_components_actions_absolute_stream_and_t0(fitted_law):
    case = fitted_law
    inputs = copy.deepcopy(case.inputs)
    means = case.means.clone()+torch.arange(32, dtype=torch.float64)[:, None, None, None, None, None]/100
    reference_means = means.clone()
    means[:, :, 0] += 17.125  # a learned mean has no guaranteed t0 identity
    inputs.posterior.support.initial_field = torch.arange(32, dtype=torch.float64)[None, :, None, None, None].expand(1, 32, 6, 6, 26)/10
    inputs.validity = inputs.validity.clone()
    inputs.validity[:, 0, 0, 0, -1] = False
    initial_before = inputs.posterior.support.initial_field.clone()
    mask_before = inputs.validity.clone()
    untouched = means.clone()
    samples = evaluation.draw_native_comparator_residual(case.config, inputs, means, case.normalizer, case.artifact)
    again = evaluation.draw_native_comparator_residual(case.config, inputs, means, case.normalizer, case.artifact)
    assert samples.shape == (32, 2, 3, 4, 6, 6, 26)
    assert torch.equal(samples, again) and torch.equal(means, untouched)
    assert torch.equal(inputs.posterior.support.initial_field, initial_before) and torch.equal(inputs.validity, mask_before)
    mask = inputs.validity.reshape(1, 32, 3, 4, 6, 6, 26)[0]
    initial = initial_before[0, :, None].expand(32, 3, 6, 6, 26)
    initial = torch.where(mask[:, :, 0], initial, 0.)
    assert torch.equal(samples[:, :, :, 0], initial[:, None].expand_as(samples[:, :, :, 0]))
    assert not torch.equal(samples[:, 0, :, 0], means[:, :, 0])
    future_reference = evaluation.draw_native_comparator_residual(case.config, inputs, reference_means, case.normalizer, case.artifact)
    assert torch.equal(samples[:, :, :, 1:], future_reference[:, :, :, 1:])
    assert (samples[:, :, :, 1:] < 0).any()  # unbounded readouts were not clipped to physical states
    normalized = case.normalizer.normalize(means, mask)
    indices = torch.arange(32).repeat_interleave(2)
    ids = list(range(64))
    pieces = [residual_draws(normalized, case.artifact["scales"], inputs.grid["time_hours"], 672., indices[left:right],
        episode_uid=inputs.sample_uids[0], residual_seed=case.config["native_controlled_comparators"]["residual_calibration"]["residual_seed"],
        draw_ids=ids[left:right], law=CONTROLLED_LAW) for left, right in ((0, 17), (17, 64))]
    expected = case.normalizer.inverse(torch.cat(pieces), mask[indices])
    expected[:, :, 0] = initial[indices]
    assert torch.equal(samples.reshape_as(expected), expected)
    with pytest.raises(ValueError, match="all32"):
        evaluation.draw_native_comparator_residual(case.config, inputs, means[:-1], case.normalizer, case.artifact)


@pytest.mark.parametrize("calibrated", [False, True])
def test_calibrated_query_fixes_public_loss_before_private_target_boundary(fitted_law, monkeypatch, tmp_path, calibrated):
    case = fitted_law; bind_declarations(case, monkeypatch)
    inputs = copy.deepcopy(case.inputs)
    inputs.sample_uids = ("independent-query-fixture",)
    inputs.regimen_uids = ("a", "b", "c")
    inputs.programs = {"records": [[[{} for _ in range(3)] for _ in range(32)]]}
    events = []
    def predict(*args):
        events.append("predict")
        return case.means.reshape(96, 4, 6, 6, 26)
    class PublicLoss:
        def __init__(self, *args): pass
        def __call__(self, values, *args):
            assert events == ["predict"]
            assert values.shape == (32, 2 if calibrated else 1, 3, 4, 6, 6, 26)
            if calibrated: assert values[:, :, :, 1:].std() > 0
            events.append("public_loss")
            return {"loss": values.square().mean(dim=(-4, -3, -2, -1))}
    class TargetBoundary:
        group_uids = ["independent-query-fixture"]
        def __getitem__(self, index):
            assert events == ["predict", "public_loss"]
            events.append("private_target_boundary")
            raise RuntimeError("intentional_private_target_boundary_no_payload")
    model = SimpleNamespace(prediction_kind="offline_one_step_mechanistic_residual_conditional_mean", predict_native=predict)
    monkeypatch.setattr(predose, "predose_source_admission", lambda *args: {"scope": "admitted_controlled_source" if calibrated else "artificial_software_fixture"})
    monkeypatch.setattr(predose, "load_family_registries", lambda *args: {})
    monkeypatch.setattr(predose, "load_native_prediction_inputs", lambda *args: inputs)
    selector = inputs.grid["species_id"].index("calcium_concentration")
    monkeypatch.setattr(predose, "native_predose_channel", lambda *args: SimpleNamespace(readout=SimpleNamespace(
        species_ids=inputs.grid["species_id"], selector=((0, 0, selector),))))
    monkeypatch.setattr(predose, "compiled_action_burden", lambda *args: {"status": "defined", "normalized_burden": 0.})
    monkeypatch.setattr(predose, "NativePublicCNSLoss", PublicLoss)
    report = predose.evaluate_native_predose_queries(case.config, model, SimpleNamespace(field_normalization=case.normalizer),
        case.binding, TargetBoundary(), {"independent-query-fixture": {}}, {"reference_id": "unit-fixture"}, tmp_path,
        residual_path=case.path if calibrated else None)
    assert events == ["predict", "public_loss", "private_target_boundary"]
    assert report["groups"][0]["failure"]["message"] == "intentional_private_target_boundary_no_payload"
    if calibrated:
        assert report["residual_calibration"]["scientific_check_pass"] is None
        assert report["residual_calibration"]["physical_state_certification"] == "not_certified_unclipped_teacher_readout"
    else:
        assert "residual_calibration" not in report
        assert report["groups"][0]["conditional_law"] == "one_deterministic_path_per_tuple_no_residual_likelihood_qualification"


def test_retained_checkpoint_cli_rejects_missing_calibration_before_arrays(tmp_path):
    raw = os.environ.get("NRNSIFT_RHS_FIXTURE")
    selected = os.environ.get("NRNSIFT_HYBRID_FIT")
    if not raw or not selected:
        pytest.skip("Requires retained cpu04 artificial corpus and selected hybrid checkpoint")
    fixture, fit = Path(raw), Path(selected)
    guard = tmp_path/"guard"; guard.mkdir()
    guard_log = tmp_path/"target_access.json"
    script = '''import atexit,builtins,io,json,os
from pathlib import Path
root=os.environ["NRNSIFT_TEST_TARGET_ROOT"]
log=Path(os.environ["NRNSIFT_TEST_GUARD_LOG"])
reads=[]
original_open=builtins.open
original_io_open=io.open
def wrap(function):
 def checked(path,*args,**kwargs):
  if isinstance(path,(str,bytes,os.PathLike)) and os.fsdecode(path).startswith(root+os.sep):
   reads.append(os.fsdecode(path));raise RuntimeError("calibration_target_access_guard")
  return function(path,*args,**kwargs)
 return checked
builtins.open=wrap(original_open)
io.open=wrap(original_io_open)
atexit.register(lambda:log.write_text(json.dumps({"guard_installed":True,"target_access_attempts":reads})))
'''
    (guard/"sitecustomize.py").write_text(script)
    env = dict(os.environ, PYTHONPATH=str(guard)+os.pathsep+"Libs", NRNSIFT_TEST_TARGET_ROOT=str(fixture/"sources"),
               NRNSIFT_TEST_GUARD_LOG=str(guard_log))
    probe = subprocess.run([sys.executable, "-c", "open("+repr(str(fixture/"sources/guard-probe"))+")"],
                           env=env, capture_output=True, text=True)
    assert probe.returncode != 0 and "calibration_target_access_guard" in probe.stderr
    assert len(json.loads(guard_log.read_text())["target_access_attempts"]) == 1
    command = [sys.executable, "Libs/exp/baselines/run_external_baseline_bridge.py", "--protocol-scope", "manuscript-native",
        "--baseline-name", "mechanistic_residual", "--command", "calibrate_baseline", "--config", str(fit.parent/"hybrid_software_config.yaml"),
        "--conditioning-config", str(fixture/"config.yaml"), "--conditioning-manifest", str(fixture/"native_fit/manifest.json"),
        "--training-manifest", str(fit/"native_comparator_training_manifest.json"), "--output-dir", str(tmp_path/"calibration"), "--device", "cpu"]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    (tmp_path/"calibrate_cli.log").write_text(result.stdout+result.stderr)
    (tmp_path/"command.json").write_text(json.dumps({"argv": command, "returncode": result.returncode,
        "scope": "retained_checkpoint_missing_independent_calibration_no_new_fit"}, indent=2))
    assert result.returncode != 0 and "Invalid or missing declared independent calibration-fit groups" in result.stderr
    assert json.loads(guard_log.read_text()) == {"guard_installed": True, "target_access_attempts": []}
    assert not (tmp_path/"calibration/native_comparator_residual.json").exists()


def test_existing_mean_evaluator_reuses_validated_draw_law(fitted_law, monkeypatch, tmp_path):
    case = fitted_law; bind_declarations(case, monkeypatch)
    inputs = copy.deepcopy(case.inputs)
    uid = "statistical:independent-evaluation"
    inputs.sample_uids = (uid,)
    inputs.regimen_uids = ("a", "b", "c")
    mask = torch.ones_like(case.means[0], dtype=torch.bool)
    target = residual_draws(case.normalizer.normalize(case.means, mask), torch.linspace(.3, 1.2, 26),
        inputs.grid["time_hours"], 672., torch.tensor([0]), episode_uid=uid,
        residual_seed=8041, draw_ids=[0], law=CONTROLLED_LAW)[0]
    target = case.normalizer.inverse(target, mask)
    target[:, 0] = case.means[0, :, 0]
    declaration = {"sample_uid": uid, "partition_uid": "independent-statistical-evaluation",
        "family_uid": "statistical_readout_law", "calibration_reservation": None}
    dataset = StatisticalDataset([declaration], {uid: {}})
    dataset.split_name = "validation"
    monkeypatch.setattr(evaluation, "prepare_comparator_group", lambda *args:
        (inputs, target[None], mask[None], torch.ones(1, 3)))
    monkeypatch.setattr(evaluation, "comparator_mean", lambda *args, **kwargs:
        (case.means.mean(0)[None], case.means[None]))
    parameter = torch.nn.Parameter(torch.zeros(()))
    belief = SimpleNamespace(field_normalization=case.normalizer, parameters=lambda: iter([parameter]))
    report = evaluation.evaluate_native_comparator(case.config, "mechanistic_residual", None, belief,
        case.binding, dataset, {uid: declaration}, tmp_path, case.path)
    assert report["status"] == "complete" and report["defined_group_count"] == 1
    assert report["groups"][0]["metrics"]["normalized_rmse"] >= 0
    assert report["residual_calibration"]["scientific_check_pass"] is None
    assert report["residual_calibration"]["physical_state_certification"] == "not_certified_unclipped_teacher_readout"


def test_retained_adapter_metadata_uses_supplied_roles_and_actions_only(tmp_path, monkeypatch):
    location = os.environ.get("NRNSIFT_RHS_FIXTURE")
    if not location:
        pytest.skip("Requires retained public registry metadata")
    registry = json.loads((Path(location)/"electrophysiology_conductance_world.json").read_text())
    teachers = [row for row in registry["teachers"] if row["split_name"] == "train"]
    declaration = {"sample_uid": teachers[0]["sample_uid"], "family_uid": teachers[0]["family_uid"],
        "role": teachers[0]["split_name"], "calibration_reservation": teachers[0]["calibration_reservation"]}
    actions = [row["regimen_uid"] for row in teachers]
    path = tmp_path/"group_0_conditioning.json"
    path.write_text(json.dumps({"particle_programs": {"regimen_uids": actions}}))
    read_text = Path.read_text
    reads = []
    def public_only(current, *args, **kwargs):
        assert current == path, "Metadata adapter attempted another source/producer/target read"
        reads.append(str(current))
        return read_text(current, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", public_only)
    role, groups = retained_group_metadata(tmp_path, [(0, declaration)])
    assert role == "train" and reads == [str(path)]
    child, positions, _ = groups[0]
    exposed = [child.native_records[i]["teacher"] for i in positions]
    assert [row["regimen_uid"] for row in exposed] == actions
    assert all(row["sample_uid"] == declaration["sample_uid"] and row["split_name"] == declaration["role"]
        and row["source_holdout_role"] == declaration["role"] and row["calibration_reservation"] is None for row in exposed)
    missing = dict(declaration); del missing["role"]
    with pytest.raises(KeyError): retained_group_metadata(tmp_path, [(0, missing)])


def test_loader_retains_legal_missing_pit_quantities_and_covariance_strata(fitted_law, monkeypatch, tmp_path):
    case = fitted_law; bind_declarations(case, monkeypatch)
    artifact = copy.deepcopy(case.artifact)
    check = artifact["checks"][0]
    missing = check["pit"]["by_species"][0]
    check["pit"]["future"]["count"] -= missing["count"]
    missing.update(count=0, mean=None)
    for summary in check["covariance"].values():
        summary.update(measured_stratum_count=0, missing_stratum_count=summary["declared_stratum_count"],
            count=0, centered_product_mean=None, total_covariance_mean=None)
    path = tmp_path/"missing-diagnostics.json"; path.write_text(json.dumps(artifact))
    restored = evaluation.load_native_comparator_residual(path, case.config, case.binding)
    assert restored["checks"][0] == check and check["scientific_check_pass"] is None
