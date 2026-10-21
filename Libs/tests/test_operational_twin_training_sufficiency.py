from __future__ import annotations

from copy import deepcopy
import json
import os
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError
import pytest
import torch
import yaml

from Libs.utils.evaluation.operational_twin_training_sufficiency import (
    CLOSED,
    SCHEMA_PATH,
    SCHEMA_VERSION,
    STATES,
    _tensor_equal,
    aggregate_methods,
    classify_optimization,
    diagnose_assignment,
    independent_pair_metrics,
    independent_statistics,
    late_diagnostic,
    safe_path,
)
from Libs.utils.evaluation.operational_twin_development_evidence import (
    build_development_comparisons,
)
from Libs.utils.evaluation.operational_twin_training_protocol import REQUIRED_METHOD_IDS
from Libs.utils.exp_utils import load_config_stack

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def policy():
    return load_config_stack()["operational_twin_training_sufficiency"]


def diagnostic(change=0.0, noise=0.0):
    return {"normalized_late_change": change, "normalized_residual_iqr": noise}


def test_plateau_does_not_require_positive_utility(policy):
    # No regret/action/trajectory utility argument enters this classifier.
    assert (
        classify_optimization(diagnostic(), diagnostic(), policy["curves"])[0]
        == STATES[0]
    )


@pytest.mark.parametrize(
    "train,dev,expected",
    [
        (-0.1, -0.1, "CLEARLY_UNDERTRAINED"),
        (-0.1, 0.2, "INDETERMINATE_WITHIN_FROZEN_BUDGET"),
        (0, 0.1, "INDETERMINATE_WITHIN_FROZEN_BUDGET"),
        (-0.03, -0.03, "INDETERMINATE_WITHIN_FROZEN_BUDGET"),
    ],
)
def test_joint_improvement_overfit_and_ambiguity(policy, train, dev, expected):
    assert (
        classify_optimization(diagnostic(train), diagnostic(dev), policy["curves"])[0]
        == expected
    )


def test_oscillation_and_numerical_failure_are_distinct(policy):
    assert (
        classify_optimization(diagnostic(0.2, 0.4), diagnostic(), policy["curves"])[0]
        == STATES[2]
    )
    assert (
        classify_optimization(
            diagnostic(), diagnostic(), policy["curves"], numerical_valid=False
        )[0]
        == STATES[3]
    )
    assert classify_optimization(None, None, policy["curves"])[0] == STATES[4]
    assert (
        classify_optimization(None, None, policy["curves"], no_fit=True)[0] == STATES[0]
    )
    assert (
        classify_optimization(
            None, None, policy["curves"], no_fit=True, mandatory_complete=False
        )[0]
        == STATES[4]
    )


def test_negative_nll_and_scaling(policy):
    values = [-i / 20 for i in range(100)]
    d = late_diagnostic(values, policy["curves"])
    e = late_diagnostic([v * 7 for v in values], policy["curves"])
    assert d["normalized_late_change"] == pytest.approx(e["normalized_late_change"])
    assert d["median_pairwise_slope"] == pytest.approx(-0.05)
    with pytest.raises(ValueError):
        late_diagnostic([1, float("nan")] * 50, policy["curves"])
    with pytest.raises(ValueError):
        late_diagnostic([1.0] * 10, policy["curves"])


def test_alias_mechanism_is_observable_without_training():
    model = torch.nn.Linear(2, 1)
    payload = model.state_dict()
    frozen = {k: v.clone() for k, v in payload.items()}
    with torch.no_grad():
        model.weight.add_(1.0)
    assert _tensor_equal(payload, model.state_dict())
    assert not _tensor_equal(frozen, model.state_dict())


def assignment_fixture(policy):
    cfg = load_config_stack()["operational_twin_training_protocol"]
    method = deepcopy(cfg["methods"][1])
    method["budget"]["epochs"] = 100
    from Libs.utils.evaluation.operational_twin_training_protocol import (
        resolve_method_execution_contract,
    )

    assignment = {
        "assignment_uid": "linear_seed",
        "method_id": "linear_or_delta",
        "seed": 360101,
        "runtime_contract": resolve_method_execution_contract("linear_or_delta"),
    }
    rows = [
        {
            "epoch": i + 1,
            "training_nll": 1.0,
            "reconstruction_rmse": 1.0,
            "loss_contract": "masked_mse_linear_delta",
            "family_metrics": {
                f: {}
                for f in (
                    "calcium_synaptic_world",
                    "electrophysiology_conductance_world",
                    "molecular_pharmacological_causal_world",
                )
            },
        }
        for i in range(100)
    ]
    status = {"epochs_completed": 100, "steps_completed": 200}
    return assignment, method, status, rows


@pytest.mark.parametrize(
    "mutation", ["epochs", "loss", "family", "missing", "nonfinite"]
)
def test_malformed_mandatory_curves_fail_closed(policy, mutation):
    a, m, s, rows = assignment_fixture(policy)
    if mutation == "epochs":
        rows[2]["epoch"] = rows[1]["epoch"]
    elif mutation == "loss":
        rows[0]["loss_contract"] = "masked_gaussian_nll"
    elif mutation == "family":
        rows[0]["family_metrics"] = {}
    elif mutation == "missing":
        rows[1].pop("training_nll")
    elif mutation == "nonfinite":
        rows[-1]["training_nll"] = float("inf")
    out = diagnose_assignment(a, m, s, rows, {}, policy)
    assert out["classification"] == (
        STATES[3] if mutation == "nonfinite" else STATES[4]
    )


def test_full_budget_is_not_sufficiency(policy):
    a, m, s, rows = assignment_fixture(policy)
    for i, r in enumerate(rows):
        r["training_nll"] = 100 - i
        r["reconstruction_rmse"] = (100 - i) ** 0.5
    out = diagnose_assignment(a, m, s, rows, {}, policy)
    assert out["budget"]["full_budget_completed"] is True
    assert out["classification"] == STATES[1]


def test_method_aggregation_retains_exact_all_seed_matrix():
    seeds = [360101, 360211, 360307]
    rows = [
        {
            "method_id": m,
            "seed": s,
            "classification": STATES[0],
            "identity_blockers": [],
        }
        for m in REQUIRED_METHOD_IDS
        for s in seeds
    ]
    rows[0]["identity_blockers"] = ["selection_defect"]
    out = aggregate_methods(rows, REQUIRED_METHOD_IDS, seeds)
    assert (
        out[0]["classification"] == STATES[0] and out[0]["identity_blocker_count"] == 1
    )
    for invalid in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError):
            aggregate_methods(invalid, REQUIRED_METHOD_IDS, seeds)


@pytest.mark.parametrize(
    "path",
    [
        "../outside",
        "/etc/passwd",
        "Input/private/data.json",
        "Input/protected_holdout/x",
        "Input/six_context/results",
        "Input/processed/sealed/payload",
        "Input/benchmark/test_manifest.json",
        "Input/benchmark/ood_manifest.json",
    ],
)
def test_forbidden_paths_denied_before_read(tmp_path, path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, path)


def test_symlink_denied_before_read(tmp_path):
    (tmp_path / "linked").symlink_to("/etc")
    with pytest.raises(ValueError):
        safe_path(tmp_path, "linked/hosts")


def pair_fixture():
    worlds = [
        {
            "world_uid": u,
            "method_id": "method",
            "seed": 1,
            "family_uid": "family",
            "reference_utility": truth,
            "predicted_utility": prediction,
            "trajectory_rmse": 0.4,
        }
        for u, truth, prediction in [("a", 2.0, 3.0), ("b", 3.0, 1.0)]
    ]
    pairs = [
        {
            "counterfactual_pair_uid": "p",
            "family_uid": "family",
            "approved_pair": True,
            "member_world_uids": ["a", "b"],
        }
    ]
    return worlds, pairs


def test_independent_metric_ties_undefined_and_missing():
    worlds, pairs = pair_fixture()
    r = independent_pair_metrics(worlds, pairs)[0]
    assert (
        r["intervention_ranking_regret"] == 1
        and r["action_contrast_error"] == 3
        and r["effect_direction_error"] == 1
    )
    worlds[1]["reference_utility"] = 2
    assert independent_pair_metrics(worlds, pairs)[0]["effect_direction_error"] is None
    with pytest.raises(ValueError):
        independent_pair_metrics(worlds[:1], pairs)
    worlds[1]["predicted_utility"] = float("nan")
    with pytest.raises(ValueError):
        independent_pair_metrics(worlds, pairs)


def test_independent_bootstrap_permutation_holm_matches_existing():
    families = [
        "calcium_synaptic_world",
        "electrophysiology_conductance_world",
        "molecular_pharmacological_causal_world",
    ]
    from Libs.utils.evaluation.operational_twin_development_evidence import (
        FULL_METHOD_ID,
        PRIMARY_BASELINE_METHODS,
        LOWER_IS_BETTER_METRICS,
    )

    rows = []
    for method_i, method in enumerate([FULL_METHOD_ID, *PRIMARY_BASELINE_METHODS]):
        for seed in [1, 2, 3]:
            for index in range(6):
                rows.append(
                    {
                        "method_id": method,
                        "seed": seed,
                        "counterfactual_pair_uid": str(index),
                        "family_uid": families[index // 2],
                        **{
                            m: (method_i + 0.1 * index + 0.01 * seed)
                            for m in LOWER_IS_BETTER_METRICS
                        },
                    }
                )
    p = {
        "analysis_seed": 17,
        "bootstrap": {"resamples": 200, "confidence_level": 0.95},
        "conditional_permutation": {"permutations": 200},
    }
    independent = independent_statistics(rows, [1, 2, 3], p)
    canonical = build_development_comparisons(rows, p)["comparisons"]
    for a, b in zip(independent, canonical):
        assert a["estimate"] == pytest.approx(
            b["family_stratified_bootstrap_ci95"]["estimate"]
        )
        assert a["lower"] == pytest.approx(
            b["family_stratified_bootstrap_ci95"]["lower"]
        )
        assert a["upper"] == pytest.approx(
            b["family_stratified_bootstrap_ci95"]["upper"]
        )
        assert a["holm_adjusted_p_value"] == b["holm_adjusted_p_value"]
    with pytest.raises(KeyError):
        independent_statistics(rows[:-1], [1, 2, 3], p)


def test_schema_and_real_stack_reject_claim_promotion(policy):
    schema = json.loads((ROOT / SCHEMA_PATH).read_text())
    Draft202012Validator.check_schema(schema)
    assert (
        load_config_stack()["schemas"]["operational_twin_training_sufficiency_schema"]
        == SCHEMA_PATH
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "decision",
        "evidence_class": "controlled_world_non_empirical",
        "policy_sha256": "0" * 64,
        "audit_execution_sha256": "0" * 64,
        "boundary": dict(CLOSED),
        "terminal": "OPERATIONAL_TWIN_TRAINING_EVIDENCE_INVALID",
        "assignment_count": 15,
        "classification_counts": {STATES[0]: 15},
        "identity_invalid_assignment_count": 1,
        "independent_review_status": "PENDING_NOT_SELF_APPROVED",
        "scientific_twin_result": "NOT_RUN",
        "continuation_authorized": False,
        "presealed_freeze_created": False,
        "next_action": "review",
        "irreducible_boundary": "identity invalid",
    }
    validator = Draft202012Validator(schema)
    validator.validate(artifact)
    artifact["boundary"]["sufficient_training"] = True
    with pytest.raises(ValidationError):
        validator.validate(artifact)


@pytest.mark.parametrize(
    "role,expected_nll",
    [("development", 0.125 + __import__("math").log(2)), ("calibration", 0.5)],
)
def test_replay_calibration_is_role_specific(monkeypatch, role, expected_nll):
    import Libs.utils.evaluation.operational_twin_training_sufficiency as audit

    class Model(torch.nn.Module):
        def forward(self, inputs, input_mask, action, family):
            return {
                "predictive_mean": torch.ones_like(inputs),
                "predictive_scale": torch.ones_like(inputs) * 2,
            }

    monkeypatch.setattr(audit, "build_method_model", lambda *args: Model())
    checkpoint = {
        "feature_names": ["f"],
        "model_config": {},
        "model_state_dict": {},
        "normalization_mean": torch.zeros(1),
        "normalization_std": torch.ones(1),
        "calibration_scale": torch.ones(3) * 0.5,
        "method_protocol": {"budget": {"batch_size": 1}},
    }
    item = {
        "world_uid": "world",
        "family_index": 0,
        "input_observation": torch.zeros(2, 1),
        "input_mask": torch.tensor([[True], [False]]),
        "target_observation": torch.zeros(2, 1),
        "target_mask": torch.tensor([[False], [True]]),
        "action_encoding": torch.zeros(1),
        "no_action_encoding": torch.zeros(1),
    }
    archived = [
        {
            "world_uid": "world",
            "role": role,
            "trajectory_rmse": 1.0,
            "gaussian_nll": expected_nll,
            "coverage_95": 1.0,
        }
    ]
    assignment = {"method_id": audit.FULL_METHOD_ID}
    tolerance = {"relative_tolerance": 1e-5, "absolute_tolerance": 1e-6}
    result = audit.replay_checkpoint(
        checkpoint, [item], assignment, archived, tolerance, role
    )
    assert result["archived_metric_mismatches"] == []
    assert result["calibration_applied"] is (role == "calibration")
    with pytest.raises(ValueError):
        audit.replay_checkpoint(
            checkpoint, [item], assignment, archived, tolerance, "test"
        )


@pytest.mark.parametrize(
    "steps,label,expected", [([1, 7, 10], 10, False), ([1, 7, 10], 8, True)]
)
def test_optimizer_partial_counters_vs_post_label_updates(
    tmp_path, monkeypatch, policy, steps, label, expected
):
    import hashlib
    import Libs.utils.evaluation.operational_twin_training_sufficiency as audit

    method = "linear_or_delta"
    contract = audit.resolve_method_execution_contract(method)
    checkpoint = {
        "assignment_uid": "a",
        "seed": 360101,
        "epoch": 1,
        "step": label,
        "method_contract": contract,
        "method_protocol": {"budget": {"learning_rate": 0.1}},
        "model_state_dict": {"p": torch.ones(1)},
        "optimizer_state_dict": {
            "state": {i: {"step": torch.tensor(float(v))} for i, v in enumerate(steps)},
            "param_groups": [{"lr": 0.1}],
        },
        "dev_metrics": {"reconstruction_rmse": 0.2},
    }
    path = tmp_path / "best_dev.pt"
    torch.save(checkpoint, path)
    assignment = {"assignment_uid": "a", "seed": 360101, "method_id": method}
    ledger = {
        "checkpoint": {
            "path": "best_dev.pt",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    }
    monkeypatch.setattr(
        audit,
        "replay_checkpoint",
        lambda *args: {"reconstruction_rmse": 0.2, "archived_metric_mismatches": []},
    )
    identity, problems = audit.checkpoint_identity(
        tmp_path,
        assignment,
        ledger,
        [{"epoch": 1, "reconstruction_rmse": 0.2}],
        [],
        None,
        {},
        policy,
    )
    assert (
        "optimizer_state_disagrees_with_labeled_checkpoint_step" in problems
    ) is expected
    assert identity["selected_metric_reproduced"] is True


@pytest.mark.parametrize(
    "key,value",
    [
        ("late_fraction", 0),
        ("minimum_late_epochs", 2.5),
        ("absolute_scale_floor", -1),
        ("dev_mse_weighting", "batch_mean"),
    ],
)
def test_policy_rejects_invalid_or_wrong_semantics(policy, key, value):
    from Libs.utils.evaluation.operational_twin_training_sufficiency import (
        validate_policy,
    )

    altered = deepcopy(policy)
    altered["curves"][key] = value
    with pytest.raises(ValueError):
        validate_policy(altered)


@pytest.mark.parametrize(
    "key,value",
    [
        ("torch_intra_op_threads", 4),
        ("torch_inter_op_threads", True),
        ("deterministic_algorithms", False),
        ("fp32_precision", "tf32"),
        ("environment", {"OMP_NUM_THREADS": "1"}),
    ],
)
def test_cpu_execution_profile_rejects_unsupported_settings(policy, key, value):
    from Libs.utils.evaluation.operational_twin_training_sufficiency import (
        validate_cpu_audit_profile,
    )

    changed = deepcopy(policy["audit_cpu_execution"])
    changed[key] = value
    with pytest.raises(ValueError, match="CPU audit execution profile"):
        validate_cpu_audit_profile(changed)


def test_cpu_audit_rejects_environment_only_repair(policy):
    code = """
import os
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation.operational_twin_training_sufficiency import configure_cpu_audit_execution
policy = load_config_stack()["operational_twin_training_sufficiency"]
for key, value in policy["audit_cpu_execution"]["environment"].items():
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
configure_cpu_audit_execution(policy)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "fresh-interpreter CLI environment" in result.stderr


def test_ambient_threads_cannot_change_canonical_model_replay(tmp_path, policy):
    # Exercise the real model and audit checkpoint replay; no fitting or source
    # checkpoints enter this regression. The all-seed CLI check uses frozen data.
    script = tmp_path / "profile_replay.py"
    script.write_text("""
import json
import torch
from Libs.utils.exp_utils import load_config_stack
from Libs.models.networks.operational_twin import OperationalTwin
from Libs.models.modules.world_action_encoder import WorldActionEncoder
from Libs.utils.evaluation.operational_twin_training_sufficiency import (
    FULL_METHOD_ID, bootstrap_cpu_audit_cli, configure_cpu_audit_execution, replay_checkpoint)
policy = load_config_stack()["operational_twin_training_sufficiency"]
bootstrap_cpu_audit_cli(policy)
execution = configure_cpu_audit_execution(policy)
assert execution == configure_cpu_audit_execution(policy)
torch.manual_seed(310031)
config = {"hidden_dim": 48, "posterior_dim": 12, "depth": 2, "head_count": 4,
          "mechanism_token_count": 4, "dropout_rate": 0.0, "minimum_scale": 0.0001}
model = OperationalTwin(feature_count=13, action_dim=WorldActionEncoder.output_dim, config=config).eval()
checkpoint = {"feature_names": list(range(13)), "model_config": config,
              "model_state_dict": model.state_dict(), "normalization_mean": torch.zeros(13),
              "normalization_std": torch.ones(13), "method_protocol": {"budget": {"batch_size": 2}},
              "calibration_scale": torch.ones(3)}
dataset = []
archived = []
for index in range(3):
    mask = torch.zeros(8, 13, dtype=torch.bool)
    mask[:4, :5] = True
    dataset.append({"world_uid": str(index), "family_index": index,
                    "input_observation": torch.randn(8, 13), "input_mask": mask,
                    "target_observation": torch.randn(8, 13), "target_mask": ~mask,
                    "action_encoding": torch.randn(WorldActionEncoder.output_dim),
                    "no_action_encoding": torch.zeros(WorldActionEncoder.output_dim)})
    archived.append({"world_uid": str(index), "role": "development", "trajectory_rmse": 0.0,
                     "gaussian_nll": 0.0, "coverage_95": 0.0})
replay = replay_checkpoint(checkpoint, dataset, {"method_id": FULL_METHOD_ID}, archived,
                           policy["checkpoint_replay"])
print(json.dumps({"execution": execution, "replay": replay}, sort_keys=True, allow_nan=False))
""")
    records = []
    thread_environment_keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    for ambient_threads in ("1", "4"):
        env = {**os.environ, "PYTHONPATH": str(ROOT) + ":" + str(ROOT / "Libs")}
        for key, value in policy["audit_cpu_execution"]["environment"].items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        for key in thread_environment_keys:
            env[key] = ambient_threads
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        lines = [json.loads(line) for line in result.stdout.splitlines()]
        if ambient_threads == "1":
            assert len(lines) == 1
            record = lines[0]
        else:
            assert len(lines) == 2
            restart, record = lines
            assert restart["cpu_audit_environment_restart"] is True
            assert restart["effective_requested"] == policy["audit_cpu_execution"][
                "environment"
            ]
            assert all(
                restart["ambient"][key] == ambient_threads
                for key in thread_environment_keys
            )
        assert set(record) == {"execution", "replay"}
        records.append(record)
    assert records[0] == records[1]
    execution = records[0]["execution"]
    assert (
        execution["torch_intra_op_threads"] == execution["torch_inter_op_threads"] == 1
    )
    assert execution["environment"] == policy["audit_cpu_execution"]["environment"]
    assert records[0]["replay"]["archived_metric_mismatches"]
    schema = json.loads((ROOT / SCHEMA_PATH).read_text())
    validator = Draft202012Validator(
        {"$ref": "#/$defs/execution_profile", "$defs": schema["$defs"]}
    )
    validator.validate(execution)
    for key in ("torch_inter_op_threads", "numpy_build_config", "environment"):
        altered = deepcopy(execution)
        altered.pop(key)
        with pytest.raises(ValidationError):
            validator.validate(altered)
    execution["torch_intra_op_threads"] = 4
    with pytest.raises(ValidationError):
        validator.validate(execution)


def test_discrepancy_history_and_criterion_remain_git_bound(policy):
    from Libs.utils.evaluation.operational_twin_training_sufficiency import (
        historical_replay_evidence,
    )

    history = historical_replay_evidence(ROOT, policy)
    old = history["pre_profile_audit"]
    assert old["source"]["commit"] == "417623382420a51ef3701d427471270d7f827e15"
    assert len(old["archived_metric_mismatches"]) == 3
    assert {row["seed"] for row in old["archived_metric_mismatches"]} == {360211}
    assert {row["metric"] for row in old["archived_metric_mismatches"]} == {
        "gaussian_nll"
    }
    review = history["independent_ambient_thread_review"]
    assert review["source"]["commit"] == "2c6fb750a7cc79d37976310715d3b241040fc584"
    assert "electrophysiology0005/0015" in review["document"]
    assert "molecular0018" in review["document"]
    altered = deepcopy(policy)
    altered["checkpoint_replay"]["absolute_tolerance"] *= 2
    with pytest.raises(
        ValueError, match="original frozen sufficiency criteria changed"
    ):
        historical_replay_evidence(ROOT, altered)


# Native scope deliberately shares this audit entry point with the historical units.
from Libs.utils.evaluation.operational_twin_training_sufficiency import (
    NATIVE_CONFIG_PATH, build_native_audit, native_evidence_path,
    replay_native_selection, strict_json,
)
from Libs.utils.model.native_training_selection import NativeTrainingSelection


def native_selection(policy, losses, *, steps=None, elapsed=None):
    selection = NativeTrainingSelection(policy)
    for epoch, loss in enumerate(losses, 1):
        stop = selection.update(role="development", epoch=epoch,
            steps=(steps or list(range(1, len(losses) + 1)))[epoch - 1],
            training_loss=float(loss + 1), development_loss=float(loss),
            checkpoint_path=f"checkpoint_{epoch}.pt",
            elapsed_seconds=float((elapsed or list(range(1, len(losses) + 1)))[epoch - 1]))
        if stop:
            break
    return selection.payload()


@pytest.fixture
def scalar_native_policy():
    return {"role": "development", "max_epochs": 10, "minimum_epochs": 2,
            "minimum_steps": 2, "patience": 2, "min_delta": .001,
            "max_steps": 10, "max_seconds": 100}


def test_native_best_last_and_budget_precedence(scalar_native_policy):
    payload = native_selection(scalar_native_policy, [3., 2., 2., 2.])
    result = replay_native_selection(payload, scalar_native_policy)
    assert result["terminal"] == "DEVELOPMENT_PLATEAU_REQUIRES_SUFFICIENCY_REVIEW"
    assert result["best"]["epoch"] == 2 and result["last"]["epoch"] == 4
    capped = native_selection(scalar_native_policy, [3., 2., 2., 2.], steps=[1, 2, 3, 10])
    assert replay_native_selection(capped, scalar_native_policy)["terminal"] == "INCOMPLETE_BUDGET_CAP"
    time_capped = native_selection(scalar_native_policy, [3., 2., 2., 2.], elapsed=[1, 2, 3, 100])
    assert replay_native_selection(time_capped, scalar_native_policy)["terminal"] == "INCOMPLETE_BUDGET_CAP"


@pytest.mark.parametrize("fault", ["role", "best", "last", "terminal", "history", "nonfinite", "promotion", "duplicate", "mapping"])
def test_native_selection_faults_rejected(scalar_native_policy, fault):
    payload = native_selection(scalar_native_policy, [3., 2., 2., 2.])
    if fault == "role": payload["selection_role"] = "calibration"
    elif fault == "best": payload["best"]["epoch"] = 3
    elif fault == "last": payload["last"]["steps"] = 8
    elif fault == "terminal": payload["terminal"] = "SUFFICIENTLY_TRAINED"
    elif fault == "history": payload.pop("history")
    elif fault == "nonfinite": payload["history"][0]["training_loss"] = float("nan")
    elif fault == "promotion": payload["sufficient_training"] = True
    elif fault == "duplicate": payload["history"][1]["epoch"] = 1
    elif fault == "mapping": payload = []
    with pytest.raises((ValueError, FloatingPointError)):
        replay_native_selection(payload, scalar_native_policy)


@pytest.mark.parametrize("text", ['{"a":NaN}', '{"a":Infinity}', '{"a":1,"a":2}', '[]'])
def test_native_strict_json_rejects_malformed(tmp_path, text):
    path = tmp_path / "artifact.json"
    path.write_text(text)
    with pytest.raises(ValueError): strict_json(path)


def native_metadata_root(tmp_path):
    source = Path(__file__).resolve().parents[2]
    paths = [NATIVE_CONFIG_PATH, SCHEMA_PATH,
        "Libs/utils/evaluation/operational_twin_training_sufficiency.py",
        "Libs/utils/evaluation/manuscript_native_protocol.py",
        "Libs/utils/model/native_training_selection.py", "Libs/utils/model/native_scientific_source.py",
        "Libs/utils/exp/native_comparator_training.py",
        "Libs/exp/training/pretrain_microenvironment_world_model.py",
        "Libs/scripts/builders/build_operational_twin_training_sufficiency.py",
        "Libs/scripts/validators/validate_operational_twin_training_sufficiency.py"]
    for ref in paths:
        target = tmp_path / ref
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / ref).read_bytes())
    return tmp_path


@pytest.mark.parametrize("seed_policy,seeds", [
    ("manuscript_native_five", [40,41,42,43,44]),
    ("manuscript_native_three", [41,42,43]),
], ids=["historical_five", "canonical_three"])
def test_native_complete_denominator_missing_no_fit_and_gated(tmp_path, monkeypatch, seed_policy, seeds):
    from Libs.utils.data.raw_data_registry import load_config
    from Libs.utils.evaluation.manuscript_native_protocol import method_roster
    import Libs.utils.evaluation.manuscript_native_protocol as protocol_module
    import Libs.utils.exp.native_comparator_training as comparator_module
    def forbidden(*args, **kwargs):
        raise AssertionError("audit called a producer or constructed a model")
    monkeypatch.setattr(protocol_module, "build_package", forbidden)
    monkeypatch.setattr(comparator_module, "make_comparator", forbidden)
    monkeypatch.setattr(comparator_module, "fit_native_comparator", forbidden)
    import Libs.utils.model.native_scientific_source as source_module
    import Libs.utils.evaluation.operational_twin_training_sufficiency as audit_module
    monkeypatch.setattr(source_module, "integrate_coupled", forbidden)
    monkeypatch.setattr(source_module, "original_physical_trajectory", forbidden)
    monkeypatch.setattr(audit_module, "classify_optimization", forbidden)
    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden)
    cfg = load_config(NATIVE_CONFIG_PATH)
    assert cfg["manuscript_native_protocol"]["training_seeds"] == [41,42,43]
    if seed_policy == "manuscript_native_five":
        # Historical fixture has no prospective execution-review authority.
        cfg["manuscript_native_protocol"].update(seed_policy=seed_policy, training_seeds=seeds)
        cfg["manuscript_native_protocol"]["training_evidence"]["approved_execution_profiles"] = {}
        cfg["manuscript_native_statistics"].update(seed_policy=seed_policy, training_seeds=seeds)
    result = build_native_audit(native_metadata_root(tmp_path), cfg)
    assert result["required_row_count"] == len(method_roster(cfg)) * len(seeds) == len(result["rows"])
    assert {row["seed"] for row in result["rows"]} == set(seeds)
    assert all(row["training_subset_percentage"] == 100 and row["required"] for row in result["rows"])
    assert result["distinct_retained_checkpoint_payloads"] == 0
    assert result["sufficient_training"] is False and result["ScientificTwin"] == "NOT_RUN"
    assert result["excluded_fraction_evidence"] == [10, 25, 50]
    assert all(row["issues"] for row in result["rows"])
    assert next(row for row in result["rows"] if row["method"] == "persistence")["status"] == "NO_FIT_OR_DEPENDENT_CHECKPOINT"
    assert next(row for row in result["rows"] if row["method"] == "scvi")["status"] == "EXTERNAL_OR_LATER_GATE_BLOCKED"
    assert next(row for row in result["rows"] if row["method"] == "large")["checkpoint_owner_profile"].startswith("native_wam_")
    schema = strict_json(tmp_path / SCHEMA_PATH)
    from jsonschema import Draft202012Validator
    Draft202012Validator(schema).validate(result)
    for key, value in [("sufficient_training", True), ("ScientificTwin", "PASS"), ("required_training_seeds", [40])]:
        changed = dict(result, **{key: value})
        with pytest.raises(Exception): Draft202012Validator(schema).validate(changed)


@pytest.mark.parametrize("fault", ["seed", "role", "profile", "checkpoint",
    "selection_list", "last_list", "identity_list", "conditioning_list", "checkpoint_list"])
def test_native_retained_faults_stay_in_denominator(tmp_path, monkeypatch, fault):
    from Libs.utils.data.raw_data_registry import load_config
    from Libs.utils.evaluation.manuscript_native_protocol import training_profile
    from Libs.utils.exp.native_comparator_training import comparator_identity
    cfg = load_config(NATIVE_CONFIG_PATH)
    root = native_metadata_root(tmp_path)
    effective, spec = training_profile(cfg, "gru", 41)
    profile = root / spec["config_path"]
    profile.parent.mkdir(parents=True, exist_ok=True)
    if fault == "profile": effective["native_profile_identity"]["seed"] = 42
    profile.write_text(yaml.safe_dump(effective))
    selection = native_selection(effective["native_scientific_training"]["selection"], [2.])
    if fault == "role": selection["selection_role"] = "calibration"
    binding = dict(spec["training_seed_binding"])
    if fault == "seed": binding["model_seed"] = 42
    manifest = {"method": "gru", "identity": comparator_identity(effective, "gru", {}),
        "training_seed_binding": binding, "selection": selection, "executed_steps": 1,
        "executed_group_visits": 1, "actual_predictor_parameter_count": 1, "failure": None}
    path = root / spec["training_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    # Both references point to one retained metadata payload.
    selection["checkpoint_path_base"] = "manifest_directory"
    for name in ("best", "last"): selection[name]["checkpoint_path"] = "one.pt"
    torch.save({"model_state_dict": {}, "optimizer_state_dict": {}, "epoch": 7 if fault == "checkpoint" else 1,
        "executed_steps": 1, "training_seed_binding": binding, "native_comparator": manifest["identity"]}, path.parent / "one.pt")
    if fault == "selection_list": manifest["selection"] = []
    elif fault == "last_list": selection["last"] = []
    elif fault == "identity_list": manifest["identity"] = []
    elif fault == "conditioning_list": manifest["identity"]["conditioning"] = []
    elif fault == "checkpoint_list": torch.save([], path.parent / "one.pt")
    path.write_text(json.dumps(manifest, allow_nan=False))
    original_load = torch.load
    loaded = []
    def counted_load(path, **kwargs):
        loaded.append(str(path))
        return original_load(path, **kwargs)
    monkeypatch.setattr(torch, "load", counted_load)
    result = build_native_audit(root, cfg)
    assert loaded == ([] if fault in {"selection_list", "identity_list", "conditioning_list"}
                      else [str(path.parent / "one.pt")])
    row = next(row for row in result["rows"] if row["method"] == "gru" and row["seed"] == 41)
    assert row["required"] and row["status"] == "INCOMPLETE_OR_INVALID"
    assert result["distinct_retained_checkpoint_payloads"] == (
        0 if fault in {"selection_list", "identity_list", "conditioning_list", "checkpoint_list"} else 1)
    expected = {"seed": "training_seed_binding_mismatch", "role": "selection_replay_invalid",
                "profile": "actual_profile_identity_or_budget_mismatch", "checkpoint": "best_checkpoint_snapshot_mismatch",
                "selection_list": "native selection must be a mapping", "last_list": "last_snapshot_missing_or_malformed",
                "identity_list": "comparator identity must be a mapping", "conditioning_list": "comparator conditioning must be a mapping",
                "checkpoint_list": "native checkpoint payload must be a mapping"}[fault]
    assert any(expected in issue for issue in row["issues"])
    assert row["visitation_proof"] == "UNAVAILABLE_NO_PER_GROUP_SEQUENCE"


def test_native_locator_preserves_identity_and_denies_protected_paths(tmp_path):
    path = native_evidence_path(tmp_path, "Output/training/model/manifest.json", tmp_path / "relocated", "Output/training")
    assert path == tmp_path / "relocated/model/manifest.json"
    with pytest.raises(ValueError): native_evidence_path(tmp_path, "Output/protected_holdout/manifest.json")
    with pytest.raises(ValueError): native_evidence_path(tmp_path, "../manifest.json")


def test_native_builder_nooverwrite_and_validator_readonly(tmp_path, monkeypatch):
    from Libs.scripts.builders import build_operational_twin_training_sufficiency as builder
    from Libs.scripts.validators import validate_operational_twin_training_sufficiency as validator
    from Libs.utils.data.raw_data_registry import load_config
    import sys
    root = native_metadata_root(tmp_path / "root")
    cfg = load_config(NATIVE_CONFIG_PATH)
    monkeypatch.setattr(builder, "ROOT", root)
    monkeypatch.setattr(validator, "ROOT", root)
    monkeypatch.setattr(builder, "load_config_stack", lambda *args: cfg)
    monkeypatch.setattr(validator, "load_config_stack", lambda *args: cfg)
    output = tmp_path / "evidence"
    args = ["command", "--protocol-scope", "manuscript-native", "--output-dir", str(output)]
    monkeypatch.setattr(sys, "argv", args)
    assert builder.main() == 2
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError): builder.main()
    monkeypatch.setattr(sys, "argv", args + ["--read-only"])
    assert validator.main() == 0
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}
    artifact_path = output / "manuscript_native_training_evidence.json"
    changed = json.loads(artifact_path.read_text())
    changed["row_status_counts"] = {}
    artifact_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="recomputation mismatch"): validator.main()
    artifact_path.write_bytes(before[artifact_path.name])
    source = root / "Libs/utils/model/native_training_selection.py"
    source.write_text(source.read_text() + "\n# source mutation\n")
    with pytest.raises(ValueError, match="recomputation mismatch"): validator.main()
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


@pytest.fixture(scope="module")
def frozen_native_first_fit():
    from Libs.utils.data.raw_data_registry import load_config
    from Libs.utils.evaluation.manuscript_native_protocol import training_profile
    package = ROOT / "Output/audits/native_source_admission/source_bound_consumer_execution/dependent_input_staging_612505"
    chain = strict_json(package / "execution_chain.json")
    profile_bytes = (package / chain["profile_file"]).read_bytes()
    actual = yaml.safe_load(profile_bytes)
    historical = load_config(NATIVE_CONFIG_PATH)
    historical["manuscript_native_protocol"] = deepcopy(actual["manuscript_native_protocol"])
    historical["manuscript_native_statistics"] = deepcopy(actual["manuscript_native_statistics"])
    # The frozen fit predates audit metadata. Preserve its independently reviewed
    # historical authority rather than the active policy's execution approvals.
    reviewed_config = yaml.safe_load(subprocess.check_output([
        "git", "show", "5a0ccb3c6e75fc20556ea42057dc70ce18edcb79:Libs/configs/exp/manuscript_native_protocol.yaml"],
        cwd=ROOT))
    historical["manuscript_native_protocol"]["training_evidence"] = deepcopy(
        reviewed_config["manuscript_native_protocol"]["training_evidence"])
    current, spec = training_profile(historical, "native_wam", 40)
    return chain, profile_bytes, actual, current, spec


def retained_native_chain(tmp_path, frozen):
    chain, profile_bytes, actual, current, spec = frozen
    chain = deepcopy(chain)
    package = tmp_path / "frozen_package"
    package.mkdir(parents=True)
    chain["frozen_package"] = str(package)
    (package / chain["profile_file"]).write_bytes(profile_bytes)
    (package / "execution_chain.json").write_text(json.dumps(chain))
    manifest = tmp_path / "primary" / "microenvironment_pretraining_manifest.json"
    manifest.parent.mkdir()
    (manifest.parent / "execution_chain.json").write_text(json.dumps(chain))
    return chain, package, manifest


def test_native_fixed_execution_accepts_additive_audit_and_calibration(tmp_path, frozen_native_first_fit):
    from Libs.utils.evaluation.operational_twin_training_sufficiency import native_executed_profile
    chain, package, manifest = retained_native_chain(tmp_path, frozen_native_first_fit)
    current = deepcopy(frozen_native_first_fit[3])
    current["manuscript_native_protocol"]["training_evidence"]["approved_execution_profiles"][frozen_native_first_fit[4]["profile_id"]]["execution_chain"] = str(package / "execution_chain.json")
    current["manuscript_native_protocol"]["training_evidence"]["extra_reporting_setting"] = "metadata_only"
    current["native_adaptive_residual_calibration"]["extra_reporting_setting"] = "later_calibration"
    actual, evidence = native_executed_profile(ROOT, manifest, current, frozen_native_first_fit[4],
                                              lambda path: Path(path), strict_json)
    assert actual == frozen_native_first_fit[2]
    assert evidence["source_commit"] == chain["source_commit"]
    assert evidence["profile_sha256"] == chain["profile_sha256"]
    assert evidence["scientific_contract"] == "MATCH"


@pytest.mark.parametrize("fault", ["source", "byte_hash", "content_hash", "profile_bytes", "source_code", "package_chain", "profile_reference"])
def test_native_fixed_execution_rejects_changed_authority(tmp_path, frozen_native_first_fit, fault):
    from Libs.utils.evaluation.operational_twin_training_sufficiency import native_executed_profile
    chain, package, manifest = retained_native_chain(tmp_path, frozen_native_first_fit)
    if fault == "source": chain["source_commit"] = "0" * 40
    elif fault == "byte_hash": chain["profile_sha256"] = "0" * 64
    elif fault == "content_hash": chain["effective_configuration_sha256"] = "0" * 64
    elif fault == "profile_bytes": (package / chain["profile_file"]).write_bytes(frozen_native_first_fit[1] + b"\n")
    elif fault == "source_code": chain["code_sha256"]["Libs/utils/model/native_scientific_source.py"] = "0" * 64
    elif fault == "package_chain": chain["groups"]["train"] = 1
    elif fault == "profile_reference": chain["profile_reference"] = "Output/wrong_profile.yaml"
    (manifest.parent / "execution_chain.json").write_text(json.dumps(chain))
    # Mutate both chains for source/hash faults. The reviewed authority must still
    # reject; agreement between mutable files alone cannot establish identity.
    if fault in {"source", "byte_hash", "content_hash", "source_code"}:
        (package / "execution_chain.json").write_text(json.dumps(chain))
    with pytest.raises(ValueError):
        current = deepcopy(frozen_native_first_fit[3])
        current["manuscript_native_protocol"]["training_evidence"]["approved_execution_profiles"][frozen_native_first_fit[4]["profile_id"]]["execution_chain"] = str(package / "execution_chain.json")
        native_executed_profile(ROOT, manifest, current, frozen_native_first_fit[4],
                                lambda path: Path(path), strict_json)


@pytest.mark.parametrize("fault", ["architecture", "sampling", "groups", "roles", "seed", "source", "budget", "objective", "normalization", "family"])
def test_native_fixed_execution_rejects_scientific_contract_drift(tmp_path, frozen_native_first_fit, fault):
    from Libs.utils.evaluation.operational_twin_training_sufficiency import native_executed_profile
    chain, package, manifest = retained_native_chain(tmp_path, frozen_native_first_fit)
    current = deepcopy(frozen_native_first_fit[3])
    current["manuscript_native_protocol"]["training_evidence"]["approved_execution_profiles"][frozen_native_first_fit[4]["profile_id"]]["execution_chain"] = str(package / "execution_chain.json")
    if fault == "architecture":
        current["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["operator_hidden_dim"] += 1
    elif fault == "sampling": current["native_scientific_training"]["predictive_budget"]["conditional_draws_per_particle"] += 1
    elif fault == "groups": current["native_observable_bridge"]["group_partitions"][0]["groups_per_family"] += 1
    elif fault == "roles": current["native_observable_bridge"]["group_partitions"][0]["role"] = "development"
    elif fault == "seed": current["screening_pipeline"]["screening_benchmark"]["training"]["random_seed"] = 41
    elif fault == "source": current["native_observable_bridge"]["source_uid"] += "_changed"
    elif fault == "budget": current["native_scientific_training"]["selection"]["max_steps"] += 1
    elif fault == "objective": current["native_scientific_training"]["objectives"]["crps"] += 1
    elif fault == "normalization": current["native_diffusion_data"]["field_normalization"]["fit_rule"] += "_changed"
    elif fault == "family": current["native_scientific_training"]["family_scope"] = current["native_scientific_training"]["family_scope"][:1]
    with pytest.raises(ValueError, match="scientific contract mismatch"):
        native_executed_profile(ROOT, manifest, current, frozen_native_first_fit[4],
                                lambda path: Path(path), strict_json)


def test_native_fixed_incomplete_manifest_uses_executed_profile_and_stays_closed(tmp_path, monkeypatch, frozen_native_first_fit):
    from collections import Counter
    from Libs.utils.evaluation.manuscript_native_protocol import training_profile
    from Libs.utils.model.native_scientific_source import native_training_profile, selected_group_definitions
    import Libs.utils.evaluation.operational_twin_training_sufficiency as audit_module
    actual, current, spec = frozen_native_first_fit[2:]
    current = deepcopy(current)
    root = native_metadata_root(tmp_path / "root")
    chain, package, unused = retained_native_chain(tmp_path / "retained", frozen_native_first_fit)
    current["manuscript_native_protocol"]["training_evidence"]["approved_execution_profiles"][spec["profile_id"]]["execution_chain"] = str(package / "execution_chain.json")
    manifest_path = root / spec["training_manifest"]
    manifest_path.parent.mkdir(parents=True)
    (manifest_path.parent / "execution_chain.json").write_text(json.dumps(chain))
    selection = NativeTrainingSelection(actual["native_scientific_training"]["selection"]).payload()
    manifest = {"training_seed_binding": spec["training_seed_binding"], "selection": selection,
                "executed_steps": 0, "executed_group_visits": 0, "failure": None,
                "native_scientific_profile": native_training_profile(actual, "metadata_fixture"),
                "actual_total_parameter_count": 491138172, "checkpoint_scale": spec["checkpoint_scale"],
                "inference_profile": spec["inference_profile"]}
    for role in ("train", "development"):
        groups = selected_group_definitions(actual, role)
        manifest[role + "_group_count"] = len(groups)
        manifest[role + "_family_group_counts"] = dict(Counter(group["family_uid"] for group in groups))
    manifest_path.write_text(json.dumps(manifest))
    original = audit_module.subprocess.check_output
    def source_git(command, **kwargs):
        if command[:2] == ["git", "show"]: kwargs["cwd"] = ROOT
        return original(command, **kwargs)
    monkeypatch.setattr(audit_module.subprocess, "check_output", source_git)
    monkeypatch.setattr(torch, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("checkpoint read")))
    result = build_native_audit(root, current)
    row = next(row for row in result["rows"] if row["method"] == "native_wam" and row["seed"] == 40)
    assert row["profile_evidence"]["scientific_contract"] == "MATCH"
    assert row["profile_evidence"]["effective_configuration_sha256"] == chain["effective_configuration_sha256"]
    assert row["status"] == "INCOMPLETE_OR_INVALID"
    assert "training_partial_failed_or_capped" in row["issues"]
    assert not any("profile_mismatch" in issue or "budget_mismatch" in issue for issue in row["issues"])
    assert result["required_row_count"] == 415 and result["required_training_seeds"] == [40,41,42,43,44]
    assert result["required_fraction"] == 100 and result["sufficient_training"] is False
    assert result["ScientificTwin"] == "NOT_RUN" and result["boundary"]["formal_qualification"] is False


@pytest.mark.parametrize("approved", [True, False])
def test_native_fixed_generic_second_profile_requires_reviewed_binding(tmp_path, frozen_native_first_fit, approved):
    import hashlib
    from Libs.utils.evaluation.manuscript_native_protocol import content_hash, training_profile
    from Libs.utils.evaluation.operational_twin_training_sufficiency import native_executed_profile
    actual, spec = training_profile(frozen_native_first_fit[2], "gru", 41)
    current, _ = training_profile(frozen_native_first_fit[3], "gru", 41)
    package = tmp_path / "frozen_comparator"
    package.mkdir()
    profile_bytes = yaml.safe_dump(actual).encode()
    chain = deepcopy(frozen_native_first_fit[0])
    chain.update(frozen_package=str(package), profile_file=Path(spec["config_path"]).name,
                 profile_reference=spec["config_path"], output_reference=spec["output_root"],
                 profile_sha256=hashlib.sha256(profile_bytes).hexdigest(),
                 effective_configuration_sha256=content_hash(actual))
    (package / chain["profile_file"]).write_bytes(profile_bytes)
    (package / "execution_chain.json").write_text(json.dumps(chain))
    manifest = tmp_path / "comparator" / "native_comparator_training_manifest.json"
    manifest.parent.mkdir()
    (manifest.parent / "execution_chain.json").write_text(json.dumps(chain))
    if approved:
        current["manuscript_native_protocol"]["training_evidence"]["approved_execution_profiles"][spec["profile_id"]] = {
            "execution_chain": str(package / "execution_chain.json"),
            **{key: chain[key] for key in ("source_commit", "profile_sha256", "effective_configuration_sha256")}}
        resolved, evidence = native_executed_profile(ROOT, manifest, current, spec, lambda path: Path(path), strict_json)
        assert resolved["native_profile_identity"]["method"] == "gru"
        assert resolved["native_profile_identity"]["seed"] == 41
        assert evidence["scientific_contract"] == "MATCH"
    else:
        with pytest.raises(ValueError, match="no independently reviewed authority"):
            native_executed_profile(ROOT, manifest, current, spec, lambda path: Path(path), strict_json)


@pytest.mark.parametrize("fault", [None, "approval", "chain_hash", "profile_hash", "historical_policy"])
def test_active_shared_execution_approval_binds_actual_profile_data_and_fit_contract(tmp_path, monkeypatch, fault):
    import hashlib
    from Libs.utils.data.raw_data_registry import load_config
    from Libs.utils.evaluation.manuscript_native_protocol import training_profile, content_hash
    from Libs.utils.evaluation.operational_twin_training_sufficiency import native_executed_profile
    cfg=load_config(NATIVE_CONFIG_PATH)
    # Current active profile; no scientific execution is performed.
    cfg["manuscript_native_protocol"].update(seed_policy="manuscript_native_three", training_seeds=[41,42,43])
    cfg["manuscript_native_statistics"].update(seed_policy="manuscript_native_three", training_seeds=[41,42,43])
    actual,spec=training_profile(cfg,"native_wam",41)
    package=tmp_path/"package";package.mkdir()
    profile_bytes=yaml.safe_dump(actual).encode()
    name=Path(spec["config_path"]).name;(package/name).write_bytes(profile_bytes)
    data={"admission_job":"612505","admission_consumer_commit":"reviewed"}
    data_path=tmp_path/"data_chain.json";data_path.write_text(json.dumps(data))
    references=("Libs/utils/evaluation/manuscript_native_protocol.py", "Libs/utils/model/native_scientific_source.py", "Libs/exp/training/pretrain_microenvironment_world_model.py")
    source={ref:(ROOT/ref).read_bytes() for ref in references}
    chain={"schema_version":"native_source_comparison_execution.v1", "input_mode":"completed_shared_stage",
        "source_commit":"a"*40, "profile_file":name, "profile_reference":spec["config_path"], "output_reference":spec["output_root"],
        "profile_sha256":hashlib.sha256(profile_bytes).hexdigest(), "effective_configuration_sha256":content_hash(actual),
        "frozen_package":str(package),"admission":data,
        "code_sha256":{ref:hashlib.sha256(value).hexdigest() for ref,value in source.items()},
        "shared_input_stage":{"execution_chain_reference":str(data_path),"execution_chain_sha256":hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "job_id":"612799", "stage":"/tmp/NrnSift_native_training_inputs_612799"}}
    chain_bytes=json.dumps(chain).encode();(package/"execution_chain.json").write_bytes(chain_bytes)
    manifest=tmp_path/"primary/microenvironment_pretraining_manifest.json";manifest.parent.mkdir();(manifest.parent/"execution_chain.json").write_bytes(chain_bytes)
    review={"status":"APPROVED_COMPONENT", "profile_id":spec["profile_id"], "source_commit":chain["source_commit"],
        "execution_chain_sha256":hashlib.sha256(chain_bytes).hexdigest(),"profile_sha256":chain["profile_sha256"],
        "effective_configuration_sha256":chain["effective_configuration_sha256"]}
    if fault=="approval":review["status"]="PENDING"
    elif fault=="chain_hash":review["execution_chain_sha256"]="0"*64
    elif fault=="profile_hash":review["profile_sha256"]="0"*64
    review_path=tmp_path/"review.json";review_path.write_text(json.dumps(review))
    current=deepcopy(actual)
    current["manuscript_native_protocol"]["training_evidence"]["approved_execution_profiles"][spec["profile_id"]]={
        "binding_kind":"native_source_comparison_execution.v1","execution_chain":str(package/"execution_chain.json"),"component_review":str(review_path)}
    if fault=="historical_policy":
        current["screening_pipeline"]["screening_benchmark"]["training"]["seed_policy"]="manuscript_native_five"
    monkeypatch.setattr(subprocess,"check_output",lambda argv,**kwargs:source[argv[2].split(":",1)[1]])
    if fault:
        with pytest.raises(ValueError):native_executed_profile(ROOT,manifest,current,spec,lambda path:Path(path),strict_json)
    else:
        resolved,evidence=native_executed_profile(ROOT,manifest,current,spec,lambda path:Path(path),strict_json)
        assert resolved==actual and evidence["scientific_contract"]=="MATCH"
