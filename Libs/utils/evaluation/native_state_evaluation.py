"""Held-readout and evaluator-owned factor recovery for native state probes."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from Libs.models.modules.native_state_probe import held_readout_log_likelihood
from Libs.models.modules.world_state import FACTOR_NAMES
from Libs.utils.data.raw_data_registry import resolve_repo_path, write_json


def controlled_factor_truth(support):
    """Only this evaluator opens the producer's hidden selection; estimators never do."""
    hypotheses = support.physical_parameter_hypotheses
    if hypotheses is None or len(hypotheses["source_record_refs"]) != 1:
        raise ValueError("Factor recovery requires one admitted producer reference per complete group")
    record = json.loads(resolve_repo_path(hypotheses["source_record_refs"][0]).read_text())
    index = record.get("hidden_index")
    if type(index) is not int or not 0 <= index < support.particle_count:
        raise ValueError("Controlled factor oracle has no valid generating whole-particle index")
    declarations = hypotheses["declarations"]
    candidates = record.get("parameters")
    if not isinstance(candidates, list) or len(candidates) != support.particle_count:
        raise ValueError("Controlled factor oracle candidate cardinality differs from admitted support")
    expected = torch.as_tensor(hypotheses["values"])[0]
    values = torch.tensor([[candidate[item["name"]] for item in declarations] for candidate in candidates],
                          dtype=expected.dtype, device=expected.device)
    if values.shape != expected.shape or not torch.equal(values.to(expected), expected):
        raise ValueError("Controlled factor oracle candidates differ from the admitted parameter table")
    return index, {name: support.factors[name][0, index] for name in FACTOR_NAMES}


@torch.no_grad()
def evaluate_native_state(config, method, model, binding, dataset, declared, output):
    """Keep each declared group, source role and unavailable metric in the report."""
    output = Path(output)
    report = {"schema_version": "native_state_evaluation.v1", "method": method,
        "binding": binding, "declared_group_count": len(declared), "groups": [],
        "truth_access": "evaluator_only_after_observation_inference",
        "factor_semantics": "controlled_physical_prior_coordinates_not_empirical_subtypes",
        "scientific_twin": "NOT_RUN", "formal_qualification": False}
    if method in {"state_single_source","state_mixed_source"}:
        report["source_diversity_contrast"] = {**dict(config["native_state_comparison"]["single_source"]),
            "source_scope":config["native_scientific_training"]["source_scope"],
            "normalization_identity":binding["normalization_id"],
            "inference_source_scope":"checkpoint_bound_isolated_training_groups"}
    path = output / "native_state_evaluation.json"
    if path.exists():
        raise FileExistsError("State evaluation output already exists")
    model.eval()
    for index in range(len(dataset)):
        row = {"status": "started"}
        report["groups"].append(row)
        try:
            source = dataset[index]["native_joint_group"]["joint_source"]
            support, observations, logs = model.infer(source)
            if len(support.sample_uids) != 1 or support.sample_uids[0] not in declared:
                raise ValueError("State evaluation group is absent from the predeclared denominator")
            if support.role not in {"validation", "development"}:
                raise ValueError("State evaluation requires development or untouched validation groups")
            uid = support.sample_uids[0]
            row.update(group_uid=uid, family_uid=support.source_family_uid,
                source_uid=support.source_uid, source_snapshot_ref=support.source_snapshot_ref,
                source_variant=declared[uid].get("source_variant"),
                partition_uid=declared[uid].get("partition_uid"), role=support.role)
            held = model.held_scores(source)
            truth_index, truth = controlled_factor_truth(support)
            summaries = model.posterior_factor_summary(source)
            errors = {name: float((summaries[name][0] - truth[name]).square().mean()) for name in FACTOR_NAMES}
            missing = []
            for target_index, block in enumerate(model.blocks):
                for removed_index, removed in enumerate(model.blocks):
                    if target_index == removed_index:
                        continue
                    visible = set(range(model.width)) - set(block) - set(removed)
                    s, obs, weights = model.infer(source, visible_indices=visible)
                    if not any((item.observed | item.left_censored)[:, [
                            i for i, value in enumerate(item.encoder_indices) if value in block
                            ]].any() for item in obs):
                        missing.append({"held_block": target_index, "removed_block": removed_index,
                                        "status": "no_held_observation"})
                        continue
                    score, count = held_readout_log_likelihood(s, obs, weights, block, model.assays)
                    missing.append({"held_block": target_index, "removed_block": removed_index,
                        "status": "defined", "held_readout_log_likelihood": float(score.mean()),
                        "held_feature_count": int(count.sum())})
            metrics = {"held_readout_log_likelihood": float(held.mean()),
                "factor_recovery_mse": sum(errors.values()) / len(errors),
                "factor_mse": errors, "generating_atom_log_probability": float(logs[0, truth_index])}
            if not all(math.isfinite(metrics[key]) for key in (
                    "held_readout_log_likelihood", "factor_recovery_mse", "generating_atom_log_probability")):
                raise FloatingPointError("Nonfinite native state metric")
            row.update(status="defined", metrics=metrics, missing_modality=missing,
                posterior_weights=logs.exp().cpu().tolist()[0])
        except (ValueError, OSError, KeyError, FloatingPointError) as error:
            row.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        write_json(path, report)
    seen = [row.get("group_uid") for row in report["groups"]]
    if len(seen) != len(set(seen)) or set(seen) != set(declared):
        report["denominator_failure"] = "missing_duplicate_or_undeclared_groups"
    report["defined_group_count"] = sum(row["status"] == "defined" for row in report["groups"])
    report["status"] = ("complete" if report["defined_group_count"] == len(declared)
                        and not report.get("denominator_failure") else "incomplete")
    write_json(path, report)
    return report
