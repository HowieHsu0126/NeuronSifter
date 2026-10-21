"""Selected learned native teacher binding; separate from solver reconstruction."""
from __future__ import annotations

import json
import math

import torch

from Libs.utils.model.native_training_selection import training_checkpoint_path
from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path
from Libs.utils.exp.experiment_manifests import resolve_training_seed, validate_training_seed_checkpoint
from Libs.utils.model.native_direct_mean import file_sha256


def validate_distillation_policy(policy: dict) -> None:
    weight = policy.get("weight")
    if (policy.get("loss") != "masked_categorical_kl" or type(weight) not in (int, float)
            or not math.isfinite(weight) or weight <= 0):
        raise ValueError("Learned distillation requires an explicit positive masked-categorical KL weight")
    if policy.get("development_selection") != "unchanged_native_objective":
        raise ValueError("Learned distillation must preserve the matched native development objective")


def selected_teacher_identity(config: dict, conditioning: dict | None) -> dict | None:
    """Validate actual selected optimizer state without constructing another teacher."""
    policy = config.get("native_learned_distillation", {})
    if not policy.get("enabled", False):
        return None
    validate_distillation_policy(policy)
    if not conditioning:
        raise ValueError("Learned native distillation requires the frozen selected mainline posterior")
    config_path = resolve_repo_path(policy["teacher_config"])
    manifest_path = resolve_repo_path(policy["teacher_manifest"])
    if (str(manifest_path) != conditioning["manifest_path"]
            or file_sha256(config_path) != conditioning["conditioning_config_sha256"]):
        raise ValueError("Teacher must be the identical selected common posterior configuration and manifest")
    teacher_config = load_config(str(config_path))
    if config.get("native_scientific_training",{}).get("source_scope","all") != "all":
        from Libs.utils.model.native_scientific_source import training_data_scope
        if training_data_scope(teacher_config) != training_data_scope(config):
            raise ValueError("Isolated source learner cannot use a differently scoped learned teacher")
    report = json.loads(manifest_path.read_text())
    best = report.get("selection", {}).get("best") or {}
    if (report.get("selection", {}).get("selection_role") != "development"
            or type(best.get("steps")) is not int or best["steps"] < 1
            or type(report.get("executed_steps")) is not int or report["executed_steps"] < best["steps"]):
        raise ValueError("Learned teacher requires an actually fitted development-selected checkpoint")
    checkpoint_path = training_checkpoint_path(manifest_path,report["selection"],best["checkpoint_path"])
    if str(checkpoint_path) != conditioning["checkpoint_path"]:
        raise ValueError("Teacher checkpoint differs from the selected common posterior")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    validate_training_seed_checkpoint(checkpoint, resolve_training_seed(config, require_selected=True))
    optimizer_state = checkpoint.get("optimizer_state_dict", {}).get("state", {})
    steps = [float(state.get("step", 0)) for state in optimizer_state.values()]
    if (checkpoint.get("executed_steps") != best["steps"] or not steps
            or max(steps) != best["steps"]):
        raise ValueError("Learned teacher selected checkpoint lacks the declared real optimizer updates")
    if teacher_config.get("native_learned_distillation", {}).get("enabled", False):
        raise ValueError("The selected mainline teacher must not recursively use learned distillation")
    binding = {key: conditioning[key] for key in ("checkpoint_path", "checkpoint_sha256",
        "conditioning_config_sha256", "manifest_path", "training_seed_binding", "normalization_id")}
    return {"binding": binding, "selected_teacher_optimizer_updates": best["steps"],
        "teacher_config": str(config_path), "teacher_manifest": str(manifest_path),
        "loss": policy["loss"], "weight": policy["weight"], "teacher_eval": True,
        "teacher_gradients": False, "development_selection": policy["development_selection"],
        "solver_reconstruction_is_separate": True}


def validate_saved_learned_teacher(checkpoint: dict, report: dict, config: dict, conditioning: dict) -> None:
    expected = selected_teacher_identity(config, conditioning)
    if checkpoint.get("native_learned_teacher") != expected or report.get("learned_teacher") != expected:
        raise ValueError("Native checkpoint learned-teacher binding mismatch")
    calls = report.get("learned_teacher_forward_calls", 0)
    if type(calls) is not int or calls < 0 or (expected is not None and calls == 0) or (expected is None and calls):
        raise ValueError("Native checkpoint has inconsistent actual learned-teacher forward calls")


def load_selected_native_teacher(model, config: dict, device):
    identity = selected_teacher_identity(config, getattr(model, "frozen_conditioning_identity", None))
    if identity is None:
        return None
    teacher_config = load_config(identity["teacher_config"])
    config_path = resolve_repo_path(identity["teacher_config"])
    manifest_path = resolve_repo_path(identity["teacher_manifest"])
    # The maintained loader imports the pretraining factory; defer to avoid a cycle.
    from Libs.utils.evaluation.native_comparator_evaluation import load_native_comparator
    # Additional teacher construction must not move the learner's dropout/init stream.
    with torch.random.fork_rng(devices=[]):
        wrapper, _, binding = load_native_comparator(
            teacher_config, teacher_config.get("native_profile_identity",{}).get("method","native_wam"), config_path, manifest_path, manifest_path, None, device)
    teacher = wrapper.mainline
    if (binding["checkpoint_sha256"] != identity["binding"]["checkpoint_sha256"]
            or teacher.field_normalization.identity != model.field_normalization.identity):
        raise ValueError("Teacher and learner must share their selected common posterior and normalization")
    if (teacher.training or any(parameter.requires_grad for parameter in teacher.parameters())
            or not torch.equal(teacher.node_adjacency, model.node_adjacency)):
        raise ValueError("Selected teacher must remain frozen/eval on the learner's declared graph")
    learner_sampler, teacher_sampler = model.quantized_mddm, teacher.quantized_mddm
    if (teacher_sampler.quantity_profile != learner_sampler.quantity_profile
            or teacher_sampler.action_policy_name != learner_sampler.action_policy_name
            or teacher_sampler.posterior_context_dim != learner_sampler.posterior_context_dim
            or teacher.operator.tokenizer.wavelet_level_count != model.operator.tokenizer.wavelet_level_count
            or not torch.equal(teacher_sampler.codebook_values, learner_sampler.codebook_values)
            or not torch.equal(teacher_sampler.codebook_boundaries, learner_sampler.codebook_boundaries)):
        raise ValueError("Learned teacher must share native quantities, action, context, Haar and codebook semantics")
    teacher.native_teacher_checkpoint_sha256 = binding["checkpoint_sha256"]
    model.native_teacher_forward_calls = 0
    model.native_learned_teacher_identity = identity
    return teacher
