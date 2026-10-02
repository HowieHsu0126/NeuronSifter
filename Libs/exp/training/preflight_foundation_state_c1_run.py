from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.foundation_state_c1 import (  # noqa: E402
    load_c1_split_units,
    require_live_c1_preflight,
)
from Libs.utils.data.foundation_state_c1_capabilities import (  # noqa: E402
    SAMPLER_CONTRACT_VERSION,
    c1_sampler_contract_binding,
    load_c1_objective_capability_manifest,
    resolve_c1_single_source_selection,
    select_c1_sampler_structural_feasibility,
)
from Libs.utils.data.foundation_state_c1_loader import (  # noqa: E402
    C1_LOADER_IMPLEMENTATION_VERSION,
    C1ReplayScopeAttestation,
    C1RunScopeAttestation,
    C1TranscriptomicsLoader,
    attest_c1_run_scope,
    authorize_c1_run_scope_replay,
    c1_batch_hash,
    c1_sampler_profile_hash,
    release_c1_run_scope_attestation,
    validate_c1_batch_contract,
)
from Libs.utils.data.foundation_state_c1_sampler import (  # noqa: E402
    C1_REPLAY_COORDINATES,
    C1_REPLAY_POLICY,
    C1_MAX_OBSERVATIONS_PER_BIOLOGICAL_UNIT,
    C1_SAMPLER_BATCH_SIZE,
)
from Libs.utils.data.public_foundation_corpus import canonical_sha256  # noqa: E402
from Libs.utils.data.public_foundation_evidence import (  # noqa: E402
    safe_repo_path,
    sha256_file,
)
from Libs.utils.data.raw_data_registry import (  # noqa: E402
    load_config,
    resolve_repo_path,
    write_json,
)
from Libs.scripts.validators.build_c1_preflight import build as build_unified_c1_preflight  # noqa: E402


# Backward-compatible read-only alias for tests and downstream audit helpers.
REPLAY_COORDINATES = C1_REPLAY_COORDINATES


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _require_unified_headline_preflight(
    config: Mapping[str, object], repo_root: Path
) -> dict[str, object]:
    """Require the persisted unified preflight to be current and headline-ready."""

    contract = config["foundation_state_c1"]
    execution = contract["execution"]
    path = safe_repo_path(repo_root, str(execution["unified_preflight_path"]))
    stored = _load_json(path)
    rebuilt = build_unified_c1_preflight(repo_root)
    if canonical_sha256(stored) != canonical_sha256(rebuilt):
        raise ValueError(
            "C1 headline run is blocked: unified preflight is stale or differs "
            "from live reconstruction"
        )
    if rebuilt.get("headline_training_allowed") is not True:
        raise ValueError(
            "C1 headline run is blocked: "
            + str(rebuilt.get("headline_requirements", rebuilt.get("blockers")))
        )
    if str(rebuilt.get("contract_hash")) != canonical_sha256(contract):
        raise ValueError("C1 unified preflight contract hash drift")
    return rebuilt


def _validate_run_preflight_report(
    config: Mapping[str, object], report: Mapping[str, object]
) -> None:
    schema_path = safe_repo_path(
        REPO_ROOT,
        str(config["foundation_state_c1"]["artifacts"]["run_preflight_schema_path"]),
    )
    schema = _load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(report),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        raise ValueError(
            "C1 run preflight violates its schema: "
            + "; ".join(error.message for error in errors)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one frozen-corpus C1 baseline run without starting training."
    )
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--scale", default="Student")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--single-source",
        default="",
        help=(
            "Assertion-only source for the deterministic single-source baseline; "
            "it never overrides the capability-manifest selection."
        ),
    )
    parser.add_argument(
        "--single-family",
        default="",
        help="Required exact family UID for the single-family baseline variant.",
    )
    parser.add_argument("--headline", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def _split_manifest(
    config: Mapping[str, object], split_uid: str
) -> Mapping[str, object]:
    frozen = config["foundation_state_c1"]["frozen_corpus"]
    bundle = _load_json(safe_repo_path(REPO_ROOT, str(frozen["evidence_bundle_path"])))
    split_artifact_path = bundle["evidence"]["frozen_splits"]["path"]
    split_artifact = _load_json(safe_repo_path(REPO_ROOT, str(split_artifact_path)))
    matches = [
        manifest
        for manifest in split_artifact["manifests"]
        if manifest.get("split_uid") == split_uid
    ]
    if len(matches) != 1:
        raise ValueError(f"C1 split UID must resolve exactly once: {split_uid}")
    return matches[0]


def _vocabulary_manifest(config: Mapping[str, object]) -> Mapping[str, object]:
    contract = config["foundation_state_c1"]
    input_contract = contract["encoder"]["input_contract"]
    path = safe_repo_path(REPO_ROOT, str(input_contract["vocabulary_manifest_path"]))
    if sha256_file(path) != str(input_contract["vocabulary_manifest_sha256"]):
        raise ValueError("C1 vocabulary manifest hash drift")
    manifest = _load_json(path)
    if manifest.get("corpus_hash") != contract["frozen_corpus"]["identity_hash"]:
        raise ValueError("C1 vocabulary manifest corpus identity drift")
    return manifest


def _replay_loader_batches(
    config: Mapping[str, object],
    *,
    role_binding_path: Path,
    run_scope_binding: Mapping[str, object],
    seed: int,
    runtime_profile: Mapping[str, object],
) -> tuple[dict[str, object], C1ReplayScopeAttestation]:
    if seed != run_scope_binding["seed"]:
        raise ValueError("C1 replay seed does not match the exact run scope")
    replay_authorization = authorize_c1_run_scope_replay(run_scope_binding)
    source_id = run_scope_binding["source_id"]
    family_uid = run_scope_binding["family_uid"]
    loader = C1TranscriptomicsLoader(
        REPO_ROOT,
        config,
        role_binding_path=role_binding_path,
        expected_role="train",
        batch_size=int(run_scope_binding["batch_size"]),
        profile=str(run_scope_binding["model_profile"]),
        seed=seed,
        sampling_policy=str(run_scope_binding["sampling_policy"]),
        max_observations_per_unit=int(
            run_scope_binding["max_observations_per_biological_unit"]
        ),
        objective_set=str(run_scope_binding["objective_set"]),
        source_id=None if source_id is None else str(source_id),
        family_uid=None if family_uid is None else str(family_uid),
        run_scope_attestation=replay_authorization,
        run_scope_mode="preflight_replay",
    )
    records: list[dict[str, object]] = []
    for epoch, batch_index in REPLAY_COORDINATES:
        first = loader.batch(epoch=epoch, batch_index=batch_index)
        first_hash = c1_batch_hash(first)
        postconditions = validate_c1_batch_contract(
            first,
            active_objectives=list(run_scope_binding["active_objectives"]),
            batch_size=int(run_scope_binding["batch_size"]),
            token_budget=loader.token_budget,
            max_observations_per_unit=int(
                run_scope_binding["max_observations_per_biological_unit"]
            ),
            feature_vocabulary_size=loader.vocabulary.feature_vocabulary_size,
            cell_ontology_size=loader.vocabulary.cell_ontology_size,
            region_ontology_size=loader.vocabulary.region_ontology_size,
            source_class_count=loader.vocabulary.source_count,
            family_class_count=loader.vocabulary.family_count,
            action_feature_dim=first.action_features.shape[1],
            family_balance_required=(
                runtime_profile["family_balance_requirement"] == "required"
            ),
        )
        repeated = loader.batch(epoch=epoch, batch_index=batch_index)
        repeated_hash = c1_batch_hash(repeated)
        if repeated_hash != first_hash:
            raise ValueError(
                "C1 loader replay is non-deterministic at "
                f"epoch={epoch},batch_index={batch_index}"
            )
        records.append(
            {
                "epoch": epoch,
                "batch_index": batch_index,
                "batch_hash": first_hash,
                "repeated_batch_hash": repeated_hash,
                "deterministic": True,
                "postconditions": postconditions,
            }
        )
    return (
        {
            "status": "verified",
            "policy": C1_REPLAY_POLICY,
            "coordinates": records,
            "replay_verified": True,
            "batch_guarantee": True,
            "blocker": "none",
        },
        replay_authorization,
    )


def _run_scope_binding(
    config: Mapping[str, object],
    *,
    baseline_id: str,
    split_uid: str,
    split_manifest_hash: str,
    role_binding_path: Path,
    role_binding_sha256: str,
    source_id: str | None,
    family_uid: str | None,
    model_scale: str,
    objective_set: str,
    active_objectives: Sequence[str],
    sampler_profile: Mapping[str, object],
    seed: int,
    live_preflight_digest: str,
    vocabulary_manifest: Mapping[str, object],
) -> dict[str, object]:
    contract = config["foundation_state_c1"]
    frozen = contract["frozen_corpus"]
    input_contract = contract["encoder"]["input_contract"]
    binding: dict[str, object] = {
        "corpus_uid": frozen["corpus_uid"],
        "corpus_hash": frozen["identity_hash"],
        "observation_binding_hash": frozen["required_observation_binding"][
            "binding_hash"
        ],
        "split_uid": split_uid,
        "split_manifest_hash": split_manifest_hash,
        "split_role": "train",
        "role_binding_path": role_binding_path.relative_to(REPO_ROOT).as_posix(),
        "role_binding_sha256": role_binding_sha256,
        "source_scope": "single_source" if source_id is not None else "all_sources",
        "source_id": source_id,
        "family_scope": "single_family" if family_uid is not None else "all_families",
        "family_uid": family_uid,
        "baseline_id": baseline_id,
        "model_profile": model_scale,
        "objective_set": objective_set,
        "active_objectives": list(active_objectives),
        "sampling_policy": sampler_profile["sampling_policy"],
        "weighting_policy": sampler_profile["weighting_policy"],
        "sampler_contract_version": sampler_profile["sampler_contract_version"],
        "batch_size": int(sampler_profile["batch_size"]),
        "max_observations_per_biological_unit": int(
            sampler_profile["max_observations_per_biological_unit"]
        ),
        "seed": seed,
        "contract_digest": canonical_sha256(contract),
        "live_preflight_digest": live_preflight_digest,
        "loader_implementation_version": C1_LOADER_IMPLEMENTATION_VERSION,
        "vocabulary_manifest_sha256": input_contract["vocabulary_manifest_sha256"],
        "action_descriptor_sha256": vocabulary_manifest["action_descriptor_sha256"],
        "qualified_action_subset_hash": frozen["qualified_action_subset_hash"],
        "objective_capability_manifest_sha256": contract["artifacts"][
            "objective_capability_manifest_sha256"
        ],
        "frozen_block_binding_hash": canonical_sha256(dict(frozen["required_blocks"])),
    }
    binding["sampler_profile_hash"] = c1_sampler_profile_hash(binding)
    binding["run_scope_binding_hash"] = canonical_sha256(binding)
    return binding


def _build_run_preflight_with_authorization(
    config: Mapping[str, object],
    *,
    baseline_id: str,
    split_uid: str,
    model_scale: str,
    seed: int,
    single_source: str = "",
    single_family: str = "",
    headline: bool = False,
) -> tuple[dict[str, object], C1RunScopeAttestation | None]:
    corpus_report = require_live_c1_preflight(config, REPO_ROOT)
    if headline and not corpus_report["headline_training_allowed"]:
        raise ValueError(
            "C1 headline run is blocked: "
            + str(
                corpus_report.get(
                    "headline_training_blocker",
                    corpus_report.get("headline_requirements", corpus_report.get("blockers")),
                )
            )
        )
    if headline:
        _require_unified_headline_preflight(config, REPO_ROOT)

    contract = config["foundation_state_c1"]
    frozen = contract["frozen_corpus"]
    baselines = contract["baselines"]
    if baseline_id not in baselines["required"]:
        raise ValueError(f"Unknown required C1 baseline: {baseline_id}")
    runtime_profile = dict(baselines["runtime_profiles"][baseline_id])
    if model_scale not in contract["encoder"]["profiles"]:
        raise ValueError(f"Unknown C1 model scale: {model_scale}")
    if headline and model_scale != str(contract["execution"]["headline_model_scale"]):
        raise ValueError(
            "C1 headline run must use the frozen model scale "
            + str(contract["execution"]["headline_model_scale"])
        )
    if seed < 0:
        raise ValueError("C1 run seed must be non-negative")
    if split_uid not in set(contract["evaluation"]["frozen_split_uids"]):
        raise ValueError(
            f"C1 split is outside the frozen evaluation matrix: {split_uid}"
        )

    objective_set = str(runtime_profile["objective_set"])
    active_objectives = (
        []
        if objective_set == "none"
        else [
            str(value) for value in contract["encoder"]["objective_sets"][objective_set]
        ]
    )
    vocabulary_manifest = _vocabulary_manifest(config)
    if (
        "masked_gene_program_prediction" in active_objectives
        and vocabulary_manifest.get("program_vocab_status") != "available"
    ):
        raise ValueError(
            "C1 objective requires a qualified program vocabulary, but "
            f"program_vocab_status={vocabulary_manifest.get('program_vocab_status')}"
        )

    capability_manifest = load_c1_objective_capability_manifest(
        config,
        REPO_ROOT,
        verify_live=True,
    )
    sampler_profiles = capability_manifest.get("sampler_profiles")
    if not isinstance(sampler_profiles, Mapping):
        raise ValueError("C1 capability manifest has no sampler profiles")
    sampler_profile_value = sampler_profiles.get(baseline_id)
    if not isinstance(sampler_profile_value, Mapping):
        raise ValueError(f"C1 sampler profile is unavailable: {baseline_id}")
    sampler_profile = dict(sampler_profile_value)
    if sampler_profile.get("active_objectives") != active_objectives:
        raise ValueError("C1 sampler profile objective binding drift")
    if sampler_profile.get("objective_set") != objective_set:
        raise ValueError("C1 sampler profile objective-set binding drift")

    sampler_contract = c1_sampler_contract_binding(config)
    if (
        sampler_contract["version"] != SAMPLER_CONTRACT_VERSION
        or sampler_profile.get("sampler_contract_version")
        != sampler_contract["version"]
        or int(sampler_profile.get("batch_size", 0))
        != int(sampler_contract["batch_size"])
        or int(sampler_profile.get("max_observations_per_biological_unit", 0))
        != int(sampler_contract["max_observations_per_biological_unit"])
    ):
        raise ValueError("C1 sampler profile/contract binding drift")
    if (
        int(sampler_contract["batch_size"]) != C1_SAMPLER_BATCH_SIZE
        or int(sampler_contract["max_observations_per_biological_unit"])
        != C1_MAX_OBSERVATIONS_PER_BIOLOGICAL_UNIT
    ):
        raise ValueError("C1 run preflight requires the frozen B=6/cap=8 sampler")

    pretraining_scope = str(runtime_profile["pretraining_scope"])
    source_selection: Mapping[str, object] | None = None
    selected_source: str | None = None
    selected_family: str | None = None
    if pretraining_scope == "one_train_source_per_run":
        source_selection = resolve_c1_single_source_selection(
            capability_manifest,
            split_uid,
            asserted_source_id=single_source,
        )
        selected_source = str(source_selection["selected_source"])
    elif pretraining_scope == "one_train_family_per_run":
        family_contract = baselines["single_family_selection"]
        allowed_families = set(family_contract["allowed_families"])
        if not single_family:
            raise ValueError("C1 single-family baseline requires --single-family")
        if single_family not in allowed_families:
            raise ValueError(f"C1 single-family baseline family is not preregistered: {single_family}")
        selected_family = single_family
    elif single_source:
        raise ValueError(
            "--single-source is assertion-only and valid only for the single-source baseline"
        )
    elif single_family:
        raise ValueError(
            "--single-family is valid only for the single-family baseline"
        )

    manifest = _split_manifest(config, split_uid)
    if manifest.get("manifest_hash") != frozen["required_split_manifests"][split_uid]:
        raise ValueError("C1 split manifest hash drift")
    query_path = safe_repo_path(REPO_ROOT, str(frozen["required_query_domain"]["path"]))
    role_counts: dict[str, int] = {}
    train_units = []
    train_binding: Mapping[str, object] | None = None
    train_role_path: Path | None = None
    for role in ("train", "validation", "test"):
        binding = manifest["role_bindings"][role]
        role_path = safe_repo_path(REPO_ROOT, str(binding["path"]))
        if sha256_file(role_path) != str(binding["sha256"]):
            raise ValueError(f"C1 frozen {role} role binding hash drift")
        units = load_c1_split_units(query_path, role_path)
        role_counts[role] = len(units)
        if role == "train":
            train_units = units
            train_binding = binding
            train_role_path = role_path
    if train_binding is None or train_role_path is None:
        raise ValueError("C1 split has no frozen train role binding")
    if selected_source is not None:
        train_units = [
            unit for unit in train_units if unit.source_id == selected_source
        ]
        if not train_units:
            raise ValueError(
                "Deterministically selected C1 source is absent from the frozen train role"
            )
    if selected_family is not None:
        train_units = [
            unit for unit in train_units if unit.family_uid == selected_family
        ]
        if not train_units:
            raise ValueError(
                "Selected C1 family is absent from the frozen train role"
            )
    train_sources = {unit.source_id for unit in train_units}
    train_families = {unit.family_uid for unit in train_units}

    structural_feasibility = select_c1_sampler_structural_feasibility(
        capability_manifest,
        split_uid,
        baseline_id,
        source_id=selected_source,
        family_uid=selected_family,
    )
    expected_source_scope = selected_source or (
        f"family:{selected_family}" if selected_family else "all_sources"
    )
    if structural_feasibility.get("source_scope") != expected_source_scope:
        raise ValueError("C1 sampler structural feasibility source scope drift")
    if structural_feasibility.get(
        "sampler_profile_contract_hash"
    ) != sampler_profile.get("sampler_profile_contract_hash"):
        raise ValueError("C1 sampler structural feasibility profile binding drift")
    if pretraining_scope == "none":
        if (
            structural_feasibility.get("status") != "not_applicable"
            or structural_feasibility.get("blocker")
            != "no_pretraining_sampler_required"
            or structural_feasibility.get("replay_required") is not False
        ):
            raise ValueError("C1 no-pretraining sampler scope is not explicit")
    elif (
        structural_feasibility.get("status") != "structurally_feasible"
        or structural_feasibility.get("plan_fit") is not True
        or structural_feasibility.get("replay_required") is not True
        or structural_feasibility.get("runtime_postcondition_required") is not True
    ):
        raise ValueError(
            "C1 sampler is not structurally feasible for "
            f"{split_uid}:{expected_source_scope}:{baseline_id}:"
            f"{structural_feasibility.get('blocker')}"
        )

    runtime_flags = {
        "use_ontology_context": (
            "ontology_aware_gene_cell_region_embedding" in active_objectives
        ),
        "use_action_context": (
            "action_conditioned_perturbation_representation" in active_objectives
        ),
    }
    expected_flags = {
        "use_ontology_context": bool(runtime_profile["ontology_context"]),
        "use_action_context": bool(runtime_profile["action_context"]),
    }
    if runtime_flags != expected_flags:
        raise ValueError(
            f"C1 baseline objective/context profile mismatch: {baseline_id}:{runtime_flags}"
        )

    unified_preflight = _load_json(
        safe_repo_path(
            REPO_ROOT, str(contract["execution"]["unified_preflight_path"])
        )
    )
    run_scope_binding = _run_scope_binding(
        config,
        baseline_id=baseline_id,
        split_uid=split_uid,
        split_manifest_hash=str(manifest["manifest_hash"]),
        role_binding_path=train_role_path,
        role_binding_sha256=str(train_binding["sha256"]),
        source_id=selected_source,
        family_uid=selected_family,
        model_scale=model_scale,
        objective_set=objective_set,
        active_objectives=active_objectives,
        sampler_profile=sampler_profile,
        seed=seed,
        live_preflight_digest=canonical_sha256(unified_preflight),
        vocabulary_manifest=vocabulary_manifest,
    )
    replay_attestation: C1ReplayScopeAttestation | None = None
    if pretraining_scope == "none":
        runtime_replay = {
            "status": "not_applicable",
            "policy": "no_pretraining_sampler_required",
            "coordinates": [],
            "replay_verified": False,
            "batch_guarantee": False,
            "blocker": "no_pretraining_sampler_required",
        }
        training_authorization_status = "not_applicable_no_pretraining"
    else:
        runtime_replay, replay_attestation = _replay_loader_batches(
            config,
            role_binding_path=train_role_path,
            run_scope_binding=run_scope_binding,
            seed=seed,
            runtime_profile=runtime_profile,
        )
        if (
            runtime_replay.get("replay_verified") is not True
            or runtime_replay.get("batch_guarantee") is not True
        ):
            raise ValueError("C1 runtime replay did not establish a batch guarantee")
        training_authorization_status = "runtime_replay_verified"

    required_blocks = dict(frozen["required_blocks"])
    report = {
        "schema_version": "neuronsifter-c1-run-preflight-v2",
        "status": "passed",
        "run_kind": "headline" if headline else "development_or_baseline",
        "claim_id": "C1",
        "baseline_id": baseline_id,
        "model_scale": model_scale,
        "seed": seed,
        "corpus_uid": corpus_report["corpus_uid"],
        "corpus_hash": corpus_report["corpus_hash"],
        "observation_binding_hash": corpus_report["observation_binding_hash"],
        "qualified_action_subset_hash": frozen["qualified_action_subset_hash"],
        "frozen_block_bindings": required_blocks,
        "frozen_block_binding_hash": canonical_sha256(required_blocks),
        "scope": corpus_report["scope"],
        "split_uid": split_uid,
        "split_manifest_hash": manifest["manifest_hash"],
        "train_role_binding": {
            "role": "train",
            "path": train_role_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": train_binding["sha256"],
        },
        "role_counts": role_counts,
        "train_sources": sorted(train_sources),
        "train_families": sorted(train_families),
        "effective_train_unit_count": len(train_units),
        "objective_capability_manifest_sha256": str(
            contract["artifacts"]["objective_capability_manifest_sha256"]
        ),
        "single_source": selected_source,
        "single_source_assertion": single_source or None,
        "single_source_selection": (
            dict(source_selection) if source_selection is not None else None
        ),
        "single_family": selected_family,
        "runtime_profile": runtime_profile,
        "sampler_profile": sampler_profile,
        "sampler_profile_contract_hash": sampler_profile[
            "sampler_profile_contract_hash"
        ],
        "sampler_structural_feasibility": dict(structural_feasibility),
        "active_objectives": active_objectives,
        "runtime_flags": runtime_flags,
        "run_scope_binding": run_scope_binding,
        "run_scope_binding_hash": run_scope_binding["run_scope_binding_hash"],
        "runtime_replay": runtime_replay,
        "replay_verified": runtime_replay["replay_verified"],
        "batch_guarantee": runtime_replay["batch_guarantee"],
        "training_authorization_status": training_authorization_status,
        "training_started": False,
        "result_status": "requires_experiment_result",
        "g3_status": "not_evaluated",
    }
    _validate_run_preflight_report(config, report)
    training_attestation = None
    if pretraining_scope != "none":
        training_attestation = attest_c1_run_scope(report, replay_attestation)
    return report, training_attestation


def build_run_preflight(
    config: Mapping[str, object],
    *,
    baseline_id: str,
    split_uid: str,
    model_scale: str,
    seed: int,
    single_source: str = "",
    single_family: str = "",
    headline: bool = False,
) -> dict[str, object]:
    """Build a serializable audit report that cannot authorize training."""

    report, training_attestation = _build_run_preflight_with_authorization(
        config,
        baseline_id=baseline_id,
        split_uid=split_uid,
        model_scale=model_scale,
        seed=seed,
        single_source=single_source,
        single_family=single_family,
        headline=headline,
    )
    if training_attestation is not None:
        release_c1_run_scope_attestation(training_attestation)
    return report


def build_authorized_run_preflight(
    config: Mapping[str, object],
    *,
    baseline_id: str,
    split_uid: str,
    model_scale: str,
    seed: int,
    single_source: str = "",
    single_family: str = "",
    headline: bool = False,
) -> tuple[dict[str, object], C1RunScopeAttestation]:
    """Return the report and its exact process-local training capability."""

    report, training_attestation = _build_run_preflight_with_authorization(
        config,
        baseline_id=baseline_id,
        split_uid=split_uid,
        model_scale=model_scale,
        seed=seed,
        single_source=single_source,
        single_family=single_family,
        headline=headline,
    )
    if training_attestation is None:
        raise ValueError(
            "C1 baseline has no loader-backed pretraining scope to authorize"
        )
    return report, training_attestation


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    report = build_run_preflight(
        config,
        baseline_id=args.baseline,
        split_uid=args.split,
        model_scale=args.scale,
        seed=args.seed,
        single_source=args.single_source,
        single_family=args.single_family,
        headline=args.headline,
    )
    output = args.output or (
        "Output/audits/c1_run_preflight_"
        f"{args.baseline}_{args.split}_{args.scale}_{args.seed}.json"
    )
    output_path = resolve_repo_path(output)
    write_json(output_path, report)
    print("c1_run_preflight_status=passed")
    print(f"c1_run_preflight_baseline={args.baseline}")
    print(f"c1_run_preflight_split={args.split}")
    print(f"c1_run_preflight_replay_verified={report['replay_verified']}")
    print(f"c1_run_preflight_output={output_path.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
