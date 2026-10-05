"""Split- and role-scoped C1 objective capability audits.

The frozen corpus contains real observations and actions, but that does not
make every auxiliary objective estimable.  This module materializes the
qualified semantic-key, physical-exposure, and qualified-action intersections
for each frozen split role.  It records static plan fit only; exact loader
replay and runtime postconditions own batch guarantees.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import pyarrow.parquet as pq
from jsonschema import Draft202012Validator

from Libs.utils.data.foundation_state_c1 import (
    C1Unit,
    load_c1_split_units,
    require_live_c1_preflight,
)
from Libs.utils.data.foundation_state_c1_actions import qualified_physical_exposure
from Libs.utils.data.foundation_state_c1_sampler import (
    C1_MAX_OBSERVATIONS_PER_BIOLOGICAL_UNIT,
    C1_SAMPLER_BATCH_SIZE,
    OBJECTIVE_CONSTRAINED_ROW_FILL_POLICY,
    SAMPLER_CONTRACT_VERSION,
    C1ObjectiveRequirementRef,
    deduplicate_objective_requirement_refs,
    plan_objective_requirements,
)
from Libs.utils.data.foundation_state_c1_semantics import (
    SEMANTIC_ALIGNMENT_POLICY_VERSION,
    qualified_semantic_alignment_key,
)
from Libs.utils.data.public_foundation_corpus import canonical_sha256
from Libs.utils.data.public_foundation_evidence import safe_repo_path, sha256_file


CAPABILITY_SCHEMA_VERSION = "neuronsifter-c1-objective-capability-v2"
SINGLE_SOURCE_SELECTION_POLICY_VERSION = (
    "train_role_action_units_desc_then_source_id_asc_v1"
)

_PHYSICAL_EXPOSURE_COLUMNS = (
    "action_uid",
    "physical_dose",
    "dose_value",
    "dose_unit",
    "duration",
    "duration_unit",
    "schedule",
    "route",
    "route_or_not_applicable",
)


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "nan", "<na>"} else text


def _binding_rows(
    repo_root: Path, config: Mapping[str, object]
) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    frozen = config["foundation_state_c1"]["frozen_corpus"]
    required = frozen["required_observation_binding"]
    manifest_path = safe_repo_path(repo_root, str(required["manifest_path"]))
    row_path = safe_repo_path(repo_root, str(required["row_table_path"]))
    if sha256_file(manifest_path) != str(required["manifest_sha256"]):
        raise ValueError("C1 capability binding manifest hash drift")
    if sha256_file(row_path) != str(required["row_table_sha256"]):
        raise ValueError("C1 capability binding row-table hash drift")
    manifest = _load_json(manifest_path)
    with row_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    by_uid: dict[str, dict[str, str]] = {}
    for row in rows:
        uid = _text(row.get("canonical_uid"))
        if not uid or uid in by_uid:
            raise ValueError("C1 capability row binding has duplicate/blank UID")
        by_uid[uid] = {str(key): str(value) for key, value in row.items()}
    if len(by_uid) != int(required["biological_unit_count"]):
        raise ValueError("C1 capability row binding count drift")
    if manifest.get("binding_hash") != required["binding_hash"]:
        raise ValueError("C1 capability binding identity drift")
    return by_uid, manifest


def _qualified_action_exposures(
    repo_root: Path, config: Mapping[str, object]
) -> tuple[dict[str, tuple[object, ...]], str]:
    frozen = config["foundation_state_c1"]["frozen_corpus"]
    manifest_path = safe_repo_path(
        repo_root, str(frozen["qualified_action_subset_manifest_path"])
    )
    manifest = _load_json(manifest_path)
    if manifest.get("corpus_hash") != frozen["identity_hash"]:
        raise ValueError("C1 capability action manifest corpus drift")
    subset_hash = str(manifest.get("qualified_action_subset_hash") or "")
    if subset_hash != str(frozen["qualified_action_subset_hash"]):
        raise ValueError("C1 capability action subset hash drift")
    action_exposures: dict[str, tuple[object, ...]] = {}
    action_record_count = 0
    for catalog in manifest.get("qualified_catalogs", []):
        if not isinstance(catalog, Mapping):
            raise ValueError("C1 capability action catalog is malformed")
        path = safe_repo_path(repo_root, str(catalog["path"]))
        if sha256_file(path) != str(catalog["sha256"]):
            raise ValueError("C1 capability action catalog content drift")
        names = set(pq.ParquetFile(path).schema_arrow.names)
        missing = set(_PHYSICAL_EXPOSURE_COLUMNS) - names
        if missing:
            raise ValueError(
                "C1 qualified action catalog lacks physical exposure fields: "
                f"{path}:{sorted(missing)}"
            )
        table = pq.read_table(path, columns=list(_PHYSICAL_EXPOSURE_COLUMNS))
        data = table.to_pydict()
        for index in range(table.num_rows):
            row = {column: data[column][index] for column in data}
            uid = _text(row.get("action_uid"))
            if not uid or uid in action_exposures:
                raise ValueError("C1 qualified action UID is blank or duplicated")
            action_exposures[uid] = qualified_physical_exposure(row).canonical_key
        action_record_count += table.num_rows
    if action_record_count != int(frozen["required_qualified_action_count"]):
        raise ValueError("C1 capability action record count drift")
    return action_exposures, subset_hash


def _metadata_capabilities(
    repo_root: Path,
    binding: Mapping[str, object],
    qualified_action_exposures: Mapping[str, tuple[object, ...]],
) -> dict[str, dict[str, dict[str, object]]]:
    """Return per-block/key semantic labels and qualified-action counts."""

    capabilities: dict[str, dict[str, dict[str, object]]] = {}
    for block in binding["blocks"]:
        block_uid = str(block["block_uid"])
        metadata = block["metadata"]
        path = safe_repo_path(repo_root, str(metadata["path"]))
        names = set(pq.ParquetFile(path).schema_arrow.names)
        key_column = str(metadata["key_column"])
        columns = [key_column]
        for candidate in (
            "disease_state_canonical",
            "brain_region_ontology_id",
            "brain_region_canonical",
            "action_uid",
        ):
            if candidate in names and candidate not in columns:
                columns.append(candidate)
        table = pq.read_table(path, columns=columns)
        data = table.to_pydict()
        by_key: dict[str, dict[str, object]] = {}
        for index in range(table.num_rows):
            key = _text(data[key_column][index])
            if not key:
                raise ValueError(f"C1 capability metadata has an empty key: {path}")
            row = {column: data[column][index] for column in columns}
            entry = by_key.setdefault(
                key,
                {
                    "semantic_keys": set(),
                    "semantic_refs": set(),
                    "observation_count": 0,
                    "action_observation_count": 0,
                    "action_uids": set(),
                    "action_refs": set(),
                },
            )
            entry["observation_count"] = int(entry["observation_count"]) + 1
            semantic_key = qualified_semantic_alignment_key(row)
            if semantic_key:
                entry["semantic_keys"].add(semantic_key)
                entry["semantic_refs"].add((semantic_key, index))
            action_uid = _text(row.get("action_uid"))
            if action_uid:
                if action_uid not in qualified_action_exposures:
                    raise ValueError(
                        f"C1 metadata references an action outside the qualified subset: {action_uid}"
                    )
                entry["action_observation_count"] = (
                    int(entry["action_observation_count"]) + 1
                )
                entry["action_uids"].add(action_uid)
                entry["action_refs"].add((action_uid, index))
        capabilities[block_uid] = by_key
    return capabilities


def _pair_counts(
    views: set[tuple[str, str, str]],
) -> tuple[int, int, int, int]:
    """Return anchors, unordered positive pairs, units, and semantic groups."""

    by_label: dict[str, list[tuple[str, str]]] = defaultdict(list)
    all_labels = {label for _, label, _ in views}
    for unit_uid, label, source_id in views:
        by_label[label].append((unit_uid, source_id))
    anchors: set[tuple[str, str, str]] = set()
    positive_pairs = 0
    for label, entries in by_label.items():
        for index, (unit_a, source_a) in enumerate(entries):
            for unit_b, source_b in entries[index + 1 :]:
                if source_a == source_b or unit_a == unit_b:
                    continue
                positive_pairs += 1
                if len(all_labels - {label}) > 0:
                    anchors.add((unit_a, label, source_a))
                    anchors.add((unit_b, label, source_b))
    return (
        len(anchors),
        positive_pairs,
        len({unit for unit, _, _ in anchors}),
        len(by_label),
    )


def c1_sampler_contract_binding(config: Mapping[str, object]) -> dict[str, object]:
    """Resolve the sampler fields that every capability and run must bind."""

    contract = config["foundation_state_c1"]
    loader = contract["loader"]
    sampler = loader.get("sampler_contract")
    if not isinstance(sampler, Mapping):
        raise ValueError("C1 sampler contract is missing")
    version = str(sampler.get("version") or "")
    if version != SAMPLER_CONTRACT_VERSION:
        raise ValueError(f"Unsupported C1 sampler contract version: {version}")
    batch_size = int(sampler.get("batch_size", 0))
    max_observations = int(loader.get("max_observations_per_biological_unit", 0))
    if (
        batch_size != C1_SAMPLER_BATCH_SIZE
        or max_observations != C1_MAX_OBSERVATIONS_PER_BIOLOGICAL_UNIT
    ):
        raise ValueError(
            "C1 frozen sampler requires "
            f"batch_size={C1_SAMPLER_BATCH_SIZE} and "
            "max_observations_per_biological_unit="
            f"{C1_MAX_OBSERVATIONS_PER_BIOLOGICAL_UNIT}"
        )
    return {
        "version": version,
        "batch_size": batch_size,
        "max_observations_per_biological_unit": max_observations,
        "observation_reservation_policy": str(
            sampler.get("observation_reservation_policy") or ""
        ),
        "runtime_postcondition_policy": str(
            sampler.get("runtime_postcondition_policy") or ""
        ),
    }


def c1_single_source_selection_contract(
    config: Mapping[str, object],
) -> dict[str, str]:
    """Resolve the deterministic single-source baseline selection contract."""

    selection = config["foundation_state_c1"]["baselines"].get(
        "single_source_selection"
    )
    if not isinstance(selection, Mapping):
        raise ValueError("C1 single-source selection contract is missing")
    resolved = {
        "policy": str(selection.get("policy") or ""),
        "candidate_eligibility": str(selection.get("candidate_eligibility") or ""),
        "manual_source_argument_policy": str(
            selection.get("manual_source_argument_policy") or ""
        ),
    }
    expected = {
        "policy": SINGLE_SOURCE_SELECTION_POLICY_VERSION,
        "candidate_eligibility": (
            "single_source_profile_structurally_feasible_with_identifiable_action"
        ),
        "manual_source_argument_policy": "assertion_only",
    }
    if resolved != expected:
        raise ValueError("Unsupported C1 single-source selection contract")
    return resolved


def c1_sampler_profile_bindings(
    config: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Build canonical sampler bindings for every frozen baseline profile."""

    contract = config["foundation_state_c1"]
    baselines = contract["baselines"]
    profiles = baselines["runtime_profiles"]
    objective_sets = contract["encoder"]["objective_sets"]
    sampler = c1_sampler_contract_binding(config)
    bindings: dict[str, dict[str, object]] = {}
    for baseline_id in baselines["required"]:
        profile = profiles.get(baseline_id)
        if not isinstance(profile, Mapping):
            raise ValueError(f"C1 sampler runtime profile is missing: {baseline_id}")
        objective_set = str(profile.get("objective_set") or "")
        if objective_set == "none":
            active_objectives: list[str] = []
            sampler_mode = "not_applicable"
        else:
            objectives = objective_sets.get(objective_set)
            if not isinstance(objectives, Sequence) or isinstance(
                objectives, (str, bytes)
            ):
                raise ValueError(f"C1 objective set is unavailable: {objective_set}")
            active_objectives = [str(value) for value in objectives]
            objective_aware = any(
                value
                in {
                    "source_aware_contrastive_alignment",
                    "action_conditioned_perturbation_representation",
                }
                for value in active_objectives
            )
            sampling_policy = str(profile["sampling_policy"])
            if sampling_policy == OBJECTIVE_CONSTRAINED_ROW_FILL_POLICY:
                sampler_mode = OBJECTIVE_CONSTRAINED_ROW_FILL_POLICY
            else:
                sampler_mode = (
                    f"{sampling_policy}_objective_aware"
                    if objective_aware
                    else f"{sampling_policy}_plain"
                )
        payload: dict[str, object] = {
            "baseline_id": str(baseline_id),
            "sampler_contract_version": sampler["version"],
            "sampler_mode": sampler_mode,
            "pretraining_scope": str(profile.get("pretraining_scope") or ""),
            "source_scope_policy": str(profile.get("source_scope_policy") or ""),
            "objective_set": objective_set,
            "active_objectives": active_objectives,
            "sampling_policy": str(profile.get("sampling_policy") or ""),
            "weighting_policy": str(profile.get("weighting_policy") or ""),
            "row_sampling_interpretation": (
                "objective_required_unit_reservation_then_row_proportional_fill_not_unbiased_raw_rows"
                if str(profile.get("sampling_policy") or "")
                == OBJECTIVE_CONSTRAINED_ROW_FILL_POLICY
                else "not_applicable"
            ),
            "family_balance_requirement": str(
                profile.get("family_balance_requirement") or ""
            ),
            "batch_size": sampler["batch_size"],
            "max_observations_per_biological_unit": sampler[
                "max_observations_per_biological_unit"
            ],
            "within_unit_aggregation": str(
                contract["loader"]["within_unit_aggregation"]
            ),
            "observation_reservation_policy": sampler["observation_reservation_policy"],
            "runtime_postcondition_policy": sampler["runtime_postcondition_policy"],
        }
        payload["sampler_profile_contract_hash"] = canonical_sha256(payload)
        bindings[str(baseline_id)] = payload
    return bindings


def _profile_sampler_structural_feasibility(
    profile: Mapping[str, object],
    units: Sequence[C1Unit],
    semantic_refs: Sequence[C1ObjectiveRequirementRef],
    action_refs: Sequence[C1ObjectiveRequirementRef],
    *,
    source_scope: str,
    action_representation_identifiable: bool,
    action_identifiability_blocker: str,
) -> dict[str, object]:
    active = set(str(value) for value in profile["active_objectives"])
    alignment_required = "source_aware_contrastive_alignment" in active
    action_required = "action_conditioned_perturbation_representation" in active
    nuisance_required = "source_family_nuisance_regularization" in active
    source_count = len({unit.source_id for unit in units})
    family_units: dict[str, set[str]] = defaultdict(set)
    for unit in units:
        family_units[unit.family_uid].add(unit.biological_unit_uid)
    family_count = len(family_units)
    batch_size = int(profile["batch_size"])
    max_observations = int(profile["max_observations_per_biological_unit"])
    source_policy = str(profile["source_scope_policy"])
    pretraining_scope = str(profile["pretraining_scope"])
    family_requirement = str(profile["family_balance_requirement"])
    family_status = (
        "not_applicable"
        if family_requirement == "not_applicable_no_pretraining"
        else (
            "disabled_by_declared_ablation"
            if family_requirement == "disabled_by_declared_ablation"
            else "not_fit"
        )
    )
    base: dict[str, object] = {
        "status": "not_structurally_feasible",
        "plan_fit": False,
        "blocker": "structural_feasibility_not_evaluated",
        "sampler_profile_contract_hash": profile["sampler_profile_contract_hash"],
        "source_scope": source_scope,
        "scope_unit_hash": canonical_sha256(
            sorted(unit.canonical_uid for unit in units)
        ),
        "unit_count": len({unit.biological_unit_uid for unit in units}),
        "represented_source_count": source_count,
        "represented_family_count": family_count,
        "alignment_required": alignment_required,
        "alignment_plan_fit": not alignment_required,
        "action_required": action_required,
        "action_plan_fit": not action_required,
        "nuisance_scope_required": nuisance_required,
        "nuisance_scope_fit": not nuisance_required,
        "family_balance_plan_status": family_status,
        "within_unit_aggregation_fit": (
            profile["within_unit_aggregation"] == "required_before_loss"
        ),
        "replay_required": pretraining_scope != "none",
        "runtime_postcondition_required": pretraining_scope != "none",
        "required_biological_unit_count": 0,
        "required_observation_count": 0,
    }

    def blocked(
        code: str, *, status: str = "not_structurally_feasible"
    ) -> dict[str, object]:
        base["status"] = status
        base["blocker"] = code
        return base

    if pretraining_scope == "none":
        return blocked("no_pretraining_sampler_required", status="not_applicable")
    if source_policy == "all_train_sources" and source_scope != "all_sources":
        return blocked("source_scope_policy_mismatch", status="not_applicable")
    if source_policy == "all_sources_within_selected_family" and not source_scope.startswith(
        "family:"
    ):
        return blocked("family_scope_policy_mismatch", status="not_applicable")
    if (
        source_policy == SINGLE_SOURCE_SELECTION_POLICY_VERSION
        and source_scope == "all_sources"
    ):
        return blocked("source_scope_policy_mismatch", status="not_applicable")
    if not units:
        return blocked("empty_train_scope")
    if alignment_required:
        semantic_views = {
            (ref.biological_unit_uid, ref.semantic_key, ref.source_id)
            for ref in semantic_refs
        }
        if _pair_counts(semantic_views)[0] == 0:
            return blocked("no_qualified_cross_source_anchor")
    if action_required and not action_refs:
        return blocked("no_qualified_action_observation")
    if action_required and not action_representation_identifiable:
        return blocked(action_identifiability_blocker)
    if nuisance_required and (source_count < 2 or family_count < 2):
        return blocked("insufficient_source_or_family_classes")

    family_quotas: dict[str, int] = {}
    sampling_policy = str(profile["sampling_policy"])
    unit_count = int(base["unit_count"])
    if sampling_policy in {
        "family_balanced",
        "family_and_biological_unit_balanced",
    }:
        if family_count <= 0 or batch_size % family_count != 0:
            return blocked("family_balanced_batch_size_not_divisible")
        quota = batch_size // family_count
        if any(len(values) < quota for values in family_units.values()):
            return blocked("insufficient_family_units_for_batch_quota")
        family_quotas = {family: quota for family in family_units}
        if family_requirement not in {"required", "not_applicable_single_family"}:
            return blocked("family_balance_profile_contract_mismatch")
        base["family_balance_plan_status"] = (
            "not_applicable_single_family"
            if family_requirement == "not_applicable_single_family"
            else "plan_fit"
        )
    elif sampling_policy == "biological_unit_balanced":
        if batch_size > unit_count:
            return blocked("insufficient_units_for_biological_unit_balanced_batch")
        if family_requirement != "not_applicable_single_family":
            return blocked("unit_balanced_profile_contract_mismatch")
        base["family_balance_plan_status"] = "not_applicable_single_family"
    elif sampling_policy == OBJECTIVE_CONSTRAINED_ROW_FILL_POLICY:
        if batch_size > unit_count:
            return blocked("insufficient_units_for_objective_constrained_row_fill")
        if family_requirement != "disabled_by_declared_ablation":
            return blocked("row_proportional_profile_lacks_declared_ablation")
        base["family_balance_plan_status"] = "disabled_by_declared_ablation"
    else:
        return blocked("unsupported_sampling_policy")

    plan = plan_objective_requirements(
        semantic_refs,
        action_refs,
        alignment_required=alignment_required,
        action_required=action_required,
        batch_size=batch_size,
        max_observations_per_unit=max_observations,
        family_quotas=family_quotas,
        ranker=lambda purpose, value: canonical_sha256(
            {"purpose": purpose, "value": list(value)}
        ),
    )
    if plan is None:
        return blocked("objective_requirements_do_not_fit_sampler_contract")
    unique_plan = deduplicate_objective_requirement_refs(plan)
    base.update(
        {
            "status": "structurally_feasible",
            "plan_fit": True,
            "blocker": "runtime_replay_required",
            "alignment_plan_fit": True,
            "action_plan_fit": True,
            "nuisance_scope_fit": True,
            "required_biological_unit_count": len(
                {ref.biological_unit_uid for ref in unique_plan}
            ),
            "required_observation_count": len(unique_plan),
        }
    )
    return base


def _scope_summary(
    units: list[C1Unit],
    binding_rows: Mapping[str, Mapping[str, str]],
    metadata: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    qualified_action_exposures: Mapping[str, tuple[object, ...]],
    sampler_profiles: Mapping[str, Mapping[str, object]],
    source_scope: str,
) -> dict[str, object]:
    selected_sources = {unit.source_id for unit in units}
    selected_families = {unit.family_uid for unit in units}
    views: set[tuple[str, str, str]] = set()
    action_units: set[str] = set()
    action_observation_count = 0
    action_uids: set[str] = set()
    action_source_families: set[tuple[str, str]] = set()
    non_action_source_families: set[tuple[str, str]] = set()
    semantic_refs: set[C1ObjectiveRequirementRef] = set()
    action_refs: set[C1ObjectiveRequirementRef] = set()
    observation_count = 0
    for unit in units:
        row = binding_rows.get(unit.canonical_uid)
        if row is None:
            raise ValueError(
                f"C1 capability unit is absent from row binding: {unit.canonical_uid}"
            )
        block_uid = str(row["block_uid"])
        key = str(row["metadata_key_value"])
        entry = metadata.get(block_uid, {}).get(key)
        if entry is None:
            raise ValueError(
                f"C1 capability metadata selector is absent: {block_uid}:{key}"
            )
        unit_observation_count = int(entry["observation_count"])
        observation_count += unit_observation_count
        for semantic_key in entry["semantic_keys"]:
            views.add((unit.biological_unit_uid, str(semantic_key), unit.source_id))
        for semantic_key, ordinal in entry["semantic_refs"]:
            semantic_refs.add(
                C1ObjectiveRequirementRef(
                    canonical_uid=unit.canonical_uid,
                    biological_unit_uid=unit.biological_unit_uid,
                    family_uid=unit.family_uid,
                    source_id=unit.source_id,
                    ordinal=int(ordinal),
                    semantic_key=str(semantic_key),
                )
            )
        action_count = int(entry["action_observation_count"])
        if action_count:
            action_units.add(unit.biological_unit_uid)
            action_source_families.add((unit.source_id, unit.family_uid))
            action_observation_count += action_count
            action_uids.update(str(value) for value in entry["action_uids"])
            for _, ordinal in entry["action_refs"]:
                action_refs.add(
                    C1ObjectiveRequirementRef(
                        canonical_uid=unit.canonical_uid,
                        biological_unit_uid=unit.biological_unit_uid,
                        family_uid=unit.family_uid,
                        source_id=unit.source_id,
                        ordinal=int(ordinal),
                        action_available=True,
                    )
                )
        if unit_observation_count > action_count:
            non_action_source_families.add((unit.source_id, unit.family_uid))

    anchor_count, pair_count, anchor_unit_count, semantic_group_count = _pair_counts(
        views
    )
    alignment_data_status = "available" if anchor_count > 0 else "not_available"
    action_data_status = "available" if action_units else "not_available"
    within_source_family_non_action_contrast = bool(
        action_source_families & non_action_source_families
    )
    physical_exposure_keys = {
        qualified_action_exposures[action_uid] for action_uid in action_uids
    }
    action_identifiable = bool(action_units) and (
        len(physical_exposure_keys) >= 2 or within_source_family_non_action_contrast
    )
    action_identifiability_blocker = (
        "none"
        if action_identifiable
        else (
            "no_qualified_action_observation"
            if not action_units
            else "single_physical_exposure_family_confounded"
        )
    )
    summary = {
        "unit_count": len({unit.biological_unit_uid for unit in units}),
        "observation_count": observation_count,
        "represented_source_count": len(selected_sources),
        "represented_family_count": len(selected_families),
        "alignment": {
            "status": "not_estimable",
            "data_status": alignment_data_status,
            "semantic_policy_version": SEMANTIC_ALIGNMENT_POLICY_VERSION,
            "eligible_anchor_count": anchor_count,
            "eligible_positive_pair_count": pair_count,
            "eligible_biological_unit_count": anchor_unit_count,
            "semantic_group_count": semantic_group_count,
            "represented_source_count": len({source for _, _, source in views}),
            "batch_guarantee": False,
            "blocker": (
                "no_qualified_cross_source_anchor"
                if anchor_count == 0
                else "profile_scoped_runtime_replay_required"
            ),
        },
        "physical_exposure_context": {
            "status": "not_estimable",
            "data_status": action_data_status,
            "action_row_runtime_path": (
                "available" if action_units else "not_available"
            ),
            "representation_identifiability_status": (
                "structurally_identifiable"
                if action_identifiable
                else "not_structurally_identifiable"
            ),
            "eligible_observation_count": action_observation_count,
            "eligible_biological_unit_count": len(action_units),
            "distinct_qualified_action_uid_count": len(
                action_uids & set(qualified_action_exposures)
            ),
            "distinct_physical_exposure_vector_count": len(physical_exposure_keys),
            "within_source_family_non_action_contrast_available": (
                within_source_family_non_action_contrast
            ),
            "represented_source_count": len(
                {
                    unit.source_id
                    for unit in units
                    if unit.biological_unit_uid in action_units
                }
            ),
            "batch_guarantee": False,
            "blocker": (
                action_identifiability_blocker
                if not action_identifiable
                else "profile_scoped_runtime_replay_required"
            ),
        },
        "profile_sampler_structural_feasibility": {
            baseline_id: _profile_sampler_structural_feasibility(
                profile,
                units,
                sorted(
                    semantic_refs,
                    key=lambda ref: (
                        ref.semantic_key,
                        ref.source_id,
                        ref.biological_unit_uid,
                        ref.canonical_uid,
                        ref.ordinal,
                    ),
                ),
                sorted(
                    action_refs,
                    key=lambda ref: (
                        ref.biological_unit_uid,
                        ref.canonical_uid,
                        ref.ordinal,
                    ),
                ),
                source_scope=source_scope,
                action_representation_identifiable=action_identifiable,
                action_identifiability_blocker=action_identifiability_blocker,
            )
            for baseline_id, profile in sampler_profiles.items()
        },
    }
    return summary


def _derive_single_source_selection(
    train_role_scopes: Mapping[str, object],
    selection_contract: Mapping[str, str],
) -> dict[str, object]:
    """Select one structurally feasible action-bearing train source."""

    source_scopes = train_role_scopes.get("sources")
    if not isinstance(source_scopes, Mapping):
        raise ValueError("C1 train role has no source-scoped capabilities")
    candidates: list[dict[str, object]] = []
    for source_id, scope in source_scopes.items():
        if not isinstance(scope, Mapping):
            raise ValueError(f"C1 source capability is malformed: {source_id}")
        feasibility_by_profile = scope.get("profile_sampler_structural_feasibility")
        action = scope.get("physical_exposure_context")
        if not isinstance(feasibility_by_profile, Mapping) or not isinstance(
            action, Mapping
        ):
            raise ValueError(f"C1 source capability is incomplete: {source_id}")
        feasibility = feasibility_by_profile.get("single_source_pretraining")
        if not isinstance(feasibility, Mapping):
            raise ValueError(
                f"C1 single-source structural feasibility is missing: {source_id}"
            )
        eligible = (
            feasibility.get("status") == "structurally_feasible"
            and feasibility.get("plan_fit") is True
            and feasibility.get("action_required") is True
            and feasibility.get("action_plan_fit") is True
            and feasibility.get("replay_required") is True
            and feasibility.get("runtime_postcondition_required") is True
            and action.get("action_row_runtime_path") == "available"
            and action.get("representation_identifiability_status")
            == "structurally_identifiable"
        )
        if not eligible:
            continue
        candidates.append(
            {
                "source_id": str(source_id),
                "eligible_action_biological_unit_count": int(
                    action["eligible_biological_unit_count"]
                ),
                "eligible_action_observation_count": int(
                    action["eligible_observation_count"]
                ),
                "distinct_qualified_action_uid_count": int(
                    action["distinct_qualified_action_uid_count"]
                ),
                "distinct_physical_exposure_vector_count": int(
                    action["distinct_physical_exposure_vector_count"]
                ),
                "sampler_profile_contract_hash": str(
                    feasibility["sampler_profile_contract_hash"]
                ),
                "scope_unit_hash": str(feasibility["scope_unit_hash"]),
            }
        )
    candidates.sort(
        key=lambda candidate: (
            -int(candidate["eligible_action_biological_unit_count"]),
            str(candidate["source_id"]),
        )
    )
    ranked_candidates = [
        {"rank": rank, **candidate}
        for rank, candidate in enumerate(candidates, start=1)
    ]
    selected_source = (
        str(ranked_candidates[0]["source_id"]) if ranked_candidates else None
    )
    return {
        **dict(selection_contract),
        "selection_status": "selected" if selected_source else "not_estimable",
        "candidates": ranked_candidates,
        "selected_source": selected_source,
        "blocker": (
            "none"
            if selected_source
            else "no_structurally_feasible_identifiable_action_source"
        ),
    }


def build_c1_objective_capability_manifest(
    config: Mapping[str, object], repo_root: Path, *, verify_live: bool = True
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    if verify_live:
        require_live_c1_preflight(config, repo_root)
    contract = config["foundation_state_c1"]
    frozen = contract["frozen_corpus"]
    is_v2 = frozen["corpus_uid"] == "public_foundation_corpus_rescue"
    sampler_contract = c1_sampler_contract_binding(config)
    sampler_profiles = c1_sampler_profile_bindings(config)
    single_source_selection_contract = c1_single_source_selection_contract(config)
    binding_rows, binding = _binding_rows(repo_root, config)
    qualified_action_exposures, action_subset_hash = _qualified_action_exposures(
        repo_root, config
    )
    metadata = _metadata_capabilities(repo_root, binding, qualified_action_exposures)
    evidence = _load_json(
        safe_repo_path(repo_root, str(frozen["evidence_bundle_path"]))
    )
    splits_path = safe_repo_path(
        repo_root, str(evidence["evidence"]["frozen_splits"]["path"])
    )
    split_artifact = _load_json(splits_path)
    if sha256_file(splits_path) != str(evidence["evidence"]["frozen_splits"]["sha256"]):
        raise ValueError("C1 capability frozen split artifact hash drift")
    query_path = safe_repo_path(repo_root, str(frozen["required_query_domain"]["path"]))
    split_summaries: dict[str, object] = {}
    for split in split_artifact["manifests"]:
        split_uid = str(split["split_uid"])
        split_summaries[split_uid] = {
            "split_manifest_hash": str(split["manifest_hash"]),
            "roles": {},
        }
        for role in ("train", "validation", "test"):
            role_binding = split["role_bindings"][role]
            role_path = safe_repo_path(repo_root, str(role_binding["path"]))
            if sha256_file(role_path) != str(role_binding["sha256"]):
                raise ValueError(
                    f"C1 capability split role hash drift: {split_uid}:{role}"
                )
            units = load_c1_split_units(query_path, role_path)
            scopes: dict[str, object] = {
                "all_sources": _scope_summary(
                    units,
                    binding_rows,
                    metadata,
                    qualified_action_exposures=qualified_action_exposures,
                    sampler_profiles=sampler_profiles,
                    source_scope="all_sources",
                ),
                "sources": {},
                "families": {},
            }
            for source_id in sorted({unit.source_id for unit in units}):
                scopes["sources"][source_id] = _scope_summary(
                    [unit for unit in units if unit.source_id == source_id],
                    binding_rows,
                    metadata,
                    qualified_action_exposures=qualified_action_exposures,
                    sampler_profiles=sampler_profiles,
                    source_scope=source_id,
                )
            for family_uid in sorted({unit.family_uid for unit in units}):
                scopes["families"][family_uid] = _scope_summary(
                    [unit for unit in units if unit.family_uid == family_uid],
                    binding_rows,
                    metadata,
                    qualified_action_exposures=qualified_action_exposures,
                    sampler_profiles=sampler_profiles,
                    source_scope=f"family:{family_uid}",
                )
            split_summaries[split_uid]["roles"][role] = scopes
        split_summaries[split_uid]["single_source_selection"] = (
            _derive_single_source_selection(
                split_summaries[split_uid]["roles"]["train"],
                single_source_selection_contract,
            )
        )

    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "corpus_uid": frozen["corpus_uid"],
        "corpus_hash": frozen["identity_hash"],
        "scope": frozen["scope"],
        "derivation_policy": (
            "read_only_derived_artifact_for_frozen_rescue"
            if is_v2
            else "read_only_derived_artifact_outside_frozen_v1"
        ),
        "observation_binding_hash": frozen["required_observation_binding"][
            "binding_hash"
        ],
        "qualified_action_subset_hash": action_subset_hash,
        "required_blocks": dict(sorted(frozen["required_blocks"].items())),
        "semantic_policy_version": SEMANTIC_ALIGNMENT_POLICY_VERSION,
        "sampler_contract": sampler_contract,
        "sampler_profiles": sampler_profiles,
        "single_source_selection_contract": single_source_selection_contract,
        "runtime_guarantee_policy": {
            "structural_feasibility": "static_plan_fit_only_not_a_batch_guarantee",
            "batch_guarantee": "exact_loader_replay_and_runtime_postcondition_required",
            "profile_scope": "split_role_source_profile_exact_match_required",
        },
        "splits": split_summaries,
    }


def load_c1_objective_capability_manifest(
    config: Mapping[str, object],
    repo_root: Path,
    *,
    verify_live: bool = False,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    artifacts = config["foundation_state_c1"]["artifacts"]
    path = safe_repo_path(
        repo_root, str(artifacts["objective_capability_manifest_path"])
    )
    expected_hash = str(artifacts["objective_capability_manifest_sha256"])
    if sha256_file(path) != expected_hash:
        raise ValueError("C1 objective capability manifest hash drift")
    payload = _load_json(path)
    schema_path = safe_repo_path(
        repo_root, str(artifacts["objective_capability_manifest_schema_path"])
    )
    schema = _load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        raise ValueError(
            "C1 objective capability manifest violates its schema: "
            + "; ".join(error.message for error in errors)
        )
    frozen = config["foundation_state_c1"]["frozen_corpus"]
    if (
        payload.get("corpus_hash") != frozen["identity_hash"]
        or payload.get("observation_binding_hash")
        != frozen["required_observation_binding"]["binding_hash"]
        or payload.get("qualified_action_subset_hash")
        != frozen["qualified_action_subset_hash"]
        or payload.get("required_blocks") != frozen["required_blocks"]
    ):
        raise ValueError("C1 objective capability manifest corpus binding drift")
    expected_sampler_contract = c1_sampler_contract_binding(config)
    expected_sampler_profiles = c1_sampler_profile_bindings(config)
    expected_single_source_selection = c1_single_source_selection_contract(config)
    if payload.get("sampler_contract") != expected_sampler_contract:
        raise ValueError("C1 objective capability sampler contract drift")
    if payload.get("sampler_profiles") != expected_sampler_profiles:
        raise ValueError("C1 objective capability sampler profile drift")
    if (
        payload.get("single_source_selection_contract")
        != expected_single_source_selection
    ):
        raise ValueError("C1 objective capability single-source policy drift")
    split_hashes = {
        str(split_uid): str(split["split_manifest_hash"])
        for split_uid, split in payload["splits"].items()
    }
    if split_hashes != frozen["required_split_manifests"]:
        raise ValueError("C1 objective capability split binding drift")
    for split_uid, split in payload["splits"].items():
        for role, role_scopes in split["roles"].items():
            scopes = {"all_sources": role_scopes["all_sources"]}
            scopes.update(role_scopes["sources"])
            scopes.update(
                {f"family:{key}": value for key, value in role_scopes["families"].items()}
            )
            for source_scope, scope_payload in scopes.items():
                feasibility_by_profile = scope_payload[
                    "profile_sampler_structural_feasibility"
                ]
                if set(feasibility_by_profile) != set(expected_sampler_profiles):
                    raise ValueError(
                        "C1 objective capability sampler profile coverage drift: "
                        f"{split_uid}:{role}:{source_scope}"
                    )
                for baseline_id, feasibility in feasibility_by_profile.items():
                    expected_hash = expected_sampler_profiles[baseline_id][
                        "sampler_profile_contract_hash"
                    ]
                    if (
                        feasibility.get("sampler_profile_contract_hash")
                        != expected_hash
                        or feasibility.get("source_scope") != source_scope
                        or "batch_guarantee" in feasibility
                    ):
                        raise ValueError(
                            "C1 objective capability scoped sampler feasibility drift: "
                            f"{split_uid}:{role}:{source_scope}:{baseline_id}"
                        )
        expected_selection = _derive_single_source_selection(
            split["roles"]["train"],
            expected_single_source_selection,
        )
        if split.get("single_source_selection") != expected_selection:
            raise ValueError(
                "C1 objective capability deterministic single-source selection drift: "
                f"{split_uid}"
            )
    if verify_live:
        rebuilt = build_c1_objective_capability_manifest(
            config,
            repo_root,
            verify_live=False,
        )
        if canonical_sha256(payload) != canonical_sha256(rebuilt):
            raise ValueError(
                "C1 objective capability manifest differs from live reconstruction"
            )
    return payload


def select_c1_objective_capability(
    manifest: Mapping[str, object], split_uid: str, *, source_id: str | None = None,
    family_uid: str | None = None,
) -> Mapping[str, object]:
    if source_id is not None and family_uid is not None:
        raise ValueError("C1 capability scope cannot select source and family together")
    try:
        split = manifest["splits"][split_uid]
        scopes = split["roles"]["train"]
        if source_id is not None:
            return scopes["sources"][source_id]
        if family_uid is not None:
            return scopes["families"][family_uid]
        return scopes["all_sources"]
    except (KeyError, TypeError) as exc:
        scope = source_id or (f"family:{family_uid}" if family_uid else "all_sources")
        raise ValueError(
            f"C1 objective capability is unavailable for {split_uid}:train:{scope}"
        ) from exc


def select_c1_sampler_structural_feasibility(
    manifest: Mapping[str, object],
    split_uid: str,
    baseline_id: str,
    *,
    source_id: str | None = None,
    family_uid: str | None = None,
) -> Mapping[str, object]:
    scope = select_c1_objective_capability(
        manifest,
        split_uid,
        source_id=source_id,
        family_uid=family_uid,
    )
    try:
        feasibility = scope["profile_sampler_structural_feasibility"][baseline_id]
    except (KeyError, TypeError) as exc:
        source_scope = source_id or (f"family:{family_uid}" if family_uid else "all_sources")
        raise ValueError(
            "C1 sampler structural feasibility is unavailable for "
            f"{split_uid}:train:{source_scope}:{baseline_id}"
        ) from exc
    if not isinstance(feasibility, Mapping):
        raise ValueError("C1 sampler structural feasibility is malformed")
    if "batch_guarantee" in feasibility:
        raise ValueError("C1 structural feasibility must not claim a batch guarantee")
    return feasibility


def resolve_c1_single_source_selection(
    manifest: Mapping[str, object],
    split_uid: str,
    *,
    asserted_source_id: str = "",
) -> Mapping[str, object]:
    """Resolve the frozen policy selection; a CLI source is assertion-only."""

    try:
        selection = manifest["splits"][split_uid]["single_source_selection"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"C1 single-source selection is unavailable for {split_uid}:train"
        ) from exc
    if not isinstance(selection, Mapping):
        raise ValueError("C1 single-source selection is malformed")
    selected_source = selection.get("selected_source")
    if (
        selection.get("policy") != SINGLE_SOURCE_SELECTION_POLICY_VERSION
        or selection.get("manual_source_argument_policy") != "assertion_only"
    ):
        raise ValueError("C1 single-source selection policy drift")
    if selection.get("selection_status") != "selected" or not isinstance(
        selected_source, str
    ):
        raise ValueError(
            f"C1 single-source selection is not_estimable for {split_uid}: "
            f"{selection.get('blocker')}"
        )
    if asserted_source_id and asserted_source_id != selected_source:
        raise ValueError(
            "C1 --single-source is assertion-only and does not match the "
            f"deterministic selection: asserted={asserted_source_id}, "
            f"selected={selected_source}"
        )
    return selection


def _c1_holdout_family(split_uid: str) -> str:
    if split_uid == "source_publication_holdout":
        return "source_publication_holdout"
    if split_uid.startswith("family_holdout_"):
        return "family_holdout"
    if split_uid == "regimen_holdout":
        return "regimen_holdout"
    if split_uid == "target_program_holdout":
        return "target_program_holdout"
    raise ValueError(f"C1 split is not mapped to a holdout family: {split_uid}")


def derive_c1_headline_feasibility(
    manifest: Mapping[str, object],
    task_registry: Mapping[str, object],
    *,
    runtime_profile_id: str,
    required_task_family_count: int,
    required_holdout_family_count: int,
) -> dict[str, object]:
    """Derive headline coverage from live objective capability evidence.

    A task/holdout pair is reachable only when the exact all-source runtime
    profile is structurally feasible for its frozen training split.  This
    deliberately reports reachability before any empirical result so a job
    matrix cannot submit runs whose required objective evidence is absent.
    """

    if not isinstance(task_registry.get("tasks"), Sequence):
        raise ValueError("C1 task registry has no task rows")
    split_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    reachable_task_families: set[str] = set()
    reachable_holdout_families: set[str] = set()
    for task in task_registry["tasks"]:
        if not isinstance(task, Mapping):
            raise ValueError("C1 task registry contains a malformed task row")
        task_family = str(task["task_family"])
        applicable = task.get("applicable_holdouts", [])
        if not isinstance(applicable, Sequence):
            raise ValueError(f"C1 task has malformed holdout list: {task_family}")
        for split_uid_value in applicable:
            split_uid = str(split_uid_value)
            holdout_family = _c1_holdout_family(split_uid)
            try:
                feasibility = select_c1_sampler_structural_feasibility(
                    manifest,
                    split_uid,
                    runtime_profile_id,
                )
                structural_status = str(feasibility.get("status", "unavailable"))
                blocker = str(feasibility.get("blocker", ""))
                split_hash = str(manifest["splits"][split_uid]["split_manifest_hash"])
            except (KeyError, TypeError, ValueError) as exc:
                structural_status = "unavailable"
                blocker = type(exc).__name__
                split_hash = ""
            reachable = structural_status == "structurally_feasible"
            row = {
                "task_family": task_family,
                "split_uid": split_uid,
                "split_hash": split_hash,
                "holdout_family": holdout_family,
                "structural_status": structural_status,
                "blocker": blocker,
                "reachable": reachable,
            }
            split_rows.append(row)
            task_rows.append(row)
            if reachable:
                reachable_task_families.add(task_family)
                reachable_holdout_families.add(holdout_family)

    blockers: list[dict[str, object]] = []
    if len(reachable_task_families) < required_task_family_count:
        blockers.append(
            {
                "code": "headline_task_family_coverage_unreachable",
                "required": required_task_family_count,
                "maximum_reachable": len(reachable_task_families),
                "families": sorted(reachable_task_families),
            }
        )
    if len(reachable_holdout_families) < required_holdout_family_count:
        blockers.append(
            {
                "code": "headline_holdout_family_coverage_unreachable",
                "required": required_holdout_family_count,
                "maximum_reachable": len(reachable_holdout_families),
                "families": sorted(reachable_holdout_families),
            }
        )
    return {
        "status": "reachable" if not blockers else "unreachable",
        "runtime_profile_id": runtime_profile_id,
        "minimum_task_family_count": required_task_family_count,
        "minimum_holdout_family_count": required_holdout_family_count,
        "maximum_reachable_task_family_count": len(reachable_task_families),
        "maximum_reachable_holdout_family_count": len(reachable_holdout_families),
        "reachable_task_families": sorted(reachable_task_families),
        "reachable_holdout_families": sorted(reachable_holdout_families),
        "split_rows": split_rows,
        "blockers": blockers,
    }
