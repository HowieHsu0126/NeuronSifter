"""Disposable metadata evidence; no neural modules, optimization or private worlds."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch
import yaml
from jsonschema import ValidationError

from Libs.tests.test_operational_twin_training_sufficiency import native_metadata_root, native_selection
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.evaluation import operational_twin_training_sufficiency as audit_module
from Libs.utils.evaluation.manuscript_native_protocol import (
    content_hash, method_roster, training_component_definition, training_profile,
)
from Libs.utils.exp.native_comparator_training import comparator_identity
from Libs.utils.model.native_scientific_source import native_training_profile, selected_group_definitions

COMPONENT = "residual_mpc_predictor"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return path


def git(root, *args):
    return subprocess.check_output(["git", "-c", "user.name=Disposable Fixture",
        "-c", "user.email=fixture@example.invalid", *args], cwd=root, stderr=subprocess.PIPE, text=True).strip()


@pytest.fixture(scope="module")
def retained_component(tmp_path_factory):
    root = native_metadata_root(tmp_path_factory.mktemp("predictor_review"))
    cfg = load_config(audit_module.NATIVE_CONFIG_PATH)
    git(root, "init", "--quiet")
    git(root, "add", "Libs")
    git(root, "commit", "--quiet", "-m", "Disposable source identities")
    source_commit = git(root, "rev-parse", "HEAD")
    sources = ["Libs/utils/evaluation/manuscript_native_protocol.py",
        "Libs/utils/model/native_scientific_source.py",
        "Libs/exp/training/pretrain_microenvironment_world_model.py"]
    bindings = {}
    effective_profiles = {}
    common = {}
    for method in ("native_wam", "mechanistic_residual"):
        for seed in (41, 42, 43):
            effective, spec = training_profile(cfg, method, seed,
                training_component=COMPONENT if method == "mechanistic_residual" else None)
            effective_profiles[(method, seed)] = effective
            profile = root / spec["config_path"]
            profile.parent.mkdir(parents=True, exist_ok=True)
            profile.write_text(yaml.safe_dump(effective))
            manifest_path = root / spec["training_manifest"]
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            selection = native_selection(effective["native_scientific_training"]["selection"],
                [1.] * 200, steps=[63 * epoch for epoch in range(1, 201)])
            assert selection["terminal"] == "DEVELOPMENT_PLATEAU_REQUIRES_SUFFICIENCY_REVIEW"
            selection["checkpoint_path_base"] = "manifest_directory"
            native = method == "native_wam"
            identity = native_training_profile(effective, "disposable_normalizer") if native else comparator_identity(effective, method, common[seed])
            manifest = {"training_seed_binding": spec["training_seed_binding"], "selection": selection,
                "executed_steps": selection["last"]["steps"], "executed_group_visits": 636000,
                "failure": None, "scientific_twin": "NOT_RUN", "method": method}
            for role in ("train", "development"):
                groups = selected_group_definitions(effective, role)
                manifest[role + "_group_count"] = len(groups)
                manifest[role + "_family_group_counts"] = dict(Counter(row["family_uid"] for row in groups))
            if native:
                manifest.update(native_scientific_profile=identity, actual_total_parameter_count=491138172,
                    checkpoint_scale=spec["checkpoint_scale"], inference_profile=spec["inference_profile"])
            else:
                manifest.update(identity=identity, actual_predictor_parameter_count=1000000)
            for label in ("best", "last"):
                selected = selection[label]
                selected["checkpoint_path"] = label + ".pt"
                # These are serialization fixtures, with no fitted tensor state.
                torch.save({"model_state_dict": {}, "optimizer_state_dict": {},
                    "epoch": selected["epoch"], "executed_steps": selected["steps"],
                    "training_seed_binding": spec["training_seed_binding"],
                    "native_scientific_profile" if native else "native_comparator": identity}, manifest_path.parent / (label + ".pt"))
            write(manifest_path, manifest)
            selected_path = manifest_path.parent / "best.pt"
            if native:
                common[seed] = {"checkpoint_path": str(selected_path), "checkpoint_sha256": sha(selected_path),
                    "conditioning_config_sha256": sha(profile), "manifest_path": str(manifest_path),
                    "training_seed_binding": spec["training_seed_binding"], "normalization_id": "disposable_normalizer",
                    "selection_role": "development", "sufficient_training": False}
            else:
                bindings[seed] = {"model_identity": identity, "training_manifest": str(manifest_path),
                    "checkpoint_path": str(selected_path), "checkpoint_sha256": sha(selected_path),
                    "normalization_id": "disposable_normalizer"}
            package = root / "fixture_packages" / spec["profile_id"]
            package.mkdir(parents=True)
            (package / "profile.yaml").write_bytes(profile.read_bytes())
            data_chain = write(package / "data_chain.json", {"source_identity": "disposable_source"})
            chain = {"schema_version": "native_source_comparison_execution.v1", "input_mode": "completed_shared_stage",
                "source_commit": source_commit, "profile_sha256": sha(profile),
                "effective_configuration_sha256": content_hash(effective),
                "profile_reference": spec["config_path"], "output_reference": spec["output_root"],
                "profile_file": "profile.yaml", "frozen_package": str(package),
                "code_sha256": {ref: sha(root / ref) for ref in sources},
                "admission": {"source_identity": "disposable_source"},
                "shared_input_stage": {"stage": "/tmp/NrnSift_native_training_inputs_fixture", "job_id": "fixture",
                    "execution_chain_reference": str(data_chain), "execution_chain_sha256": sha(data_chain)}}
            retained_chain = write(package / "execution_chain.json", chain)
            write(manifest_path.parent / "execution_chain.json", chain)
            review = write(package / "execution_review.json", {"status": "APPROVED_COMPONENT",
                "profile_id": spec["profile_id"], "execution_chain_sha256": sha(retained_chain),
                **{key: chain[key] for key in ("source_commit", "profile_sha256", "effective_configuration_sha256")}})
            cfg["manuscript_native_protocol"]["training_evidence"]["approved_execution_profiles"][spec["profile_id"]] = {
                "binding_kind": chain["schema_version"], "execution_chain": str(retained_chain), "component_review": str(review)}
            write(manifest_path.parent / "execution.json", {"scope": "disposable_software_metadata", "elapsed_seconds": 159})
            (manifest_path.parent / "process_resources.log").write_text("disposable measured resource receipt\n")
    base_ref = str(Path(cfg["native_observable_bridge"]["output_root"]) / "native_consumption.json")
    named_ref = str(Path(cfg["native_diffusion_data"]["registry_path"]).parent / "native_consumption.json")
    base = {"status": "complete", "admitted_count": cfg["native_observable_bridge"]["budget"]["cases"],
        "source_identity": "disposable_source"}
    write(root / base_ref, base)
    write(root / named_ref, {**base, "operation": "bind_retained_physical_candidates",
        "array_values_unchanged": True, "compiled_programs_unchanged": True, "original_consumption_ref": base_ref})
    artifact = audit_module.build_native_audit(root, cfg, training_component=COMPONENT)
    assert artifact["component_issues"] == [], [(row["method"], row["seed"], row["issues"]) for row in artifact["rows"]]
    path = write(root / "fixture_review" / "audit.json", artifact)
    definition = training_component_definition(cfg, COMPONENT)
    verdict = {"schema_version": "native_predictor_training_verdict.v1", "training_component": COMPONENT,
        "owner_method": "residual_mpc", "review_scope": definition["review_scope"], "decision": "APPROVED",
        "reviewer": {"identity": "disposable independent reviewer fixture", "role": "independent_scientific_reviewer"},
        "audit": {"path": str(path.relative_to(root)), "sha256": sha(path)}, "boundary": dict(audit_module.CLOSED),
        "findings": {key: {"status": "supported", "rationale": "Disposable transport fixture; no scientific result.",
            "evidence": [{"path": str(path.relative_to(root)), "sha256": sha(path)}]} for key in definition["required_review_findings"]}}
    admission = {"scope": "admitted_controlled_source", "source_identity": "disposable_source",
        "base_admission": str(root / base_ref), "named_admission": str(root / named_ref)}
    return SimpleNamespace(root=root, config=cfg, artifact=artifact, bindings=bindings,
        verdict=verdict, admission=admission, original_profiles=effective_profiles)


def register(fixture, verdict=None):
    path = write(fixture.root / "fixture_review" / "verdict.json", verdict or fixture.verdict)
    git(fixture.root, "add", "fixture_review/verdict.json")
    git(fixture.root, "commit", "--quiet", "--allow-empty", "-m", "Disposable independent verdict")
    pin = {"path": "fixture_review/verdict.json", "sha256": sha(path), "commit": git(fixture.root, "rev-parse", "HEAD")}
    registry = fixture.root / audit_module.NATIVE_REVIEW_REGISTRY
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(yaml.safe_dump({"schema_version": "native_predictor_training_review_authority.v1", "verdicts": {COMPONENT: pin}}))
    git(fixture.root, "add", audit_module.NATIVE_REVIEW_REGISTRY)
    git(fixture.root, "commit", "--quiet", "--allow-empty", "-m", "Disposable registry pin")
    return path, registry


@pytest.fixture
def reviewed(retained_component, monkeypatch):
    fixture = retained_component
    register(fixture)
    monkeypatch.setattr(audit_module, "NATIVE_REVIEW_ROOT", fixture.root)
    import Libs.utils.data.raw_data_registry as registry
    monkeypatch.setattr(registry, "load_config", lambda path: deepcopy(fixture.config))
    def forbidden(*args, **kwargs):
        raise AssertionError("metadata verification initialized a neural model")
    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden)
    return fixture


def consume(fixture, binding=None, config=None, admission=None):
    from Libs.utils.evaluation.native_residual_control_evaluation import _collection_selected_predictor
    current, _ = training_profile(fixture.config, "mechanistic_residual", 41, training_component=COMPONENT)
    return _collection_selected_predictor(binding or fixture.bindings[41], artificial=False,
        config=config or current, source_admission=admission or fixture.admission)


def test_component_full_matrix_keeps_global_gates_closed(reviewed):
    audit = reviewed.artifact
    assert len(audit["rows"]) == 6 and audit["required_training_seeds"] == [41, 42, 43]
    assert audit["sufficient_training"] is False and audit["independent_review"] == "PENDING"
    assert all(row["training_subset_percentage"] == 100 for row in audit["rows"])
    assert not next(row for row in method_roster(reviewed.config) if row["method_id"] == "residual_mpc")["software_interface_bound"]
    accepted = consume(reviewed)
    assert accepted["scientific_checkpoint_sufficiency"] and not accepted["boundary"]["formal_qualification"]
    assert audit == audit_module.build_native_audit(reviewed.root, reviewed.config, training_component=COMPONENT)


@pytest.mark.parametrize("fault", ["absent", "uncommitted_registry", "uncommitted_verdict", "wrong_hash",
    "pending", "wrong_scope", "generic_component", "unsupported", "unresolved", "missing_finding", "missing_reference"])
def test_predictor_review_authority_rejects_before_acceptance(reviewed, fault):
    verdict = deepcopy(reviewed.verdict)
    if fault == "pending": verdict["decision"] = "PENDING"
    elif fault == "wrong_scope": verdict["review_scope"] = "whole_twin"
    elif fault == "generic_component": verdict["decision"] = "APPROVED_COMPONENT"
    elif fault in {"unsupported", "unresolved"}: verdict["findings"]["native_objective_learning_curves"]["status"] = fault
    elif fault == "missing_finding": verdict["findings"].pop("architecture_and_uncertainty_objectives")
    elif fault == "missing_reference": verdict["findings"]["measured_resource_and_budget_evidence"]["evidence"] = []
    path, registry = register(reviewed, verdict)
    original_path, original_registry = path.read_bytes(), registry.read_bytes()
    try:
        if fault == "uncommitted_registry": registry.write_bytes(original_registry + b"\n")
        elif fault == "uncommitted_verdict": path.write_bytes(original_path + b"\n")
        elif fault in {"absent", "wrong_hash"}:
            payload = yaml.safe_load(original_registry)
            if fault == "absent": payload["verdicts"] = {}
            else: payload["verdicts"][COMPONENT]["sha256"] = "0" * 64
            registry.write_text(yaml.safe_dump(payload))
            git(reviewed.root, "add", audit_module.NATIVE_REVIEW_REGISTRY)
            git(reviewed.root, "commit", "--quiet", "-m", "Disposable rejecting registry")
        with pytest.raises((ValueError, ValidationError)):
            consume(reviewed)
    finally:
        path.write_bytes(original_path)
        registry.write_bytes(original_registry)


@pytest.mark.parametrize("fault", ["seed", "predictor", "conditioning", "normalization", "source", "profile", "checkpoint_bytes", "audit_bytes"])
def test_reviewed_predictor_rejects_stale_runtime_identities(reviewed, fault):
    binding, admission = deepcopy(reviewed.bindings[41]), deepcopy(reviewed.admission)
    config, _ = training_profile(reviewed.config, "mechanistic_residual", 41, training_component=COMPONENT)
    touched = None
    if fault == "seed": binding["model_identity"]["seed_binding"]["model_seed"] = 42
    elif fault == "predictor": binding["model_identity"]["method"] = "gru"
    elif fault == "conditioning": binding["model_identity"]["conditioning"] = deepcopy(reviewed.bindings[42]["model_identity"]["conditioning"])
    elif fault == "normalization": binding["normalization_id"] = "changed"
    elif fault == "source": admission["source_identity"] = "changed"
    elif fault == "profile": config["native_scientific_training"]["selection"]["minimum_steps"] -= 1
    else:
        touched = Path(binding["checkpoint_path"]) if fault == "checkpoint_bytes" else reviewed.root / "fixture_review/audit.json"
        old = touched.read_bytes(); touched.write_bytes(old + b"\n")
    try:
        with pytest.raises((ValueError, ValidationError)):
            consume(reviewed, binding, config, admission)
    finally:
        if touched: touched.write_bytes(old)


def test_component_metadata_preserves_untagged_frozen_conditioning(reviewed):
    effective, spec = training_profile(reviewed.config, "native_wam", 41)
    original = reviewed.original_profiles[("native_wam", 41)]
    assert "training_component" not in effective["native_profile_identity"]
    assert audit_module.native_fit_contract(effective) == audit_module.native_fit_contract(original)
    manifest = reviewed.root / spec["training_manifest"]
    retained, evidence = audit_module.native_executed_profile(reviewed.root, manifest, effective, spec, Path, audit_module.strict_json)
    assert retained == original and evidence["scientific_contract"] == "MATCH"


def test_component_builder_validator_and_no_automatic_sufficiency(tmp_path, monkeypatch):
    import sys
    from Libs.scripts.builders import build_operational_twin_training_sufficiency as builder
    from Libs.scripts.validators import validate_operational_twin_training_sufficiency as validator
    cfg = load_config(audit_module.NATIVE_CONFIG_PATH)
    root = native_metadata_root(tmp_path / "missing_evidence")
    for module in (builder, validator):
        monkeypatch.setattr(module, "ROOT", root)
        monkeypatch.setattr(module, "load_config_stack", lambda *args: cfg)
    output = tmp_path / "audit"
    args = ["command", "--protocol-scope", "manuscript-native", "--training-component", COMPONENT, "--output-dir", str(output)]
    monkeypatch.setattr(sys, "argv", args)
    assert builder.main() == 2
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    artifact = json.loads(before[audit_module.NATIVE_COMPONENT_OUTPUT_NAME])
    assert artifact["prerequisite_status"] == "INCOMPLETE" and artifact["sufficient_training"] is False
    with pytest.raises(FileExistsError): builder.main()
    monkeypatch.setattr(sys, "argv", args + ["--read-only"])
    assert validator.main() == 0
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


@pytest.mark.parametrize("payload_kind", ["component_audit", "approved_verdict"])
def test_component_approval_cannot_satisfy_whole_matrix_validator(reviewed, tmp_path, monkeypatch, payload_kind):
    import sys
    from Libs.scripts.validators import validate_operational_twin_training_sufficiency as validator
    consume(reviewed)
    monkeypatch.setattr(validator, "ROOT", reviewed.root)
    monkeypatch.setattr(validator, "load_config_stack", lambda *args: deepcopy(reviewed.config))
    whole = audit_module.build_native_audit(reviewed.root, reviewed.config)
    assert whole["sufficient_training"] is False and whole["independent_review"] == "PENDING"
    assert whole["artifact_type"] == audit_module.NATIVE_ARTIFACT_TYPE
    write(tmp_path / audit_module.NATIVE_OUTPUT_NAME, reviewed.artifact if payload_kind == "component_audit" else reviewed.verdict)
    (tmp_path / "report.md").write_text(audit_module.native_report(reviewed.artifact))
    monkeypatch.setattr(sys, "argv", ["command", "--read-only", "--protocol-scope", "manuscript-native",
        "--output-dir", str(tmp_path)])
    with pytest.raises((ValueError, ValidationError)):
        validator.main()


@pytest.mark.parametrize("caller", ["collection", "evaluation"])
@pytest.mark.parametrize("pending", [False, True])
def test_both_residual_callers_transport_only_pinned_component_authority(reviewed, tmp_path, monkeypatch, caller, pending):
    from Libs.utils.evaluation import native_residual_control_evaluation as consumer
    from Libs.utils.model.native_observable_source import MOLECULAR
    config, _ = training_profile(reviewed.config, "mechanistic_residual", 41, training_component=COMPONENT)
    binding = deepcopy(reviewed.bindings[41])
    selected = consume(reviewed)
    if pending:
        verdict = deepcopy(reviewed.verdict); verdict["decision"] = "PENDING"
        register(reviewed, verdict)
    families = config["native_observable_bridge"]["quantity_profile"]["families"]
    partitions = {role: {f"{role}:{family}:{index}": {"sample_uid": f"{role}:{family}:{index}",
        "family_uid": family, "role": "calibration", "calibration_reservation": role}
        for family in families for index in range(512)} for role in ("fit", "check", "threshold")}
    monkeypatch.setattr(consumer, "predose_source_admission", lambda *args: deepcopy(reviewed.admission))
    calls = []
    def private(*args, **kwargs):
        calls.append("private"); raise AssertionError("No private callback belongs in metadata transport tests")
    if caller == "collection":
        monkeypatch.setattr(consumer, "admit_native_residual_calibration", lambda *args: {key: partitions[key] for key in ("fit", "check")})
        monkeypatch.setattr(consumer, "selected_group_definitions", lambda *args: list(partitions["threshold"].values()))
        def evaluate(*args, **kwargs):
            context = kwargs["collection_context"]
            assert context["denominator"]["production_complete"]
            calls.append("selected"); return context
        monkeypatch.setattr(consumer, "evaluate_native_residual_controls", evaluate)
        invoke = lambda: consumer.collect_native_residual_calibration(config, object(), object(), binding,
            object(), object(), {}, tmp_path, reservation="fit", residual_path=Path("unused"), world_factory=private)
    else:
        # A blocked molecular row reaches collection binding but cannot open a
        # private world. It proves authority transport, not calibration success.
        uid = next(key for key, value in partitions["fit"].items() if value["family_uid"] == MOLECULAR)
        declaration = {uid: partitions["fit"][uid]}
        tensor = torch.zeros(1)
        model = SimpleNamespace(parameters=lambda: iter([tensor]), buffers=lambda: iter([]), config=config,
            prediction_kind="offline_one_step_mechanistic_residual_conditional_mean")
        residual = {"law": "disposable_offline_residual", "fit_group_uids": [], "check_group_uids": []}
        monkeypatch.setattr(consumer, "load_native_comparator_residual", lambda *args: residual)
        monkeypatch.setattr(consumer, "NativePublicCNSLoss", lambda *args: object())
        monkeypatch.setattr(consumer, "load_family_registries", lambda *args: {})
        context = {"purpose": consumer.residual_collection_policy(config, "fit")["purpose"], "reservation": "fit",
            "source_admission": deepcopy(reviewed.admission), "selected_predictor": selected,
            "denominator": {"software_fixture_only": False}, "roles": {"fit": [uid], "check": []},
            "threshold_group_uids": []}
        invoke = lambda: consumer.evaluate_native_residual_controls(config, model,
            SimpleNamespace(field_normalization=SimpleNamespace(identity=binding["normalization_id"])), binding,
            SimpleNamespace(group_uids=[uid]), declaration, {"reference_id": "disposable_reference"}, tmp_path,
            initial_index=0, residual_path=Path("unused"), world_factory=private,
            adaptive_readout=True, collection_context=context)
    if pending:
        with pytest.raises(ValueError, match="pending"):
            invoke()
        assert calls == []
    else:
        result = invoke()
        transported = result["selected_predictor"] if caller == "collection" else result["collection_binding"]["selected_predictor"]
        assert transported == selected and "private" not in calls
        if caller == "evaluation":
            assert result["status"] == "incomplete" and not result["full_residual_mpc_eligible"]


def test_predictor_profile_is_full_fraction_internal_dependency():
    config = load_config(audit_module.NATIVE_CONFIG_PATH)
    for seed in (41, 42, 43):
        effective, spec = training_profile(config, "mechanistic_residual", seed, training_component=COMPONENT)
        assert spec["training_owner_method"] == "residual_mpc" and spec["conditioning_method"] == "native_wam"
        assert spec["budget"]["minimum_optimizer_updates"] == 10000
        assert effective["native_scientific_training"]["source_scope"] == "all"
        assert len(effective["native_scientific_training"]["family_scope"]) == 4
    for kwargs in ({"percentage": 50}, {"budget": {"max_steps": 1}}):
        with pytest.raises(ValueError):
            training_profile(config, "mechanistic_residual", 41, training_component=COMPONENT, **kwargs)



def stored_component_report(fixture, tmp_path, monkeypatch):
    """Exercise real report admission; replace only downstream episode decoding."""
    from Libs.utils.evaluation import native_residual_control_evaluation as consumer
    config, _ = training_profile(fixture.config, "mechanistic_residual", 41, training_component=COMPONENT)
    binding = deepcopy(fixture.bindings[41])
    families = config["native_observable_bridge"]["quantity_profile"]["families"]
    required = config["native_adaptive_residual_calibration"]["groups_per_family"]
    declarations = {f"fit:{family}:{index}": {"sample_uid": f"fit:{family}:{index}",
        "family_uid": family, "role": "calibration", "calibration_reservation": "fit"}
        for family in families for index in range(required)}
    monkeypatch.setattr(consumer, "declared_dataset", lambda *args, **kwargs:
        (SimpleNamespace(group_uids=list(declarations)), declarations))
    monkeypatch.setattr(consumer, "predose_source_admission", lambda *args: deepcopy(fixture.admission))
    monkeypatch.setattr(consumer, "selected_group_definitions", lambda *args: list(declarations.values()))
    context = {"purpose": consumer.residual_collection_policy(config, "fit", "adaptive_fit")["purpose"],
        "reservation": "fit", "phase": "adaptive_fit", "selected_predictor": consume(fixture),
        "source_admission": deepcopy(fixture.admission), "roles": {"fit": list(declarations), "check": []},
        "threshold_group_uids": []}
    frozen = consumer._collection_binding(config, binding, {"reference_id": "disposable_reference"},
        {"fit_group_uids": [], "check_group_uids": []}, context)
    frozen_id = consumer.digest(frozen)
    report = {"schema_version": "native_residual_calibration_collection.v1", "status": "collection_complete",
        "reservation": "fit", "binding": binding, "collection_binding": frozen, "collection_binding_id": frozen_id,
        "denominator": {"production_complete": True, "software_fixture_only": False},
        "reference_id": "disposable_reference", "pre_execution_policy_snapshot": {}, "groups": []}
    readout_fields = {key: [] for key in ("region_ids", "compartment_ids", "species_ids", "scales", "observed", "selector", "matrix")}
    for uid, declaration in declarations.items():
        fixed = {"readout": deepcopy(readout_fields), "sample_uids": [uid], "particle_uids": [[]],
            "residual_scales": [], "inverse_normalizer_scales": [], "source_times": [], "time_hours": [],
            "predictor_binding": json.dumps(binding, sort_keys=True, allow_nan=False),
            "normalization_id": binding["normalization_id"]}
        report["groups"].append({"group_uid": uid, "family_uid": declaration["family_uid"],
            "status": "component_complete", "episode_collection_binding": {
                "collection_binding_id": frozen_id, "declaration": consumer._collection_public_config(declaration),
                "adaptive_reference": {"fixed": fixed, "original_actions": []}, "initial_log_weights": [0.]},
            "actual_adaptive_readout_history": {"initial_log_weights": [0.]}})
    # These decoding stubs are downstream of real source, authority, report,
    # denominator and serialized binding checks. They do not simulate histories.
    monkeypatch.setattr(consumer, "NativeCNSReadout", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(consumer, "native_cns_readout", lambda *args: SimpleNamespace(**{key: () for key in readout_fields}))
    monkeypatch.setattr(consumer, "FixedReadoutReference", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(consumer, "AdaptiveReadoutReference", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(consumer, "native_actuation_library", lambda *args: ())
    callbacks = []
    def reload_history(payload, *, expected_reference, expected_report_context):
        uid = expected_report_context["group_uid"]
        assert expected_reference.fixed.sample_uids == (uid,)
        callbacks.append(uid)
        return SimpleNamespace(status="complete", history=("disposable_decoding_marker",), initial_log_weights=(0.,))
    monkeypatch.setattr(consumer, "reload_adaptive_readout_history", reload_history)
    monkeypatch.setattr(consumer, "AdaptiveCalibrationEpisode", lambda uid, reference, history, weights:
        SimpleNamespace(group_uid=uid, reference=reference, history=history, initial_log_weights=weights))
    path = write(tmp_path / "stored_collection.json", report)
    return SimpleNamespace(config=config, path=path, report=report, callbacks=callbacks,
        invoke=lambda: consumer._read_collection_report(path, config, "fit", "adaptive_fit"))


@pytest.mark.parametrize("authority", ["valid", "absent", "pending"])
def test_stored_report_requires_current_typed_authority_before_episode_callbacks(reviewed, tmp_path, monkeypatch, authority):
    stored = stored_component_report(reviewed, tmp_path, monkeypatch)
    if authority == "pending":
        verdict = deepcopy(reviewed.verdict); verdict["decision"] = "PENDING"
        register(reviewed, verdict)
    elif authority == "absent":
        registry = reviewed.root / audit_module.NATIVE_REVIEW_REGISTRY
        registry.write_text(yaml.safe_dump({"schema_version": "native_predictor_training_review_authority.v1", "verdicts": {}}))
        git(reviewed.root, "add", audit_module.NATIVE_REVIEW_REGISTRY)
        git(reviewed.root, "commit", "--quiet", "-m", "Disposable absent report authority")
    if authority != "valid":
        with pytest.raises(ValueError, match="absent|pending"):
            stored.invoke()
        assert stored.callbacks == []
    else:
        report, episodes, identity = stored.invoke()
        assert report == stored.report and len(episodes) == len(report["groups"]) == 2048
        assert len(stored.callbacks) == 2048 and identity["sha256"] == sha(stored.path)
        assert identity["collection_binding_id"] == report["collection_binding_id"]


@pytest.mark.parametrize("change", ["unrelated_commit", "registry", "verdict", "evidence"])
def test_stored_report_uses_stable_pinned_authority_and_rejects_changed_evidence(reviewed, tmp_path, monkeypatch, change):
    stored = stored_component_report(reviewed, tmp_path, monkeypatch)
    accepted = consume(reviewed)
    first_report, first_episodes, first_identity = stored.invoke()
    stored.callbacks.clear()
    previous_head = git(reviewed.root, "rev-parse", "HEAD")
    touched, original = None, None
    try:
        if change == "unrelated_commit":
            note = reviewed.root / "fixture_review/unrelated_note.txt"
            note.write_text("Unrelated disposable repository note.\n")
            git(reviewed.root, "add", "fixture_review/unrelated_note.txt")
            git(reviewed.root, "commit", "--quiet", "--allow-empty", "-m", "Unrelated disposable change")
        elif change == "registry":
            touched = reviewed.root / audit_module.NATIVE_REVIEW_REGISTRY
            original = touched.read_bytes(); touched.write_bytes(original + b"\n")
            git(reviewed.root, "add", audit_module.NATIVE_REVIEW_REGISTRY)
            git(reviewed.root, "commit", "--quiet", "-m", "Changed disposable registry bytes")
        elif change == "verdict":
            verdict = deepcopy(reviewed.verdict)
            verdict["findings"]["failures_and_limitations"]["rationale"] += " Revised independent fixture finding."
            register(reviewed, verdict)
        else:
            row = reviewed.artifact["rows"][0]
            touched = Path(row["training_evidence"]["path"]).parent / "process_resources.log"
            original = touched.read_bytes(); touched.write_bytes(original + b"Changed resource evidence.\n")
        if change == "unrelated_commit":
            assert git(reviewed.root, "rev-parse", "HEAD") != previous_head
            assert consume(reviewed) == accepted
            report, episodes, identity = stored.invoke()
            assert report == first_report and identity == first_identity
            assert len(episodes) == len(first_episodes) == len(stored.callbacks) == 2048
        else:
            with pytest.raises(ValueError, match="authority differ|evidence changed"):
                stored.invoke()
            assert stored.callbacks == []
    finally:
        if touched is not None:
            touched.write_bytes(original)
