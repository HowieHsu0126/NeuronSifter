from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from Libs.utils.data.controlled_research_config import (
    assert_controlled_governance_approval,
    assert_controlled_network_credentials,
    load_controlled_research_config,
)
from Libs.utils.data.public_mainline_isolation import (
    build_public_mainline_isolation_reports,
    validate_public_mainline_isolation_payload,
)
from Libs.utils.data import public_mainline_isolation
from Libs.utils.data.raw_data_registry import load_config, load_config_with_provenance, write_tsv
from Libs.scripts.data_acquisition.sync_adni_controlled_access import parse_args as parse_adni_sync_args
from Libs.scripts.builders.build_adni_controlled_registry import parse_args as parse_adni_builder_args
from Libs.scripts.validators.validate_adni_controlled_access import parse_args as parse_adni_validator_args
from Libs.exp.evaluation.run_adni_controlled_state_inference_benchmark import parse_args as parse_adni_benchmark_args


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_CONFIG = REPO_ROOT / "Libs/configs/config_stack.yaml"
CONTROLLED_CONFIG = REPO_ROOT / "Libs/configs/config_stack_controlled_research.yaml"


def blocker_codes(report: dict[str, object]) -> set[str]:
    return {str(row["code"]) for row in report["blockers"]}


def public_context() -> tuple[dict, dict[str, object]]:
    return load_config_with_provenance(PUBLIC_CONFIG)


def test_public_mainline_resolves_zero_controlled_sources_in_every_route_class() -> None:
    graph, audit, remediation = build_public_mainline_isolation_reports(PUBLIC_CONFIG)
    assert graph["status"] == "pass"
    assert graph["controlled_nodes_reachable_count"] == 0
    assert graph["loaded_controlled_sections"] == []
    assert graph["isolated_controlled_overlays"][0]["status"] == "isolated"
    assert graph["isolated_controlled_overlays"][0]["governance_contract_valid"] is True
    assert audit["status"] == "pass"
    assert audit["blocker_count"] == audit["controlled_reachability_count"] == 0
    assert audit["unregistered_config_findings"] == []
    assert all(row["status"] == "excluded_governance_only" for row in audit["governance_context_findings"])
    assert audit["route_class_counts"] == {
        "config": 0,
        "experiment": 0,
        "tool": 0,
        "report_release": 0,
        "anonymous_package": 0,
    }
    assert all(row["status"] == "guarded" for row in audit["controlled_entrypoint_findings"])
    assert remediation["blocker_id"] == "G0-PUBLIC-MAINLINE-CONTROLLED-INCLUDE"
    assert remediation["candidate_status"] == "dedicated_validator_passed"
    assert remediation["dedicated_g0_rerun_required"] is False
    assert remediation["aggregate_g0_rerun_required"] is True
    assert remediation["dedicated_validator_passed"] is True
    assert remediation["original_blocker_closed"] is True
    assert remediation["phase1_open"] is False
    assert remediation["implementation_enabled"] is False
    config = load_config(PUBLIC_CONFIG)
    schemas = config["schemas"]
    assert set(schemas["public_mainline_isolation_contract_fields"]).issubset(config["public_mainline_isolation"])
    controlled_config = load_config(CONTROLLED_CONFIG)
    assert set(schemas["controlled_research_stack_fields"]).issubset(controlled_config["controlled_research_stack"])
    assert set(schemas["public_mainline_config_graph_fields"]).issubset(graph)
    assert set(schemas["public_mainline_isolation_audit_fields"]).issubset(audit)
    assert set(schemas["g0_public_mainline_remediation_report_fields"]).issubset(remediation)
    assert "Libs/configs/model/neuronsifter_scale_registry.yaml" in graph["nodes"]
    assert "Libs/configs/model/mind_dino_scale_registry.yaml" not in graph["nodes"]


def test_direct_and_transitive_controlled_includes_fail_closed(tmp_path: Path) -> None:
    direct = tmp_path / "direct.yaml"
    direct.write_text(
        "config_includes:\n"
        f"- {PUBLIC_CONFIG.as_posix()}\n"
        f"- {(REPO_ROOT / 'Libs/configs/data/adni_controlled_access.yaml').as_posix()}\n",
        encoding="utf-8",
    )
    _, direct_audit, _ = build_public_mainline_isolation_reports(direct)
    assert {"public_entrypoint_mismatch", "public_config_reaches_controlled_node", "public_config_loads_controlled_section"}.issubset(blocker_codes(direct_audit))

    child = tmp_path / "child.yaml"
    child.write_text(
        "config_includes:\n"
        f"- {(REPO_ROOT / 'Libs/configs/data/adni_controlled_access.yaml').as_posix()}\n",
        encoding="utf-8",
    )
    transitive = tmp_path / "transitive.yaml"
    transitive.write_text(
        "config_includes:\n"
        f"- {PUBLIC_CONFIG.as_posix()}\n"
        "- child.yaml\n",
        encoding="utf-8",
    )
    _, transitive_audit, _ = build_public_mainline_isolation_reports(transitive)
    assert {"public_config_reaches_controlled_node", "public_config_loads_controlled_section"}.issubset(blocker_codes(transitive_audit))


def test_environment_include_indirection_fails_closed(tmp_path: Path) -> None:
    stack = tmp_path / "environment.yaml"
    stack.write_text("config_includes:\n- ${CONTROLLED_CONFIG}\n", encoding="utf-8")
    _, audit, remediation = build_public_mainline_isolation_reports(stack)
    assert audit["status"] == remediation["candidate_status"] == "blocked"
    assert "config_include_resolution_failed" in blocker_codes(audit)


@pytest.mark.parametrize(
    ("dotted_path", "value", "route_class"),
    [
        ("paper_experiments.ad_override", {"config": "Libs/configs/data/adni_controlled_access.yaml"}, "experiment"),
        ("data.world_state_ready_registry.default_paths", ["Input/private/adni"], "config"),
        ("screening_pipeline.screening_benchmark.active_learning.tool_input", "Input/private/adni", "tool"),
        ("paper.publication_claim_suite.final_report_commands", ["python report.py --input Input/processed/adni"], "report_release"),
        ("paper.publication_readiness.submission_policy.anonymous_package_contents", ["code", "adni_controlled_access"], "anonymous_package"),
    ],
)
def test_public_route_injection_fails_in_its_route_class(dotted_path: str, value: object, route_class: str) -> None:
    config, provenance = public_context()
    mutated = deepcopy(config)
    current: dict = mutated
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value
    _, audit, _ = validate_public_mainline_isolation_payload(mutated, provenance, PUBLIC_CONFIG)
    assert audit["route_class_counts"][route_class] > 0
    assert "public_route_reaches_controlled_source" in blocker_codes(audit)


def test_wildcard_registry_discovery_fails_closed() -> None:
    config, provenance = public_context()
    mutated = deepcopy(config)
    mutated["paper_experiments"]["wildcard_override"] = {"config": "Libs/configs/data/*.yaml"}
    _, audit, _ = validate_public_mainline_isolation_payload(mutated, provenance, PUBLIC_CONFIG)
    assert "public_route_reaches_controlled_source" in blocker_codes(audit)
    matches = [
        match
        for row in audit["route_findings"]
        for match in row["controlled_matches"]
    ]
    assert any(match["kind"] == "wildcard_controlled_config" for match in matches)


def test_controlled_mapping_key_and_source_class_fail_closed() -> None:
    config, provenance = public_context()
    keyed = deepcopy(config)
    keyed["paper_experiments"]["ADNI"] = {"enabled": True}
    _, keyed_audit, _ = validate_public_mainline_isolation_payload(keyed, provenance, PUBLIC_CONFIG)
    assert keyed_audit["route_class_counts"]["experiment"] > 0
    assert any(
        match["location"] == "mapping_key"
        for row in keyed_audit["route_findings"]
        for match in row["controlled_matches"]
    )

    classified = deepcopy(config)
    classified["paper_experiments"]["controlled_probe"] = {
        "access_class": "controlled_access_participant_data"
    }
    _, classified_audit, _ = validate_public_mainline_isolation_payload(
        classified,
        provenance,
        PUBLIC_CONFIG,
    )
    assert "public_route_reaches_controlled_source" in blocker_codes(classified_audit)


def test_unregistered_public_config_namespace_fails_closed() -> None:
    config, provenance = public_context()
    mutated = deepcopy(config)
    mutated["new_public_release_route"] = {"source": "Input/private/adni"}
    _, audit, _ = validate_public_mainline_isolation_payload(mutated, provenance, PUBLIC_CONFIG)
    assert "unregistered_public_config_reaches_controlled_source" in blocker_codes(audit)
    assert audit["route_class_counts"]["config"] > 0
    assert audit["unregistered_config_findings"][0]["value_path"].startswith(
        "new_public_release_route"
    )


def test_governance_only_contexts_must_resolve_and_remain_non_executable() -> None:
    config, provenance = public_context()
    missing = deepcopy(config)
    missing["public_mainline_isolation"]["governance_only_contexts"].append(
        {
            "dotted_path": "missing.governance.path",
            "context_role": "controlled_exclusion_policy",
            "executable": False,
            "public_evidence_allowed": False,
        }
    )
    _, missing_audit, _ = validate_public_mainline_isolation_payload(
        missing,
        provenance,
        PUBLIC_CONFIG,
    )
    assert "missing_governance_context" in blocker_codes(missing_audit)

    executable = deepcopy(config)
    executable["public_mainline_isolation"]["governance_only_contexts"][0]["executable"] = True
    _, executable_audit, _ = validate_public_mainline_isolation_payload(
        executable,
        provenance,
        PUBLIC_CONFIG,
    )
    assert "invalid_governance_context_metadata" in blocker_codes(executable_audit)


def test_static_public_default_path_and_wildcard_import_fail_closed(tmp_path: Path) -> None:
    config, provenance = public_context()
    route = tmp_path / "public_route.py"
    route.write_text(
        'from public_registry import *\nDEFAULT_INPUT = "Input/private/adni"\n',
        encoding="utf-8",
    )
    mutated = deepcopy(config)
    contract = mutated["public_mainline_isolation"]
    contract["static_scan_roots"] = [str(tmp_path)]
    contract["controlled_entrypoints"] = []
    contract["governance_only_static_entrypoints"] = []
    _, audit, _ = validate_public_mainline_isolation_payload(mutated, provenance, PUBLIC_CONFIG)
    assert "static_public_route_indirection" in blocker_codes(audit)
    assert audit["static_route_findings"] == [
        {
            "path": str(route),
            "route_class": "config",
            "status": "blocked",
            "matches": ["controlled_literal", "wildcard_import"],
            "controlled_literals": ["ADNI", "Input/private/adni"],
        }
    ]
    assert audit["route_class_counts"]["config"] > 0


def test_static_environment_and_yaml_registry_discovery_fail_closed(tmp_path: Path) -> None:
    config, provenance = public_context()
    route = tmp_path / "environment_registry.py"
    route.write_text(
        'import os\nfrom pathlib import Path\nCONFIG = os.environ.get("CONTROLLED_CONFIG")\n'
        'CONFIGS = list(Path("Libs/configs/data").glob("*.yaml"))\n',
        encoding="utf-8",
    )
    mutated = deepcopy(config)
    contract = mutated["public_mainline_isolation"]
    contract["static_scan_roots"] = [str(tmp_path)]
    contract["controlled_entrypoints"] = []
    contract["governance_only_static_entrypoints"] = []
    _, audit, _ = validate_public_mainline_isolation_payload(mutated, provenance, PUBLIC_CONFIG)
    assert "static_public_route_indirection" in blocker_codes(audit)
    assert audit["static_route_findings"][0]["matches"] == [
        "environment_conditioned_config_default",
        "registry_or_wildcard_discovery",
    ]
    assert audit["controlled_reachability_count"] > 0


def test_static_scan_skips_external_references_but_scans_project_routes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    external_route = repo_root / "Libs/external/reference/wildcard_reference.py"
    project_route = repo_root / "Libs/scripts/public_route.py"
    external_route.parent.mkdir(parents=True)
    project_route.parent.mkdir(parents=True)
    external_route.write_text("from reference_registry import *\n", encoding="utf-8")
    project_route.write_text("from public_registry import *\n", encoding="utf-8")
    contract = {
        "controlled_entrypoints": [],
        "governance_only_static_entrypoints": [],
        "static_scan_roots": ["Libs"],
        "static_route_class_prefixes": {"tool": ["Libs/scripts"]},
        "controlled_source_identifiers": [],
        "controlled_source_classes": [],
        "controlled_boolean_keys": [],
        "controlled_path_roots": [],
        "controlled_config_paths": [],
    }
    monkeypatch.setattr(public_mainline_isolation, "REPO_ROOT", repo_root)
    public_mainline_isolation._cached_static_route_findings.cache_clear()
    try:
        findings, blockers = public_mainline_isolation._static_route_findings(contract)
    finally:
        public_mainline_isolation._cached_static_route_findings.cache_clear()
    assert blockers == [
        {
            "code": "static_public_route_indirection",
            "message": "A public code route has uncontrolled config or import indirection.",
            "context": {
                "path": "Libs/scripts/public_route.py",
                "route_class": "tool",
                "status": "blocked",
                "matches": ["wildcard_import"],
                "controlled_literals": [],
            },
        }
    ]
    assert findings == [
        {
            "path": "Libs/scripts/public_route.py",
            "route_class": "tool",
            "status": "blocked",
            "matches": ["wildcard_import"],
            "controlled_literals": [],
        }
    ]


def test_controlled_overlay_requires_explicit_invocation_and_governance(tmp_path: Path) -> None:
    public = load_config(PUBLIC_CONFIG)
    with pytest.raises(ValueError, match="explicit --config"):
        load_controlled_research_config(PUBLIC_CONFIG)
    controlled = load_controlled_research_config(CONTROLLED_CONFIG)
    assert "adni_controlled_access" in controlled
    metadata = controlled["controlled_research_stack"]
    assert metadata["non_public"] is True
    assert metadata["fail_closed_without_approval"] is True
    assert metadata["fail_closed_without_credentials"] is True
    assert metadata["public_fallback_allowed"] is False
    assert metadata["synthetic_fallback_allowed"] is False
    with pytest.raises(FileNotFoundError, match="approval evidence"):
        assert_controlled_governance_approval(controlled)
    authenticated_http = controlled["adni_controlled_access"]["sync"]["authenticated_http"]
    authenticated_http["cookie_jar_path"] = str(tmp_path / "missing-cookie.jar")
    authenticated_http["candidate_cookie_jar_paths"] = []
    with pytest.raises(RuntimeError, match="cookie jar"):
        assert_controlled_network_credentials(controlled)
    assert "adni_controlled_access" not in public


@pytest.mark.parametrize(
    "parser",
    [parse_adni_sync_args, parse_adni_builder_args, parse_adni_validator_args, parse_adni_benchmark_args],
)
def test_controlled_cli_requires_explicit_config(monkeypatch: pytest.MonkeyPatch, parser) -> None:
    monkeypatch.setattr("sys.argv", ["controlled_command.py"])
    with pytest.raises(SystemExit):
        parser()


def test_tsv_writer_is_lf_only_and_byte_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "audit.tsv"
    rows = [{"name": "one", "status": "pass"}, {"name": "two", "status": "pass"}]
    write_tsv(path, ["name", "status"], rows)
    first = path.read_bytes()
    first_hash = hashlib.sha256(first).hexdigest()
    write_tsv(path, ["name", "status"], rows)
    second = path.read_bytes()
    assert b"\r\n" not in second
    assert first == second
    assert first_hash == hashlib.sha256(second).hexdigest()
