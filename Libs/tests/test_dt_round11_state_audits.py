from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.digital_twin_state_audits import AUDIT_KEYS, build_round11_audits, validate_round11_audits, write_round11_audits
from Libs.utils.data.raw_data_registry import load_config


def test_round11_audit_set_is_complete_and_fail_closed() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    audits = build_round11_audits(config, repository_root=REPO_ROOT)
    assert tuple(audits) == AUDIT_KEYS
    for payload in audits.values():
        assert payload["state"] == "in-progress"
        assert payload["evidence_fabric_sealed"] is False
        assert payload["dt_core"] == "not_evaluated"
        assert payload["evidence_materialized"] is False
        assert payload["dt-evidence-fabric-state"] == "in-progress"
        assert payload["typed-state-blocks"] == "implemented"
        assert payload["partial-state-masks"] == "implemented"
        assert payload["evidence-fabric-sealed"] is False
        assert payload["DT_CORE"] == "not_evaluated"
        assert payload["round_exit_status"] == "round11_blocked_full_suite"
    aggregate = audits["legacy_aggregate"]["aggregate_record"]
    assert aggregate["aggregation_method"] == "legacy_aggregate_unknown"
    assert aggregate["source_record_count"] == 222336
    assert aggregate["independent_unit_count"] is None
    assert aggregate["headline_eligible"] is False
    triage = audits["full_suite_failure_triage"]
    assert triage["classification"] == "unresolved"
    assert triage["round_exit_status"] == "round11_blocked_full_suite"
    assert all(triage[field]["status"] == "unavailable" for field in ("failure_scope", "failing_test_ids", "first_known_commit", "owner", "preserved_log"))
    assert triage["result"] == {"passed": 269, "failed": 17, "errors": 16, "duration_seconds": 1327.27}


def _tmp_audit_config(tmp_path: Path) -> dict[str, object]:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    config = deepcopy(config)
    config["digital_twin_canonical_state"]["audit_output_paths"] = {key: str(tmp_path / f"{key}.json") for key in AUDIT_KEYS}
    write_round11_audits(config, build_round11_audits(config, repository_root=REPO_ROOT), repository_root=REPO_ROOT)
    return config


def test_round11_audit_readback_passes_for_generated_outputs(tmp_path: Path) -> None:
    config = _tmp_audit_config(tmp_path)
    report = validate_round11_audits(config, repository_root=REPO_ROOT)
    assert report["status"] == "passed"
    assert report["blocker_count"] == 0


def test_round11_audit_validator_rejects_missing_required_field(tmp_path: Path) -> None:
    config = _tmp_audit_config(tmp_path)
    path = tmp_path / "state_path_inventory.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["entries"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_round11_audits(config, repository_root=REPO_ROOT)
    assert any(item["code"] == "missing_audit_fields" and item["audit"] == "state_path_inventory" for item in report["blockers"])


def test_round11_audit_validator_rejects_missing_required_status(tmp_path: Path) -> None:
    config = _tmp_audit_config(tmp_path)
    path = tmp_path / "execution.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["DT_CORE"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_round11_audits(config, repository_root=REPO_ROOT)
    assert any(item["code"] == "missing_audit_fields" and item["audit"] == "execution" for item in report["blockers"])


def test_round11_audit_validator_rejects_invalid_exit_status(tmp_path: Path) -> None:
    config = _tmp_audit_config(tmp_path)
    path = tmp_path / "execution.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["round_exit_status"] = "round11_unknown"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_round11_audits(config, repository_root=REPO_ROOT)
    assert any(item["code"] == "invalid_round_exit_status" for item in report["blockers"])


def test_round11_audit_validator_rejects_invalid_component_status(tmp_path: Path) -> None:
    config = _tmp_audit_config(tmp_path)
    path = tmp_path / "mask_semantics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["partial-state-masks"] = "unknown"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_round11_audits(config, repository_root=REPO_ROOT)
    assert any(item["code"] == "invalid_partial_mask_status" for item in report["blockers"])


def test_round11_audit_validator_rejects_hash_mismatch_and_private_path(tmp_path: Path) -> None:
    config = _tmp_audit_config(tmp_path)
    path = tmp_path / "state_path_inventory.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["path"] = "/private/forbidden.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_round11_audits(config, repository_root=REPO_ROOT)
    codes = {item["code"] for item in report["blockers"]}
    assert "private_absolute_path_in_audit" in codes
    assert "audit_hash_mismatch" in codes
