from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.validators.validate_formal_experiment_readiness import OVERCLAIMED
from Libs.scripts.validators.validate_formal_experiment_readiness import build_report
from Libs.scripts.validators.validate_formal_experiment_readiness import build_requirement_rows
from Libs.scripts.validators.validate_formal_experiment_readiness import markdown_report
from Libs.utils.data.raw_data_registry import load_config


def requirement_by_id(report: dict[str, object], requirement_id: str) -> dict[str, object]:
    rows = {
        str(row["requirement_id"]): row
        for row in report["requirements"]
    }
    return rows[requirement_id]


def test_formal_readiness_report_contains_core_matrix_and_g2_g5_status() -> None:
    report = build_report(REPO_ROOT / "Libs/configs/config_stack.yaml")
    requirement_ids = {str(row["requirement_id"]) for row in report["requirements"]}
    assert {
        "WM-001",
        "DATA-005",
        "NEURONSIFTER-001",
        "DOC-001",
        "REPORT-001",
        "ACTION-001",
        "TRAIN-001",
        "VALID-002",
        "SERVE-003",
        "SLURM-002",
    } <= requirement_ids
    assert report["g2_gate"] == "g2_100k"
    assert report["g5_status"] == "G5_not_passed"
    assert report["current_full_microenvironment_trajectory_count"] < report["g5_store_trajectory_min_count"]
    assert report["current_mesoscopic_trajectory_count"] < report["g5_store_trajectory_min_count"]
    data_row = requirement_by_id(report, "DATA-005")
    assert "full_count=" in ";".join(data_row["artifact_evidence"])
    report_row = requirement_by_id(report, "REPORT-001")
    assert report_row["status"] == "implemented"


def test_formal_readiness_detects_g5_overclaim(tmp_path: Path) -> None:
    config = deepcopy(load_config(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    report_dir = tmp_path / "paper_reports"
    report_dir.mkdir()
    config["paper"]["publication_readiness"]["report_outputs"]["report_dir"] = str(report_dir)
    (report_dir / "g5_effective_sample_audit.json").write_text(
        json.dumps({"gate_status": "passed"}),
        encoding="utf-8",
    )
    rows = {
        item.requirement_id: item
        for item in build_requirement_rows(config)
    }
    assert rows["DATA-005"].status == OVERCLAIMED


def test_formal_readiness_markdown_renders_required_columns() -> None:
    report = build_report(REPO_ROOT / "Libs/configs/config_stack.yaml")
    rendered = markdown_report(report)
    assert "| requirement_id | status | severity | requirement | artifact_evidence | required_fix |" in rendered
    assert "DATA-005" in rendered
    assert "G5_not_passed" in rendered


def test_publication_submission_runs_formal_readiness_preflight() -> None:
    script_text = (REPO_ROOT / "Libs/exp/submission/submit_publication_claim_suite.sh").read_text(encoding="utf-8")
    assert "validate_simulation_readiness_audit.py" in script_text
    assert "validate_supplemental_milestone_contracts.py" in script_text
    assert "validate_formal_experiment_readiness.py" in script_text
    assert "validate_formal_experiment_freeze.py" in script_text
    assert "--no-write" in script_text
