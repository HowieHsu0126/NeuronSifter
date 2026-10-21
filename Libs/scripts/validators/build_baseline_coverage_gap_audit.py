from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.raw_data_registry import write_json
from Libs.utils.data.raw_data_registry import write_text_atomic


DEFAULT_JSON_OUTPUT = "Output/audits/baseline_coverage_gap_audit.json"
DEFAULT_MD_OUTPUT = "Output/audits/baseline_coverage_gap_audit.md"
REQUIRED_COMPLETION_ITEMS = {
    "kg_graph_drug_repurposing_baseline": "local matched-split public graph evidence baseline",
    "pdediff_adapter": "paper-grade registration for existing direct wrapper",
}
NOT_RECOMMENDED_LOCAL_REPRODUCTIONS = [
    "TxGNN",
    "KGiA",
    "ExPDrug",
    "REx",
    "HiMAL",
    "DeepDrug",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the AD paper-grade baseline coverage gap audit.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--json-output", default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", default=DEFAULT_MD_OUTPUT)
    return parser.parse_args()


def artifact_path(path_value: object) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return resolved.as_posix()


def registry_baselines(config: dict[str, object]) -> list[dict[str, object]]:
    return [dict(row) for row in dict(config["paper_grade_baseline_registry"])["baselines"]]


def runnable_baselines(config: dict[str, object]) -> list[dict[str, object]]:
    return [dict(row) for row in dict(config["paper_experiments"])["runnable_baselines"]]


def suite_config(config: dict[str, object]) -> dict[str, object]:
    return dict(config["paper_grade_full_suite"])


def coverage_by_category(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    coverage: dict[str, dict[str, object]] = {}
    for row in rows:
        family = str(row["baseline_family"])
        payload = coverage.setdefault(
            family,
            {
                "baseline_ids": [],
                "main_table_runnable_count": 0,
                "appendix_runnable_count": 0,
                "reference_only_count": 0,
            },
        )
        payload["baseline_ids"].append(str(row["baseline_id"]))
        payload["main_table_runnable_count"] = int(payload["main_table_runnable_count"]) + int(bool(row["main_table_runnable"]))
        payload["appendix_runnable_count"] = int(payload["appendix_runnable_count"]) + int(bool(row["appendix_runnable"]))
        payload["reference_only_count"] = int(payload["reference_only_count"]) + int(bool(row["reference_only"]))
    return coverage


def literature_comparators(config: dict[str, object]) -> list[dict[str, object]]:
    comparators = dict(dict(config["paper_experiments"])["baseline_sota_comparators"])
    return [dict(row) for row in comparators.get("task_matched", [])]


def adapter_verdict(config: dict[str, object], adapter_id: str) -> dict[str, object]:
    registry_by_id = {str(row["baseline_id"]): row for row in registry_baselines(config)}
    runnable_by_id = {str(row["baseline_name"]): row for row in runnable_baselines(config)}
    suite = suite_config(config)
    suite_models = dict(suite["models"])
    submission_adapters = {str(value) for value in dict(suite["submission_order"])["baseline_adapters"]}
    registry_row = dict(registry_by_id.get(adapter_id, {}))
    runnable_row = dict(runnable_by_id.get(adapter_id, {}))
    conditions = {
        "runnable_adapter_registered": bool(runnable_row),
        "paper_registry_main_table": bool(registry_row.get("main_table_runnable", False)),
        "paper_registry_appendix_detail": bool(registry_row.get("appendix_runnable", False)),
        "suite_model_registered": adapter_id in suite_models,
        "submission_order_registered": adapter_id in submission_adapters,
    }
    return {
        "adapter_id": adapter_id,
        "verdict": "paper_grade_registered_main_table_runnable" if all(conditions.values()) else "registration_incomplete",
        "conditions": conditions,
        "adapter_mode": str(runnable_row.get("adapter_mode", "")),
        "implementation_status": str(runnable_row.get("implementation_status", "")),
    }


def build_report(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    rows = registry_baselines(config)
    row_by_id = {str(row["baseline_id"]): row for row in rows}
    main_table_list = [str(row["baseline_id"]) for row in rows if bool(row["main_table_runnable"])]
    appendix_list = [str(row["baseline_id"]) for row in rows if bool(row["appendix_runnable"])]
    missing_required = [
        item_id
        for item_id in REQUIRED_COMPLETION_ITEMS
        if item_id not in row_by_id or not bool(row_by_id[item_id].get("main_table_runnable", False))
    ]
    required_item_status = [
        {
            "baseline_id": item_id,
            "required_action": action,
            "status": "present_after_repair" if item_id not in missing_required else "missing",
        }
        for item_id, action in REQUIRED_COMPLETION_ITEMS.items()
    ]
    literature_rows = literature_comparators(config)
    return {
        "audit_name": "baseline_coverage_gap_audit",
        "config_path": repo_relative(artifact_path(config_path)),
        "needs_baseline_completion": True,
        "missing_required_baselines": missing_required,
        "required_completion_items": required_item_status,
        "optional_baselines": [
            {
                "baseline_id": "poseidon_style_pde_foundation_reference",
                "recommendation": "appendix_reference_only_until_a_matched_local_adapter_exists",
            }
        ],
        "not_recommended_local_reproductions": [
            {
                "comparator_name": name,
                "recommendation": "literature_comparator_only_without_matched_split_local_code",
            }
            for name in NOT_RECOMMENDED_LOCAL_REPRODUCTIONS
        ],
        "coverage_by_category": coverage_by_category(rows),
        "main_table_runnable_recommended_list": main_table_list,
        "main_table_runnable_count": len(main_table_list),
        "appendix_runnable_detail_list": appendix_list,
        "appendix_runnable_count": len(appendix_list),
        "literature_comparator_list": [str(row["comparator_name"]) for row in literature_rows],
        "literature_comparators": literature_rows,
        "pdediff_adapter_verdict": adapter_verdict(config, "pdediff_adapter"),
        "wdno_adapter_verdict": adapter_verdict(config, "wdno_adapter"),
        "requires_refreeze_before_live": True,
        "claim_scope": "AD-only intervention-prioritization baseline coverage audit",
    }


def markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Baseline Coverage Gap Audit",
        "",
        f"- needs_baseline_completion: `{report['needs_baseline_completion']}`",
        f"- requires_refreeze_before_live: `{report['requires_refreeze_before_live']}`",
        f"- main_table_runnable_count: `{report['main_table_runnable_count']}`",
        f"- appendix_runnable_count: `{report['appendix_runnable_count']}`",
        "",
        "## Required Completion Items",
        "| baseline_id | status | required_action |",
        "| --- | --- | --- |",
    ]
    for row in report["required_completion_items"]:
        payload = dict(row)
        lines.append(f"| {payload['baseline_id']} | {payload['status']} | {payload['required_action']} |")
    lines.extend(["", "## Main Table Runnable List"])
    lines.extend([f"- `{value}`" for value in report["main_table_runnable_recommended_list"]])
    lines.extend(["", "## Appendix Detail List"])
    lines.extend([f"- `{value}`" for value in report["appendix_runnable_detail_list"]])
    lines.extend(["", "## Literature Comparators"])
    lines.extend([f"- `{value}`" for value in report["literature_comparator_list"]])
    lines.extend(
        [
            "",
            "## Adapter Verdicts",
            f"- pdediff_adapter: `{dict(report['pdediff_adapter_verdict'])['verdict']}`",
            f"- wdno_adapter: `{dict(report['wdno_adapter_verdict'])['verdict']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report = build_report(args.config)
    write_json(artifact_path(args.json_output), report)
    write_text_atomic(artifact_path(args.md_output), markdown_report(report))


if __name__ == "__main__":
    main()
