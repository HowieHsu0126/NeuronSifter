from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import ensure_directory
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.raw_data_registry import write_json
from Libs.utils.data.raw_data_registry import write_text_atomic


SEED_PATTERN = re.compile(r"seed[_-](\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrospective published AD intervention-prioritization consistency.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--model-key", default="neuronsifter_mainline")
    parser.add_argument("--allow-pending", action="store_true")
    parser.add_argument("--json-output", default="")
    parser.add_argument("--md-output", default="")
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


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def suite_config(config: dict[str, object]) -> dict[str, object]:
    payload = config.get("paper_grade_full_suite", {})
    if not isinstance(payload, dict) or not payload:
        raise KeyError("Missing paper_grade_full_suite config.")
    return payload


def seed_from_path(path: Path) -> int:
    match = SEED_PATTERN.search(path.as_posix())
    if match is None:
        return -1
    return int(match.group(1))


def result_catalog_paths(config: dict[str, object], model_key: str) -> list[tuple[int, Path]]:
    suite = suite_config(config)
    model_payload = dict(suite["models"][model_key])
    template = str(model_payload["result_catalog_template"])
    paths = []
    for seed in [int(value) for value in suite["formal_seed_values"]]:
        path = artifact_path(template.format(seed=seed))
        paths.append((seed, path))
    return paths


def metric_values(rows: list[dict[str, str]], evaluation_scope: str, metric_name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        if str(row.get("evaluation_scope", "")) != evaluation_scope:
            continue
        if str(row.get("metric_name", "")) != metric_name:
            continue
        values.append(float(row["metric_value"]))
    return values


def any_metric_values(rows: list[dict[str, str]], metric_names: list[str]) -> list[float]:
    values: list[float] = []
    metric_name_set = set(metric_names)
    for row in rows:
        if str(row.get("metric_name", "")) not in metric_name_set:
            continue
        values.append(float(row["metric_value"]))
    return values


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    current_mean = mean(values)
    assert current_mean is not None
    variance = sum((value - current_mean) ** 2 for value in values) / (len(values) - 1)
    return float(math.sqrt(variance))


def ci95(values: list[float]) -> list[float | None]:
    current_mean = mean(values)
    current_std = std(values)
    if current_mean is None or current_std is None:
        return [None, None]
    half_width = 1.96 * current_std / math.sqrt(max(len(values), 1))
    return [float(current_mean - half_width), float(current_mean + half_width)]


def summarize_values(values: list[float]) -> dict[str, object]:
    return {
        "values": values,
        "mean": mean(values),
        "std": std(values),
        "ci95": ci95(values),
        "status": "available" if values else "pending",
    }


def per_seed_metrics(seed: int, rows: list[dict[str, str]]) -> dict[str, object]:
    direction_hit_values = metric_values(rows, "quantitative_case_validation", "direction_hit")
    rank_hit_values = metric_values(rows, "quantitative_case_validation", "rank_recovery_hit")
    tri_values = metric_values(rows, "quantitative_case_validation", "tri_gate_consistency")
    effect_error_values = metric_values(rows, "quantitative_case_validation", "standardized_effect_error")
    pairwise_values = any_metric_values(rows, ["pairwise_anchor_accuracy", "rank_recovery_hit"])
    pkpd_values = any_metric_values(rows, ["pkpd_consistency", "pk_feasibility_top_10", "occupancy_nrmse"])
    expert_values = any_metric_values(rows, ["expert_activation", "expert_utilization", "router_entropy"])
    mixed_case_calibration = None
    if effect_error_values:
        mixed_case_calibration = max(0.0, 1.0 - min(1.0, mean(effect_error_values) or 0.0))
    return {
        "seed": seed,
        "direction_hit_rate": mean(direction_hit_values),
        "pairwise_anchor_accuracy": mean(pairwise_values),
        "rank_ndcg": mean(rank_hit_values),
        "mixed_case_calibration": mixed_case_calibration,
        "tri_consistency": mean(tri_values),
        "pkpd_consistency": mean(pkpd_values),
        "expert_activation": mean(expert_values),
        "case_row_count": sum(1 for row in rows if str(row.get("evaluation_scope", "")) == "quantitative_case_validation"),
    }


def aggregate_metric(per_seed_rows: list[dict[str, object]], metric_name: str) -> dict[str, object]:
    values = [
        float(row[metric_name])
        for row in per_seed_rows
        if row.get(metric_name) is not None
    ]
    return summarize_values(values)


def build_report(config_path: str | Path, model_key: str = "neuronsifter_mainline", allow_pending: bool = False) -> dict[str, object]:
    config = load_config(config_path)
    suite = suite_config(config)
    output_config = dict(suite["published_ad_sift_reproduction"])
    catalog_paths = result_catalog_paths(config, model_key)
    per_seed_rows: list[dict[str, object]] = []
    missing_catalogs = []
    source_catalogs = []
    for seed, path in catalog_paths:
        if not path.exists():
            missing_catalogs.append(repo_relative(path))
            continue
        rows = read_tsv_rows(path)
        if seed < 0:
            seed = seed_from_path(path)
        source_catalogs.append(repo_relative(path))
        per_seed_rows.append(per_seed_metrics(seed, rows))
    if not per_seed_rows and not allow_pending:
        raise FileNotFoundError(
            "No published AD reproduction result catalogs are available for model "
            f"{model_key}: " + ", ".join(missing_catalogs)
        )
    metrics = {
        metric_name: aggregate_metric(per_seed_rows, metric_name)
        for metric_name in [
            "direction_hit_rate",
            "pairwise_anchor_accuracy",
            "rank_ndcg",
            "mixed_case_calibration",
            "tri_consistency",
            "pkpd_consistency",
            "expert_activation",
        ]
    }
    available_required_metrics = [
        metric_name
        for metric_name, payload in metrics.items()
        if payload["status"] == "available"
    ]
    status = "available" if {"direction_hit_rate", "pairwise_anchor_accuracy", "rank_ndcg"} <= set(available_required_metrics) else "pending"
    return {
        "audit_name": "published_ad_sift_reproduction",
        "model_key": model_key,
        "claim_scope": "retrospective literature-direction recovery and intervention-prioritization consistency",
        "forbidden_claims": list(output_config["claim_language"]["forbidden"]),
        "status": status,
        "source_catalogs": source_catalogs,
        "missing_catalogs": missing_catalogs,
        "per_seed": per_seed_rows,
        "metrics": metrics,
        "result_language": {
            "allowed": list(output_config["claim_language"]["allowed"]),
            "statement": "Metrics summarize retrospective literature-direction recovery from formal result catalogs only.",
        },
    }


def markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Published AD Sift Reproduction Audit",
        "",
        f"- status: `{report['status']}`",
        f"- model_key: `{report['model_key']}`",
        f"- claim_scope: `{report['claim_scope']}`",
        "",
        "| metric | status | mean | std | ci95 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for metric_name, payload in report["metrics"].items():
        ci = payload["ci95"]
        ci_text = "pending" if ci == [None, None] else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        mean_text = "pending" if payload["mean"] is None else f"{payload['mean']:.4f}"
        std_text = "pending" if payload["std"] is None else f"{payload['std']:.4f}"
        lines.append(f"| {metric_name} | {payload['status']} | {mean_text} | {std_text} | {ci_text} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_config = suite_config(config)["published_ad_sift_reproduction"]
    json_output = args.json_output or str(output_config["audit_json"])
    md_output = args.md_output or str(output_config["audit_md"])
    report = build_report(args.config, args.model_key, args.allow_pending)
    write_json(artifact_path(json_output), report)
    write_text_atomic(artifact_path(md_output), markdown_report(report))
    output_root = artifact_path(output_config["output_root"])
    ensure_directory(output_root)
    write_json(output_root / "published_ad_sift_reproduction_summary.json", report)
    if report["status"] != "available" and not args.allow_pending:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
