from __future__ import annotations

import argparse
import csv
import math
import subprocess
from collections.abc import Iterator
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.benchmark_datasets import benchmark_program_config, load_benchmark_catalogs
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_tsv
from Libs.utils.exp.runnable_baselines import baseline_entry


EXTERNAL_EVIDENCE_BASELINE_NAME = "external_evidence_only"
KG_GRAPH_BASELINE_NAME = "kg_graph_drug_repurposing_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a project-owned runnable baseline across benchmark folds.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--disease-program", default="ad_control")
    parser.add_argument("--seed", default="")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fold-names", nargs="*", default=None)
    return parser.parse_args()


def seeded_output_root(config: dict, disease_program: str, baseline_name: str, seed_value: str) -> Path:
    root = (
        REPO_ROOT
        / str(config["baseline_adapter_registry"]["output_roots"]["runnable_baseline_root"])
        / disease_program
        / baseline_name
    )
    if seed_value:
        return root / f"seed_{int(seed_value)}"
    return root


def direction_label(score: float, threshold: float) -> int:
    return 1 if score >= threshold else 0


def pairwise_anchor_accuracy(predicted_scores: dict[str, float], reference_scores: dict[str, float]) -> float:
    labels = sorted(predicted_scores)
    comparable_count = 0
    correct_count = 0
    for left_index, left_label in enumerate(labels):
        for right_label in labels[left_index + 1:]:
            predicted_delta = predicted_scores[left_label] - predicted_scores[right_label]
            reference_delta = reference_scores[left_label] - reference_scores[right_label]
            if predicted_delta == 0.0 or reference_delta == 0.0:
                continue
            comparable_count += 1
            if predicted_delta * reference_delta > 0.0:
                correct_count += 1
    if comparable_count == 0:
        return 0.0
    return float(correct_count / comparable_count)


def ndcg_at_k(predicted_scores: dict[str, float], reference_scores: dict[str, float], k_value: int) -> float:
    ranked_labels = sorted(predicted_scores, key=lambda label: predicted_scores[label], reverse=True)[:k_value]
    ideal_labels = sorted(reference_scores, key=lambda label: reference_scores[label], reverse=True)[:k_value]

    def discounted_gain(labels: list[str]) -> float:
        gain = 0.0
        for index, label in enumerate(labels):
            relevance = max(float(reference_scores[label]), 0.0)
            gain += relevance / math.log2(index + 2)
        return gain

    ideal_gain = discounted_gain(ideal_labels)
    if ideal_gain == 0.0:
        return 0.0
    return float(discounted_gain(ranked_labels) / ideal_gain)


def iter_tsv_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def numeric_cell(row: dict[str, str], key: str) -> float:
    value = str(row.get(key, "")).strip()
    if not value:
        return 0.0
    return float(value)


def split_uid_set(value: object) -> list[str]:
    return [item for item in str(value).split("|") if item]


def required_kg_value(entry: dict[str, object], key: str) -> object:
    scoring = dict(entry.get("kg_graph_scoring", {}))
    if key not in scoring:
        raise KeyError(f"{entry['baseline_name']} is missing kg_graph_scoring.{key}")
    return scoring[key]


def best_target_evidence_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in iter_tsv_rows(path):
        target_uid = str(row["target_uid"]).strip()
        score = numeric_cell(row, "evidence_score")
        scores[target_uid] = max(scores.get(target_uid, 0.0), score)
    return scores


def affinity_to_graph_score(affinity_value: float, affinity_floor_nm: float) -> float:
    if affinity_floor_nm <= 0.0:
        raise ValueError("kg_graph_scoring.affinity_floor_nm must be positive.")
    if affinity_value == 0.0:
        return 0.0
    if affinity_value < 0.0:
        raise ValueError("Drug-target affinity values must be non-negative.")
    scaled_affinity = max(affinity_value, affinity_floor_nm) / affinity_floor_nm
    return float(1.0 / (1.0 + math.log10(scaled_affinity)))


def best_edge_scores(path: Path, affinity_floor_nm: float) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    evidence_scores: dict[tuple[str, str], float] = {}
    affinity_scores: dict[tuple[str, str], float] = {}
    for row in iter_tsv_rows(path):
        compound_uid = str(row["compound_uid"]).strip()
        target_uid = str(row["target_uid"]).strip()
        key = (compound_uid, target_uid)
        evidence_scores[key] = max(evidence_scores.get(key, 0.0), numeric_cell(row, "evidence_score"))
        affinity_scores[key] = max(
            affinity_scores.get(key, 0.0),
            affinity_to_graph_score(numeric_cell(row, "affinity_value"), affinity_floor_nm),
        )
    return evidence_scores, affinity_scores


def mean_value(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def weighted_graph_score(
    target_score: float,
    edge_evidence_score: float,
    edge_affinity_score: float,
    regimen_graph_score: float,
    scoring: dict[str, object],
) -> float:
    weights = {
        "target_evidence_weight": float(scoring["target_evidence_weight"]),
        "edge_evidence_weight": float(scoring["edge_evidence_weight"]),
        "edge_affinity_weight": float(scoring["edge_affinity_weight"]),
        "regimen_graph_aggregate_weight": float(scoring["regimen_graph_aggregate_weight"]),
    }
    total_weight = sum(weights.values())
    if total_weight <= 0.0:
        raise ValueError("KG graph scoring weights must sum to a positive value.")
    weighted_sum = (
        target_score * weights["target_evidence_weight"]
        + edge_evidence_score * weights["edge_evidence_weight"]
        + edge_affinity_score * weights["edge_affinity_weight"]
        + regimen_graph_score * weights["regimen_graph_aggregate_weight"]
    )
    return float(weighted_sum / total_weight)


def kg_ranked_regimen_rows(
    entry: dict[str, object],
    disease_program: str,
    seed_value: str,
    config: dict[str, object],
) -> list[dict[str, object]]:
    inputs = dict(entry["kg_graph_inputs"])
    scoring = dict(entry["kg_graph_scoring"])
    target_scores = best_target_evidence_scores(REPO_ROOT / str(inputs["target_evidence_catalog"]))
    edge_evidence_scores, edge_affinity_scores = best_edge_scores(
        REPO_ROOT / str(inputs["drug_target_edge_catalog"]),
        float(required_kg_value(entry, "affinity_floor_nm")),
    )
    rows: list[dict[str, object]] = []
    for row in iter_tsv_rows(REPO_ROOT / str(inputs["regimen_catalog"])):
        compounds = split_uid_set(row["compound_uid_set"])
        targets = split_uid_set(row["target_uid_set"])
        target_score = mean_value([target_scores.get(target_uid, 0.0) for target_uid in targets])
        edge_keys = [(compound_uid, target_uid) for compound_uid in compounds for target_uid in targets]
        edge_evidence_score = mean_value([edge_evidence_scores.get(key, 0.0) for key in edge_keys])
        edge_affinity_score = mean_value([edge_affinity_scores.get(key, 0.0) for key in edge_keys])
        regimen_graph_score = mean_value(
            [
                numeric_cell(row, "aggregate_target_evidence_score"),
                numeric_cell(row, "aggregate_affinity_score"),
            ]
        )
        rows.append(
            {
                "regimen_uid": row["regimen_uid"],
                "disease_program": disease_program,
                "seed": seed_value or "deterministic",
                "graph_prior_score": weighted_graph_score(
                    target_score,
                    edge_evidence_score,
                    edge_affinity_score,
                    regimen_graph_score,
                    scoring,
                ),
                "target_evidence_score": target_score,
                "edge_evidence_score": edge_evidence_score,
                "edge_affinity_score": edge_affinity_score,
                "regimen_graph_aggregate_score": regimen_graph_score,
                "compound_count": len(compounds),
                "target_count": len(targets),
                "source_evidence": "target_evidence_catalog|drug_target_edge_catalog|regimen_catalog",
            }
        )
    return sorted(rows, key=lambda payload: (-float(payload["graph_prior_score"]), str(payload["regimen_uid"])))


def write_kg_graph_drug_repurposing_baseline(config: dict, entry: dict[str, object], disease_program: str, seed_value: str, output_root: Path) -> None:
    ranked_rows = kg_ranked_regimen_rows(entry, disease_program, seed_value, config)
    scoring = dict(entry["kg_graph_scoring"])
    top_k = int(scoring["top_k_regimens"])
    ranked_path = output_root / "kg_graph_ranked_regimens.tsv"
    write_tsv(ranked_path, config["schemas"]["kg_graph_baseline_ranked_fields"], ranked_rows)
    top_rows = ranked_rows[:top_k]
    metric_values = {
        "graph_prior_score": mean_value([float(row["graph_prior_score"]) for row in top_rows]),
        "mean_graph_prior_score": mean_value([float(row["graph_prior_score"]) for row in ranked_rows]),
        "target_evidence_coverage": mean_value([float(float(row["target_evidence_score"]) > 0.0) for row in ranked_rows]),
        "edge_evidence_coverage": mean_value([float(float(row["edge_evidence_score"]) > 0.0) for row in ranked_rows]),
        "edge_affinity_coverage": mean_value([float(float(row["edge_affinity_score"]) > 0.0) for row in ranked_rows]),
        "ranked_regimen_count": float(len(ranked_rows)),
    }
    seed_text = str(seed_value) if seed_value else "deterministic"
    result_rows = [
        {
            "benchmark_name": f"{disease_program}_kg_graph_drug_repurposing_seed_{seed_text}",
            "disease_program": disease_program,
            "baseline_family": str(entry["baseline_family"]),
            "split_name": str(scoring["split_name"]),
            "evaluation_scope": str(scoring["evaluation_scope"]),
            "task_name": str(scoring["task_name"]),
            "task_family": str(scoring["task_family"]),
            "model_name": KG_GRAPH_BASELINE_NAME,
            "modality": "public_knowledge_graph",
            "sample_count": len(ranked_rows),
            "metric_name": metric_name,
            "metric_value": metric_value,
            "artifact_path": str(ranked_path),
            "case_study_name": "",
            "scale_classification": "",
            "is_case_study_anchor": "no",
            "note": "public_graph_evidence_only_no_neuronsifter_rollout_no_tri_no_endpoint_labels",
        }
        for metric_name, metric_value in metric_values.items()
    ]
    write_tsv(output_root / "result_catalog.tsv", config["schemas"]["screening_result_fields"], result_rows)


def write_external_evidence_only_baseline(config: dict, disease_program: str, seed_value: str, output_root: Path) -> None:
    comparison_config = config["screening_pipeline"]["screening_benchmark"]["case_study_comparison"]
    threshold = float(config["paper_grade_full_suite"]["external_evidence_only_baseline"]["direction_threshold"])
    score_matrix = dict(comparison_config["anchor_score_matrix"])
    predicted_scores = {
        str(label): float(payload["external_evidence_only"])
        for label, payload in score_matrix.items()
    }
    reference_scores = {
        str(label): float(payload["endpoint_direction_score"])
        for label, payload in score_matrix.items()
    }
    direction_hits = [
        float(direction_label(predicted_scores[label], threshold) == direction_label(reference_scores[label], threshold))
        for label in sorted(predicted_scores)
    ]
    metrics = {
        "direction_hit": float(sum(direction_hits) / len(direction_hits)) if direction_hits else 0.0,
        "pairwise_anchor_accuracy": pairwise_anchor_accuracy(predicted_scores, reference_scores),
        "rank_recovery_hit": ndcg_at_k(predicted_scores, reference_scores, len(predicted_scores)),
    }
    ensure_rows: list[dict[str, object]] = []
    seed_text = str(seed_value) if seed_value else "deterministic"
    for metric_name, metric_value in metrics.items():
        ensure_rows.append(
            {
                "benchmark_name": f"{disease_program}_external_evidence_only_seed_{seed_text}",
                "disease_program": disease_program,
                "baseline_family": "evidence_adapter",
                "split_name": "source_holdout",
                "evaluation_scope": "quantitative_case_validation",
                "task_name": "published_ad_sift_reproduction",
                "task_family": "retrospective_literature_direction_recovery",
                "model_name": EXTERNAL_EVIDENCE_BASELINE_NAME,
                "modality": "literature_aggregate",
                "sample_count": len(score_matrix),
                "metric_name": metric_name,
                "metric_value": metric_value,
                "artifact_path": str(output_root / "result_catalog.tsv"),
                "case_study_name": "ad_literature_anchor_matrix",
                "scale_classification": "",
                "is_case_study_anchor": "yes",
                "note": "configured_external_evidence_only_retrospective_direction_bridge",
            }
        )
    write_tsv(output_root / "result_catalog.tsv", config["schemas"]["screening_result_fields"], ensure_rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["benchmark_registry"].update(benchmark_program_config(config, args.disease_program))
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    output_root = seeded_output_root(config, args.disease_program, args.baseline_name, args.seed)
    if args.baseline_name == EXTERNAL_EVIDENCE_BASELINE_NAME:
        write_external_evidence_only_baseline(config, args.disease_program, args.seed, output_root)
        complete_script_logging(
            pipeline_logger,
            pipeline_started_at,
            baseline_name=args.baseline_name,
            disease_program=args.disease_program,
            seed=args.seed or "deterministic",
            artifact=output_root / "result_catalog.tsv",
        )
        return
    entry = baseline_entry(config, args.baseline_name)
    if args.baseline_name == KG_GRAPH_BASELINE_NAME:
        write_kg_graph_drug_repurposing_baseline(config, entry, args.disease_program, args.seed, output_root)
        complete_script_logging(
            pipeline_logger,
            pipeline_started_at,
            baseline_name=args.baseline_name,
            disease_program=args.disease_program,
            seed=args.seed or "deterministic",
            artifact=output_root / "result_catalog.tsv",
        )
        return
    _cohort_rows, fold_rows, dataset_rows = load_benchmark_catalogs(config, args.disease_program)
    required_test_modality = str(entry.get("fold_contract", {}).get("required_test_modality", "")).strip()
    target_folds = [
        row["fold_name"]
        for row in fold_rows
        if args.fold_names is None or row["fold_name"] in args.fold_names
    ]
    if required_test_modality:
        target_folds = [
            fold_name
            for fold_name in target_folds
            if any(
                row["fold_name"] == fold_name
                and row["dataset_role"] == "test"
                and row["modality"] == required_test_modality
                and int(row["sample_count"]) > 0
                for row in dataset_rows
            )
        ]
    for fold_name in target_folds:
        result_catalog_path = output_root / fold_name / "result_catalog.tsv"
        if args.skip_existing and result_catalog_path.exists():
            pipeline_logger.progress("fold_skipped", fold_name=fold_name, reason="existing_result_catalog")
            continue
        for command_name in ["prepare_inputs", "train_baseline", "evaluate_baseline"]:
            pipeline_logger.progress("fold_command_started", fold_name=fold_name, command=command_name, baseline_name=args.baseline_name)
            subprocess.run(
                [
                    sys.executable,
                    "Libs/exp/baselines/run_runnable_baseline.py",
                    "--config",
                    args.config,
                    "--baseline-name",
                    args.baseline_name,
                    "--command",
                    command_name,
                    "--disease-program",
                    args.disease_program,
                    "--fold-name",
                    fold_name,
                    "--baseline-output-dir",
                    output_root.as_posix(),
                ],
                check=True,
                cwd=REPO_ROOT,
            )
            pipeline_logger.progress("fold_command_completed", fold_name=fold_name, command=command_name, baseline_name=args.baseline_name)

    aggregate_rows = []
    for fold_name in target_folds:
        result_catalog_path = output_root / fold_name / "result_catalog.tsv"
        if result_catalog_path.exists():
            aggregate_rows.extend(read_tsv(result_catalog_path))
    if aggregate_rows:
        write_tsv(output_root / "result_catalog.tsv", list(aggregate_rows[0].keys()), aggregate_rows)

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        baseline_name=args.baseline_name,
        disease_program=args.disease_program,
        fold_count=len(target_folds),
    )


if __name__ == "__main__":
    main()
