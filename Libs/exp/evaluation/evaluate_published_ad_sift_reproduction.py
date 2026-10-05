from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.evaluation.evaluate_published_ad_sift_reproduction import build_report
from Libs.scripts.evaluation.evaluate_published_ad_sift_reproduction import markdown_report
from Libs.utils.data.raw_data_registry import ensure_directory
from Libs.utils.data.raw_data_registry import write_json
from Libs.utils.data.raw_data_registry import write_text_atomic
from Libs.utils.exp.paper_grade_entrypoint import artifact_path
from Libs.utils.exp.paper_grade_entrypoint import load_effective_config
from Libs.utils.exp.paper_grade_entrypoint import parse_standard_args
from Libs.utils.exp.paper_grade_entrypoint import suite_config
from Libs.utils.exp.paper_grade_entrypoint import write_entrypoint_outputs


def main() -> None:
    args = parse_standard_args(
        "Evaluate retrospective published AD intervention-prioritization consistency.",
        "published_ad_sift_reproduction",
    )
    config = load_effective_config(args)
    output_config = suite_config(config)["published_ad_sift_reproduction"]
    if args.dry_run:
        result = write_entrypoint_outputs(args, config, {"metrics": {}, "source_status": "dry_run"}, [], "dry_run")
        print(json.dumps(result, sort_keys=True))
        return
    report = build_report(args.config, args.model_family, args.allow_pending)
    json_output = artifact_path(output_config["audit_json"])
    md_output = artifact_path(output_config["audit_md"])
    write_json(json_output, report)
    write_text_atomic(md_output, markdown_report(report))
    output_root = artifact_path(output_config["output_root"])
    ensure_directory(output_root)
    summary_path = output_root / "published_ad_sift_reproduction_summary.json"
    write_json(summary_path, report)
    status = "materialized" if report["status"] == "available" else "pending"
    result = write_entrypoint_outputs(args, config, report, [summary_path, json_output, md_output], status)
    print(json.dumps(result, sort_keys=True))
    if status != "materialized" and not args.allow_pending:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
