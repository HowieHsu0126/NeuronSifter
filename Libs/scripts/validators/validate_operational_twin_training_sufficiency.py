"""Recompute the open-only sufficiency audit without modifying any artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jsonschema import Draft202012Validator  # noqa: E402
from Libs.utils.evaluation.operational_twin_training_sufficiency import (  # noqa: E402
    SCHEMA_PATH,
    NATIVE_CONFIG_PATH,
    NATIVE_OUTPUT_NAME,
    NATIVE_COMPONENT_OUTPUT_NAME,
    build_native_audit,
    native_report,
    strict_json,
    bootstrap_cpu_audit_cli,
    build_audit,
    render_report,
)
from Libs.utils.exp_utils import load_config_stack  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-only", action="store_true", required=True)
    parser.add_argument("--protocol-scope", choices=("historical-v1", "manuscript-native"), default="historical-v1")
    parser.add_argument("--config")
    parser.add_argument("--output-dir")
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--training-component", choices=("residual_mpc_predictor",))
    args = parser.parse_args()
    if args.protocol_scope == "manuscript-native":
        if not args.output_dir:
            parser.error("manuscript-native requires a separate --output-dir")
        output = Path(args.output_dir).resolve()
        output_name = NATIVE_COMPONENT_OUTPUT_NAME if args.training_component else NATIVE_OUTPUT_NAME
        config_path = args.config or NATIVE_CONFIG_PATH
        expected = build_native_audit(ROOT, load_config_stack(config_path), config_path=config_path,
                                      training_root=args.training_root, training_component=args.training_component)
        schema = strict_json(ROOT / SCHEMA_PATH)
        Draft202012Validator.check_schema(schema)
        existing = strict_json(output / output_name)
        Draft202012Validator(schema).validate(existing)
        if existing != expected:
            raise ValueError("native artifact or source evidence recomputation mismatch")
        if (output / "report.md").read_text() != native_report(expected):
            raise ValueError("native sufficiency report drift")
        print(json.dumps({"artifact_validation": "PASS", "read_only": True,
                          "independent_scientific_approval": False, "ScientificTwin": "NOT_RUN"}))
        return 0
    if args.output_dir or args.training_root or args.training_component:
        parser.error("native locators are supported only in manuscript-native scope")
    config = load_config_stack(args.config) if args.config else load_config_stack()
    bootstrap_cpu_audit_cli(config["operational_twin_training_sufficiency"])
    outputs = config["operational_twin_training_sufficiency"]["outputs"]
    expected = build_audit(ROOT, config)
    schema = json.loads((ROOT / SCHEMA_PATH).read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for key, value in expected.items():
        existing = json.loads((ROOT / outputs[key]).read_text())
        validator.validate(existing)
        if existing != value:
            raise ValueError("audit recomputation mismatch: " + key)
    if (ROOT / outputs["report"]).read_text() != render_report(expected):
        raise ValueError("sufficiency report drift")
    print(
        json.dumps(
            {
                "artifact_validation": "PASS",
                "independent_scientific_approval": False,
                "terminal": expected["decision"]["terminal"],
                "read_only": True,
                "sealed_test_access": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
