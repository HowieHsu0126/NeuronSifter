"""Build the open-only Twin optimization/selection audit; never grant readiness."""

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
        paths = (output / output_name, output / "report.md")
        if any(path.exists() for path in paths):
            raise FileExistsError("Native evidence outputs already exist; overwrite is prohibited")
        config_path = args.config or NATIVE_CONFIG_PATH
        artifact = build_native_audit(ROOT, load_config_stack(config_path), config_path=config_path,
                                      training_root=args.training_root, training_component=args.training_component)
        schema = strict_json(ROOT / SCHEMA_PATH)
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(artifact)
        output.mkdir(parents=True, exist_ok=True)
        with paths[0].open("x") as stream:
            stream.write(json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False) + "\n")
        with paths[1].open("x") as stream:
            stream.write(native_report(artifact))
        print(json.dumps({"artifact_integrity": "VALID", "sufficient_training": False,
                          "ScientificTwin": "NOT_RUN", "terminal": artifact["terminal"]}))
        return 2
    if args.output_dir or args.training_root or args.training_component:
        parser.error("native locators are supported only in manuscript-native scope")
    config = load_config_stack(args.config) if args.config else load_config_stack()
    bootstrap_cpu_audit_cli(config["operational_twin_training_sufficiency"])
    artifacts = build_audit(ROOT, config)
    schema = json.loads((ROOT / SCHEMA_PATH).read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for artifact in artifacts.values():
        validator.validate(artifact)
    outputs = config["operational_twin_training_sufficiency"]["outputs"]
    for key, artifact in artifacts.items():
        path = ROOT / outputs[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
    path = ROOT / outputs["report"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(artifacts))
    print(
        json.dumps(
            {
                "terminal": artifacts["decision"]["terminal"],
                "artifact_integrity": "VALID",
                "sufficient_training": False,
                "sealed_test_access": 0,
            }
        )
    )
    # A valid negative audit is not a passing readiness gate.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
