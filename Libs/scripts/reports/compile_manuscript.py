from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile the paper manuscript using the configured LaTeX toolchain.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--main-tex", default="")
    return parser.parse_args()


def vscode_settings() -> dict:
    settings_path = REPO_ROOT / ".vscode/settings.json"
    filtered_lines = []
    for line in settings_path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("//"):
            continue
        filtered_lines.append(line)
    return yaml.safe_load("\n".join(filtered_lines)) or {}


def tool_lookup(settings: dict) -> dict[str, dict]:
    tools = settings.get("latex-workshop.latex.tools", [])
    return {str(tool["name"]): dict(tool) for tool in tools}


def tool_args(tool_config: dict, main_tex_path: Path) -> list[str]:
    args: list[str] = []
    doc_name = main_tex_path.name
    docfile = main_tex_path.stem
    for value in tool_config.get("args", []):
        text = str(value).replace("%DOC%", doc_name).replace("%DOCFILE%", docfile)
        args.append(text)
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    settings = vscode_settings()
    tool_by_name = tool_lookup(settings)
    main_tex = Path(args.main_tex or config["paper"]["paths"]["main_tex_path"]).resolve()
    manuscript_root = main_tex.parent
    recipe = config["paper"]["latex_build"]["toolchain"]

    for step in recipe:
        tool_name = str(step["tool"])
        if tool_name in tool_by_name:
            command = str(tool_by_name[tool_name]["command"])
            arguments = tool_args(tool_by_name[tool_name], main_tex)
        else:
            command = str(step["command"])
            arguments = [str(value) for value in step.get("args", [])]
        pipeline_logger.progress(
            "latex_step_started",
            tool=tool_name,
            command=command,
        )
        subprocess.run(
            [command, *arguments],
            cwd=manuscript_root,
            check=True,
        )
        pipeline_logger.progress(
            "latex_step_completed",
            tool=tool_name,
            command=command,
        )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=main_tex.with_suffix(".pdf"),
    )


if __name__ == "__main__":
    main()
