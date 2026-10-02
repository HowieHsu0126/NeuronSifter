"""Derive current C1 contract artifacts without treating derivation as training authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.foundation_state_c1_capabilities import (  # noqa: E402
    build_c1_objective_capability_manifest,
)
from Libs.utils.data.foundation_state_c1_loader import (  # noqa: E402
    build_c1_vocabulary_from_frozen_bindings,
)
from Libs.utils.data.public_foundation_evidence import safe_repo_path  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, write_json, write_text_atomic  # noqa: E402


CONFIG_PATH = "Libs/configs/config_stack_c1_rescue.yaml"
SCHEMA_PATH = "Libs/configs/schema/current_c1_contract_artifact_projection.schema.json"
DEFAULT_OUTPUT_PATH = "Output/audits/current_c1_contract_artifact_projection.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _serialized(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _artifact_projection(
    path: str,
    payload: Mapping[str, object],
    configured_sha256: str,
) -> tuple[dict[str, object], str]:
    text = _serialized(payload)
    digest = _sha256_text(text)
    return (
        {
            "path": path,
            "derived_sha256": digest,
            "configured_sha256": configured_sha256,
            "configured_sha256_matches": digest == configured_sha256,
        },
        text,
    )


def build_projection(repo_root: Path = REPO_ROOT) -> tuple[dict[str, object], dict[str, str]]:
    repo_root = repo_root.resolve()
    config = load_config(repo_root / CONFIG_PATH)
    contract = config["foundation_state_c1"]
    frozen = contract["frozen_corpus"]
    encoder = contract["encoder"]["input_contract"]
    artifacts = contract["artifacts"]

    _, vocabulary_manifest = build_c1_vocabulary_from_frozen_bindings(repo_root, config)
    capability_manifest = build_c1_objective_capability_manifest(
        config,
        repo_root,
        verify_live=False,
    )
    vocabulary, vocabulary_text = _artifact_projection(
        str(encoder["vocabulary_manifest_path"]),
        vocabulary_manifest,
        str(encoder["vocabulary_manifest_sha256"]),
    )
    capability, capability_text = _artifact_projection(
        str(artifacts["objective_capability_manifest_path"]),
        capability_manifest,
        str(artifacts["objective_capability_manifest_sha256"]),
    )
    blockers: list[dict[str, str]] = []
    for name, payload in (
        ("vocabulary_manifest", vocabulary_manifest),
        ("capability_manifest", capability_manifest),
    ):
        if payload.get("corpus_uid") != frozen["corpus_uid"]:
            blockers.append(
                {
                    "code": "current_c1_artifact_corpus_uid_mismatch",
                    "path": name,
                    "message": str(payload.get("corpus_uid")),
                }
            )
        if payload.get("corpus_hash") != frozen["identity_hash"]:
            blockers.append(
                {
                    "code": "current_c1_artifact_corpus_hash_mismatch",
                    "path": name,
                    "message": str(payload.get("corpus_hash")),
                }
            )
    projection = {
        "schema_version": "neuronsifter-current-c1-contract-artifact-projection",
        "status": "derived" if not blockers else "blocked",
        "training_allowed": False,
        "corpus_uid": frozen["corpus_uid"],
        "corpus_hash": frozen["identity_hash"],
        "vocabulary_manifest": vocabulary,
        "capability_manifest": capability,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "next_action": (
            "repair_current_c1_derivation_input"
            if blockers
            else (
                "continue_current_c1_preflight_and_claim_scoped_qualification"
                if all(
                    record["configured_sha256_matches"] is True
                    for record in (vocabulary, capability)
                )
                else "bind_current_contract_hashes_then_seal_current_schemas"
            )
        ),
    }
    return projection, {
        str(vocabulary["path"]): vocabulary_text,
        str(capability["path"]): capability_text,
    }


def validate_schema(payload: Mapping[str, object], repo_root: Path = REPO_ROOT) -> None:
    schema = _load_json(safe_repo_path(repo_root.resolve(), SCHEMA_PATH))
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(payload))
    if errors:
        raise ValueError(
            "Current C1 contract projection schema violation: "
            + "; ".join(error.message for error in errors)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--write-manifests",
        action="store_true",
        help="Write derived manifests only after their configured hashes match.",
    )
    args = parser.parse_args()
    projection, manifest_text = build_projection(REPO_ROOT)
    validate_schema(projection, REPO_ROOT)
    write_json(safe_repo_path(REPO_ROOT, args.output), projection)
    if args.write_manifests:
        unmatched = [
            name
            for name in ("vocabulary_manifest", "capability_manifest")
            if projection[name]["configured_sha256_matches"] is not True
        ]
        if unmatched:
            raise SystemExit(
                "Refusing to write manifests before the active contract binds their derived hashes: "
                + ", ".join(unmatched)
            )
        for relative_path, text in manifest_text.items():
            write_text_atomic(safe_repo_path(REPO_ROOT, relative_path), text)
    print(json.dumps(projection, sort_keys=True))
    if projection["status"] != "derived":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
