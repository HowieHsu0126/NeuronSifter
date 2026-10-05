from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from Libs.utils.data.raw_data_registry import load_config_with_provenance, repo_root


CONTROLLED_RESEARCH_ENTRYPOINT = "Libs/configs/config_stack_controlled_research.yaml"
CONTROLLED_RESEARCH_REQUIRED_METADATA: dict[str, object] = {
    "entrypoint": CONTROLLED_RESEARCH_ENTRYPOINT,
    "public_entrypoint": "Libs/configs/config_stack.yaml",
    "controlled_config": "Libs/configs/data/adni_controlled_access.yaml",
    "non_public": True,
    "non_paper": True,
    "paper_evidence_allowed": False,
    "c1_c5_evidence_allowed": False,
    "explicit_invocation_required": True,
    "governance_approval_required": True,
    "approval_evidence_required": True,
    "network_credentials_required_for_sync": True,
    "fail_closed_without_approval": True,
    "fail_closed_without_credentials": True,
    "silent_fallback_allowed": False,
    "public_fallback_allowed": False,
    "synthetic_fallback_allowed": False,
}


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (repo_root() / path).resolve()


def assert_controlled_research_entrypoint(config_path: str | Path) -> None:
    expected = _resolve(CONTROLLED_RESEARCH_ENTRYPOINT)
    actual = _resolve(config_path)
    if actual != expected:
        raise ValueError(
            "Controlled research commands require explicit --config "
            f"{CONTROLLED_RESEARCH_ENTRYPOINT}; received {config_path}."
        )


def load_controlled_research_config(config_path: str | Path) -> dict:
    assert_controlled_research_entrypoint(config_path)
    config, provenance = load_config_with_provenance(config_path)
    metadata = config.get("controlled_research_stack")
    if not isinstance(metadata, Mapping):
        raise ValueError("Controlled research stack metadata is missing.")
    mismatches = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in CONTROLLED_RESEARCH_REQUIRED_METADATA.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Controlled research stack metadata is invalid: {mismatches}")
    if provenance.get("entrypoint") != CONTROLLED_RESEARCH_ENTRYPOINT:
        raise ValueError("Controlled research stack provenance entrypoint is invalid.")
    required_includes = {
        "Libs/configs/config_stack.yaml",
        "Libs/configs/data/adni_controlled_access.yaml",
    }
    if not required_includes.issubset(set(provenance.get("include_order", []))):
        raise ValueError("Controlled research stack does not include its public base and controlled overlay.")
    if "adni_controlled_access" not in config:
        raise ValueError("Controlled research stack is missing the ADNI controlled-access section.")
    return config


def assert_controlled_governance_approval(config: Mapping[str, object]) -> None:
    controlled = config.get("adni_controlled_access")
    if not isinstance(controlled, Mapping):
        raise ValueError("ADNI controlled-access configuration is missing.")
    governance = controlled.get("governance")
    paths = controlled.get("paths")
    if not isinstance(governance, Mapping) or not isinstance(paths, Mapping):
        raise ValueError("ADNI controlled-access governance metadata is incomplete.")
    if governance.get("data_use_agreement_required") is not True:
        raise ValueError("ADNI data-use approval must remain required.")
    if governance.get("approved_application_scope_required") is not True:
        raise ValueError("ADNI approved application scope must remain required.")
    evidence = governance.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("ADNI governance evidence contract is missing.")
    private_root = _resolve(str(paths.get("private_root", "")))
    governance_root = private_root / str(evidence.get("governance_root", ""))
    missing = []
    for row in evidence.get("required_documents", []):
        if not isinstance(row, Mapping) or row.get("required") is not True:
            continue
        document_path = governance_root / str(row.get("relative_path", ""))
        if not document_path.is_file():
            missing.append(document_path.as_posix())
    if missing:
        raise FileNotFoundError(f"Required controlled-research approval evidence is missing: {missing}")


def assert_controlled_network_credentials(config: Mapping[str, object]) -> None:
    controlled = config.get("adni_controlled_access")
    if not isinstance(controlled, Mapping):
        raise ValueError("ADNI controlled-access configuration is missing.")
    sync = controlled.get("sync")
    credentials = controlled.get("credentials")
    if not isinstance(sync, Mapping) or not isinstance(credentials, Mapping):
        raise ValueError("ADNI controlled-access authentication metadata is incomplete.")
    auth = sync.get("authenticated_http")
    if not isinstance(auth, Mapping):
        raise ValueError("ADNI controlled-access HTTP authentication contract is missing.")
    order = str(auth.get("authentication_order", ""))
    if order == "browser_cookie_jar_first":
        candidates = [auth.get("cookie_jar_path"), *auth.get("candidate_cookie_jar_paths", [])]
        if not any(_resolve(str(value)).is_file() for value in candidates if value):
            raise RuntimeError("Missing required ADNI approved private cookie jar for controlled network access.")
        return
    if order == "password_login":
        required_names = [str(credentials.get("email_env", "")), str(credentials.get("password_env", ""))]
        missing_names = [name for name in required_names if not name or not os.environ.get(name)]
        if missing_names:
            raise RuntimeError("Missing required ADNI controlled-access environment credentials.")
        return
    raise ValueError(f"Unsupported controlled network authentication order: {order}")
