from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from Libs.utils.evaluation import operational_twin_successor_lineage_protocol as successor
from Libs.utils.evaluation.operational_twin_training_protocol import (
    CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS,
    OperationalTwinProtocolError,
    canonical_hash,
)
from Libs.utils.data.raw_data_registry import load_config_with_provenance
from Libs.utils.exp_utils import load_config_stack


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def unpublished_successor_repository(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copy declared public inputs and retain the outer repository's Git identity."""
    config, provenance = load_config_with_provenance(ROOT / "Libs/configs/config_stack.yaml")
    contract = config[successor.CONFIG_KEY]
    manifests = contract["predecessor_authority"]["round_34"]["manifests"]
    manifest_paths = {row["path"] for row in manifests.values()}
    paths = set(provenance["include_order"]) | set(contract["source_bindings"]) | manifest_paths
    # These are the config closure, source-identity inputs and the three public
    # manifests read by validate_successor_config. Historical authorities remain
    # genuine Git reads through the same outer object store.
    assert all(path.startswith("Libs/") or path in manifest_paths for path in paths)
    assert paths.isdisjoint(successor.publication_paths(config).values())
    repository = tmp_path_factory.mktemp("unpublished_successor")
    for relative in sorted(paths):
        assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
        source = ROOT / relative
        assert source.is_file() and not source.is_symlink(), relative
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        shutil.copymode(source, target)

    # Reuse the local-fixture alternate mechanism. Under the validation gate,
    # ROOT already contains only the admitted isolated object projection.
    subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True, capture_output=True)
    objects = (ROOT / _git(ROOT, "rev-parse", "--git-path", "objects")).resolve()
    (repository / ".git/objects/info/alternates").write_text(str(objects) + "\n")
    (repository / ".git/HEAD").write_text(_git(ROOT, "rev-parse", "HEAD") + "\n")
    protocol_root = repository / contract["output_namespace"]["protocol_root"]
    if protocol_root.is_dir() and not any(protocol_root.iterdir()):
        protocol_root.rmdir()
    return repository


@pytest.fixture
def prepublication_successor_root(
    unpublished_successor_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(globals(), "ROOT", unpublished_successor_repository)


def _configured_successor() -> tuple[dict[str, object], dict[str, object]]:
    config = deepcopy(load_config_stack(ROOT / "Libs/configs/config_stack.yaml"))
    contract = config[successor.CONFIG_KEY]
    identities = successor.compute_successor_identity_declarations(config, ROOT)
    contract["identity"].update(
        {key: value for key, value in identities.items() if value is not None}
    )
    contract["environment_contract"]["environment_contract_uid"] = identities[
        "environment_contract_uid"
    ]
    contract["resource_contract"]["resource_contract_uid"] = identities[
        "resource_contract_uid"
    ]
    manifest = successor.build_pre_change_identity_manifest(config, ROOT)
    identities = successor.compute_successor_identity_declarations(
        config, ROOT, manifest
    )
    contract["identity"].update(identities)
    return config, manifest


def _blocker_codes(rows: list[dict[str, str]]) -> set[str]:
    return {row["code"] for row in rows}


@pytest.mark.usefixtures("prepublication_successor_root")
def test_successor_derivation_is_deterministic_disjoint_and_non_executing() -> None:
    config, manifest = _configured_successor()
    assert successor.validate_successor_config(
        config,
        ROOT,
        pre_change_manifest=manifest,
    ) == []

    first = successor.build_successor_artifacts(config, ROOT, manifest)
    second = successor.build_successor_artifacts(config, ROOT, manifest)
    assert first == second
    identities = first["identities"]
    predecessor = config[successor.CONFIG_KEY]["predecessor_authority"]
    assert identities["candidate_lineage_uid"] != predecessor[
        "predecessor_candidate_lineage_uid"
    ]
    assert identities["protocol_uid"] != predecessor["predecessor_protocol_uid"]
    assert identities["preflight_uid"] != predecessor["predecessor_preflight_uid"]

    assignments = first["assignment_manifest"]
    commands = first["assignment_commands"]
    assert assignments["assignment_count"] == 15
    assert assignments["attempt_slot_count"] == 30
    assert commands["command_count"] == 30
    assert [row["command_index"] for row in commands["commands"]] == list(
        range(30)
    )
    assert successor._validate_bijections(
        first, predecessor["predecessor_candidate_lineage_uid"]
    ) == []
    assert successor._validate_commands(config, first) == []
    assert successor._validate_future_absence(
        config,
        ROOT,
        first["protocol"],
        assignments["assignments"],
    ) == []

    inherited = config["operational_twin_candidate_lineage_protocol"]
    assert first["protocol"]["scientific_contract"] == inherited[
        "scientific_contract"
    ]
    assert first["protocol"]["execution_contract"] == inherited[
        "execution_contract"
    ]
    assert first["protocol"]["scheduler_evidence_contract"] == inherited[
        "scheduler_evidence_contract"
    ]
    assert first["candidate_lineage_manifest"]["candidate_dispatch_authorized"] is False
    assert first["candidate_lineage_manifest"]["scientific_twin_result"] == "NOT_RUN"
    assert set(
        first["candidate_lineage_manifest"]["access_counters"].values()
    ) == {0}
    initialization = first["candidate_lineage_manifest"]["initialization"]
    assert initialization["initialization_source_candidate_lineage_uid"] is None
    assert initialization["historical_state_use_allowed"] is False
    assert all(initialization[field] is None for field in CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            lambda contract, inherited: contract["initialization"].__setitem__(
                "checkpoint_input", "historical.pt"
            ),
            "successor_historical_state_input",
        ),
        (
            lambda contract, inherited: inherited["scientific_contract"][
                "seeds"
            ].append(999999),
            "successor_seed_identity_drift",
        ),
        (
            lambda contract, inherited: inherited["scientific_contract"][
                "methods"
            ][0]["budget"].__setitem__("epochs", 1),
            "successor_scientific_contract_drift",
        ),
        (
            lambda contract, inherited: contract["resource_contract"]["profiles"][
                "gpu_single"
            ].__setitem__("exclusive", True),
            "successor_resource_contract_drift",
        ),
        (
            lambda contract, inherited: contract["review_contract"].__setitem__(
                "decision", "rejected"
            ),
            "successor_review_contract_mismatch",
        ),
        (
            lambda contract, inherited: contract["access_counters"].__setitem__(
                "protected_holdout_access", 1
            ),
            "successor_access_boundary_open",
        ),
    ],
)
@pytest.mark.usefixtures("prepublication_successor_root")
def test_successor_config_drift_fails_closed(
    mutation: object,
    expected_code: str,
) -> None:
    config, manifest = _configured_successor()
    contract = config[successor.CONFIG_KEY]
    inherited = config["operational_twin_candidate_lineage_protocol"]
    mutation(contract, inherited)
    blockers = successor.validate_successor_config(
        config,
        ROOT,
        pre_change_manifest=manifest,
    )
    assert expected_code in _blocker_codes(blockers)


@pytest.mark.usefixtures("prepublication_successor_root")
def test_successor_bijections_reject_gaps_duplicates_and_old_identity_reuse() -> None:
    config, manifest = _configured_successor()
    built = successor.build_successor_artifacts(config, ROOT, manifest)
    predecessor_uid = config[successor.CONFIG_KEY]["predecessor_authority"][
        "predecessor_candidate_lineage_uid"
    ]

    missing = deepcopy(built)
    missing["mappings"]["assignment_bijection"].pop()
    assert "successor_assignment_bijection_incomplete" in _blocker_codes(
        successor._validate_bijections(missing, predecessor_uid)
    )

    duplicate = deepcopy(built)
    duplicate["mappings"]["attempt_slot_bijection"][1][
        "successor_attempt_uid"
    ] = duplicate["mappings"]["attempt_slot_bijection"][0][
        "successor_attempt_uid"
    ]
    assert "successor_attempt_bijection_not_one_to_one" in _blocker_codes(
        successor._validate_bijections(duplicate, predecessor_uid)
    )

    reused = deepcopy(built)
    reused["mappings"]["assignment_bijection"][0][
        "successor_assignment_uid"
    ] = reused["mappings"]["assignment_bijection"][0][
        "predecessor_assignment_uid"
    ]
    assert "successor_assignment_identity_collision" in _blocker_codes(
        successor._validate_bijections(reused, predecessor_uid)
    )


@pytest.mark.usefixtures("prepublication_successor_root")
def test_pre_change_manifest_is_hashed_and_records_absence() -> None:
    config, manifest = _configured_successor()
    assert manifest["manifest_hash"] == canonical_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    assert all(
        row["git_object_id_at_planner_commit"] is None
        and row["filesystem_absent_when_recorded"] is True
        for row in (
            list(manifest["successor_paths_absent"])
            + list(manifest["reserved_retry_paths_absent"])
        )
    )
    assert successor.validate_pre_change_identity_manifest(
        config, ROOT, manifest
    ) == []
    tampered = deepcopy(manifest)
    tampered["successor_paths_absent"][0]["path"] = "Output/elsewhere"
    assert {
        "successor_pre_change_identity_manifest_hash_mismatch",
        "successor_pre_change_identity_manifest_drift",
    }.issubset(
        _blocker_codes(
            successor.validate_pre_change_identity_manifest(
                config, ROOT, tampered
            )
        )
    )


def test_path_validation_rejects_escape_alias_and_ancestor_symlink(
    tmp_path: Path,
) -> None:
    assert "successor_path_not_canonical" in _blocker_codes(
        successor._validate_path(
            tmp_path,
            "../escape.json",
            allowed_root="Output/audits",
            require_absent=True,
            require_git_absent=False,
        )
    )
    assert "successor_path_not_canonical" in _blocker_codes(
        successor._validate_path(
            tmp_path,
            "Output/audits//alias.json",
            allowed_root="Output/audits",
            require_absent=True,
            require_git_absent=False,
        )
    )
    (tmp_path / "Output").mkdir()
    os.symlink(tmp_path / "outside", tmp_path / "Output/audits")
    assert "successor_path_symlink" in _blocker_codes(
        successor._validate_path(
            tmp_path,
            "Output/audits/protocol.json",
            allowed_root="Output/audits",
            require_absent=True,
            require_git_absent=False,
        )
    )


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _reviewed_package_fixture(tmp_path: Path) -> tuple[dict[str, object], Path]:
    root = tmp_path / "repository"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "successor@example.invalid"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Successor Fixture"],
        cwd=root,
        check=True,
    )
    candidate_uid = "operational_twin_successor_lineage_fixture"
    output_namespace = {
        "candidate_lineage_manifest": "Output/results/protocol/candidate_lineage_manifest.json",
        "assignment_manifest": "Output/results/protocol/assignment_manifest.json",
        "assignment_commands": "Output/results/protocol/assignment_commands.json",
        "preflight_authority": "Output/audits/preflight.json",
        "protocol_audit": "Output/audits/protocol.json",
        "preflight_report": "Output/reports/preflight.md",
        "protocol_report": "Output/reports/protocol.md",
    }
    review_contract = {
        "path": "orchestrator/rounds/round-49/review-record.json",
        "schema_version": "review-record-v4",
        "round_id": "round-49-freeze-operational-twin-successor-lineage-protocol",
        "roadmap_id": "roadmap",
        "roadmap_revision": "revision",
        "roadmap_dir": "orchestrator/roadmaps/roadmap/revision",
        "milestone_id": "operational-twin-qualification",
        "direction_id": "freeze-operational-twin-successor-lineage-protocol",
        "extracted_item_id": "operational-twin-disjoint-successor-lineage-protocol-freeze",
        "decision": "approved",
    }
    config = {
        successor.CONFIG_KEY: {
            "identity": {
                "candidate_lineage_uid": candidate_uid,
                "protocol_uid": "operational_twin_successor_protocol_fixture",
                "preflight_uid": "operational_twin_successor_preflight_fixture",
            },
            "output_namespace": output_namespace,
            "review_contract": review_contract,
            "access_counters": {
                "sealed_material_access": 0,
                "test_outcome_access": 0,
                "protected_holdout_access": 0,
                "six_context_outcome_access": 0,
                "tool_access": 0,
                "mcp_access": 0,
                "agent_trace_access": 0,
                "agent_access": 0,
                "memory_access": 0,
                "coevolution_access": 0,
            },
        }
    }
    for key, path_value in output_namespace.items():
        if key not in successor.PUBLICATION_KEYS:
            continue
        path = root / path_value
        if key in successor.JSON_PUBLICATION_KEYS:
            _write_json(path, {"artifact": key})
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {key}\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "artifacts"], cwd=root, check=True)
    identities = {
        key: successor._git_identity(root, "HEAD", path)
        for key, path in successor.publication_paths(config).items()
    }
    binding_uid = successor.package_binding_uid(identities)
    review = {
        "schema_version": review_contract["schema_version"],
        "round_id": review_contract["round_id"],
        "roadmap_id": review_contract["roadmap_id"],
        "roadmap_revision": review_contract["roadmap_revision"],
        "roadmap_dir": review_contract["roadmap_dir"],
        "milestone_id": review_contract["milestone_id"],
        "direction_id": review_contract["direction_id"],
        "extracted_item_id": review_contract["extracted_item_id"],
        "decision": "approved",
        "evidence_summary": (
            f"{successor.POSITIVE_TERMINAL} {candidate_uid} {binding_uid}"
        ),
        "roadmap_closeout": {},
    }
    _write_json(root / review_contract["path"], review)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "review"], cwd=root, check=True)
    return config, root


def test_reviewed_package_gate_is_read_only_and_returns_no_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root = _reviewed_package_fixture(tmp_path)
    monkeypatch.setattr(
        successor,
        "validate_materialized_successor_package",
        lambda loaded_config, loaded_root: [],
    )
    before = _git(root, "status", "--short", "--untracked-files=all")
    result = successor.require_approved_successor_lineage_protocol(config, root)
    assert result["candidate_dispatch_authorized"] is False
    assert result["execution_handoff_required"] is True
    assert result["execution_authority"] is None
    assert result["resource_environment_receipt"] is None
    assert result["scientific_twin_result"] == "NOT_RUN"
    assert before == _git(root, "status", "--short", "--untracked-files=all") == ""


@pytest.mark.parametrize(
    "state",
    ["missing", "rejected", "uncommitted", "binding", "artifact"],
)
def test_reviewed_package_gate_rejects_review_or_artifact_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    config, root = _reviewed_package_fixture(tmp_path)
    monkeypatch.setattr(
        successor,
        "validate_materialized_successor_package",
        lambda loaded_config, loaded_root: [],
    )
    review_path = root / config[successor.CONFIG_KEY]["review_contract"]["path"]
    if state == "missing":
        review_path.unlink()
        subprocess.run(["git", "add", "-u"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "remove review"], cwd=root, check=True)
    elif state == "artifact":
        artifact = root / successor.publication_paths(config)["protocol_report"]
        artifact.write_text("tampered\n", encoding="utf-8")
    else:
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if state == "rejected":
            review["decision"] = "rejected"
        elif state == "binding":
            review["evidence_summary"] = successor.POSITIVE_TERMINAL
        else:
            review["evidence_summary"] += " uncommitted"
        _write_json(review_path, review)
        if state != "uncommitted":
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "commit", "-q", "-m", state], cwd=root, check=True)
    with pytest.raises(OperationalTwinProtocolError):
        successor.require_approved_successor_lineage_protocol(config, root)
