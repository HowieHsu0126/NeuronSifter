from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from Libs.utils.data import mechanism_basis_qualification as mbq


REPO_ROOT = Path(__file__).resolve().parents[2]


def _publication_slot() -> dict:
    _, raw, _ = mbq.load_frozen_git_json(
        REPO_ROOT,
        mbq.SOURCE_QUEUE_PATH,
        mbq.EXPECTED_UPSTREAM[mbq.SOURCE_QUEUE_PATH][0],
    )
    queue = json.loads(raw)
    return deepcopy(next(row for row in queue["slots"] if row["disposition"] == "eligible"))


def _xml(pmid: str, sections: list[tuple[str | None, str]]) -> bytes:
    parts = []
    for label, text in sections:
        attribute = "" if label is None else f' Label="{label}" NlmCategory="{label}"'
        parts.append(f"<AbstractText{attribute}>{text}</AbstractText>")
    return (
        "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
        f"<PMID>{pmid}</PMID><Article><Abstract>{''.join(parts)}</Abstract></Article>"
        "</MedlineCitation></PubmedArticle></PubmedArticleSet>"
    ).encode()


def test_contract_and_code_policy_are_independently_pinned() -> None:
    contract, _ = mbq.validate_contract(REPO_ROOT)
    assert contract["frozen_queue"]["eligible_slot_count"] == 15
    assert contract["retrieval_policy"]["maximum_total_requests"] == 45
    assert contract["retrieval_policy"]["retries"] == 0
    assert mbq._code_policy_sha256() == mbq.EXPECTED_CODE_POLICY_SHA256


def test_all_five_schemas_are_strict_draft_2020_12() -> None:
    for relative, expected_hash in mbq.EXPECTED_SCHEMA_SHA256.items():
        schema = json.loads((REPO_ROOT / relative).read_text())
        Draft202012Validator.check_schema(schema)
        assert mbq.sha256_bytes((REPO_ROOT / relative).read_bytes()) == expected_hash
        assert schema["additionalProperties"] is False


def test_skip_worktree_queue_loads_only_from_current_commit_blob() -> None:
    queue, raw, source = mbq.load_frozen_git_json(
        REPO_ROOT,
        mbq.SOURCE_QUEUE_PATH,
        mbq.EXPECTED_UPSTREAM[mbq.SOURCE_QUEUE_PATH][0],
    )
    assert source == "current_commit_git_blob"
    assert len(queue["slots"]) == 72
    assert mbq.sha256_bytes(raw) == mbq.EXPECTED_UPSTREAM[mbq.SOURCE_QUEUE_PATH][0]


def test_noncanonical_or_untracked_git_blob_fails_closed(tmp_path: Path) -> None:
    root = tmp_path.resolve()

    def reader(_: Path, args: list[str]) -> bytes:
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return f"{root}\n".encode()
        if args[0] == "ls-files":
            return b"fixture.json\n"
        return b'{"b":2, "a":1}\n'

    digest = mbq.sha256_bytes(b'{"b":2, "a":1}\n')
    with pytest.raises(mbq.MechanismBasisQualificationError, match="canonical JSON"):
        mbq.load_frozen_git_json(root, "fixture.json", digest, git_reader=reader)


def test_requirement_registry_is_deterministic_and_preserves_optional_prior_roles() -> None:
    authorities = mbq.load_frozen_authorities(REPO_ROOT)
    first = mbq.derive_requirement_registry(authorities["construct"], authorities["prior"])
    second = mbq.derive_requirement_registry(authorities["construct"], authorities["prior"])
    assert first == second
    assert len({row["requirement_uid"] for row in first}) == len(first)
    optional = {row["subject_uid"] for row in first if not row["required"]}
    assert optional == {"conservative_support_sensitivity", "broad_parameter_sensitivity"}


def test_exact_request_plan_rejects_search_related_and_unqueued_identity() -> None:
    slot = _publication_slot()
    request = mbq._request_plan(slot, "project@example.org")[0]
    mbq.validate_exact_request_plan(request, slot)
    for url in (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=x",
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&id=1&linkname=pubmed_pubmed",
        "https://publisher.example/article/1",
    ):
        mutated = {**request, "url": url}
        mutated["request_uid"] = mbq._request_uid(slot["slot_uid"], "pubmed_exact_fetch", url)
        with pytest.raises(mbq.MechanismBasisQualificationError):
            mbq.validate_exact_request_plan(mutated, slot)
    changed = deepcopy(slot)
    changed["metadata"]["pmid"] = "999"
    with pytest.raises(mbq.MechanismBasisQualificationError, match="exact-ID"):
        mbq.validate_exact_request_plan(request, changed)


def test_projector_reads_only_allowlisted_structured_sections() -> None:
    projected = mbq.project_pubmed_xml(
        _xml(
            "123",
            [
                ("BACKGROUND", "A source-authored mechanism definition."),
                ("RESULTS", "FORBIDDEN_EFFECT_SIZE_VALUE"),
                ("CONCLUSIONS", "FORBIDDEN_RECOMMENDATION"),
            ],
        ),
        expected_pmid="123",
        allowed_labels=["BACKGROUND", "METHODS", "LIMITATIONS"],
        denied_labels=["RESULTS", "CONCLUSIONS"],
    )
    assert [row["section_label"] for row in projected["allowed_sections"]] == ["BACKGROUND"]
    assert "FORBIDDEN" not in json.dumps(projected)
    assert projected["forbidden_sections_skipped"] == 2


def test_unstructured_or_identity_mismatched_content_fails_closed() -> None:
    projected = mbq.project_pubmed_xml(
        _xml("123", [(None, "Mixed result and mechanism prose")]),
        expected_pmid="123",
        allowed_labels=["BACKGROUND"],
        denied_labels=["RESULTS", "CONCLUSIONS"],
    )
    assert projected["disposition"] == "forbidden_mixed_content"
    assert projected["allowed_sections"] == []
    with pytest.raises(mbq.MechanismBasisQualificationError, match="identifier mismatch"):
        mbq.project_pubmed_xml(
            _xml("999", [("BACKGROUND", "text")]),
            expected_pmid="123",
            allowed_labels=["BACKGROUND"],
            denied_labels=["RESULTS"],
        )


def test_local_content_requires_bound_identity_license_and_safe_regular_path(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    source_root = root / "Input" / "public"
    source_root.mkdir(parents=True)
    source = source_root / "record.xml"
    source.write_bytes(b"safe")
    slot = {"canonical_source_identity": "pmid:123"}
    manifest = {
        "canonical_source_identity": "pmid:123",
        "license_status": "public_use_authorized",
        "file_sha256": mbq.sha256_bytes(b"safe"),
    }
    assert mbq.open_local_exact_source(
        repository_root=root,
        input_root=source_root,
        relative_path="record.xml",
        manifest_entry=manifest,
        slot=slot,
    ) == b"safe"
    denied = {**manifest, "license_status": "ambiguous"}
    with pytest.raises(mbq.MechanismBasisQualificationError, match="license"):
        mbq.open_local_exact_source(
            repository_root=root,
            input_root=source_root,
            relative_path="record.xml",
            manifest_entry=denied,
            slot=slot,
        )
    (source_root / "protected").mkdir()
    (source_root / "protected" / "record.xml").write_bytes(b"safe")
    with pytest.raises(mbq.MechanismBasisQualificationError, match="protected"):
        mbq.open_local_exact_source(
            repository_root=root,
            input_root=source_root,
            relative_path="protected/record.xml",
            manifest_entry=manifest,
            slot=slot,
        )


def test_transactional_install_is_idempotent_and_rejects_partial_or_overwrite(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    items = [(left, b"left\n"), (right, b"right\n")]
    assert mbq._install_many(items)
    assert not mbq._install_many(items)
    with pytest.raises(mbq.MechanismBasisQualificationError, match="overwrite"):
        mbq._install_many([(left, b"changed\n"), (right, b"right\n")])
    right.unlink()
    with pytest.raises(mbq.MechanismBasisQualificationError, match="partial"):
        mbq._install_many(items)


def test_claim_graph_rejects_unsupported_grade_and_dangling_edges() -> None:
    ledger = {"slots": [{"slot_uid": "slot-1", "grade": "CONTEXT_ONLY"}]}
    graph = {
        "claims": [{"claim_uid": "claim-1"}],
        "support_edges": [
            {
                "claim_uid": "claim-1",
                "slot_uid": "slot-1",
                "grade": "DIRECT_MECHANISM",
            }
        ],
        "contradictions": [],
        "counts": {
            "claim_count": 1,
            "support_edge_count": 1,
            "contradiction_count": 0,
            "unresolved_essential_contradiction_count": 0,
        },
    }
    with pytest.raises(mbq.MechanismBasisQualificationError, match="unsupported grade"):
        mbq._validate_claim_graph(graph, ledger)
    graph["support_edges"][0]["slot_uid"] = "missing"
    with pytest.raises(mbq.MechanismBasisQualificationError, match="dangling"):
        mbq._validate_claim_graph(graph, ledger)


def test_security_critical_closures_remain_false() -> None:
    assert not any(mbq.DOWNSTREAM_CLOSURE.values())
    assert mbq.ZERO_AUTHORIZATION_COUNTERS["six_context_outcome_access"] == 0
    assert mbq.ZERO_AUTHORIZATION_COUNTERS["protected_holdout_access"] == 0
    assert mbq.CODE_POLICY["retries"] == 0
    assert mbq.CODE_POLICY["concurrency"] == 1


@pytest.mark.parametrize(
    ("claim_type", "grade"),
    [("causal_dependency", "DIRECT_MECHANISM"), ("parameter_range", "PARAMETER_ANCHOR"), ("support_boundary", "BOUNDARY_CONSTRAINT")],
)
def test_support_grade_precedence(claim_type: str, grade: str) -> None:
    assert mbq._derive_grade([{"claim_type": claim_type}], admissible_context=True) == grade


def test_bidirectional_requirement_coverage_and_optional_gap() -> None:
    requirements = [
        {"requirement_uid": "required", "required": True},
        {"requirement_uid": "optional", "required": False},
    ]
    claims = [{"claim_uid": "claim", "slot_uid": "slot", "requirement_uids": ["required"]}]
    rows, terminal = mbq.derive_requirement_coverage(requirements, claims, [])
    assert terminal == "MECHANISM_BASIS_QUALIFIED"
    assert [row["coverage_status"] for row in rows] == ["covered", "uncovered"]
    rows, terminal = mbq.derive_requirement_coverage(requirements, [], [])
    assert terminal == "MECHANISM_BASIS_INSUFFICIENT"
    assert rows[0]["coverage_status"] == "uncovered"


def test_unresolved_contradiction_blocks_qualified_terminal() -> None:
    requirements = [{"requirement_uid": "required", "required": True}]
    claims = [
        {"claim_uid": "left", "slot_uid": "slot-a", "requirement_uids": ["required"]},
        {"claim_uid": "right", "slot_uid": "slot-b", "requirement_uids": ["required"]},
    ]
    contradictions = [{"contradiction_uid": "conflict", "claim_uids": ["left", "right"], "status": "unresolved"}]
    rows, terminal = mbq.derive_requirement_coverage(requirements, claims, contradictions)
    assert terminal == "MECHANISM_BASIS_INSUFFICIENT"
    assert rows[0]["coverage_status"] == "contradictory"
    contradictions[0]["status"] = "resolved"
    rows, terminal = mbq.derive_requirement_coverage(requirements, claims, contradictions)
    assert terminal == "MECHANISM_BASIS_QUALIFIED"


def test_operational_failure_blocks_scientific_review() -> None:
    authorized = [{"request_plan": [{"request_uid": "request-a"}]}]
    failed = [{"request_uid": "request-a", "attempt_ordinal": 1, "response_sha256": None}]
    with pytest.raises(mbq.MechanismBasisQualificationError, match="operational"):
        mbq.validate_operational_review(failed, authorized)


def test_semantic_review_requires_bound_locator_requirement_and_parameter_units() -> None:
    slot = {"slot_uid": "mechanism-source-slot-" + "a" * 20}
    projection = {
        "allowed_sections": [
            {
                "locator": "abstract[1]",
                "content_sha256": "b" * 64,
                "content_class": "measurement_definition",
            }
        ],
        "disposition": "reviewed_allowed_projection",
    }
    requirement = {
        "requirement_uid": "mechanism-requirement-" + "c" * 20,
        "category": "parameter_domain",
        "subject_uid": "positive_real",
        "allowed_claim_types": ["parameter_range"],
    }
    spec = {"projection_sha256": mbq.canonical_value_sha256(projection), "rationale": "Typed bounded parameter evidence.", "claims": [{"claim_type": "parameter_range", "paraphrase": "A positive bounded dose range is declared.", "locator": "abstract[1]", "content_sha256": "b" * 64, "requirement": ["parameter_domain", "positive_real"], "units": ["mg/kg"], "range_context": "0.1-10 mg/kg", "assumptions": [], "limitations": [], "support_boundary": "Declared study design only.", "contradicts_claim_uids": ["claim-other"]}]}
    review = mbq._semantic_review(slot, projection, [requirement], spec)
    assert review["grade"] == "PARAMETER_ANCHOR"
    assert review["claims_supported"] == [review["semantic_claims"][0]["claim_uid"]]
    assert review["semantic_claims"][0]["contradicts_claim_uids"] == ["claim-other"]
    invalid = deepcopy(spec)
    invalid["claims"][0]["units"] = []
    with pytest.raises(mbq.MechanismBasisQualificationError, match="units"):
        mbq._semantic_review(slot, projection, [requirement], invalid)


def test_semantic_review_rejects_disallowed_requirement_and_section_types() -> None:
    slot = {"slot_uid": "mechanism-source-slot-" + "a" * 20}
    projection = {"allowed_sections": [{"locator": "abstract[1]", "content_sha256": "b" * 64, "content_class": "mechanism_definition"}], "disposition": "reviewed_allowed_projection"}
    requirement = {"requirement_uid": "mechanism-requirement-" + "c" * 20, "category": "parameter_domain", "subject_uid": "positive_real", "allowed_claim_types": ["parameter_range"]}
    spec = {"projection_sha256": mbq.canonical_value_sha256(projection), "rationale": "review", "claims": [{"claim_type": "measurement_definition", "paraphrase": "A measurement is defined.", "locator": "abstract[1]", "content_sha256": "b" * 64, "requirement": ["parameter_domain", "positive_real"], "units": [], "range_context": None, "assumptions": [], "limitations": [], "support_boundary": None}]}
    with pytest.raises(mbq.MechanismBasisQualificationError, match="not allowed"):
        mbq._semantic_review(slot, projection, [requirement], spec)
    requirement["allowed_claim_types"] = ["measurement_definition"]
    with pytest.raises(mbq.MechanismBasisQualificationError, match="section class"):
        mbq._semantic_review(slot, projection, [requirement], spec)


def _synthetic_claim(slot_uid: str, claim_uid: str, other: list[str] | None = None, paraphrase: str = "A mechanism dependency is declared.") -> dict:
    claim = {"claim_type": "causal_dependency", "paraphrase": paraphrase, "content_locator": "abstract[1]", "content_sha256": "b" * 64, "requirement_uids": ["required"], "units": [], "range_context": None, "assumptions": [], "limitations": [], "support_boundary": None, "contradicts_claim_uids": other or []}
    claim["claim_uid"] = mbq._claim_uid(slot_uid, claim)
    return claim


def _synthetic_ledger(claims: list[dict], status: str = "none_observed") -> dict:
    return {"slots": [{"slot_uid": "slot-a", "grade": "DIRECT_MECHANISM", "contradiction_status": status, "content_receipt": {"allowed_sections": [{"locator": "abstract[1]", "content_sha256": "b" * 64, "content_class": "mechanism_definition"}]}, "semantic_claims": claims}]}


def _synthetic_prior_contract() -> tuple[list[dict], dict]:
    requirements = [
        {
            "requirement_uid": "required",
            "category": "prior_role",
            "subject_uid": "canonical-role",
            "required": True,
            "allowed_claim_types": ["causal_dependency"],
        },
        {
            "requirement_uid": "optional-a",
            "category": "prior_role",
            "subject_uid": "sensitivity-a",
            "required": False,
            "allowed_claim_types": ["causal_dependency"],
        },
        {
            "requirement_uid": "optional-b",
            "category": "prior_role",
            "subject_uid": "sensitivity-b",
            "required": False,
            "allowed_claim_types": ["causal_dependency"],
        },
    ]
    prior = {
        "prior_roles": {
            "ordered_roles": [
                {
                    "role_uid": "canonical-role",
                    "role_class": "canonical",
                    "required": True,
                },
                {
                    "role_uid": "sensitivity-a",
                    "role_class": "sensitivity",
                    "required": False,
                },
                {
                    "role_uid": "sensitivity-b",
                    "role_class": "sensitivity",
                    "required": False,
                },
            ]
        }
    }
    return requirements, prior


def _synthetic_manifest(requirements: list[dict]) -> dict:
    return {
        "authorization_sha256": "a" * 64,
        "requirement_registry": requirements,
    }


def _synthetic_schema_bindings() -> dict[str, dict[str, str]]:
    return {
        path: {"path": path, "sha256": f"{index:x}" * 64}
        for index, path in enumerate(mbq.EXPECTED_SCHEMA_SHA256, start=1)
    }


def _synthetic_final_chain(
    requirements: list[dict], ledger: dict, prior: dict
) -> tuple[dict, dict, dict]:
    manifest = _synthetic_manifest(requirements)
    schemas = _synthetic_schema_bindings()
    contract_binding = {
        "path": mbq.CONTRACT_PATH,
        "sha256": mbq.sha256_bytes(b"synthetic-contract"),
    }
    authorization_binding = mbq._artifact_from_payload(
        mbq.AUTHORIZATION_PATH, manifest, "authorization_sha256"
    )
    ledger_binding = mbq._artifact_from_payload(
        mbq.LEDGER_PATH, ledger, "ledger_sha256"
    )
    claims, edges, contradictions = mbq.derive_graph_components(ledger, requirements)
    graph = {
        "contract": contract_binding,
        "schema": schemas[mbq.GRAPH_SCHEMA_PATH],
        "evidence_ledger": ledger_binding,
        "claims": claims,
        "support_edges": edges,
        "contradictions": contradictions,
        "counts": {
            "claim_count": len(claims),
            "support_edge_count": len(edges),
            "contradiction_count": len(contradictions),
            "unresolved_essential_contradiction_count": sum(
                row["essential"] and row["status"] == "unresolved"
                for row in contradictions
            ),
        },
    }
    graph["graph_sha256"] = mbq._self_hash(graph, "graph_sha256")
    coverage, terminal = mbq.derive_requirement_coverage(
        requirements, claims, contradictions
    )
    required = [row for row in coverage if row["required"]]
    uncovered = [
        row["requirement_uid"]
        for row in required
        if row["coverage_status"] == "uncovered"
    ]
    contradictory = [
        row["requirement_uid"]
        for row in required
        if row["coverage_status"] == "contradictory"
    ]
    canonical_prior, sensitivity_priors = mbq.derive_prior_support(
        requirements, coverage, prior
    )
    assessment = {
        "contract": contract_binding,
        "schema": schemas[mbq.ASSESSMENT_SCHEMA_PATH],
        "authorization": authorization_binding,
        "evidence_ledger": ledger_binding,
        "requirements": coverage,
        "coverage_counts": {
            "requirement_count": len(coverage),
            "required_count": len(required),
            "covered_required_count": len(required) - len(uncovered) - len(contradictory),
            "uncovered_required_count": len(uncovered),
            "contradictory_required_count": len(contradictory),
        },
        "uncovered_required_requirement_uids": uncovered,
        "contradictory_required_requirement_uids": contradictory,
        "canonical_prior_support": canonical_prior,
        "sensitivity_prior_support": sensitivity_priors,
        "decision_candidate": terminal,
        "access_counters": ledger["access_counters"],
        "downstream_authorization": deepcopy(mbq.DOWNSTREAM_CLOSURE),
        "source_claim_graph": mbq._artifact_from_payload(
            mbq.GRAPH_PATH, graph, "graph_sha256"
        ),
    }
    assessment["assessment_sha256"] = mbq._self_hash(
        assessment, "assessment_sha256"
    )
    decision = {
        "status": terminal,
        "contract": contract_binding,
        "schema": schemas[mbq.DECISION_SCHEMA_PATH],
        "authorization": authorization_binding,
        "evidence_ledger": ledger_binding,
        "missing_required_requirement_uids": uncovered,
        "contradictory_required_requirement_uids": contradictory,
        "scientific_interpretation": mbq.scientific_interpretation_for_terminal(
            terminal
        ),
        "access_counters": ledger["access_counters"],
        "downstream_authorization": deepcopy(mbq.DOWNSTREAM_CLOSURE),
        "source_claim_graph": mbq._artifact_from_payload(
            mbq.GRAPH_PATH, graph, "graph_sha256"
        ),
        "assessment": mbq._artifact_from_payload(
            mbq.ASSESSMENT_PATH, assessment, "assessment_sha256"
        ),
    }
    decision["decision_sha256"] = mbq._self_hash(decision, "decision_sha256")
    return graph, assessment, decision


def _validate_synthetic_final_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    requirements: list[dict],
    ledger: dict,
    graph: dict,
    assessment: dict,
    decision: dict,
    prior: dict,
) -> None:
    manifest = _synthetic_manifest(requirements)
    schemas = _synthetic_schema_bindings()
    outputs = {
        mbq.GRAPH_PATH: graph,
        mbq.ASSESSMENT_PATH: assessment,
        mbq.DECISION_PATH: decision,
    }
    monkeypatch.setattr(
        mbq,
        "load_frozen_authorities",
        lambda *_args, **_kwargs: {
            "prior": prior,
            "contract_raw": b"synthetic-contract",
        },
    )
    monkeypatch.setattr(mbq, "_schema_bindings", lambda *_args, **_kwargs: schemas)
    monkeypatch.setattr(mbq, "validate_authorization", lambda **_: manifest)
    monkeypatch.setattr(mbq, "validate_evidence_ledger", lambda **_: ledger)
    monkeypatch.setattr(
        mbq,
        "_load_output",
        lambda _root, relative, _schema, _hash: (outputs[relative], b""),
    )
    mbq.validate_final(repository_root=tmp_path)


def _rebind_assessment_and_decision(
    assessment: dict, decision: dict
) -> None:
    assessment["assessment_sha256"] = mbq._self_hash(
        assessment, "assessment_sha256"
    )
    decision["assessment"] = mbq._artifact_from_payload(
        mbq.ASSESSMENT_PATH, assessment, "assessment_sha256"
    )
    decision["decision_sha256"] = mbq._self_hash(decision, "decision_sha256")


def _rebind_graph_and_downstream(
    graph: dict, assessment: dict, decision: dict
) -> None:
    graph["graph_sha256"] = mbq._self_hash(graph, "graph_sha256")
    graph_binding = mbq._artifact_from_payload(
        mbq.GRAPH_PATH, graph, "graph_sha256"
    )
    assessment["source_claim_graph"] = graph_binding
    decision["source_claim_graph"] = graph_binding
    _rebind_assessment_and_decision(assessment, decision)


def _contradictory_ledger(left_status: str, right_status: str) -> dict:
    left = _synthetic_claim("slot-a", "unused")
    right = _synthetic_claim("slot-b", "unused")
    left["contradicts_claim_uids"] = [right["claim_uid"]]
    right["contradicts_claim_uids"] = [left["claim_uid"]]
    return {
        "slots": [
            _synthetic_ledger([left], left_status)["slots"][0],
            {
                **_synthetic_ledger([right], right_status)["slots"][0],
                "slot_uid": "slot-b",
            },
        ]
    }


def test_graph_is_reconstructed_from_ledger_and_rejects_forbidden_substitution() -> None:
    claim = _synthetic_claim("slot-a", "unused")
    ledger = _synthetic_ledger([claim])
    requirements = [{"requirement_uid": "required", "allowed_claim_types": ["causal_dependency"]}]
    claims, edges, contradictions = mbq.derive_graph_components(ledger, requirements)
    graph = {"claims": claims, "support_edges": edges, "contradictions": contradictions}
    mbq.validate_graph_matches_ledger(graph, ledger, requirements)
    graph = deepcopy(graph)
    graph["claims"][0]["paraphrase"] = "Efficacy benefit ranking is proven."
    with pytest.raises(mbq.MechanismBasisQualificationError, match="diverges"):
        mbq.validate_graph_matches_ledger(graph, ledger, requirements)
    ledger["slots"][0]["semantic_claims"][0]["paraphrase"] = "Efficacy benefit ranking is proven."
    semantic_claim = ledger["slots"][0]["semantic_claims"][0]
    semantic_claim["claim_uid"] = mbq._claim_uid(
        ledger["slots"][0]["slot_uid"],
        {key: value for key, value in semantic_claim.items() if key != "claim_uid"},
    )
    with pytest.raises(mbq.MechanismBasisQualificationError, match="forbidden"):
        mbq.derive_graph_components(ledger, requirements)


@pytest.mark.parametrize(
    ("row_status", "graph_status", "resolution_dimension"),
    [
        ("unresolved", "unresolved", "not_resolved"),
        ("resolved_by_scope", "resolved", "scope"),
    ],
)
def test_contradiction_lineage_is_symmetric_and_preserved(
    row_status: str, graph_status: str, resolution_dimension: str
) -> None:
    ledger = _contradictory_ledger(row_status, row_status)
    requirements = [{"requirement_uid": "required", "allowed_claim_types": ["causal_dependency"]}]
    _, _, contradictions = mbq.derive_graph_components(ledger, requirements)
    assert contradictions[0]["status"] == graph_status
    assert contradictions[0]["resolution_dimension"] == resolution_dimension
    ledger["slots"][1]["semantic_claims"][0]["contradicts_claim_uids"] = []
    with pytest.raises(mbq.MechanismBasisQualificationError, match="asymmetric"):
        mbq.derive_graph_components(ledger, requirements)


@pytest.mark.parametrize("failure", ["self", "dangling", "duplicate"])
def test_contradiction_lineage_rejects_malformed_declarations(failure: str) -> None:
    left = _synthetic_claim("slot-a", "unused")
    right = _synthetic_claim("slot-b", "unused")
    if failure == "self":
        left["contradicts_claim_uids"] = [left["claim_uid"]]
    elif failure == "dangling":
        left["contradicts_claim_uids"] = ["missing-claim"]
    else:
        left["contradicts_claim_uids"] = [right["claim_uid"], right["claim_uid"]]
        right["contradicts_claim_uids"] = [left["claim_uid"]]
    ledger = {
        "slots": [
            _synthetic_ledger([left], "unresolved")["slots"][0],
            {
                **_synthetic_ledger([right], "unresolved")["slots"][0],
                "slot_uid": "slot-b",
            },
        ]
    }
    requirements = [
        {"requirement_uid": "required", "allowed_claim_types": ["causal_dependency"]}
    ]
    with pytest.raises(mbq.MechanismBasisQualificationError, match="malformed"):
        mbq.derive_graph_components(ledger, requirements)


@pytest.mark.parametrize("status", ["unresolved", "resolved_by_scope"])
def test_contradiction_lineage_rejects_orphan_disposition(status: str) -> None:
    ledger = _synthetic_ledger([_synthetic_claim("slot-a", "unused")], status)
    requirements = [
        {"requirement_uid": "required", "allowed_claim_types": ["causal_dependency"]}
    ]
    with pytest.raises(mbq.MechanismBasisQualificationError, match="orphan"):
        mbq.derive_graph_components(ledger, requirements)


@pytest.mark.parametrize(
    "status", ["none_observed", "not_assessed", "inadmissible"]
)
def test_contradiction_lineage_rejects_neutral_status_with_relation(
    status: str,
) -> None:
    ledger = _contradictory_ledger(status, status)
    requirements = [
        {"requirement_uid": "required", "allowed_claim_types": ["causal_dependency"]}
    ]
    with pytest.raises(mbq.MechanismBasisQualificationError, match="neutral"):
        mbq.derive_graph_components(ledger, requirements)


def test_contradiction_lineage_rejects_incompatible_dispositions() -> None:
    ledger = _contradictory_ledger("unresolved", "resolved_by_scope")
    requirements = [
        {"requirement_uid": "required", "allowed_claim_types": ["causal_dependency"]}
    ]
    with pytest.raises(mbq.MechanismBasisQualificationError, match="mismatch"):
        mbq.derive_graph_components(ledger, requirements)


@pytest.mark.parametrize(
    ("include_claim", "expected_terminal"),
    [
        (True, "MECHANISM_BASIS_QUALIFIED"),
        (False, "MECHANISM_BASIS_INSUFFICIENT"),
    ],
)
def test_full_final_chain_supports_both_terminals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    include_claim: bool,
    expected_terminal: str,
) -> None:
    requirements, prior = _synthetic_prior_contract()
    claims = [_synthetic_claim("slot-a", "unused")] if include_claim else []
    ledger = _synthetic_ledger(claims)
    ledger["access_counters"] = {
        "source_content_access": 0,
        "successful_source_content_opens": 0,
        "unqueued_source_attempts": 0,
        "forbidden_value_reads": 0,
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
    }
    ledger["ledger_sha256"] = mbq._self_hash(ledger, "ledger_sha256")
    graph, assessment, decision = _synthetic_final_chain(
        requirements, ledger, prior
    )
    assert decision["status"] == expected_terminal
    _validate_synthetic_final_chain(
        monkeypatch,
        tmp_path,
        requirements,
        ledger,
        graph,
        assessment,
        decision,
        prior,
    )


def test_final_validation_rejects_rehashed_graph_and_coverage_mutations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requirements, prior = _synthetic_prior_contract()
    ledger = _synthetic_ledger([_synthetic_claim("slot-a", "unused")])
    ledger["access_counters"] = {
        "source_content_access": 0,
        "successful_source_content_opens": 0,
        "unqueued_source_attempts": 0,
        "forbidden_value_reads": 0,
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
    }
    ledger["ledger_sha256"] = mbq._self_hash(ledger, "ledger_sha256")
    graph, assessment, decision = _synthetic_final_chain(
        requirements, ledger, prior
    )

    mutated_graph = deepcopy(graph)
    mutated_assessment = deepcopy(assessment)
    mutated_decision = deepcopy(decision)
    claim = mutated_graph["claims"][0]
    claim["paraphrase"] = "Efficacy benefit ranking is proven."
    claim["claim_sha256"] = mbq.canonical_value_sha256(
        {key: value for key, value in claim.items() if key != "claim_sha256"}
    )
    mutated_graph["graph_sha256"] = mbq._self_hash(
        mutated_graph, "graph_sha256"
    )
    mutated_assessment["source_claim_graph"] = mbq._artifact_from_payload(
        mbq.GRAPH_PATH, mutated_graph, "graph_sha256"
    )
    mutated_assessment["assessment_sha256"] = mbq._self_hash(
        mutated_assessment, "assessment_sha256"
    )
    mutated_decision["source_claim_graph"] = mutated_assessment[
        "source_claim_graph"
    ]
    mutated_decision["assessment"] = mbq._artifact_from_payload(
        mbq.ASSESSMENT_PATH, mutated_assessment, "assessment_sha256"
    )
    mutated_decision["decision_sha256"] = mbq._self_hash(
        mutated_decision, "decision_sha256"
    )
    with pytest.raises(mbq.MechanismBasisQualificationError, match="diverges"):
        _validate_synthetic_final_chain(
            monkeypatch,
            tmp_path,
            requirements,
            ledger,
            mutated_graph,
            mutated_assessment,
            mutated_decision,
            prior,
        )

    graph, assessment, decision = _synthetic_final_chain(
        requirements, ledger, prior
    )
    assessment["requirements"][0]["source_slot_uids"] = []
    assessment["assessment_sha256"] = mbq._self_hash(
        assessment, "assessment_sha256"
    )
    decision["assessment"] = mbq._artifact_from_payload(
        mbq.ASSESSMENT_PATH, assessment, "assessment_sha256"
    )
    decision["decision_sha256"] = mbq._self_hash(decision, "decision_sha256")
    with pytest.raises(
        mbq.MechanismBasisQualificationError, match="coverage diverges"
    ):
        _validate_synthetic_final_chain(
            monkeypatch,
            tmp_path,
            requirements,
            ledger,
            graph,
            assessment,
            decision,
            prior,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "canonical_unsupported",
        "canonical_contradictory",
        "canonical_role_uid",
        "canonical_required",
        "sensitivity_supported",
        "sensitivity_contradictory",
        "sensitivity_role_uid",
        "sensitivity_required",
        "sensitivity_order",
    ],
)
def test_final_validation_rejects_rehashed_prior_support_mutations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    requirements, prior = _synthetic_prior_contract()
    ledger = _synthetic_ledger([_synthetic_claim("slot-a", "unused")])
    ledger["access_counters"] = {
        "source_content_access": 0,
        "successful_source_content_opens": 0,
        "unqueued_source_attempts": 0,
        "forbidden_value_reads": 0,
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
    }
    ledger["ledger_sha256"] = mbq._self_hash(ledger, "ledger_sha256")
    graph, assessment, decision = _synthetic_final_chain(
        requirements, ledger, prior
    )
    if mutation == "canonical_unsupported":
        assessment["canonical_prior_support"]["support_status"] = "unsupported"
    elif mutation == "canonical_contradictory":
        assessment["canonical_prior_support"]["support_status"] = "contradictory"
    elif mutation == "canonical_role_uid":
        assessment["canonical_prior_support"]["role_uid"] = "other-canonical"
    elif mutation == "canonical_required":
        assessment["canonical_prior_support"]["required"] = False
    elif mutation == "sensitivity_supported":
        assessment["sensitivity_prior_support"][0]["support_status"] = "supported"
    elif mutation == "sensitivity_contradictory":
        assessment["sensitivity_prior_support"][0]["support_status"] = "contradictory"
    elif mutation == "sensitivity_role_uid":
        assessment["sensitivity_prior_support"][0]["role_uid"] = "other-sensitivity"
    elif mutation == "sensitivity_required":
        assessment["sensitivity_prior_support"][0]["required"] = True
    else:
        assessment["sensitivity_prior_support"].reverse()
    _rebind_assessment_and_decision(assessment, decision)
    with pytest.raises(mbq.MechanismBasisQualificationError, match="prior support"):
        _validate_synthetic_final_chain(
            monkeypatch,
            tmp_path,
            requirements,
            ledger,
            graph,
            assessment,
            decision,
            prior,
        )


@pytest.mark.parametrize(
    "forbidden_term",
    ["efficacy", "benefit", "outcome", "ranking", "recommendation"],
)
def test_final_validation_rejects_rehashed_forbidden_interpretation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    forbidden_term: str,
) -> None:
    requirements, prior = _synthetic_prior_contract()
    ledger = _synthetic_ledger([_synthetic_claim("slot-a", "unused")])
    ledger["access_counters"] = {
        "source_content_access": 0,
        "successful_source_content_opens": 0,
        "unqueued_source_attempts": 0,
        "forbidden_value_reads": 0,
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
    }
    ledger["ledger_sha256"] = mbq._self_hash(ledger, "ledger_sha256")
    graph, assessment, decision = _synthetic_final_chain(
        requirements, ledger, prior
    )
    decision["scientific_interpretation"] = f"{forbidden_term} is proven."
    decision["decision_sha256"] = mbq._self_hash(decision, "decision_sha256")
    with pytest.raises(mbq.MechanismBasisQualificationError, match="forbidden"):
        _validate_synthetic_final_chain(
            monkeypatch,
            tmp_path,
            requirements,
            ledger,
            graph,
            assessment,
            decision,
            prior,
        )


@pytest.mark.parametrize("include_claim", [True, False])
def test_final_validation_rejects_terminal_mismatched_interpretation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, include_claim: bool
) -> None:
    requirements, prior = _synthetic_prior_contract()
    claims = [_synthetic_claim("slot-a", "unused")] if include_claim else []
    ledger = _synthetic_ledger(claims)
    ledger["access_counters"] = {
        "source_content_access": 0,
        "successful_source_content_opens": 0,
        "unqueued_source_attempts": 0,
        "forbidden_value_reads": 0,
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
    }
    ledger["ledger_sha256"] = mbq._self_hash(ledger, "ledger_sha256")
    graph, assessment, decision = _synthetic_final_chain(
        requirements, ledger, prior
    )
    other_terminal = (
        "MECHANISM_BASIS_INSUFFICIENT"
        if decision["status"] == "MECHANISM_BASIS_QUALIFIED"
        else "MECHANISM_BASIS_QUALIFIED"
    )
    decision["scientific_interpretation"] = (
        mbq.scientific_interpretation_for_terminal(other_terminal)
    )
    decision["decision_sha256"] = mbq._self_hash(decision, "decision_sha256")
    with pytest.raises(mbq.MechanismBasisQualificationError, match="terminal mismatch"):
        _validate_synthetic_final_chain(
            monkeypatch,
            tmp_path,
            requirements,
            ledger,
            graph,
            assessment,
            decision,
            prior,
        )


@pytest.mark.parametrize(
    ("artifact_name", "field"),
    [
        ("graph", "contract"),
        ("graph", "schema"),
        ("graph", "evidence_ledger"),
        ("assessment", "contract"),
        ("assessment", "schema"),
        ("assessment", "authorization"),
        ("assessment", "evidence_ledger"),
        ("assessment", "source_claim_graph"),
        ("decision", "contract"),
        ("decision", "schema"),
        ("decision", "authorization"),
        ("decision", "evidence_ledger"),
        ("decision", "source_claim_graph"),
        ("decision", "assessment"),
    ],
)
def test_final_validation_rejects_fully_rehashed_binding_mutations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifact_name: str,
    field: str,
) -> None:
    requirements, prior = _synthetic_prior_contract()
    ledger = _synthetic_ledger([_synthetic_claim("slot-a", "unused")])
    ledger["access_counters"] = {
        "source_content_access": 0,
        "successful_source_content_opens": 0,
        "unqueued_source_attempts": 0,
        "forbidden_value_reads": 0,
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
    }
    ledger["ledger_sha256"] = mbq._self_hash(ledger, "ledger_sha256")
    graph, assessment, decision = _synthetic_final_chain(
        requirements, ledger, prior
    )
    artifacts = {
        "graph": graph,
        "assessment": assessment,
        "decision": decision,
    }
    target = artifacts[artifact_name]
    if field == "contract":
        target[field] = deepcopy(target["schema"])
    elif field == "schema":
        target[field] = deepcopy(target["contract"])
    elif field == "authorization":
        target[field] = deepcopy(target["evidence_ledger"])
    elif field == "evidence_ledger":
        target[field] = {
            "path": "Input/processed/mechanistic_belief/wrong_ledger.json",
            "file_sha256": "e" * 64,
            "semantic_sha256": "f" * 64,
        }
    elif field == "source_claim_graph":
        target[field] = deepcopy(target["authorization"])
    else:
        target[field] = deepcopy(target["source_claim_graph"])
    if artifact_name == "graph":
        _rebind_graph_and_downstream(graph, assessment, decision)
    elif artifact_name == "assessment":
        _rebind_assessment_and_decision(assessment, decision)
    else:
        decision["decision_sha256"] = mbq._self_hash(
            decision, "decision_sha256"
        )
    with pytest.raises(mbq.MechanismBasisQualificationError, match="binding drift"):
        _validate_synthetic_final_chain(
            monkeypatch,
            tmp_path,
            requirements,
            ledger,
            graph,
            assessment,
            decision,
            prior,
        )


@pytest.mark.parametrize(
    ("artifact_name", "field"),
    [
        ("authorization", "contract"),
        ("authorization", "schema"),
        ("authorization", "upstream_authorities"),
        ("authorization", "queue"),
        ("authorization", "chronology"),
        ("ledger", "contract"),
        ("ledger", "schema"),
        ("ledger", "authorization"),
        ("ledger", "queue"),
    ],
)
def test_upstream_binding_validators_reject_fully_rehashed_mutations(
    monkeypatch: pytest.MonkeyPatch, artifact_name: str, field: str
) -> None:
    schemas = _synthetic_schema_bindings()
    loaded = {
        path: ({}, f"upstream-{index}".encode())
        for index, path in enumerate(mbq.EXPECTED_UPSTREAM)
    }
    authorities = {
        "contract": {"retrieval_policy": {}},
        "contract_raw": b"synthetic-contract",
        "loaded": loaded,
        "queue_raw": loaded[mbq.SOURCE_QUEUE_PATH][1],
        "queue": {
            "queue_sha256": mbq.EXPECTED_UPSTREAM[mbq.SOURCE_QUEUE_PATH][2],
            "slots": [],
        },
        "chronology_raw": loaded[mbq.SOURCE_CHRONOLOGY_PATH][1],
        "chronology": {
            "chronology_sha256": mbq.EXPECTED_UPSTREAM[
                mbq.SOURCE_CHRONOLOGY_PATH
            ][2]
        },
    }
    authorization = {
        "contract": mbq._contract_binding(authorities),
        "schema": schemas[mbq.AUTHORIZATION_SCHEMA_PATH],
        "upstream_authorities": mbq._upstream_authority_bindings(authorities),
        "queue": mbq._artifact_binding(
            mbq.SOURCE_QUEUE_PATH,
            authorities["queue_raw"],
            authorities["queue"]["queue_sha256"],
        ),
        "chronology": mbq._artifact_binding(
            mbq.SOURCE_CHRONOLOGY_PATH,
            authorities["chronology_raw"],
            authorities["chronology"]["chronology_sha256"],
        ),
    }
    authorization["authorization_sha256"] = mbq._self_hash(
        authorization, "authorization_sha256"
    )
    ledger = {
        "contract": mbq._contract_binding(authorities),
        "schema": schemas[mbq.LEDGER_SCHEMA_PATH],
        "authorization": mbq._artifact_from_payload(
            mbq.AUTHORIZATION_PATH, authorization, "authorization_sha256"
        ),
        "queue": mbq._artifact_binding(
            mbq.SOURCE_QUEUE_PATH,
            authorities["queue_raw"],
            authorities["queue"]["queue_sha256"],
        ),
    }
    ledger["ledger_sha256"] = mbq._self_hash(ledger, "ledger_sha256")
    target = deepcopy(authorization if artifact_name == "authorization" else ledger)
    if field == "contract":
        target[field] = deepcopy(target["schema"])
    elif field == "schema":
        target[field] = deepcopy(target["contract"])
    elif field == "upstream_authorities":
        target[field] = list(reversed(target[field]))
    elif field == "queue" and artifact_name == "authorization":
        target[field] = deepcopy(target["chronology"])
    elif field == "chronology":
        target[field] = deepcopy(target["queue"])
    elif field == "authorization":
        target[field] = deepcopy(target["queue"])
    else:
        target[field] = deepcopy(target["authorization"])
    hash_field = (
        "authorization_sha256"
        if artifact_name == "authorization"
        else "ledger_sha256"
    )
    target[hash_field] = mbq._self_hash(target, hash_field)
    monkeypatch.setattr(mbq, "load_frozen_authorities", lambda *_args, **_kwargs: authorities)
    monkeypatch.setattr(mbq, "_schema_bindings", lambda *_args, **_kwargs: schemas)
    monkeypatch.setattr(
        mbq,
        "_load_output",
        lambda *_args, **_kwargs: (target, mbq.canonical_json_bytes(target)),
    )
    if artifact_name == "ledger":
        monkeypatch.setattr(mbq, "validate_authorization", lambda **_: authorization)
    validator = (
        mbq.validate_authorization
        if artifact_name == "authorization"
        else mbq.validate_evidence_ledger
    )
    with pytest.raises(mbq.MechanismBasisQualificationError, match="binding drift"):
        validator(repository_root=REPO_ROOT)


def _canonical_artifact_chain() -> dict[str, dict]:
    return {
        path: json.loads((REPO_ROOT / path).read_text())
        for path in (
            mbq.AUTHORIZATION_PATH,
            mbq.LEDGER_PATH,
            mbq.GRAPH_PATH,
            mbq.ASSESSMENT_PATH,
            mbq.DECISION_PATH,
        )
    }


def _rebind_canonical_chain(artifacts: dict[str, dict], start: str) -> None:
    authorization = artifacts[mbq.AUTHORIZATION_PATH]
    ledger = artifacts[mbq.LEDGER_PATH]
    graph = artifacts[mbq.GRAPH_PATH]
    assessment = artifacts[mbq.ASSESSMENT_PATH]
    decision = artifacts[mbq.DECISION_PATH]
    if start == "authorization":
        authorization["authorization_sha256"] = mbq._self_hash(
            authorization, "authorization_sha256"
        )
        ledger["authorization"] = mbq._artifact_from_payload(
            mbq.AUTHORIZATION_PATH, authorization, "authorization_sha256"
        )
        start = "ledger"
    if start == "ledger":
        ledger["ledger_sha256"] = mbq._self_hash(ledger, "ledger_sha256")
        graph["evidence_ledger"] = mbq._artifact_from_payload(
            mbq.LEDGER_PATH, ledger, "ledger_sha256"
        )
        assessment["evidence_ledger"] = graph["evidence_ledger"]
        decision["evidence_ledger"] = graph["evidence_ledger"]
        start = "graph"
    if start == "graph":
        graph["graph_sha256"] = mbq._self_hash(graph, "graph_sha256")
        binding = mbq._artifact_from_payload(mbq.GRAPH_PATH, graph, "graph_sha256")
        assessment["source_claim_graph"] = binding
        decision["source_claim_graph"] = binding
        start = "assessment"
    if start == "assessment":
        assessment["assessment_sha256"] = mbq._self_hash(
            assessment, "assessment_sha256"
        )
        decision["assessment"] = mbq._artifact_from_payload(
            mbq.ASSESSMENT_PATH, assessment, "assessment_sha256"
        )
    decision["decision_sha256"] = mbq._self_hash(decision, "decision_sha256")


def _validate_mutated_canonical_chain(
    monkeypatch: pytest.MonkeyPatch, artifacts: dict[str, dict]
) -> None:
    real_load = mbq._load_output
    authorities = mbq.load_frozen_authorities(REPO_ROOT)
    schemas = mbq._schema_bindings(REPO_ROOT)
    monkeypatch.setattr(mbq, "load_frozen_authorities", lambda *_a, **_k: authorities)
    monkeypatch.setattr(mbq, "_schema_bindings", lambda *_a, **_k: schemas)

    def load_override(root: Path, relative: str, schema: str, hash_field: str):
        if relative in artifacts:
            payload = artifacts[relative]
            return payload, mbq.canonical_json_bytes(payload)
        return real_load(root, relative, schema, hash_field)

    monkeypatch.setattr(mbq, "_load_output", load_override)
    mbq.validate_final(repository_root=REPO_ROOT)


@pytest.mark.parametrize(
    "field",
    [
        "queue_order", "slot_uid", "canonical_intervention_uid", "validation_uid",
        "family_uid", "canonical_source_identity", "pmid", "doi", "official_date",
        "queue_disposition", "public_access_status", "license_status", "request_uid",
        "request_kind", "method", "url", "host", "allowed_reorder", "allowed_add",
        "allowed_drop", "allowed_substitute", "denied_reorder", "denied_add",
        "denied_drop", "denied_substitute",
    ],
)
def test_authorization_slot_and_policy_mutation_matrix(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    artifacts = _canonical_artifact_chain()
    authorization = artifacts[mbq.AUTHORIZATION_PATH]
    row = authorization["eligible_slots"][0]
    request = row["request_plan"][0]
    if field.startswith("allowed_") or field.startswith("denied_"):
        key, operation = field.split("_", 1)
        values = authorization[f"{key}_content_classes"]
        if operation == "reorder":
            values.reverse()
        elif operation == "add":
            values.append("fabricated_class")
        elif operation == "drop":
            values.pop()
        else:
            values[0] = "fabricated_class"
    elif field in request:
        request[field] = "POST" if field == "method" else "fabricated"
    else:
        row[field] = row[field] + 1 if isinstance(row[field], int) else "fabricated"
    _rebind_canonical_chain(artifacts, "authorization")
    with pytest.raises(mbq.MechanismBasisQualificationError):
        _validate_mutated_canonical_chain(monkeypatch, artifacts)


@pytest.mark.parametrize(
    "field",
    [
        "queue_order", "slot_uid", "queue_slot_sha256", "canonical_intervention_uid",
        "validation_uid", "family_uid", "canonical_source_identity", "official_date",
        "queue_disposition", "receipt_uid", "attempt_ordinal", "request_uid",
        "request_kind", "method", "url", "retrieved_at", "final_host", "content_type",
        "response_bytes", "response_sha256", "parser_version", "public_access_status",
        "license_status", "disposition", "allowed_sections", "forbidden_sections_skipped",
        "mixed_sections_rejected", "roster_reorder", "roster_drop", "roster_duplicate",
        "row_receipt_rebound", "claim_locator", "claim_hash",
    ],
)
def test_ledger_row_receipt_and_linkage_mutation_matrix(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    artifacts = _canonical_artifact_chain()
    ledger = artifacts[mbq.LEDGER_PATH]
    row = ledger["slots"][0]
    receipt = ledger["request_receipts"][0]
    if field.startswith("roster_"):
        if field == "roster_reorder":
            ledger["request_receipts"][0:2] = reversed(
                ledger["request_receipts"][0:2]
            )
        elif field == "roster_drop":
            ledger["request_receipts"].pop()
        else:
            ledger["request_receipts"].append(deepcopy(receipt))
    elif field == "row_receipt_rebound":
        next(r for r in ledger["slots"] if r["content_receipt"] is not None)["content_receipt"] = deepcopy(receipt)
    elif field in {"claim_locator", "claim_hash"}:
        claim = next(c for r in ledger["slots"] for c in r["semantic_claims"])
        claim["content_locator" if field == "claim_locator" else "content_sha256"] = "fabricated"
        claim["claim_uid"] = mbq._claim_uid(next(r["slot_uid"] for r in ledger["slots"] if claim in r["semantic_claims"]), {k:v for k,v in claim.items() if k!="claim_uid"})
    elif field in receipt:
        receipt[field] = receipt[field] + 1 if isinstance(receipt[field], int) else ([{"locator": "fabricated", "section_label": "BACKGROUND", "content_class": "mechanism_definition", "content_sha256": "f" * 64}] if isinstance(receipt[field], list) else "fabricated")
        if field != "receipt_uid" and receipt.get("response_sha256") is not None:
            receipt["receipt_uid"] = mbq._receipt_uid(receipt)
    else:
        row[field] = row[field] + 1 if isinstance(row[field], int) else "fabricated"
    _rebind_canonical_chain(artifacts, "ledger")
    with pytest.raises((mbq.MechanismBasisQualificationError, KeyError)):
        _validate_mutated_canonical_chain(monkeypatch, artifacts)


def test_cumulative_receipts_and_immutable_installation(tmp_path: Path) -> None:
    authorized = [
        {"request_plan": [{"request_uid": f"request-{index:02d}"}]}
        for index in range(15)
    ]
    receipts = [
        {
            "request_uid": f"request-{index:02d}",
            "attempt_ordinal": attempt,
            "response_sha256": f"{index:02x}" * 32,
        }
        for index in range(15)
        for attempt in (1, 2)
    ]
    mbq.validate_operational_review(receipts, authorized)
    ledger_path = tmp_path / "mechanism_evidence_ledger.json"
    original = mbq.canonical_json_bytes({"request_receipts": receipts})
    assert mbq._install_many([(ledger_path, original)])
    assert not mbq._install_many([(ledger_path, original)])
    changed_receipts = deepcopy(receipts)
    changed_receipts[0]["response_sha256"] = "f" * 64
    changed = mbq.canonical_json_bytes({"request_receipts": changed_receipts})
    with pytest.raises(mbq.MechanismBasisQualificationError, match="overwrite"):
        mbq._install_many([(ledger_path, changed)])
    persisted = json.loads(ledger_path.read_bytes())
    assert len(persisted["request_receipts"]) == 30
    assert persisted["request_receipts"] == receipts


def test_semantic_packet_finalization_cannot_replace_existing_ledger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger_path = tmp_path / mbq.LEDGER_PATH
    ledger_path.parent.mkdir(parents=True)
    receipts = [
        {
            "request_uid": f"request-{index:02d}",
            "attempt_ordinal": attempt,
            "response_sha256": f"{index:02x}" * 32,
        }
        for index in range(15)
        for attempt in (1, 2)
    ]
    original = mbq.canonical_json_bytes({"request_receipts": receipts})
    ledger_path.write_bytes(original)
    packet_path = tmp_path / "semantic-review-packet.json"
    packet_path.write_bytes(
        mbq.canonical_json_bytes(
            {
                "schema": "node_local_semantic_review_packet_v1",
                "projections": [
                    {
                        "slot_uid": "slot-a",
                        "projection": {"allowed_sections": []},
                    }
                ],
                "request_receipts": [
                    {
                        "request_uid": "request-a",
                        "attempt_ordinal": 1,
                        "response_sha256": "a" * 64,
                        "response_bytes": 1,
                    }
                ],
            }
        )
    )
    authorities = {
        "contract_raw": b"contract",
        "queue_raw": b"queue",
        "queue": {
            "queue_sha256": "q" * 64,
            "slots": [{"slot_uid": "slot-a", "disposition": "eligible"}],
        },
    }
    manifest = {
        "authorization_sha256": "a" * 64,
        "eligible_slots": [
            {"slot_uid": "slot-a", "request_plan": [{"request_uid": "request-a"}]}
        ],
        "requirement_registry": [],
    }
    semantic_review = {
        "grade": "CONTEXT_ONLY",
        "semantic_claims": [],
        "claims_supported": [],
    }
    monkeypatch.setattr(mbq, "load_frozen_authorities", lambda *_: authorities)
    monkeypatch.setattr(mbq, "validate_authorization", lambda **_: manifest)
    monkeypatch.setattr(mbq, "_base_evidence_row", lambda _: {})
    monkeypatch.setattr(mbq, "_semantic_review", lambda *_: semantic_review)
    monkeypatch.setattr(mbq, "derive_graph_components", lambda *_: ([], [], []))
    monkeypatch.setattr(mbq, "_schema", lambda *_: ({}, "s" * 64))
    monkeypatch.setattr(mbq, "_schema_validate", lambda *_args, **_kwargs: None)
    with pytest.raises(mbq.MechanismBasisQualificationError, match="overwrite"):
        mbq.finalize_semantic_review_packet(
            repository_root=tmp_path, packet_path=packet_path
        )
    assert ledger_path.read_bytes() == original
    assert len(json.loads(ledger_path.read_bytes())["request_receipts"]) == 30
    assert packet_path.exists()
