from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from Libs.utils.data.claim_gated_wording import audit_document_text, build_claim_wording_audit
from Libs.utils.data.raw_data_registry import load_config


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"


def findings(text: str, *, role: str = "current_public", path: str = "README.md") -> list[dict[str, object]]:
    config = load_config(CONFIG_PATH)
    return audit_document_text(document_path=path, document_role=role, text=text, config=config)


def test_target_wording_is_allowed_before_c1() -> None:
    rows = findings("NeuronSifter targets a public-data CNS foundation digital twin architecture.")
    assert rows and all(row["status"] == "allowed" for row in rows)


def test_completed_foundation_wording_is_rejected_before_c1() -> None:
    rows = findings("NeuronSifter is a validated CNS foundation model.")
    assert any(row["status"] == "violation" for row in rows)
    assert any(row["category"] == "prohibited_completion_language" for row in rows)


def test_ambiguous_present_tense_is_rejected_before_c1() -> None:
    rows = findings("NeuronSifter is a CNS foundation model.")
    assert any(row["status"] == "violation" for row in rows)
    assert any(row["category"] == "missing_pending_gate_qualification" for row in rows)


def test_gate_qualification_and_contract_definition_are_allowed() -> None:
    qualified = findings("The C1 gate is pending; NeuronSifter targets a foundation model architecture.")
    contract = findings("C4 TRI-gated reliable screening requires its gate evidence.", role="design_contract", path="Docs/design/iclr/07_REVISED_CLAIM_EXPERIMENT_MATRIX.md")
    assert qualified and all(row["status"] == "allowed" for row in qualified)
    assert contract and all(row["status"] == "allowed" for row in contract)


def test_historical_quote_is_allowed() -> None:
    rows = findings(
        "Historical quotation: NeuronSifter is a validated foundation model.",
        role="design_contract",
        path="Docs/design/iclr/CHANGELOG.md",
    )
    assert rows and all(row["status"] == "allowed" for row in rows)


def test_passed_gate_allows_result_wording() -> None:
    config = deepcopy(load_config(CONFIG_PATH))
    for claim in config["iclr_design_contract_union"]["claims"]:
        if claim["claim_id"] == "C1":
            claim["result_status"] = "completed_empirical"
            claim["claim_gate_status"] = "manuscript_allowed"
    rows = audit_document_text(
        document_path="README.md",
        document_role="current_public",
        text="NeuronSifter is a validated foundation model.",
        config=config,
    )
    assert rows and all(row["status"] == "allowed" for row in rows)


def test_cost_and_clinical_causality_overclaims_are_rejected() -> None:
    rows = findings("NeuronSifter reduces wet-lab cost and provides clinical prediction of causal effect.")
    violations = [row for row in rows if row["status"] == "violation"]
    assert {row["gated_term"] for row in violations} >= {"reduces wet-lab cost", "clinical prediction", "causal effect"}


def test_current_public_documents_pass_claim_wording_audit() -> None:
    report, _ = build_claim_wording_audit(CONFIG_PATH)
    assert report["status"] == "pass"
    assert report["violation_count"] == 0
