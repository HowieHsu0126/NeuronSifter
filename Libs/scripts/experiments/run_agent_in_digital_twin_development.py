"""Run a bounded development-only Agent-in-Digital-Twin smoke campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.models.modules.agent_in_twin_runtime import TwinGroundedReActController
from Libs.models.modules.coevolution_runtime import AgentTwinCoEvolutionRuntime, CoEvolutionVariant
from Libs.models.modules.decision_consistent_runtime import (
    DecisionConsistentTwin,
    InformationAction,
    ToolResult,
    WorldAction,
)
from Libs.models.modules.scientific_tool_fabric import ScientificToolFabric
from Libs.models.modules.intervention_optimizer import InterventionOptimizer, OptimizerWeights
from Libs.scripts.validators.validate_agent_in_digital_twin_development import validate
from Libs.utils.data.raw_data_registry import load_config


DEFAULT_OUTPUT = REPO_ROOT / "Output/results/agent_in_digital_twin_development"
TOOL_ID = "evidence.assay.replay"


def _world_actions() -> list[WorldAction]:
    return [
        WorldAction(
            target_set=("amyloid_beta",),
            drug_set=("candidate_alpha",),
            dose=1.0,
            schedule="q24h",
            route="oral",
            modality="pharmacological",
            duration="14d",
            biological_context={"safety_risk": 0.1},
        ),
        WorldAction(
            target_set=("neuroinflammation",),
            drug_set=("candidate_beta",),
            dose=0.5,
            schedule="q48h",
            route="iv",
            modality="pharmacological",
            duration="7d",
            biological_context={"safety_risk": 0.2},
        ),
    ]


def _information_action() -> InformationAction:
    return InformationAction(
        tool_id=TOOL_ID,
        query={"expected_decision_relevance": 0.8, "tool_reliability": 0.95, "assay": "functional_replay"},
        fidelity="development_replay",
        expected_cost=1.0,
    )


def _new_twin() -> DecisionConsistentTwin:
    def rollout(posterior: Any, action: WorldAction) -> dict[str, Any]:
        dose = float(action.dose)
        baseline = float(posterior.state.get("baseline", 0.0))
        benefit = baseline + (1.0 if action.drug_set == ("candidate_alpha",) else 0.7) * dose
        return {
            "trajectory": [baseline, benefit],
            "uncertainty": {"epistemic": 0.08, "aleatoric": 0.04},
            "diagnostics": {"benefit": benefit, "safety_risk": action.biological_context.get("safety_risk", 0.0), "status": "development_simulator"},
        }

    twin = DecisionConsistentTwin(rollout_fn=rollout)
    twin.infer_belief(
        {"state": {"baseline": 0.2}, "uncertainty": {"epistemic": 0.2, "aleatoric": 0.1}, "support": {"support_score": 1.0}, "reliability": {"tri_score": 1.0}},
        {"regions": ["hippocampus"], "compartments": ["isf"]},
    )
    return twin


def run_smoke(output_root: str | Path | None = None) -> dict[str, Any]:
    contract_report = validate()
    runtime_config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    contract = runtime_config["agent_in_digital_twin"]
    controller_config = contract["controller"]
    objective_config = contract["intervention_objective"]
    optimizer = InterventionOptimizer(
        weights=OptimizerWeights(
            uncertainty=float(objective_config["uncertainty_weight"]),
            safety=float(objective_config["safety_weight"]),
            cost=float(objective_config["cost_weight"]),
            violation=float(objective_config["violation_weight"]),
            support=float(objective_config["support_weight"]),
        )
    )
    fabric = ScientificToolFabric()

    def assay_handler(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "observation": {"functional_rescue_proxy": 0.65, "assay": payload.get("information_action", {}).get("tool_id", TOOL_ID)},
            "uncertainty": {"epistemic": 0.05, "aleatoric": 0.05},
            "support": {"support_score": 0.9},
            "evidence_grade": "development_diagnostic",
            "cost": 1.0,
            "runtime": 0.01,
        }

    fabric.register(TOOL_ID, boundary="neuronsifter-evidence-mcp", handler=assay_handler, version="development-replay-1")
    twin = _new_twin()
    controller = TwinGroundedReActController(
        twin,
        tools=fabric,
        optimizer=optimizer,
        controller_config=controller_config,
    )
    agent_result = controller.run("optimize CNS functional rescue", _world_actions(), [_information_action()])

    coevolution_runs: dict[str, Any] = {}
    for variant in CoEvolutionVariant:
        local_twin = _new_twin()
        runtime = AgentTwinCoEvolutionRuntime(local_twin, initial_agent_policy_uid="agent-policy-development")
        info = _information_action()

        def select(round_index: int, posterior: Any, policy_uid: str) -> dict[str, Any]:
            return {
                "evidence_uids": (f"development-evidence-{round_index}",),
                "calibration_drift": 0.01,
                "old_domain_retention": 0.99,
                "tool_result": ToolResult(
                    posterior_uid=posterior.posterior_uid,
                    action_uid=str(info.action_uid),
                    status="completed",
                    observation={"round": round_index, "policy_uid": policy_uid},
                    uncertainty={"epistemic": 0.03},
                ),
            }

        result = runtime.run(variant, rounds=3, select_evidence=select)
        coevolution_runs[variant.value] = result.to_record()

    report = {
        "schema_version": "neuronsifter-agent-in-digital-twin-development-smoke-v1",
        "development_only": True,
        "publication_claim_eligible": False,
        "formal_evidence_gate": contract_report["formal_evidence_gate"],
        "agent_run": agent_result.to_record(),
        "coevolution": coevolution_runs,
        "posterior_lineage": [row for row in twin.decisions.trace if row.get("operation") in {"rollout", "assimilate_observation", "rank_interventions", "value_information_actions"}],
    }
    if output_root is not None:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "development_smoke_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    report = run_smoke(None if args.no_write else args.output_root)
    print(json.dumps({"status": "passed", "development_only": report["development_only"], "agent_rounds": len(report["agent_run"]["decisions"]), "coevolution_variants": len(report["coevolution"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
