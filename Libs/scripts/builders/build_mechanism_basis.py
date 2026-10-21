"""Build bounded mechanism-basis authorization, review, and adjudication artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Libs.utils.data.mechanism_basis_qualification import (
    CONTRACT_PATH,
    MechanismBasisQualificationError,
    adjudicate_mechanism_basis,
    build_content_authorization,
    review_mechanism_basis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", default=CONTRACT_PATH)
    parser.add_argument("--mode", choices=("authorize", "review", "adjudicate"), required=True)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--projection-packet", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "authorize":
            manifest, created = build_content_authorization(
                repository_root=args.repository_root, contract_path=args.contract
            )
            result = {
                "status": manifest["status"],
                "created": created,
                "authorization_sha256": manifest["authorization_sha256"],
                "eligible_slot_count": len(manifest["eligible_slots"]),
                "requirement_count": len(manifest["requirement_registry"]),
                "network_requests_performed": 0,
                "access_counters": manifest["access_counters"],
            }
        elif args.mode == "review":
            if args.projection_packet is not None:
                from Libs.utils.data.mechanism_basis_qualification import finalize_semantic_review_packet

                ledger, created = finalize_semantic_review_packet(
                    repository_root=args.repository_root,
                    contract_path=args.contract,
                    packet_path=args.projection_packet,
                )
            else:
                if args.scratch_dir is None:
                    raise MechanismBasisQualificationError("review mode requires --scratch-dir")
                ledger, created = review_mechanism_basis(
                    repository_root=args.repository_root,
                    contract_path=args.contract,
                    scratch_dir=args.scratch_dir,
                )
            result = {
                "status": ledger["status"],
                "created": created,
                "ledger_sha256": ledger["ledger_sha256"],
                "counts": ledger["counts"],
                "network_summary": ledger["network_summary"],
                "access_counters": ledger["access_counters"],
            }
        else:
            graph, assessment, decision, created = adjudicate_mechanism_basis(
                repository_root=args.repository_root, contract_path=args.contract
            )
            result = {
                "status": decision["status"],
                "created": created,
                "graph_sha256": graph["graph_sha256"],
                "assessment_sha256": assessment["assessment_sha256"],
                "decision_sha256": decision["decision_sha256"],
                "missing_required_count": len(decision["missing_required_requirement_uids"]),
                "contradictory_required_count": len(decision["contradictory_required_requirement_uids"]),
                "access_counters": decision["access_counters"],
            }
    except MechanismBasisQualificationError as exc:
        print(json.dumps({"status": "MECHANISM_BASIS_BUILD_FAILED", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
