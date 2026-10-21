"""Preregister or materialize the frozen metadata-only mechanism-source queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Libs.utils.data.mechanism_source_queue import (
    CONTRACT_PATH,
    MechanismSourceQueueError,
    build_preregistration,
    materialize_queue,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", default=CONTRACT_PATH)
    parser.add_argument("--mode", choices=("preregister", "materialize"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "preregister":
            preregistration, created = build_preregistration(
                repository_root=args.repository_root,
                contract_path=args.contract,
            )
            result = {
                "status": preregistration["status"],
                "created": created,
                "preregistration_sha256": preregistration["preregistration_sha256"],
                "context_count": len(preregistration["context_action_projections"]),
                "family_count": len(preregistration["source_families"]),
                "request_plan_count": len(preregistration["expanded_requests"]),
                "network_requests_performed": 0,
                "access_counters": preregistration["access_counters"],
            }
        else:
            queue, chronology, authorization, created = materialize_queue(
                repository_root=args.repository_root,
                contract_path=args.contract,
            )
            result = {
                "status": queue["status"],
                "created": created,
                "queue_sha256": queue["queue_sha256"],
                "chronology_sha256": chronology["chronology_sha256"],
                "authorization_sha256": authorization["authorization_sha256"],
                "slot_count": queue["counts"]["slot_count"],
                "request_count": queue["network_summary"]["request_count"],
                "dispositions": queue["counts"]["dispositions"],
                "content_access_authorized": authorization["content_access_authorized"],
                "access_counters": queue["access_counters"],
            }
    except MechanismSourceQueueError as exc:
        print(json.dumps({"status": "MECHANISM_SOURCE_QUEUE_BUILD_FAILED", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
