#!/usr/bin/env bash
set -euo pipefail
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_profile_suite.sh" neurodegeneration_without_pkpd_prior "${1:-}"
