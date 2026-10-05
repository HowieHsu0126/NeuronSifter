#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/share/home/anonymous/Researches/NeuronSifter"
CANDIDATE_ROOT="/share/home/anonymous/Researches/.baseline_first_candidates"
HISTORICAL_SOURCE_SNAPSHOT="${CANDIDATE_ROOT}/candidate_baseline_first_primary_source_snapshot_v5"
HISTORICAL_BASE_SNAPSHOT="${CANDIDATE_ROOT}/candidate_baseline_first_primary_source_snapshot_v2"
SOURCE="${CANDIDATE_ROOT}/baseline_first_source_composite"
CANDIDATE="${CANDIDATE_ROOT}/candidate_baseline_first_primary_source_snapshot"

test -d "${HISTORICAL_SOURCE_SNAPSHOT}/.git"
test -d "${HISTORICAL_BASE_SNAPSHOT}/.git"
test ! -e "${SOURCE}"
test ! -e "${CANDIDATE}"

mkdir "${SOURCE}" "${CANDIDATE}"
git -C "${HISTORICAL_SOURCE_SNAPSHOT}" archive --format=tar HEAD | tar -xf - -C "${SOURCE}"
git -C "${HISTORICAL_BASE_SNAPSHOT}" archive --format=tar HEAD | tar -xf - -C "${CANDIDATE}"

cp "${REPO_ROOT}/PROJECT_PREFIX" "${SOURCE}/PROJECT_PREFIX"
cp "${REPO_ROOT}/Libs/configs/schema/evidence_union.schema.json" \
  "${SOURCE}/Libs/configs/schema/evidence_union.schema.json"

git -C "${SOURCE}" init -q
git -C "${SOURCE}" config user.email baseline_first-source@example.invalid
git -C "${SOURCE}" config user.name "NeuronSifter BASELINE_FIRST source"
git -C "${SOURCE}" add --all
git -C "${SOURCE}" add --force -- PROJECT_PREFIX \
  Libs/configs/schema/evidence_union.schema.json
git -C "${SOURCE}" commit --no-verify -q \
  -m "BASELINE_FIRST: source composite with missing runtime inputs"

git -C "${CANDIDATE}" init -q
git -C "${CANDIDATE}" config user.email baseline_first-candidate@example.invalid
git -C "${CANDIDATE}" config user.name "NeuronSifter BASELINE_FIRST candidate"
git -C "${CANDIDATE}" add --all
git -C "${CANDIDATE}" commit --no-verify -q \
  -m "BASELINE_FIRST: candidate base source snapshot"

python "${REPO_ROOT}/Libs/scripts/builders/materialize_mainline_candidate.py" \
  --source-root "${SOURCE}" \
  --candidate-root "${CANDIDATE}" \
  --manifest-out "${REPO_ROOT}/Output/audits/mainline_candidate_materialization.json" \
  --commit-message "BASELINE_FIRST: materialize runtime contract inputs"

test -z "$(git -C "${SOURCE}" status --porcelain --untracked-files=all)"
test -z "$(git -C "${CANDIDATE}" status --porcelain --untracked-files=all)"
git -C "${CANDIDATE}" log -2 --format='%H %T %s'
