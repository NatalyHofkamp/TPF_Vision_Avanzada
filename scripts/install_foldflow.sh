#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="https://github.com/DreamFold/FoldFlow.git"
REVISION="9d2c260813da3c9a2bc944973953f63ea6d71203"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="${FOLDFLOW_ROOT:-${PROJECT_ROOT}/third_party/foldflow_official}"
TEMP_REPOSITORY="$(mktemp -d)"

cleanup() {
  rm -rf "${TEMP_REPOSITORY}"
}
trap cleanup EXIT

git clone --filter=blob:none --no-checkout "${REPOSITORY}" "${TEMP_REPOSITORY}/repo"
git -C "${TEMP_REPOSITORY}/repo" fetch --depth 1 origin "${REVISION}"

rm -rf "${DESTINATION}"
mkdir -p "${DESTINATION}"
git -C "${TEMP_REPOSITORY}/repo" archive "${REVISION}" \
  foldflow openfold runner/config LICENSE | tar -x -C "${DESTINATION}"
printf '%s\n' "${REVISION}" > "${DESTINATION}/REVISION"

echo "Installed official FoldFlow ${REVISION} at ${DESTINATION}"
