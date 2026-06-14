#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/foldflow"
VARIANT="${1:-sfm}"
mkdir -p "${OUTPUT_DIR}"

download() {
  local name="$1"
  local expected_sha256="$2"
  local url="https://github.com/DreamFold/FoldFlow/releases/download/0.1.0/${name}"
  local destination="${OUTPUT_DIR}/${name}"
  if [[ ! -f "${destination}" ]]; then
    curl -L --fail --retry 3 --output "${destination}.tmp" "${url}"
    mv "${destination}.tmp" "${destination}"
  fi
  if [[ -n "${expected_sha256}" ]]; then
    printf '%s  %s\n' "${expected_sha256}" "${destination}" | shasum -a 256 -c -
  else
    echo "Upstream publishes no checksum for ${name}; downloaded file was not hash-pinned."
  fi
}

case "${VARIANT}" in
  base)
    download "foldflow-base.pth" ""
    ;;
  ot)
    download "foldflow-ot.pth" ""
    ;;
  sfm)
    download "foldflow-sfm.pth" "d1dc0328c1b9d8b8f8900e3e41e2f8d43886eff5e9fab4a2a611a671708085f7"
    ;;
  all)
    download "foldflow-base.pth" ""
    download "foldflow-ot.pth" ""
    download "foldflow-sfm.pth" "d1dc0328c1b9d8b8f8900e3e41e2f8d43886eff5e9fab4a2a611a671708085f7"
    ;;
  *)
    echo "Usage: $0 [base|ot|sfm|all]" >&2
    exit 2
    ;;
esac
