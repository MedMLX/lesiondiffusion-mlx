#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  --help)
    echo "Usage: environment/setup.sh [--host-check]"
    exit 0
    ;;
  ""|--host-check) ;;
  *) echo "Unsupported setup argument: $1" >&2; exit 2 ;;
esac

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "LesionControl requires macOS arm64 and MLX Metal." >&2
  exit 1
fi
if [[ "${1:-}" == --host-check ]]; then
  echo "macos-arm64"
  exit 0
fi

project_root="$(cd "$(dirname "$0")/.." && pwd)"
for source_path in ../.. ../../../mlxradiomics ../../../medmlx/packages/core; do
  if [[ ! -f "$project_root/$source_path/pyproject.toml" ]]; then
    echo "Required workspace source is missing: $source_path" >&2
    exit 1
  fi
done
cd "$project_root"
uv sync --locked --python 3.13
