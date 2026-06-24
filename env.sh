#!/usr/bin/env bash
# Local dev environment for running submission.py with the vendored CuTe DSL + uv venv torch.
# Usage: source ./env.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${HERE}/.cutedsl:${HERE}/.cutedsl/nvidia_cutlass_dsl/python_packages:${PYTHONPATH}"

# Activate uv venv that holds torch (cu130, matches cutlass-dsl cu13 build).
if [ -f "${HERE}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${HERE}/.venv/bin/activate"
fi

if [ "${CUDA_VISIBLE_DEVICES:-}" = "all" ]; then
  unset CUDA_VISIBLE_DEVICES
fi
