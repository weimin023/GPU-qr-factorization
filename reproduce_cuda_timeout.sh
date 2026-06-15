#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONPATH="/workspace/.cutedsl:/workspace/.cutedsl/nvidia_cutlass_dsl/python_packages:${PYTHONPATH:-}"
if [ "${CUDA_VISIBLE_DEVICES:-}" = "all" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("CUDA is not available in this local environment.")
    print("Cannot reproduce the B200 custom-kernel timeout locally here.")
    sys.exit(2)

print("CUDA available:", torch.cuda.get_device_name(0))
PY

python verify_submission.py \
  --module submission_cuda_experiment \
  --remote-cases \
  --trials 1 \
  "$@"
