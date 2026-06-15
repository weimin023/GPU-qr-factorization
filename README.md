# GPU-qr-factorization

2026 Linear Algebra Kernels For The Age Of Research.

This repository contains a single-file QR factorization submission:

- `submission.py`: final file to submit.
- `qr_official.py`: official local reference/checker.
- `verify_submission.py`: local wrapper for running the official test cases.
- `cutedsl_env.sh`: local environment setup for CuTe DSL and CUDA visibility.
- `docs/`: notes, announcement summary, implementation plan, profiling notes.

## Problem Summary

The competition target is batched real square QR factorization:

```text
A shape: (batch, n, n)
output:  (H, tau)
```

The output follows `torch.geqrf` compact Householder format. The checker rebuilds `Q` from `(H, tau)`, extracts `R = triu(H)`, and validates factorization, orthogonality, reconstruction, and triangularity.

## Environment Setup

From the repository root:

```bash
cd /workspace/GPU-qr-factorization
source ./cutedsl_env.sh
```

The environment script sets `PYTHONPATH` for the workspace-local CuTe DSL install and fixes CUDA visibility when `CUDA_VISIBLE_DEVICES=all`.

Quick CUDA/CuTe check:

```bash
python - <<'PY'
import torch
import cutlass
import cutlass.cute as cute

print("torch cuda:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
print("cute import ok")
PY
```

## Local Verification

Run the official checker wrapper:

```bash
cd /workspace/GPU-qr-factorization
source ./cutedsl_env.sh
python verify_submission.py
```

The wrapper creates a local fake `task` module, imports `qr_official.py`, and checks `submission.custom_kernel` on the official-style cases:

- `dense`
- `upper`
- `diagonal`
- `rankdef`
- `nearrank`
- `clustered`
- `band`
- `nearcollinear`
- `rowscale`

Each line reports correctness, end-to-end elapsed time, and scaled residuals:

```text
dense          True e2e_ms=...; factor_rtol=...; scaled_factor_residual=...
```

Syntax-only check:

```bash
python -m py_compile submission.py qr_official.py verify_submission.py
```

## Running The Submission Manually

Minimal local smoke test:

```bash
cd /workspace/GPU-qr-factorization
source ./cutedsl_env.sh

python - <<'PY'
import sys, types, torch

task = types.ModuleType("task")
task.input_t = torch.Tensor
task.output_t = tuple[torch.Tensor, torch.Tensor]
sys.modules["task"] = task

import qr_official
import submission

data = qr_official.generate_input(batch=2, n=16, cond=2, seed=123, case="dense")
ok, msg = qr_official.check_implementation(data, submission.custom_kernel(data))
print(ok, msg)
PY
```

## Popcorn CLI Setup

If `popcorn-cli` was installed into `/root/.local/bin` but the command is not found, refresh the shell or update `PATH`:

```bash
source /root/.bashrc
# or:
export PATH="/root/.local/bin:$PATH"
```

Check the command:

```bash
which popcorn-cli
popcorn-cli --help
```

Register with GitHub:

```bash
popcorn-cli register github
```

## Submission Command

Submit only the final single file:

```bash
cd /workspace/GPU-qr-factorization
source ./cutedsl_env.sh
popcorn-cli submit --gpu <GPU_NAME> --leaderboard <LEADERBOARD_NAME> --mode test submission.py
```

Replace `<GPU_NAME>` and `<LEADERBOARD_NAME>` with the values for the QR leaderboard.

Examples of modes supported by the CLI:

```bash
popcorn-cli submit --gpu <GPU_NAME> --leaderboard <LEADERBOARD_NAME> --mode test submission.py
popcorn-cli submit --gpu <GPU_NAME> --leaderboard <LEADERBOARD_NAME> --mode benchmark submission.py
popcorn-cli submit --gpu <GPU_NAME> --leaderboard <LEADERBOARD_NAME> --mode profile submission.py
```

## Notes

- `submission.py` is self-contained and does not import local helper files.
- The CUDA path uses an in-file CuTe DSL panel kernel plus a raw CUDA tiled WY trailing update compiled with `torch.utils.cpp_extension.load_inline`.
- The first run may include CUDA extension compilation overhead. Subsequent runs reuse the compiled extension cache.
