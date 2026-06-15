# GPU-qr-factorization

2026 Linear Algebra Kernels For The Age Of Research.

This repository contains a single-file QR factorization submission:

- `submission.py`: final file to submit. The current checked-in version is the verified `qr_v2`/B200 baseline.
- `qr_official.py`: official local reference/checker.
- `verify_submission.py`: local wrapper for running the official test cases.
- `cutedsl_env.sh`: local environment setup for CUDA visibility.
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

The current leaderboard file, `submission.py`, uses the official `torch.geqrf` baseline and includes Popcorn directives for `qr_v2` on B200.

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
popcorn-cli submit --gpu B200 --leaderboard qr_v2 --mode leaderboard --no-tui submission.py
```

The `qr` and `qr_v2` leaderboards currently list B200 as the available GPU. `test` mode may be disabled for these leaderboards, so use `leaderboard` mode when `test` returns a `0/0 test submissions per hour` quota error.

Examples of modes supported by the CLI:

```bash
popcorn-cli submit --gpu B200 --leaderboard qr_v2 --mode leaderboard --no-tui submission.py
popcorn-cli submit --gpu B200 --leaderboard qr_v2 --mode profile --no-tui submission.py
```

## Notes

- `submission.py` is self-contained and does not import local helper files.
- The active submission path is the official `torch.geqrf` baseline. It passed the remote `qr_v2` leaderboard run on B200.
- A hand-written PyTorch Householder version passed local verification but timed out remotely at 300 seconds on `qr_v2`.
