"""Correctness and timing harness for the raw CUDA small-square QR extension."""

from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

task_module = types.ModuleType("task")
task_module.input_t = torch.Tensor
task_module.output_t = tuple[torch.Tensor, torch.Tensor]
sys.modules["task"] = task_module

import qr_official  # noqa: E402
import submission  # noqa: E402


CASES = ["dense", "upper", "diagonal", "rankdef", "clustered", "band", "rowscale"]


def time_fn(fn, data: torch.Tensor, warmup: int, trials: int) -> float:
    for _ in range(warmup):
        fn(data)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(trials):
        fn(data)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / trials


def run_case(case: str, batch: int, n: int, cond: int, seed: int, trials: int) -> bool:
    data = qr_official.generate_input(batch=batch, n=n, cond=cond, seed=seed, case=case)
    output = submission.custom_kernel(data)
    ok, msg = qr_official.check_implementation(data, output)

    custom_ms = time_fn(submission.custom_kernel, data, warmup=3, trials=trials)
    geqrf_ms = time_fn(torch.geqrf, data, warmup=3, trials=trials)
    ratio = custom_ms / geqrf_ms if geqrf_ms > 0 else float("inf")

    print(
        f"{case:12s} ok={ok} batch={batch:3d} n={n:2d} "
        f"custom_ms={custom_ms:.4f} geqrf_ms={geqrf_ms:.4f} "
        f"custom/geqrf={ratio:.3f}; {msg}",
        flush=True,
    )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--n", type=int, nargs="*", default=[16, 32, 64])
    parser.add_argument("--cond", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--case", choices=CASES, nargs="*", default=CASES)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not visible to PyTorch; skipping correctness/benchmark run.")
        raise SystemExit(2)

    all_ok = True
    for n in args.n:
      if n < 1 or n > 64:
          raise ValueError("small-square raw CUDA path supports 1 <= n <= 64")
      for case in args.case:
          all_ok = run_case(case, args.batch, n, args.cond, args.seed, args.trials) and all_ok

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
