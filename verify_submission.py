"""Local verification wrapper for submission.py against qr_official.py."""

import sys
import time
import types

import torch


task_module = types.ModuleType("task")
task_module.input_t = torch.Tensor
task_module.output_t = tuple[torch.Tensor, torch.Tensor]
sys.modules["task"] = task_module

import qr_official  # noqa: E402
import submission  # noqa: E402


CASES = [
    "dense",
    "upper",
    "diagonal",
    "rankdef",
    "nearrank",
    "clustered",
    "band",
    "nearcollinear",
    "rowscale",
]


def time_custom_kernel(data: torch.Tensor, warmup: int = 1, trials: int = 3) -> float:
    for _ in range(warmup):
        submission.custom_kernel(data)
    if data.is_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(trials):
        submission.custom_kernel(data)
    if data.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / trials


def main() -> None:
    all_ok = True
    for case in CASES:
        data = qr_official.generate_input(batch=2, n=16, cond=2, seed=123, case=case)
        elapsed_ms = time_custom_kernel(data)
        ok, msg = qr_official.check_implementation(data, submission.custom_kernel(data))
        all_ok = all_ok and ok
        print(f"{case:14s} {ok} e2e_ms={elapsed_ms:.3f}; {msg}", flush=True)

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
