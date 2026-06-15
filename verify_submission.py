"""Local verification wrapper for submission.py against qr_official.py."""

import argparse
import importlib
import sys
import time
import types

import torch


task_module = types.ModuleType("task")
task_module.input_t = torch.Tensor
task_module.output_t = tuple[torch.Tensor, torch.Tensor]
sys.modules["task"] = task_module

import qr_official  # noqa: E402
submission = None


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

REMOTE_CASES = [
    {"n": 32, "cond": 1, "seed": 53124, "batch": 20, "case": "dense"},
    {"n": 176, "cond": 1, "seed": 3321, "batch": 40, "case": "dense"},
    {"n": 352, "cond": 1, "seed": 1200, "batch": 40, "case": "dense"},
    {"n": 512, "cond": 2, "seed": 32523, "batch": 16, "case": "dense"},
    {"n": 1024, "cond": 2, "seed": 4327, "batch": 4, "case": "dense"},
    {"n": 4096, "cond": 1, "seed": 75342, "batch": 1, "case": "dense"},
    {"n": 512, "cond": 4, "seed": 32524, "batch": 16, "case": "dense"},
    {"n": 512, "cond": 0, "seed": 32525, "batch": 16, "case": "rankdef"},
    {"n": 512, "cond": 0, "seed": 32526, "batch": 16, "case": "clustered"},
    {"n": 512, "cond": 0, "seed": 32527, "batch": 16, "case": "band"},
    {"n": 512, "cond": 0, "seed": 32528, "batch": 16, "case": "rowscale"},
    {"n": 512, "cond": 0, "seed": 32529, "batch": 16, "case": "nearcollinear"},
    {"n": 1024, "cond": 4, "seed": 4328, "batch": 4, "case": "dense"},
    {"n": 1024, "cond": 0, "seed": 4329, "batch": 4, "case": "rankdef"},
    {"n": 1024, "cond": 0, "seed": 4330, "batch": 4, "case": "nearrank"},
    {"n": 1024, "cond": 0, "seed": 4331, "batch": 4, "case": "clustered"},
    {"n": 2048, "cond": 2, "seed": 224466, "batch": 2, "case": "dense"},
    {"n": 2048, "cond": 0, "seed": 224467, "batch": 2, "case": "rankdef"},
    {"n": 4096, "cond": 0, "seed": 75343, "batch": 1, "case": "upper"},
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


def run_case(label: str, *, batch: int, n: int, cond: int, seed: int, case: str, trials: int) -> bool:
    data = qr_official.generate_input(batch=batch, n=n, cond=cond, seed=seed, case=case)
    elapsed_ms = time_custom_kernel(data, warmup=1, trials=trials)
    ok, msg = qr_official.check_implementation(data, submission.custom_kernel(data))
    print(
        f"{label:26s} {ok} e2e_ms={elapsed_ms:.3f}; "
        f"case={case}; batch={batch}; n={n}; cond={cond}; seed={seed}; {msg}",
        flush=True,
    )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module",
        default="submission",
        help="Python module containing custom_kernel, default: submission",
    )
    parser.add_argument(
        "--remote-cases",
        action="store_true",
        help="run the qr_v2 cases copied from the remote failure log",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="timing trials per case",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=0,
        help="when set, skip cases with n larger than this value",
    )
    args = parser.parse_args()

    global submission
    submission = importlib.import_module(args.module)

    all_ok = True
    if args.remote_cases:
        print(
            "note: local qr_official.py has no `mixed` generator, so remote mixed cases are not included.",
            flush=True,
        )
        for spec in REMOTE_CASES:
            if args.max_n and spec["n"] > args.max_n:
                continue
            label = f"remote/{spec['case']}/{spec['n']}"
            all_ok = run_case(label, trials=args.trials, **spec) and all_ok
    else:
        for case in CASES:
            all_ok = run_case(
                case,
                batch=2,
                n=16,
                cond=2,
                seed=123,
                case=case,
                trials=args.trials,
            ) and all_ok

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
