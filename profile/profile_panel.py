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
task_module.output_t = tuple
sys.modules["task"] = task_module

import qr_official
import submission as S


def time_cuda(fn, warmup: int = 3, trials: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(trials)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(trials)]
    for idx in range(trials):
        starts[idx].record()
        fn()
        ends[idx].record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / trials


def sigma_like_row_major(h: torch.Tensor, j_start: int, j_end: int) -> None:
    j = j_start
    while j < j_end:
        _ = torch.sum(h[:, j + 1 :, j] * h[:, j + 1 :, j], dim=1)
        j += 1


def sigma_like_transposed(ht: torch.Tensor, j_start: int, j_end: int) -> None:
    j = j_start
    while j < j_end:
        _ = torch.sum(ht[:, j, j + 1 :] * ht[:, j, j + 1 :], dim=1)
        j += 1


def scale_like_row_major(h: torch.Tensor, scales: torch.Tensor, j_start: int, j_end: int) -> None:
    j = j_start
    while j < j_end:
        h[:, j + 1 :, j] *= scales[:, j - j_start].unsqueeze(-1)
        j += 1


def scale_like_transposed(ht: torch.Tensor, scales: torch.Tensor, j_start: int, j_end: int) -> None:
    j = j_start
    while j < j_end:
        ht[:, j, j + 1 :] *= scales[:, j - j_start].unsqueeze(-1)
        j += 1


def apply_like_row_major(h: torch.Tensor, tau: torch.Tensor, j_start: int, j_end: int) -> None:
    j = j_start
    while j < j_end:
        v = h[:, j:, j]
        if v.shape[1] > 0:
            v[:, 0] = 1.0
        c = h[:, j:, j + 1 : j_end]
        if c.shape[2] > 0:
            dots = torch.bmm(v.unsqueeze(1), c).squeeze(1)
            w = tau[:, j - j_start : j - j_start + 1] * dots
            c -= v.unsqueeze(-1) * w.unsqueeze(1)
        j += 1


def apply_like_transposed(ht: torch.Tensor, tau: torch.Tensor, j_start: int, j_end: int) -> None:
    j = j_start
    while j < j_end:
        v = ht[:, j, j:].clone()
        if v.shape[1] > 0:
            v[:, 0] = 1.0
        c = ht[:, j + 1 : j_end, j:]
        if c.shape[1] > 0:
            dots = torch.sum(c * v.unsqueeze(1), dim=2)
            w = tau[:, j - j_start : j - j_start + 1] * dots
            c -= w.unsqueeze(-1) * v.unsqueeze(1)
        j += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--nb", type=int, default=64)
    parser.add_argument("--cond", type=int, default=1)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--case", default="dense")
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    data = qr_official.generate_input(
        batch=args.batch,
        n=args.n,
        cond=args.cond,
        seed=args.seed,
        case=args.case,
    )
    j_start = 0
    j_end = min(args.nb, args.n)

    tau_full = torch.zeros(args.batch, args.n, device=data.device, dtype=data.dtype)

    panel_ms = time_cuda(
        lambda: S._panel_factor_apply_cutedsl_mvp(
            data.clone(),
            torch.zeros_like(tau_full),
            j_start,
            j_end,
        ),
        trials=args.trials,
    )

    sigma_rm = time_cuda(
        lambda: sigma_like_row_major(data.clone(), j_start, j_end),
        trials=args.trials,
    )
    sigma_t = time_cuda(
        lambda: sigma_like_transposed(data.transpose(-1, -2).contiguous(), j_start, j_end),
        trials=args.trials,
    )

    scales = torch.full((args.batch, j_end - j_start), 0.99, device=data.device, dtype=data.dtype)
    scale_rm = time_cuda(
        lambda: scale_like_row_major(data.clone(), scales, j_start, j_end),
        trials=args.trials,
    )
    scale_t = time_cuda(
        lambda: scale_like_transposed(data.transpose(-1, -2).contiguous(), scales, j_start, j_end),
        trials=args.trials,
    )

    tau_panel = torch.full((args.batch, j_end - j_start), 0.5, device=data.device, dtype=data.dtype)
    apply_rm = time_cuda(
        lambda: apply_like_row_major(data.clone(), tau_panel, j_start, j_end),
        trials=args.trials,
    )
    apply_t = time_cuda(
        lambda: apply_like_transposed(data.transpose(-1, -2).contiguous(), tau_panel, j_start, j_end),
        trials=args.trials,
    )

    print(
        {
            "shape": {"batch": args.batch, "n": args.n, "nb": j_end - j_start, "case": args.case},
            "panel_kernel_ms": panel_ms,
            "sigma_row_major_ms": sigma_rm,
            "sigma_transposed_ms": sigma_t,
            "scale_row_major_ms": scale_rm,
            "scale_transposed_ms": scale_t,
            "apply_row_major_ms": apply_rm,
            "apply_transposed_ms": apply_t,
            "sigma_row_over_transposed": sigma_rm / sigma_t if sigma_t else None,
            "scale_row_over_transposed": scale_rm / scale_t if scale_t else None,
            "apply_row_over_transposed": apply_rm / apply_t if apply_t else None,
        }
    )


if __name__ == "__main__":
    main()
