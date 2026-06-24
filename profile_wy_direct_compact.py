"""Profile the side-only compact-tree WY -> standard (H,tau) converter path."""

from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
PROFILE_DIR = ROOT / "profile"
for path in (ROOT, PROFILE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

task_module = types.ModuleType("task")
task_module.input_t = torch.Tensor
task_module.output_t = tuple[torch.Tensor, torch.Tensor]
sys.modules.setdefault("task", task_module)

import qr_official  # noqa: E402
from prototype_cuda_tsqr_panel import load_ext as load_tsqr_ext  # noqa: E402
from prototype_tsqr_checker_bridge import tsqr_blocked_paper_wy_compact_thin_standard_output  # noqa: E402


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_once(fn):
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return (time.perf_counter() - t0) * 1000.0, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", default="128,256,512")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--case", default="dense")
    parser.add_argument("--cond", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--nb", type=int, default=16)
    parser.add_argument("--row-tile", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--avoid-qthin", action="store_true", help="use direct tree->WY path without materializing Q_thin")
    args = parser.parse_args()

    ext = load_tsqr_ext()
    for n in [int(x) for x in args.ns.split(",") if x.strip()]:
        data = qr_official.generate_input(args.batch, n, args.cond, args.seed, args.case)
        for _ in range(args.warmup):
            tsqr_blocked_paper_wy_compact_thin_standard_output(
                ext,
                data,
                args.nb,
                args.row_tile,
                avoid_qthin=args.avoid_qthin,
            )
        ms, (h, tau, metrics) = time_once(
            lambda: tsqr_blocked_paper_wy_compact_thin_standard_output(
                ext,
                data,
                args.nb,
                args.row_tile,
                avoid_qthin=args.avoid_qthin,
            ),
        )
        ok, msg = qr_official.check_implementation(data, (h, tau))
        print(
            f"n={n} wy_direct_compact={ms:.3f} ms "
            f"panel_q_err={metrics['max_panel_q_rel_err']:.3e} "
            f"T_lower={metrics['max_panel_t_lower_norm']:.3e} "
            f"T_upper={metrics['max_panel_t_upper_norm']:.3e} "
            f"ok={ok}; {msg}",
            flush=True,
        )


if __name__ == "__main__":
    main()
