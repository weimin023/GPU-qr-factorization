"""NCU target for the side-only direct TSQR/WY compact path."""

from __future__ import annotations

import argparse
import importlib.util
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--case", default="dense")
    parser.add_argument("--cond", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--nb", type=int, default=16)
    parser.add_argument("--row-tile", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--avoid-qthin", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--prebuilt-ext", default="", help="load prebuilt TSQR CUDA extension .so instead of JIT")
    args = parser.parse_args()

    if args.prebuilt_ext:
        spec = importlib.util.spec_from_file_location("qr_cuda_tsqr_panel_proof_ext_v13", args.prebuilt_ext)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load prebuilt extension: {args.prebuilt_ext}")
        ext = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ext)
    else:
        ext = load_tsqr_ext()
    data = qr_official.generate_input(args.batch, args.n, args.cond, args.seed, args.case)
    for _ in range(args.warmup):
        tsqr_blocked_paper_wy_compact_thin_standard_output(
            ext,
            data,
            args.nb,
            args.row_tile,
            avoid_qthin=args.avoid_qthin,
        )
    sync()

    torch.cuda.cudart().cudaProfilerStart()
    t0 = time.perf_counter()
    h, tau, _metrics = tsqr_blocked_paper_wy_compact_thin_standard_output(
        ext,
        data,
        args.nb,
        args.row_tile,
        avoid_qthin=args.avoid_qthin,
    )
    sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    torch.cuda.cudart().cudaProfilerStop()

    if args.check:
        ok, msg = qr_official.check_implementation(data, (h, tau))
        print(f"n={args.n} elapsed_ms={elapsed_ms:.3f} ok={ok}; {msg}", flush=True)
    else:
        print(f"n={args.n} elapsed_ms={elapsed_ms:.3f}", flush=True)


if __name__ == "__main__":
    main()
