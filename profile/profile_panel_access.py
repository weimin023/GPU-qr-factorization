import argparse
import sys
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

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.cute.runtime import from_dlpack
except ModuleNotFoundError as exc:
    raise SystemExit(f"cutlass is required for this profiler: {exc}")


@cute.kernel
def _sigma_scan_rowmajor_kernel(
    h: cute.Tensor,
    out: cute.Tensor,
    batch_count: Int32,
    m: Int32,
    j: Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    if bidx < batch_count:
        local = h[bidx, j, j] * 0.0
        row = j + 1 + tidx
        while row < m:
            x = h[bidx, row, j]
            local += x * x
            row += 128
        out[bidx, tidx] = local


@cute.kernel
def _sigma_scan_transposed_kernel(
    ht: cute.Tensor,
    out: cute.Tensor,
    batch_count: Int32,
    m: Int32,
    j: Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    if bidx < batch_count:
        local = ht[bidx, j, j] * 0.0
        row = j + 1 + tidx
        while row < m:
            x = ht[bidx, j, row]
            local += x * x
            row += 128
        out[bidx, tidx] = local


@cute.kernel
def _apply_dot_rowmajor_kernel(
    h: cute.Tensor,
    out: cute.Tensor,
    batch_count: Int32,
    m: Int32,
    j: Int32,
    j_end: Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    if bidx < batch_count:
        target = j + 1 + tidx
        if target < j_end:
            dot = h[bidx, j, target]
            row = j + 1
            while row < m:
                dot += h[bidx, row, j] * h[bidx, row, target]
                row += 1
            out[bidx, tidx] = dot


@cute.kernel
def _apply_dot_transposed_kernel(
    ht: cute.Tensor,
    out: cute.Tensor,
    batch_count: Int32,
    m: Int32,
    j: Int32,
    j_end: Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    if bidx < batch_count:
        target = j + 1 + tidx
        if target < j_end:
            dot = ht[bidx, target, j]
            row = j + 1
            while row < m:
                dot += ht[bidx, j, row] * ht[bidx, target, row]
                row += 1
            out[bidx, tidx] = dot


@cute.kernel
def _apply_update_rowmajor_kernel(
    h: cute.Tensor,
    wvals: cute.Tensor,
    out: cute.Tensor,
    batch_count: Int32,
    m: Int32,
    j: Int32,
    j_end: Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    if bidx < batch_count:
        target = j + 1 + tidx
        if target < j_end:
            w = wvals[bidx, tidx]
            row = j + 1
            while row < m:
                out[bidx, row, target] = h[bidx, row, target] - h[bidx, row, j] * w
                row += 1


@cute.kernel
def _apply_update_transposed_kernel(
    ht: cute.Tensor,
    wvals: cute.Tensor,
    outt: cute.Tensor,
    batch_count: Int32,
    m: Int32,
    j: Int32,
    j_end: Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    if bidx < batch_count:
        target = j + 1 + tidx
        if target < j_end:
            w = wvals[bidx, tidx]
            row = j + 1
            while row < m:
                outt[bidx, target, row] = ht[bidx, target, row] - ht[bidx, j, row] * w
                row += 1


@cute.jit
def sigma_scan_rowmajor_cuda(h: cute.Tensor, out: cute.Tensor, batch_count: Int32, m: Int32, j: Int32):
    _sigma_scan_rowmajor_kernel(h, out, batch_count, m, j).launch(grid=(batch_count, 1, 1), block=(128, 1, 1))


@cute.jit
def sigma_scan_transposed_cuda(ht: cute.Tensor, out: cute.Tensor, batch_count: Int32, m: Int32, j: Int32):
    _sigma_scan_transposed_kernel(ht, out, batch_count, m, j).launch(grid=(batch_count, 1, 1), block=(128, 1, 1))


@cute.jit
def apply_dot_rowmajor_cuda(
    h: cute.Tensor, out: cute.Tensor, batch_count: Int32, m: Int32, j: Int32, j_end: Int32
):
    _apply_dot_rowmajor_kernel(h, out, batch_count, m, j, j_end).launch(grid=(batch_count, 1, 1), block=(128, 1, 1))


@cute.jit
def apply_dot_transposed_cuda(
    ht: cute.Tensor, out: cute.Tensor, batch_count: Int32, m: Int32, j: Int32, j_end: Int32
):
    _apply_dot_transposed_kernel(ht, out, batch_count, m, j, j_end).launch(grid=(batch_count, 1, 1), block=(128, 1, 1))


@cute.jit
def apply_update_rowmajor_cuda(
    h: cute.Tensor,
    wvals: cute.Tensor,
    out: cute.Tensor,
    batch_count: Int32,
    m: Int32,
    j: Int32,
    j_end: Int32,
):
    _apply_update_rowmajor_kernel(h, wvals, out, batch_count, m, j, j_end).launch(
        grid=(batch_count, 1, 1), block=(128, 1, 1)
    )


@cute.jit
def apply_update_transposed_cuda(
    ht: cute.Tensor,
    wvals: cute.Tensor,
    outt: cute.Tensor,
    batch_count: Int32,
    m: Int32,
    j: Int32,
    j_end: Int32,
):
    _apply_update_transposed_kernel(ht, wvals, outt, batch_count, m, j, j_end).launch(
        grid=(batch_count, 1, 1), block=(128, 1, 1)
    )


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
    data_t = data.transpose(-1, -2).contiguous()
    batch = data.shape[0]
    m = data.shape[1]
    j = 0
    j_end = min(args.nb, args.n)

    partial = torch.empty(batch, 128, device=data.device, dtype=data.dtype)
    wvals = torch.full((batch, 128), 0.5, device=data.device, dtype=data.dtype)
    out_rm = data.clone()
    out_t = data_t.clone()

    sigma_rm = time_cuda(
        lambda: sigma_scan_rowmajor_cuda(from_dlpack(data), from_dlpack(partial), batch, m, j),
        trials=args.trials,
    )
    sigma_t = time_cuda(
        lambda: sigma_scan_transposed_cuda(from_dlpack(data_t), from_dlpack(partial), batch, m, j),
        trials=args.trials,
    )
    dot_rm = time_cuda(
        lambda: apply_dot_rowmajor_cuda(from_dlpack(data), from_dlpack(partial), batch, m, j, j_end),
        trials=args.trials,
    )
    dot_t = time_cuda(
        lambda: apply_dot_transposed_cuda(from_dlpack(data_t), from_dlpack(partial), batch, m, j, j_end),
        trials=args.trials,
    )
    update_rm = time_cuda(
        lambda: apply_update_rowmajor_cuda(
            from_dlpack(data),
            from_dlpack(wvals),
            from_dlpack(out_rm),
            batch,
            m,
            j,
            j_end,
        ),
        trials=args.trials,
    )
    update_t = time_cuda(
        lambda: apply_update_transposed_cuda(
            from_dlpack(data_t),
            from_dlpack(wvals),
            from_dlpack(out_t),
            batch,
            m,
            j,
            j_end,
        ),
        trials=args.trials,
    )

    print(
        {
            "shape": {"batch": args.batch, "n": args.n, "nb": j_end, "case": args.case},
            "sigma_rowmajor_ms": sigma_rm,
            "sigma_transposed_ms": sigma_t,
            "dot_rowmajor_ms": dot_rm,
            "dot_transposed_ms": dot_t,
            "update_rowmajor_ms": update_rm,
            "update_transposed_ms": update_t,
            "sigma_row_over_transposed": sigma_rm / sigma_t if sigma_t else None,
            "dot_row_over_transposed": dot_rm / dot_t if dot_t else None,
            "update_row_over_transposed": update_rm / update_t if update_t else None,
        }
    )


if __name__ == "__main__":
    main()
