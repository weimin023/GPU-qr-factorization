"""Checker-compatible bridge for TSQR / GEQRT-like panel experiments.

This is a side prototype, not a submission path.  It answers two questions before
we invest in a real tiled-reflector implementation:

1. Can a TSQR panel representation drive the blocked trailing update correctly?
2. Can the resulting explicit-Q QR be converted into checker-compatible `(H, tau)`?

The implementation intentionally materializes explicit Q panels and converts the
final explicit Q through GEQRF.  That is too slow for leaderboard use, but gives a
clean correctness target for later tile-reflector / representation-converter work.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl


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
from prototype_cuda_tsqr_panel import top_compact_apply_q_triton  # noqa: E402
from prototype_cuda_tsqr_panel import top_compact_apply_qt_triton  # noqa: E402


@dataclass(frozen=True)
class Timing:
    median_ms: float
    min_ms: float
    max_ms: float


@dataclass
class TSQRPanelTree:
    local_qs: list[list[torch.Tensor]]
    sizes: list[list[int]]
    coord_positions: list[torch.Tensor]
    q_tops: torch.Tensor
    r_panel: torch.Tensor


@dataclass
class TSQRCompactTree:
    h_tiles: torch.Tensor
    tau_tiles: torch.Tensor
    h_tops: torch.Tensor
    tau_tops: torch.Tensor
    coord_positions: list[torch.Tensor]
    r_panel: torch.Tensor
    sizes: list[list[int]]


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_fn(fn, warmup: int, trials: int):
    out = None
    for _ in range(warmup):
        out = fn()
    sync()
    times: list[float] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        out = fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return Timing(statistics.median(times), min(times), max(times)), out


def residuals(a: torch.Tensor, q: torch.Tensor, r: torch.Tensor) -> tuple[float, float, float]:
    a64 = a.double()
    q64 = q.double()
    r64 = r.double()
    recon = torch.linalg.matrix_norm(torch.bmm(q64, r64) - a64, ord=1, dim=(-2, -1)).amax()
    scale = torch.linalg.matrix_norm(a64, ord=1, dim=(-2, -1)).amax().clamp_min(1e-30)
    eye = torch.eye(q.shape[-1], device=a.device, dtype=torch.float64).expand(a.shape[0], q.shape[-1], q.shape[-1])
    orth = torch.linalg.matrix_norm(torch.bmm(q64.transpose(1, 2), q64) - eye, ord=1, dim=(-2, -1)).amax()
    lower = torch.linalg.matrix_norm(torch.tril(r64, diagonal=-1), ord=1, dim=(-2, -1)).amax()
    epsn = torch.finfo(torch.float32).eps * max(a.shape[-1], 1)
    return (recon / (epsn * scale)).item(), (orth / epsn).item(), (lower / (epsn * scale)).item()


def explicit_qr_to_compact(a: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert explicit Q/R into the official checker's compact Householder layout."""

    hq, tau = torch.geqrf(q.contiguous())
    qh = torch.linalg.householder_product(hq, tau)
    r = torch.bmm(qh.transpose(1, 2), a)
    h = torch.tril(hq, diagonal=-1) + torch.triu(r)
    return h.contiguous(), tau.contiguous()


@triton.jit
def _wy_tau_diag16_triton_kernel(
    w_ptr,
    y_ptr,
    tau_ptr,
    rows: tl.constexpr,
    k: tl.constexpr,
):
    batch = tl.program_id(0)
    offs = tl.arange(0, 16)
    mask = offs < k
    base = batch * rows * k
    y_top = tl.load(
        y_ptr + base + offs[:, None] * k + offs[None, :],
        mask=mask[:, None] & mask[None, :],
        other=0.0,
    )
    w_top = tl.load(
        w_ptr + base + offs[:, None] * k + offs[None, :],
        mask=mask[:, None] & mask[None, :],
        other=0.0,
    )
    x = tl.zeros((16, 16), tl.float32)

    # Unit-lower forward solve: y_top * x = w_top.  Only diag(x) is stored,
    # but off-diagonal x values are needed by later diagonal columns.
    for r in tl.static_range(0, 16):
        if r < k:
            row_mask = offs == r
            y_row = tl.sum(tl.where(row_mask[:, None], y_top, 0.0), axis=0)
            w_row = tl.sum(tl.where(row_mask[:, None], w_top, 0.0), axis=0)
            v = tl.sum(tl.where(offs[:, None] < r, y_row[:, None] * x, 0.0), axis=0)
            row_x = w_row - v
            x = tl.where(offs[:, None] == r, row_x[None, :], x)

    diag = tl.sum(tl.where(offs[:, None] == offs[None, :], x, 0.0), axis=1)
    tl.store(tau_ptr + batch * k + offs, diag, mask=mask)


def wy_tau_diag16_triton(w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Extract tau=diag(T) for fixed small panels from W=Y*T.T."""

    batch, rows, k = y.shape
    if k > 16:
        raise ValueError("wy_tau_diag16_triton supports k <= 16")
    tau = torch.empty((batch, k), device=y.device, dtype=y.dtype)
    _wy_tau_diag16_triton_kernel[(batch,)](
        w.contiguous(),
        y.contiguous(),
        tau,
        rows=rows,
        k=k,
    )
    return tau


@triton.jit
def _scatter_top_coord_to_basis_kernel(
    coord_ptr,
    basis_ptr,
    rows: tl.constexpr,
    jb: tl.constexpr,
    row_tile: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    row_block = tl.program_id(1)
    rows_off = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    tile = rows_off // row_tile
    local = rows_off - tile * row_tile
    coord_row = tile * jb + local
    vals = tl.load(
        coord_ptr + batch * ((rows + row_tile - 1) // row_tile * jb) * jb + coord_row[:, None] * jb + cols[None, :],
        mask=(rows_off[:, None] < rows) & (local[:, None] < jb) & (cols[None, :] < jb),
        other=0.0,
    )
    tl.store(
        basis_ptr + batch * rows * jb + rows_off[:, None] * jb + cols[None, :],
        vals,
        mask=(rows_off[:, None] < rows) & (cols[None, :] < jb),
    )


def scatter_top_coord_to_basis_triton(coord: torch.Tensor, rows: int, row_tile: int) -> torch.Tensor:
    batch, _stack_rows, jb = coord.shape
    basis = torch.empty((batch, rows, jb), device=coord.device, dtype=coord.dtype)
    grid = (batch, triton.cdiv(rows, 32))
    _scatter_top_coord_to_basis_kernel[grid](
        coord.contiguous(),
        basis,
        rows=rows,
        jb=jb,
        row_tile=row_tile,
        BLOCK_M=32,
        BLOCK_N=16,
    )
    return basis


def wy_to_standard_compact_diag_trial(
    a: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    signs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Try a cheap WY -> sequential compact-Householder conversion.

    This is intentionally a diagnostic converter, not a claimed algorithm.  If a
    paper-style WY representation already matches the standard compact WY
    convention, then `Y` can serve as the reflector storage `V`, `W = V*T.T`,
    and `diag(T)` should be the per-reflector `tau`.  A large Q mismatch here
    falsifies that optimistic shortcut and tells us a real WY-to-sequential
    factorization is required.
    """

    batch, m, n = a.shape
    k = y.shape[-1]
    if k != n:
        raise ValueError("trial converter currently expects a full n-column WY basis")
    if torch.max(torch.abs(signs - 1)).item() > 1e-5:
        raise ValueError("trial converter only supports identity sign convention")

    y_top = y[:, :k, :]
    w_top = w[:, :k, :]
    # y_top is unit lower triangular in the paper reconstruction.  Solve
    # y_top * T.T = w_top for the candidate compact-WY T.T.
    t_trans = torch.linalg.solve_triangular(
        y_top,
        w_top,
        upper=False,
        unitriangular=True,
    )
    tau = torch.diagonal(t_trans, dim1=1, dim2=2).contiguous()

    q_wy = materialize_wy_q(w, y, signs)
    q_trial = torch.linalg.householder_product(y.contiguous(), tau)
    q_err = torch.linalg.matrix_norm((q_trial - q_wy).double(), ord=1, dim=(-2, -1)).amax()
    q_scale = torch.linalg.matrix_norm(q_wy.double(), ord=1, dim=(-2, -1)).amax().clamp_min(1e-30)

    r = torch.bmm(q_trial.transpose(1, 2), a)
    h = torch.tril(y, diagonal=-1) + torch.triu(r)
    metrics = {
        "q_rel_err": (q_err / q_scale).item(),
        "t_lower_norm": torch.linalg.matrix_norm(torch.tril(t_trans, diagonal=-1).double(), ord=1, dim=(-2, -1)).amax().item(),
        "t_upper_norm": torch.linalg.matrix_norm(torch.triu(t_trans, diagonal=1).double(), ord=1, dim=(-2, -1)).amax().item(),
    }
    return h.contiguous(), tau.contiguous(), metrics


def wy_to_standard_panel_tau_diag_trial(
    w: torch.Tensor,
    y: torch.Tensor,
    signs: torch.Tensor,
    validate: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Extract panel `tau` from paper-style WY via the optimistic diag(T) rule."""

    k = y.shape[-1]
    if validate and torch.max(torch.abs(signs - 1)).item() > 1e-5:
        raise ValueError("panel trial converter only supports identity sign convention")
    y_top = y[:, :k, :]
    w_top = w[:, :k, :]
    if (not validate) and k <= 16 and y.is_cuda and y.dtype == torch.float32:
        tau = wy_tau_diag16_triton(w, y)
        t_trans = None
    else:
        t_trans = torch.linalg.solve_triangular(
            y_top,
            w_top,
            upper=False,
            unitriangular=True,
        )
        tau = torch.diagonal(t_trans, dim1=1, dim2=2).contiguous()
    if validate:
        q_wy = materialize_wy_q(w, y, signs)
        q_trial = torch.linalg.householder_product(y.contiguous(), tau)
        q_wy_thin = q_wy[:, :, :k]
        q_err = torch.linalg.matrix_norm((q_trial - q_wy_thin).double(), ord=1, dim=(-2, -1)).amax()
        q_scale = torch.linalg.matrix_norm(q_wy_thin.double(), ord=1, dim=(-2, -1)).amax().clamp_min(1e-30)
        panel_q_rel_err = (q_err / q_scale).item()
    else:
        panel_q_rel_err = float("nan")
    if validate:
        if t_trans is None:
            t_trans = torch.linalg.solve_triangular(
                y_top,
                w_top,
                upper=False,
                unitriangular=True,
            )
        panel_t_lower_norm = torch.linalg.matrix_norm(
            torch.tril(t_trans, diagonal=-1).double(),
            ord=1,
            dim=(-2, -1),
        ).amax().item()
        panel_t_upper_norm = torch.linalg.matrix_norm(
            torch.triu(t_trans, diagonal=1).double(),
            ord=1,
            dim=(-2, -1),
        ).amax().item()
    else:
        panel_t_lower_norm = float("nan")
        panel_t_upper_norm = float("nan")
    metrics = {
        "panel_q_rel_err": panel_q_rel_err,
        "panel_t_lower_norm": panel_t_lower_norm,
        "panel_t_upper_norm": panel_t_upper_norm,
    }
    return tau, metrics


def nonpivoting_lu_square(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Small batched no-pivot LU used by the paper-style WY reconstruction."""

    batch, n, n2 = a.shape
    if n != n2:
        raise ValueError("nonpivoting_lu_square expects square matrices")
    u = a.clone()
    l = torch.eye(n, device=a.device, dtype=a.dtype).expand(batch, n, n).clone()
    for j in range(n - 1):
        pivot = u[:, j, j]
        if torch.any(pivot.abs() < 1e-12):
            raise RuntimeError("non-pivot LU hit a near-zero pivot; try a different sign mode")
        factors = u[:, j + 1 :, j] / pivot[:, None]
        l[:, j + 1 :, j] = factors
        u[:, j + 1 :, j:] -= factors[:, :, None] * u[:, None, j, j:]
        u[:, j + 1 :, j] = 0.0
    if torch.any(u[:, -1, -1].abs() < 1e-12):
        raise RuntimeError("non-pivot LU hit a near-zero final pivot; try a different sign mode")
    return l, u


def reconstruct_wy_from_explicit_q(
    q: torch.Tensor,
    k: int,
    sign_mode: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct `Q ~= S - W Y.T` from explicit Q using the paper's LU route.

    The paper states the practical variant factors `Q - S`, where `S` is a
    diagonal sign matrix, to avoid rank-deficient non-pivot LU.  We keep a tiny
    `auto` search here because TSQR sign conventions differ from torch.geqrf and
    from our CUDA local Householder kernel.
    """

    batch, rows, cols = q.shape
    if cols < k:
        raise ValueError("explicit Q must have at least k columns")
    eye_thin = torch.eye(rows, k, device=q.device, dtype=q.dtype).expand(batch, rows, k)
    q_thin = q[:, :, :k]

    modes = [sign_mode]
    if sign_mode == "auto":
        modes = ["identity_minus_q", "q_minus_identity", "diag_minus_q", "q_minus_diag"]

    last_error: Exception | None = None
    for mode in modes:
        try:
            if mode == "identity_minus_q":
                signs = torch.ones(batch, k, device=q.device, dtype=q.dtype)
                a = eye_thin - q_thin
            elif mode == "q_minus_identity":
                signs = torch.ones(batch, k, device=q.device, dtype=q.dtype)
                a = q_thin - eye_thin
            elif mode in {"diag_minus_q", "q_minus_diag"}:
                diag = torch.diagonal(q[:, :k, :k], dim1=1, dim2=2)
                signs = torch.where(diag >= 0, -torch.ones_like(diag), torch.ones_like(diag))
                s_thin = eye_thin * signs[:, None, :]
                a = s_thin - q_thin if mode == "diag_minus_q" else q_thin - s_thin
            else:
                raise ValueError(f"unknown sign_mode={sign_mode!r}")

            l1, u = nonpivoting_lu_square(a[:, :k, :])
            if rows > k:
                # L2 * U = A_bottom, so solve U.T * L2.T = A_bottom.T.
                l2_t = torch.linalg.solve_triangular(
                    u.transpose(1, 2),
                    a[:, k:, :].transpose(1, 2),
                    upper=False,
                )
                y = torch.cat([l1, l2_t.transpose(1, 2)], dim=1)
            else:
                y = l1

            # Use the square leading block of Y.T.  This is equivalent to the
            # paper's right division A / Y.T because Y has full column rank.
            w_t = torch.linalg.solve_triangular(
                l1,
                a.transpose(1, 2),
                upper=False,
                unitriangular=True,
            )
            w = w_t.transpose(1, 2)

            # Only the first k columns are uniquely defined by a thin QR panel.
            # Complete-Q nullspace columns can differ wildly between TSQR trees
            # without changing the factorization.
            q_recon_thin = eye_thin * signs[:, None, :] - torch.bmm(w, y[:, :k, :].transpose(1, 2))
            err = torch.linalg.matrix_norm((q_recon_thin - q_thin).double(), ord=1, dim=(-2, -1)).amax()
            scale = torch.linalg.matrix_norm(q_thin.double(), ord=1, dim=(-2, -1)).amax().clamp_min(1e-30)
            if torch.isfinite(err) and (err / scale) < 1e-3:
                s_full = torch.eye(rows, device=q.device, dtype=q.dtype).expand(batch, rows, rows).clone()
                diag_idx = torch.arange(k, device=q.device)
                s_full[:, diag_idx, diag_idx] = signs
                q_recon = s_full - torch.bmm(w, y.transpose(1, 2))
                return w.contiguous(), y.contiguous(), signs.contiguous(), q_recon.contiguous()
            last_error = RuntimeError(f"{mode} produced WY rel err {(err / scale).item():.3e}")
        except Exception as exc:  # noqa: BLE001 - keep auto mode resilient for prototype probing.
            last_error = exc
            continue
    raise RuntimeError(f"WY reconstruction failed: {last_error}")


def wy_reconstruction_residual(q: torch.Tensor, w: torch.Tensor, y: torch.Tensor, signs: torch.Tensor) -> float:
    rows = q.shape[1]
    k = signs.shape[1]
    eye_thin = torch.eye(rows, k, device=q.device, dtype=q.dtype).expand(q.shape[0], rows, k)
    q_recon_thin = eye_thin * signs[:, None, :] - torch.bmm(w, y[:, :k, :].transpose(1, 2))
    q_thin = q[:, :, :k]
    err = torch.linalg.matrix_norm((q_recon_thin - q_thin).double(), ord=1, dim=(-2, -1)).amax()
    scale = torch.linalg.matrix_norm(q_thin.double(), ord=1, dim=(-2, -1)).amax().clamp_min(1e-30)
    return (err / scale).item()


@triton.jit
def _wy_tmp_triton_kernel(
    w_ptr,
    c_ptr,
    tmp_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    k: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    col_tile = tl.program_id(1)
    kk = tl.program_id(2)
    row_offsets = tl.arange(0, BLOCK_ROWS)
    col_offsets = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    w_vals = tl.load(
        w_ptr + batch * rows * k + row_offsets * k + kk,
        mask=row_offsets < rows,
        other=0.0,
    )
    c_vals = tl.load(
        c_ptr + batch * rows * cols + row_offsets[:, None] * cols + col_offsets[None, :],
        mask=(row_offsets[:, None] < rows) & (col_offsets[None, :] < cols),
        other=0.0,
    )
    tmp = tl.sum(w_vals[:, None] * c_vals, axis=0)
    tl.store(
        tmp_ptr + batch * k * cols + kk * cols + col_offsets,
        tmp,
        mask=col_offsets < cols,
    )


@triton.jit
def _wy_apply_triton_kernel(
    y_ptr,
    signs_ptr,
    c_ptr,
    tmp_ptr,
    out_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    row_tile = tl.program_id(1)
    col_tile = tl.program_id(2)
    row_offsets = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    c_vals = tl.load(
        c_ptr + batch * rows * cols + row_offsets[:, None] * cols + col_offsets[None, :],
        mask=(row_offsets[:, None] < rows) & (col_offsets[None, :] < cols),
        other=0.0,
    )
    scale = tl.full((BLOCK_M,), 1.0, tl.float32)
    sign_vals = tl.load(signs_ptr + batch * k + row_offsets, mask=row_offsets < k, other=1.0)
    scale = tl.where(row_offsets < k, sign_vals, scale)
    out = scale[:, None] * c_vals
    for kk in tl.static_range(0, 64):
        if kk < k:
            y_vals = tl.load(
                y_ptr + batch * rows * k + row_offsets * k + kk,
                mask=row_offsets < rows,
                other=0.0,
            )
            tmp_vals = tl.load(
                tmp_ptr + batch * k * cols + kk * cols + col_offsets,
                mask=col_offsets < cols,
                other=0.0,
            )
            out -= y_vals[:, None] * tmp_vals[None, :]
    tl.store(
        out_ptr + batch * rows * cols + row_offsets[:, None] * cols + col_offsets[None, :],
        out,
        mask=(row_offsets[:, None] < rows) & (col_offsets[None, :] < cols),
    )


def reconstruct_wy_from_explicit_q_cuda(ext, q: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CUDA version of the paper's fixed-small-k no-pivot LU/TRSM converter."""

    q = q[:, :, :k]
    if not q.is_contiguous():
        q = q.contiguous()
    w, y, signs = ext.reconstruct_wy_lu(q, int(k))
    return w.contiguous(), y.contiguous(), signs.contiguous()


def apply_wy_qt_triton(w: torch.Tensor, y: torch.Tensor, signs: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Apply `(S - W Y.T).T @ C` without materializing the dense Q."""

    batch, rows, k = w.shape
    if y.shape != (batch, rows, k):
        raise ValueError("y shape mismatch")
    if signs.shape != (batch, k):
        raise ValueError("signs shape mismatch")
    if c.shape[0] != batch or c.shape[1] != rows:
        raise ValueError("c shape mismatch")
    cols = c.shape[2]
    block_rows = 1 << (rows - 1).bit_length()
    tmp = torch.empty((batch, k, cols), device=c.device, dtype=c.dtype)
    out = torch.empty_like(c)
    grid_tmp = (batch, triton.cdiv(cols, 16), k)
    _wy_tmp_triton_kernel[grid_tmp](
        w.contiguous(),
        c.contiguous(),
        tmp,
        rows=rows,
        cols=cols,
        k=k,
        BLOCK_ROWS=block_rows,
        BLOCK_N=16,
        num_warps=8,
    )
    grid_apply = (batch, triton.cdiv(rows, 16), triton.cdiv(cols, 16))
    _wy_apply_triton_kernel[grid_apply](
        y.contiguous(),
        signs.contiguous(),
        c.contiguous(),
        tmp,
        out,
        rows=rows,
        cols=cols,
        k=k,
        BLOCK_M=16,
        BLOCK_N=16,
        num_warps=4,
    )
    return out


def materialize_wy_q(w: torch.Tensor, y: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Side-only dense materialization of `Q = S - W Y.T` for checker bridge."""

    batch, rows, k = w.shape
    q = torch.eye(rows, device=w.device, dtype=w.dtype).expand(batch, rows, rows).clone()
    diag_idx = torch.arange(k, device=w.device)
    q[:, diag_idx, diag_idx] = signs
    q -= torch.bmm(w, y.transpose(1, 2))
    return q.contiguous()


def tsqr_panel_factor_explicit(panel: torch.Tensor, row_tile: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor one skinny panel with a two-level TSQR tree and materialize Q."""

    batch, rows, jb = panel.shape
    if row_tile < jb:
        raise ValueError("row_tile must be >= panel width")
    q_panels = []
    r_finals = []
    for b in range(batch):
        local_qs = []
        r_blocks = []
        sizes = []
        coord_positions = []
        cursor = 0
        for row0 in range(0, rows, row_tile):
            row1 = min(row0 + row_tile, rows)
            q_i, r_i = torch.linalg.qr(panel[b : b + 1, row0:row1, :], mode="complete")
            q_i = q_i[0]
            r_i = r_i[0]
            local_qs.append(q_i)
            r_rows = min(row1 - row0, jb)
            r_blocks.append(r_i[:r_rows, :])
            sizes.append(row1 - row0)
            coord_positions.extend(range(cursor, cursor + r_rows))
            cursor += row1 - row0

        r_stack = torch.cat(r_blocks, dim=0).unsqueeze(0)
        q_top, r_final = torch.linalg.qr(r_stack, mode="complete")
        q_top = q_top[0]

        q_local_block = torch.zeros(rows, rows, device=panel.device, dtype=panel.dtype)
        cursor = 0
        for q_i, size in zip(local_qs, sizes):
            q_local_block[cursor : cursor + size, cursor : cursor + size] = q_i
            cursor += size

        q_embed = torch.eye(rows, device=panel.device, dtype=panel.dtype)
        pos = torch.tensor(coord_positions, device=panel.device, dtype=torch.long)
        q_embed[pos[:, None], pos[None, :]] = q_top
        q_panels.append(torch.mm(q_local_block, q_embed))
        r_finals.append(r_final[0, :jb, :])

    return torch.stack(q_panels, dim=0), torch.stack(r_finals, dim=0)


def tsqr_panel_factor_thin(panel: torch.Tensor, row_tile: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor one skinny panel and materialize only the first `jb` TSQR Q columns."""

    batch, rows, jb = panel.shape
    if row_tile < jb:
        raise ValueError("row_tile must be >= panel width")
    q_thins = []
    r_finals = []
    for b in range(batch):
        local_q_thins = []
        r_blocks = []
        sizes = []
        coord_positions = []
        cursor = 0
        for row0 in range(0, rows, row_tile):
            row1 = min(row0 + row_tile, rows)
            q_i, r_i = torch.linalg.qr(panel[b : b + 1, row0:row1, :], mode="reduced")
            size = row1 - row0
            local_q_thins.append(q_i[0, :, :jb])
            r_blocks.append(r_i[0, :jb, :])
            sizes.append(size)
            coord_positions.extend(range(cursor, cursor + jb))
            cursor += size

        r_stack = torch.cat(r_blocks, dim=0).unsqueeze(0)
        q_top, r_final = torch.linalg.qr(r_stack, mode="reduced")
        q_top_thin = q_top[0, :, :jb]

        q_thin = torch.zeros(rows, jb, device=panel.device, dtype=panel.dtype)
        for tile, (q_i_thin, size) in enumerate(zip(local_q_thins, sizes)):
            row0 = tile * row_tile
            row1 = row0 + size
            top_slice = q_top_thin[tile * jb : (tile + 1) * jb, :]
            q_thin[row0:row1, :] = torch.mm(q_i_thin, top_slice)
        q_thins.append(q_thin)
        r_finals.append(r_final[0, :jb, :])

    return torch.stack(q_thins, dim=0), torch.stack(r_finals, dim=0)


def tsqr_panel_factor_tree(panel: torch.Tensor, row_tile: int) -> TSQRPanelTree:
    """Factor one panel and keep the TSQR tree representation.

    This stores local tile Q_i and the top-tree Q, but does not assemble the full
    `rows x rows` Q_panel.  It is still a Torch oracle; a production path would
    store compact tile reflectors instead of explicit local Q_i.
    """

    batch, rows, jb = panel.shape
    if row_tile < jb:
        raise ValueError("row_tile must be >= panel width")
    all_local_qs: list[list[torch.Tensor]] = []
    all_sizes: list[list[int]] = []
    all_pos: list[torch.Tensor] = []
    q_tops = []
    r_finals = []
    for b in range(batch):
        local_qs = []
        r_blocks = []
        sizes = []
        coord_positions = []
        cursor = 0
        for row0 in range(0, rows, row_tile):
            row1 = min(row0 + row_tile, rows)
            q_i, r_i = torch.linalg.qr(panel[b : b + 1, row0:row1, :], mode="complete")
            q_i = q_i[0]
            r_i = r_i[0]
            local_qs.append(q_i)
            r_rows = min(row1 - row0, jb)
            r_blocks.append(r_i[:r_rows, :])
            sizes.append(row1 - row0)
            coord_positions.extend(range(cursor, cursor + r_rows))
            cursor += row1 - row0

        r_stack = torch.cat(r_blocks, dim=0).unsqueeze(0)
        q_top, r_final = torch.linalg.qr(r_stack, mode="complete")
        all_local_qs.append(local_qs)
        all_sizes.append(sizes)
        all_pos.append(torch.tensor(coord_positions, device=panel.device, dtype=torch.long))
        q_tops.append(q_top[0])
        r_finals.append(r_final[0, :jb, :])

    return TSQRPanelTree(
        local_qs=all_local_qs,
        sizes=all_sizes,
        coord_positions=all_pos,
        q_tops=torch.stack(q_tops, dim=0),
        r_panel=torch.stack(r_finals, dim=0),
    )


def tsqr_panel_factor_compact_cuda(ext, panel: torch.Tensor, row_tile: int) -> TSQRCompactTree:
    """Factor one panel using CUDA tile compact reflectors plus a top compact QR."""

    batch, rows, jb = panel.shape
    h_tiles, tau_tiles, r_blocks = ext.local_tile_householder_compact(panel.contiguous(), row_tile)
    num_tiles = h_tiles.shape[1]
    stacked = r_blocks.reshape(batch, num_tiles * jb, jb).contiguous()
    h_top, tau_top = torch.geqrf(stacked)
    r_panel = torch.triu(h_top[:, :jb, :])

    all_pos: list[torch.Tensor] = []
    all_sizes: list[list[int]] = []
    for _b in range(batch):
        pos = []
        sizes = []
        for tile in range(num_tiles):
            row0 = tile * row_tile
            row1 = min(row0 + row_tile, rows)
            size = row1 - row0
            sizes.append(size)
            pos.extend(range(row0, row0 + min(size, jb)))
        all_pos.append(torch.tensor(pos, device=panel.device, dtype=torch.long))
        all_sizes.append(sizes)

    return TSQRCompactTree(
        h_tiles=h_tiles,
        tau_tiles=tau_tiles,
        h_tops=h_top,
        tau_tops=tau_top,
        coord_positions=all_pos,
        r_panel=r_panel,
        sizes=all_sizes,
    )


def tsqr_panel_factor_thin_compact_cuda_triton(ext, panel: torch.Tensor, row_tile: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor panel with compact CUDA TSQR and form only Q_thin via CUDA/Triton applies."""

    batch, rows, jb = panel.shape
    tree = tsqr_panel_factor_compact_cuda(ext, panel, row_tile)
    basis = torch.zeros((batch, rows, jb), device=panel.device, dtype=panel.dtype)
    diag = torch.arange(jb, device=panel.device)
    basis[:, diag, diag] = 1.0

    for b in range(batch):
        pos = tree.coord_positions[b]
        coord = basis[b : b + 1, pos, :].contiguous()
        coord = top_compact_apply_q_triton(
            tree.h_tops[b : b + 1, :, :].contiguous(),
            tree.tau_tops[b : b + 1, :].contiguous(),
            coord,
        )
        basis[b, pos, :] = coord[0]

    q_thin = ext.local_compact_apply_q(
        tree.h_tiles.contiguous(),
        tree.tau_tiles.contiguous(),
        basis.contiguous(),
        int(jb),
    )
    return q_thin.contiguous(), tree.r_panel.contiguous()


def tsqr_panel_factor_thin_compact_cuda_triton_fast(ext, panel: torch.Tensor, row_tile: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Same Q_thin path with less Python indexing/scatter overhead."""

    batch, rows, jb = panel.shape
    h_tiles, tau_tiles, r_blocks = ext.local_tile_householder_compact(panel.contiguous(), row_tile)
    num_tiles = h_tiles.shape[1]
    stacked = r_blocks.reshape(batch, num_tiles * jb, jb).contiguous()
    h_top, tau_top = torch.geqrf(stacked)
    r_panel = torch.triu(h_top[:, :jb, :])

    coord = torch.zeros((batch, num_tiles * jb, jb), device=panel.device, dtype=panel.dtype)
    diag = torch.arange(jb, device=panel.device)
    coord[:, diag, diag] = 1.0
    coord = top_compact_apply_q_triton(h_top.contiguous(), tau_top.contiguous(), coord)
    basis = scatter_top_coord_to_basis_triton(coord, rows, row_tile)
    q_thin = ext.local_compact_apply_q(
        h_tiles.contiguous(),
        tau_tiles.contiguous(),
        basis.contiguous(),
        int(jb),
    )
    return q_thin.contiguous(), r_panel.contiguous()


def tsqr_panel_wy_compact_tree_direct(ext, panel: torch.Tensor, row_tile: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Factor panel and reconstruct WY without materializing full Q_thin."""

    batch, rows, jb = panel.shape
    h_tiles, tau_tiles, r_blocks = ext.local_tile_householder_compact(panel.contiguous(), row_tile)
    num_tiles = h_tiles.shape[1]
    stacked = r_blocks.reshape(batch, num_tiles * jb, jb).contiguous()
    h_top, tau_top = torch.geqrf(stacked)
    r_panel = torch.triu(h_top[:, :jb, :])

    coord = torch.zeros((batch, num_tiles * jb, jb), device=panel.device, dtype=panel.dtype)
    diag = torch.arange(jb, device=panel.device)
    coord[:, diag, diag] = 1.0
    coord = top_compact_apply_q_triton(h_top.contiguous(), tau_top.contiguous(), coord)
    basis = scatter_top_coord_to_basis_triton(coord, rows, row_tile)
    w, y, signs = ext.reconstruct_wy_lu_from_tree(
        h_tiles.contiguous(),
        tau_tiles.contiguous(),
        basis.contiguous(),
        int(jb),
    )
    return w.contiguous(), y.contiguous(), signs.contiguous(), r_panel.contiguous()


def _apply_compact_qt(h: torch.Tensor, tau: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Apply Q.T from compact Householder storage to C."""

    out = c.clone()
    rows, cols = out.shape
    k = tau.shape[0]
    for j in range(k):
        tau_j = tau[j]
        if tau_j == 0:
            continue
        dot = out[j, :].clone()
        if j + 1 < rows:
            v_tail = h[j + 1 :, j]
            dot += torch.mv(out[j + 1 :, :].transpose(0, 1), v_tail)
            out[j + 1 :, :] -= v_tail[:, None] * (tau_j * dot)[None, :]
        out[j, :] -= tau_j * dot
    return out


def _apply_compact_q(h: torch.Tensor, tau: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Apply Q from compact Householder storage to C."""

    out = c.clone()
    rows, cols = out.shape
    k = tau.shape[0]
    for j in range(k - 1, -1, -1):
        tau_j = tau[j]
        if tau_j == 0:
            continue
        dot = out[j, :].clone()
        if j + 1 < rows:
            v_tail = h[j + 1 :, j]
            dot += torch.mv(out[j + 1 :, :].transpose(0, 1), v_tail)
            out[j + 1 :, :] -= v_tail[:, None] * (tau_j * dot)[None, :]
        out[j, :] -= tau_j * dot
    return out


def apply_tsqr_tree_transpose(tree: TSQRPanelTree, c: torch.Tensor) -> torch.Tensor:
    """Apply Q_panel.T to `c` using local tile Qs and the top-tree Q.

    This is the key GEQRT/TSQRT-like apply oracle: no full Q_panel is assembled.
    """

    batch, rows, cols = c.shape
    out = torch.empty_like(c)
    for b in range(batch):
        cursor = 0
        for q_i, size in zip(tree.local_qs[b], tree.sizes[b]):
            out[b, cursor : cursor + size, :] = torch.mm(
                q_i.transpose(0, 1),
                c[b, cursor : cursor + size, :],
            )
            cursor += size

        pos = tree.coord_positions[b]
        coord = out[b, pos, :]
        out[b, pos, :] = torch.mm(tree.q_tops[b].transpose(0, 1), coord)
    return out


def apply_tsqr_tree_transpose_bmm(tree: TSQRPanelTree, c: torch.Tensor) -> torch.Tensor:
    """Apply Q_panel.T using packed batched GEMM for local tiles.

    This is a kernelized oracle for the future tile-reflector apply. It still uses
    explicit local Q_i, but the expensive row-tile loop is represented as one
    batched GEMM plus one top-tree GEMM per batch item.
    """

    batch, _rows, cols = c.shape
    out = torch.empty_like(c)
    for b in range(batch):
        num_tiles = len(tree.local_qs[b])
        tile_rows = max(tree.sizes[b])
        q_stack = torch.zeros(num_tiles, tile_rows, tile_rows, device=c.device, dtype=c.dtype)
        c_stack = torch.zeros(num_tiles, tile_rows, cols, device=c.device, dtype=c.dtype)
        cursor = 0
        for tile, (q_i, size) in enumerate(zip(tree.local_qs[b], tree.sizes[b])):
            q_stack[tile, :size, :size] = q_i
            c_stack[tile, :size, :] = c[b, cursor : cursor + size, :]
            cursor += size

        applied = torch.bmm(q_stack.transpose(1, 2), c_stack)
        cursor = 0
        for tile, size in enumerate(tree.sizes[b]):
            out[b, cursor : cursor + size, :] = applied[tile, :size, :]
            cursor += size

        pos = tree.coord_positions[b]
        out[b, pos, :] = torch.mm(tree.q_tops[b].transpose(0, 1), out[b, pos, :])
    return out


def apply_tsqr_tree_to_basis(tree: TSQRPanelTree, q_embed_tail: torch.Tensor) -> torch.Tensor:
    """Apply Q_panel to a dense basis matrix without building Q_panel directly."""

    batch, rows, cols = q_embed_tail.shape
    out = q_embed_tail.clone()
    for b in range(batch):
        pos = tree.coord_positions[b]
        coord = out[b, pos, :]
        out[b, pos, :] = torch.mm(tree.q_tops[b], coord)

        cursor = 0
        for q_i, size in zip(tree.local_qs[b], tree.sizes[b]):
            out[b, cursor : cursor + size, :] = torch.mm(
                q_i,
                out[b, cursor : cursor + size, :],
            )
            cursor += size
    return out


def apply_tsqr_tree_to_basis_bmm(tree: TSQRPanelTree, q_embed_tail: torch.Tensor) -> torch.Tensor:
    """Apply Q_panel to a dense basis using packed batched GEMM local tiles."""

    batch, _rows, cols = q_embed_tail.shape
    out = q_embed_tail.clone()
    for b in range(batch):
        pos = tree.coord_positions[b]
        out[b, pos, :] = torch.mm(tree.q_tops[b], out[b, pos, :])

        num_tiles = len(tree.local_qs[b])
        tile_rows = max(tree.sizes[b])
        q_stack = torch.zeros(num_tiles, tile_rows, tile_rows, device=out.device, dtype=out.dtype)
        c_stack = torch.zeros(num_tiles, tile_rows, cols, device=out.device, dtype=out.dtype)
        cursor = 0
        for tile, (q_i, size) in enumerate(zip(tree.local_qs[b], tree.sizes[b])):
            q_stack[tile, :size, :size] = q_i
            c_stack[tile, :size, :] = out[b, cursor : cursor + size, :]
            cursor += size

        applied = torch.bmm(q_stack, c_stack)
        cursor = 0
        for tile, size in enumerate(tree.sizes[b]):
            out[b, cursor : cursor + size, :] = applied[tile, :size, :]
            cursor += size
    return out


def apply_compact_tree_transpose(tree: TSQRCompactTree, c: torch.Tensor) -> torch.Tensor:
    """Apply TSQR tree Q.T using compact tile and top reflectors."""

    batch, _rows, _cols = c.shape
    out = torch.empty_like(c)
    for b in range(batch):
        cursor = 0
        for tile, size in enumerate(tree.sizes[b]):
            h_tile = tree.h_tiles[b, tile, :size, :]
            tau_tile = tree.tau_tiles[b, tile, :]
            out[b, cursor : cursor + size, :] = _apply_compact_qt(
                h_tile,
                tau_tile,
                c[b, cursor : cursor + size, :],
            )
            cursor += size

        pos = tree.coord_positions[b]
        coord = out[b, pos, :]
        h_top = tree.h_tops[b, :, :]
        tau_top = tree.tau_tops[b, :]
        out[b, pos, :] = _apply_compact_qt(h_top, tau_top, coord)
    return out


def apply_compact_tree_transpose_cuda_local(ext, tree: TSQRCompactTree, c: torch.Tensor, block_n: int = 16) -> torch.Tensor:
    """Apply local compact reflectors with CUDA, then top compact reflectors."""

    out = ext.local_compact_apply_qt(
        tree.h_tiles.contiguous(),
        tree.tau_tiles.contiguous(),
        c.contiguous(),
        int(block_n),
    )
    batch = c.shape[0]
    for b in range(batch):
        pos = tree.coord_positions[b]
        coord = out[b, pos, :]
        h_top = tree.h_tops[b, :, :]
        tau_top = tree.tau_tops[b, :]
        out[b, pos, :] = _apply_compact_qt(h_top, tau_top, coord)
    return out


def apply_compact_tree_transpose_cuda_triton(ext, tree: TSQRCompactTree, c: torch.Tensor, block_n: int = 8) -> torch.Tensor:
    """Apply local compact reflectors with CUDA and top compact rows with Triton."""

    out = ext.local_compact_apply_qt(
        tree.h_tiles.contiguous(),
        tree.tau_tiles.contiguous(),
        c.contiguous(),
        int(block_n),
    )
    batch = c.shape[0]
    for b in range(batch):
        pos = tree.coord_positions[b]
        coord = out[b : b + 1, pos, :].contiguous()
        top = top_compact_apply_qt_triton(
            tree.h_tops[b : b + 1, :, :].contiguous(),
            tree.tau_tops[b : b + 1, :].contiguous(),
            coord,
        )
        out[b, pos, :] = top[0]
    return out


def apply_compact_tree_to_basis(tree: TSQRCompactTree, q_embed_tail: torch.Tensor) -> torch.Tensor:
    """Apply TSQR tree Q using compact tile and top reflectors."""

    batch, _rows, _cols = q_embed_tail.shape
    out = q_embed_tail.clone()
    for b in range(batch):
        pos = tree.coord_positions[b]
        h_top = tree.h_tops[b, :, :]
        tau_top = tree.tau_tops[b, :]
        out[b, pos, :] = _apply_compact_q(h_top, tau_top, out[b, pos, :])

        cursor = 0
        for tile, size in enumerate(tree.sizes[b]):
            h_tile = tree.h_tiles[b, tile, :size, :]
            tau_tile = tree.tau_tiles[b, tile, :]
            out[b, cursor : cursor + size, :] = _apply_compact_q(
                h_tile,
                tau_tile,
                out[b, cursor : cursor + size, :],
            )
            cursor += size
    return out


def tsqr_blocked_explicit(a: torch.Tensor, nb: int, row_tile: int, max_panels: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Blocked QR using explicit TSQR panel Q and explicit trailing update."""

    batch, m, n = a.shape
    work = a.clone()
    q_total = torch.eye(m, device=a.device, dtype=a.dtype).expand(batch, m, m).clone()
    panels_done = 0
    for j in range(0, min(m, n), nb):
        if max_panels and panels_done >= max_panels:
            break
        jb = min(nb, min(m, n) - j)
        q_panel, r_panel = tsqr_panel_factor_explicit(work[:, j:, j : j + jb], row_tile)

        # This is the crucial bridge: the panel representation applies to all
        # remaining columns, not just the panel, matching blocked QR semantics.
        work[:, j:, j:] = torch.bmm(q_panel.transpose(1, 2), work[:, j:, j:])
        work[:, j + jb :, j : j + jb] = 0.0
        work[:, j : j + jb, j : j + jb] = r_panel

        q_embed = torch.eye(m, device=a.device, dtype=a.dtype).expand(batch, m, m).clone()
        q_embed[:, j:, j:] = q_panel
        q_total = torch.bmm(q_total, q_embed)
        panels_done += 1
    if max_panels:
        return q_total, work
    return q_total, torch.triu(work)


def tsqr_blocked_paper_wy_apply(a: torch.Tensor, nb: int, row_tile: int, max_panels: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Blocked QR using TSQR thin-Q reconstruction into paper-style WY completion."""

    batch, m, n = a.shape
    work = a.clone()
    q_total = torch.eye(m, device=a.device, dtype=a.dtype).expand(batch, m, m).clone()
    panels_done = 0
    for j in range(0, min(m, n), nb):
        if max_panels and panels_done >= max_panels:
            break
        jb = min(nb, min(m, n) - j)
        q_panel, _r_panel = tsqr_panel_factor_explicit(work[:, j:, j : j + jb], row_tile)
        _w, _y, _signs, q_panel_wy = reconstruct_wy_from_explicit_q(q_panel, jb)

        work[:, j:, j:] = torch.bmm(q_panel_wy.transpose(1, 2), work[:, j:, j:])
        work[:, j + jb :, j : j + jb] = 0.0

        q_embed = torch.eye(m, device=a.device, dtype=a.dtype).expand(batch, m, m).clone()
        q_embed[:, j:, j:] = q_panel_wy
        q_total = torch.bmm(q_total, q_embed)
        panels_done += 1
    if max_panels:
        return q_total, work
    return q_total, torch.triu(work)


def tsqr_blocked_paper_wy_cuda_triton_apply(
    ext,
    a: torch.Tensor,
    nb: int,
    row_tile: int,
    max_panels: int = 1,
) -> torch.Tensor:
    """Partial blocked QR update using CUDA WY reconstruction and Triton WY apply."""

    work = a.clone()
    panels_done = 0
    for j in range(0, min(a.shape[1], a.shape[2]), nb):
        if max_panels and panels_done >= max_panels:
            break
        jb = min(nb, min(a.shape[1], a.shape[2]) - j)
        q_panel, _r_panel = tsqr_panel_factor_explicit(work[:, j:, j : j + jb], row_tile)
        w, y, signs = reconstruct_wy_from_explicit_q_cuda(ext, q_panel, jb)
        work[:, j:, j:] = apply_wy_qt_triton(w, y, signs, work[:, j:, j:])
        work[:, j + jb :, j : j + jb] = 0.0
        panels_done += 1
    return work


def tsqr_blocked_paper_wy_thin_cuda_triton_apply(
    ext,
    a: torch.Tensor,
    nb: int,
    row_tile: int,
    max_panels: int = 1,
) -> torch.Tensor:
    """Partial blocked QR update using only TSQR thin-Q as converter input."""

    work = a.clone()
    panels_done = 0
    for j in range(0, min(a.shape[1], a.shape[2]), nb):
        if max_panels and panels_done >= max_panels:
            break
        jb = min(nb, min(a.shape[1], a.shape[2]) - j)
        q_thin, _r_panel = tsqr_panel_factor_thin(work[:, j:, j : j + jb], row_tile)
        w, y, signs = reconstruct_wy_from_explicit_q_cuda(ext, q_thin, jb)
        work[:, j:, j:] = apply_wy_qt_triton(w, y, signs, work[:, j:, j:])
        work[:, j + jb :, j : j + jb] = 0.0
        panels_done += 1
    return work


def tsqr_blocked_paper_wy_compact_thin_cuda_triton_apply(
    ext,
    a: torch.Tensor,
    nb: int,
    row_tile: int,
    max_panels: int = 1,
) -> torch.Tensor:
    """Partial blocked QR update using compact-tree-generated TSQR thin-Q."""

    work = a.clone()
    panels_done = 0
    for j in range(0, min(a.shape[1], a.shape[2]), nb):
        if max_panels and panels_done >= max_panels:
            break
        jb = min(nb, min(a.shape[1], a.shape[2]) - j)
        q_thin, _r_panel = tsqr_panel_factor_thin_compact_cuda_triton_fast(ext, work[:, j:, j : j + jb], row_tile)
        w, y, signs = reconstruct_wy_from_explicit_q_cuda(ext, q_thin, jb)
        work[:, j:, j:] = apply_wy_qt_triton(w, y, signs, work[:, j:, j:])
        work[:, j + jb :, j : j + jb] = 0.0
        panels_done += 1
    return work


def tsqr_blocked_paper_wy_compact_thin_full(
    ext,
    a: torch.Tensor,
    nb: int,
    row_tile: int,
    max_panels: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full blocked QR using compact-tree thin-Q plus CUDA/Triton WY update.

    This is still a checker bridge: it materializes dense panel Q and Q_total to
    produce a final official `(H,tau)` through `explicit_qr_to_compact`.
    """

    batch, m, n = a.shape
    work = a.clone()
    q_total = torch.eye(m, device=a.device, dtype=a.dtype).expand(batch, m, m).clone()
    panels_done = 0
    for j in range(0, min(m, n), nb):
        if max_panels and panels_done >= max_panels:
            break
        jb = min(nb, min(m, n) - j)
        q_thin, _r_panel = tsqr_panel_factor_thin_compact_cuda_triton(ext, work[:, j:, j : j + jb], row_tile)
        w, y, signs = reconstruct_wy_from_explicit_q_cuda(ext, q_thin, jb)
        work[:, j:, j:] = apply_wy_qt_triton(w, y, signs, work[:, j:, j:])
        work[:, j + jb :, j : j + jb] = 0.0

        q_tail = materialize_wy_q(w, y, signs)
        q_embed = torch.eye(m, device=a.device, dtype=a.dtype).expand(batch, m, m).clone()
        q_embed[:, j:, j:] = q_tail
        q_total = torch.bmm(q_total, q_embed)
        panels_done += 1
    if max_panels:
        return q_total, work
    return q_total, torch.triu(work)


def tsqr_blocked_paper_wy_compact_thin_standard_output(
    ext,
    a: torch.Tensor,
    nb: int,
    row_tile: int,
    validate_panels: bool = False,
    avoid_qthin: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Full compact-tree WY QR that emits checker-style `(H,tau)` directly.

    This side prototype avoids the previous dense `Q_total -> geqrf` bridge.
    Each panel's paper-style WY is converted into sequential compact
    Householder storage by using `Y` as the reflector matrix and `diag(T)` as
    `tau`, where `W = Y*T.T`.
    """

    batch, m, n = a.shape
    work = a.clone()
    h_lower = torch.zeros_like(a)
    tau_out = torch.zeros(batch, min(m, n), device=a.device, dtype=a.dtype)
    max_panel_q_err = 0.0
    max_t_lower = 0.0
    max_t_upper = 0.0

    for j in range(0, min(m, n), nb):
        jb = min(nb, min(m, n) - j)
        if avoid_qthin:
            w, y, signs, _r_panel = tsqr_panel_wy_compact_tree_direct(ext, work[:, j:, j : j + jb], row_tile)
        else:
            q_thin, _r_panel = tsqr_panel_factor_thin_compact_cuda_triton_fast(ext, work[:, j:, j : j + jb], row_tile)
            w, y, signs = reconstruct_wy_from_explicit_q_cuda(ext, q_thin, jb)
        tau_panel, metrics = wy_to_standard_panel_tau_diag_trial(w, y, signs, validate=validate_panels)
        tau_out[:, j : j + jb] = tau_panel
        h_lower[:, j:, j : j + jb] = torch.tril(y, diagonal=-1)
        max_panel_q_err = max(max_panel_q_err, metrics["panel_q_rel_err"])
        max_t_lower = max(max_t_lower, metrics["panel_t_lower_norm"])
        max_t_upper = max(max_t_upper, metrics["panel_t_upper_norm"])

        work[:, j:, j:] = apply_wy_qt_triton(w, y, signs, work[:, j:, j:])
        work[:, j + jb :, j : j + jb] = 0.0

    h = torch.triu(work) + torch.tril(h_lower, diagonal=-1)
    metrics = {
        "max_panel_q_rel_err": max_panel_q_err,
        "max_panel_t_lower_norm": max_t_lower,
        "max_panel_t_upper_norm": max_t_upper,
    }
    return h.contiguous(), tau_out.contiguous(), metrics


def tsqr_blocked_tree_apply(a: torch.Tensor, nb: int, row_tile: int, max_panels: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Blocked QR using TSQR tree apply, avoiding full Q_panel materialization."""

    batch, m, n = a.shape
    work = a.clone()
    q_total = torch.eye(m, device=a.device, dtype=a.dtype).expand(batch, m, m).clone()
    panels_done = 0
    for j in range(0, min(m, n), nb):
        if max_panels and panels_done >= max_panels:
            break
        jb = min(nb, min(m, n) - j)
        tree = tsqr_panel_factor_tree(work[:, j:, j : j + jb], row_tile)

        work[:, j:, j:] = apply_tsqr_tree_transpose(tree, work[:, j:, j:])
        work[:, j + jb :, j : j + jb] = 0.0
        work[:, j : j + jb, j : j + jb] = tree.r_panel

        # Converter oracle: update dense Q_total by applying the same tree to the
        # tail of an embedded basis. This avoids materializing Q_panel itself, but
        # still materializes Q_total because the official checker requires (H,tau).
        q_embed_tail = torch.eye(m - j, device=a.device, dtype=a.dtype).expand(batch, m - j, m - j).clone()
        q_embed_tail = apply_tsqr_tree_to_basis(tree, q_embed_tail)
        q_embed = torch.eye(m, device=a.device, dtype=a.dtype).expand(batch, m, m).clone()
        q_embed[:, j:, j:] = q_embed_tail
        q_total = torch.bmm(q_total, q_embed)
        panels_done += 1
    if max_panels:
        return q_total, work
    return q_total, torch.triu(work)


def tsqr_blocked_tree_apply_bmm(a: torch.Tensor, nb: int, row_tile: int, max_panels: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Blocked QR using packed-GEMM TSQR tree apply."""

    batch, m, n = a.shape
    work = a.clone()
    q_total = torch.eye(m, device=a.device, dtype=a.dtype).expand(batch, m, m).clone()
    panels_done = 0
    for j in range(0, min(m, n), nb):
        if max_panels and panels_done >= max_panels:
            break
        jb = min(nb, min(m, n) - j)
        tree = tsqr_panel_factor_tree(work[:, j:, j : j + jb], row_tile)
        work[:, j:, j:] = apply_tsqr_tree_transpose_bmm(tree, work[:, j:, j:])
        work[:, j + jb :, j : j + jb] = 0.0
        work[:, j : j + jb, j : j + jb] = tree.r_panel

        q_embed_tail = torch.eye(m - j, device=a.device, dtype=a.dtype).expand(batch, m - j, m - j).clone()
        q_embed_tail = apply_tsqr_tree_to_basis_bmm(tree, q_embed_tail)
        q_embed = torch.eye(m, device=a.device, dtype=a.dtype).expand(batch, m, m).clone()
        q_embed[:, j:, j:] = q_embed_tail
        q_total = torch.bmm(q_total, q_embed)
        panels_done += 1
    if max_panels:
        return q_total, work
    return q_total, torch.triu(work)


def tsqr_blocked_compact_apply_cuda(ext, a: torch.Tensor, nb: int, row_tile: int, max_panels: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Blocked QR using CUDA local compact reflectors and compact tree apply."""

    batch, m, n = a.shape
    work = a.clone()
    q_total = torch.eye(m, device=a.device, dtype=a.dtype).expand(batch, m, m).clone()
    panels_done = 0
    for j in range(0, min(m, n), nb):
        if max_panels and panels_done >= max_panels:
            break
        jb = min(nb, min(m, n) - j)
        tree = tsqr_panel_factor_compact_cuda(ext, work[:, j:, j : j + jb], row_tile)
        work[:, j:, j:] = apply_compact_tree_transpose(tree, work[:, j:, j:])
        work[:, j + jb :, j : j + jb] = 0.0
        work[:, j : j + jb, j : j + jb] = tree.r_panel

        q_embed_tail = torch.eye(m - j, device=a.device, dtype=a.dtype).expand(batch, m - j, m - j).clone()
        q_embed_tail = apply_compact_tree_to_basis(tree, q_embed_tail)
        q_embed = torch.eye(m, device=a.device, dtype=a.dtype).expand(batch, m, m).clone()
        q_embed[:, j:, j:] = q_embed_tail
        q_total = torch.bmm(q_total, q_embed)
        panels_done += 1
    if max_panels:
        return q_total, work
    return q_total, torch.triu(work)


def tsqr_blocked_compact_apply_cuda_local(ext, a: torch.Tensor, nb: int, row_tile: int, max_panels: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Blocked QR using CUDA local reflector apply and compact top apply."""

    batch, m, n = a.shape
    work = a.clone()
    q_total = torch.eye(m, device=a.device, dtype=a.dtype).expand(batch, m, m).clone()
    panels_done = 0
    for j in range(0, min(m, n), nb):
        if max_panels and panels_done >= max_panels:
            break
        jb = min(nb, min(m, n) - j)
        tree = tsqr_panel_factor_compact_cuda(ext, work[:, j:, j : j + jb], row_tile)
        work[:, j:, j:] = apply_compact_tree_transpose_cuda_local(ext, tree, work[:, j:, j:], block_n=16)
        work[:, j + jb :, j : j + jb] = 0.0
        work[:, j : j + jb, j : j + jb] = tree.r_panel

        # Keep converter oracle on the slower compact path; active submission work
        # should avoid dense q_total entirely.
        q_embed_tail = torch.eye(m - j, device=a.device, dtype=a.dtype).expand(batch, m - j, m - j).clone()
        q_embed_tail = apply_compact_tree_to_basis(tree, q_embed_tail)
        q_embed = torch.eye(m, device=a.device, dtype=a.dtype).expand(batch, m, m).clone()
        q_embed[:, j:, j:] = q_embed_tail
        q_total = torch.bmm(q_total, q_embed)
        panels_done += 1
    if max_panels:
        return q_total, work
    return q_total, torch.triu(work)


def tsqr_blocked_compact_apply_cuda_triton(ext, a: torch.Tensor, nb: int, row_tile: int, max_panels: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Blocked QR with CUDA local apply and Triton top compact-coordinate apply."""

    batch, m, n = a.shape
    work = a.clone()
    q_total = torch.eye(m, device=a.device, dtype=a.dtype).expand(batch, m, m).clone()
    panels_done = 0
    for j in range(0, min(m, n), nb):
        if max_panels and panels_done >= max_panels:
            break
        jb = min(nb, min(m, n) - j)
        tree = tsqr_panel_factor_compact_cuda(ext, work[:, j:, j : j + jb], row_tile)
        work[:, j:, j:] = apply_compact_tree_transpose_cuda_triton(ext, tree, work[:, j:, j:], block_n=8)
        work[:, j + jb :, j : j + jb] = 0.0
        work[:, j : j + jb, j : j + jb] = tree.r_panel

        q_embed_tail = torch.eye(m - j, device=a.device, dtype=a.dtype).expand(batch, m - j, m - j).clone()
        q_embed_tail = apply_compact_tree_to_basis(tree, q_embed_tail)
        q_embed = torch.eye(m, device=a.device, dtype=a.dtype).expand(batch, m, m).clone()
        q_embed[:, j:, j:] = q_embed_tail
        q_total = torch.bmm(q_total, q_embed)
        panels_done += 1
    if max_panels:
        return q_total, work
    return q_total, torch.triu(work)


def tsqr_checker_bridge(a: torch.Tensor, nb: int, row_tile: int) -> tuple[torch.Tensor, torch.Tensor]:
    q, _ = tsqr_blocked_explicit(a, nb, row_tile)
    return explicit_qr_to_compact(a, q)


def tsqr_tree_checker_bridge(a: torch.Tensor, nb: int, row_tile: int) -> tuple[torch.Tensor, torch.Tensor]:
    q, _ = tsqr_blocked_tree_apply(a, nb, row_tile)
    return explicit_qr_to_compact(a, q)


def tsqr_compact_checker_bridge(ext, a: torch.Tensor, nb: int, row_tile: int) -> tuple[torch.Tensor, torch.Tensor]:
    q, _ = tsqr_blocked_compact_apply_cuda(ext, a, nb, row_tile)
    return explicit_qr_to_compact(a, q)


def run(args: argparse.Namespace) -> None:
    ext = load_tsqr_ext()
    data = qr_official.generate_input(args.batch, args.n, args.cond, args.seed, args.case)
    print(
        f"case n={args.n} batch={args.batch} case={args.case} nb={args.nb} "
        f"row_tile={args.row_tile} device={data.device}",
        flush=True,
    )

    explicit_time, (q, r) = time_fn(
        lambda: tsqr_blocked_explicit(data, args.nb, args.row_tile, args.max_panels),
        args.warmup,
        args.trials,
    )
    recon, orth, lower = residuals(data, q, r)
    label = "tsqr_partial_update" if args.max_panels else "tsqr_explicit"
    print(
        f"{label:20s} median={explicit_time.median_ms:.3f} ms "
        f"min={explicit_time.min_ms:.3f} max={explicit_time.max_ms:.3f} "
        f"scaled_recon={recon:.3g} scaled_orth={orth:.3g} scaled_lower={lower:.3g}",
        flush=True,
    )

    wy_k = min(args.nb if args.max_panels else args.n, q.shape[-1])
    try:
        wy_time, (w_wy, y_wy, signs_wy, _q_wy) = time_fn(
            lambda: reconstruct_wy_from_explicit_q(q, wy_k, args.wy_sign_mode),
            0,
            1,
        )
        wy_err = wy_reconstruction_residual(q, w_wy, y_wy, signs_wy)
        print(
            f"paper_wy_recon     one_shot={wy_time.median_ms:.3f} ms "
            f"k={wy_k} wy_rel_err={wy_err:.3e}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic prototype path.
        print(f"paper_wy_recon     failed k={wy_k}: {exc}", flush=True)

    if args.max_panels:
        paper_wy_time, (q_paper_wy, r_paper_wy) = time_fn(
            lambda: tsqr_blocked_paper_wy_apply(data, args.nb, args.row_tile, args.max_panels),
            0,
            1,
        )
        recon_pw, orth_pw, _lower_pw = residuals(data, q_paper_wy, r_paper_wy)
        diff_pw = torch.linalg.matrix_norm((r_paper_wy - r).double(), ord=1, dim=(-2, -1)).amax()
        scale_pw = torch.linalg.matrix_norm(r.double(), ord=1, dim=(-2, -1)).amax().clamp_min(1e-30)
        print(
            f"paper_wy_partial   one_shot={paper_wy_time.median_ms:.3f} ms "
            f"scaled_recon={recon_pw:.3g} scaled_orth={orth_pw:.3g} "
            f"work_rel_diff={(diff_pw / scale_pw).item():.3e}",
            flush=True,
        )
        cuda_wy_time, work_cuda_wy = time_fn(
            lambda: tsqr_blocked_paper_wy_cuda_triton_apply(ext, data, args.nb, args.row_tile, args.max_panels),
            0,
            1,
        )
        diff_cuda_wy = torch.linalg.matrix_norm((work_cuda_wy - r_paper_wy).double(), ord=1, dim=(-2, -1)).amax()
        print(
            f"paper_wy_cuda_tri  one_shot={cuda_wy_time.median_ms:.3f} ms "
            f"vs_torch_wy_rel_diff={(diff_cuda_wy / scale_pw).item():.3e}",
            flush=True,
        )
        thin_cuda_wy_time, work_thin_cuda_wy = time_fn(
            lambda: tsqr_blocked_paper_wy_thin_cuda_triton_apply(ext, data, args.nb, args.row_tile, args.max_panels),
            0,
            1,
        )
        diff_thin_cuda_wy = torch.linalg.matrix_norm((work_thin_cuda_wy - r_paper_wy).double(), ord=1, dim=(-2, -1)).amax()
        diff_thin_vs_complete = torch.linalg.matrix_norm(
            (work_thin_cuda_wy - work_cuda_wy).double(),
            ord=1,
            dim=(-2, -1),
        ).amax()
        print(
            f"paper_wy_thin_tri  one_shot={thin_cuda_wy_time.median_ms:.3f} ms "
            f"vs_torch_wy_rel_diff={(diff_thin_cuda_wy / scale_pw).item():.3e} "
            f"vs_complete_cuda_diff={(diff_thin_vs_complete / scale_pw).item():.3e}",
            flush=True,
        )
        compact_thin_time, work_compact_thin = time_fn(
            lambda: tsqr_blocked_paper_wy_compact_thin_cuda_triton_apply(
                ext,
                data,
                args.nb,
                args.row_tile,
                args.max_panels,
            ),
            0,
            1,
        )
        diff_compact_thin = torch.linalg.matrix_norm((work_compact_thin - r_paper_wy).double(), ord=1, dim=(-2, -1)).amax()
        diff_compact_vs_thin = torch.linalg.matrix_norm(
            (work_compact_thin - work_thin_cuda_wy).double(),
            ord=1,
            dim=(-2, -1),
        ).amax()
        print(
            f"paper_wy_cmpthin  one_shot={compact_thin_time.median_ms:.3f} ms "
            f"vs_torch_wy_rel_diff={(diff_compact_thin / scale_pw).item():.3e} "
            f"vs_thin_cuda_diff={(diff_compact_vs_thin / scale_pw).item():.3e}",
            flush=True,
        )
        tree_time, (q_tree, r_tree) = time_fn(
            lambda: tsqr_blocked_tree_apply(data, args.nb, args.row_tile, args.max_panels),
            0,
            1,
        )
        recon_t, orth_t, lower_t = residuals(data, q_tree, r_tree)
        diff = torch.linalg.matrix_norm((r_tree - r).double(), ord=1, dim=(-2, -1)).amax()
        scale = torch.linalg.matrix_norm(r.double(), ord=1, dim=(-2, -1)).amax().clamp_min(1e-30)
        print(
            f"tsqr_tree_partial  one_shot={tree_time.median_ms:.3f} ms "
            f"scaled_recon={recon_t:.3g} scaled_orth={orth_t:.3g} "
            f"work_rel_diff={(diff / scale).item():.3e}",
            flush=True,
        )
        bmm_time, (q_bmm, r_bmm) = time_fn(
            lambda: tsqr_blocked_tree_apply_bmm(data, args.nb, args.row_tile, args.max_panels),
            0,
            1,
        )
        recon_b, orth_b, _lower_b = residuals(data, q_bmm, r_bmm)
        diff_b = torch.linalg.matrix_norm((r_bmm - r).double(), ord=1, dim=(-2, -1)).amax()
        diff_tree_b = torch.linalg.matrix_norm((r_bmm - r_tree).double(), ord=1, dim=(-2, -1)).amax()
        print(
            f"tsqr_bmm_partial   one_shot={bmm_time.median_ms:.3f} ms "
            f"scaled_recon={recon_b:.3g} scaled_orth={orth_b:.3g} "
            f"work_rel_diff={(diff_b / scale).item():.3e} "
            f"vs_tree_rel_diff={(diff_tree_b / scale).item():.3e}",
            flush=True,
        )
        compact_time, (q_compact, r_compact) = time_fn(
            lambda: tsqr_blocked_compact_apply_cuda(ext, data, args.nb, args.row_tile, args.max_panels),
            0,
            1,
        )
        recon_c, orth_c, _lower_c = residuals(data, q_compact, r_compact)
        diff_c = torch.linalg.matrix_norm((r_compact - r).double(), ord=1, dim=(-2, -1)).amax()
        diff_tree_c = torch.linalg.matrix_norm((r_compact - r_tree).double(), ord=1, dim=(-2, -1)).amax()
        print(
            f"tsqr_compact_part  one_shot={compact_time.median_ms:.3f} ms "
            f"scaled_recon={recon_c:.3g} scaled_orth={orth_c:.3g} "
            f"work_rel_diff={(diff_c / scale).item():.3e} "
            f"vs_tree_rel_diff={(diff_tree_c / scale).item():.3e}",
            flush=True,
        )
        cuda_local_time, (q_cuda_local, r_cuda_local) = time_fn(
            lambda: tsqr_blocked_compact_apply_cuda_local(ext, data, args.nb, args.row_tile, args.max_panels),
            0,
            1,
        )
        recon_l, orth_l, _lower_l = residuals(data, q_cuda_local, r_cuda_local)
        diff_l = torch.linalg.matrix_norm((r_cuda_local - r).double(), ord=1, dim=(-2, -1)).amax()
        diff_compact_l = torch.linalg.matrix_norm((r_cuda_local - r_compact).double(), ord=1, dim=(-2, -1)).amax()
        print(
            f"tsqr_cuda_local    one_shot={cuda_local_time.median_ms:.3f} ms "
            f"scaled_recon={recon_l:.3g} scaled_orth={orth_l:.3g} "
            f"work_rel_diff={(diff_l / scale).item():.3e} "
            f"vs_compact_rel_diff={(diff_compact_l / scale).item():.3e}",
            flush=True,
        )
        cuda_triton_time, (q_cuda_triton, r_cuda_triton) = time_fn(
            lambda: tsqr_blocked_compact_apply_cuda_triton(ext, data, args.nb, args.row_tile, args.max_panels),
            0,
            1,
        )
        recon_tn, orth_tn, _lower_tn = residuals(data, q_cuda_triton, r_cuda_triton)
        diff_tn = torch.linalg.matrix_norm((r_cuda_triton - r).double(), ord=1, dim=(-2, -1)).amax()
        diff_local_tn = torch.linalg.matrix_norm((r_cuda_triton - r_cuda_local).double(), ord=1, dim=(-2, -1)).amax()
        print(
            f"tsqr_cuda_triton   one_shot={cuda_triton_time.median_ms:.3f} ms "
            f"scaled_recon={recon_tn:.3g} scaled_orth={orth_tn:.3g} "
            f"work_rel_diff={(diff_tn / scale).item():.3e} "
            f"vs_cuda_local_diff={(diff_local_tn / scale).item():.3e}",
            flush=True,
        )
        print("compact conversion skipped because --max-panels produced a partial QR", flush=True)
        return

    tree_time, (q_tree, r_tree) = time_fn(
        lambda: tsqr_blocked_tree_apply(data, args.nb, args.row_tile),
        args.warmup,
        args.trials,
    )
    recon_t, orth_t, lower_t = residuals(data, q_tree, r_tree)
    r_diff = torch.linalg.matrix_norm((r_tree - r).double(), ord=1, dim=(-2, -1)).amax()
    r_scale = torch.linalg.matrix_norm(r.double(), ord=1, dim=(-2, -1)).amax().clamp_min(1e-30)
    q_diff = torch.linalg.matrix_norm((q_tree - q).double(), ord=1, dim=(-2, -1)).amax()
    print(
        f"tsqr_tree_apply    median={tree_time.median_ms:.3f} ms "
        f"min={tree_time.min_ms:.3f} max={tree_time.max_ms:.3f} "
        f"scaled_recon={recon_t:.3g} scaled_orth={orth_t:.3g} scaled_lower={lower_t:.3g} "
        f"r_rel_diff={(r_diff / r_scale).item():.3e} q_l1_diff={q_diff.item():.3e}",
        flush=True,
    )

    bmm_tree_time, (q_bmm, r_bmm) = time_fn(
        lambda: tsqr_blocked_tree_apply_bmm(data, args.nb, args.row_tile),
        args.warmup,
        args.trials,
    )
    recon_b, orth_b, lower_b = residuals(data, q_bmm, r_bmm)
    r_diff_b = torch.linalg.matrix_norm((r_bmm - r).double(), ord=1, dim=(-2, -1)).amax()
    q_diff_b = torch.linalg.matrix_norm((q_bmm - q).double(), ord=1, dim=(-2, -1)).amax()
    print(
        f"tsqr_bmm_tree      median={bmm_tree_time.median_ms:.3f} ms "
        f"min={bmm_tree_time.min_ms:.3f} max={bmm_tree_time.max_ms:.3f} "
        f"scaled_recon={recon_b:.3g} scaled_orth={orth_b:.3g} scaled_lower={lower_b:.3g} "
        f"r_rel_diff={(r_diff_b / r_scale).item():.3e} q_l1_diff={q_diff_b.item():.3e}",
        flush=True,
    )

    paper_wy_time, (q_paper_wy, r_paper_wy) = time_fn(
        lambda: tsqr_blocked_paper_wy_apply(data, args.nb, args.row_tile),
        0,
        1,
    )
    recon_pw, orth_pw, lower_pw = residuals(data, q_paper_wy, r_paper_wy)
    h_pw, tau_pw = explicit_qr_to_compact(data, q_paper_wy)
    ok_pw, msg_pw = qr_official.check_implementation(data, (h_pw, tau_pw))
    print(
        f"paper_wy_blocked   one_shot={paper_wy_time.median_ms:.3f} ms "
        f"scaled_recon={recon_pw:.3g} scaled_orth={orth_pw:.3g} scaled_lower={lower_pw:.3g} "
        f"checker_ok={ok_pw}; {msg_pw}",
        flush=True,
    )

    compact_thin_full_time, (q_compact_thin_wy, r_compact_thin_wy) = time_fn(
        lambda: tsqr_blocked_paper_wy_compact_thin_full(ext, data, args.nb, args.row_tile),
        0,
        1,
    )
    recon_ctw, orth_ctw, lower_ctw = residuals(data, q_compact_thin_wy, r_compact_thin_wy)
    h_ctw, tau_ctw = explicit_qr_to_compact(data, q_compact_thin_wy)
    ok_ctw, msg_ctw = qr_official.check_implementation(data, (h_ctw, tau_ctw))
    print(
        f"cmpthin_wy_full   one_shot={compact_thin_full_time.median_ms:.3f} ms "
        f"scaled_recon={recon_ctw:.3g} scaled_orth={orth_ctw:.3g} scaled_lower={lower_ctw:.3g} "
        f"checker_ok={ok_ctw}; {msg_ctw}",
        flush=True,
    )
    direct_wy_time, (h_direct_wy, tau_direct_wy, direct_wy_metrics) = time_fn(
        lambda: tsqr_blocked_paper_wy_compact_thin_standard_output(ext, data, args.nb, args.row_tile),
        0,
        1,
    )
    ok_direct_wy, msg_direct_wy = qr_official.check_implementation(data, (h_direct_wy, tau_direct_wy))
    print(
        f"wy_direct_compact one_shot={direct_wy_time.median_ms:.3f} ms "
        f"panel_q_err={direct_wy_metrics['max_panel_q_rel_err']:.3e} "
        f"T_lower={direct_wy_metrics['max_panel_t_lower_norm']:.3e} "
        f"T_upper={direct_wy_metrics['max_panel_t_upper_norm']:.3e} "
        f"checker_ok={ok_direct_wy}; {msg_direct_wy}",
        flush=True,
    )
    if args.try_wy_compact_converter:
        try:
            def build_wy_diag_trial():
                w_trial, y_trial, signs_trial, _ = reconstruct_wy_from_explicit_q(
                    q_compact_thin_wy,
                    args.n,
                    "identity_minus_q",
                )
                return wy_to_standard_compact_diag_trial(data, w_trial, y_trial, signs_trial)

            wy_compact_time, (h_wyc, tau_wyc, wyc_metrics) = time_fn(
                build_wy_diag_trial,
                0,
                1,
            )
            ok_wyc, msg_wyc = qr_official.check_implementation(data, (h_wyc, tau_wyc))
            print(
                f"wy_diag_converter one_shot={wy_compact_time.median_ms:.3f} ms "
                f"q_rel_err={wyc_metrics['q_rel_err']:.3e} "
                f"T_lower={wyc_metrics['t_lower_norm']:.3e} "
                f"T_upper={wyc_metrics['t_upper_norm']:.3e} "
                f"checker_ok={ok_wyc}; {msg_wyc}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic prototype path.
            print(f"wy_diag_converter failed: {exc}", flush=True)

    compact_tree_time, (q_compact, r_compact) = time_fn(
        lambda: tsqr_blocked_compact_apply_cuda(ext, data, args.nb, args.row_tile),
        args.warmup,
        args.trials,
    )
    recon_c, orth_c, lower_c = residuals(data, q_compact, r_compact)
    r_diff_c = torch.linalg.matrix_norm((r_compact - r).double(), ord=1, dim=(-2, -1)).amax()
    q_diff_c = torch.linalg.matrix_norm((q_compact - q).double(), ord=1, dim=(-2, -1)).amax()
    print(
        f"tsqr_compact_tree  median={compact_tree_time.median_ms:.3f} ms "
        f"min={compact_tree_time.min_ms:.3f} max={compact_tree_time.max_ms:.3f} "
        f"scaled_recon={recon_c:.3g} scaled_orth={orth_c:.3g} scaled_lower={lower_c:.3g} "
        f"r_rel_diff={(r_diff_c / r_scale).item():.3e} q_l1_diff={q_diff_c.item():.3e}",
        flush=True,
    )

    cuda_local_tree_time, (q_cuda_local, r_cuda_local) = time_fn(
        lambda: tsqr_blocked_compact_apply_cuda_local(ext, data, args.nb, args.row_tile),
        args.warmup,
        args.trials,
    )
    recon_l, orth_l, lower_l = residuals(data, q_cuda_local, r_cuda_local)
    r_diff_l = torch.linalg.matrix_norm((r_cuda_local - r).double(), ord=1, dim=(-2, -1)).amax()
    q_diff_l = torch.linalg.matrix_norm((q_cuda_local - q).double(), ord=1, dim=(-2, -1)).amax()
    print(
        f"tsqr_cuda_local   median={cuda_local_tree_time.median_ms:.3f} ms "
        f"min={cuda_local_tree_time.min_ms:.3f} max={cuda_local_tree_time.max_ms:.3f} "
        f"scaled_recon={recon_l:.3g} scaled_orth={orth_l:.3g} scaled_lower={lower_l:.3g} "
        f"r_rel_diff={(r_diff_l / r_scale).item():.3e} q_l1_diff={q_diff_l.item():.3e}",
        flush=True,
    )

    compact_time, (h, tau) = time_fn(
        lambda: explicit_qr_to_compact(data, q),
        args.warmup,
        args.trials,
    )
    ok, msg = qr_official.check_implementation(data, (h, tau))
    print(
        f"compact_converter   median={compact_time.median_ms:.3f} ms "
        f"min={compact_time.min_ms:.3f} max={compact_time.max_ms:.3f} ok={ok}; {msg}",
        flush=True,
    )

    bridge_time, (h2, tau2) = time_fn(
        lambda: tsqr_checker_bridge(data, args.nb, args.row_tile),
        0,
        1,
    )
    ok2, msg2 = qr_official.check_implementation(data, (h2, tau2))
    print(
        f"tsqr_checker_bridge one_shot={bridge_time.median_ms:.3f} ms ok={ok2}; {msg2}",
        flush=True,
    )

    tree_bridge_time, (h3, tau3) = time_fn(
        lambda: tsqr_tree_checker_bridge(data, args.nb, args.row_tile),
        0,
        1,
    )
    ok3, msg3 = qr_official.check_implementation(data, (h3, tau3))
    print(
        f"tsqr_tree_bridge    one_shot={tree_bridge_time.median_ms:.3f} ms ok={ok3}; {msg3}",
        flush=True,
    )

    compact_bridge_time, (h4, tau4) = time_fn(
        lambda: tsqr_compact_checker_bridge(ext, data, args.nb, args.row_tile),
        0,
        1,
    )
    ok4, msg4 = qr_official.check_implementation(data, (h4, tau4))
    print(
        f"tsqr_compact_bridge one_shot={compact_bridge_time.median_ms:.3f} ms ok={ok4}; {msg4}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--case", default="dense")
    parser.add_argument("--cond", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--nb", type=int, default=16)
    parser.add_argument("--row-tile", type=int, default=128)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-panels", type=int, default=0, help="debug only: stop after this many panels")
    parser.add_argument(
        "--try-wy-compact-converter",
        action="store_true",
        help="diagnose a cheap WY/tree -> standard compact (H,tau) shortcut",
    )
    parser.add_argument(
        "--wy-sign-mode",
        default="auto",
        choices=["auto", "identity_minus_q", "q_minus_identity", "diag_minus_q", "q_minus_diag"],
        help="sign convention for paper-style explicit-Q to WY reconstruction",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
