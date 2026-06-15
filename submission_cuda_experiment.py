#!POPCORN leaderboard qr_v2
#!POPCORN gpu B200

from __future__ import annotations

import os
from typing import Literal

import torch

try:
    from task import input_t, output_t
except ModuleNotFoundError:
    input_t = torch.Tensor
    output_t = tuple[torch.Tensor, torch.Tensor]

BackendName = Literal["auto", "torch_cuda", "torch_cpu", "cutedsl", "cuda"]

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.cute.runtime import from_dlpack
    _CUTE_AVAILABLE = True
except Exception:
    cutlass = None
    cute = None
    Int32 = int
    from_dlpack = None
    _CUTE_AVAILABLE = False

if _CUTE_AVAILABLE:

    @cute.kernel
    def _part2_3_factor_apply_panel_kernel(
        h: cute.Tensor,
        tau: cute.Tensor,
        batch_count: Int32,
        m: Int32,
        n: Int32,
        j_start: Int32,
        j_end: Int32,
    ):
        from cutlass.utils.smem_allocator import SmemAllocator

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        smem = SmemAllocator()
        partial = smem.allocate_tensor(cutlass.Float32, 128, byte_alignment=16)

        if bidx < batch_count:
            j = j_start
            while j < j_end:
                alpha = h[bidx, j, j]
                local = alpha * 0.0
                row = j + 1 + tidx
                while row < m:
                    x = h[bidx, row, j]
                    local += x * x
                    row += 128

                partial[tidx] = local
                cute.arch.sync_threads()

                if tidx < 64:
                    partial[tidx] = partial[tidx] + partial[tidx + 64]
                cute.arch.sync_threads()
                if tidx < 32:
                    partial[tidx] = partial[tidx] + partial[tidx + 32]
                cute.arch.sync_threads()
                if tidx < 16:
                    partial[tidx] = partial[tidx] + partial[tidx + 16]
                cute.arch.sync_threads()
                if tidx < 8:
                    partial[tidx] = partial[tidx] + partial[tidx + 8]
                cute.arch.sync_threads()
                if tidx < 4:
                    partial[tidx] = partial[tidx] + partial[tidx + 4]
                cute.arch.sync_threads()
                if tidx < 2:
                    partial[tidx] = partial[tidx] + partial[tidx + 2]
                cute.arch.sync_threads()
                if tidx < 1:
                    partial[tidx] = partial[tidx] + partial[tidx + 1]
                cute.arch.sync_threads()

                if tidx == 0:
                    sigma = partial[0]
                    x_norm = cute.math.sqrt(alpha * alpha + sigma)
                    beta = x_norm
                    if alpha >= 0.0:
                        beta = -x_norm

                    tau_j = alpha * 0.0
                    scale = alpha * 0.0
                    if sigma > 0.0:
                        tau_j = (beta - alpha) / beta
                        scale = 1.0 / (alpha - beta)
                        h[bidx, j, j] = beta
                    else:
                        h[bidx, j, j] = alpha

                    tau[bidx, j] = tau_j
                    partial[0] = scale
                    partial[1] = tau_j

                cute.arch.sync_threads()
                scale = partial[0]
                tau_j = partial[1]
                row = j + 1 + tidx
                while row < m:
                    h[bidx, row, j] = h[bidx, row, j] * scale
                    row += 128
                cute.arch.sync_threads()

                target = j + 1 + tidx
                if target < j_end:
                    dot = h[bidx, j, target]
                    row2 = j + 1
                    while row2 < m:
                        dot += h[bidx, row2, j] * h[bidx, row2, target]
                        row2 += 1

                    w = tau_j * dot
                    h[bidx, j, target] = h[bidx, j, target] - w
                    row2 = j + 1
                    while row2 < m:
                        h[bidx, row2, target] = h[bidx, row2, target] - h[bidx, row2, j] * w
                        row2 += 1

                cute.arch.sync_threads()
                j += 1


    @cute.jit
    def part2_3_factor_apply_panel_cuda(
        h: cute.Tensor,
        tau: cute.Tensor,
        batch_count: Int32,
        m: Int32,
        n: Int32,
        j_start: Int32,
        j_end: Int32,
    ):
        _part2_3_factor_apply_panel_kernel(h, tau, batch_count, m, n, j_start, j_end).launch(
            grid=(batch_count, 1, 1), block=(128, 1, 1)
        )


    @cute.kernel
    def _part5_apply_panel_wy_gemv_kernel(
        h: cute.Tensor,
        tau: cute.Tensor,
        tmat: cute.Tensor,
        batch_count: Int32,
        m: Int32,
        n: Int32,
        j_start: Int32,
        j_end: Int32,
    ):
        from cutlass.utils.smem_allocator import SmemAllocator

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        col = j_end + bidy
        panel_cols = j_end - j_start

        smem = SmemAllocator()
        partial = smem.allocate_tensor(cutlass.Float32, 128, byte_alignment=16)
        y_buf = smem.allocate_tensor(cutlass.Float32, 32, byte_alignment=16)
        z_buf = smem.allocate_tensor(cutlass.Float32, 32, byte_alignment=16)

        if bidx < batch_count and col < n:
            i = 0
            while i < panel_cols:
                diag = j_start + i
                local = h[bidx, j_start, col] * 0.0
                row = j_start + tidx
                while row < m:
                    v_val = h[bidx, j_start, col] * 0.0
                    if row == diag:
                        v_val = 1.0
                    if row > diag:
                        v_val = h[bidx, row, diag]
                    local += v_val * h[bidx, row, col]
                    row += 128

                partial[tidx] = local
                cute.arch.sync_threads()

                if tidx < 64:
                    partial[tidx] = partial[tidx] + partial[tidx + 64]
                cute.arch.sync_threads()
                if tidx < 32:
                    partial[tidx] = partial[tidx] + partial[tidx + 32]
                cute.arch.sync_threads()
                if tidx < 16:
                    partial[tidx] = partial[tidx] + partial[tidx + 16]
                cute.arch.sync_threads()
                if tidx < 8:
                    partial[tidx] = partial[tidx] + partial[tidx + 8]
                cute.arch.sync_threads()
                if tidx < 4:
                    partial[tidx] = partial[tidx] + partial[tidx + 4]
                cute.arch.sync_threads()
                if tidx < 2:
                    partial[tidx] = partial[tidx] + partial[tidx + 2]
                cute.arch.sync_threads()
                if tidx < 1:
                    partial[tidx] = partial[tidx] + partial[tidx + 1]
                cute.arch.sync_threads()

                if tidx == 0:
                    y_buf[i] = partial[0]
                cute.arch.sync_threads()
                i += 1

            if tidx == 0:
                row_i = 0
                while row_i < panel_cols:
                    acc = h[bidx, j_start, col] * 0.0
                    k = 0
                    while k < panel_cols:
                        acc += tmat[bidx, row_i, k] * y_buf[k]
                        k += 1
                    z_buf[row_i] = acc
                    row_i += 1
            cute.arch.sync_threads()

            row = j_start + tidx
            while row < m:
                update = h[bidx, j_start, col] * 0.0
                i2 = 0
                while i2 < panel_cols:
                    diag = j_start + i2
                    v_val = h[bidx, j_start, col] * 0.0
                    if row == diag:
                        v_val = 1.0
                    if row > diag:
                        v_val = h[bidx, row, diag]
                    update += v_val * z_buf[i2]
                    i2 += 1
                h[bidx, row, col] = h[bidx, row, col] - update
                row += 128


    @cute.jit
    def part5_apply_panel_wy_gemv_cuda(
        h: cute.Tensor,
        tau: cute.Tensor,
        tmat: cute.Tensor,
        batch_count: Int32,
        m: Int32,
        n: Int32,
        j_start: Int32,
        j_end: Int32,
    ):
        cols = n - j_end
        _part5_apply_panel_wy_gemv_kernel(
            h, tau, tmat, batch_count, m, n, j_start, j_end
        ).launch(grid=(batch_count, cols, 1), block=(128, 1, 1))

from torch.utils.cpp_extension import load_inline


_EXT = None


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

    cpp_src = r"""
#include <torch/extension.h>

void apply_panel_wy_tiled_cuda(torch::Tensor h, torch::Tensor tmat, int64_t j_start, int64_t j_end);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("apply_panel_wy_tiled_cuda", &apply_panel_wy_tiled_cuda, "Tiled compact WY trailing update (CUDA)");
}
"""

    cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

namespace {

constexpr int THREADS = 128;
constexpr int MAX_NB = 32;
constexpr int ROW_TILE = 64;
constexpr int FUSED_ROWS = 128;
constexpr int COL_TILE = 16;

__device__ __forceinline__ float v_value(const float* h, int b, int m, int n, int row, int diag) {
  if (row == diag) {
    return 1.0f;
  }
  if (row > diag) {
    return h[(static_cast<int64_t>(b) * m + row) * n + diag];
  }
  return 0.0f;
}

__global__ void compute_y_partial_kernel(
    const float* __restrict__ h,
    float* __restrict__ y_partial,
    int batch,
    int m,
    int n,
    int j_start,
    int j_end,
    int trailing_cols,
    int row_tiles) {
  __shared__ float partial[THREADS];

  int col_off = blockIdx.x;
  int b = blockIdx.y;
  int z_index = blockIdx.z;
  int panel_i = z_index / row_tiles;
  int row_tile = z_index - panel_i * row_tiles;
  int col = j_end + col_off;
  int diag = j_start + panel_i;
  int row_begin = j_start + row_tile * ROW_TILE;
  int row_end = row_begin + ROW_TILE;
  if (row_end > m) {
    row_end = m;
  }

  float acc = 0.0f;
  for (int row = row_begin + threadIdx.x; row < row_end; row += blockDim.x) {
    float v = v_value(h, b, m, n, row, diag);
    acc += v * h[(static_cast<int64_t>(b) * m + row) * n + col];
  }

  partial[threadIdx.x] = acc;
  __syncthreads();

  for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    int64_t idx = (((static_cast<int64_t>(b) * (j_end - j_start) + panel_i) * trailing_cols + col_off) * row_tiles + row_tile);
    y_partial[idx] = partial[0];
  }
}

__global__ void reduce_y_compute_z_kernel(
    const float* __restrict__ y_partial,
    const float* __restrict__ tmat,
    float* __restrict__ z,
    int batch,
    int nb,
    int trailing_cols,
    int row_tiles) {
  __shared__ float partial[THREADS];
  __shared__ float y_buf[MAX_NB];

  int col_off = blockIdx.x;
  int b = blockIdx.y;

  for (int panel_i = 0; panel_i < nb; ++panel_i) {
    float acc = 0.0f;
    for (int rt = threadIdx.x; rt < row_tiles; rt += blockDim.x) {
      int64_t idx = (((static_cast<int64_t>(b) * nb + panel_i) * trailing_cols + col_off) * row_tiles + rt);
      acc += y_partial[idx];
    }
    partial[threadIdx.x] = acc;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] += partial[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      y_buf[panel_i] = partial[0];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    for (int i = 0; i < nb; ++i) {
      float acc = 0.0f;
      for (int k = 0; k < nb; ++k) {
        acc += tmat[(static_cast<int64_t>(b) * nb + i) * nb + k] * y_buf[k];
      }
      z[(static_cast<int64_t>(b) * nb + i) * trailing_cols + col_off] = acc;
    }
  }
}

__global__ void update_c_tiled_kernel(
    float* __restrict__ h,
    const float* __restrict__ z,
    int batch,
    int m,
    int n,
    int j_start,
    int j_end,
    int nb,
    int trailing_cols) {
  int col_lane = threadIdx.x;
  int row_lane = threadIdx.y;
  int col_off = blockIdx.x * COL_TILE + col_lane;
  int row = j_start + blockIdx.y * COL_TILE + row_lane;
  int b = blockIdx.z;

  if (b >= batch || row >= m || col_off >= trailing_cols) {
    return;
  }

  float update = 0.0f;
  for (int i = 0; i < nb; ++i) {
    int diag = j_start + i;
    float v = v_value(h, b, m, n, row, diag);
    update += v * z[(static_cast<int64_t>(b) * nb + i) * trailing_cols + col_off];
  }

  int col = j_end + col_off;
  int64_t idx = (static_cast<int64_t>(b) * m + row) * n + col;
  h[idx] -= update;
}

__global__ void apply_panel_wy_fused_tile_kernel(
    float* __restrict__ h,
    const float* __restrict__ tmat,
    int batch,
    int m,
    int n,
    int j_start,
    int j_end,
    int nb,
    int trailing_cols,
    int active_rows) {
  __shared__ float c_tile[FUSED_ROWS * COL_TILE];
  __shared__ float v_tile[FUSED_ROWS * MAX_NB];
  __shared__ float t_tile[MAX_NB * MAX_NB];
  __shared__ float y_tile[MAX_NB * COL_TILE];
  __shared__ float z_tile[MAX_NB * COL_TILE];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int block_threads = blockDim.x * blockDim.y;
  int col_base = blockIdx.x * COL_TILE;
  int b = blockIdx.y;

  for (int idx = tid; idx < active_rows * COL_TILE; idx += block_threads) {
    int r = idx / COL_TILE;
    int c = idx - r * COL_TILE;
    int col_off = col_base + c;
    float value = 0.0f;
    if (b < batch && col_off < trailing_cols) {
      int row = j_start + r;
      int col = j_end + col_off;
      value = h[(static_cast<int64_t>(b) * m + row) * n + col];
    }
    c_tile[idx] = value;
  }

  for (int idx = tid; idx < active_rows * nb; idx += block_threads) {
    int r = idx / nb;
    int i = idx - r * nb;
    int row = j_start + r;
    int diag = j_start + i;
    v_tile[r * MAX_NB + i] = v_value(h, b, m, n, row, diag);
  }

  for (int idx = tid; idx < nb * nb; idx += block_threads) {
    int r = idx / nb;
    int c = idx - r * nb;
    t_tile[r * MAX_NB + c] = tmat[(static_cast<int64_t>(b) * nb + r) * nb + c];
  }

  __syncthreads();

  for (int idx = tid; idx < nb * COL_TILE; idx += block_threads) {
    int i = idx / COL_TILE;
    int c = idx - i * COL_TILE;
    float acc = 0.0f;
    for (int r = 0; r < active_rows; ++r) {
      acc += v_tile[r * MAX_NB + i] * c_tile[r * COL_TILE + c];
    }
    y_tile[i * COL_TILE + c] = acc;
  }

  __syncthreads();

  for (int idx = tid; idx < nb * COL_TILE; idx += block_threads) {
    int i = idx / COL_TILE;
    int c = idx - i * COL_TILE;
    float acc = 0.0f;
    for (int k = 0; k < nb; ++k) {
      acc += t_tile[i * MAX_NB + k] * y_tile[k * COL_TILE + c];
    }
    z_tile[i * COL_TILE + c] = acc;
  }

  __syncthreads();

  for (int idx = tid; idx < active_rows * COL_TILE; idx += block_threads) {
    int r = idx / COL_TILE;
    int c = idx - r * COL_TILE;
    int col_off = col_base + c;
    if (b < batch && col_off < trailing_cols) {
      float update = 0.0f;
      for (int i = 0; i < nb; ++i) {
        update += v_tile[r * MAX_NB + i] * z_tile[i * COL_TILE + c];
      }
      int row = j_start + r;
      int col = j_end + col_off;
      int64_t h_idx = (static_cast<int64_t>(b) * m + row) * n + col;
      h[h_idx] = c_tile[idx] - update;
    }
  }
}

}  // namespace

void apply_panel_wy_tiled_cuda(torch::Tensor h, torch::Tensor tmat, int64_t j_start, int64_t j_end) {
  TORCH_CHECK(h.is_cuda(), "h must be CUDA");
  TORCH_CHECK(tmat.is_cuda(), "tmat must be CUDA");
  TORCH_CHECK(h.dtype() == torch::kFloat32, "h must be float32");
  TORCH_CHECK(tmat.dtype() == torch::kFloat32, "tmat must be float32");
  TORCH_CHECK(h.is_contiguous(), "h must be contiguous");
  TORCH_CHECK(tmat.is_contiguous(), "tmat must be contiguous");

  int batch = static_cast<int>(h.size(0));
  int m = static_cast<int>(h.size(1));
  int n = static_cast<int>(h.size(2));
  int js = static_cast<int>(j_start);
  int je = static_cast<int>(j_end);
  int nb = je - js;
  int trailing_cols = n - je;
  if (trailing_cols <= 0) {
    return;
  }
  TORCH_CHECK(nb > 0 && nb <= MAX_NB, "panel width must be in 1..32");

  int active_rows = m - js;
  if (active_rows <= FUSED_ROWS) {
    dim3 block_fused(COL_TILE, COL_TILE);
    dim3 grid_fused((trailing_cols + COL_TILE - 1) / COL_TILE, batch);
    apply_panel_wy_fused_tile_kernel<<<grid_fused, block_fused>>>(
        h.data_ptr<float>(),
        tmat.data_ptr<float>(),
        batch,
        m,
        n,
        js,
        je,
        nb,
        trailing_cols,
        active_rows);
    return;
  }

  int row_tiles = (active_rows + ROW_TILE - 1) / ROW_TILE;

  auto opts = h.options();
  auto y_partial = torch::empty({batch, nb, trailing_cols, row_tiles}, opts);
  auto z = torch::empty({batch, nb, trailing_cols}, opts);

  dim3 block1(THREADS);
  dim3 grid1(trailing_cols, batch, nb * row_tiles);
  compute_y_partial_kernel<<<grid1, block1>>>(
      h.data_ptr<float>(),
      y_partial.data_ptr<float>(),
      batch,
      m,
      n,
      js,
      je,
      trailing_cols,
      row_tiles);

  dim3 block2(THREADS);
  dim3 grid2(trailing_cols, batch);
  reduce_y_compute_z_kernel<<<grid2, block2>>>(
      y_partial.data_ptr<float>(),
      tmat.data_ptr<float>(),
      z.data_ptr<float>(),
      batch,
      nb,
      trailing_cols,
      row_tiles);

  dim3 block3(COL_TILE, COL_TILE);
  dim3 grid3((trailing_cols + COL_TILE - 1) / COL_TILE, (active_rows + COL_TILE - 1) / COL_TILE, batch);
  update_c_tiled_kernel<<<grid3, block3>>>(
      h.data_ptr<float>(),
      z.data_ptr<float>(),
      batch,
      m,
      n,
      js,
      je,
      nb,
      trailing_cols);
}
"""

    _EXT = load_inline(
        name="submission_cuda_wy_tiled_ext_v1",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _EXT


def apply_panel_wy_tiled_cuda(h: torch.Tensor, tmat: torch.Tensor, j_start: int, j_end: int) -> None:
    """Apply C <- C - V T (V^T C) using raw CUDA tiled kernels."""

    _load_ext().apply_panel_wy_tiled_cuda(h, tmat, int(j_start), int(j_end))


def _runtime_backend(a: torch.Tensor, backend: BackendName) -> str:
    if backend == "auto":
        return "torch_cuda" if a.is_cuda else "torch_cpu"
    if backend == "cutedsl":
        if not a.is_cuda:
            raise ValueError("CuTe DSL backend requires a CUDA tensor")
        if not _CUTE_AVAILABLE:
            raise RuntimeError("CuTe DSL is not importable")
        return "cutedsl_mvp_torch_cuda"
    if backend == "cuda":
        if not a.is_cuda:
            raise ValueError("cuda backend requires a CUDA tensor")
        if not _CUTE_AVAILABLE:
            raise RuntimeError("cuda backend requires the in-file CuTe DSL panel kernel")
        return "cuda_tiled_wy"
    if backend == "torch_cuda" and not a.is_cuda:
        raise ValueError("torch_cuda backend requires a CUDA tensor")
    return backend


def _sign_nonzero(alpha: torch.Tensor) -> torch.Tensor:
    """LAPACK-style sign choice: sign(0) is treated as +1."""

    return torch.where(alpha >= 0, torch.ones_like(alpha), -torch.ones_like(alpha))


def _safe_tau(beta: torch.Tensor, alpha: torch.Tensor, nonzero: torch.Tensor) -> torch.Tensor:
    tau = (beta - alpha) / beta
    return torch.where(nonzero & (beta != 0), tau, torch.zeros_like(alpha))


def _panel_factor_apply_torch(
    h: torch.Tensor,
    tau: torch.Tensor,
    j_start: int,
    j_end: int,
) -> None:
    """
    Parts 2 and 3: factor one panel and apply reflectors inside the panel.

    Mathematical operation per reflector:
        H_j = I - tau_j v_j v_j^T
        A[j:, j+1:j_end] <- H_j A[j:, j+1:j_end]
    """

    batch, m, _ = h.shape
    for j in range(j_start, j_end):
        x = h[:, j:, j].clone()
        alpha = x[:, 0].clone()
        x_tail = x[:, 1:]
        sigma = (x_tail * x_tail).sum(dim=-1)
        x_norm = torch.sqrt(alpha * alpha + sigma)
        beta = -_sign_nonzero(alpha) * x_norm

        nonzero = sigma > 0
        tau_j = _safe_tau(beta, alpha, nonzero)
        tau[:, j] = tau_j

        denom = alpha - beta
        scale = torch.where(
            nonzero & (denom != 0),
            1.0 / denom,
            torch.zeros_like(alpha),
        )

        h[:, j, j] = torch.where(nonzero, beta, alpha)
        h[:, j + 1 :, j] = x_tail * scale.unsqueeze(-1)

        if j < j_end - 1:
            v = torch.ones(batch, m - j, device=h.device, dtype=h.dtype)
            v[:, 1:] = h[:, j + 1 :, j]
            trailing = h[:, j:, j + 1 : j_end]
            v_col = v.unsqueeze(-1)
            w = torch.bmm(v_col.transpose(-1, -2), trailing)
            w = tau_j.view(batch, 1, 1) * w
            h[:, j:, j + 1 : j_end] = trailing - torch.bmm(v_col, w)


def _panel_factor_apply_cutedsl_mvp(
    h: torch.Tensor,
    tau: torch.Tensor,
    j_start: int,
    j_end: int,
) -> None:
    """
    Parts 2 and 3 fused into one real CuTe DSL panel kernel.

    part2_3_factor_apply_panel_cuda performs parallel sigma reduction, writes beta/tau
    and v_tail, then applies each reflector to the remaining panel columns.
    """

    batch, m, n = h.shape
    part2_3_factor_apply_panel_cuda(
        from_dlpack(h),
        from_dlpack(tau),
        batch,
        m,
        n,
        j_start,
        j_end,
    )


def _build_panel_v_torch(h: torch.Tensor, j_start: int, actual_nb: int) -> torch.Tensor:
    """
    Part 4 helper: materialize compact WY V from the Householder vectors in H.

    Mathematical object:
        V = [v_0, v_1, ..., v_{b-1}], with v_i[i] = 1 and v_i[:i] = 0.
    """

    batch, m, _ = h.shape
    v = torch.zeros(batch, m - j_start, actual_nb, device=h.device, dtype=h.dtype)
    for jj in range(actual_nb):
        col = j_start + jj
        v[:, jj, jj] = 1.0
        v[:, jj + 1 :, jj] = h[:, col + 1 :, col]
    return v


def _build_compact_wy_t_torch(
    v: torch.Tensor,
    panel_tau: torch.Tensor,
) -> torch.Tensor:
    """
    Part 4: build reverse-order compact WY T for QR trailing updates.

    The panel factorization applies reflectors to the active matrix as
    H_last ... H_first, so T is lower triangular for V columns stored in
    chronological order.

    Recurrence for P_j = H_j P_{j-1} = I - V_j T_j V_j^T:
        T[j, j] = tau_j
        T[j, :j] = -tau_j * (v_j^T V[:,:j]) @ T[:j, :j]
    """

    batch, _, nb = v.shape
    t = torch.zeros(batch, nb, nb, device=v.device, dtype=v.dtype)
    for jj in range(nb):
        tau_j = panel_tau[:, jj]
        t[:, jj, jj] = tau_j
        if jj > 0:
            z = torch.bmm(v[:, :, jj].unsqueeze(1), v[:, :, :jj]).squeeze(1)
            row = -tau_j.unsqueeze(-1) * z
            t[:, jj, :jj] = torch.bmm(row.unsqueeze(1), t[:, :jj, :jj]).squeeze(1)
    return t


def _apply_trailing_update_wy_torch(
    h: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
    j_start: int,
    j_end: int,
) -> None:
    """
    Part 5 optimized target: apply a compact WY panel update.

    Mathematical operation:
        C <- (I - V T V^T) C = C - V T (V^T C)

    This function is kept as the optimization boundary. The default runtime uses
    the sequential Torch CUDA update below because it matches the compact
    Householder order expected by torch.linalg.householder_product.
    """

    trailing = h[:, j_start:, j_end:]
    if trailing.numel() == 0:
        return
    w = torch.bmm(v.transpose(-1, -2), trailing)
    w = torch.bmm(t, w)
    h[:, j_start:, j_end:] = trailing - torch.bmm(v, w)


def _apply_trailing_update_torch(
    h: torch.Tensor,
    tau: torch.Tensor,
    j_start: int,
    j_end: int,
) -> None:
    """
    Part 5 MVP: apply the panel reflectors sequentially to the trailing matrix.

    Mathematical operation:
        for j in panel:
            C <- (I - tau_j v_j v_j^T) C

    This is less fused than the WY update, but it is the correctness-preserving
    Torch CUDA fallback used before replacing Part 5 with a verified CuTe/WY
    kernel.
    """

    if j_end >= h.shape[-1]:
        return
    batch, m, _ = h.shape
    for j in range(j_start, j_end):
        v = torch.ones(batch, m - j, device=h.device, dtype=h.dtype)
        v[:, 1:] = h[:, j + 1 :, j]
        trailing = h[:, j:, j_end:]
        v_col = v.unsqueeze(-1)
        w = torch.bmm(v_col.transpose(-1, -2), trailing)
        w = tau[:, j].view(batch, 1, 1) * w
        h[:, j:, j_end:] = trailing - torch.bmm(v_col, w)


def householder_qr_blocked(
    A: torch.Tensor,
    nb: int = 32,
    backend: BackendName = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Blocked Householder QR factorization.

    Args:
        A: (batch, m, n) tensor.
        nb: panel width.
        backend: "auto", "torch_cuda", "torch_cpu", "cutedsl", or "cuda".
            The "cutedsl" path uses a fused CuTe DSL panel kernel and a compact
            WY CuTe DSL trailing-update kernel. The "cuda" path reuses the same
            panel kernel and applies Part 5 with raw CUDA tiled WY kernels.

    Returns:
        H: upper triangle is R; strict lower triangle stores Householder vectors.
        tau: scalar factors for each Householder reflector.
    """

    if A.ndim != 3:
        raise ValueError(f"A must have shape (batch, m, n), got {tuple(A.shape)}")
    if A.dtype != torch.float32:
        raise ValueError(f"A must be torch.float32, got {A.dtype}")
    if nb <= 0:
        raise ValueError("nb must be positive")

    runtime = _runtime_backend(A, backend)
    batch, m, n = A.shape
    k = min(m, n)
    h = A.clone()
    tau = torch.zeros(batch, k, device=A.device, dtype=A.dtype)

    for j_start in range(0, k, nb):
        j_end = min(j_start + nb, k)
        actual_nb = j_end - j_start

        if runtime in ("cutedsl_mvp_torch_cuda", "cuda_tiled_wy"):
            _panel_factor_apply_cutedsl_mvp(h, tau, j_start, j_end)
        else:
            _panel_factor_apply_torch(h, tau, j_start, j_end)

        if j_end < n:
            if runtime in ("cutedsl_mvp_torch_cuda", "cuda_tiled_wy"):
                v = _build_panel_v_torch(h, j_start, actual_nb)
                t = _build_compact_wy_t_torch(v, tau[:, j_start:j_end])
                if runtime == "cuda_tiled_wy":
                    apply_panel_wy_tiled_cuda(h, t, j_start, j_end)
                else:
                    part5_apply_panel_wy_gemv_cuda(
                        from_dlpack(h),
                        from_dlpack(tau),
                        from_dlpack(t),
                        batch,
                        m,
                        n,
                        j_start,
                        j_end,
                    )
            else:
                _apply_trailing_update_torch(h, tau, j_start, j_end)

    return h, tau


def custom_kernel(data: input_t) -> output_t:
    """Official harness entry point.

    Prefer the real CuTe DSL WY backend on CUDA. Fall back to the
    Torch implementation when CuTe DSL is unavailable or the input is on CPU.
    """

    backend: BackendName = "cuda" if data.is_cuda and _CUTE_AVAILABLE else "auto"
    return householder_qr_blocked(data, nb=16, backend=backend)

