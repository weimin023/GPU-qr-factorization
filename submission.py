#!POPCORN leaderboard qr_v2
#!POPCORN gpu B200

"""Single-file QR submission.

Entry point: custom_kernel(data) -> (H, tau).
The active path uses an inline CUDA C small-square kernel and CuTe DSL kernels
for the blocked panel/T/WY stages. torch.geqrf is kept only as ref_kernel.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_SMALL_SQUARE_QR_EXT = None

_SMALL_SQUARE_QR_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>

extern "C" cudaError_t small_square_qr_cuda(
    const float* a,
    float* h,
    float* tau,
    int batch_count,
    int n);

std::vector<torch::Tensor> small_square_qr(torch::Tensor a) {
  TORCH_CHECK(a.is_cuda(), "small_square_qr expects a CUDA tensor");
  TORCH_CHECK(a.dtype() == torch::kFloat32, "small_square_qr expects float32 input");
  TORCH_CHECK(a.dim() == 3, "small_square_qr expects shape (batch, n, n)");
  TORCH_CHECK(a.size(1) == a.size(2), "small_square_qr expects square matrices");
  TORCH_CHECK(a.size(1) > 0 && a.size(1) <= 64, "small_square_qr supports 1 <= n <= 64");
  TORCH_CHECK(a.is_contiguous(), "small_square_qr expects contiguous row-major input");

  const int batch_count = static_cast<int>(a.size(0));
  const int n = static_cast<int>(a.size(1));
  auto h = torch::empty_like(a);
  auto tau = torch::empty({batch_count, n}, a.options());

  C10_CUDA_CHECK(small_square_qr_cuda(
      a.data_ptr<float>(),
      h.data_ptr<float>(),
      tau.data_ptr<float>(),
      batch_count,
      n));

  return {h, tau};
}
"""

_SMALL_SQUARE_QR_CUDA_SOURCE = r"""
#include <cuda_runtime.h>
#include <stdint.h>

namespace ssqr {

constexpr int kMaxN = 64;
constexpr int kThreads = 128;

struct LaunchConfig {
  dim3 grid;
  dim3 block;
  size_t shared_bytes;
};

__device__ __forceinline__ float block_sum(float value, float* scratch) {
  const int tid = threadIdx.x;
  scratch[tid] = value;
  __syncthreads();

  if (tid < 64) scratch[tid] += scratch[tid + 64];
  __syncthreads();
  if (tid < 32) scratch[tid] += scratch[tid + 32];
  __syncthreads();
  if (tid < 16) scratch[tid] += scratch[tid + 16];
  __syncthreads();
  if (tid < 8) scratch[tid] += scratch[tid + 8];
  __syncthreads();
  if (tid < 4) scratch[tid] += scratch[tid + 4];
  __syncthreads();
  if (tid < 2) scratch[tid] += scratch[tid + 2];
  __syncthreads();
  if (tid == 0) scratch[0] += scratch[1];
  __syncthreads();

  return scratch[0];
}

__global__ void small_square_qr_kernel(
    const float* __restrict__ a,
    float* __restrict__ h,
    float* __restrict__ tau,
    int batch_count,
    int n) {
  extern __shared__ float smem[];
  float* s_a = smem;
  float* s_tau = s_a + kMaxN * kMaxN;
  float* s_reduce = s_tau + kMaxN;
  float* s_scalars = s_reduce + kThreads;

  const int tid = threadIdx.x;
  const int batch = blockIdx.x;
  if (batch >= batch_count) return;

  const int matrix_elems = n * n;
  const int64_t matrix_base = static_cast<int64_t>(batch) * matrix_elems;

  for (int idx = tid; idx < matrix_elems; idx += blockDim.x) {
    s_a[idx] = a[matrix_base + idx];
  }
  for (int idx = tid; idx < n; idx += blockDim.x) {
    s_tau[idx] = 0.0f;
  }
  __syncthreads();

  for (int j = 0; j < n; ++j) {
    const int diag = j * n + j;
    const float alpha = s_a[diag];

    float local_sigma = 0.0f;
    for (int row = j + 1 + tid; row < n; row += blockDim.x) {
      const float x = s_a[row * n + j];
      local_sigma += x * x;
    }
    const float sigma = block_sum(local_sigma, s_reduce);

    if (tid == 0) {
      const float x_norm = sqrtf(alpha * alpha + sigma);
      const float beta = (alpha >= 0.0f) ? -x_norm : x_norm;

      float tau_j = 0.0f;
      float scale = 0.0f;
      if (sigma > 0.0f) {
        tau_j = (beta - alpha) / beta;
        scale = 1.0f / (alpha - beta);
        s_a[diag] = beta;
      } else {
        s_a[diag] = alpha;
      }

      s_tau[j] = tau_j;
      s_scalars[0] = scale;
      s_scalars[1] = tau_j;
    }
    __syncthreads();

    const float scale = s_scalars[0];
    const float tau_j = s_scalars[1];

    for (int row = j + 1 + tid; row < n; row += blockDim.x) {
      s_a[row * n + j] *= scale;
    }
    __syncthreads();

    for (int col = j + 1; col < n; ++col) {
      float local_dot = 0.0f;
      for (int row = j + tid; row < n; row += blockDim.x) {
        const float v = (row == j) ? 1.0f : s_a[row * n + j];
        local_dot += v * s_a[row * n + col];
      }
      const float dot = block_sum(local_dot, s_reduce);

      if (tid == 0) s_scalars[2] = tau_j * dot;
      __syncthreads();

      const float w = s_scalars[2];
      for (int row = j + tid; row < n; row += blockDim.x) {
        const float v = (row == j) ? 1.0f : s_a[row * n + j];
        s_a[row * n + col] -= v * w;
      }
      __syncthreads();
    }
  }

  for (int idx = tid; idx < matrix_elems; idx += blockDim.x) {
    h[matrix_base + idx] = s_a[idx];
  }
  for (int idx = tid; idx < n; idx += blockDim.x) {
    tau[static_cast<int64_t>(batch) * n + idx] = s_tau[idx];
  }
}

inline LaunchConfig make_launch_config(int batch_count) {
  LaunchConfig cfg;
  cfg.grid = dim3(batch_count, 1, 1);
  cfg.block = dim3(kThreads, 1, 1);
  cfg.shared_bytes = sizeof(float) * (kMaxN * kMaxN + kMaxN + kThreads + 4);
  return cfg;
}

}  // namespace ssqr

extern "C" cudaError_t small_square_qr_cuda(
    const float* a,
    float* h,
    float* tau,
    int batch_count,
    int n) {
  if (a == nullptr || h == nullptr || tau == nullptr) {
    return cudaErrorInvalidDevicePointer;
  }
  if (batch_count < 0 || n <= 0 || n > ssqr::kMaxN) {
    return cudaErrorInvalidValue;
  }
  if (batch_count == 0) {
    return cudaSuccess;
  }

  const auto cfg = ssqr::make_launch_config(batch_count);
  ssqr::small_square_qr_kernel<<<cfg.grid, cfg.block, cfg.shared_bytes>>>(
      a, h, tau, batch_count, n);
  return cudaGetLastError();
}

"""


def _small_square_arch_tag() -> tuple[str, str]:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f"{major}.{minor}", f"sm{major}{minor}"
    return os.environ.get("TORCH_CUDA_ARCH_LIST", "12.0"), "nogpu"


def _load_small_square_qr_ext():
    global _SMALL_SQUARE_QR_EXT
    if _SMALL_SQUARE_QR_EXT is not None:
        return _SMALL_SQUARE_QR_EXT

    from torch.utils.cpp_extension import load_inline

    arch_list, arch_tag = _small_square_arch_tag()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch_list)
    _SMALL_SQUARE_QR_EXT = load_inline(
        name=f"small_square_qr_inline_ext_{arch_tag}",
        cpp_sources=[_SMALL_SQUARE_QR_CPP_SOURCE],
        cuda_sources=[_SMALL_SQUARE_QR_CUDA_SOURCE],
        functions=["small_square_qr"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-DSMALL_SQUARE_QR_TORCH_EXTENSION"],
        verbose=bool(int(os.environ.get("SMALL_SQUARE_QR_VERBOSE_BUILD", "0"))),
    )
    return _SMALL_SQUARE_QR_EXT


_PANEL_QR_EXT = None

_PANEL_QR_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#ifndef ROW_SPLIT_ROWS_PER_TILE
#define ROW_SPLIT_ROWS_PER_TILE 64
#endif

#ifndef ROW_SPLIT_MIN_M
#define ROW_SPLIT_MIN_M 4096
#endif

#ifndef PANEL_COOP_PANEL
#define PANEL_COOP_PANEL 0
#endif

#ifndef PANEL_COOP_ROW_TILE
#define PANEL_COOP_ROW_TILE 128
#endif

extern "C" cudaError_t panel_factor_apply_cuda(
    float* h,
    float* tau,
    float* sigma_ws,
    float* dot_ws,
    int batch_count,
    int m,
    int n,
    int j_start,
    int j_end,
    int max_row_tiles);

extern "C" cudaError_t build_compact_wy_t_raw_cuda(
    const float* h,
    const float* tau,
    float* tmat,
    float* dot_ws,
    int batch_count,
    int m,
    int n,
    int tau_stride,
    int t_ld,
    int j_start,
    int j_end);

extern "C" cudaError_t build_compact_wy_t_finish_cuda(
    const float* tau,
    float* tmat,
    const float* dot_ws,
    int batch_count,
    int tau_stride,
    int t_ld,
    int j_start,
    int panel_cols);

extern "C" cudaError_t apply_panel_wy_fused_update_raw_cuda(
    float* h,
    const float* tmat,
    float* y_partial,
    int batch_count,
    int m,
    int n,
    int t_ld,
    int y_panel_cap,
    int y_trailing_cap,
    int y_row_tile_cap,
    int j_start,
    int j_end,
    int row_tiles);

void panel_factor_apply(torch::Tensor h, torch::Tensor tau, int64_t j_start, int64_t j_end) {
  TORCH_CHECK(h.is_cuda(), "panel_factor_apply expects CUDA H");
  TORCH_CHECK(tau.is_cuda(), "panel_factor_apply expects CUDA tau");
  TORCH_CHECK(h.dtype() == torch::kFloat32, "panel_factor_apply expects float32 H");
  TORCH_CHECK(tau.dtype() == torch::kFloat32, "panel_factor_apply expects float32 tau");
  TORCH_CHECK(h.dim() == 3, "panel_factor_apply expects H shape (batch, m, n)");
  TORCH_CHECK(tau.dim() == 2, "panel_factor_apply expects tau shape (batch, k)");
  TORCH_CHECK(h.is_contiguous(), "panel_factor_apply expects contiguous H");
  TORCH_CHECK(tau.is_contiguous(), "panel_factor_apply expects contiguous tau");

  const int batch_count = static_cast<int>(h.size(0));
  const int m = static_cast<int>(h.size(1));
  const int n = static_cast<int>(h.size(2));
  const int js = static_cast<int>(j_start);
  const int je = static_cast<int>(j_end);
  TORCH_CHECK(0 <= js && js <= je && je <= n, "invalid panel bounds");
  TORCH_CHECK(je - js <= 128, "panel_factor_apply supports panel width <= 128");

  constexpr int workspace_rows_per_tile = ROW_SPLIT_ROWS_PER_TILE < 256 ? ROW_SPLIT_ROWS_PER_TILE : 256;
  const int max_row_tiles = (m + workspace_rows_per_tile - 1) / workspace_rows_per_tile;
  auto sigma_ws = torch::empty({batch_count, je - js, max_row_tiles}, h.options());
  auto dot_ws = torch::empty({batch_count, je - js, je - js, max_row_tiles}, h.options());

  C10_CUDA_CHECK(panel_factor_apply_cuda(
      h.data_ptr<float>(),
      tau.data_ptr<float>(),
      sigma_ws.data_ptr<float>(),
      dot_ws.data_ptr<float>(),
      batch_count,
      m,
      n,
      js,
      je,
      max_row_tiles));
}

void build_compact_wy_t_raw(
    torch::Tensor h,
    torch::Tensor tau,
    torch::Tensor tmat,
    int64_t j_start,
    int64_t j_end) {
  TORCH_CHECK(h.is_cuda(), "build_compact_wy_t_raw expects CUDA H");
  TORCH_CHECK(tau.is_cuda(), "build_compact_wy_t_raw expects CUDA tau");
  TORCH_CHECK(tmat.is_cuda(), "build_compact_wy_t_raw expects CUDA T");
  TORCH_CHECK(h.dtype() == torch::kFloat32, "build_compact_wy_t_raw expects float32 H");
  TORCH_CHECK(tau.dtype() == torch::kFloat32, "build_compact_wy_t_raw expects float32 tau");
  TORCH_CHECK(tmat.dtype() == torch::kFloat32, "build_compact_wy_t_raw expects float32 T");
  TORCH_CHECK(h.dim() == 3, "build_compact_wy_t_raw expects H shape (batch, m, n)");
  TORCH_CHECK(tau.dim() == 2, "build_compact_wy_t_raw expects tau shape (batch, k)");
  TORCH_CHECK(tmat.dim() == 3, "build_compact_wy_t_raw expects T shape (batch, nb, nb)");
  TORCH_CHECK(h.is_contiguous(), "build_compact_wy_t_raw expects contiguous H");
  TORCH_CHECK(tau.is_contiguous(), "build_compact_wy_t_raw expects contiguous tau");
  TORCH_CHECK(tmat.is_contiguous(), "build_compact_wy_t_raw expects contiguous T");

  const int batch_count = static_cast<int>(h.size(0));
  const int m = static_cast<int>(h.size(1));
  const int n = static_cast<int>(h.size(2));
  const int tau_stride = static_cast<int>(tau.size(1));
  const int t_ld = static_cast<int>(tmat.size(1));
  const int js = static_cast<int>(j_start);
  const int je = static_cast<int>(j_end);
  TORCH_CHECK(tmat.size(0) == h.size(0), "T batch must match H batch");
  TORCH_CHECK(tmat.size(1) == tmat.size(2), "T must be square per batch");
  TORCH_CHECK(0 <= js && js <= je && je <= n, "invalid panel bounds");
  TORCH_CHECK(je - js <= t_ld, "T workspace too small for panel");
  TORCH_CHECK(je <= tau_stride, "tau is too small for panel");

  auto dot_ws = torch::empty({batch_count, t_ld, t_ld}, h.options());

  C10_CUDA_CHECK(build_compact_wy_t_raw_cuda(
      h.data_ptr<float>(),
      tau.data_ptr<float>(),
      tmat.data_ptr<float>(),
      dot_ws.data_ptr<float>(),
      batch_count,
      m,
      n,
      tau_stride,
      t_ld,
      js,
      je));
}

void build_compact_wy_t_finish(
    torch::Tensor tau,
    torch::Tensor tmat,
    torch::Tensor dot_ws,
    int64_t j_start,
    int64_t j_end) {
  TORCH_CHECK(tau.is_cuda() && tmat.is_cuda() && dot_ws.is_cuda(),
              "build_compact_wy_t_finish expects CUDA tensors");
  TORCH_CHECK(tau.dtype() == torch::kFloat32 && tmat.dtype() == torch::kFloat32 &&
              dot_ws.dtype() == torch::kFloat32,
              "build_compact_wy_t_finish expects float32 tensors");
  TORCH_CHECK(tau.is_contiguous() && tmat.is_contiguous() && dot_ws.is_contiguous(),
              "build_compact_wy_t_finish expects contiguous tensors");
  TORCH_CHECK(tau.dim() == 2 && tmat.dim() == 3 && dot_ws.dim() == 3,
              "invalid build_compact_wy_t_finish tensor ranks");

  const int batch_count = static_cast<int>(tau.size(0));
  const int tau_stride = static_cast<int>(tau.size(1));
  const int t_ld = static_cast<int>(tmat.size(1));
  const int js = static_cast<int>(j_start);
  const int je = static_cast<int>(j_end);
  TORCH_CHECK(tmat.size(0) == batch_count && dot_ws.size(0) == batch_count,
              "batch dimensions must match");
  TORCH_CHECK(tmat.size(1) == tmat.size(2) && dot_ws.size(1) == t_ld &&
              dot_ws.size(2) == t_ld, "T and dot workspace shapes must match");
  TORCH_CHECK(0 <= js && js <= je && je <= tau_stride && je - js <= t_ld,
              "invalid panel bounds");

  C10_CUDA_CHECK(build_compact_wy_t_finish_cuda(
      tau.data_ptr<float>(), tmat.data_ptr<float>(), dot_ws.data_ptr<float>(),
      batch_count, tau_stride, t_ld, js, je - js));
}

void apply_panel_wy_fused_update_raw(
    torch::Tensor h,
    torch::Tensor tmat,
    torch::Tensor y_partial,
    int64_t j_start,
    int64_t j_end,
    int64_t row_tiles) {
  TORCH_CHECK(h.is_cuda(), "apply_panel_wy_fused_update_raw expects CUDA H");
  TORCH_CHECK(tmat.is_cuda(), "apply_panel_wy_fused_update_raw expects CUDA T");
  TORCH_CHECK(y_partial.is_cuda(), "apply_panel_wy_fused_update_raw expects CUDA Y workspace");
  TORCH_CHECK(h.dtype() == torch::kFloat32, "apply_panel_wy_fused_update_raw expects float32 H");
  TORCH_CHECK(tmat.dtype() == torch::kFloat32, "apply_panel_wy_fused_update_raw expects float32 T");
  TORCH_CHECK(y_partial.dtype() == torch::kFloat32, "apply_panel_wy_fused_update_raw expects float32 Y workspace");
  TORCH_CHECK(h.dim() == 3, "apply_panel_wy_fused_update_raw expects H shape (batch, m, n)");
  TORCH_CHECK(tmat.dim() == 3, "apply_panel_wy_fused_update_raw expects T shape (batch, nb, nb)");
  TORCH_CHECK(y_partial.dim() == 4, "apply_panel_wy_fused_update_raw expects Y shape (batch, nb, trailing, row_tiles)");
  TORCH_CHECK(h.is_contiguous(), "apply_panel_wy_fused_update_raw expects contiguous H");
  TORCH_CHECK(tmat.is_contiguous(), "apply_panel_wy_fused_update_raw expects contiguous T");
  TORCH_CHECK(y_partial.is_contiguous(), "apply_panel_wy_fused_update_raw expects contiguous Y workspace");

  const int batch_count = static_cast<int>(h.size(0));
  const int m = static_cast<int>(h.size(1));
  const int n = static_cast<int>(h.size(2));
  const int t_ld = static_cast<int>(tmat.size(1));
  const int y_panel_cap = static_cast<int>(y_partial.size(1));
  const int y_trailing_cap = static_cast<int>(y_partial.size(2));
  const int y_row_tile_cap = static_cast<int>(y_partial.size(3));
  const int js = static_cast<int>(j_start);
  const int je = static_cast<int>(j_end);
  const int rt = static_cast<int>(row_tiles);
  TORCH_CHECK(tmat.size(0) == h.size(0), "T batch must match H batch");
  TORCH_CHECK(y_partial.size(0) == h.size(0), "Y batch must match H batch");
  TORCH_CHECK(0 <= js && js <= je && je <= n, "invalid panel bounds");
  TORCH_CHECK(je - js <= t_ld && je - js <= y_panel_cap, "workspace too small for panel");
  TORCH_CHECK(n - je <= y_trailing_cap, "Y workspace too small for trailing columns");
  TORCH_CHECK(rt <= y_row_tile_cap, "Y workspace too small for row tiles");

  C10_CUDA_CHECK(apply_panel_wy_fused_update_raw_cuda(
      h.data_ptr<float>(),
      tmat.data_ptr<float>(),
      y_partial.data_ptr<float>(),
      batch_count,
      m,
      n,
      t_ld,
      y_panel_cap,
      y_trailing_cap,
      y_row_tile_cap,
      js,
      je,
      rt));
}
"""

_PANEL_QR_CUDA_SOURCE = r"""
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>

namespace panel_qr {
namespace cg = cooperative_groups;

constexpr int kThreads = 256;
constexpr int kRowsPerTile = 256;
constexpr int kWarps = kThreads / 32;

#ifndef ROW_SPLIT_MULTI8_MIN_TARGETS
#define ROW_SPLIT_MULTI8_MIN_TARGETS 8
#endif

#ifndef ROW_SPLIT_TARGET_TILE
#define ROW_SPLIT_TARGET_TILE 8
#endif

#ifndef ROW_SPLIT_ROWS_PER_TILE
#define ROW_SPLIT_ROWS_PER_TILE 64
#endif

__device__ __forceinline__ float block_sum(float value) {
  __shared__ float scratch[kWarps];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  if (lane == 0) scratch[warp] = value;
  __syncthreads();

  value = (warp == 0 && lane < kWarps) ? scratch[lane] : 0.0f;
  if (warp == 0) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

__global__ void factor_single_tile_kernel(
    float* __restrict__ h,
    float* __restrict__ tau,
    int batch_count,
    int m,
    int n,
    int j) {
  const int batch = blockIdx.x;
  const int tid = threadIdx.x;
  if (batch >= batch_count) return;

  const int64_t base = static_cast<int64_t>(batch) * m * n;
  float sigma = 0.0f;
  for (int row = j + 1 + tid; row < m; row += blockDim.x) {
    const float x = h[base + static_cast<int64_t>(row) * n + j];
    sigma += x * x;
  }
  sigma = block_sum(sigma);

  __shared__ float scale_s;
  if (tid == 0) {
    const float alpha = h[base + static_cast<int64_t>(j) * n + j];
    const float x_norm = sqrtf(alpha * alpha + sigma);
    const float beta = (alpha >= 0.0f) ? -x_norm : x_norm;
    float tau_j = 0.0f;
    float scale = 0.0f;
    if (sigma > 0.0f) {
      tau_j = (beta - alpha) / beta;
      scale = 1.0f / (alpha - beta);
      h[base + static_cast<int64_t>(j) * n + j] = beta;
    } else {
      h[base + static_cast<int64_t>(j) * n + j] = alpha;
    }
    tau[static_cast<int64_t>(batch) * n + j] = tau_j;
    scale_s = scale;
  }
  __syncthreads();

  const float scale = scale_s;
  for (int row = j + 1 + tid; row < m; row += blockDim.x) {
    h[base + static_cast<int64_t>(row) * n + j] *= scale;
  }
}

__global__ void sigma_partial_kernel(
    float* __restrict__ h,
    float* __restrict__ sigma_ws,
    int batch_count,
    int m,
    int n,
    int j_start,
    int panel_cols,
    int j,
    int row_tiles,
    int max_row_tiles) {
  const int tile = blockIdx.x;
  const int batch = blockIdx.y;
  const int tid = threadIdx.x;
  if (batch >= batch_count) return;

  float local = 0.0f;
  const int row0 = j + 1 + tile * kRowsPerTile;
  const int row1 = min(row0 + kRowsPerTile, m);
  const int64_t base = static_cast<int64_t>(batch) * m * n;
  for (int row = row0 + tid; row < row1; row += blockDim.x) {
    const float x = h[base + static_cast<int64_t>(row) * n + j];
    local += x * x;
  }
  const float sum = block_sum(local);
  if (tid == 0) {
    const int pj = j - j_start;
    sigma_ws[(static_cast<int64_t>(batch) * panel_cols + pj) * max_row_tiles + tile] = sum;
  }
}

__global__ void finalize_scale_kernel(
    float* __restrict__ h,
    float* __restrict__ tau,
    float* __restrict__ sigma_ws,
    int batch_count,
    int m,
    int n,
    int j_start,
    int panel_cols,
    int j,
    int row_tiles,
    int max_row_tiles) {
  const int batch = blockIdx.x;
  const int tid = threadIdx.x;
  if (batch >= batch_count) return;

  const int64_t base = static_cast<int64_t>(batch) * m * n;
  const int pj = j - j_start;
  float sigma = 0.0f;
  for (int tile = tid; tile < row_tiles; tile += blockDim.x) {
    sigma += sigma_ws[(static_cast<int64_t>(batch) * panel_cols + pj) * max_row_tiles + tile];
  }
  sigma = block_sum(sigma);

  __shared__ float scalars[2];
  if (tid == 0) {
    const float alpha = h[base + static_cast<int64_t>(j) * n + j];
    const float x_norm = sqrtf(alpha * alpha + sigma);
    const float beta = (alpha >= 0.0f) ? -x_norm : x_norm;
    float tau_j = 0.0f;
    float scale = 0.0f;
    if (sigma > 0.0f) {
      tau_j = (beta - alpha) / beta;
      scale = 1.0f / (alpha - beta);
      h[base + static_cast<int64_t>(j) * n + j] = beta;
    } else {
      h[base + static_cast<int64_t>(j) * n + j] = alpha;
    }
    tau[static_cast<int64_t>(batch) * n + j] = tau_j;
    scalars[0] = scale;
    scalars[1] = tau_j;
  }
  __syncthreads();

  const float scale = scalars[0];
  for (int row = j + 1 + tid; row < m; row += blockDim.x) {
    h[base + static_cast<int64_t>(row) * n + j] *= scale;
  }
}

__global__ void dot_partial_kernel(
    float* __restrict__ h,
    float* __restrict__ dot_ws,
    int batch_count,
    int m,
    int n,
    int j_start,
    int panel_cols,
    int j,
    int target_count,
    int row_tiles,
    int max_row_tiles) {
  const int tile = blockIdx.x;
  const int target_off = blockIdx.y;
  const int batch = blockIdx.z;
  const int tid = threadIdx.x;
  if (batch >= batch_count || target_off >= target_count) return;

  const int target = j + 1 + target_off;
  const int target_panel = target - j_start;
  const int row0 = j + tile * ROW_SPLIT_ROWS_PER_TILE;
  const int row1 = min(row0 + ROW_SPLIT_ROWS_PER_TILE, m);
  const int64_t base = static_cast<int64_t>(batch) * m * n;

  float local = 0.0f;
  for (int row = row0 + tid; row < row1; row += blockDim.x) {
    const float v = (row == j) ? 1.0f : h[base + static_cast<int64_t>(row) * n + j];
    local += v * h[base + static_cast<int64_t>(row) * n + target];
  }
  const float sum = block_sum(local);
  if (tid == 0) {
    const int pj = j - j_start;
    dot_ws[((static_cast<int64_t>(batch) * panel_cols + pj) * panel_cols + target_panel) * max_row_tiles + tile] = sum;
  }
}

__global__ void update_target_kernel(
    float* __restrict__ h,
    float* __restrict__ tau,
    float* __restrict__ dot_ws,
    int batch_count,
    int m,
    int n,
    int j_start,
    int panel_cols,
    int j,
    int target_count,
    int row_tiles,
    int max_row_tiles) {
  const int target_off = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int batch = blockIdx.z;
  const int tid = threadIdx.x;
  if (batch >= batch_count || target_off >= target_count || row_tile >= row_tiles) return;

  const int target = j + 1 + target_off;
  const int pj = j - j_start;
  const int target_panel = target - j_start;
  const int64_t base = static_cast<int64_t>(batch) * m * n;

  float local = 0.0f;
  for (int tile = tid; tile < row_tiles; tile += blockDim.x) {
    local += dot_ws[((static_cast<int64_t>(batch) * panel_cols + pj) * panel_cols + target_panel) * max_row_tiles + tile];
  }
  const float dot = block_sum(local);

  __shared__ float w_s;
  if (tid == 0) {
    const float tau_j = tau[static_cast<int64_t>(batch) * n + j];
    w_s = tau_j * dot;
  }
  __syncthreads();

  const float w = w_s;
  const int row0 = j + row_tile * ROW_SPLIT_ROWS_PER_TILE;
  const int row1 = min(row0 + ROW_SPLIT_ROWS_PER_TILE, m);
  for (int row = row0 + tid; row < row1; row += blockDim.x) {
    const float v = (row == j) ? 1.0f : h[base + static_cast<int64_t>(row) * n + j];
    h[base + static_cast<int64_t>(row) * n + target] -= v * w;
  }
}

__global__ void dot_partial_multi16_kernel(
    float* __restrict__ h,
    float* __restrict__ dot_ws,
    int batch_count,
    int m,
    int n,
    int j_start,
    int panel_cols,
    int j,
    int target_count,
    int row_tiles,
    int max_row_tiles) {
  constexpr int kTargetTile = 16;
  constexpr int kRowLanes = kThreads / kTargetTile;
  __shared__ float partial[kRowLanes][kTargetTile];

  const int tile = blockIdx.x;
  const int col_lane = threadIdx.x & (kTargetTile - 1);
  const int row_lane = threadIdx.x / kTargetTile;
  const int target_off = blockIdx.y * kTargetTile + col_lane;
  const int batch = blockIdx.z;
  const bool valid = batch < batch_count && target_off < target_count;
  const int target = j + 1 + target_off;
  const int row0 = j + tile * ROW_SPLIT_ROWS_PER_TILE;
  const int row1 = min(row0 + ROW_SPLIT_ROWS_PER_TILE, m);
  const int64_t base = static_cast<int64_t>(batch) * m * n;

  float local = 0.0f;
  if (valid) {
    for (int row = row0 + row_lane; row < row1; row += kRowLanes) {
      const float v =
          (row == j) ? 1.0f : h[base + static_cast<int64_t>(row) * n + j];
      local += v * h[base + static_cast<int64_t>(row) * n + target];
    }
  }
  partial[row_lane][col_lane] = local;
  __syncthreads();

  if (row_lane == 0 && valid) {
    float sum = 0.0f;
    #pragma unroll
    for (int lane = 0; lane < kRowLanes; ++lane) {
      sum += partial[lane][col_lane];
    }
    const int pj = j - j_start;
    const int target_panel = target - j_start;
    dot_ws[((static_cast<int64_t>(batch) * panel_cols + pj) * panel_cols +
            target_panel) * max_row_tiles + tile] = sum;
  }
}

__global__ void update_target_multi16_kernel(
    float* __restrict__ h,
    float* __restrict__ tau,
    float* __restrict__ dot_ws,
    int batch_count,
    int m,
    int n,
    int j_start,
    int panel_cols,
    int j,
    int target_count,
    int row_tiles,
    int max_row_tiles) {
  constexpr int kTargetTile = 16;
  constexpr int kRowLanes = kThreads / kTargetTile;
  __shared__ float weights[kTargetTile];

  const int col_lane = threadIdx.x & (kTargetTile - 1);
  const int row_lane = threadIdx.x / kTargetTile;
  const int target_off = blockIdx.x * kTargetTile + col_lane;
  const int row_tile = blockIdx.y;
  const int batch = blockIdx.z;
  const bool valid = batch < batch_count && target_off < target_count;
  const int target = j + 1 + target_off;
  const int pj = j - j_start;
  const int target_panel = target - j_start;
  const int64_t base = static_cast<int64_t>(batch) * m * n;

  if (row_lane == 0) {
    float dot = 0.0f;
    if (valid) {
      for (int tile = 0; tile < row_tiles; ++tile) {
        dot += dot_ws[((static_cast<int64_t>(batch) * panel_cols + pj) *
                       panel_cols + target_panel) * max_row_tiles + tile];
      }
    }
    weights[col_lane] =
        valid ? tau[static_cast<int64_t>(batch) * n + j] * dot : 0.0f;
  }
  __syncthreads();

  if (valid) {
    const float w = weights[col_lane];
    const int row0 = j + row_tile * ROW_SPLIT_ROWS_PER_TILE;
    const int row1 = min(row0 + ROW_SPLIT_ROWS_PER_TILE, m);
    for (int row = row0 + row_lane; row < row1; row += kRowLanes) {
      const float v =
          (row == j) ? 1.0f : h[base + static_cast<int64_t>(row) * n + j];
      h[base + static_cast<int64_t>(row) * n + target] -= v * w;
    }
  }
}

__global__ void dot_partial_multi8_kernel(
    float* __restrict__ h,
    float* __restrict__ dot_ws,
    int batch_count,
    int m,
    int n,
    int j_start,
    int panel_cols,
    int j,
    int target_count,
    int row_tiles,
    int max_row_tiles) {
  constexpr int kTargetTile = 8;
  constexpr int kRowLanes = kThreads / kTargetTile;
  __shared__ float partial[kRowLanes][kTargetTile];

  const int tile = blockIdx.x;
  const int col_lane = threadIdx.x & (kTargetTile - 1);
  const int row_lane = threadIdx.x / kTargetTile;
  const int target_off = blockIdx.y * kTargetTile + col_lane;
  const int batch = blockIdx.z;
  const bool valid = batch < batch_count && target_off < target_count;
  const int target = j + 1 + target_off;
  const int row0 = j + tile * ROW_SPLIT_ROWS_PER_TILE;
  const int row1 = min(row0 + ROW_SPLIT_ROWS_PER_TILE, m);
  const int64_t base = static_cast<int64_t>(batch) * m * n;

  float local = 0.0f;
  if (valid) {
    for (int row = row0 + row_lane; row < row1; row += kRowLanes) {
      const float v =
          (row == j) ? 1.0f : h[base + static_cast<int64_t>(row) * n + j];
      local += v * h[base + static_cast<int64_t>(row) * n + target];
    }
  }
  partial[row_lane][col_lane] = local;
  __syncthreads();

  if (row_lane == 0 && valid) {
    float sum = 0.0f;
    for (int lane = 0; lane < kRowLanes; ++lane) {
      sum += partial[lane][col_lane];
    }
    const int pj = j - j_start;
    const int target_panel = target - j_start;
    dot_ws[((static_cast<int64_t>(batch) * panel_cols + pj) * panel_cols +
            target_panel) * max_row_tiles + tile] = sum;
  }
}

__global__ void update_target_multi8_kernel(
    float* __restrict__ h,
    float* __restrict__ tau,
    float* __restrict__ dot_ws,
    int batch_count,
    int m,
    int n,
    int j_start,
    int panel_cols,
    int j,
    int target_count,
    int row_tiles,
    int max_row_tiles) {
  constexpr int kTargetTile = 8;
  constexpr int kRowLanes = kThreads / kTargetTile;
  __shared__ float weights[kTargetTile];

  const int col_lane = threadIdx.x & (kTargetTile - 1);
  const int row_lane = threadIdx.x / kTargetTile;
  const int target_off = blockIdx.x * kTargetTile + col_lane;
  const int row_tile = blockIdx.y;
  const int batch = blockIdx.z;
  const bool valid = batch < batch_count && target_off < target_count;
  const int target = j + 1 + target_off;
  const int pj = j - j_start;
  const int target_panel = target - j_start;
  const int64_t base = static_cast<int64_t>(batch) * m * n;

  if (row_lane == 0) {
    float dot = 0.0f;
    if (valid) {
      for (int tile = 0; tile < row_tiles; ++tile) {
        dot += dot_ws[((static_cast<int64_t>(batch) * panel_cols + pj) *
                       panel_cols + target_panel) * max_row_tiles + tile];
      }
    }
    weights[col_lane] =
        valid ? tau[static_cast<int64_t>(batch) * n + j] * dot : 0.0f;
  }
  __syncthreads();

  if (valid) {
    const float w = weights[col_lane];
    const int row0 = j + row_tile * ROW_SPLIT_ROWS_PER_TILE;
    const int row1 = min(row0 + ROW_SPLIT_ROWS_PER_TILE, m);
    for (int row = row0 + row_lane; row < row1; row += kRowLanes) {
      const float v =
          (row == j) ? 1.0f : h[base + static_cast<int64_t>(row) * n + j];
      h[base + static_cast<int64_t>(row) * n + target] -= v * w;
    }
  }
}


__global__ void dot_partial_multi4_kernel(
    float* __restrict__ h,
    float* __restrict__ dot_ws,
    int batch_count,
    int m,
    int n,
    int j_start,
    int panel_cols,
    int j,
    int target_count,
    int row_tiles,
    int max_row_tiles) {
  constexpr int kTargetTile = 4;
  constexpr int kRowLanes = kThreads / kTargetTile;
  __shared__ float partial[kRowLanes][kTargetTile];

  const int tile = blockIdx.x;
  const int col_lane = threadIdx.x & (kTargetTile - 1);
  const int row_lane = threadIdx.x / kTargetTile;
  const int target_off = blockIdx.y * kTargetTile + col_lane;
  const int batch = blockIdx.z;
  const bool valid = batch < batch_count && target_off < target_count;
  const int target = j + 1 + target_off;
  const int row0 = j + tile * ROW_SPLIT_ROWS_PER_TILE;
  const int row1 = min(row0 + ROW_SPLIT_ROWS_PER_TILE, m);
  const int64_t base = static_cast<int64_t>(batch) * m * n;

  float local = 0.0f;
  if (valid) {
    for (int row = row0 + row_lane; row < row1; row += kRowLanes) {
      const float v =
          (row == j) ? 1.0f : h[base + static_cast<int64_t>(row) * n + j];
      local += v * h[base + static_cast<int64_t>(row) * n + target];
    }
  }
  partial[row_lane][col_lane] = local;
  __syncthreads();

  if (row_lane == 0 && valid) {
    float sum = 0.0f;
    for (int lane = 0; lane < kRowLanes; ++lane) {
      sum += partial[lane][col_lane];
    }
    const int pj = j - j_start;
    const int target_panel = target - j_start;
    dot_ws[((static_cast<int64_t>(batch) * panel_cols + pj) * panel_cols +
            target_panel) * max_row_tiles + tile] = sum;
  }
}

__global__ void update_target_multi4_kernel(
    float* __restrict__ h,
    float* __restrict__ tau,
    float* __restrict__ dot_ws,
    int batch_count,
    int m,
    int n,
    int j_start,
    int panel_cols,
    int j,
    int target_count,
    int row_tiles,
    int max_row_tiles) {
  constexpr int kTargetTile = 4;
  constexpr int kRowLanes = kThreads / kTargetTile;
  __shared__ float weights[kTargetTile];

  const int col_lane = threadIdx.x & (kTargetTile - 1);
  const int row_lane = threadIdx.x / kTargetTile;
  const int target_off = blockIdx.x * kTargetTile + col_lane;
  const int row_tile = blockIdx.y;
  const int batch = blockIdx.z;
  const bool valid = batch < batch_count && target_off < target_count;
  const int target = j + 1 + target_off;
  const int pj = j - j_start;
  const int target_panel = target - j_start;
  const int64_t base = static_cast<int64_t>(batch) * m * n;

  if (row_lane == 0) {
    float dot = 0.0f;
    if (valid) {
      for (int tile = 0; tile < row_tiles; ++tile) {
        dot += dot_ws[((static_cast<int64_t>(batch) * panel_cols + pj) *
                       panel_cols + target_panel) * max_row_tiles + tile];
      }
    }
    weights[col_lane] =
        valid ? tau[static_cast<int64_t>(batch) * n + j] * dot : 0.0f;
  }
  __syncthreads();

  if (valid) {
    const float w = weights[col_lane];
    const int row0 = j + row_tile * ROW_SPLIT_ROWS_PER_TILE;
    const int row1 = min(row0 + ROW_SPLIT_ROWS_PER_TILE, m);
    for (int row = row0 + row_lane; row < row1; row += kRowLanes) {
      const float v =
          (row == j) ? 1.0f : h[base + static_cast<int64_t>(row) * n + j];
      h[base + static_cast<int64_t>(row) * n + target] -= v * w;
    }
  }
}

__global__ void apply_target_fused_kernel(
    float* __restrict__ h,
    float* __restrict__ tau,
    int batch_count,
    int m,
    int n,
    int j,
    int target_count) {
  const int target_off = blockIdx.x;
  const int batch = blockIdx.y;
  const int tid = threadIdx.x;
  if (batch >= batch_count || target_off >= target_count) return;

  const int target = j + 1 + target_off;
  const int64_t base = static_cast<int64_t>(batch) * m * n;

  float local = 0.0f;
  for (int row = j + tid; row < m; row += blockDim.x) {
    const float v = (row == j) ? 1.0f : h[base + static_cast<int64_t>(row) * n + j];
    local += v * h[base + static_cast<int64_t>(row) * n + target];
  }
  const float dot = block_sum(local);

  __shared__ float w_s;
  if (tid == 0) {
    const float tau_j = tau[static_cast<int64_t>(batch) * n + j];
    w_s = tau_j * dot;
  }
  __syncthreads();

  const float w = w_s;
  for (int row = j + tid; row < m; row += blockDim.x) {
    const float v = (row == j) ? 1.0f : h[base + static_cast<int64_t>(row) * n + j];
    h[base + static_cast<int64_t>(row) * n + target] -= v * w;
  }
}

template <int kTargetTile>
__global__ void apply_target_tiled_kernel(
    float* __restrict__ h,
    float* __restrict__ tau,
    int batch_count,
    int m,
    int n,
    int j,
    int target_count) {
  constexpr int kRowLanes = kThreads / kTargetTile;
  __shared__ float partial[kTargetTile][kRowLanes];
  __shared__ float weights[kTargetTile];

  const int batch = blockIdx.y;
  const int col_lane = threadIdx.x & (kTargetTile - 1);
  const int row_lane = threadIdx.x / kTargetTile;
  const int target_off = blockIdx.x * kTargetTile + col_lane;
  const bool valid = batch < batch_count && target_off < target_count;
  const int target = j + 1 + target_off;
  const int64_t base = static_cast<int64_t>(batch) * m * n;

  float local = 0.0f;
  if (valid) {
    for (int row = j + row_lane; row < m; row += kRowLanes) {
      const float v =
          (row == j) ? 1.0f : h[base + static_cast<int64_t>(row) * n + j];
      local += v * h[base + static_cast<int64_t>(row) * n + target];
    }
  }
  partial[col_lane][row_lane] = local;
  __syncthreads();

  if (row_lane == 0) {
    float dot = 0.0f;
    for (int lane = 0; lane < kRowLanes; ++lane) {
      dot += partial[col_lane][lane];
    }
    weights[col_lane] =
        valid ? tau[static_cast<int64_t>(batch) * n + j] * dot : 0.0f;
  }
  __syncthreads();

  if (valid) {
    const float w = weights[col_lane];
    for (int row = j + row_lane; row < m; row += kRowLanes) {
      const float v =
          (row == j) ? 1.0f : h[base + static_cast<int64_t>(row) * n + j];
      h[base + static_cast<int64_t>(row) * n + target] -= v * w;
    }
  }
}

__global__ void compute_y_partial_raw_kernel(
    const float* __restrict__ h,
    float* __restrict__ y_partial,
    int batch_count,
    int m,
    int n,
    int y_panel_cap,
    int y_trailing_cap,
    int y_row_tile_cap,
    int j_start,
    int j_end,
    int trailing_cols,
    int row_tiles) {
  const int col_off = blockIdx.x;
  const int batch = blockIdx.y;
  const int zidx = blockIdx.z;
  const int tid = threadIdx.x;
  const int panel_cols = j_end - j_start;
  const int panel_i = zidx / row_tiles;
  const int row_tile = zidx - panel_i * row_tiles;
  if (batch >= batch_count || col_off >= trailing_cols || panel_i >= panel_cols) return;

  const int diag = j_start + panel_i;
  const int col = j_end + col_off;
  const int row_begin = j_start + row_tile * 64;
  const int row_end = min(row_begin + 64, m);
  const int64_t h_base = static_cast<int64_t>(batch) * m * n;

  float local = 0.0f;
  for (int row = row_begin + tid; row < row_end; row += blockDim.x) {
    float v_val = 0.0f;
    if (row == diag) {
      v_val = 1.0f;
    } else if (row > diag) {
      v_val = h[h_base + static_cast<int64_t>(row) * n + diag];
    }
    local += v_val * h[h_base + static_cast<int64_t>(row) * n + col];
  }
  const float sum = block_sum(local);
  if (tid == 0) {
    y_partial[(((static_cast<int64_t>(batch) * y_panel_cap + panel_i) * y_trailing_cap + col_off) * y_row_tile_cap) + row_tile] = sum;
  }
}

__global__ void reduce_z_update_fused_raw_kernel(
    float* __restrict__ h,
    const float* __restrict__ tmat,
    const float* __restrict__ y_partial,
    int batch_count,
    int m,
    int n,
    int t_ld,
    int y_panel_cap,
    int y_trailing_cap,
    int y_row_tile_cap,
    int j_start,
    int j_end,
    int trailing_cols,
    int row_tiles) {
  const int col_lane = threadIdx.x;
  const int row_lane = threadIdx.y;
  const int col_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int batch = blockIdx.z;
  const int tid = row_lane * 16 + col_lane;
  const int panel_cols = j_end - j_start;
  const int col_off = col_tile * 16 + col_lane;
  const int row = j_start + row_tile * 16 + row_lane;

  __shared__ float y_buf[2048];
  __shared__ float z_buf[2048];

  if (batch >= batch_count) return;
  const int64_t h_base = static_cast<int64_t>(batch) * m * n;
  const int64_t t_base = static_cast<int64_t>(batch) * t_ld * t_ld;

  for (int idx = tid; idx < panel_cols * 16; idx += 256) {
    const int panel_i = idx / 16;
    const int c = idx - panel_i * 16;
    const int col = col_tile * 16 + c;
    float acc = 0.0f;
    if (col < trailing_cols) {
      for (int rt = 0; rt < row_tiles; ++rt) {
        acc += y_partial[(((static_cast<int64_t>(batch) * y_panel_cap + panel_i) * y_trailing_cap + col) * y_row_tile_cap) + rt];
      }
    }
    y_buf[idx] = acc;
  }
  __syncthreads();

  for (int idx = tid; idx < panel_cols * 16; idx += 256) {
    const int panel_i = idx / 16;
    const int c = idx - panel_i * 16;
    float acc = 0.0f;
    for (int k = 0; k < panel_cols; ++k) {
      acc += tmat[t_base + static_cast<int64_t>(panel_i) * t_ld + k] * y_buf[k * 16 + c];
    }
    z_buf[idx] = acc;
  }
  __syncthreads();

  if (row < m && col_off < trailing_cols) {
    float update = 0.0f;
    for (int i = 0; i < panel_cols; ++i) {
      const int diag = j_start + i;
      float v_val = 0.0f;
      if (row == diag) {
        v_val = 1.0f;
      } else if (row > diag) {
        v_val = h[h_base + static_cast<int64_t>(row) * n + diag];
      }
      update += v_val * z_buf[i * 16 + col_lane];
    }
    h[h_base + static_cast<int64_t>(row) * n + (j_end + col_off)] -= update;
  }
}

__global__ void build_t_dot_kernel(
    const float* __restrict__ h,
    float* __restrict__ dot_ws,
    int batch_count,
    int m,
    int n,
    int t_ld,
    int j_start,
    int panel_cols,
    int pair_count) {
  const int pair = blockIdx.x;
  const int batch = blockIdx.y;
  const int tid = threadIdx.x;
  if (batch >= batch_count || pair >= pair_count) return;

  int jj = 1;
  int base = 0;
  while (pair >= base + jj) {
    base += jj;
    ++jj;
  }
  const int prev = pair - base;

  const int64_t h_base = static_cast<int64_t>(batch) * m * n;
  float local = 0.0f;
  for (int rr = jj + tid; rr < m - j_start; rr += blockDim.x) {
    const float vj = (rr == jj)
        ? 1.0f
        : h[h_base + static_cast<int64_t>(j_start + rr) * n + (j_start + jj)];
    const float vi = h[h_base + static_cast<int64_t>(j_start + rr) * n + (j_start + prev)];
    local += vj * vi;
  }
  const float dot = block_sum(local);
  if (tid == 0) {
    dot_ws[(static_cast<int64_t>(batch) * t_ld + jj) * t_ld + prev] = dot;
  }
}

__global__ void build_t_finish_kernel(
    const float* __restrict__ tau,
    float* __restrict__ tmat,
    const float* __restrict__ dot_ws,
    int batch_count,
    int tau_stride,
    int t_ld,
    int j_start,
    int panel_cols) {
  const int batch = blockIdx.x;
  const int tid = threadIdx.x;
  if (batch >= batch_count) return;

  const int64_t tau_base = static_cast<int64_t>(batch) * tau_stride;
  const int64_t t_base = static_cast<int64_t>(batch) * t_ld * t_ld;
  const int64_t dot_base = static_cast<int64_t>(batch) * t_ld * t_ld;

  for (int idx = tid; idx < t_ld * t_ld; idx += blockDim.x) {
    tmat[t_base + idx] = 0.0f;
  }
  __syncthreads();

  for (int jj = 0; jj < panel_cols; ++jj) {
    const float tau_j = tau[tau_base + j_start + jj];
    for (int col = tid; col <= jj; col += blockDim.x) {
      if (col == jj) {
        tmat[t_base + static_cast<int64_t>(jj) * t_ld + col] = tau_j;
      } else {
        float value = 0.0f;
        for (int prev = 0; prev < jj; ++prev) {
          const float work =
              -tau_j * dot_ws[dot_base + static_cast<int64_t>(jj) * t_ld + prev];
          value += work * tmat[t_base + static_cast<int64_t>(prev) * t_ld + col];
        }
        tmat[t_base + static_cast<int64_t>(jj) * t_ld + col] = value;
      }
    }
    __syncthreads();
  }
}

#if PANEL_COOP_PANEL
__global__ void panel_factor_apply_coop_kernel(
    float* __restrict__ h,
    float* __restrict__ tau,
    float* __restrict__ sigma_ws,
    float* __restrict__ dot_ws,
    int batch_count,
    int m,
    int n,
    int j_start,
    int panel_cols,
    int row_tiles,
    int max_row_tiles) {
  cg::grid_group grid = cg::this_grid();
  constexpr int kTargetTile = 16;
  constexpr int kRowLanes = kThreads / kTargetTile;

  const int row_tile = blockIdx.x;
  const int target_tile = blockIdx.y;
  const int batch = blockIdx.z;
  const int tid = threadIdx.x;
  const int col_lane = tid & (kTargetTile - 1);
  const int row_lane = tid / kTargetTile;
  const int64_t base = static_cast<int64_t>(batch) * m * n;

  if (batch >= batch_count) return;

  for (int local_j = 0; local_j < panel_cols; ++local_j) {
    const int j = j_start + local_j;
    const int pj = local_j;

    if (target_tile == 0) {
      float local_sigma = 0.0f;
      const int row0 = j + 1 + row_tile * PANEL_COOP_ROW_TILE;
      const int row1 = min(row0 + PANEL_COOP_ROW_TILE, m);
      for (int row = row0 + tid; row < row1; row += blockDim.x) {
        const float x = h[base + static_cast<int64_t>(row) * n + j];
        local_sigma += x * x;
      }
      const float sum = block_sum(local_sigma);
      if (tid == 0) {
        sigma_ws[(static_cast<int64_t>(batch) * panel_cols + pj) * max_row_tiles + row_tile] = sum;
      }
    }
    grid.sync();

    if (row_tile == 0 && target_tile == 0) {
      float sigma = 0.0f;
      for (int tile = tid; tile < row_tiles; tile += blockDim.x) {
        sigma += sigma_ws[(static_cast<int64_t>(batch) * panel_cols + pj) * max_row_tiles + tile];
      }
      sigma = block_sum(sigma);
      if (tid == 0) {
        const float alpha = h[base + static_cast<int64_t>(j) * n + j];
        const float x_norm = sqrtf(alpha * alpha + sigma);
        const float beta = (alpha >= 0.0f) ? -x_norm : x_norm;
        float tau_j = 0.0f;
        float scale = 0.0f;
        if (sigma > 0.0f) {
          tau_j = (beta - alpha) / beta;
          scale = 1.0f / (alpha - beta);
          h[base + static_cast<int64_t>(j) * n + j] = beta;
        } else {
          h[base + static_cast<int64_t>(j) * n + j] = alpha;
        }
        tau[static_cast<int64_t>(batch) * n + j] = tau_j;
        sigma_ws[(static_cast<int64_t>(batch) * panel_cols + pj) * max_row_tiles + 0] = scale;
      }
    }
    grid.sync();

    if (target_tile == 0) {
      const float scale = sigma_ws[(static_cast<int64_t>(batch) * panel_cols + pj) * max_row_tiles + 0];
      const int row0 = j + 1 + row_tile * PANEL_COOP_ROW_TILE;
      const int row1 = min(row0 + PANEL_COOP_ROW_TILE, m);
      for (int row = row0 + tid; row < row1; row += blockDim.x) {
        h[base + static_cast<int64_t>(row) * n + j] *= scale;
      }
    }
    grid.sync();

    const int target_count = panel_cols - local_j - 1;
    const int target_off = target_tile * kTargetTile + col_lane;
    const bool valid_target = target_off < target_count;
    const int target = j + 1 + target_off;
    if (valid_target) {
      float local_dot = 0.0f;
      const int row0 = j + row_tile * PANEL_COOP_ROW_TILE;
      const int row1 = min(row0 + PANEL_COOP_ROW_TILE, m);
      for (int row = row0 + row_lane; row < row1; row += kRowLanes) {
        const float v = (row == j) ? 1.0f : h[base + static_cast<int64_t>(row) * n + j];
        local_dot += v * h[base + static_cast<int64_t>(row) * n + target];
      }
      __shared__ float partial[kRowLanes][kTargetTile];
      partial[row_lane][col_lane] = local_dot;
      __syncthreads();
      if (row_lane == 0) {
        float sum = 0.0f;
        for (int lane = 0; lane < kRowLanes; ++lane) {
          sum += partial[lane][col_lane];
        }
        const int target_panel = target - j_start;
        dot_ws[((static_cast<int64_t>(batch) * panel_cols + pj) * panel_cols + target_panel) * max_row_tiles + row_tile] = sum;
      }
    }
    grid.sync();

    if (row_tile == 0 && valid_target) {
      float dot = 0.0f;
      const int target_panel = target - j_start;
      for (int tile = tid; tile < row_tiles; tile += blockDim.x) {
        dot += dot_ws[((static_cast<int64_t>(batch) * panel_cols + pj) * panel_cols + target_panel) * max_row_tiles + tile];
      }
      dot = block_sum(dot);
      if (tid == 0) {
        const float tau_j = tau[static_cast<int64_t>(batch) * n + j];
        dot_ws[((static_cast<int64_t>(batch) * panel_cols + pj) * panel_cols + target_panel) * max_row_tiles + 0] = tau_j * dot;
      }
    }
    grid.sync();

    if (valid_target) {
      const int target_panel = target - j_start;
      const float w = dot_ws[((static_cast<int64_t>(batch) * panel_cols + pj) * panel_cols + target_panel) * max_row_tiles + 0];
      const int row0 = j + row_tile * PANEL_COOP_ROW_TILE;
      const int row1 = min(row0 + PANEL_COOP_ROW_TILE, m);
      for (int row = row0 + row_lane; row < row1; row += kRowLanes) {
        const float v = (row == j) ? 1.0f : h[base + static_cast<int64_t>(row) * n + j];
        h[base + static_cast<int64_t>(row) * n + target] -= v * w;
      }
    }
    grid.sync();
  }
}
#endif

}  // namespace panel_qr

extern "C" cudaError_t panel_factor_apply_cuda(
    float* h,
    float* tau,
    float* sigma_ws,
    float* dot_ws,
    int batch_count,
    int m,
    int n,
    int j_start,
    int j_end,
    int max_row_tiles) {
  if (h == nullptr || tau == nullptr || sigma_ws == nullptr || dot_ws == nullptr) {
    return cudaErrorInvalidDevicePointer;
  }
  if (batch_count < 0 || m <= 0 || n <= 0 || j_start < 0 || j_end < j_start || j_end > n || j_end - j_start > 128) {
    return cudaErrorInvalidValue;
  }
  const int panel_cols = j_end - j_start;
#if PANEL_COOP_PANEL
  if (m >= 2048 && panel_cols <= 16) {
    int coop_j_start = j_start;
    int coop_panel_cols = panel_cols;
    int coop_row_tiles = (m - j_start + PANEL_COOP_ROW_TILE - 1) / PANEL_COOP_ROW_TILE;
    const int target_tiles = 1;
    dim3 grid(coop_row_tiles, target_tiles, batch_count);
    dim3 block(panel_qr::kThreads, 1, 1);
    void* args[] = {&h, &tau, &sigma_ws, &dot_ws, &batch_count, &m, &n,
                    &coop_j_start, &coop_panel_cols, &coop_row_tiles, &max_row_tiles};
    cudaError_t coop_status = cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(panel_qr::panel_factor_apply_coop_kernel),
        grid, block, args, 0, nullptr);
    return coop_status == cudaSuccess ? cudaGetLastError() : coop_status;
  }
#endif
  auto factor_one = [&](int jj) {
    const int sigma_rows = m - (jj + 1);
    const int sigma_tiles = sigma_rows > 0 ? (sigma_rows + panel_qr::kRowsPerTile - 1) / panel_qr::kRowsPerTile : 0;
    if (m <= 1024 && sigma_tiles <= 1) {
      panel_qr::factor_single_tile_kernel<<<dim3(batch_count, 1, 1), panel_qr::kThreads>>>(
          h, tau, batch_count, m, n, jj);
    } else {
      if (sigma_tiles > 0) {
        panel_qr::sigma_partial_kernel<<<dim3(sigma_tiles, batch_count, 1), panel_qr::kThreads>>>(
            h, sigma_ws, batch_count, m, n, j_start, panel_cols, jj, sigma_tiles, max_row_tiles);
      }
      panel_qr::finalize_scale_kernel<<<dim3(batch_count, 1, 1), panel_qr::kThreads>>>(
          h, tau, sigma_ws, batch_count, m, n, j_start, panel_cols, jj, sigma_tiles, max_row_tiles);
    }
  };

  for (int j = j_start; j < j_end;) {
    factor_one(j);

    const int target_count = j_end - j - 1;
    if (target_count > 0) {
      const int apply_row_tiles_split = (m - j + ROW_SPLIT_ROWS_PER_TILE - 1) / ROW_SPLIT_ROWS_PER_TILE;
      const bool use_row_split = m >= ROW_SPLIT_MIN_M && target_count >= 8 && apply_row_tiles_split >= 4;
      if (use_row_split && ROW_SPLIT_TARGET_TILE == 16 && target_count >= 16) {
        const int apply_row_tiles = apply_row_tiles_split;
        panel_qr::dot_partial_multi16_kernel<<<dim3(apply_row_tiles, (target_count + 15) / 16, batch_count), panel_qr::kThreads>>>(
            h, dot_ws, batch_count, m, n, j_start, panel_cols, j, target_count,
            apply_row_tiles, max_row_tiles);
        panel_qr::update_target_multi16_kernel<<<dim3((target_count + 15) / 16, apply_row_tiles, batch_count), panel_qr::kThreads>>>(
            h, tau, dot_ws, batch_count, m, n, j_start, panel_cols, j, target_count,
            apply_row_tiles, max_row_tiles);
      } else if (use_row_split && target_count >= ROW_SPLIT_MULTI8_MIN_TARGETS) {
        const int apply_row_tiles = apply_row_tiles_split;
        panel_qr::dot_partial_multi8_kernel<<<dim3(apply_row_tiles, (target_count + 7) / 8, batch_count), panel_qr::kThreads>>>(
            h, dot_ws, batch_count, m, n, j_start, panel_cols, j, target_count,
            apply_row_tiles, max_row_tiles);
        panel_qr::update_target_multi8_kernel<<<dim3((target_count + 7) / 8, apply_row_tiles, batch_count), panel_qr::kThreads>>>(
            h, tau, dot_ws, batch_count, m, n, j_start, panel_cols, j, target_count,
            apply_row_tiles, max_row_tiles);
      } else if (use_row_split) {
        const int apply_row_tiles = apply_row_tiles_split;
        panel_qr::dot_partial_multi4_kernel<<<dim3(apply_row_tiles, (target_count + 3) / 4, batch_count), panel_qr::kThreads>>>(
            h, dot_ws, batch_count, m, n, j_start, panel_cols, j, target_count,
            apply_row_tiles, max_row_tiles);
        panel_qr::update_target_multi4_kernel<<<dim3((target_count + 3) / 4, apply_row_tiles, batch_count), panel_qr::kThreads>>>(
            h, tau, dot_ws, batch_count, m, n, j_start, panel_cols, j, target_count,
            apply_row_tiles, max_row_tiles);
      } else {
        const bool use_tiled2 = m >= 2048 || target_count <= 8;
        if (use_tiled2) {
          panel_qr::apply_target_tiled_kernel<2><<<dim3((target_count + 1) / 2, batch_count, 1), panel_qr::kThreads>>>(
              h, tau, batch_count, m, n, j, target_count);
        } else {
          panel_qr::apply_target_tiled_kernel<4><<<dim3((target_count + 3) / 4, batch_count, 1), panel_qr::kThreads>>>(
              h, tau, batch_count, m, n, j, target_count);
        }
      }
    }
    ++j;
  }
  return cudaGetLastError();
}

extern "C" cudaError_t build_compact_wy_t_raw_cuda(
    const float* h,
    const float* tau,
    float* tmat,
    float* dot_ws,
    int batch_count,
    int m,
    int n,
    int tau_stride,
    int t_ld,
    int j_start,
    int j_end) {
  if (h == nullptr || tau == nullptr || tmat == nullptr || dot_ws == nullptr) {
    return cudaErrorInvalidDevicePointer;
  }
  if (batch_count < 0 || m <= 0 || n <= 0 || tau_stride <= 0 || t_ld <= 0 ||
      j_start < 0 || j_end < j_start || j_end > n || j_end - j_start > t_ld) {
    return cudaErrorInvalidValue;
  }
  const int panel_cols = j_end - j_start;
  if (panel_cols == 0 || batch_count == 0) {
    return cudaSuccess;
  }
  const int pair_count = panel_cols * (panel_cols - 1) / 2;
  if (pair_count > 0) {
    panel_qr::build_t_dot_kernel<<<dim3(pair_count, batch_count, 1), panel_qr::kThreads>>>(
        h, dot_ws, batch_count, m, n, t_ld, j_start, panel_cols, pair_count);
  }
  panel_qr::build_t_finish_kernel<<<dim3(batch_count, 1, 1), panel_qr::kThreads>>>(
      tau, tmat, dot_ws, batch_count, tau_stride, t_ld, j_start, panel_cols);
  return cudaGetLastError();
}

extern "C" cudaError_t build_compact_wy_t_finish_cuda(
    const float* tau,
    float* tmat,
    const float* dot_ws,
    int batch_count,
    int tau_stride,
    int t_ld,
    int j_start,
    int panel_cols) {
  if (tau == nullptr || tmat == nullptr || dot_ws == nullptr) {
    return cudaErrorInvalidDevicePointer;
  }
  if (batch_count < 0 || tau_stride <= 0 || t_ld <= 0 || j_start < 0 ||
      panel_cols < 0 || panel_cols > t_ld || j_start + panel_cols > tau_stride) {
    return cudaErrorInvalidValue;
  }
  if (panel_cols == 0 || batch_count == 0) {
    return cudaSuccess;
  }
  panel_qr::build_t_finish_kernel<<<dim3(batch_count, 1, 1), panel_qr::kThreads>>>(
      tau, tmat, dot_ws, batch_count, tau_stride, t_ld, j_start, panel_cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apply_panel_wy_fused_update_raw_cuda(
    float* h,
    const float* tmat,
    float* y_partial,
    int batch_count,
    int m,
    int n,
    int t_ld,
    int y_panel_cap,
    int y_trailing_cap,
    int y_row_tile_cap,
    int j_start,
    int j_end,
    int row_tiles) {
  if (h == nullptr || tmat == nullptr || y_partial == nullptr) {
    return cudaErrorInvalidDevicePointer;
  }
  if (batch_count < 0 || m <= 0 || n <= 0 || t_ld <= 0 || y_panel_cap <= 0 ||
      y_trailing_cap < 0 || y_row_tile_cap <= 0 || j_start < 0 || j_end < j_start ||
      j_end > n || j_end - j_start > t_ld || j_end - j_start > y_panel_cap ||
      n - j_end > y_trailing_cap || row_tiles > y_row_tile_cap) {
    return cudaErrorInvalidValue;
  }
  const int panel_cols = j_end - j_start;
  const int trailing_cols = n - j_end;
  if (batch_count == 0 || panel_cols == 0 || trailing_cols == 0) {
    return cudaSuccess;
  }
  panel_qr::compute_y_partial_raw_kernel<<<dim3(trailing_cols, batch_count, panel_cols * row_tiles), panel_qr::kThreads>>>(
      h, y_partial, batch_count, m, n, y_panel_cap, y_trailing_cap, y_row_tile_cap,
      j_start, j_end, trailing_cols, row_tiles);
  panel_qr::reduce_z_update_fused_raw_kernel<<<dim3((trailing_cols + 15) / 16, (m - j_start + 15) / 16, batch_count), dim3(16, 16, 1)>>>(
      h, tmat, y_partial, batch_count, m, n, t_ld, y_panel_cap, y_trailing_cap, y_row_tile_cap,
      j_start, j_end, trailing_cols, row_tiles);
  return cudaGetLastError();
}
"""


def _load_panel_qr_ext():
    global _PANEL_QR_EXT
    if _PANEL_QR_EXT is not None:
        return _PANEL_QR_EXT

    from torch.utils.cpp_extension import load_inline

    arch_list, arch_tag = _small_square_arch_tag()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch_list)
    _PANEL_QR_EXT = load_inline(
        name=(
            f"panel_qr_inline_ext_{arch_tag}_bt2_bf1_"
            f"parowsplit{os.environ.get('ROW_SPLIT_TARGET_TILE', '8')}x8x4x_t{os.environ.get('ROW_SPLIT_MULTI8_MIN_TARGETS', '8')}_r{os.environ.get('ROW_SPLIT_ROWS_PER_TILE', '64')}_m{os.environ.get('ROW_SPLIT_MIN_M', '4096')}_coop{os.environ.get('PANEL_COOP_PANEL', '0')}_cr{os.environ.get('PANEL_COOP_ROW_TILE', '128')}_tdot1_pf1"
        ),
        cpp_sources=[_PANEL_QR_CPP_SOURCE],
        cuda_sources=[_PANEL_QR_CUDA_SOURCE],
        functions=["panel_factor_apply", "build_compact_wy_t_raw", "build_compact_wy_t_finish", "apply_panel_wy_fused_update_raw"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            f"-DROW_SPLIT_MULTI8_MIN_TARGETS={int(os.environ.get('ROW_SPLIT_MULTI8_MIN_TARGETS', '8'))}",
            f"-DROW_SPLIT_TARGET_TILE={int(os.environ.get('ROW_SPLIT_TARGET_TILE', '8'))}",
            f"-DROW_SPLIT_ROWS_PER_TILE={int(os.environ.get('ROW_SPLIT_ROWS_PER_TILE', '64'))}",
            f"-DROW_SPLIT_MIN_M={int(os.environ.get('ROW_SPLIT_MIN_M', '4096'))}",
            f"-DPANEL_COOP_PANEL={int(os.environ.get('PANEL_COOP_PANEL', '0'))}",
            f"-DPANEL_COOP_ROW_TILE={int(os.environ.get('PANEL_COOP_ROW_TILE', '128'))}",
        ],
        extra_cflags=[
            f"-DROW_SPLIT_ROWS_PER_TILE={int(os.environ.get('ROW_SPLIT_ROWS_PER_TILE', '64'))}",
        ],
        verbose=bool(int(os.environ.get("PANEL_QR_VERBOSE_BUILD", "0"))),
    )
    return _PANEL_QR_EXT

_FUSED_PANEL_QR_EXT = None

_FUSED_PANEL_QR_CPP_SOURCE = '#include <torch/extension.h>\n#include <c10/cuda/CUDAException.h>\n#include <cuda_runtime.h>\n#include <algorithm>\n\nextern "C" cudaError_t panel_factor_apply_fused_cuda(\n    float* h, float* tau, float* sigma_ws, float* scale_ws,\n    float* dot_ws,\n    int batch_count, int m, int n, int j_start, int j_end,\n    int max_row_tiles);\n\nextern "C" cudaError_t panel_factor_apply_fused2_cuda(\n    float* h, float* tau, float* sigma_ws, float* scale_ws,\n    float* dot_ws, float* w_ws,\n    int batch_count, int m, int n, int j_start, int j_end,\n    int max_row_tiles);\n\nvoid panel_factor_apply_fused(\n    torch::Tensor h, torch::Tensor tau, int64_t j_start, int64_t j_end) {\n  TORCH_CHECK(h.is_cuda() && tau.is_cuda(), "fused panel: cuda tensors required");\n  TORCH_CHECK(h.dtype() == torch::kFloat32, "fused panel: float32 required");\n  TORCH_CHECK(h.is_contiguous() && tau.is_contiguous(), "contiguous required");\n  TORCH_CHECK(h.dim() == 3 && tau.dim() == 2, "shape mismatch");\n  const int batch_count = (int)h.size(0);\n  const int m = (int)h.size(1);\n  const int n = (int)h.size(2);\n  const int js = (int)j_start, je = (int)j_end;\n  TORCH_CHECK(0 <= js && js <= je && je <= n, "bad bounds");\n  TORCH_CHECK(je - js <= 128, "panel_cols <= 128");\n\n  // workspaces\n  const int kRowsFactor = 256;\n  const int max_row_tiles_factor = (m + kRowsFactor - 1) / kRowsFactor;\n  const int kRowsApply = 64;\n  const int max_row_tiles_apply = (m + kRowsApply - 1) / kRowsApply;\n  const int max_row_tiles = std::max(max_row_tiles_factor, max_row_tiles_apply);\n\n  auto sigma_ws = torch::empty({batch_count, max_row_tiles}, h.options());\n  auto scale_ws = torch::empty({batch_count}, h.options());\n  // dot_ws layout reused for apply: [batch, panel_cols, panel_cols, max_row_tiles]\n  const int pcols = je - js;\n  auto dot_ws = torch::empty({batch_count, pcols, pcols, max_row_tiles}, h.options());\n\n  C10_CUDA_CHECK(panel_factor_apply_fused_cuda(\n      h.data_ptr<float>(), tau.data_ptr<float>(),\n      sigma_ws.data_ptr<float>(), scale_ws.data_ptr<float>(),\n      dot_ws.data_ptr<float>(),\n      batch_count, m, n, js, je, max_row_tiles));\n}\n\nvoid panel_factor_apply_fused2(\n    torch::Tensor h, torch::Tensor tau, int64_t j_start, int64_t j_end) {\n  TORCH_CHECK(h.is_cuda() && tau.is_cuda(), "fused2 panel: cuda tensors required");\n  TORCH_CHECK(h.dtype() == torch::kFloat32, "fused2 panel: float32 required");\n  TORCH_CHECK(h.is_contiguous() && tau.is_contiguous(), "contiguous required");\n  TORCH_CHECK(h.dim() == 3 && tau.dim() == 2, "shape mismatch");\n  const int batch_count = (int)h.size(0);\n  const int m = (int)h.size(1);\n  const int n = (int)h.size(2);\n  const int js = (int)j_start, je = (int)j_end;\n  TORCH_CHECK(0 <= js && js <= je && je <= n, "bad bounds");\n  TORCH_CHECK(je - js <= 128, "panel_cols <= 128");\n\n  const int kRowsFactor = 256;\n  const int max_row_tiles_factor = (m + kRowsFactor - 1) / kRowsFactor;\n  const int kRowsApply = 64;\n  const int max_row_tiles_apply = (m + kRowsApply - 1) / kRowsApply;\n  const int max_row_tiles = std::max(max_row_tiles_factor, max_row_tiles_apply);\n  const int pcols = je - js;\n\n  auto sigma_ws = torch::empty({batch_count, max_row_tiles}, h.options());\n  auto scale_ws = torch::empty({batch_count}, h.options());\n  auto dot_ws = torch::empty({batch_count, pcols, pcols, max_row_tiles}, h.options());\n  auto w_ws = torch::empty({batch_count, pcols}, h.options());\n\n  C10_CUDA_CHECK(panel_factor_apply_fused2_cuda(\n      h.data_ptr<float>(), tau.data_ptr<float>(),\n      sigma_ws.data_ptr<float>(), scale_ws.data_ptr<float>(),\n      dot_ws.data_ptr<float>(), w_ws.data_ptr<float>(),\n      batch_count, m, n, js, je, max_row_tiles));\n}'

_FUSED_PANEL_QR_CUDA_SOURCE = '#include <cuda_runtime.h>\n#include <cooperative_groups.h>\n#include <stdint.h>\n\nnamespace pf {\nnamespace cg = cooperative_groups;\n\nconstexpr int kThreads = 256;\nconstexpr int kRowsFactor = 256;\nconstexpr int kRowsApply = 64;\nconstexpr int kWarps = kThreads / 32;\n\n__device__ __forceinline__ float block_sum(float v) {\n  __shared__ float scratch[kWarps];\n  const int tid = threadIdx.x;\n  const int lane = tid & 31;\n  const int warp = tid >> 5;\n  #pragma unroll\n  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);\n  if (lane == 0) scratch[warp] = v;\n  __syncthreads();\n  v = (warp == 0 && lane < kWarps) ? scratch[lane] : 0.0f;\n  if (warp == 0) {\n    #pragma unroll\n    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);\n    if (lane == 0) scratch[0] = v;\n  }\n  __syncthreads();\n  return scratch[0];\n}\n\n// FACTOR fused: sigma_partial + finalize + scale_vtail in one cooperative launch.\n// grid = (row_tiles, batch_count), block = kThreads.\n__global__ void factor_coop_kernel(\n    float* __restrict__ h,\n    float* __restrict__ tau,\n    float* __restrict__ sigma_ws,\n    float* __restrict__ scale_ws,\n    int batch_count, int m, int n, int j, int row_tiles, int max_row_tiles) {\n  cg::grid_group grid = cg::this_grid();\n  const int tile = blockIdx.x;\n  const int batch = blockIdx.y;\n  const int tid = threadIdx.x;\n  const int64_t base = (int64_t)batch * m * n;\n\n  // Phase 1: partial sigma over rows j+1 .. m-1 within this tile\n  float local = 0.0f;\n  const int row0 = j + 1 + tile * kRowsFactor;\n  const int row1 = min(row0 + kRowsFactor, m);\n  for (int row = row0 + tid; row < row1; row += blockDim.x) {\n    const float x = h[base + (int64_t)row * n + j];\n    local += x * x;\n  }\n  const float partial = block_sum(local);\n  if (tid == 0) sigma_ws[(int64_t)batch * max_row_tiles + tile] = partial;\n\n  grid.sync();\n\n  // Phase 2: CTA 0 sums partials, computes scalars, broadcasts scale\n  if (tile == 0) {\n    float sigma = 0.0f;\n    for (int t = tid; t < row_tiles; t += blockDim.x)\n      sigma += sigma_ws[(int64_t)batch * max_row_tiles + t];\n    sigma = block_sum(sigma);\n    if (tid == 0) {\n      const float alpha = h[base + (int64_t)j * n + j];\n      const float x_norm = sqrtf(alpha * alpha + sigma);\n      const float beta = (alpha >= 0.0f) ? -x_norm : x_norm;\n      float tau_j = 0.0f, scale = 0.0f;\n      if (sigma > 0.0f) {\n        tau_j = (beta - alpha) / beta;\n        scale = 1.0f / (alpha - beta);\n        h[base + (int64_t)j * n + j] = beta;\n      } else {\n        h[base + (int64_t)j * n + j] = alpha;\n      }\n      tau[(int64_t)batch * n + j] = tau_j;\n      scale_ws[batch] = scale;\n    }\n  }\n\n  grid.sync();\n\n  // Phase 3: every CTA scales its row tile of v_tail\n  const float scale = scale_ws[batch];\n  for (int row = row0 + tid; row < row1; row += blockDim.x) {\n    h[base + (int64_t)row * n + j] *= scale;\n  }\n}\n\n// Standard multi8 dot+update path (copy from main panel ext, kept identical)\n__global__ void dot_partial_multi8_kernel(\n    float* __restrict__ h, float* __restrict__ dot_ws,\n    int batch_count, int m, int n, int j_start, int panel_cols, int j,\n    int target_count, int row_tiles, int max_row_tiles) {\n  const int tile = blockIdx.x;\n  const int tgroup = blockIdx.y;\n  const int batch = blockIdx.z;\n  const int tid = threadIdx.x;\n  const int target_lane = tid & 7;\n  const int row_lane = tid >> 3;\n  const int target_off = tgroup * 8 + target_lane;\n  const bool valid = target_off < target_count;\n  const int target = j + 1 + target_off;\n  const int target_panel = target - j_start;\n  const int64_t base = (int64_t)batch * m * n;\n\n  const int row0 = j + tile * kRowsApply;\n  const int row1 = min(row0 + kRowsApply, m);\n  float local = 0.0f;\n  if (valid) {\n    for (int row = row0 + row_lane; row < row1; row += 32) {\n      const float v = (row == j) ? 1.0f : h[base + (int64_t)row * n + j];\n      local += v * h[base + (int64_t)row * n + target];\n    }\n  }\n  __shared__ float partial[32][8];\n  partial[row_lane][target_lane] = local;\n  __syncthreads();\n  if (row_lane == 0 && valid) {\n    float sum = 0.0f;\n    for (int k = 0; k < 32; ++k) sum += partial[k][target_lane];\n    const int pj = j - j_start;\n    dot_ws[(((int64_t)batch * panel_cols + pj) * panel_cols + target_panel) * max_row_tiles + tile] = sum;\n  }\n}\n\n__global__ void update_target_multi8_kernel(\n    float* __restrict__ h, const float* __restrict__ tau, float* __restrict__ dot_ws,\n    int batch_count, int m, int n, int j_start, int panel_cols, int j,\n    int target_count, int row_tiles, int max_row_tiles) {\n  const int tgroup = blockIdx.x;\n  const int row_tile = blockIdx.y;\n  const int batch = blockIdx.z;\n  const int tid = threadIdx.x;\n  const int target_lane = tid & 7;\n  const int row_lane = tid >> 3;\n  const int target_off = tgroup * 8 + target_lane;\n  const bool valid = target_off < target_count;\n  const int target = j + 1 + target_off;\n  const int pj = j - j_start;\n  const int target_panel = target - j_start;\n  const int64_t base = (int64_t)batch * m * n;\n\n  __shared__ float w_s[8];\n  if (row_lane == 0 && valid) {\n    float dot = 0.0f;\n    for (int t = 0; t < row_tiles; ++t)\n      dot += dot_ws[(((int64_t)batch * panel_cols + pj) * panel_cols + target_panel) * max_row_tiles + t];\n    const float tau_j = tau[(int64_t)batch * n + j];\n    w_s[target_lane] = tau_j * dot;\n  }\n  __syncthreads();\n  if (!valid) return;\n  const float w = w_s[target_lane];\n  const int row0 = j + row_tile * kRowsApply;\n  const int row1 = min(row0 + kRowsApply, m);\n  for (int row = row0 + row_lane; row < row1; row += 32) {\n    const float v = (row == j) ? 1.0f : h[base + (int64_t)row * n + j];\n    h[base + (int64_t)row * n + target] -= v * w;\n  }\n}\n\n// APPLY fused: dot reduction + update in one cooperative launch (per column j).\n// grid = (apply_row_tiles, target_groups, batch), block = kThreads.\n// 8 targets per block (target_lane), 32 row lanes.\n__global__ void apply_coop_kernel(\n    float* __restrict__ h, const float* __restrict__ tau,\n    float* __restrict__ dot_ws, float* __restrict__ w_ws,\n    int batch_count, int m, int n, int j_start, int panel_cols, int j,\n    int target_count, int row_tiles, int max_row_tiles) {\n  cg::grid_group grid = cg::this_grid();\n  const int row_tile = blockIdx.x;\n  const int tgroup = blockIdx.y;\n  const int batch = blockIdx.z;\n  const int tid = threadIdx.x;\n  const int target_lane = tid & 7;\n  const int row_lane = tid >> 3;\n  const int target_off = tgroup * 8 + target_lane;\n  const bool valid = target_off < target_count;\n  const int target = j + 1 + target_off;\n  const int target_panel = target - j_start;\n  const int64_t base = (int64_t)batch * m * n;\n\n  // Phase 1: partial dot v^T c over this row tile\n  const int row0 = j + row_tile * kRowsApply;\n  const int row1 = min(row0 + kRowsApply, m);\n  float local = 0.0f;\n  if (valid) {\n    for (int row = row0 + row_lane; row < row1; row += 32) {\n      const float v = (row == j) ? 1.0f : h[base + (int64_t)row * n + j];\n      local += v * h[base + (int64_t)row * n + target];\n    }\n  }\n  __shared__ float part[32][8];\n  part[row_lane][target_lane] = local;\n  __syncthreads();\n  if (row_lane == 0 && valid) {\n    float s = 0.0f;\n    for (int k = 0; k < 32; ++k) s += part[k][target_lane];\n    dot_ws[(((int64_t)batch * panel_cols + 0) * panel_cols + target_panel) * max_row_tiles + row_tile] = s;\n  }\n\n  grid.sync();\n\n  // Phase 2: row_tile 0 reduces partials over all row tiles -> w = tau*dot\n  if (row_tile == 0 && valid) {\n    float dot = 0.0f;\n    for (int t = 0; t < row_tiles; ++t)\n      dot += dot_ws[(((int64_t)batch * panel_cols + 0) * panel_cols + target_panel) * max_row_tiles + t];\n    const float tau_j = tau[(int64_t)batch * n + j];\n    w_ws[(int64_t)batch * panel_cols + target_off] = tau_j * dot;\n  }\n\n  grid.sync();\n\n  // Phase 3: every block updates its row tile\n  if (valid) {\n    const float w = w_ws[(int64_t)batch * panel_cols + target_off];\n    for (int row = row0 + row_lane; row < row1; row += 32) {\n      const float v = (row == j) ? 1.0f : h[base + (int64_t)row * n + j];\n      h[base + (int64_t)row * n + target] -= v * w;\n    }\n  }\n}\n\n}  // namespace pf\n\nextern "C" cudaError_t panel_factor_apply_fused2_cuda(\n    float* h, float* tau, float* sigma_ws, float* scale_ws,\n    float* dot_ws, float* w_ws,\n    int batch_count, int m, int n, int j_start, int j_end,\n    int max_row_tiles) {\n  int panel_cols = j_end - j_start;\n  for (int j = j_start; j < j_end; ++j) {\n    int row_tiles_f = (m - (j + 1) + pf::kRowsFactor - 1) / pf::kRowsFactor;\n    if (row_tiles_f < 1) row_tiles_f = 1;\n    {\n      dim3 grid(row_tiles_f, batch_count, 1);\n      dim3 block(pf::kThreads, 1, 1);\n      void* args[] = {&h, &tau, &sigma_ws, &scale_ws,\n                      &batch_count, &m, &n, &j, &row_tiles_f, &max_row_tiles};\n      cudaError_t st = cudaLaunchCooperativeKernel(\n          (void*)pf::factor_coop_kernel, grid, block, args, 0, nullptr);\n      if (st != cudaSuccess) return st;\n    }\n\n    int target_count = j_end - j - 1;\n    if (target_count > 0) {\n      int apply_row_tiles = (m - j + pf::kRowsApply - 1) / pf::kRowsApply;\n      if (apply_row_tiles < 1) apply_row_tiles = 1;\n      int target_groups = (target_count + 7) / 8;\n      dim3 grid(apply_row_tiles, target_groups, batch_count);\n      dim3 block(pf::kThreads, 1, 1);\n      void* args[] = {&h, &tau, &dot_ws, &w_ws,\n                      &batch_count, &m, &n, &j_start, &panel_cols, &j,\n                      &target_count, &apply_row_tiles, &max_row_tiles};\n      cudaError_t st = cudaLaunchCooperativeKernel(\n          (void*)pf::apply_coop_kernel, grid, block, args, 0, nullptr);\n      if (st != cudaSuccess) return st;\n    }\n  }\n  return cudaGetLastError();\n}\n\nextern "C" cudaError_t panel_factor_apply_fused_cuda(\n    float* h, float* tau, float* sigma_ws, float* scale_ws,\n    float* dot_ws,\n    int batch_count, int m, int n, int j_start, int j_end,\n    int max_row_tiles) {\n  const int panel_cols = j_end - j_start;\n  for (int j = j_start; j < j_end; ++j) {\n    int row_tiles = (m - (j + 1) + pf::kRowsFactor - 1) / pf::kRowsFactor;\n    if (row_tiles > 0) {\n      dim3 grid(row_tiles, batch_count, 1);\n      dim3 block(pf::kThreads, 1, 1);\n      void* args[] = {&h, &tau, &sigma_ws, &scale_ws,\n                      &batch_count, &m, &n, &j, &row_tiles, &max_row_tiles};\n      cudaError_t st = cudaLaunchCooperativeKernel(\n          (void*)pf::factor_coop_kernel, grid, block, args, 0, nullptr);\n      if (st != cudaSuccess) return st;\n    } else {\n      // m - j - 1 == 0: trivial, just zero the tau and leave alpha.\n      // Run a tiny kernel: skip for simplicity (this happens at j = m-1 only).\n      dim3 grid(1, batch_count, 1);\n      dim3 block(pf::kThreads, 1, 1);\n      int row_tiles_one = 1;\n      void* args[] = {&h, &tau, &sigma_ws, &scale_ws,\n                      &batch_count, &m, &n, &j, &row_tiles_one, &max_row_tiles};\n      cudaError_t st = cudaLaunchCooperativeKernel(\n          (void*)pf::factor_coop_kernel, grid, block, args, 0, nullptr);\n      if (st != cudaSuccess) return st;\n    }\n\n    const int target_count = j_end - j - 1;\n    if (target_count > 0) {\n      const int apply_row_tiles = (m - j + pf::kRowsApply - 1) / pf::kRowsApply;\n      pf::dot_partial_multi8_kernel<<<dim3(apply_row_tiles, (target_count + 7) / 8, batch_count), pf::kThreads>>>(\n          h, dot_ws, batch_count, m, n, j_start, panel_cols, j, target_count,\n          apply_row_tiles, max_row_tiles);\n      pf::update_target_multi8_kernel<<<dim3((target_count + 7) / 8, apply_row_tiles, batch_count), pf::kThreads>>>(\n          h, tau, dot_ws, batch_count, m, n, j_start, panel_cols, j, target_count,\n          apply_row_tiles, max_row_tiles);\n    }\n  }\n  return cudaGetLastError();\n}'


def _load_fused_panel_qr_ext():
    global _FUSED_PANEL_QR_EXT
    if _FUSED_PANEL_QR_EXT is not None:
        return _FUSED_PANEL_QR_EXT

    from torch.utils.cpp_extension import load_inline

    arch_list, arch_tag = _small_square_arch_tag()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch_list)
    _FUSED_PANEL_QR_EXT = load_inline(
        name=f"panel_qr_fused_factor_ext_{arch_tag}_v2",
        cpp_sources=[_FUSED_PANEL_QR_CPP_SOURCE],
        cuda_sources=[_FUSED_PANEL_QR_CUDA_SOURCE],
        functions=["panel_factor_apply_fused", "panel_factor_apply_fused2"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=bool(int(os.environ.get("PANEL_FUSED_VERBOSE_BUILD", "0"))),
    )
    return _FUSED_PANEL_QR_EXT


@triton.jit
def _build_t_dot_triton_kernel(
    h_ptr,
    dot_ptr,
    m,
    n,
    t_ld: tl.constexpr,
    j_start,
    panel_cols: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_PREV: tl.constexpr,
):
    jj = tl.program_id(0)
    prev = tl.program_id(1) * BLOCK_PREV + tl.arange(0, BLOCK_PREV)
    batch = tl.program_id(2)
    valid_prev = prev < jj
    acc = tl.zeros((BLOCK_PREV,), dtype=tl.float32)
    row_limit = m - j_start

    for row_base in tl.range(0, row_limit, BLOCK_ROWS):
        rr = row_base + tl.arange(0, BLOCK_ROWS)
        valid_row = (rr >= jj) & (rr < row_limit)
        h_base = batch * m * n + (j_start + rr) * n + j_start
        vj_loaded = tl.load(h_ptr + h_base + jj, mask=valid_row, other=0.0)
        vj = tl.where(rr == jj, 1.0, vj_loaded)
        vi = tl.load(
            h_ptr + h_base[:, None] + prev[None, :],
            mask=valid_row[:, None] & valid_prev[None, :],
            other=0.0,
        )
        acc += tl.sum(vj[:, None] * vi, axis=0)

    out = batch * t_ld * t_ld + jj * t_ld + prev
    tl.store(dot_ptr + out, acc, mask=valid_prev)


_TRITON_BUILD_T_CONFIG = None

_TRITON_PANEL_MAX_ROWS = 1024


@triton.jit
def _panel_factor_apply_register_triton_kernel(
    h_ptr,
    tau_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    j_start: tl.constexpr,
    PANEL_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    batch = tl.program_id(0)
    base = batch * m * n
    row_offsets = tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, PANEL_N)
    rows = j_start + row_offsets
    cols = j_start + col_offsets
    valid = (rows[:, None] < m) & (cols[None, :] < n)
    panel = tl.load(h_ptr + base + rows[:, None] * n + cols[None, :], mask=valid, other=0.0)

    for local_j in tl.static_range(0, 32):
        if local_j < PANEL_N:
            is_col = col_offsets == local_j
            col_vec = tl.sum(tl.where(is_col[None, :], panel, 0.0), axis=1)
            alpha = tl.sum(tl.where(row_offsets == local_j, col_vec, 0.0), axis=0)
            tail = tl.where(row_offsets > local_j, col_vec, 0.0)
            sigma = tl.sum(tail * tail, axis=0)
            norm = tl.sqrt(alpha * alpha + sigma)
            beta = tl.where(alpha >= 0.0, -norm, norm)
            active = sigma > 0.0
            beta_safe = tl.where(active, beta, 1.0)
            denom_safe = tl.where(active, alpha - beta, 1.0)
            tau_j = tl.where(active, (beta - alpha) / beta_safe, 0.0)
            scale = tl.where(active, 1.0 / denom_safe, 0.0)
            v = tl.where(row_offsets == local_j, 1.0, tl.where(row_offsets > local_j, col_vec * scale, 0.0))
            new_col = tl.where(
                row_offsets == local_j,
                tl.where(active, beta, alpha),
                tl.where(row_offsets > local_j, col_vec * scale, col_vec),
            )
            panel = tl.where(is_col[None, :], new_col[:, None], panel)
            tl.store(tau_ptr + batch * n + j_start + local_j, tau_j)

            active_rows = row_offsets >= local_j
            dots = tl.sum(tl.where(active_rows[:, None], v[:, None] * panel, 0.0), axis=0)
            update_cols = col_offsets > local_j
            panel = tl.where(
                active_rows[:, None] & update_cols[None, :],
                panel - v[:, None] * (tau_j * dots)[None, :],
                panel,
            )

    tl.store(h_ptr + base + rows[:, None] * n + cols[None, :], panel, mask=valid)


@triton.jit
def _panel_factor_apply_fused_triton_kernel(
    h_ptr,
    tau_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    j_start: tl.constexpr,
    panel_cols: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    base = batch * m * n
    row_offsets = tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, BLOCK_N)

    for local_j in tl.static_range(0, 128):
        if local_j < panel_cols:
            j = j_start + local_j
            rows_tail = j + 1 + row_offsets
            x = tl.load(
                h_ptr + base + rows_tail * n + j,
                mask=rows_tail < m,
                other=0.0,
            )
            sigma = tl.sum(x * x, axis=0)
            alpha = tl.load(h_ptr + base + j * n + j)
            norm = tl.sqrt(alpha * alpha + sigma)
            beta = tl.where(alpha >= 0.0, -norm, norm)
            active = sigma > 0.0
            beta_safe = tl.where(active, beta, 1.0)
            denom_safe = tl.where(active, alpha - beta, 1.0)
            tau_j = tl.where(active, (beta - alpha) / beta_safe, 0.0)
            scale = tl.where(active, 1.0 / denom_safe, 0.0)
            rr = j + row_offsets
            orig_v_tail = tl.load(
                h_ptr + base + rr * n + j,
                mask=(rr > j) & (rr < m),
                other=0.0,
            )
            v = tl.where(rr == j, 1.0, orig_v_tail * scale)

            tl.store(h_ptr + base + j * n + j, tl.where(active, beta, alpha))
            tl.store(tau_ptr + batch * n + j, tau_j)
            tl.store(h_ptr + base + rows_tail * n + j, x * scale, mask=rows_tail < m)
            for target_base in tl.static_range(0, 128, BLOCK_N):
                if target_base < panel_cols:
                    target_local = target_base + col_offsets
                    cc = j_start + target_local
                    valid_col = (target_local < panel_cols) & (target_local > local_j)
                    c = tl.load(
                        h_ptr + base + rr[:, None] * n + cc[None, :],
                        mask=(rr[:, None] < m) & valid_col[None, :],
                        other=0.0,
                    )
                    dot = tl.sum(v[:, None] * c, axis=0)
                    out = c - v[:, None] * (tau_j * dot)[None, :]
                    tl.store(
                        h_ptr + base + rr[:, None] * n + cc[None, :],
                        out,
                        mask=(rr[:, None] < m) & valid_col[None, :],
                    )


@triton.jit
def _panel_factor_col_triton_kernel(
    h_ptr,
    tau_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    j: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    batch = tl.program_id(0)
    offs = tl.arange(0, BLOCK_M)
    rows_tail = j + 1 + offs
    base = batch * m * n
    x = tl.load(h_ptr + base + rows_tail * n + j, mask=rows_tail < m, other=0.0)
    sigma = tl.sum(x * x, axis=0)
    alpha = tl.load(h_ptr + base + j * n + j)
    norm = tl.sqrt(alpha * alpha + sigma)
    beta = tl.where(alpha >= 0.0, -norm, norm)
    active = sigma > 0.0
    tau_j = tl.where(active, (beta - alpha) / beta, 0.0)
    scale = tl.where(active, 1.0 / (alpha - beta), 0.0)
    tl.store(h_ptr + base + j * n + j, tl.where(active, beta, alpha))
    tl.store(tau_ptr + batch * n + j, tau_j)
    tl.store(h_ptr + base + rows_tail * n + j, x * scale, mask=rows_tail < m)


@triton.jit
def _panel_apply_targets_triton_kernel(
    h_ptr,
    tau_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    j: tl.constexpr,
    target_start: tl.constexpr,
    target_count: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0)
    batch = tl.program_id(1)
    rr = j + tl.arange(0, BLOCK_M)
    cc = target_start + tile * BLOCK_N + tl.arange(0, BLOCK_N)
    base = batch * m * n
    v_loaded = tl.load(h_ptr + base + rr * n + j, mask=rr < m, other=0.0)
    v = tl.where(rr == j, 1.0, v_loaded)
    c = tl.load(
        h_ptr + base + rr[:, None] * n + cc[None, :],
        mask=(rr[:, None] < m) & (cc[None, :] < target_start + target_count),
        other=0.0,
    )
    dot = tl.sum(v[:, None] * c, axis=0)
    tau_j = tl.load(tau_ptr + batch * n + j)
    w = tau_j * dot
    out = c - v[:, None] * w[None, :]
    tl.store(
        h_ptr + base + rr[:, None] * n + cc[None, :],
        out,
        mask=(rr[:, None] < m) & (cc[None, :] < target_start + target_count),
    )


import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

CUSTOM_KERNEL_BACKEND = "raw-cuda-small-square+shape-triton-register-panel-p32p16+triton-dot-raw-finish-t+fp32-gemm-wy"
_TSQR_WY_TREE_EXT_BACKEND = "tsqr_wy_tree"
_TSQR_WY_DIRECT_EXT_BACKEND = "tsqr_wy_direct"


def _runtime_triton_panel_n(n: int) -> int:
    """Register-panel width for Triton side tuning. Triton arange needs powers of two."""

    default = 32 if n in (352, 512) else 16
    value = os.environ.get("QR_TRITON_PANEL_N")
    if not value:
        return default
    try:
        panel_n = int(value)
    except ValueError:
        return default
    if panel_n in (16, 32):
        return panel_n
    return default


def _runtime_block_nb(n: int) -> int:
    """Shape-aware block size with optional side-experiment overrides."""

    value = os.environ.get("QR_BLOCK_NB")
    if value:
        try:
            nb = int(value)
        except ValueError:
            nb = 64
        return max(1, min(nb, 128))

    if n in (352, 512, 1024):
        return _runtime_triton_panel_n(n)
    return 64


def _runtime_panel_backend(n: int) -> str:
    """Optional panel backend override with a shape-aware Triton-register default."""

    value = os.environ.get("QR_PANEL_BACKEND")
    if value and value not in (_TSQR_WY_TREE_EXT_BACKEND, _TSQR_WY_DIRECT_EXT_BACKEND):
        return value
    if n in (352, 512, 1024):
        return "triton_register"
    return "raw"


def _runtime_panel_width(n: int, j_start: int, base_nb: int) -> int:
    """Dynamic panel width for medium-shape Triton-register panels."""

    if os.environ.get("QR_BLOCK_NB"):
        return base_nb
    if n in (352, 512, 1024):
        return _runtime_triton_panel_n(n)
    return base_nb


def _runtime_panel_backend_for(n: int, j_start: int) -> str:
    value = os.environ.get("QR_PANEL_BACKEND")
    if value and value not in (_TSQR_WY_TREE_EXT_BACKEND, _TSQR_WY_DIRECT_EXT_BACKEND):
        return value
    if n in (352, 512, 1024):
        return "triton_register"
    return "raw"


def _runtime_tsqr_wy_tree_max_n() -> int:
    value = os.environ.get("QR_TSQR_WY_MAX_N")
    if not value:
        return 512
    try:
        return max(0, int(value))
    except ValueError:
        return 512


def _runtime_tsqr_wy_direct_max_n() -> int:
    value = os.environ.get("QR_TSQR_WY_DIRECT_MAX_N")
    if not value:
        return 4096
    try:
        return max(0, int(value))
    except ValueError:
        return 4096


try:
    from task import input_t, output_t
except ModuleNotFoundError:
    input_t = torch.Tensor
    output_t = tuple[torch.Tensor, torch.Tensor]

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
    lane = cute.arch.lane_idx()
    warp_id = tidx // 32
    bidx, _, _ = cute.arch.block_idx()
    smem = SmemAllocator()
    partial = smem.allocate_tensor(cutlass.Float32, 64, byte_alignment=16)

    if bidx < batch_count:
        j = j_start
        while j < j_end:
            alpha = h[bidx, j, j]
            local = alpha * 0.0
            row = j + 1 + tidx
            while row < m:
                x = h[bidx, row, j]
                local += x * x
                row += 64

            warp_sum = local
            offset = 1
            while offset < cute.arch.WARP_SIZE:
                value = cute.arch.shuffle_sync_up(warp_sum, offset, mask_and_clamp=0)
                if lane >= offset:
                    warp_sum += value
                offset = offset << 1

            if lane == cute.arch.WARP_SIZE - 1:
                partial[warp_id] = warp_sum
            cute.arch.sync_threads()

            if warp_id == 0:
                block_sum = alpha * 0.0
                if lane < 2:
                    block_sum = partial[lane]

                offset2 = 1
                while offset2 < 2:
                    value2 = cute.arch.shuffle_sync_up(block_sum, offset2, mask_and_clamp=0)
                    if lane >= offset2:
                        block_sum += value2
                    offset2 = offset2 << 1

                if lane == 1:
                    partial[0] = block_sum
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
                row += 64
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
        grid=(batch_count, 1, 1), block=(64, 1, 1)
    )


@cute.kernel
def _part5_compute_y_partial_kernel(
    h: cute.Tensor,
    y_partial: cute.Tensor,
    batch_count: Int32,
    m: Int32,
    n: Int32,
    j_start: Int32,
    j_end: Int32,
    trailing_cols: Int32,
    row_tiles: Int32,
):
    from cutlass.utils.smem_allocator import SmemAllocator

    tidx, _, _ = cute.arch.thread_idx()
    col_off, bidx, zidx = cute.arch.block_idx()
    panel_cols = j_end - j_start
    panel_i = zidx // row_tiles
    row_tile = zidx - panel_i * row_tiles
    diag = j_start + panel_i
    col = j_end + col_off
    row_begin = j_start + row_tile * 64
    row_end = row_begin + 64
    if row_end > m:
        row_end = m

    smem = SmemAllocator()
    partial = smem.allocate_tensor(cutlass.Float32, 128, byte_alignment=16)

    local = h[bidx, j_start, col] * 0.0
    if bidx < batch_count and col_off < trailing_cols and panel_i < panel_cols:
        row = row_begin + tidx
        while row < row_end:
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
        y_partial[bidx, panel_i, col_off, row_tile] = partial[0]


@cute.kernel
def _part5_reduce_z_update_c_fused_kernel(
    h: cute.Tensor,
    y_partial: cute.Tensor,
    tmat: cute.Tensor,
    batch_count: Int32,
    m: Int32,
    n: Int32,
    j_start: Int32,
    j_end: Int32,
    panel_cols: Int32,
    trailing_cols: Int32,
    row_tiles: Int32,
):
    from cutlass.utils.smem_allocator import SmemAllocator

    col_lane, row_lane, _ = cute.arch.thread_idx()
    col_tile, row_tile, bidx = cute.arch.block_idx()
    tid = row_lane * 16 + col_lane
    col_off = col_tile * 16 + col_lane
    row = j_start + row_tile * 16 + row_lane

    smem = SmemAllocator()
    y_buf = smem.allocate_tensor(cutlass.Float32, 1024, byte_alignment=16)
    z_buf = smem.allocate_tensor(cutlass.Float32, 1024, byte_alignment=16)

    if bidx < batch_count:
        idx = tid
        while idx < panel_cols * 16:
            panel_i = idx // 16
            c = idx - panel_i * 16
            col = col_tile * 16 + c
            acc = tmat[bidx, 0, 0] * 0.0
            if col < trailing_cols:
                rt = 0
                while rt < row_tiles:
                    acc += y_partial[bidx, panel_i, col, rt]
                    rt += 1
            y_buf[idx] = acc
            idx += 256

        cute.arch.sync_threads()

        idx2 = tid
        while idx2 < panel_cols * 16:
            panel_i2 = idx2 // 16
            c2 = idx2 - panel_i2 * 16
            acc2 = tmat[bidx, 0, 0] * 0.0
            k = 0
            while k < panel_cols:
                acc2 += tmat[bidx, panel_i2, k] * y_buf[k * 16 + c2]
                k += 1
            z_buf[idx2] = acc2
            idx2 += 256

        cute.arch.sync_threads()

        if row < m and col_off < trailing_cols:
            update = h[bidx, j_start, j_end + col_off] * 0.0
            i = 0
            while i < panel_cols:
                diag = j_start + i
                v_val = h[bidx, j_start, j_end + col_off] * 0.0
                if row == diag:
                    v_val = 1.0
                if row > diag:
                    v_val = h[bidx, row, diag]
                update += v_val * z_buf[i * 16 + col_lane]
                i += 1
            h[bidx, row, j_end + col_off] = h[bidx, row, j_end + col_off] - update


@cute.jit
def part5_apply_panel_wy_fused_update_cuda(
    h: cute.Tensor,
    tmat: cute.Tensor,
    y_partial: cute.Tensor,
    batch_count: Int32,
    m: Int32,
    n: Int32,
    j_start: Int32,
    j_end: Int32,
    row_tiles: Int32,
):
    panel_cols = j_end - j_start
    trailing_cols = n - j_end
    _part5_compute_y_partial_kernel(
        h, y_partial, batch_count, m, n, j_start, j_end, trailing_cols, row_tiles
    ).launch(grid=(trailing_cols, batch_count, panel_cols * row_tiles), block=(128, 1, 1))
    _part5_reduce_z_update_c_fused_kernel(
        h, y_partial, tmat, batch_count, m, n, j_start, j_end, panel_cols, trailing_cols, row_tiles
    ).launch(
        grid=((trailing_cols + 15) // 16, (m - j_start + 15) // 16, batch_count),
        block=(16, 16, 1),
    )


@cute.kernel
def _build_compact_wy_t_kernel(
    h: cute.Tensor,
    tau: cute.Tensor,
    tmat: cute.Tensor,
    batch_count: Int32,
    m: Int32,
    n: Int32,
    j_start: Int32,
    j_end: Int32,
):
    bidx, _, _ = cute.arch.block_idx()
    panel_cols = j_end - j_start

    if bidx < batch_count:
        jj = 0
        while jj < panel_cols:
            tau_j = tau[bidx, j_start + jj]

            row = 0
            while row < panel_cols:
                tmat[bidx, jj, row] = tau_j * 0.0
                row += 1

            tmat[bidx, jj, jj] = tau_j

            if jj > 0:
                prev = 0
                while prev < jj:
                    dot = tau_j * 0.0
                    rr = jj
                    while rr < m - j_start:
                        vj = tau_j * 0.0
                        vi = tau_j * 0.0

                        if rr == jj:
                            vj = 1.0
                        if rr > jj:
                            vj = h[bidx, j_start + rr, j_start + jj]

                        if rr == prev:
                            vi = 1.0
                        if rr > prev:
                            vi = h[bidx, j_start + rr, j_start + prev]

                        dot += vj * vi
                        rr += 1

                    work = -tau_j * dot
                    col = 0
                    while col < jj:
                        tmat[bidx, jj, col] = tmat[bidx, jj, col] + work * tmat[bidx, prev, col]
                        col += 1
                    prev += 1
            jj += 1


@cute.jit
def build_compact_wy_t_cuda(
    h: cute.Tensor,
    tau: cute.Tensor,
    tmat: cute.Tensor,
    batch_count: Int32,
    m: Int32,
    n: Int32,
    j_start: Int32,
    j_end: Int32,
):
    _build_compact_wy_t_kernel(h, tau, tmat, batch_count, m, n, j_start, j_end).launch(
        grid=(batch_count, 1, 1), block=(1, 1, 1)
    )


def _panel_factor_apply_cutedsl_mvp(
    h: torch.Tensor,
    tau: torch.Tensor,
    j_start: int,
    j_end: int,
) -> None:
    """Legacy single-CTA CuTe DSL panel kernel kept as a fallback/reference."""

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


def _panel_factor_apply_raw_cuda(
    h: torch.Tensor,
    tau: torch.Tensor,
    j_start: int,
    j_end: int,
) -> None:
    """Raw CUDA panel path with target-column CTAs to avoid grid=(batch,1,1)."""

    ext = _load_panel_qr_ext()
    ext.panel_factor_apply(h, tau, int(j_start), int(j_end))


def _fused_panel_min_n() -> int:
    return int(os.environ.get("QR_FUSED_PANEL_MIN_N", "2048"))


def _use_fused_raw_panel(h: torch.Tensor, j_start: int, j_end: int) -> bool:
    if os.environ.get("QR_FUSED_PANEL", "1") == "0":
        return False
    _batch, m, n = h.shape
    panel_cols = int(j_end) - int(j_start)
    return n >= _fused_panel_min_n() and panel_cols > 0 and panel_cols <= 128 and m == n


def _panel_factor_apply_raw_cuda_default(
    h: torch.Tensor,
    tau: torch.Tensor,
    j_start: int,
    j_end: int,
) -> None:
    if _use_fused_raw_panel(h, j_start, j_end):
        ext = _load_fused_panel_qr_ext()
        ext.panel_factor_apply_fused(h, tau, int(j_start), int(j_end))
        return
    _panel_factor_apply_raw_cuda(h, tau, j_start, j_end)


def _triton_panel_block_rows(rows: int) -> int:
    if rows <= 256:
        return 256
    if rows <= 512:
        return 512
    return 1024


def _panel_factor_apply_triton_singleblock(
    h: torch.Tensor,
    tau: torch.Tensor,
    j_start: int,
    j_end: int,
) -> bool:
    """Experimental Triton panel path for rows <= 1024; returns False to fallback."""

    _batch, m, n = h.shape
    if m - j_start > _TRITON_PANEL_MAX_ROWS:
        return False
    for j in range(int(j_start), int(j_end)):
        rows = m - j
        block_m = _triton_panel_block_rows(rows)
        _panel_factor_col_triton_kernel[(h.size(0),)](
            h,
            tau,
            m,
            n,
            j,
            BLOCK_M=block_m,
            num_warps=8,
        )
        target_count = int(j_end) - j - 1
        if target_count > 0:
            block_n = 16
            grid = (triton.cdiv(target_count, block_n), h.size(0))
            _panel_apply_targets_triton_kernel[grid](
                h,
                tau,
                m,
                n,
                j,
                j + 1,
                target_count,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=8,
            )
    return True


def _panel_factor_apply_triton_fused(
    h: torch.Tensor,
    tau: torch.Tensor,
    j_start: int,
    j_end: int,
) -> bool:
    """Experimental one-launch Triton panel path for rows <= 1024."""

    _batch, m, n = h.shape
    panel_cols = int(j_end) - int(j_start)
    if m - j_start > _TRITON_PANEL_MAX_ROWS or panel_cols <= 0 or panel_cols > 128:
        return False
    block_m = _triton_panel_block_rows(m - int(j_start))
    _panel_factor_apply_fused_triton_kernel[(h.size(0),)](
        h,
        tau,
        m,
        n,
        int(j_start),
        panel_cols,
        BLOCK_M=block_m,
        BLOCK_N=16,
        num_warps=8,
    )
    return True


def _panel_factor_apply_triton_register(
    h: torch.Tensor,
    tau: torch.Tensor,
    j_start: int,
    j_end: int,
) -> bool:
    """Experimental true persistent/register-panel Triton path for panel_cols <= 32."""

    _batch, m, n = h.shape
    panel_cols = int(j_end) - int(j_start)
    if m - j_start > _TRITON_PANEL_MAX_ROWS or panel_cols <= 0 or panel_cols > 32:
        return False
    block_m = _triton_panel_block_rows(m - int(j_start))
    _panel_factor_apply_register_triton_kernel[(h.size(0),)](
        h,
        tau,
        m,
        n,
        int(j_start),
        panel_cols,
        BLOCK_M=block_m,
        num_warps=8,
    )
    return True


def _build_compact_wy_t_raw_cuda(
    h: torch.Tensor,
    tau: torch.Tensor,
    t: torch.Tensor,
    j_start: int,
    j_end: int,
) -> None:
    """Raw CUDA compact-WY T build with parallel dot reductions."""

    ext = _load_panel_qr_ext()
    ext.build_compact_wy_t_raw(h, tau, t, int(j_start), int(j_end))


def _build_compact_wy_t_triton_dot(
    h: torch.Tensor,
    tau: torch.Tensor,
    t: torch.Tensor,
    j_start: int,
    j_end: int,
) -> None:
    """Build compact-WY T with a coalesced multi-prev Triton dot kernel."""

    batch, m, n = h.shape
    panel_cols = j_end - j_start
    t_ld = t.size(1)
    dot_ws = torch.empty(batch, t_ld, t_ld, device=h.device, dtype=h.dtype)
    rows = m - j_start
    if _TRITON_BUILD_T_CONFIG is not None:
        block_rows, block_prev, num_warps = _TRITON_BUILD_T_CONFIG
    elif rows <= 1024:
        block_rows, block_prev, num_warps = 256, 32, 8
    elif rows <= 2048:
        block_rows, block_prev, num_warps = 512, 64, 8
    else:
        block_rows, block_prev, num_warps = 1024, 32, 8
    grid = (panel_cols, triton.cdiv(panel_cols, block_prev), batch)
    _build_t_dot_triton_kernel[grid](
        h,
        dot_ws,
        m,
        n,
        t_ld,
        int(j_start),
        panel_cols,
        BLOCK_ROWS=block_rows,
        BLOCK_PREV=block_prev,
        num_warps=num_warps,
    )
    ext = _load_panel_qr_ext()
    ext.build_compact_wy_t_finish(tau, t, dot_ws, int(j_start), int(j_end))


def _apply_panel_wy_fused_update_raw_cuda(
    h: torch.Tensor,
    t: torch.Tensor,
    y_partial: torch.Tensor,
    j_start: int,
    j_end: int,
    row_tiles: int,
) -> None:
    """Raw CUDA row-tiled fused WY trailing update."""

    ext = _load_panel_qr_ext()
    ext.apply_panel_wy_fused_update_raw(h, t, y_partial, int(j_start), int(j_end), int(row_tiles))


def _apply_panel_wy_update_gemm(
    h: torch.Tensor,
    t: torch.Tensor,
    j_start: int,
    j_end: int,
) -> None:
    """Apply compact WY with IEEE FP32 batched GEMMs."""

    batch = h.size(0)
    panel_cols = j_end - j_start
    v = h[:, j_start:, j_start:j_end].clone()
    top = torch.tril(v[:, :panel_cols, :], diagonal=-1)
    eye = torch.eye(panel_cols, device=h.device, dtype=h.dtype).expand(
        batch, panel_cols, panel_cols
    )
    v[:, :panel_cols, :] = top + eye

    c = h[:, j_start:, j_end:]
    t_panel = t[:, :panel_cols, :panel_cols]
    y = torch.bmm(v.transpose(1, 2), c)
    z = torch.bmm(t_panel, y)
    c.sub_(torch.bmm(v, z))


def householder_qr_blocked(
    A: torch.Tensor,
    nb: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blocked Householder QR with raw panel, Triton/CUDA T, and FP32 GEMM."""

    batch, m, n = A.shape
    nb = min(int(nb), 128)
    k = min(m, n)
    h = A.clone()
    tau = torch.zeros(batch, k, device=A.device, dtype=A.dtype)
    if n > nb:
        t_workspace = torch.empty(batch, nb, nb, device=A.device, dtype=A.dtype)

    old_precision = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        j_start = 0
        while j_start < k:
            panel_width = _runtime_panel_width(n, j_start, nb)
            j_end = min(j_start + panel_width, k)

            panel_backend = _runtime_panel_backend_for(n, j_start)
            if panel_backend == "triton_register":
                used_triton_panel = _panel_factor_apply_triton_register(
                    h, tau, j_start, j_end
                )
                if not used_triton_panel:
                    _panel_factor_apply_raw_cuda_default(h, tau, j_start, j_end)
            elif panel_backend == "triton_fused":
                used_triton_panel = _panel_factor_apply_triton_fused(
                    h, tau, j_start, j_end
                )
                if not used_triton_panel:
                    _panel_factor_apply_raw_cuda_default(h, tau, j_start, j_end)
            elif panel_backend == "triton":
                used_triton_panel = _panel_factor_apply_triton_singleblock(
                    h, tau, j_start, j_end
                )
                if not used_triton_panel:
                    _panel_factor_apply_raw_cuda_default(h, tau, j_start, j_end)
            else:
                _panel_factor_apply_raw_cuda_default(h, tau, j_start, j_end)

            if j_end < n:
                _build_compact_wy_t_triton_dot(h, tau, t_workspace, j_start, j_end)
                _apply_panel_wy_update_gemm(h, t_workspace, j_start, j_end)
            j_start = j_end
    finally:
        torch.backends.cuda.matmul.fp32_precision = old_precision

    return h, tau


def _tsqr_wy_tree_experimental(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Side-prototype backend for local A/B only; never active by default."""

    if os.environ.get("QR_PANEL_BACKEND") != _TSQR_WY_TREE_EXT_BACKEND:
        return None
    if data.dim() != 3 or data.size(1) != data.size(2):
        return None
    n = int(data.size(2))
    if n > _runtime_tsqr_wy_tree_max_n():
        return None

    try:
        from prototype_cuda_tsqr_panel import load_ext as _load_tsqr_ext
        from prototype_tsqr_checker_bridge import (
            explicit_qr_to_compact as _tsqr_explicit_qr_to_compact,
        )
        from prototype_tsqr_checker_bridge import (
            tsqr_blocked_paper_wy_compact_thin_full as _tsqr_wy_full,
        )
    except Exception:
        return None

    try:
        ext = _load_tsqr_ext()
        nb = int(os.environ.get("QR_TSQR_WY_NB", "16"))
        row_tile = int(os.environ.get("QR_TSQR_WY_ROW_TILE", "128"))
        q_total, _r = _tsqr_wy_full(ext, data.contiguous(), nb=nb, row_tile=row_tile)
        return _tsqr_explicit_qr_to_compact(data, q_total)
    except Exception:
        return None


def _tsqr_wy_direct_experimental(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Direct compact TSQR/WY backend for local A/B only; never active by default."""

    if os.environ.get("QR_PANEL_BACKEND") != _TSQR_WY_DIRECT_EXT_BACKEND:
        return None
    if data.dim() != 3 or data.size(1) != data.size(2):
        return None
    n = int(data.size(2))
    if n > _runtime_tsqr_wy_direct_max_n():
        return None

    try:
        from prototype_cuda_tsqr_panel import load_ext as _load_tsqr_ext
        from prototype_tsqr_checker_bridge import (
            tsqr_blocked_paper_wy_compact_thin_standard_output as _tsqr_wy_direct,
        )
    except Exception:
        return None

    try:
        ext = _load_tsqr_ext()
        nb = int(os.environ.get("QR_TSQR_WY_NB", "16"))
        row_tile = int(os.environ.get("QR_TSQR_WY_ROW_TILE", "128"))
        avoid_qthin = os.environ.get("QR_TSQR_WY_AVOID_QTHIN") == "1"
        h, tau, _metrics = _tsqr_wy_direct(
            ext,
            data.contiguous(),
            nb=nb,
            row_tile=row_tile,
            avoid_qthin=avoid_qthin,
        )
        return h, tau
    except Exception:
        return None


def _small_square_qr_cuda(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ext = _load_small_square_qr_ext()
    h, tau = ext.small_square_qr(data.contiguous())
    return h, tau


def custom_kernel(data: input_t) -> output_t:
    """Official harness entry point."""

    tsqr_wy_output = _tsqr_wy_tree_experimental(data)
    if tsqr_wy_output is not None:
        return tsqr_wy_output
    tsqr_wy_direct_output = _tsqr_wy_direct_experimental(data)
    if tsqr_wy_direct_output is not None:
        return tsqr_wy_direct_output

    if (
        data.dim() == 3
        and data.is_cuda
        and data.dtype == torch.float32
        and data.size(1) == data.size(2)
        and 0 < data.size(1) <= 64
    ):
        return _small_square_qr_cuda(data)

    return householder_qr_blocked(data, nb=_runtime_block_nb(int(data.size(2))))


def ref_kernel(data: input_t) -> output_t:
    return torch.geqrf(data)
