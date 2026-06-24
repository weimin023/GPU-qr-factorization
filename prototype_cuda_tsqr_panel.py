"""CUDA proof for one-panel TSQR / GEQRT-like large-panel factorization.

This is a side prototype and intentionally does not touch submission.py.  The
kernel factors independent row tiles of one skinny panel with Householder QR and
emits local R blocks.  A second small CUDA kernel can also factor the stacked R blocks.
"""

from __future__ import annotations

import argparse
import os
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


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float block_sum(float x) {
    static __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    if (lane == 0) {
        warp_sums[warp] = x;
    }
    __syncthreads();
    x = (threadIdx.x < ((blockDim.x + 31) >> 5)) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            x += __shfl_down_sync(0xffffffff, x, offset);
        }
        if (lane == 0) {
            warp_sums[0] = x;
        }
    }
    __syncthreads();
    return warp_sums[0];
}

__global__ void local_tile_householder_r_kernel(
    const float* __restrict__ panel,
    float* __restrict__ r_blocks,
    int batch,
    int rows,
    int nb,
    int row_tile,
    int num_tiles) {

    extern __shared__ float s[];
    int tile_id = blockIdx.x;
    int b = blockIdx.y;
    int row0 = tile_id * row_tile;
    int tile_rows = min(row_tile, rows - row0);
    int tid = threadIdx.x;

    for (int idx = tid; idx < row_tile * nb; idx += blockDim.x) {
        int r = idx / nb;
        int c = idx - r * nb;
        float v = 0.0f;
        if (r < tile_rows) {
            v = panel[(b * rows + row0 + r) * nb + c];
        }
        s[r * nb + c] = v;
    }
    __syncthreads();

    for (int j = 0; j < nb; ++j) {
        float local = 0.0f;
        for (int r = j + tid; r < tile_rows; r += blockDim.x) {
            float v = s[r * nb + j];
            local += v * v;
        }
        float norm2 = block_sum(local);
        float tau = 0.0f;
        float scale = 0.0f;
        float beta = 0.0f;
        if (tid == 0) {
            float alpha = s[j * nb + j];
            float norm = sqrtf(norm2);
            beta = (alpha >= 0.0f) ? -norm : norm;
            if (norm > 0.0f) {
                tau = (beta - alpha) / beta;
                scale = 1.0f / (alpha - beta);
                s[j * nb + j] = beta;
            }
            s[row_tile * nb + 0] = tau;
            s[row_tile * nb + 1] = scale;
        }
        __syncthreads();
        tau = s[row_tile * nb + 0];
        scale = s[row_tile * nb + 1];

        for (int r = j + 1 + tid; r < tile_rows; r += blockDim.x) {
            s[r * nb + j] *= scale;
        }
        __syncthreads();

        for (int c = j + 1; c < nb; ++c) {
            float dot = 0.0f;
            if (tid == 0) {
                dot += s[j * nb + c];
            }
            for (int r = j + 1 + tid; r < tile_rows; r += blockDim.x) {
                dot += s[r * nb + j] * s[r * nb + c];
            }
            dot = block_sum(dot);
            float w = tau * dot;
            if (tid == 0) {
                s[j * nb + c] -= w;
            }
            for (int r = j + 1 + tid; r < tile_rows; r += blockDim.x) {
                s[r * nb + c] -= s[r * nb + j] * w;
            }
            __syncthreads();
        }
    }

    float* out = r_blocks + (((b * num_tiles + tile_id) * nb) * nb);
    for (int idx = tid; idx < nb * nb; idx += blockDim.x) {
        int r = idx / nb;
        int c = idx - r * nb;
        out[idx] = (r <= c && r < tile_rows) ? s[r * nb + c] : 0.0f;
    }
}

__global__ void local_tile_householder_compact_kernel(
    const float* __restrict__ panel,
    float* __restrict__ h_tiles,
    float* __restrict__ tau_tiles,
    float* __restrict__ r_blocks,
    int batch,
    int rows,
    int nb,
    int row_tile,
    int num_tiles) {

    extern __shared__ float s[];
    int tile_id = blockIdx.x;
    int b = blockIdx.y;
    int row0 = tile_id * row_tile;
    int tile_rows = min(row_tile, rows - row0);
    int tid = threadIdx.x;
    float* s_tau = s + row_tile * nb;
    float* s_scalars = s_tau + nb;

    for (int idx = tid; idx < row_tile * nb; idx += blockDim.x) {
        int r = idx / nb;
        int c = idx - r * nb;
        float v = 0.0f;
        if (r < tile_rows) {
            v = panel[(b * rows + row0 + r) * nb + c];
        }
        s[r * nb + c] = v;
    }
    for (int j = tid; j < nb; j += blockDim.x) {
        s_tau[j] = 0.0f;
    }
    __syncthreads();

    for (int j = 0; j < nb; ++j) {
        float local = 0.0f;
        for (int r = j + tid; r < tile_rows; r += blockDim.x) {
            float v = s[r * nb + j];
            local += v * v;
        }
        float norm2 = block_sum(local);
        float tau = 0.0f;
        float scale = 0.0f;
        float beta = 0.0f;
        if (tid == 0) {
            float alpha = s[j * nb + j];
            float norm = sqrtf(norm2);
            beta = (alpha >= 0.0f) ? -norm : norm;
            if (norm > 0.0f) {
                tau = (beta - alpha) / beta;
                scale = 1.0f / (alpha - beta);
                s[j * nb + j] = beta;
            }
            s_tau[j] = tau;
            s_scalars[0] = tau;
            s_scalars[1] = scale;
        }
        __syncthreads();
        tau = s_scalars[0];
        scale = s_scalars[1];

        for (int r = j + 1 + tid; r < tile_rows; r += blockDim.x) {
            s[r * nb + j] *= scale;
        }
        __syncthreads();

        for (int c = j + 1; c < nb; ++c) {
            float dot = 0.0f;
            if (tid == 0) {
                dot += s[j * nb + c];
            }
            for (int r = j + 1 + tid; r < tile_rows; r += blockDim.x) {
                dot += s[r * nb + j] * s[r * nb + c];
            }
            dot = block_sum(dot);
            float w = tau * dot;
            if (tid == 0) {
                s[j * nb + c] -= w;
            }
            for (int r = j + 1 + tid; r < tile_rows; r += blockDim.x) {
                s[r * nb + c] -= s[r * nb + j] * w;
            }
            __syncthreads();
        }
    }

    float* h_out = h_tiles + (((b * num_tiles + tile_id) * row_tile) * nb);
    for (int idx = tid; idx < row_tile * nb; idx += blockDim.x) {
        int r = idx / nb;
        h_out[idx] = (r < tile_rows) ? s[idx] : 0.0f;
    }
    float* tau_out = tau_tiles + ((b * num_tiles + tile_id) * nb);
    for (int j = tid; j < nb; j += blockDim.x) {
        tau_out[j] = s_tau[j];
    }
    float* r_out = r_blocks + (((b * num_tiles + tile_id) * nb) * nb);
    for (int idx = tid; idx < nb * nb; idx += blockDim.x) {
        int r = idx / nb;
        int c = idx - r * nb;
        r_out[idx] = (r <= c && r < tile_rows) ? s[r * nb + c] : 0.0f;
    }
}


__global__ void stacked_householder_r_kernel(
    const float* __restrict__ r_blocks,
    float* __restrict__ r_final,
    int batch,
    int num_tiles,
    int nb) {

    extern __shared__ float s[];
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int stack_rows = num_tiles * nb;

    for (int idx = tid; idx < stack_rows * nb; idx += blockDim.x) {
        int r = idx / nb;
        int c = idx - r * nb;
        int tile = r / nb;
        int local_r = r - tile * nb;
        s[idx] = r_blocks[(((b * num_tiles + tile) * nb + local_r) * nb) + c];
    }
    __syncthreads();

    for (int j = 0; j < nb; ++j) {
        float local = 0.0f;
        for (int r = j + tid; r < stack_rows; r += blockDim.x) {
            float v = s[r * nb + j];
            local += v * v;
        }
        float norm2 = block_sum(local);
        float tau = 0.0f;
        float scale = 0.0f;
        if (tid == 0) {
            float alpha = s[j * nb + j];
            float norm = sqrtf(norm2);
            float beta = (alpha >= 0.0f) ? -norm : norm;
            if (norm > 0.0f) {
                tau = (beta - alpha) / beta;
                scale = 1.0f / (alpha - beta);
                s[j * nb + j] = beta;
            }
            s[stack_rows * nb + 0] = tau;
            s[stack_rows * nb + 1] = scale;
        }
        __syncthreads();
        tau = s[stack_rows * nb + 0];
        scale = s[stack_rows * nb + 1];

        for (int r = j + 1 + tid; r < stack_rows; r += blockDim.x) {
            s[r * nb + j] *= scale;
        }
        __syncthreads();

        for (int c = j + 1; c < nb; ++c) {
            float dot = 0.0f;
            if (tid == 0) {
                dot += s[j * nb + c];
            }
            for (int r = j + 1 + tid; r < stack_rows; r += blockDim.x) {
                dot += s[r * nb + j] * s[r * nb + c];
            }
            dot = block_sum(dot);
            float w = tau * dot;
            if (tid == 0) {
                s[j * nb + c] -= w;
            }
            for (int r = j + 1 + tid; r < stack_rows; r += blockDim.x) {
                s[r * nb + c] -= s[r * nb + j] * w;
            }
            __syncthreads();
        }
    }

    float* out = r_final + b * nb * nb;
    for (int idx = tid; idx < nb * nb; idx += blockDim.x) {
        int r = idx / nb;
        int c = idx - r * nb;
        out[idx] = (r <= c) ? s[r * nb + c] : 0.0f;
    }
}

__global__ void local_compact_apply_qt_kernel(
    const float* __restrict__ h_tiles,
    const float* __restrict__ tau_tiles,
    const float* __restrict__ c,
    float* __restrict__ out,
    int batch,
    int rows,
    int cols,
    int nb,
    int row_tile,
    int num_tiles,
    int block_n) {

    extern __shared__ float s[];
    float* s_h = s;
    float* s_c = s_h + row_tile * nb;
    float* s_dot = s_c + row_tile * block_n;

    int col_tile = blockIdx.x;
    int tile_id = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;
    int row0 = tile_id * row_tile;
    int tile_rows = min(row_tile, rows - row0);
    int col0 = col_tile * block_n;

    const float* h_base = h_tiles + (((b * num_tiles + tile_id) * row_tile) * nb);
    for (int idx = tid; idx < row_tile * nb; idx += blockDim.x) {
        s_h[idx] = h_base[idx];
    }
    for (int idx = tid; idx < row_tile * block_n; idx += blockDim.x) {
        int r = idx / block_n;
        int cn = idx - r * block_n;
        int col = col0 + cn;
        float v = 0.0f;
        if (r < tile_rows && col < cols) {
            v = c[(b * rows + row0 + r) * cols + col];
        }
        s_c[idx] = v;
    }
    __syncthreads();

    for (int j = 0; j < nb; ++j) {
        float tau = tau_tiles[(b * num_tiles + tile_id) * nb + j];
        for (int cn = 0; cn < block_n; ++cn) {
            float local = 0.0f;
            if (tid == 0) {
                local += s_c[j * block_n + cn];
            }
            for (int r = j + 1 + tid; r < tile_rows; r += blockDim.x) {
                local += s_h[r * nb + j] * s_c[r * block_n + cn];
            }
            float dot = block_sum(local);
            if (tid == 0) {
                s_dot[cn] = tau * dot;
            }
            __syncthreads();
        }

        for (int idx = tid; idx < (tile_rows - j) * block_n; idx += blockDim.x) {
            int rr = idx / block_n + j;
            int cn = idx - (rr - j) * block_n;
            float v = (rr == j) ? 1.0f : s_h[rr * nb + j];
            s_c[rr * block_n + cn] -= v * s_dot[cn];
        }
        __syncthreads();
    }

    for (int idx = tid; idx < row_tile * block_n; idx += blockDim.x) {
        int r = idx / block_n;
        int cn = idx - r * block_n;
        int col = col0 + cn;
        if (r < tile_rows && col < cols) {
            out[(b * rows + row0 + r) * cols + col] = s_c[idx];
        }
    }
}

__global__ void local_compact_apply_q_kernel(
    const float* __restrict__ h_tiles,
    const float* __restrict__ tau_tiles,
    const float* __restrict__ c,
    float* __restrict__ out,
    int batch,
    int rows,
    int cols,
    int nb,
    int row_tile,
    int num_tiles,
    int block_n) {

    extern __shared__ float s[];
    float* s_h = s;
    float* s_c = s_h + row_tile * nb;
    float* s_dot = s_c + row_tile * block_n;

    int col_tile = blockIdx.x;
    int tile_id = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;
    int row0 = tile_id * row_tile;
    int tile_rows = min(row_tile, rows - row0);
    int col0 = col_tile * block_n;

    const float* h_base = h_tiles + (((b * num_tiles + tile_id) * row_tile) * nb);
    for (int idx = tid; idx < row_tile * nb; idx += blockDim.x) {
        s_h[idx] = h_base[idx];
    }
    for (int idx = tid; idx < row_tile * block_n; idx += blockDim.x) {
        int r = idx / block_n;
        int cn = idx - r * block_n;
        int col = col0 + cn;
        float v = 0.0f;
        if (r < tile_rows && col < cols) {
            v = c[(b * rows + row0 + r) * cols + col];
        }
        s_c[idx] = v;
    }
    __syncthreads();

    for (int j = nb - 1; j >= 0; --j) {
        float tau = tau_tiles[(b * num_tiles + tile_id) * nb + j];
        for (int cn = 0; cn < block_n; ++cn) {
            float local = 0.0f;
            if (tid == 0) {
                local += s_c[j * block_n + cn];
            }
            for (int r = j + 1 + tid; r < tile_rows; r += blockDim.x) {
                local += s_h[r * nb + j] * s_c[r * block_n + cn];
            }
            float dot = block_sum(local);
            if (tid == 0) {
                s_dot[cn] = tau * dot;
            }
            __syncthreads();
        }

        for (int idx = tid; idx < (tile_rows - j) * block_n; idx += blockDim.x) {
            int rr = idx / block_n + j;
            int cn = idx - (rr - j) * block_n;
            float v = (rr == j) ? 1.0f : s_h[rr * nb + j];
            s_c[rr * block_n + cn] -= v * s_dot[cn];
        }
        __syncthreads();
    }

    for (int idx = tid; idx < row_tile * block_n; idx += blockDim.x) {
        int r = idx / block_n;
        int cn = idx - r * block_n;
        int col = col0 + cn;
        if (r < tile_rows && col < cols) {
            out[(b * rows + row0 + r) * cols + col] = s_c[idx];
        }
    }
}

__global__ void reconstruct_wy_lu_kernel(
    const float* __restrict__ q,
    float* __restrict__ w,
    float* __restrict__ y,
    float* __restrict__ signs,
    int batch,
    int rows,
    int q_cols,
    int k) {

    extern __shared__ float sh[];
    float* l = sh;
    float* u = l + k * k;
    float* s = u + k * k;
    int b = blockIdx.x;
    int tid = threadIdx.x;

    if (tid < k) {
        // Match the prototype's preferred successful path: A = I - Q_thin.
        // A diagonal sign search remains future work for harder sign cases.
        s[tid] = 1.0f;
        signs[b * k + tid] = s[tid];
    }
    __syncthreads();
    for (int idx = tid; idx < k * k; idx += blockDim.x) {
        int r = idx / k;
        int c = idx - r * k;
        float eye = (r == c) ? s[c] : 0.0f;
        u[idx] = eye - q[(b * rows + r) * q_cols + c];
        l[idx] = (r == c) ? 1.0f : 0.0f;
    }
    __syncthreads();

    if (tid == 0) {
        for (int j = 0; j < k - 1; ++j) {
            float pivot = u[j * k + j];
            for (int r = j + 1; r < k; ++r) {
                float factor = u[r * k + j] / pivot;
                l[r * k + j] = factor;
                u[r * k + j] = 0.0f;
                for (int c = j + 1; c < k; ++c) {
                    u[r * k + c] -= factor * u[j * k + c];
                }
            }
        }
    }
    __syncthreads();

    for (int row = tid; row < rows; row += blockDim.x) {
        float a_vals[64];
        float y_vals[64];
        float w_vals[64];
        for (int c = 0; c < k; ++c) {
            float eye = (row == c) ? s[c] : 0.0f;
            a_vals[c] = eye - q[(b * rows + row) * q_cols + c];
        }

        // y[row, :] solves y_row * U = a_row.
        for (int c = 0; c < k; ++c) {
            float v = a_vals[c];
            for (int p = 0; p < c; ++p) {
                v -= y_vals[p] * u[p * k + c];
            }
            y_vals[c] = v / u[c * k + c];
        }

        // w[row, :] solves L * w_row.T = a_row.T, with unit-lower L.
        for (int c = 0; c < k; ++c) {
            float v = a_vals[c];
            for (int p = 0; p < c; ++p) {
                v -= l[c * k + p] * w_vals[p];
            }
            w_vals[c] = v;
        }

        for (int c = 0; c < k; ++c) {
            y[(b * rows + row) * k + c] = y_vals[c];
            w[(b * rows + row) * k + c] = w_vals[c];
        }
    }
}

__device__ void apply_local_q_in_shared(
    const float* __restrict__ s_h,
    const float* __restrict__ s_tau,
    float* __restrict__ s_c,
    float* __restrict__ s_dot,
    int tile_rows,
    int nb,
    int block_n) {

    int tid = threadIdx.x;
    for (int j = nb - 1; j >= 0; --j) {
        float tau = s_tau[j];
        for (int cn = 0; cn < block_n; ++cn) {
            float local = 0.0f;
            if (tid == 0) {
                local += s_c[j * block_n + cn];
            }
            for (int r = j + 1 + tid; r < tile_rows; r += blockDim.x) {
                local += s_h[r * nb + j] * s_c[r * block_n + cn];
            }
            float dot = block_sum(local);
            if (tid == 0) {
                s_dot[cn] = tau * dot;
            }
            __syncthreads();
        }

        for (int idx = tid; idx < (tile_rows - j) * block_n; idx += blockDim.x) {
            int rr = idx / block_n + j;
            int cn = idx - (rr - j) * block_n;
            float v = (rr == j) ? 1.0f : s_h[rr * nb + j];
            s_c[rr * block_n + cn] -= v * s_dot[cn];
        }
        __syncthreads();
    }
}

__global__ void tree_wy_lu_prepare_kernel(
    const float* __restrict__ h_tiles,
    const float* __restrict__ tau_tiles,
    const float* __restrict__ basis,
    float* __restrict__ lu,
    float* __restrict__ signs,
    int batch,
    int rows,
    int k,
    int row_tile,
    int num_tiles) {

    extern __shared__ float sh[];
    float* s_h = sh;
    float* s_tau = s_h + row_tile * k;
    float* s_c = s_tau + k;
    float* s_dot = s_c + row_tile * k;
    float* l = s_dot + k;
    float* u = l + k * k;
    float* s_signs = u + k * k;

    int b = blockIdx.x;
    int tid = threadIdx.x;
    int tile_rows = min(row_tile, rows);

    const float* h_base = h_tiles + ((b * num_tiles) * row_tile * k);
    const float* tau_base = tau_tiles + (b * num_tiles) * k;
    for (int idx = tid; idx < row_tile * k; idx += blockDim.x) {
        s_h[idx] = h_base[idx];
    }
    for (int j = tid; j < k; j += blockDim.x) {
        s_tau[j] = tau_base[j];
        s_signs[j] = 1.0f;
        signs[b * k + j] = 1.0f;
    }
    for (int idx = tid; idx < row_tile * k; idx += blockDim.x) {
        int r = idx / k;
        int c = idx - r * k;
        float v = 0.0f;
        if (r < tile_rows) {
            v = basis[(b * rows + r) * k + c];
        }
        s_c[idx] = v;
    }
    __syncthreads();

    apply_local_q_in_shared(s_h, s_tau, s_c, s_dot, tile_rows, k, k);

    for (int idx = tid; idx < k * k; idx += blockDim.x) {
        int r = idx / k;
        int c = idx - r * k;
        float eye = (r == c) ? s_signs[c] : 0.0f;
        u[idx] = eye - s_c[r * k + c];
        l[idx] = (r == c) ? 1.0f : 0.0f;
    }
    __syncthreads();

    if (tid == 0) {
        for (int j = 0; j < k - 1; ++j) {
            float pivot = u[j * k + j];
            for (int r = j + 1; r < k; ++r) {
                float factor = u[r * k + j] / pivot;
                l[r * k + j] = factor;
                u[r * k + j] = 0.0f;
                for (int c = j + 1; c < k; ++c) {
                    u[r * k + c] -= factor * u[j * k + c];
                }
            }
        }
    }
    __syncthreads();

    float* lu_base = lu + b * 2 * k * k;
    for (int idx = tid; idx < k * k; idx += blockDim.x) {
        lu_base[idx] = l[idx];
        lu_base[k * k + idx] = u[idx];
    }
}

__global__ void tree_wy_lu_apply_kernel(
    const float* __restrict__ h_tiles,
    const float* __restrict__ tau_tiles,
    const float* __restrict__ basis,
    const float* __restrict__ lu,
    float* __restrict__ w,
    float* __restrict__ y,
    int batch,
    int rows,
    int k,
    int row_tile,
    int num_tiles) {

    extern __shared__ float sh[];
    float* s_h = sh;
    float* s_tau = s_h + row_tile * k;
    float* s_c = s_tau + k;
    float* s_dot = s_c + row_tile * k;
    float* l = s_dot + k;
    float* u = l + k * k;

    int tile_id = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    int row0 = tile_id * row_tile;
    int tile_rows = min(row_tile, rows - row0);

    const float* h_base = h_tiles + (((b * num_tiles + tile_id) * row_tile) * k);
    const float* tau_base = tau_tiles + (b * num_tiles + tile_id) * k;
    for (int idx = tid; idx < row_tile * k; idx += blockDim.x) {
        s_h[idx] = h_base[idx];
    }
    for (int j = tid; j < k; j += blockDim.x) {
        s_tau[j] = tau_base[j];
    }
    for (int idx = tid; idx < row_tile * k; idx += blockDim.x) {
        int r = idx / k;
        int c = idx - r * k;
        float v = 0.0f;
        if (r < tile_rows) {
            v = basis[(b * rows + row0 + r) * k + c];
        }
        s_c[idx] = v;
    }
    const float* lu_base = lu + b * 2 * k * k;
    for (int idx = tid; idx < k * k; idx += blockDim.x) {
        l[idx] = lu_base[idx];
        u[idx] = lu_base[k * k + idx];
    }
    __syncthreads();

    apply_local_q_in_shared(s_h, s_tau, s_c, s_dot, tile_rows, k, k);

    for (int row = tid; row < tile_rows; row += blockDim.x) {
        float a_vals[64];
        float y_vals[64];
        float w_vals[64];
        int global_row = row0 + row;
        for (int c = 0; c < k; ++c) {
            float eye = (global_row == c) ? 1.0f : 0.0f;
            a_vals[c] = eye - s_c[row * k + c];
        }

        for (int c = 0; c < k; ++c) {
            float v = a_vals[c];
            for (int p = 0; p < c; ++p) {
                v -= y_vals[p] * u[p * k + c];
            }
            y_vals[c] = v / u[c * k + c];
        }

        for (int c = 0; c < k; ++c) {
            float v = a_vals[c];
            for (int p = 0; p < c; ++p) {
                v -= l[c * k + p] * w_vals[p];
            }
            w_vals[c] = v;
        }

        for (int c = 0; c < k; ++c) {
            y[(b * rows + global_row) * k + c] = y_vals[c];
            w[(b * rows + global_row) * k + c] = w_vals[c];
        }
    }
}

torch::Tensor local_tile_householder_r(torch::Tensor panel, int64_t row_tile) {
    TORCH_CHECK(panel.is_cuda(), "panel must be CUDA");
    TORCH_CHECK(panel.dtype() == torch::kFloat32, "panel must be float32");
    TORCH_CHECK(panel.dim() == 3, "panel must have shape (batch, rows, nb)");
    TORCH_CHECK(panel.is_contiguous(), "panel must be contiguous");
    int batch = static_cast<int>(panel.size(0));
    int rows = static_cast<int>(panel.size(1));
    int nb = static_cast<int>(panel.size(2));
    int rt = static_cast<int>(row_tile);
    TORCH_CHECK(nb > 0 && nb <= 64, "nb must be in 1..64 for this proof");
    TORCH_CHECK(rt >= nb, "row_tile must be >= nb");
    int num_tiles = (rows + rt - 1) / rt;
    auto out = torch::empty({batch, num_tiles, nb, nb}, panel.options());
    int threads = 256;
    size_t shmem = static_cast<size_t>(rt) * nb * sizeof(float) + 2 * sizeof(float);
    cudaFuncSetAttribute(
        local_tile_householder_r_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem));
    dim3 grid(num_tiles, batch);
    local_tile_householder_r_kernel<<<grid, threads, shmem, at::cuda::getCurrentCUDAStream()>>>(
        panel.data_ptr<float>(), out.data_ptr<float>(), batch, rows, nb, rt, num_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> local_tile_householder_compact(torch::Tensor panel, int64_t row_tile) {
    TORCH_CHECK(panel.is_cuda(), "panel must be CUDA");
    TORCH_CHECK(panel.dtype() == torch::kFloat32, "panel must be float32");
    TORCH_CHECK(panel.dim() == 3, "panel must have shape (batch, rows, nb)");
    TORCH_CHECK(panel.is_contiguous(), "panel must be contiguous");
    int batch = static_cast<int>(panel.size(0));
    int rows = static_cast<int>(panel.size(1));
    int nb = static_cast<int>(panel.size(2));
    int rt = static_cast<int>(row_tile);
    TORCH_CHECK(nb > 0 && nb <= 64, "nb must be in 1..64 for this proof");
    TORCH_CHECK(rt >= nb, "row_tile must be >= nb");
    int num_tiles = (rows + rt - 1) / rt;
    auto h_tiles = torch::empty({batch, num_tiles, rt, nb}, panel.options());
    auto tau_tiles = torch::empty({batch, num_tiles, nb}, panel.options());
    auto r_blocks = torch::empty({batch, num_tiles, nb, nb}, panel.options());
    int threads = 256;
    size_t shmem = static_cast<size_t>(rt) * nb * sizeof(float)
        + static_cast<size_t>(nb) * sizeof(float)
        + 2 * sizeof(float);
    cudaFuncSetAttribute(
        local_tile_householder_compact_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem));
    dim3 grid(num_tiles, batch);
    local_tile_householder_compact_kernel<<<grid, threads, shmem, at::cuda::getCurrentCUDAStream()>>>(
        panel.data_ptr<float>(),
        h_tiles.data_ptr<float>(),
        tau_tiles.data_ptr<float>(),
        r_blocks.data_ptr<float>(),
        batch,
        rows,
        nb,
        rt,
        num_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {h_tiles, tau_tiles, r_blocks};
}

torch::Tensor stacked_householder_r(torch::Tensor r_blocks) {
    TORCH_CHECK(r_blocks.is_cuda(), "r_blocks must be CUDA");
    TORCH_CHECK(r_blocks.dtype() == torch::kFloat32, "r_blocks must be float32");
    TORCH_CHECK(r_blocks.dim() == 4, "r_blocks must have shape (batch, num_tiles, nb, nb)");
    TORCH_CHECK(r_blocks.is_contiguous(), "r_blocks must be contiguous");
    int batch = static_cast<int>(r_blocks.size(0));
    int num_tiles = static_cast<int>(r_blocks.size(1));
    int nb = static_cast<int>(r_blocks.size(2));
    TORCH_CHECK(r_blocks.size(3) == nb, "last two dims must be nb x nb");
    TORCH_CHECK(nb > 0 && nb <= 64, "nb must be in 1..64 for this proof");
    int stack_rows = num_tiles * nb;
    size_t shmem = static_cast<size_t>(stack_rows) * nb * sizeof(float) + 2 * sizeof(float);
    TORCH_CHECK(shmem <= 98304, "stacked-R proof exceeds 96KB dynamic shared memory; reduce nb or num_tiles");
    auto out = torch::empty({batch, nb, nb}, r_blocks.options());
    int threads = 256;
    cudaFuncSetAttribute(
        stacked_householder_r_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem));
    stacked_householder_r_kernel<<<batch, threads, shmem, at::cuda::getCurrentCUDAStream()>>>(
        r_blocks.data_ptr<float>(), out.data_ptr<float>(), batch, num_tiles, nb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> reconstruct_wy_lu(torch::Tensor q, int64_t k_arg) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(q.dtype() == torch::kFloat32, "q must be float32");
    TORCH_CHECK(q.dim() == 3, "q must have shape (batch, rows, q_cols)");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    int batch = static_cast<int>(q.size(0));
    int rows = static_cast<int>(q.size(1));
    int q_cols = static_cast<int>(q.size(2));
    int k = static_cast<int>(k_arg);
    TORCH_CHECK(k > 0 && k <= 64, "k must be in 1..64");
    TORCH_CHECK(k <= rows, "k must be <= rows");
    TORCH_CHECK(q_cols >= k, "q must contain at least k columns");
    auto w = torch::empty({batch, rows, k}, q.options());
    auto y = torch::empty({batch, rows, k}, q.options());
    auto signs = torch::empty({batch, k}, q.options());
    int threads = 256;
    size_t shmem = static_cast<size_t>(2 * k * k + k) * sizeof(float);
    reconstruct_wy_lu_kernel<<<batch, threads, shmem, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<float>(),
        w.data_ptr<float>(),
        y.data_ptr<float>(),
        signs.data_ptr<float>(),
        batch,
        rows,
        q_cols,
        k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {w, y, signs};
}

std::vector<torch::Tensor> reconstruct_wy_lu_from_tree(torch::Tensor h_tiles, torch::Tensor tau_tiles, torch::Tensor basis, int64_t k_arg) {
    TORCH_CHECK(h_tiles.is_cuda() && tau_tiles.is_cuda() && basis.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(h_tiles.dtype() == torch::kFloat32 && tau_tiles.dtype() == torch::kFloat32 && basis.dtype() == torch::kFloat32, "all tensors must be float32");
    TORCH_CHECK(h_tiles.dim() == 4, "h_tiles must have shape (batch, num_tiles, row_tile, k)");
    TORCH_CHECK(tau_tiles.dim() == 3, "tau_tiles must have shape (batch, num_tiles, k)");
    TORCH_CHECK(basis.dim() == 3, "basis must have shape (batch, rows, k)");
    TORCH_CHECK(h_tiles.is_contiguous() && tau_tiles.is_contiguous() && basis.is_contiguous(), "inputs must be contiguous");
    int batch = static_cast<int>(basis.size(0));
    int rows = static_cast<int>(basis.size(1));
    int k = static_cast<int>(k_arg);
    int num_tiles = static_cast<int>(h_tiles.size(1));
    int row_tile = static_cast<int>(h_tiles.size(2));
    TORCH_CHECK(k > 0 && k <= 64, "k must be in 1..64");
    TORCH_CHECK(h_tiles.size(0) == batch && h_tiles.size(3) == k, "h_tiles shape mismatch");
    TORCH_CHECK(tau_tiles.size(0) == batch && tau_tiles.size(1) == num_tiles && tau_tiles.size(2) == k, "tau_tiles shape mismatch");
    TORCH_CHECK(basis.size(2) == k, "basis k mismatch");
    auto w = torch::empty({batch, rows, k}, basis.options());
    auto y = torch::empty({batch, rows, k}, basis.options());
    auto signs = torch::empty({batch, k}, basis.options());
    auto lu = torch::empty({batch, 2, k, k}, basis.options());
    int threads = 256;
    size_t shmem = static_cast<size_t>(row_tile) * k * sizeof(float)
        + static_cast<size_t>(k) * sizeof(float)
        + static_cast<size_t>(row_tile) * k * sizeof(float)
        + static_cast<size_t>(k) * sizeof(float)
        + static_cast<size_t>(2 * k * k) * sizeof(float)
        + static_cast<size_t>(k) * sizeof(float);
    cudaFuncSetAttribute(
        tree_wy_lu_prepare_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem));
    cudaFuncSetAttribute(
        tree_wy_lu_apply_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem));
    tree_wy_lu_prepare_kernel<<<batch, threads, shmem, at::cuda::getCurrentCUDAStream()>>>(
        h_tiles.data_ptr<float>(),
        tau_tiles.data_ptr<float>(),
        basis.data_ptr<float>(),
        lu.data_ptr<float>(),
        signs.data_ptr<float>(),
        batch,
        rows,
        k,
        row_tile,
        num_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    dim3 grid(num_tiles, batch);
    tree_wy_lu_apply_kernel<<<grid, threads, shmem, at::cuda::getCurrentCUDAStream()>>>(
        h_tiles.data_ptr<float>(),
        tau_tiles.data_ptr<float>(),
        basis.data_ptr<float>(),
        lu.data_ptr<float>(),
        w.data_ptr<float>(),
        y.data_ptr<float>(),
        batch,
        rows,
        k,
        row_tile,
        num_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {w, y, signs};
}

torch::Tensor local_compact_apply_qt(torch::Tensor h_tiles, torch::Tensor tau_tiles, torch::Tensor c, int64_t block_n) {
    TORCH_CHECK(h_tiles.is_cuda() && tau_tiles.is_cuda() && c.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(h_tiles.dtype() == torch::kFloat32 && tau_tiles.dtype() == torch::kFloat32 && c.dtype() == torch::kFloat32, "all tensors must be float32");
    TORCH_CHECK(h_tiles.dim() == 4, "h_tiles must have shape (batch, num_tiles, row_tile, nb)");
    TORCH_CHECK(tau_tiles.dim() == 3, "tau_tiles must have shape (batch, num_tiles, nb)");
    TORCH_CHECK(c.dim() == 3, "c must have shape (batch, rows, cols)");
    TORCH_CHECK(h_tiles.is_contiguous() && tau_tiles.is_contiguous() && c.is_contiguous(), "inputs must be contiguous");
    int batch = static_cast<int>(c.size(0));
    int rows = static_cast<int>(c.size(1));
    int cols = static_cast<int>(c.size(2));
    int num_tiles = static_cast<int>(h_tiles.size(1));
    int row_tile = static_cast<int>(h_tiles.size(2));
    int nb = static_cast<int>(h_tiles.size(3));
    int bn = static_cast<int>(block_n);
    TORCH_CHECK(h_tiles.size(0) == batch, "h_tiles batch mismatch");
    TORCH_CHECK(tau_tiles.size(0) == batch && tau_tiles.size(1) == num_tiles && tau_tiles.size(2) == nb, "tau_tiles shape mismatch");
    TORCH_CHECK(nb > 0 && nb <= 64, "nb must be in 1..64");
    TORCH_CHECK(bn > 0 && bn <= 64, "block_n must be in 1..64");
    auto out = torch::empty_like(c);
    int threads = 256;
    size_t shmem = static_cast<size_t>(row_tile) * nb * sizeof(float)
        + static_cast<size_t>(row_tile) * bn * sizeof(float)
        + static_cast<size_t>(bn) * sizeof(float);
    cudaFuncSetAttribute(
        local_compact_apply_qt_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem));
    dim3 grid((cols + bn - 1) / bn, num_tiles, batch);
    local_compact_apply_qt_kernel<<<grid, threads, shmem, at::cuda::getCurrentCUDAStream()>>>(
        h_tiles.data_ptr<float>(),
        tau_tiles.data_ptr<float>(),
        c.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        rows,
        cols,
        nb,
        row_tile,
        num_tiles,
        bn);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor local_compact_apply_q(torch::Tensor h_tiles, torch::Tensor tau_tiles, torch::Tensor c, int64_t block_n) {
    TORCH_CHECK(h_tiles.is_cuda() && tau_tiles.is_cuda() && c.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(h_tiles.dtype() == torch::kFloat32 && tau_tiles.dtype() == torch::kFloat32 && c.dtype() == torch::kFloat32, "all tensors must be float32");
    TORCH_CHECK(h_tiles.dim() == 4, "h_tiles must have shape (batch, num_tiles, row_tile, nb)");
    TORCH_CHECK(tau_tiles.dim() == 3, "tau_tiles must have shape (batch, num_tiles, nb)");
    TORCH_CHECK(c.dim() == 3, "c must have shape (batch, rows, cols)");
    TORCH_CHECK(h_tiles.is_contiguous() && tau_tiles.is_contiguous() && c.is_contiguous(), "inputs must be contiguous");
    int batch = static_cast<int>(c.size(0));
    int rows = static_cast<int>(c.size(1));
    int cols = static_cast<int>(c.size(2));
    int num_tiles = static_cast<int>(h_tiles.size(1));
    int row_tile = static_cast<int>(h_tiles.size(2));
    int nb = static_cast<int>(h_tiles.size(3));
    int bn = static_cast<int>(block_n);
    TORCH_CHECK(h_tiles.size(0) == batch, "h_tiles batch mismatch");
    TORCH_CHECK(tau_tiles.size(0) == batch && tau_tiles.size(1) == num_tiles && tau_tiles.size(2) == nb, "tau_tiles shape mismatch");
    TORCH_CHECK(nb > 0 && nb <= 64, "nb must be in 1..64");
    TORCH_CHECK(bn > 0 && bn <= 64, "block_n must be in 1..64");
    auto out = torch::empty_like(c);
    int threads = 256;
    size_t shmem = static_cast<size_t>(row_tile) * nb * sizeof(float)
        + static_cast<size_t>(row_tile) * bn * sizeof(float)
        + static_cast<size_t>(bn) * sizeof(float);
    cudaFuncSetAttribute(
        local_compact_apply_q_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem));
    dim3 grid((cols + bn - 1) / bn, num_tiles, batch);
    local_compact_apply_q_kernel<<<grid, threads, shmem, at::cuda::getCurrentCUDAStream()>>>(
        h_tiles.data_ptr<float>(),
        tau_tiles.data_ptr<float>(),
        c.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        rows,
        cols,
        nb,
        row_tile,
        num_tiles,
        bn);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""


CPP_SRC = r"""
#include <torch/extension.h>

torch::Tensor local_tile_householder_r(torch::Tensor panel, int64_t row_tile);
std::vector<torch::Tensor> local_tile_householder_compact(torch::Tensor panel, int64_t row_tile);
torch::Tensor stacked_householder_r(torch::Tensor r_blocks);
torch::Tensor local_compact_apply_qt(torch::Tensor h_tiles, torch::Tensor tau_tiles, torch::Tensor c, int64_t block_n);
torch::Tensor local_compact_apply_q(torch::Tensor h_tiles, torch::Tensor tau_tiles, torch::Tensor c, int64_t block_n);
std::vector<torch::Tensor> reconstruct_wy_lu(torch::Tensor q, int64_t k);
std::vector<torch::Tensor> reconstruct_wy_lu_from_tree(torch::Tensor h_tiles, torch::Tensor tau_tiles, torch::Tensor basis, int64_t k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("local_tile_householder_r", &local_tile_householder_r, "Local tile Householder R proof");
    m.def("local_tile_householder_compact", &local_tile_householder_compact, "Local tile compact Householder proof");
    m.def("stacked_householder_r", &stacked_householder_r, "Stacked-R Householder R proof");
    m.def("local_compact_apply_qt", &local_compact_apply_qt, "Apply local compact tile reflectors");
    m.def("local_compact_apply_q", &local_compact_apply_q, "Apply local compact tile reflectors forward");
    m.def("reconstruct_wy_lu", &reconstruct_wy_lu, "Paper-style explicit-Q to WY reconstruction");
    m.def("reconstruct_wy_lu_from_tree", &reconstruct_wy_lu_from_tree, "Paper-style WY reconstruction directly from compact tree basis");
}
"""


@dataclass(frozen=True)
class Timing:
    median_ms: float
    min_ms: float
    max_ms: float


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_ext():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="qr_cuda_tsqr_panel_proof_ext_v13",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


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


def align_r_signs(r: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    diag_r = torch.diagonal(r, dim1=-2, dim2=-1)
    diag_ref = torch.diagonal(ref, dim1=-2, dim2=-1)
    signs = torch.sign(diag_r * diag_ref)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return signs.unsqueeze(-1) * r


def rel_r_error(r: torch.Tensor, ref: torch.Tensor) -> float:
    aligned = align_r_signs(r, ref)
    num = torch.linalg.matrix_norm((aligned - ref).double(), ord=1, dim=(-2, -1)).amax()
    den = torch.linalg.matrix_norm(ref.double(), ord=1, dim=(-2, -1)).amax().clamp_min(1e-30)
    return (num / den).item()


def validate_local_compact(
    panel: torch.Tensor,
    h_tiles: torch.Tensor,
    tau_tiles: torch.Tensor,
    r_blocks: torch.Tensor,
    row_tile: int,
) -> tuple[float, float, float]:
    """Validate tile-local compact Householder storage against the input panel."""

    batch, rows, nb = panel.shape
    num_tiles = h_tiles.shape[1]
    worst_recon = torch.zeros((), device=panel.device, dtype=torch.float64)
    worst_orth = torch.zeros((), device=panel.device, dtype=torch.float64)
    worst_lower = torch.zeros((), device=panel.device, dtype=torch.float64)
    epsn = torch.finfo(torch.float32).eps * max(nb, 1)
    for b in range(batch):
        for tile in range(num_tiles):
            row0 = tile * row_tile
            row1 = min(row0 + row_tile, rows)
            size = row1 - row0
            h_tile = h_tiles[b : b + 1, tile, :size, :].contiguous()
            tau_tile = tau_tiles[b : b + 1, tile, :].contiguous()
            q_tile = torch.linalg.householder_product(h_tile, tau_tile)
            tile_panel = panel[b : b + 1, row0:row1, :]
            projected = torch.bmm(q_tile.double().transpose(1, 2), tile_panel.double())
            r_ref = projected[:, :nb, :]
            r_cuda = r_blocks[b : b + 1, tile, :, :].double()
            scale = torch.linalg.matrix_norm(tile_panel.double(), ord=1, dim=(-2, -1)).amax().clamp_min(1e-30)
            recon = torch.linalg.matrix_norm(r_cuda - r_ref, ord=1, dim=(-2, -1)).amax() / (epsn * scale)
            lower = torch.linalg.matrix_norm(torch.tril(projected, diagonal=-1), ord=1, dim=(-2, -1)).amax() / (epsn * scale)
            eye = torch.eye(nb, device=panel.device, dtype=torch.float64).unsqueeze(0)
            orth = torch.linalg.matrix_norm(
                torch.bmm(q_tile.double().transpose(1, 2), q_tile.double()) - eye,
                ord=1,
                dim=(-2, -1),
            ).amax() / torch.finfo(torch.float32).eps
            worst_recon = torch.maximum(worst_recon, recon)
            worst_lower = torch.maximum(worst_lower, lower)
            worst_orth = torch.maximum(worst_orth, orth)
    return worst_recon.item(), worst_lower.item(), worst_orth.item()


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["NB", "STACK_ROWS", "BLOCK_ROWS"],
)
@triton.jit
def _stacked_householder_r_triton_kernel(
    r_blocks_ptr,
    out_ptr,
    NUM_TILES: tl.constexpr,
    NB: tl.constexpr,
    STACK_ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, NB)
    tile = rows // NB
    local_row = rows - tile * NB
    src = ((batch * NUM_TILES + tile[:, None]) * NB + local_row[:, None]) * NB + cols[None, :]
    panel = tl.load(
        r_blocks_ptr + src,
        mask=rows[:, None] < STACK_ROWS,
        other=0.0,
    )

    for local_j in tl.static_range(0, 64):
        if local_j < NB:
            is_col = cols == local_j
            col_vec = tl.sum(tl.where(is_col[None, :], panel, 0.0), axis=1)
            alpha = tl.sum(tl.where(rows == local_j, col_vec, 0.0), axis=0)
            active_rows_tail = (rows > local_j) & (rows < STACK_ROWS)
            tail = tl.where(active_rows_tail, col_vec, 0.0)
            sigma = tl.sum(tail * tail, axis=0)
            norm = tl.sqrt(alpha * alpha + sigma)
            beta = tl.where(alpha >= 0.0, -norm, norm)
            active = sigma > 0.0
            beta_safe = tl.where(active, beta, 1.0)
            denom_safe = tl.where(active, alpha - beta, 1.0)
            tau = tl.where(active, (beta - alpha) / beta_safe, 0.0)
            scale = tl.where(active, 1.0 / denom_safe, 0.0)
            v = tl.where(
                rows == local_j,
                1.0,
                tl.where(active_rows_tail, col_vec * scale, 0.0),
            )
            new_col = tl.where(
                rows == local_j,
                tl.where(active, beta, alpha),
                tl.where(active_rows_tail, col_vec * scale, col_vec),
            )
            panel = tl.where(is_col[None, :], new_col[:, None], panel)

            active_rows = (rows >= local_j) & (rows < STACK_ROWS)
            dots = tl.sum(tl.where(active_rows[:, None], v[:, None] * panel, 0.0), axis=0)
            update_cols = cols > local_j
            panel = tl.where(
                active_rows[:, None] & update_cols[None, :],
                panel - v[:, None] * (tau * dots)[None, :],
                panel,
            )

    out_vals = tl.where(rows[:, None] <= cols[None, :], panel, 0.0)
    tl.store(
        out_ptr + batch * NB * NB + rows[:, None] * NB + cols[None, :],
        out_vals,
        mask=rows[:, None] < NB,
    )


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def stacked_householder_r_triton(r_blocks: torch.Tensor) -> torch.Tensor:
    batch, num_tiles, nb, nb2 = r_blocks.shape
    if nb != nb2:
        raise ValueError("r_blocks must have shape (batch, num_tiles, nb, nb)")
    if nb > 64:
        raise ValueError("Triton stacked-R proof supports nb <= 64")
    stack_rows = num_tiles * nb
    block_rows = _next_power_of_2(stack_rows)
    out = torch.empty((batch, nb, nb), device=r_blocks.device, dtype=r_blocks.dtype)
    _stacked_householder_r_triton_kernel[(batch,)](
        r_blocks,
        out,
        NUM_TILES=num_tiles,
        NB=nb,
        STACK_ROWS=stack_rows,
        BLOCK_ROWS=block_rows,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["NB", "STACK_ROWS", "BLOCK_ROWS", "BLOCK_N"],
)
@triton.jit
def _top_compact_apply_qt_triton_kernel(
    h_ptr,
    tau_ptr,
    c_ptr,
    out_ptr,
    cols: tl.constexpr,
    NB: tl.constexpr,
    STACK_ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    col_tile = tl.program_id(1)
    rows = tl.arange(0, BLOCK_ROWS)
    col_offsets = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    c = tl.load(
        c_ptr + batch * STACK_ROWS * cols + rows[:, None] * cols + col_offsets[None, :],
        mask=(rows[:, None] < STACK_ROWS) & (col_offsets[None, :] < cols),
        other=0.0,
    )

    for j in tl.static_range(0, 64):
        if j < NB:
            h_col = tl.load(
                h_ptr + batch * STACK_ROWS * NB + rows * NB + j,
                mask=rows < STACK_ROWS,
                other=0.0,
            )
            v = tl.where(rows == j, 1.0, tl.where(rows > j, h_col, 0.0))
            active = (rows >= j) & (rows < STACK_ROWS)
            dot = tl.sum(tl.where(active[:, None], v[:, None] * c, 0.0), axis=0)
            tau = tl.load(tau_ptr + batch * NB + j)
            c = tl.where(active[:, None], c - v[:, None] * (tau * dot)[None, :], c)

    tl.store(
        out_ptr + batch * STACK_ROWS * cols + rows[:, None] * cols + col_offsets[None, :],
        c,
        mask=(rows[:, None] < STACK_ROWS) & (col_offsets[None, :] < cols),
    )


def top_compact_apply_qt_triton(h_top: torch.Tensor, tau_top: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
    batch, stack_rows, nb = h_top.shape
    if tau_top.shape != (batch, nb):
        raise ValueError("tau_top must have shape (batch, nb)")
    if coord.shape[0] != batch or coord.shape[1] != stack_rows:
        raise ValueError("coord must have shape (batch, stack_rows, cols)")
    block_rows = _next_power_of_2(stack_rows)
    out = torch.empty_like(coord)
    grid = (batch, triton.cdiv(coord.shape[2], 16))
    _top_compact_apply_qt_triton_kernel[grid](
        h_top.contiguous(),
        tau_top.contiguous(),
        coord.contiguous(),
        out,
        cols=coord.shape[2],
        NB=nb,
        STACK_ROWS=stack_rows,
        BLOCK_ROWS=block_rows,
        BLOCK_N=16,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["NB", "STACK_ROWS", "BLOCK_ROWS", "BLOCK_N"],
)
@triton.jit
def _top_compact_apply_q_triton_kernel(
    h_ptr,
    tau_ptr,
    c_ptr,
    out_ptr,
    cols: tl.constexpr,
    NB: tl.constexpr,
    STACK_ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    col_tile = tl.program_id(1)
    rows = tl.arange(0, BLOCK_ROWS)
    col_offsets = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    c = tl.load(
        c_ptr + batch * STACK_ROWS * cols + rows[:, None] * cols + col_offsets[None, :],
        mask=(rows[:, None] < STACK_ROWS) & (col_offsets[None, :] < cols),
        other=0.0,
    )

    for jj in tl.static_range(0, 64):
        j = NB - 1 - jj
        if j >= 0:
            h_col = tl.load(
                h_ptr + batch * STACK_ROWS * NB + rows * NB + j,
                mask=rows < STACK_ROWS,
                other=0.0,
            )
            v = tl.where(rows == j, 1.0, tl.where(rows > j, h_col, 0.0))
            active = (rows >= j) & (rows < STACK_ROWS)
            dot = tl.sum(tl.where(active[:, None], v[:, None] * c, 0.0), axis=0)
            tau = tl.load(tau_ptr + batch * NB + j)
            c = tl.where(active[:, None], c - v[:, None] * (tau * dot)[None, :], c)

    tl.store(
        out_ptr + batch * STACK_ROWS * cols + rows[:, None] * cols + col_offsets[None, :],
        c,
        mask=(rows[:, None] < STACK_ROWS) & (col_offsets[None, :] < cols),
    )


def top_compact_apply_q_triton(h_top: torch.Tensor, tau_top: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
    batch, stack_rows, nb = h_top.shape
    if tau_top.shape != (batch, nb):
        raise ValueError("tau_top must have shape (batch, nb)")
    if coord.shape[0] != batch or coord.shape[1] != stack_rows:
        raise ValueError("coord must have shape (batch, stack_rows, cols)")
    block_rows = _next_power_of_2(stack_rows)
    out = torch.empty_like(coord)
    grid = (batch, triton.cdiv(coord.shape[2], 16))
    _top_compact_apply_q_triton_kernel[grid](
        h_top.contiguous(),
        tau_top.contiguous(),
        coord.contiguous(),
        out,
        cols=coord.shape[2],
        NB=nb,
        STACK_ROWS=stack_rows,
        BLOCK_ROWS=block_rows,
        BLOCK_N=16,
    )
    return out


def torch_tsqr_panel_r(panel: torch.Tensor, row_tile: int) -> torch.Tensor:
    batch, rows, nb = panel.shape
    blocks = []
    for row0 in range(0, rows, row_tile):
        row1 = min(row0 + row_tile, rows)
        _, r = torch.linalg.qr(panel[:, row0:row1, :], mode="reduced")
        blocks.append(r[:, :nb, :])
    stacked = torch.cat(blocks, dim=1)
    _, r_final = torch.linalg.qr(stacked, mode="reduced")
    return r_final[:, :nb, :]


def cuda_tsqr_panel_r_torch_stack(ext, panel: torch.Tensor, row_tile: int) -> torch.Tensor:
    batch, _, nb = panel.shape
    r_blocks = ext.local_tile_householder_r(panel, row_tile)
    stacked = r_blocks.reshape(batch, -1, nb)
    _, r_final = torch.linalg.qr(stacked, mode="reduced")
    return r_final[:, :nb, :]


def cuda_tsqr_panel_r_full_cuda(ext, panel: torch.Tensor, row_tile: int) -> torch.Tensor:
    r_blocks = ext.local_tile_householder_r(panel, row_tile)
    return ext.stacked_householder_r(r_blocks)


def cuda_tsqr_panel_r_triton_stack(ext, panel: torch.Tensor, row_tile: int) -> torch.Tensor:
    r_blocks = ext.local_tile_householder_r(panel, row_tile)
    return stacked_householder_r_triton(r_blocks)


def run(args: argparse.Namespace) -> None:
    if args.cuda_home:
        os.environ["CUDA_HOME"] = args.cuda_home
    ext = load_ext()
    data = qr_official.generate_input(args.batch, args.n, args.cond, args.seed, args.case)
    panel = data[:, :, : args.nb].contiguous()
    print(
        f"case n={args.n} batch={args.batch} case={args.case} nb={args.nb} "
        f"row_tile={args.row_tile} device={panel.device}",
        flush=True,
    )

    local_time, r_blocks = time_fn(
        lambda: ext.local_tile_householder_r(panel, args.row_tile),
        args.warmup,
        args.trials,
    )
    compact_time, compact_out = time_fn(
        lambda: ext.local_tile_householder_compact(panel, args.row_tile),
        args.warmup,
        args.trials,
    )
    h_tiles, tau_tiles, r_blocks_compact = compact_out
    compact_recon, compact_lower, compact_orth = validate_local_compact(
        panel,
        h_tiles,
        tau_tiles,
        r_blocks_compact,
        args.row_tile,
    )
    r_blocks_diff = rel_r_error(
        r_blocks_compact.reshape(-1, args.nb, args.nb),
        r_blocks.reshape(-1, args.nb, args.nb),
    )
    stacked = r_blocks.reshape(args.batch, -1, args.nb)
    stacked_time, r_cuda = time_fn(
        lambda: torch.linalg.qr(stacked, mode="reduced")[1][:, : args.nb, :],
        args.warmup,
        args.trials,
    )
    stacked_cuda_time, r_cuda_stack = time_fn(
        lambda: ext.stacked_householder_r(r_blocks),
        args.warmup,
        args.trials,
    )
    stacked_triton_time, r_triton_stack = time_fn(
        lambda: stacked_householder_r_triton(r_blocks),
        args.warmup,
        args.trials,
    )
    total_time, r_total = time_fn(
        lambda: cuda_tsqr_panel_r_torch_stack(ext, panel, args.row_tile),
        args.warmup,
        args.trials,
    )
    total_cuda_time, r_total_cuda = time_fn(
        lambda: cuda_tsqr_panel_r_full_cuda(ext, panel, args.row_tile),
        args.warmup,
        args.trials,
    )
    total_triton_time, r_total_triton = time_fn(
        lambda: cuda_tsqr_panel_r_triton_stack(ext, panel, args.row_tile),
        args.warmup,
        args.trials,
    )
    torch_tsqr_time, r_torch_tsqr = time_fn(
        lambda: torch_tsqr_panel_r(panel, args.row_tile),
        args.warmup,
        args.trials,
    )
    torch_panel_time, r_ref = time_fn(
        lambda: torch.linalg.qr(panel, mode="reduced")[1],
        args.warmup,
        args.trials,
    )
    err_total = rel_r_error(r_total, r_ref)
    err_total_cuda = rel_r_error(r_total_cuda, r_ref)
    err_total_triton = rel_r_error(r_total_triton, r_ref)
    err_cuda_stack_vs_torch = rel_r_error(r_cuda_stack, r_cuda)
    err_triton_stack_vs_torch = rel_r_error(r_triton_stack, r_cuda)
    err_cuda_vs_torch_tsqr = rel_r_error(r_cuda, r_torch_tsqr)
    print(
        f"cuda_local_tile_r   median={local_time.median_ms:.3f} ms "
        f"min={local_time.min_ms:.3f} max={local_time.max_ms:.3f}",
        flush=True,
    )
    print(
        f"cuda_local_compact  median={compact_time.median_ms:.3f} ms "
        f"min={compact_time.min_ms:.3f} max={compact_time.max_ms:.3f} "
        f"tile_scaled_recon={compact_recon:.3g} tile_scaled_lower={compact_lower:.3g} "
        f"tile_scaled_orth={compact_orth:.3g} r_blocks_diff={r_blocks_diff:.3e}",
        flush=True,
    )
    print(
        f"stacked_r_torch_qr  median={stacked_time.median_ms:.3f} ms "
        f"min={stacked_time.min_ms:.3f} max={stacked_time.max_ms:.3f}",
        flush=True,
    )
    print(
        f"stacked_r_cuda_qr   median={stacked_cuda_time.median_ms:.3f} ms "
        f"min={stacked_cuda_time.min_ms:.3f} max={stacked_cuda_time.max_ms:.3f} "
        f"cuda_stack_vs_torch_err={err_cuda_stack_vs_torch:.3e}",
        flush=True,
    )
    print(
        f"stacked_r_triton_qr median={stacked_triton_time.median_ms:.3f} ms "
        f"min={stacked_triton_time.min_ms:.3f} max={stacked_triton_time.max_ms:.3f} "
        f"triton_stack_vs_torch_err={err_triton_stack_vs_torch:.3e}",
        flush=True,
    )
    print(
        f"cuda_tsqr_torchstk  median={total_time.median_ms:.3f} ms "
        f"min={total_time.min_ms:.3f} max={total_time.max_ms:.3f} "
        f"rel_r_err={err_total:.3e}",
        flush=True,
    )
    print(
        f"cuda_tsqr_fullcuda  median={total_cuda_time.median_ms:.3f} ms "
        f"min={total_cuda_time.min_ms:.3f} max={total_cuda_time.max_ms:.3f} "
        f"rel_r_err={err_total_cuda:.3e}",
        flush=True,
    )
    print(
        f"cuda_tsqr_tritonstk median={total_triton_time.median_ms:.3f} ms "
        f"min={total_triton_time.min_ms:.3f} max={total_triton_time.max_ms:.3f} "
        f"rel_r_err={err_total_triton:.3e}",
        flush=True,
    )
    print(
        f"torch_tsqr_total    median={torch_tsqr_time.median_ms:.3f} ms "
        f"min={torch_tsqr_time.min_ms:.3f} max={torch_tsqr_time.max_ms:.3f} "
        f"cuda_vs_torch_tsqr_r_err={err_cuda_vs_torch_tsqr:.3e}",
        flush=True,
    )
    print(
        f"torch_panel_qr      median={torch_panel_time.median_ms:.3f} ms "
        f"min={torch_panel_time.min_ms:.3f} max={torch_panel_time.max_ms:.3f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--case", default="dense")
    parser.add_argument("--cond", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--nb", type=int, default=32)
    parser.add_argument("--row-tile", type=int, default=512)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cuda-home", default="")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
