// Raw CUDA small-square batched QR prototype.
//
// Contract:
//   - float32, row-major contiguous input/output.
//   - A/H shape is (batch_count, n, n), tau shape is (batch_count, n).
//   - n must be 1..64.
//   - Output matches LAPACK/torch.geqrf compact Householder storage:
//       upper triangle of H is R, strict lower triangle stores reflector tails,
//       tau stores reflector scales, and every reflector has implicit v[j] = 1.
//
// The raw C ABI is independent from PyTorch.  Define
// SMALL_SQUARE_QR_TORCH_EXTENSION to also compile the pybind wrapper below.

#include <cuda_runtime.h>

#include <stdint.h>

#ifdef SMALL_SQUARE_QR_TORCH_EXTENSION
#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>

#include <vector>
#endif

namespace small_square_qr {

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

  if (tid < 64) {
    scratch[tid] += scratch[tid + 64];
  }
  __syncthreads();
  if (tid < 32) {
    scratch[tid] += scratch[tid + 32];
  }
  __syncthreads();
  if (tid < 16) {
    scratch[tid] += scratch[tid + 16];
  }
  __syncthreads();
  if (tid < 8) {
    scratch[tid] += scratch[tid + 8];
  }
  __syncthreads();
  if (tid < 4) {
    scratch[tid] += scratch[tid + 4];
  }
  __syncthreads();
  if (tid < 2) {
    scratch[tid] += scratch[tid + 2];
  }
  __syncthreads();
  if (tid == 0) {
    scratch[0] += scratch[1];
  }
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

  // Shared-memory layout, all float32:
  //   s_a       : kMaxN * kMaxN matrix tile, row-major.
  //   s_tau     : kMaxN compact Householder scalar outputs.
  //   s_reduce  : kThreads reduction scratch.
  //   s_scalars : [scale, tau_j, dot_w].
  //
  // Fixed max-sized regions keep the pointer math independent from runtime n
  // and make the launch ABI simple.  At n <= 64 this is about 17 KiB per CTA.
  float* s_a = smem;
  float* s_tau = s_a + kMaxN * kMaxN;
  float* s_reduce = s_tau + kMaxN;
  float* s_scalars = s_reduce + kThreads;

  const int tid = threadIdx.x;
  const int batch = blockIdx.x;
  if (batch >= batch_count) {
    return;
  }

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

    // Panel-internal apply:
    //   A[:, col] <- (I - tau_j * v * v^T) A[:, col], col > j.
    //
    // One target column is processed at a time.  This deliberately favors a
    // compact first raw-CUDA baseline over maximum parallelism; the next step is
    // splitting target columns/row tiles across CTAs once benchmark data says
    // where the pressure lands.
    for (int col = j + 1; col < n; ++col) {
      float local_dot = 0.0f;
      for (int row = j + tid; row < n; row += blockDim.x) {
        const float v = (row == j) ? 1.0f : s_a[row * n + j];
        local_dot += v * s_a[row * n + col];
      }
      const float dot = block_sum(local_dot, s_reduce);

      if (tid == 0) {
        s_scalars[2] = tau_j * dot;
      }
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

}  // namespace small_square_qr

extern "C" cudaError_t small_square_qr_cuda(
    const float* a,
    float* h,
    float* tau,
    int batch_count,
    int n) {
  if (a == nullptr || h == nullptr || tau == nullptr) {
    return cudaErrorInvalidDevicePointer;
  }
  if (batch_count < 0 || n <= 0 || n > small_square_qr::kMaxN) {
    return cudaErrorInvalidValue;
  }
  if (batch_count == 0) {
    return cudaSuccess;
  }

  const auto cfg = small_square_qr::make_launch_config(batch_count);
  small_square_qr::small_square_qr_kernel<<<
      cfg.grid,
      cfg.block,
      cfg.shared_bytes>>>(a, h, tau, batch_count, n);
  return cudaGetLastError();
}

#ifdef SMALL_SQUARE_QR_TORCH_EXTENSION
namespace {

void check_torch_input(const torch::Tensor& a) {
  TORCH_CHECK(a.is_cuda(), "small_square_qr expects a CUDA tensor");
  TORCH_CHECK(a.dtype() == torch::kFloat32, "small_square_qr expects float32 input");
  TORCH_CHECK(a.dim() == 3, "small_square_qr expects shape (batch, n, n)");
  TORCH_CHECK(a.size(1) == a.size(2), "small_square_qr expects square matrices");
  TORCH_CHECK(a.size(1) > 0 && a.size(1) <= 64, "small_square_qr supports 1 <= n <= 64");
  TORCH_CHECK(a.is_contiguous(), "small_square_qr expects contiguous row-major input");
}

}  // namespace

std::vector<torch::Tensor> small_square_qr_torch(torch::Tensor a) {
  check_torch_input(a);

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("small_square_qr", &small_square_qr_torch, "Raw CUDA small-square QR");
}
#endif
