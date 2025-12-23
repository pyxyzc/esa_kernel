#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <type_traits>
#include <math.h>

template <typename T>
__device__ __forceinline__ bool eq_eps(T a, T b, double eps) {
    if constexpr (std::is_floating_point<T>::value) {
        return fabs((double)a - (double)b) <= eps;
    } else {
        return a == b;
    }
}

template <typename scalar_t>
__global__ void mark_matches_kernel(
    const scalar_t* __restrict__ A, int64_t NA,
    const scalar_t* __restrict__ B, int64_t NB,
    double eps,
    uint8_t* __restrict__ matchA)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < NA; i += stride) {
        scalar_t a = A[i];
        uint8_t matched = 0;
        for (int64_t j = 0; j < NB; ++j) {
            if (eq_eps<scalar_t>(a, B[j], eps)) {
                matched = 1;
                break;
            }
        }
        matchA[i] = matched;
    }
}

struct is_zero {
    __host__ __device__ bool operator()(const uint8_t x) const { return x == 0; }
};

std::tuple<at::Tensor, at::Tensor> diff_two_map_cuda(
    at::Tensor keys,
    at::Tensor old_values,
    at::Tensor new_values,
    double eps)
{
    TORCH_CHECK(keys.is_cuda(), "keys must be a CUDA tensor");
    TORCH_CHECK(old_values.is_cuda(), "old_values must be a CUDA tensor");
    TORCH_CHECK(new_values.is_cuda(), "new_values must be a CUDA tensor");
    TORCH_CHECK(keys.dtype() == at::kLong, "keys must be int64 (torch.long)");
    TORCH_CHECK(old_values.scalar_type() == new_values.scalar_type(),
                "old_values and new_values must have the same dtype");
    TORCH_CHECK(old_values.dim() == 1 && new_values.dim() == 1 && keys.dim() == 1,
                "keys, old_values, and new_values must be 1D");
    TORCH_CHECK(keys.size(0) == old_values.size(0),
                "keys and old_values must have the same length");

    auto stream = at::cuda::getCurrentCUDAStream();

    keys = keys.contiguous();
    old_values = old_values.contiguous();
    new_values = new_values.contiguous();

    const int64_t N = old_values.size(0);
    const int64_t M = new_values.size(0);

    auto byte_opts = old_values.options().dtype(at::kByte);
    at::Tensor old_match = at::empty({N}, byte_opts);
    at::Tensor new_match = at::empty({M}, byte_opts);

    const int threads = 256;
    const int blocks_old = std::min<int64_t>( (N + threads - 1) / threads, 4096 );
    const int blocks_new = std::min<int64_t>( (M + threads - 1) / threads, 4096 );

    // Launch marking kernels
    switch (old_values.scalar_type()) {
        case at::kFloat: {
            const float* old_p = old_values.data_ptr<float>();
            const float* new_p = new_values.data_ptr<float>();
            uint8_t* old_m = old_match.data_ptr<uint8_t>();
            uint8_t* new_m = new_match.data_ptr<uint8_t>();
            mark_matches_kernel<float><<<blocks_old, threads, 0, stream>>>(
                old_p, N, new_p, M, eps, old_m);
            mark_matches_kernel<float><<<blocks_new, threads, 0, stream>>>(
                new_p, M, old_p, N, eps, new_m);
            break;
        }
        case at::kDouble: {
            const double* old_p = old_values.data_ptr<double>();
            const double* new_p = new_values.data_ptr<double>();
            uint8_t* old_m = old_match.data_ptr<uint8_t>();
            uint8_t* new_m = new_match.data_ptr<uint8_t>();
            mark_matches_kernel<double><<<blocks_old, threads, 0, stream>>>(
                old_p, N, new_p, M, eps, old_m);
            mark_matches_kernel<double><<<blocks_new, threads, 0, stream>>>(
                new_p, M, old_p, N, eps, new_m);
            break;
        }
        case at::kInt: {
            const int32_t* old_p = old_values.data_ptr<int32_t>();
            const int32_t* new_p = new_values.data_ptr<int32_t>();
            uint8_t* old_m = old_match.data_ptr<uint8_t>();
            uint8_t* new_m = new_match.data_ptr<uint8_t>();
            mark_matches_kernel<int32_t><<<blocks_old, threads, 0, stream>>>(
                old_p, N, new_p, M, 0.0, old_m);
            mark_matches_kernel<int32_t><<<blocks_new, threads, 0, stream>>>(
                new_p, M, old_p, N, 0.0, new_m);
            break;
        }
        case at::kLong: {
            const int64_t* old_p = old_values.data_ptr<int64_t>();
            const int64_t* new_p = new_values.data_ptr<int64_t>();
            uint8_t* old_m = old_match.data_ptr<uint8_t>();
            uint8_t* new_m = new_match.data_ptr<uint8_t>();
            mark_matches_kernel<int64_t><<<blocks_old, threads, 0, stream>>>(
                old_p, N, new_p, M, 0.0, old_m);
            mark_matches_kernel<int64_t><<<blocks_new, threads, 0, stream>>>(
                new_p, M, old_p, N, 0.0, new_m);
            break;
        }
        default:
            TORCH_CHECK(false, "Unsupported dtype for values: ", old_values.scalar_type());
    }

    // Use Thrust to count and compact remaining elements (where match == 0)
    auto policy = thrust::cuda::par.on(stream);

    const uint8_t* old_m_ptr = old_match.data_ptr<uint8_t>();
    const uint8_t* new_m_ptr = new_match.data_ptr<uint8_t>();

    int64_t remain_old_count = thrust::count_if(policy,
        thrust::device_pointer_cast(old_m_ptr),
        thrust::device_pointer_cast(old_m_ptr) + N,
        is_zero());

    int64_t remain_new_count = thrust::count_if(policy,
        thrust::device_pointer_cast(new_m_ptr),
        thrust::device_pointer_cast(new_m_ptr) + M,
        is_zero());

    at::Tensor remain_keys = at::empty({remain_old_count}, keys.options());
    at::Tensor remain_new_values = at::empty({remain_new_count}, new_values.options());

    // Copy with stencil (select where match == 0)
    thrust::copy_if(policy,
        thrust::device_pointer_cast(keys.data_ptr<int64_t>()),
        thrust::device_pointer_cast(keys.data_ptr<int64_t>()) + N,
        thrust::device_pointer_cast(old_m_ptr),
        thrust::device_pointer_cast(remain_keys.data_ptr<int64_t>()),
        is_zero());

    switch (new_values.scalar_type()) {
        case at::kFloat: {
            thrust::copy_if(policy,
                thrust::device_pointer_cast(new_values.data_ptr<float>()),
                thrust::device_pointer_cast(new_values.data_ptr<float>()) + M,
                thrust::device_pointer_cast(new_m_ptr),
                thrust::device_pointer_cast(remain_new_values.data_ptr<float>()),
                is_zero());
            break;
        }
        case at::kDouble: {
            thrust::copy_if(policy,
                thrust::device_pointer_cast(new_values.data_ptr<double>()),
                thrust::device_pointer_cast(new_values.data_ptr<double>()) + M,
                thrust::device_pointer_cast(new_m_ptr),
                thrust::device_pointer_cast(remain_new_values.data_ptr<double>()),
                is_zero());
            break;
        }
        case at::kInt: {
            thrust::copy_if(policy,
                thrust::device_pointer_cast(new_values.data_ptr<int32_t>()),
                thrust::device_pointer_cast(new_values.data_ptr<int32_t>()) + M,
                thrust::device_pointer_cast(new_m_ptr),
                thrust::device_pointer_cast(remain_new_values.data_ptr<int32_t>()),
                is_zero());
            break;
        }
        case at::kLong: {
            thrust::copy_if(policy,
                thrust::device_pointer_cast(new_values.data_ptr<int64_t>()),
                thrust::device_pointer_cast(new_values.data_ptr<int64_t>()) + M,
                thrust::device_pointer_cast(new_m_ptr),
                thrust::device_pointer_cast(remain_new_values.data_ptr<int64_t>()),
                is_zero());
            break;
        }
        default:
            TORCH_CHECK(false, "Unsupported dtype for values: ", new_values.scalar_type());
    }

    return {remain_keys, remain_new_values};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("diff_two_map", &diff_two_map_cuda,
          "Find non-overlapped keys (from old_values) and non-overlapped new_values (CUDA)",
          py::arg("keys"), py::arg("old_values"), py::arg("new_values"), py::arg("eps") = 1e-6);
}
