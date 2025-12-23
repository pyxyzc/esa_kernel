#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <type_traits>
#include <math.h>

template <typename T>
__global__ void add_scalar_kernel(const T* __restrict__ in, T* __restrict__ out, int64_t N, T alpha) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < N; i += stride) {
        out[i] = in[i] + alpha;
    }
}

template <typename IndexT>
__global__ void mark_from_bounds(const IndexT* __restrict__ lower,
                                 const IndexT* __restrict__ upper,
                                 int64_t N,
                                 uint8_t* __restrict__ out) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < N; i += stride) {
        out[i] = (upper[i] > lower[i]) ? 1 : 0;
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

    auto policy = thrust::cuda::par.on(stream);

    auto byte_opts = old_values.options().dtype(at::kByte);
    at::Tensor old_match = at::empty({N}, byte_opts);
    at::Tensor new_match = at::empty({M}, byte_opts);

    const int threads = 256;
    const int blocks_old = std::min<int64_t>( (N + threads - 1) / threads, 4096 );
    const int blocks_new = std::min<int64_t>( (M + threads - 1) / threads, 4096 );

    // Use sorting + batched binary search to compute membership masks in O((N+M)log(N+M))
    switch (old_values.scalar_type()) {
        case at::kFloat: {
            using T = float;
            // Sort copies for search
            at::Tensor new_sorted = new_values.clone();
            thrust::sort(policy,
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()) + M);
            at::Tensor old_sorted = old_values.clone();
            thrust::sort(policy,
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()) + N);

            // For old_values vs new_sorted
            at::Tensor old_minus = at::empty_like(old_values);
            at::Tensor old_plus  = at::empty_like(old_values);
            add_scalar_kernel<T><<<blocks_old, threads, 0, stream>>>(
                old_values.data_ptr<T>(), old_minus.data_ptr<T>(), N, -static_cast<T>(eps));
            add_scalar_kernel<T><<<blocks_old, threads, 0, stream>>>(
                old_values.data_ptr<T>(), old_plus.data_ptr<T>(), N, static_cast<T>(eps));
            at::Tensor lower_old = at::empty({N}, keys.options().dtype(at::kLong));
            at::Tensor upper_old = at::empty({N}, keys.options().dtype(at::kLong));
            thrust::lower_bound(policy,
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()) + M,
                thrust::device_pointer_cast(old_minus.data_ptr<T>()),
                thrust::device_pointer_cast(old_minus.data_ptr<T>()) + N,
                thrust::device_pointer_cast(lower_old.data_ptr<int64_t>()));
            thrust::upper_bound(policy,
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()) + M,
                thrust::device_pointer_cast(old_plus.data_ptr<T>()),
                thrust::device_pointer_cast(old_plus.data_ptr<T>()) + N,
                thrust::device_pointer_cast(upper_old.data_ptr<int64_t>()));
            mark_from_bounds<int64_t><<<blocks_old, threads, 0, stream>>>(
                lower_old.data_ptr<int64_t>(), upper_old.data_ptr<int64_t>(), N, old_match.data_ptr<uint8_t>());

            // For new_values vs old_sorted
            at::Tensor new_minus = at::empty_like(new_values);
            at::Tensor new_plus  = at::empty_like(new_values);
            add_scalar_kernel<T><<<blocks_new, threads, 0, stream>>>(
                new_values.data_ptr<T>(), new_minus.data_ptr<T>(), M, -static_cast<T>(eps));
            add_scalar_kernel<T><<<blocks_new, threads, 0, stream>>>(
                new_values.data_ptr<T>(), new_plus.data_ptr<T>(), M, static_cast<T>(eps));
            at::Tensor lower_new = at::empty({M}, keys.options().dtype(at::kLong));
            at::Tensor upper_new = at::empty({M}, keys.options().dtype(at::kLong));
            thrust::lower_bound(policy,
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()) + N,
                thrust::device_pointer_cast(new_minus.data_ptr<T>()),
                thrust::device_pointer_cast(new_minus.data_ptr<T>()) + M,
                thrust::device_pointer_cast(lower_new.data_ptr<int64_t>()));
            thrust::upper_bound(policy,
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()) + N,
                thrust::device_pointer_cast(new_plus.data_ptr<T>()),
                thrust::device_pointer_cast(new_plus.data_ptr<T>()) + M,
                thrust::device_pointer_cast(upper_new.data_ptr<int64_t>()));
            mark_from_bounds<int64_t><<<blocks_new, threads, 0, stream>>>(
                lower_new.data_ptr<int64_t>(), upper_new.data_ptr<int64_t>(), M, new_match.data_ptr<uint8_t>());
            break;
        }
        case at::kDouble: {
            using T = double;
            // Sort copies for search
            at::Tensor new_sorted = new_values.clone();
            thrust::sort(policy,
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()) + M);
            at::Tensor old_sorted = old_values.clone();
            thrust::sort(policy,
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()) + N);

            // For old_values vs new_sorted
            at::Tensor old_minus = at::empty_like(old_values);
            at::Tensor old_plus  = at::empty_like(old_values);
            add_scalar_kernel<T><<<blocks_old, threads, 0, stream>>>(
                old_values.data_ptr<T>(), old_minus.data_ptr<T>(), N, -static_cast<T>(eps));
            add_scalar_kernel<T><<<blocks_old, threads, 0, stream>>>(
                old_values.data_ptr<T>(), old_plus.data_ptr<T>(), N, static_cast<T>(eps));
            at::Tensor lower_old = at::empty({N}, keys.options().dtype(at::kLong));
            at::Tensor upper_old = at::empty({N}, keys.options().dtype(at::kLong));
            thrust::lower_bound(policy,
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()) + M,
                thrust::device_pointer_cast(old_minus.data_ptr<T>()),
                thrust::device_pointer_cast(old_minus.data_ptr<T>()) + N,
                thrust::device_pointer_cast(lower_old.data_ptr<int64_t>()));
            thrust::upper_bound(policy,
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()) + M,
                thrust::device_pointer_cast(old_plus.data_ptr<T>()),
                thrust::device_pointer_cast(old_plus.data_ptr<T>()) + N,
                thrust::device_pointer_cast(upper_old.data_ptr<int64_t>()));
            mark_from_bounds<int64_t><<<blocks_old, threads, 0, stream>>>(
                lower_old.data_ptr<int64_t>(), upper_old.data_ptr<int64_t>(), N, old_match.data_ptr<uint8_t>());

            // For new_values vs old_sorted
            at::Tensor new_minus = at::empty_like(new_values);
            at::Tensor new_plus  = at::empty_like(new_values);
            add_scalar_kernel<T><<<blocks_new, threads, 0, stream>>>(
                new_values.data_ptr<T>(), new_minus.data_ptr<T>(), M, -static_cast<T>(eps));
            add_scalar_kernel<T><<<blocks_new, threads, 0, stream>>>(
                new_values.data_ptr<T>(), new_plus.data_ptr<T>(), M, static_cast<T>(eps));
            at::Tensor lower_new = at::empty({M}, keys.options().dtype(at::kLong));
            at::Tensor upper_new = at::empty({M}, keys.options().dtype(at::kLong));
            thrust::lower_bound(policy,
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()) + N,
                thrust::device_pointer_cast(new_minus.data_ptr<T>()),
                thrust::device_pointer_cast(new_minus.data_ptr<T>()) + M,
                thrust::device_pointer_cast(lower_new.data_ptr<int64_t>()));
            thrust::upper_bound(policy,
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()) + N,
                thrust::device_pointer_cast(new_plus.data_ptr<T>()),
                thrust::device_pointer_cast(new_plus.data_ptr<T>()) + M,
                thrust::device_pointer_cast(upper_new.data_ptr<int64_t>()));
            mark_from_bounds<int64_t><<<blocks_new, threads, 0, stream>>>(
                lower_new.data_ptr<int64_t>(), upper_new.data_ptr<int64_t>(), M, new_match.data_ptr<uint8_t>());
            break;
        }
        case at::kInt: {
            using T = int32_t;
            // Sort copies for search
            at::Tensor new_sorted = new_values.clone();
            thrust::sort(policy,
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()) + M);
            at::Tensor old_sorted = old_values.clone();
            thrust::sort(policy,
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()) + N);

            // For old_values vs new_sorted
            at::Tensor lower_old = at::empty({N}, keys.options().dtype(at::kLong));
            at::Tensor upper_old = at::empty({N}, keys.options().dtype(at::kLong));
            thrust::lower_bound(policy,
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()) + M,
                thrust::device_pointer_cast(old_values.data_ptr<T>()),
                thrust::device_pointer_cast(old_values.data_ptr<T>()) + N,
                thrust::device_pointer_cast(lower_old.data_ptr<int64_t>()));
            thrust::upper_bound(policy,
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()) + M,
                thrust::device_pointer_cast(old_values.data_ptr<T>()),
                thrust::device_pointer_cast(old_values.data_ptr<T>()) + N,
                thrust::device_pointer_cast(upper_old.data_ptr<int64_t>()));
            mark_from_bounds<int64_t><<<blocks_old, threads, 0, stream>>>(
                lower_old.data_ptr<int64_t>(), upper_old.data_ptr<int64_t>(), N, old_match.data_ptr<uint8_t>());

            // For new_values vs old_sorted
            at::Tensor lower_new = at::empty({M}, keys.options().dtype(at::kLong));
            at::Tensor upper_new = at::empty({M}, keys.options().dtype(at::kLong));
            thrust::lower_bound(policy,
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()) + N,
                thrust::device_pointer_cast(new_values.data_ptr<T>()),
                thrust::device_pointer_cast(new_values.data_ptr<T>()) + M,
                thrust::device_pointer_cast(lower_new.data_ptr<int64_t>()));
            thrust::upper_bound(policy,
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()) + N,
                thrust::device_pointer_cast(new_values.data_ptr<T>()),
                thrust::device_pointer_cast(new_values.data_ptr<T>()) + M,
                thrust::device_pointer_cast(upper_new.data_ptr<int64_t>()));
            mark_from_bounds<int64_t><<<blocks_new, threads, 0, stream>>>(
                lower_new.data_ptr<int64_t>(), upper_new.data_ptr<int64_t>(), M, new_match.data_ptr<uint8_t>());
            break;
        }
        case at::kLong: {
            using T = int64_t;
            // Sort copies for search
            at::Tensor new_sorted = new_values.clone();
            thrust::sort(policy,
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()) + M);
            at::Tensor old_sorted = old_values.clone();
            thrust::sort(policy,
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()) + N);

            // For old_values vs new_sorted
            at::Tensor lower_old = at::empty({N}, keys.options().dtype(at::kLong));
            at::Tensor upper_old = at::empty({N}, keys.options().dtype(at::kLong));
            thrust::lower_bound(policy,
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()) + M,
                thrust::device_pointer_cast(old_values.data_ptr<T>()),
                thrust::device_pointer_cast(old_values.data_ptr<T>()) + N,
                thrust::device_pointer_cast(lower_old.data_ptr<int64_t>()));
            thrust::upper_bound(policy,
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(new_sorted.data_ptr<T>()) + M,
                thrust::device_pointer_cast(old_values.data_ptr<T>()),
                thrust::device_pointer_cast(old_values.data_ptr<T>()) + N,
                thrust::device_pointer_cast(upper_old.data_ptr<int64_t>()));
            mark_from_bounds<int64_t><<<blocks_old, threads, 0, stream>>>(
                lower_old.data_ptr<int64_t>(), upper_old.data_ptr<int64_t>(), N, old_match.data_ptr<uint8_t>());

            // For new_values vs old_sorted
            at::Tensor lower_new = at::empty({M}, keys.options().dtype(at::kLong));
            at::Tensor upper_new = at::empty({M}, keys.options().dtype(at::kLong));
            thrust::lower_bound(policy,
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()) + N,
                thrust::device_pointer_cast(new_values.data_ptr<T>()),
                thrust::device_pointer_cast(new_values.data_ptr<T>()) + M,
                thrust::device_pointer_cast(lower_new.data_ptr<int64_t>()));
            thrust::upper_bound(policy,
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()),
                thrust::device_pointer_cast(old_sorted.data_ptr<T>()) + N,
                thrust::device_pointer_cast(new_values.data_ptr<T>()),
                thrust::device_pointer_cast(new_values.data_ptr<T>()) + M,
                thrust::device_pointer_cast(upper_new.data_ptr<int64_t>()));
            mark_from_bounds<int64_t><<<blocks_new, threads, 0, stream>>>(
                lower_new.data_ptr<int64_t>(), upper_new.data_ptr<int64_t>(), M, new_match.data_ptr<uint8_t>());
            break;
        }
        default:
            TORCH_CHECK(false, "Unsupported dtype for values: ", old_values.scalar_type());
    }

    // Use Thrust to count and compact remaining elements (where match == 0)

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

// Row indices with any non-zero in the row (CUDA)

template <typename T>
__device__ __forceinline__ bool is_nz(T v, double /*eps*/) {
    return v != static_cast<T>(0);
}
template <>
__device__ __forceinline__ bool is_nz<float>(float v, double eps) {
    return fabsf(v) > static_cast<float>(eps);
}
template <>
__device__ __forceinline__ bool is_nz<double>(double v, double eps) {
    return fabs(v) > eps;
}

template <typename T>
__global__ void row_has_nonzero_kernel(const T* __restrict__ a,
                                       int64_t N, int64_t M, double eps,
                                       uint8_t* __restrict__ flags)
{
    int row = blockIdx.x;
    if (row >= N) return;

    __shared__ int any;
    if (threadIdx.x == 0) any = 0;
    __syncthreads();

    const T* row_ptr = a + static_cast<size_t>(row) * static_cast<size_t>(M);

    for (int64_t j = threadIdx.x; j < M; j += blockDim.x) {
        T v = row_ptr[j];
        if (is_nz<T>(v, eps)) {
            atomicExch(&any, 1);
            break;
        }
        if (any) break;
    }
    __syncthreads();
    if (threadIdx.x == 0) flags[row] = static_cast<uint8_t>(any);
}

at::Tensor row_indices_with_nonzero(at::Tensor matrix, double eps=0.0) {
    TORCH_CHECK(matrix.is_cuda(), "matrix must be a CUDA tensor");
    TORCH_CHECK(matrix.dim() == 2, "matrix must be 2D");
    matrix = matrix.contiguous();

    const int64_t N = matrix.size(0);
    const int64_t M = matrix.size(1);

    auto flags = at::empty({N}, matrix.options().dtype(at::kByte));
    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    dim3 blocks(static_cast<unsigned int>(N));

    switch (matrix.scalar_type()) {
        case at::kFloat:
            row_has_nonzero_kernel<float><<<blocks, threads, 0, stream>>>(
                matrix.data_ptr<float>(), N, M, eps, flags.data_ptr<uint8_t>());
            break;
        case at::kDouble:
            row_has_nonzero_kernel<double><<<blocks, threads, 0, stream>>>(
                matrix.data_ptr<double>(), N, M, eps, flags.data_ptr<uint8_t>());
            break;
        case at::kInt:
            row_has_nonzero_kernel<int32_t><<<blocks, threads, 0, stream>>>(
                matrix.data_ptr<int32_t>(), N, M, 0.0, flags.data_ptr<uint8_t>());
            break;
        case at::kLong:
            row_has_nonzero_kernel<int64_t><<<blocks, threads, 0, stream>>>(
                matrix.data_ptr<int64_t>(), N, M, 0.0, flags.data_ptr<uint8_t>());
            break;
        default:
            TORCH_CHECK(false, "row_indices_with_nonzero: unsupported dtype: ", matrix.scalar_type());
    }

    auto out_idx = at::empty({N}, matrix.options().dtype(at::kLong));
    auto policy = thrust::cuda::par.on(stream);

    auto begin = thrust::make_counting_iterator<int64_t>(0);
    auto end   = thrust::make_counting_iterator<int64_t>(N);

    auto stencil_begin = thrust::device_pointer_cast(flags.data_ptr<uint8_t>());
    auto out_begin = thrust::device_pointer_cast(out_idx.data_ptr<int64_t>());

    auto new_end = thrust::copy_if(
        policy, begin, end, stencil_begin, out_begin,
        [] __device__ (uint8_t f) { return f != 0; });

    int64_t K = new_end - out_begin;
    return out_idx.narrow(0, 0, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("diff_two_map", &diff_two_map_cuda,
          "Find non-overlapped keys (from old_values) and non-overlapped new_values (CUDA)",
          py::arg("keys"), py::arg("old_values"), py::arg("new_values"), py::arg("eps") = 1e-6);
    m.def("row_indices_with_nonzero", &row_indices_with_nonzero,
          "Return row indices that contain at least one non-zero (CUDA)",
          py::arg("matrix"), py::arg("eps") = 0.0);
}
