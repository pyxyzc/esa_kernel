#include "esa_utils.h"
/**
 * This kernel performs: repre_cache[repre_block_table[i]] = mean( key_cache[key_block_table[i]], 0 )
 *
 * @param key_cache: [N, block_size, dim]
 * @param repre_cache: [N, dim]
 * @param key_block_table: [S]
 * @param repre_block_table: [S]
 */
template <typename scalar_t>
__global__ void extract_repre(const scalar_t *key_cache, scalar_t *repre_cache, const int *key_block_table, const int *repre_block_table, int block_size, int dim) {
    int idx = blockIdx.x;
    int block_id = key_block_table[idx];
    int block_id_2 = repre_block_table[idx];
    const scalar_t* key_ptr = key_cache + block_id * block_size * dim;
    scalar_t* repre_ptr = repre_cache + block_id_2 * dim;
    int d = threadIdx.x;
    if (d < dim) {
        float sum = 0;
        for (int j = 0; j < block_size; ++j) {
            sum += static_cast<float>(key_ptr[j * dim + d]);
        }
        repre_ptr[d] = static_cast<scalar_t>(sum / block_size);
    }
}

/**
 * This kernel performs: score[i] = queries[batch_index[i]] * repre_cache[block_table[i]]
 *
 * @param queries: a list of tensors. { [dim] }
 * @param repre_cache: [N, dim]
 * @param score: [S]
 * @param block_table: [S]
 * @param batch_index: [S]
 */
__global__ void retrieval_kernel_fp32(float **queries, float *__restrict__ repre_cache, float *__restrict__ score, int *__restrict__ block_table, int *__restrict__ batch_index, int dim, int S){
    extern __shared__ float local_score[]; // num of threads
    int global_x = blockIdx.x;
    int local_x = threadIdx.x;
    if (global_x < S){
        const float *q = queries[batch_index[global_x]];
        const float *k = repre_cache + block_table[global_x] * dim;
        int num_tiles = (dim + 4 * blockDim.x - 1) / (4 * blockDim.x);
        float sum = 0.0f;
        for(int i = 0; i < num_tiles; ++i){
            int tile_offset = i * (4 * blockDim.x);
            int idx = tile_offset + local_x * 4;
            if(idx + 4 <= dim){
                const float4 q4 = *reinterpret_cast<const float4*>(q + idx);
                const float4 k4 = *reinterpret_cast<const float4*>(k + idx);
                sum += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
            }
        }
        local_score[local_x] = sum;
        __syncthreads();
        for(int i = blockDim.x / 2; i; i = i / 2){
            if(local_x < i){
                local_score[local_x] = local_score[local_x] + local_score[local_x + i];
            }
            __syncthreads();
        }
        score[global_x] = local_score[0];
    }
}

__global__ void retrieval_kernel_fp16(__half **queries, __half *__restrict__ repre_cache, __half *__restrict__ score, int *__restrict__ block_table, int *__restrict__ batch_index, int dim, int S){
    extern __shared__ float local_score_fp16[]; // num of threads
    int global_x = blockIdx.x;
    int local_x = threadIdx.x;
    if (global_x < S){
        const __half *q = queries[batch_index[global_x]];
        const __half *k = repre_cache + block_table[global_x] * dim;
        int num_tiles = (dim + 2 * blockDim.x - 1) / (2 * blockDim.x);
        float sum = 0.0f;
        for(int i = 0; i < num_tiles; ++i){
            int tile_offset = i * (2 * blockDim.x);
            int idx = tile_offset + local_x * 2;
            if(idx + 2 <= dim){
                __half2 q2 = *reinterpret_cast<const __half2*>(q + idx);
                __half2 k2 = *reinterpret_cast<const __half2*>(k + idx);
                __half2 p = __hmul2(q2, k2);
                sum += __half2float(p.x) + __half2float(p.y);
            }
        }
        local_score_fp16[local_x] = sum;
        __syncthreads();
        for(int i = blockDim.x / 2; i; i = i / 2){
            if(local_x < i){
                local_score_fp16[local_x] += local_score_fp16[local_x + i];
            }
            __syncthreads();
        }
        if (local_x == 0) score[global_x] = __float2half(local_score_fp16[0]);
    }
}

__global__ void retrieval_kernel_bf16(__nv_bfloat16** queries, __nv_bfloat16* __restrict__ repre_cache, __nv_bfloat16*  __restrict__ score, int* __restrict__ block_table, int* __restrict__ batch_index, int dim, int S){
    extern __shared__ float local_score_bf16[];
    int global_x = blockIdx.x;
    int local_x  = threadIdx.x;
    if (global_x >= S) return;
    const __nv_bfloat16* q = queries[batch_index[global_x]];
    const __nv_bfloat16* k = repre_cache + block_table[global_x] * dim;
    int num_tiles = (dim + 2 * blockDim.x - 1) / (2 * blockDim.x);
    float sum = 0.0f;
    for (int i = 0; i < num_tiles; ++i) {
        int idx = i * (2 * blockDim.x) + local_x * 2;
        if (idx + 2 <= dim) {
            uint4 tmp   = *reinterpret_cast<const uint4*>(q + idx);
            uint2 q2u   = make_uint2(tmp.x, tmp.y);      // 前 4 个 bf16
            tmp         = *reinterpret_cast<const uint4*>(k + idx);
            uint2 k2u   = make_uint2(tmp.x, tmp.y);
            __nv_bfloat162 q2, k2;
            asm volatile("mov.b32 {%0, %1}, %2;"
                    : "=h"(*reinterpret_cast<uint16_t*>(&q2.x)),
                    "=h"(*reinterpret_cast<uint16_t*>(&q2.y))
                    : "r"(q2u.x));
            asm volatile("mov.b32 {%0, %1}, %2;"
                    : "=h"(*reinterpret_cast<uint16_t*>(&k2.x)),
                    "=h"(*reinterpret_cast<uint16_t*>(&k2.y))
                    : "r"(k2u.x));
            __nv_bfloat162 p = __hmul2(q2, k2);
            sum += __bfloat162float(p.x) + __bfloat162float(p.y);
        }
    }
    local_score_bf16[local_x] = sum;
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (local_x < i)
            local_score_bf16[local_x] += local_score_bf16[local_x + i];
        __syncthreads();
    }
    if (local_x == 0) score[global_x] = __float2bfloat16(local_score_bf16[0]);
}

void esa_repre(torch::Tensor key_cache, torch::Tensor repre_cache, torch::Tensor block_table, torch::Tensor repre_table){
    int block_size = key_cache.size(1);
    int dim = repre_cache.size(-1);
    int threads = dim;
    int blocks = block_table.size(0);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, key_cache.scalar_type(), "esa_repre_cuda", ([&] {
        extract_repre<scalar_t><<<blocks, threads>>>(
                key_cache.data_ptr<scalar_t>(),
                repre_cache.data_ptr<scalar_t>(),
                block_table.data_ptr<int>(),
                repre_table.data_ptr<int>(),
                block_size,
                dim);
        }));
}


void esa_retrieval(const std::vector<torch::Tensor> &query_list, torch::Tensor repre_cache, torch::Tensor q_index, torch::Tensor repre_index, torch::Tensor score, torch::Tensor score_sorted, torch::Tensor index_ranged, torch::Tensor index_sorted, torch::Tensor batch_offset, torch::Tensor workspace){
    int s = q_index.size(0);
    int dim = repre_cache.size(1);
    int batch = query_list.size();
    dim3 numThreads = {(unsigned int)(32)};
    dim3 numBlocks = {(unsigned int)(s)};

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, repre_cache.scalar_type(), "esa_retrieval_cuda", [&]{
        if constexpr (std::is_same_v<scalar_t, float>) {
            float** Q_ptrs = nullptr;
            cudaMallocManaged(&Q_ptrs, batch * sizeof(float*));
            for(int i = 0; i < batch; ++i) {
            Q_ptrs[i] = query_list[i].data_ptr<float>();
            }
            printf("is float32\n");
            size_t bytes = numThreads.x * sizeof(float);
            retrieval_kernel_fp32<<<numBlocks, numThreads, bytes>>>(Q_ptrs, repre_cache.data_ptr<float>(), score.data_ptr<float>(), repre_index.data_ptr<int>(), q_index.data_ptr<int>(), dim, s);
            CUDA_CHECK(cudaFree(Q_ptrs));
            void* temp_workspace = nullptr;
            size_t temp_bytes = 0;
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                    temp_workspace, temp_bytes,
                    score.data_ptr<float>(),  score_sorted.data_ptr<float>(),
                    index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
                    s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
            temp_workspace = workspace.data_ptr<int>();
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                    temp_workspace, temp_bytes,
                    score.data_ptr<float>(),  score_sorted.data_ptr<float>(),
                    index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
                    s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
        } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
            __half** Q_ptrs = nullptr;
            cudaMallocManaged(&Q_ptrs, batch * sizeof(__half*));
            for(int i = 0; i < batch; ++i) {
                Q_ptrs[i] = reinterpret_cast<__half*>(query_list[i].data_ptr());
            }
            printf("is float16\n");
            size_t bytes = numThreads.x * sizeof(float);
            retrieval_kernel_fp16<<<numBlocks, numThreads, bytes>>>(Q_ptrs,
                    reinterpret_cast<__half*>(repre_cache.data_ptr()),
                    reinterpret_cast<__half*>(score.data_ptr()),
                    reinterpret_cast<int*>(repre_index.data_ptr()),
                    reinterpret_cast<int*>(q_index.data_ptr()),
                    dim, s);
            CUDA_CHECK(cudaFree(Q_ptrs));
            void* temp_workspace = nullptr;
            size_t temp_bytes = 0;
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                    temp_workspace, temp_bytes,
                    reinterpret_cast<__half*>(score.data_ptr()),
                    reinterpret_cast<__half*>(score_sorted.data_ptr()),
                    index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
                    s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
            temp_workspace = workspace.data_ptr<int>();
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                    temp_workspace, temp_bytes,
                    reinterpret_cast<__half*>(score.data_ptr()),
                    reinterpret_cast<__half*>(score_sorted.data_ptr()),
                    index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
                    s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
        } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
            __nv_bfloat16** Q_ptrs = nullptr;
            cudaMallocManaged(&Q_ptrs, batch * sizeof(__nv_bfloat16*));
            for(int i = 0; i < batch; ++i) {
                Q_ptrs[i] = reinterpret_cast<__nv_bfloat16*>(query_list[i].data_ptr());
            }
            printf("is bfloat16\n");
            size_t bytes = numThreads.x * sizeof(float);
            retrieval_kernel_bf16<<<numBlocks, numThreads, bytes>>>(Q_ptrs,
                    reinterpret_cast<__nv_bfloat16*>(repre_cache.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(score.data_ptr()),
                    reinterpret_cast<int*>(repre_index.data_ptr()),
                    reinterpret_cast<int*>(q_index.data_ptr()),
                    dim, s);
            CUDA_CHECK(cudaFree(Q_ptrs));
            void* temp_workspace = nullptr;
            size_t temp_bytes = 0;
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                    temp_workspace, temp_bytes,
                    reinterpret_cast<__nv_bfloat16*>(score.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(score_sorted.data_ptr()),
                    index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
                    s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
            temp_workspace = workspace.data_ptr<int>();
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                    temp_workspace, temp_bytes,
                    reinterpret_cast<__nv_bfloat16*>(score.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(score_sorted.data_ptr()),
                    index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
                    s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
        }
    });
}

void esa_topk(torch::Tensor score, torch::Tensor index, torch::Tensor offsets, torch::Tensor score_out, torch::Tensor index_out, torch::Tensor workspace){
    void* temp_workspace = nullptr;
    size_t temp_bytes = 0;
    size_t B = offsets.size(0) - 1;
    size_t total = score.size(0);
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
            temp_workspace, temp_bytes,
            score.data_ptr<float>(),  score_out.data_ptr<float>(),
            index.data_ptr<int>(), index_out.data_ptr<int>(),
            total, B, offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1);
    // NOTE: don't malloc, just reuse the workspace, but the first call of
    // SortPairsDescending is necesssary to determine the workspace size
    // CUDA_CHECK(cudaMalloc(&temp_workspace, temp_bytes));
    temp_workspace = workspace.data_ptr<int>();

    cub::DeviceSegmentedRadixSort::SortPairsDescending(
            temp_workspace, temp_bytes,
            score.data_ptr<float>(),  score_out.data_ptr<float>(),
            index.data_ptr<int>(), index_out.data_ptr<int>(),
            total, B, offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(esa_retrieval)
    TORCH_BINDING_COMMON_EXTENSION(esa_topk)
    TORCH_BINDING_COMMON_EXTENSION(esa_repre)
}
