#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#define CUDA_TRANS_UNIT_SIZE (sizeof(uint4) * 2)

constexpr __host__ __device__
int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

inline __device__ void CudaCopyUnit(const uint8_t* __restrict__ src,
                                    volatile uint8_t* __restrict__ dst)
{
    uint4 lo, hi;
    asm volatile("ld.global.cs.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(lo.x), "=r"(lo.y), "=r"(lo.z), "=r"(lo.w)
                 : "l"(src));
    asm volatile("ld.global.cs.v4.b32 {%0,%1,%2,%3}, [%4+16];"
                 : "=r"(hi.x), "=r"(hi.y), "=r"(hi.z), "=r"(hi.w)
                 : "l"(src));
    asm volatile("st.volatile.global.v4.b32 [%0], {%1,%2,%3,%4};"
                 :
                 : "l"(dst), "r"(lo.x), "r"(lo.y), "r"(lo.z), "r"(lo.w));
    asm volatile("st.volatile.global.v4.b32 [%0+16], {%1,%2,%3,%4};"
                 :
                 : "l"(dst), "r"(hi.x), "r"(hi.y), "r"(hi.z), "r"(hi.w));
}

__global__ void CudaCopyKernel(const void** src, void** dst, size_t size, size_t num)
{
    auto length = size * num;
    auto offset = (blockIdx.x * blockDim.x + threadIdx.x) * CUDA_TRANS_UNIT_SIZE;
    while (offset + CUDA_TRANS_UNIT_SIZE <= length) {
        auto idx = offset / size;
        auto off = offset % size;
        auto host = ((const uint8_t*)src[idx]) + off;
        auto device = ((uint8_t*)dst[idx]) + off;
        CudaCopyUnit(host, device);
        offset += gridDim.x * blockDim.x * CUDA_TRANS_UNIT_SIZE;
    }
}

__global__ void CudaCopyKernel(const void** src, void* dst, size_t size, size_t num)
{
    auto length = size * num;
    auto offset = (blockIdx.x * blockDim.x + threadIdx.x) * CUDA_TRANS_UNIT_SIZE;
    while (offset + CUDA_TRANS_UNIT_SIZE <= length) {
        auto idx = offset / size;
        auto off = offset % size;
        auto host = ((const uint8_t*)src[idx]) + off;
        auto device = ((uint8_t*)dst) + offset;
        CudaCopyUnit(host, device);
        offset += gridDim.x * blockDim.x * CUDA_TRANS_UNIT_SIZE;
    }
}

__global__ void CudaCopyKernel(const void* src, void** dst, size_t size, size_t num)
{
    auto length = size * num;
    auto offset = (blockIdx.x * blockDim.x + threadIdx.x) * CUDA_TRANS_UNIT_SIZE;
    while (offset + CUDA_TRANS_UNIT_SIZE <= length) {
        auto idx = offset / size;
        auto off = offset % size;
        auto host = ((const uint8_t*)src) + offset;
        auto device = ((uint8_t*)dst[idx]) + off;
        CudaCopyUnit(host, device);
        offset += gridDim.x * blockDim.x * CUDA_TRANS_UNIT_SIZE;
    }
}

__global__ void CudaCopyKernel(const void* src, void* dst, size_t size)
{
    int num_repeats = (size + gridDim.x * blockDim.x *CUDA_TRANS_UNIT_SIZE - 1) / (gridDim.x * blockDim.x * CUDA_TRANS_UNIT_SIZE);
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i = 0; i < num_repeats; ++i){
        int offset = (i * gridDim.x * blockDim.x + tid) * CUDA_TRANS_UNIT_SIZE;
        auto host = ((const uint8_t*)src) + offset;
        auto device = ((uint8_t*)dst) + offset;
        CudaCopyUnit(host, device);
    }
}


extern "C" void esa_copy(torch::Tensor src, torch::Tensor dst, size_t size)
{
    dim3 numThreads = {1024};
    int totalThreads = ceildiv(size, CUDA_TRANS_UNIT_SIZE);
    dim3 numBlocks = {static_cast<unsigned int>(ceildiv(totalThreads, numThreads.x))};
    CudaCopyKernel<<<numBlocks, numThreads>>>(
        (const void*)src.data_ptr(), (void*)dst.data_ptr(), size);
}
