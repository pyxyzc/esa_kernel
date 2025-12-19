#include <cuda.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void add_kernel(float** in, float* out, int n, int batch)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if(tid < n){
        float sum = 0.0f;
        for(int i = 0; i < batch; ++i){
            sum += in[i][tid];
        }
        out[tid] = sum;
    }
}


void launch(std::vector<torch::Tensor> in_tensors, torch::Tensor out)
{
    int n = in_tensors[0].numel();
    int batch = in_tensors.size();
    std::vector<float*> ptrs = {};
    for (int i = 0; i < batch; ++i){
        ptrs.push_back(reinterpret_cast<float*>(in_tensors[i].data_ptr()));
    }
    dim3 numThreads = {1024};
    dim3 numBlocks = {(size_t)((n + 1024 - 1) / 1024)};
    printf("%d %d\n", n, batch);
    // allocate device array of input pointers
    float** d_ptrs;
    cudaMalloc((void**)&d_ptrs, batch * sizeof(float*));
    cudaMemcpy(d_ptrs, ptrs.data(), batch * sizeof(float*), cudaMemcpyHostToDevice);
    add_kernel<<<numBlocks, numThreads>>>(d_ptrs, out.data_ptr<float>(), n, batch);
    cudaFree(d_ptrs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch, "add two CUDA tensors");
}
