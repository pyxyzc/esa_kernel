#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err)            \
                << " (" #call ") at " << __FILE__ << ":" << __LINE__    \
                << std::endl;                                           \
      std::exit(1);                                                     \
    }                                                                   \
  } while(0)


int main(){
  using clock = std::chrono::high_resolution_clock;
  size_t total_bytes = 1ULL << 30;  // 1 GiB
  size_t chunk_bytes = 1ULL << 20;  // 1 MiB
  chunk_bytes /= 5;
  int    N = total_bytes / chunk_bytes;
  int    MT = 4;                   // 多线程时使用的线程数

  // 分配页锁定主机 + 设备内存
  void *h, *d;
  CUDA_CHECK(cudaMallocHost(&h, total_bytes));
  CUDA_CHECK(cudaMalloc(&d,      total_bytes));

  // 预热
  CUDA_CHECK(cudaMemcpy(d, h, total_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());

  // —— Test A: 单线程 async（1 流） ——
  {
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));

    auto t0 = clock::now();
    for(int i = 0; i < N; i++){
      CUDA_CHECK(cudaMemcpyAsync(
        (char*)d + i*chunk_bytes,
        (char*)h + i*chunk_bytes,
        chunk_bytes,
        cudaMemcpyHostToDevice,
        s));
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
    auto t1 = clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double bw = (double)total_bytes / elapsed / 1e9;
    std::cout << "Single-thread (1-stream) bandwidth: "
              << bw << " GB/s\n";

    CUDA_CHECK(cudaStreamDestroy(s));
  }

  // —— Test B: 多线程 async（MT 线程 + MT 流） ——
  {
    // 每个线程一个流
    std::vector<cudaStream_t> streams(MT);
    for(int t = 0; t < MT; t++){
      CUDA_CHECK(cudaStreamCreate(&streams[t]));
    }

    auto t0 = clock::now();
    // 启动 MT 个线程，轮询提交它们负责的 chunk
    std::vector<std::thread> workers;
    for(int t = 0; t < MT; t++){
      workers.emplace_back([&, t](){
        for(int i = t; i < N; i += MT){
          CUDA_CHECK(cudaMemcpyAsync(
            (char*)d + i*chunk_bytes,
            (char*)h + i*chunk_bytes,
            chunk_bytes,
            cudaMemcpyHostToDevice,
            streams[t]));
        }
      });
    }
    // for(auto &th: workers) th.join();
    // 等待所有流完成
    for(auto &s: streams){
      CUDA_CHECK(cudaStreamSynchronize(s));
    }
    auto t1 = clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double bw = (double)total_bytes / elapsed / 1e9;
    std::cout << MT << "-thread bandwidth: "
              << bw << " GB/s\n";

    for(auto &th: workers) th.join();

    for(auto &s: streams){
      CUDA_CHECK(cudaStreamDestroy(s));
    }
  }

  CUDA_CHECK(cudaFree(d));
  CUDA_CHECK(cudaFreeHost(h));
  return 0;
}

