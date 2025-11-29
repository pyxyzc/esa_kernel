import torch
from torch.utils.cpp_extension import load
import time

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="retrieval_kernel",
    sources=["retrieval_kernel.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)
cuda_retrieval = lib.cuda_retrieval

b = 4
s = 10
dim = 576
N = 100
query_list = []
for i in range(b):
    query_list.append(torch.rand(dim, dtype=torch.float32).cuda())


repre_cache = torch.randn(N, dim, dtype = torch.float32).cuda()
repre_table = torch.arange(0, s, dtype = torch.int32).cuda()
q_table = torch.arange(0, s, dtype = torch.int32).cuda()
q_table = q_table % b
score = torch.zeros(s, dtype = torch.float32).cuda()
start = time.time()
cuda_retrieval(query_list, repre_cache, q_table, repre_table, score)
print("launch spent: ", time.time() - start)
torch.cuda.synchronize()
elapsed_cuda = time.time() - start
print(f"cuda_retrieval time: {elapsed_cuda:.6f} s")
print("score: ", score)


def naive_retrieval():
    query = torch.stack(query_list)
    score_gt = (query[q_table] * repre_cache[repre_table]).sum(-1)
    return score_gt

start = time.time()
score_gt = naive_retrieval()
torch.cuda.synchronize()
elapsed_naive = time.time() - start
print(f"naive_retrieval time: {elapsed_naive:.6f} s")
print("score_gt: ", score_gt)

diff = (score - score_gt).abs()
print("diff: ", diff.mean(), diff.max())
