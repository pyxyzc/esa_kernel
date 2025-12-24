import numpy as np
import torch
import random

def diff_two_map(map1, map2):
    keys2 = map2.keys()
    values2 = map2.values()
    keys2_set = set(keys2)
    values2_set = set(values2)
    diff_map = {}
    updated_map = {}
    for k1, v1 in map1.items():
        if k1 in keys2 and v1 in values2:
            updated_map[k1] = v1
            keys2_set.remove(k1)
            values2_set.remove(v1)
    for k2, v2 in zip(keys2_set, values2_set):
        diff_map[k2] = v2
        updated_map[k2] = v2
    return updated_map, diff_map



map_size = 100
map1 = {}
for i in range(map_size):
    map1[i] = i * 3 + 1
map2 = {}
map1_values = list(map1.values())
random.shuffle(map1_values)
overlap = 30
for i in range(map_size):
    if i < overlap:
        map2[i] = map1_values[i]
    else:
        map2[i] = i + i *i + map_size * 2

import time
times = []
for _ in range(100):
    begin = time.perf_counter_ns()
    res1 =diff_two_map(map1, map2)
    dura = time.perf_counter_ns() - begin
    times.append(dura)
times = times[10:]
print(f'time: {sum(times)/len(times) / 1e6}ms')


def diff_two_map_np(keys, old_values, new_values):
    diff = new_values[:, None] - old_values[None, :] # N(new_values) * N(keys)
    equal = (diff < 1e-3)
    rows = np.sum(equal, 1)
    cols = np.sum(equal, 0)
    remain_new_values = new_values[np.where(rows == 0)]
    remain_keys = keys[np.where(cols == 0)]
    return remain_keys, remain_new_values

def diff_two_map_torch(keys, old_values, new_values):
    diff = new_values[:, None] - old_values[None, :] # N(new_values) * N(keys)
    equal = (diff < 1e-3)
    rows = torch.sum(equal, 1)
    cols = torch.sum(equal, 0)
    remain_new_values = new_values[torch.where(rows == 0)]
    remain_keys = keys[torch.where(cols == 0)]
    return remain_keys, remain_new_values


keys = np.array(list(map1.keys()))
old_values = np.array(list(map1.values()))
new_values = np.array(list(map2.values()))

times = []
for _ in range(100):
    begin = time.perf_counter_ns()
    res2 = diff_two_map_np(keys, old_values, new_values)
    dura = time.perf_counter_ns() - begin
    print("np dura", dura/1e6)
    times.append(dura)
times = times[10:]
print(f'time: {sum(times)/len(times) / 1e6}ms')



keys = torch.tensor(keys).cuda()
old_values = torch.tensor(old_values).cuda()
new_values = torch.tensor(new_values).cuda()

times = []
for _ in range(100):
    begin = time.perf_counter_ns()
    res3 = diff_two_map_torch(keys, old_values, new_values)
    dura = time.perf_counter_ns() - begin
    times.append(dura)
torch.cuda.synchronize()
times = times[10:]
print(f'time: {sum(times)/len(times) / 1e6}ms')

from torch.utils.cpp_extension import load as _load_ext

_diff_map_ext = _load_ext(
    name="diff_map_ext",
    sources=["diff_map_extension.cu"],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=["-O3", "-std=c++17"],
    verbose=False
)

def diff_two_map_cuda(keys, old_values, new_values, eps=1e-6):
    return _diff_map_ext.diff_two_map(keys, old_values, new_values, float(eps))

times = []
for _ in range(100):
    begin = time.perf_counter_ns()
    rk, rnv = diff_two_map_cuda(keys, old_values, new_values, 0.0)
    dura = time.perf_counter_ns() - begin
    print("dura:", dura/1e6)
    times.append(dura)
torch.cuda.synchronize()
times = times[10:]
print(f'time: {sum(times)/len(times) / 1e6}ms, remain_keys={rk.numel()}, remain_new_values={rnv.numel()}')
