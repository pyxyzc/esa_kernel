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

def diff_two_map_simple(keys, old_values, new_values):
    remain_keys = []
    remain_new_values = []
    n = len(old_values)
    remain_flags = [True for _ in range(n)]
    for i, e in enumerate(new_values):
        index = np.where(old_values == e)[0]
        if index.shape[0] == 0:
            remain_new_values.append(e)
        else:
            remain_flags[index[0]] = False

    for i, e in enumerate(remain_flags):
        if e:
            remain_keys.append(keys[i])
    # print('remain_keys:',remain_keys)
    # print('remain_new_values:',remain_new_values)



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

keys = np.array(list(map1.keys()))
old_values = np.array(list(map1.values()))
new_values = np.array(list(map2.values()))

import time
times = []
for _ in range(100):
    begin = time.perf_counter_ns()
    # res1 =diff_two_map(map1, map2)
    diff_two_map_simple(keys, old_values, new_values)
    dura = time.perf_counter_ns() - begin
    times.append(dura)
times = times[10:]
print(f'simple time: {sum(times)/len(times) / 1e6}ms')


def diff_two_map_np(keys, old_values, new_values):
    equal = new_values[:, None] == old_values[None, :]

    # diff = new_values[:, None] - old_values[None, :] # N(new_values) * N(keys)
    # equal = (diff < 1e-3)


    # remain_keys = keys[~np.any(equal, axis=0)]
    # remain_new_values = new_values[~np.any(equal, axis=1)]

    rows = np.sum(equal, 1)
    cols = np.sum(equal, 0)
    # remain_new_values = new_values[np.where(rows == 0)]
    # remain_keys = keys[np.where(cols == 0)]

    remain_new_values = new_values[rows == 0]
    remain_keys = keys[cols == 0]

    return remain_keys, remain_new_values

def diff_two_map_torch(keys, old_values, new_values):
    equal = (new_values[:, None] == old_values[None, :])
    # diff = new_values[:, None] - old_values[None, :] # N(new_values) * N(keys)
    # equal = (diff < 1e-3)
    rows = torch.sum(equal, 1)
    cols = torch.sum(equal, 0)
    remain_new_values = new_values[rows == 0]
    remain_keys = keys[cols == 0]
    return remain_keys, remain_new_values


times = []
for _ in range(100):
    begin = time.perf_counter_ns()
    res2 = diff_two_map_np(keys, old_values, new_values)
    dura = time.perf_counter_ns() - begin
    # print("np dura", dura/1e6)
    times.append(dura)
times = times[10:]
print(f'time: {sum(times)/len(times) / 1e6}ms')



keys = torch.tensor(keys).cuda().to(torch.int32)
old_values = torch.tensor(old_values).cuda().to(torch.int32)
new_values = torch.tensor(new_values).cuda().to(torch.int32)

times = []
for _ in range(100):
    begin = time.perf_counter_ns()
    res3 = diff_two_map_torch(keys, old_values, new_values)
    torch.cuda.synchronize()
    dura = time.perf_counter_ns() - begin
    times.append(dura)
times = times[10:]
print(f'torch time: {sum(times)/len(times) / 1e6}ms')

import diff_map as diff_lib
times = []
for _ in range(100):
    begin = time.perf_counter_ns()
    unkept_keys, unkept_vals = diff_lib.filter_unkept(keys, old_values, new_values)
    torch.cuda.synchronize()
    dura = time.perf_counter_ns() - begin
    times.append(dura)
times = times[10:]
print(f'time: {sum(times)/len(times) / 1e6}ms')
