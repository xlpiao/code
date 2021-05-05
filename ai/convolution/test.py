## https://pytorch.org/docs/stable/nn.functional.html
## nn.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)

import torch
import torch.nn.functional as nn
import time
import cpu
import gpu
import pdb

stride = 1
padding = 1
dilation = 1

ifm = (torch.rand(8, 3, 32, 32) * 10).int().float()
wgt = (torch.rand(8, 3, 3, 3) * 10).int().float()
# ifm = (torch.rand(1, 1, 4096, 4096) * 10).int().float()
# wgt = (torch.rand(1, 1, 3, 3) * 10).int().float()
print("ifm.shape: ", ifm.shape)
print("wgt.shape: ", wgt.shape)
# pdb.set_trace()

t = time.perf_counter()
ofm1 = nn.conv2d(ifm, wgt, stride=stride, padding=padding, dilation=dilation)
print("nn.conv2d():\t", format(time.perf_counter() - t, ".6f") + "s")
# print("nn.conv2d() result:\t", ofm1)

t = time.perf_counter()
ofm2 = cpu.conv2d(ifm, wgt, stride, padding, dilation)
print("cpu.conv2d():\t", format(time.perf_counter() - t, ".6f") + "s")
# print("cpu.conv2d() result:\t", ofm2)

t = time.perf_counter()
ofm3 = gpu.conv2d(ifm, wgt, stride, padding, dilation)
print("gpu.conv2d():\t", format(time.perf_counter() - t, ".6f") + "s")
# print("gpu.conv2d() result:\t", ofm3)

print(ofm1.shape)
print(ofm2.shape)
print(ofm3.shape)
# print(ofm1 == ofm2)
# print(ofm1 == ofm3)
