## https://pytorch.org/docs/stable/nn.functional.html

import torch
import torch.nn.functional as F
import time
import cpu
import gpu

stride = 1
padding = 1
dilation = 1

ifm = (torch.rand(8, 1024, 32, 32) * 10).int().float()
wgt = (torch.rand(8, 1024, 3, 3) * 10).int().float()
print(ifm.shape)
print(wgt.shape)

t = time.perf_counter()
ofm1 = F.conv2d(ifm, wgt, stride=stride, padding=padding, dilation=dilation)
print(time.perf_counter() - t)
# print(ofm1)

t = time.perf_counter()
ofm2 = cpu.conv2d(ifm, wgt, stride, padding, dilation)
print(time.perf_counter() - t)
# print(ofm2)

t = time.perf_counter()
ofm3 = gpu.conv2d(ifm, wgt, stride, padding, dilation)
print(time.perf_counter() - t)
# print(ofm3)

print(ofm1.shape)
print(ofm2.shape)
print(ofm3.shape)
# print(ofm1 == ofm2)
# print(ofm1 == ofm3)
