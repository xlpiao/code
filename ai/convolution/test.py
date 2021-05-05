## https://pytorch.org/docs/stable/nn.functional.html

import torch
import torch.nn.functional as nn
import time
import cpu
import gpu
import pdb

## nn.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
## TODO: bias/groups

## torch.rand(batch, channel, width, height)
## torch.rand(4D, 3D, 2D, 1D)
ifm = (torch.rand(8, 3, 32, 32) * 10).int().float()  #input
wgt = (torch.rand(8, 3, 3, 3) * 10).int().float()  #weight
bias = (torch.rand(8) * 10).int().float()
stride = 1
padding = 1
dilation = 1
groups = 1

print("\r")
print("ifm.shape:\t", ifm.shape)
print("wgt.shape:\t", wgt.shape)
print("bias.shape:\t", bias.shape)
print("stride:\t\t", stride)
print("padding:\t", padding)
print("dilation:\t", dilation)
print("groups:\t\t", groups)
print("\r")

# pdb.set_trace()

t = time.perf_counter()
ofm1 = nn.conv2d(ifm,
                 wgt,
                 stride=stride,
                 padding=padding,
                 dilation=dilation,
                 groups=groups)
print("nn.conv2d():\t", format(time.perf_counter() - t, ".6f") + "s")
# print("nn.conv2d() result:\t", ofm1)

t = time.perf_counter()
ofm2 = cpu.conv2d(ifm, wgt, bias, stride, padding, dilation, groups)
print("cpu.conv2d():\t", format(time.perf_counter() - t, ".6f") + "s")
# print("cpu.conv2d() result:\t", ofm2)

t = time.perf_counter()
ofm3 = gpu.conv2d(ifm, wgt, bias, stride, padding, dilation, groups)
print("gpu.conv2d():\t", format(time.perf_counter() - t, ".6f") + "s")
# print("gpu.conv2d() result:\t", ofm3)

print("\r")
print(ofm1.shape)
print(ofm2.shape)
print(ofm3.shape)
print("\r")
# print(ofm1 == ofm2)
# print(ofm1 == ofm3)
