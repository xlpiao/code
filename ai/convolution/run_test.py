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
IFM_B = 8
IFM_C = 3
IFM_W = 32
IFM_H = 32
WGT_B = 8
WGT_C = IFM_C
WGT_W = 3
WGT_H = 3
BIAS_SIZE = WGT_B
STRIDE = 1
PADDING = 1
DILATION = 1
GROUPS = 1

ifm = (torch.rand(IFM_B, IFM_C, IFM_H, IFM_W) * 10).int().float()  #input
wgt = (torch.rand(WGT_B, WGT_C, WGT_H, WGT_W) * 10).int().float()  #weight
bias = (torch.rand(BIAS_SIZE) * 0).int().float()
stride = STRIDE
padding = PADDING
dilation = DILATION
groups = GROUPS
print("\r")

## unfold
"""
t = time.perf_counter()
im2col_ifm = nn.unfold(ifm, (WGT_H, WGT_W),
                       dilation=dilation,
                       padding=padding,
                       stride=stride)
print("nn.unfold():\t", format(time.perf_counter() - t, ".6f") + "s")
print(im2col_ifm.shape)
print("\r")

t = time.perf_counter()
im2col_ifm1 = cpu.unfold_v1(ifm, wgt, bias, stride, padding, dilation, groups)
print("cpu.unfold():\t", format(time.perf_counter() - t, ".6f") + "s")
print(im2col_ifm1.shape)
print((im2col_ifm == im2col_ifm1).all())
print("\r")

t = time.perf_counter()
im2col_ifm2 = cpu.unfold_v2(ifm, wgt, bias, stride, padding, dilation, groups)
print("cpu.unfold():\t", format(time.perf_counter() - t, ".6f") + "s")
print(im2col_ifm2.shape)
print((im2col_ifm == im2col_ifm2).all())
print("\r")

t = time.perf_counter()
im2col_ifm3 = gpu.unfold(ifm, wgt, bias, stride, padding, dilation, groups)
print("gpu.unfold():\t", format(time.perf_counter() - t, ".6f") + "s")
print(im2col_ifm3.shape)
print((im2col_ifm == im2col_ifm3).all())
print("\r")

print("ifm.shape:\t", ifm.shape)
print("wgt.shape:\t", wgt.shape)
print("bias.shape:\t", bias.shape)
print("stride:\t\t", stride)
print("padding:\t", padding)
print("dilation:\t", dilation)
print("groups:\t\t", groups)
print("\r")
"""

## pdb.set_trace()

t = time.perf_counter()
ofm1 = nn.conv2d(ifm,
                 wgt,
                 bias,
                 stride=stride,
                 padding=padding,
                 dilation=dilation,
                 groups=groups)
print("nn.conv2d():\t", format(time.perf_counter() - t, ".6f") + "s")
print(ofm1.shape)
print("\r")

t = time.perf_counter()
ofm2 = cpu.conv2d(ifm, wgt, bias, stride, padding, dilation, groups)
print("cpu.conv2d():\t", format(time.perf_counter() - t, ".6f") + "s")
print(ofm2.shape)
print((ofm1 == ofm2).all())
print("\r")

t = time.perf_counter()
ofm3 = gpu.conv2d(ifm, wgt, bias, stride, padding, dilation, groups)
print("gpu.conv2d():\t", format(time.perf_counter() - t, ".6f") + "s")
print(ofm3.shape)
print((ofm1 == ofm3).all())
print("\r")
