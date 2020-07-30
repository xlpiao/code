## https://pytorch.org/docs/stable/nn.functional.html

import torch
import torch.nn.functional as F
import cpu

stride = 1
padding = 1
dilation = 1

ifm = (torch.rand(1, 3, 5, 5) * 10).int().float()
wgt = (torch.rand(2, 3, 3, 3) * 10).int().float()
print(ifm)
print(wgt)

ofm1 = F.conv2d(ifm, wgt, stride = stride, padding = padding, dilation = dilation)
print(ofm1)

ofm2 = cpu.conv2d(ifm, wgt, stride, padding, dilation)
print(ofm2)

print(ofm1.shape)
print(ofm2.shape)
print(ofm1 == ofm2)
