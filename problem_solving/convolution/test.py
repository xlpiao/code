## https://pytorch.org/docs/stable/nn.functional.html

import torch
import torch.nn.functional as F
import cpu

ifm = torch.ones(1, 1, 5, 5)
wgt = torch.ones(1, 1, 3, 3) * 2
print(ifm)
print(wgt)

ofm1 = F.conv2d(ifm, wgt, stride = 1, padding = 1, dilation = 1)
print(ofm1)

ofm2 = cpu.conv2d(ifm, wgt, 1, 1, 1)
print(ofm2)
