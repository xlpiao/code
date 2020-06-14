## https://pytorch.org/docs/stable/nn.functional.html

'''
torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) → Tensor

input: input tensor of shape (\text{minibatch} , \text{in\_channels} , iH , iW)(minibatch,in_channels,iH,iW)
weight: filters of shape (\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW)(out_channels,
        groups
        in_channels
        ​
         ,kH,kW)
bias: optional bias tensor of shape (\text{out\_channels})(out_channels) . Default: None
stride: the stride of the convolving kernel. Can be a single number or a tuple (sH, sW). Default: 1
padding: implicit paddings on both sides of the input. Can be a single number or a tuple (padH, padW). Default: 0
dilation: the spacing between kernel elements. Can be a single number or a tuple (dH, dW). Default: 1
groups: split input into groups, \text{in\_channels}in_channels should be divisible by the number of groups. Default: 1
'''

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

input = torch.Tensor(np.array([[[
    [1,1,1,0,0],
    [0,1,1,1,0],
    [0,0,1,1,1],
    [0,0,1,1,0],
    [0,1,1,0,0]
]]]))

filter = torch.Tensor(np.array([[[
    [1,0,1],
    [0,1,0],
    [1,0,1]
]]]))

input = Variable(input)
filter = Variable(filter)
out = F.conv2d(input, filter)
# print(input)
# print(filter)
# print(out)


'''
torch.randn(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
'''
input = torch.randn(1,1,5,5)
filter = torch.randn(1,1,3,3)
out = F.conv2d(input, filter, padding=1)
print(input)
print(filter)
print(out)
