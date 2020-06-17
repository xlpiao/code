## https://pytorch.org/docs/stable/nn.functional.html

import torch
import torch.nn.functional as F

input = torch.ones(1, 1, 1, 8)
kernel = torch.ones(1, 1, 1, 5) * 2
output = F.conv1d(input, kernel, stride = 2, padding = 2)
print(input)
print(kernel)
print(output)

input = torch.ones(1, 1, 8, 8)
kernel = torch.ones(1, 1, 5, 5) * 2
output = F.conv2d(input, kernel, stride = 2, padding = 2)
print(input)
print(kernel)
print(output)

input = torch.ones(1, 8, 8, 8)
kernel = torch.ones(1, 5, 5, 5) * 2
input = input.unsqueeze(0)
kernel = kernel.unsqueeze(0)
output = F.conv3d(input, kernel, stride = 2, padding = 2)
print(input)
print(kernel)
print(output)
