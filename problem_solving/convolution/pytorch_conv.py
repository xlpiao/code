## https://pytorch.org/docs/stable/nn.functional.html

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

input = torch.ones(1, 1, 1, 8)
filter = torch.ones(1, 1, 1, 5) * 2
out = F.conv1d(input, filter, stride = 2, padding = 2)
print(input)
print(filter)
print(out)

input = torch.ones(1, 1, 8, 8)
filter = torch.ones(1, 1, 5, 5) * 2
out = F.conv2d(input, filter, stride = 2, padding = 2)
print(input)
print(filter)
print(out)
