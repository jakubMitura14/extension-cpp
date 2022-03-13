from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import torch

import lltm_baseline
import lltm



from torch.utils.cpp_extension import load
lltm_cuda = load(
    'lltm_cuda', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'], verbose=True)
help(lltm_cuda)



import lltm_cuda
device = torch.device("cuda")

kwargs = {'dtype': torch.float64,
          'device': device,
          'requires_grad': True}
X = torch.randn(5).to(device)
Y = torch.randn(5).to(device)
lltm_cuda.forwardB(X, Y)

#check_forward(variables, True, True)

print("x = {}".format(X.flatten()))