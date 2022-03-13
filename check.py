from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import torch

import lltm_baseline
import lltm



import lltm_cuda
device = torch.device("cuda")

kwargs = {'dtype': torch.float64,
          'device': device,
          'requires_grad': True}
X = torch.randn(3,
                17,
                **kwargs)
h = torch.randn(3, 5, **kwargs)
C = torch.randn(3, 5, **kwargs)
W = torch.randn(3 * 5, 17 +5, **kwargs)
b = torch.randn(1, 3 * 5, **kwargs)

variables = [X, W, b, h, C]

lltm_cuda.forward(X, W, b, h, C)

#check_forward(variables, True, True)

print("x = {}".format(W.flatten()))