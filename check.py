from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import torch

import lltm_baseline
import lltm


def check_equal(first, second, verbose):
    if verbose:
        print()
    for i, (x, y) in enumerate(zip(first, second)):
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        if verbose:
            print("x = {}".format(x.flatten()))
            print("y = {}".format(y.flatten()))
            print('-' * 80)
        np.testing.assert_allclose(x, y, err_msg="Index: {}".format(i))


def zero_grad(variables):
    for variable in variables:
        variable.grad.zero_()


def get_grads(variables):
    return [var.grad.clone() for var in variables]


def check_forward(variables, with_cuda, verbose):
    baseline_values = lltm_baseline.LLTMFunction.apply(*variables)
    cpp_values = lltm.LLTMFunction.apply(*variables)

    print('Forward: Baseline (Python) vs. C++ ... ', end='')
    check_equal(baseline_values, cpp_values, verbose)
    print('Ok')

    if with_cuda:
        cuda_values = lltm.LLTMFunction.apply(*variables)
        print('Forward: Baseline (Python) vs. CUDA ... ', end='')
        check_equal(baseline_values, cuda_values, verbose)
        print('Ok')



import lltm
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

check_forward(variables, True, True)

