# Copyright (c) 2022 Ximeng Sun

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import sys
sys.path.insert(0, '..')
import torch
import numpy as np
from utils.flops_benchmark import add_flops_counting_methods


def count_params(model):
    num_params = 0.
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        v_shape = v.shape
        num_params += np.prod(v_shape)

    print('Number of Parameters = %.2f M' % (num_params/1e6))


def compute_flops(model, input, kwargs_dict):
    model = add_flops_counting_methods(model)
    model = model.cuda().train()

    model.start_flops_count()

    _ = model(input, **kwargs_dict)
    gflops = model.compute_average_flops_cost()

    return gflops