import numpy as np
import torch
from operator import mul

def linear_expectation(probs, values):
    assert(len(values) == probs.ndimension() - 2)
    expectation = []
    for i in range(2, probs.ndimension()):
        # Marginalise probabilities
        marg = probs
        for j in range(probs.ndimension() - 1, 1, -1):
            if i != j:
                marg = marg.sum(j, keepdim=False)
        # Calculate expectation along axis `i`
        expectation.append((marg * values[len(expectation)]).sum(-1, keepdim=False))
    return torch.stack(expectation, -1)

def normalized_linspace(length, dtype=None, device=None):
    first = -(length - 1) /float(length)
    last = (length - 1) /float(length)
    return torch.linspace(first, last, length, dtype=dtype, device=device)

def soft_argmax(heatmaps):
    values = [normalized_linspace(d, dtype=heatmaps.dtype, device=heatmaps.device)
              for d in heatmaps.size()[2:]]
    return linear_expectation(heatmaps, values)

def flat_softmax(inp):
    """Compute the softmax with all but the first two tensor dimensions combined."""
    orig_size = inp.size()
    flat = inp.view(-1, reduce(mul, orig_size[2:]))
    flat = torch.nn.functional.softmax(flat, -1)
    return flat.view(*orig_size)


input_success = torch.Tensor([[[
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 20.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],]]])
input_fail = torch.Tensor([[[
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 10.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],]]])
probs = flat_softmax(input_success)
values = [torch.arange(d, dtype=probs.dtype, device=probs.device) for d in probs.size()[2:]]
output = linear_expectation(probs, values)
print output

probs = flat_softmax(input_fail)
values = [torch.arange(d, dtype=probs.dtype, device=probs.device) for d in probs.size()[2:]]
output = linear_expectation(probs, values)
print output

input_fail = input_fail*20
e = torch.exp(input_fail - torch.max(input_fail))
s = torch.sum(e)
print e
probs = e/s
probs = probs - probs.min()
probs = probs/probs.sum()
print probs

x = values[0].repeat(4,1)
y = values[1].repeat(4,1).transpose(0,1)
print (probs[0][0]*x).sum()
print (probs[0][0]*y).sum()

