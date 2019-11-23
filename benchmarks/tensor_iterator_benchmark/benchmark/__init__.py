import torch
from .result import results
from . import unary


def warm_up_cuda():
    a = torch.randn(100 * 1024 * 1024)
    for _ in range(10):
        _ = a + a


def run():
    warm_up_cuda()
    for title, result in unary.run():
        results[title].append(result)
