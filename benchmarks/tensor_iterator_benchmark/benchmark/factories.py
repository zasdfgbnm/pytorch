import torch

padding = 1

def trivial_1d(shape, dtype=None, device=None):
    return torch.empty(shape, dtype=dtype, device=device)

def contiguous_last_dim(shape, dtype=None, device=None):
    underlying_shape = tuple(s + padding for s in shape)
    t = trivial_1d(underlying_shape, dtype, device)
    for i, s in enumerate(shape):
        t = t.narrow(i, 0, s)
    return t

def non_contiguous(shape, dtype=None, device=None):
    return contiguous_last_dim((*shape, 1), dtype, device).squeeze(-1)
