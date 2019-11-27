from typing import NamedTuple, Tuple, Union
from . import factories
from collections import defaultdict

# all sizes are in power of 2
max_size = 29
sizes_full = range(8, max_size + 1, 2)
sizes_small = [10, 20, max_size]

layouts_full = [
    ("contiguous",),
    ("contiguous", "contiguous", "contiguous"),
    ("non_contiguous",),
    ("non_contiguous", "non_contiguous", "non_contiguous"),
    ("contiguous", "non_contiguous"),
    ("contiguous", "non_contiguous", "non_contiguous", "non_contiguous"),
]

layouts_small = [
    ("contiguous",),
    ("non_contiguous", "non_contiguous", "non_contiguous"),
    ("contiguous", "non_contiguous", "non_contiguous", "non_contiguous"),
]


class LayoutShape(NamedTuple):
    problem_size: int
    factory: callable
    shape: Tuple

    def new(self, dtype=None, device=None):
        return self.factory(self.shape, dtype, device)


def split_size(size, dims):
    s1 = size // dims
    s2 = size - s1 * (dims - 1)
    return (s1,) * (dims - 1) + (s2,)


def make_layout_shape(layout, contiguous_size=0, non_contiguous_size=0):
    countiguous_dims = 0
    non_contiguous_dims = 0
    for i in layout:
        if i == 'contiguous':
            countiguous_dims += 1
        else:
            assert i == 'non_contiguous'
            non_contiguous_dims += 1
    if non_contiguous_dims == 0:
        assert countiguous_dims > 0
        name = f"all contiguous {countiguous_dims}d"
        problem_size = contiguous_size
        factory = factories.trivial_1d
        shape = split_size(contiguous_size, countiguous_dims)
    elif countiguous_dims == 0:
        assert non_contiguous_dims > 0
        name = f"all non-contiguous {non_contiguous_dims}d"
        problem_size = non_contiguous_size
        factory = factories.non_contiguous
        shape = split_size(non_contiguous_size, non_contiguous_dims)
    else:
        assert countiguous_dims == 1
        assert non_contiguous_dims > 0
        assert layout[0] == 'contiguous'
        name = f"contiguous 1d and non-contiguous {non_contiguous_dims}d"
        problem_size = contiguous_size + non_contiguous_size
        factory = factories.contiguous_last_dim
        shape = split_size(non_contiguous_size, non_contiguous_dims) + (contiguous_size,)
    shape = tuple(2 ** x for x in shape)
    return name, LayoutShape(problem_size, factory, shape)


def combine_layouts_and_shapes(layouts, sizes):
    ret1d = defaultdict(list)
    ret2d = defaultdict(lambda: defaultdict(list))
    for layout in layouts:
        if set(layout) == {'contiguous'}:
            for size in sizes:
                name, result = make_layout_shape(layout, contiguous_size=size)
                ret1d[name].append(result)
        elif set(layout) == {'non_contiguous'}:
            for size in sizes:
                name, result = make_layout_shape(layout, non_contiguous_size=size)
                ret1d[name].append(result)
        else:
            assert set(layout) == {'contiguous', 'non_contiguous'}
            for size1 in sizes:
                for size2 in sizes:
                    if size1 + size2 <= max_size:
                        name, result = make_layout_shape(layout, size1, size2)
                        ret2d[name][size2].append(result)
    return ret1d, ret2d


full1d, full2d = combine_layouts_and_shapes(layouts_full, sizes_full)
small1d, small2d = combine_layouts_and_shapes(layouts_small, sizes_small)
