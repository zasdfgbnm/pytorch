import torch
import gc
from . import layout_shape, timing

selected_dtypes = [torch.float16, torch.float32]
all_dtypes = torch.testing.get_all_dtypes()
floating_point_dtypes = [x for x in all_dtypes if x.is_floating_point]

all_dtype_ops = ['abs', 'logical_not', 'sign']

floating_points_ops = [
    'acos', 'asin', 'ceil', 'expm1', 'frac', 'floor', 'log', 'log10', 'log2', 'log1p',
    'round', 'trunc', 'rsqrt', 'sin', 'sinh', 'sqrt', 'sigmoid', 'erfinv', 'digamma',
    'trigamma', 'lgamma',
]

selected_ops = ['logical_not', 'logical_not_', 'abs', 'abs_', 'rsqrt', 'rsqrt_', 'digamma', 'digamma_']

def compare_problem_sizes():
    title = "unary op compare problem sizes"
    for op in selected_ops:
        for dtype in selected_dtypes:

            def setup(device, non_contiguous_size=None):
                return {
                    'op': op,
                    'dtype': str(dtype),
                    'layout': name,
                    'device': device,
                    'non_contiguous_size': non_contiguous_size,
                }

            def benchmark_cpu(factories):
                print('Benchmarking', op, 'with dtype', dtype, 'and layout', name, 'on cpu')
                data = []
                for factory in factories:
                    tensor = factory.new(dtype, 'cpu')
                    f = getattr(tensor, op)
                    one_loop_timer = timing.time_one_loop(f)
                    result = timing.time_func(one_loop_timer)
                    data.append(({'problem_size': factory.problem_size, 'result': result}))
                    del tensor, one_loop_timer, f
                    gc.collect()
                return data

            def benchmark_cuda(factories):
                print('Benchmarking', op, 'with dtype', dtype, 'and layout', name, 'on cuda')
                data = []
                for factory in factories:
                    tensor = factory.new(dtype, 'cuda')
                    f = getattr(tensor, op)
                    one_loop_timer = timing.time_one_loop_cuda(f)
                    result = timing.time_func(one_loop_timer)
                    data.append(({'problem_size': factory.problem_size, 'result': result}))
                    del tensor, one_loop_timer, f
                    gc.collect()
                return data

            for name, factories in layout_shape.full1d.items():
                if dtype is not torch.float16:
                    yield (title, {'setup': setup('cpu'), 'data': benchmark_cpu(factories)})
                yield (title, {'setup': setup('cuda'), 'data': benchmark_cuda(factories)})

            for name, d in layout_shape.full2d.items():
                for non_contiguous_size, factories in d.items():
                    if dtype is not torch.float16:
                        yield (title, {'setup': setup('cpu', non_contiguous_size), 'data': benchmark_cpu(factories)})
                    yield (title, {'setup': setup('cuda', non_contiguous_size), 'data': benchmark_cuda(factories)})


def run():
    yield from compare_problem_sizes()
