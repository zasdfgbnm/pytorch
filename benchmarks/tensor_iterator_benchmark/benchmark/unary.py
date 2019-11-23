import torch
import gc
from . import layout_shape, timing


selected_dtypes = [torch.float16, torch.float32, torch.float64]
all_dtypes = torch.testing.get_all_dtypes()
floating_point_dtypes = [x for x in all_dtypes if x.is_floating_point]

all_dtype_ops = ['abs', 'logical_not', 'sign']

floating_points_ops_small = ['floor', 'sin', 'digamma']
floating_points_ops_full = [
    'acos', 'asin', 'ceil', 'expm1', 'frac', 'floor', 'log', 'log10', 'log2', 'log1p',
    'round', 'trunc', 'rsqrt', 'sin', 'sinh', 'sqrt', 'sigmoid', 'erfinv', 'digamma',
    'trigamma', 'lgamma',
]


def compare_problem_sizes():
    title = "unary op compare problem sizes"
    for op in all_dtype_ops + floating_points_ops_small:
        with_inplace = [op, op + '_']
        for op in with_inplace:
            for dtype in selected_dtypes:
                for name, factories in layout_shape.full.items():

                    def setup(device):
                        return {
                            'op': op,
                            'dtype': str(dtype),
                            'layout': name,
                            'device': device,
                        }

                    # benchmark cpu
                    if dtype is not torch.float16:
                        print('Benchmarking', op, 'with dtype', dtype, 'and layout', name, 'on cpu')
                        data = []
                        for factory in factories:
                            tensor = factory.new(dtype, 'cpu')
                            f = getattr(tensor, op)
                            one_loop_timer = timing.time_one_loop(f)
                            result = timing.time_func(one_loop_timer)
                            data.append((factory.problem_size, result))
                            del tensor, one_loop_timer, f
                            gc.collect()
                        yield (title, {'setup': setup('cpu'), 'data': data})

                    # benchmark cuda
                    print('Benchmarking', op, 'with dtype', dtype, 'and layout', name, 'on cuda')
                    data = []
                    for factory in factories:
                        tensor = factory.new(dtype, 'cuda')
                        f = getattr(tensor, op)
                        one_loop_timer = timing.time_one_loop_cuda(f)
                        result = timing.time_func(one_loop_timer)
                        data.append((factory.problem_size, result))
                        del tensor, one_loop_timer, f
                        gc.collect()
                    yield (title, {'setup': setup('cuda'), 'data': data})


def run():
    yield from compare_problem_sizes()
