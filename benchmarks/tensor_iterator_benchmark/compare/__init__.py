from . import data
from . import plot


def serve(baseline, new):
    baseline = data.load(baseline)
    new = data.load(new)
    compare = data.compare(baseline, new)
    plot.serve(compare)
    # https://matthewrocklin.com/blog//work/2017/06/28/simple-bokeh-server

