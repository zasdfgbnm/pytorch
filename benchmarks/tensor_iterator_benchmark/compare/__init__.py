from . import data
from . import plot


def generate_html(baseline, new, report):
    baseline = data.load(baseline)
    new = data.load(new)
    compare = data.compare(baseline, new)
    html = plot.make_html(compare)
    with open(report, 'w') as f:
        f.write(html)
