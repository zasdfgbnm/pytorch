import pkg_resources
from collections import OrderedDict
from jinja2 import Template
from bokeh.resources import INLINE
from bokeh.layouts import gridplot
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.models import LinearAxis, Range1d

def plot1d(data):
    baseline = data['baseline']
    new = data['new']
    compare = data['compare']
    x = [x.problem_size for x in baseline]
    y_baseline = [x.result for x in baseline]
    y_new = [x.result for x in new]
    y_compare = [x.result * 100 for x in compare]

    p1 = figure()
    p1.line(x, y_baseline, color='blue', legend_label='baseline')
    p1.line(x, y_new, color='red', legend_label='new')
    p1.xaxis.axis_label = 'Tensor size (power of 2)'
    p1.yaxis.axis_label = 'Time (seconds)'

    p2 = figure()
    p2.xaxis.axis_label = 'Tensor size (power of 2)'
    p2.yaxis.axis_label = 'Performance change (percentage)'
    p2.x_range = p1.x_range
    positive_x = []
    positive_y = []
    negative_x = []
    negative_y = []
    for x_, y_ in zip(x, y_compare):
        if y_ >= 0:
            positive_x.append(x_)
            positive_y.append(y_)
        if y_ < 0:
            negative_x.append(x_)
            negative_y.append(y_)
    sx = sorted(x)
    width = (sx[1] - sx[0]) * 0.8
    p2.vbar(x=positive_x, bottom=0, top=positive_y, color='green', width=width)
    p2.vbar(x=negative_x, bottom=0, top=negative_y, color='red', width=width)
    return gridplot([[p1, p2]])

def plot2d(data):
    p = figure()
    return p

def plot_data(data):
    if isinstance(data['baseline'][0].problem_size, int):
        return plot1d(data)
    return plot2d(data)

def make_html(compare):
    template_filename = pkg_resources.resource_filename(__name__, "template.html")
    plots = OrderedDict()
    with open(template_filename) as f:
        template = Template(f.read())
    for title, experiment in compare.items():
        for setup, data in experiment.items():
            name = str(setup)
            fig = plot_data(data)
            plots[name] = fig
            break

    script, div = components(plots)
    resources = INLINE.render()
    return template.render(resources=resources, script=script, div=div)
