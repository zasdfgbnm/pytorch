from collections import defaultdict
from bokeh.layouts import gridplot, column, row
from bokeh.models.widgets import Select
from bokeh.server.server import Server
from bokeh.palettes import Viridis256
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import Panel, Tabs, LogColorMapper
from . import data as data_

def plot1d(source1d0, source1d1, source1d2):
    p1 = figure()
    p1.line(x='x', y='y_baseline', source=source1d0, color='blue', legend_label='baseline')
    p1.line(x='x', y='y_new', source=source1d0, color='red', legend_label='new')
    p1.xaxis.axis_label = 'Tensor size (power of 2)'
    p1.yaxis.axis_label = 'Time (seconds)'

    p2 = figure()
    p2.xaxis.axis_label = 'Tensor size (power of 2)'
    p2.yaxis.axis_label = 'Performance change (percentage)'
    p2.x_range = p1.x_range
    p2.vbar(x='positive_x', bottom=0, top='positive_y', color='green', width='width', source=source1d1)
    p2.vbar(x='negative_x', bottom=0, top='negative_y', color='red', width='width', source=source1d2)
    return gridplot([[p1, p2]])

def plot2d(source):
    low = min(source.data['value_baseline'] + source.data['value_new'])
    high = max(source.data['value_baseline'] + source.data['value_new'])
    mapper = LogColorMapper(palette=Viridis256, low=low, high=high)

    p1 = figure()
    p1.rect(x='x', y='y', source=source, width=2, height=2, fill_color={'field': 'value_baseline', 'transform': mapper}, line_color=None)
    return p1

def is_1d(data):
    return isinstance(data['baseline'][0].problem_size, int)

def plot_experiment(experiment):
    values1d = defaultdict(set)
    values2d = defaultdict(set)
    for setup, data in experiment.items():
        for key, value in setup.items():
            if is_1d(data):
                values1d[key].add(value)
            else:
                values2d[key].add(value)

    # 1D plot
    selects1d = {}
    for k, v in values1d.items():
        selects1d[k] = Select(title=k, value=next(iter(v)), options=list(v))

    def get_data_1d():
        setup = {}
        for k, v in selects1d.items():
            setup[k] = v.value
        data = experiment[data_.hashabledict(setup)]

        baseline = data['baseline']
        new = data['new']
        compare = data['compare']
        x = [x.problem_size for x in baseline]
        y_baseline = [x.result for x in baseline]
        y_new = [x.result for x in new]
        y_compare = [x.result * 100 for x in compare]

        sx = sorted(x)
        if len(sx) >= 2:
            width = (sx[1] - sx[0]) * 0.8
        else:
            width = 0

        positive_x = []
        positive_y = []
        positive_width = []
        negative_x = []
        negative_y = []
        negative_width = []
        for x_, y_ in zip(x, y_compare):
            if y_ >= 0:
                positive_x.append(x_)
                positive_y.append(y_)
                positive_width.append(width)
            if y_ < 0:
                negative_x.append(x_)
                negative_y.append(y_)
                negative_width.append(width)

        return ({'x': x, 'y_baseline': y_baseline, 'y_new': y_new},
                {'positive_x': positive_x, 'positive_y': positive_y, 'width': positive_width},
                {'negative_x': negative_x, 'negative_y': negative_y, 'width': negative_width})

    d = get_data_1d()
    source1d0 = ColumnDataSource(data=d[0])
    source1d1 = ColumnDataSource(data=d[1])
    source1d2 = ColumnDataSource(data=d[2])

    plot1d_ = plot1d(source1d0, source1d1, source1d2)

    def update1d(attrname, old, new):
        source1d0.data, source1d1.data, source1d2.data = get_data_1d()

    for w in selects1d.values():
        w.on_change('value', update1d)

    plot1d_ = row(column(*selects1d.values()), plot1d_)

    # 2D plot
    selects2d = {}
    for k, v in values2d.items():
        selects2d[k] = Select(title=k, value=next(iter(v)), options=list(v))

    def get_data_2d():
        setup = {}
        for k, v in selects2d.items():
            setup[k] = v.value
        data = experiment[data_.hashabledict(setup)]

        baseline = data['baseline']
        new = data['new']
        compare = data['compare']
        x = [x.problem_size[0] for x in baseline]
        y = [x.problem_size[1] for x in baseline]
        value_baseline = [x.result for x in baseline]
        value_new = [x.result for x in new]
        value_compare = [x.result * 100 for x in compare]
        return {'x': x, 'y': y, 'value_baseline': value_baseline, 'value_new': value_new, 'value_compare': value_compare}

    source2d = ColumnDataSource(data=get_data_2d())
    plot2d_ = plot2d(source2d)

    def update2d(attrname, old, new):
        source2d.data = get_data_2d()

    for w in selects2d.values():
        w.on_change('value', update2d)

    plot2d_ = row(column(*selects2d.values()), plot2d_)

    return column(plot1d_, plot2d_)

def serve(compare):
    def make_document(doc):
        tabs = []
        for title, experiment in compare.items():
            plot = plot_experiment(experiment)
            tab = Panel(child=plot, title=title)
            tabs.append(tab)
        tabs = Tabs(tabs=tabs)
        doc.title = 'Benchmark of TensorIterator'
        doc.add_root(tabs)

    server = Server(make_document, port=5000)
    server.run_until_shutdown()
