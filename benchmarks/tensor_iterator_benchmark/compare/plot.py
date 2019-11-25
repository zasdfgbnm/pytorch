import pkg_resources

def plot_data(data):
    pass

def make_html(compare):
    html_template_filename = pkg_resources.resource_filename(__name__, "template.html")
    js_template_filename = pkg_resources.resource_filename(__name__, "template.js")
    with open(html_template_filename) as f:
        html_template = f.read()
    with open(js_template_filename) as f:
        js_template = f.read()
    for title, experiment in compare.items():
        for setup, data in experiment.items():
            plot_data(data)
    html = html_template.format(script=js_template)
    return html
