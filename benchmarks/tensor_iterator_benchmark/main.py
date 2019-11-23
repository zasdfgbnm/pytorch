import argparse

# Usage
# -----
#
# Run the benchmark and write the result to a json file:
#   python main.py benchmark output.json
#
# Compare a new benchmark result with a baseline and render it to HTML:
#   python main.py compare baseline.json new.json report.html

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Benchmark TensorIterator')
    # subs = parser.add_subparsers()
    # benchmark_parser = subs.add_parser('benchmark', description='Benchmark TensorIterator and write result to a json file')
    # benchmark_parser.add_argument('output', help='Name of the output json file')
    # compare_parser = subs.add_parser('compare', description='Compare a new benchmark result with a baseline and render it to HTML')
    # compare_parser.add_argument('baseline', help='Name of the json file used as baseline')
    # compare_parser.add_argument('new', help='Name of the json file for the new result')
    # compare_parser.add_argument('report', help='Name of the HTML file for the report')
    # args = parser.parse_args()

    import benchmark
    benchmark.run()
    print(benchmark.results)
