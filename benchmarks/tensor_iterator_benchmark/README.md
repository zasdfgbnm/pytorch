TensorIterator Benchmark
========================

TensorIterator is a performance-critical part of the codebase that influences the performance of
all elementwise operations and reductions. Historically there was some regression in TensorIterator
not discovered during code review. TensorIterator is complicated and changing some parts of it might
cause hard-to-realize regression on the other parts. So it is important to run a full benchmark on
TensorIterator for all cases when making changes to TensorIterator.

With the script here, running the benchmark is easy:

**Step 1**:
Install a PyTorch build of the master branch, and run
```
python main.py benchmark baseline.json
```

**Step 2**:
Go to your branch, build install and run
```
python main.py benchmark new.json
```

**Step 3**:
Run the following command to get the report:
```
python main.py compare baseline.json new.json report.html
```