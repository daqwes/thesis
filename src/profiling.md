# Profiling

In order to profile python code, different approaches are possible. One can target the call stack, the lines but also the memory.

## Austin library
The `austin` python library is a frame stack sampler, which also allows to generate a flamegraph as the output format.

Example usage:
```bash
austin -P python <filename.py> | flamegraph fg.svg
```

Here we use the `flamegraph` tool to generate the `svg` file, it is available [here](https://raw.githubusercontent.com/brendangregg/FlameGraph/master/flamegraph.pl).

