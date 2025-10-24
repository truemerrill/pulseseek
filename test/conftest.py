import jax
import pytest


@pytest.fixture
def compile_then_benchmark(benchmark):
    """Benchmark the execution time (not compile time)."""

    def run(fn, *args, **kwargs):
        # Compile JIT function
        jax.block_until_ready(fn(*args, **kwargs))

        # Wrapped function used in benchmarking
        def call():
            return jax.block_until_ready(fn(*args, **kwargs))
        return benchmark(call)
    return run
