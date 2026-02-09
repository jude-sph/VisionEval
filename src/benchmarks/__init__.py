"""Benchmark adapters for VLM evaluation."""

from .base import Benchmark, BenchmarkSample
from .mmmu import MMMUBenchmark
from .mmbench import MMBenchBenchmark
from .scienceqa import ScienceQABenchmark
from .pope import POPEBenchmark
from .textvqa import TextVQABenchmark
from .gqa import GQABenchmark

BENCHMARKS = {
    "mmmu": MMMUBenchmark,
    "mmbench": MMBenchBenchmark,
    "scienceqa": ScienceQABenchmark,
    "pope": POPEBenchmark,
    "textvqa": TextVQABenchmark,
    "gqa": GQABenchmark,
}


def get_benchmark(name: str) -> Benchmark:
    """Get a benchmark instance by name."""
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(BENCHMARKS.keys())}")
    return BENCHMARKS[name]()
