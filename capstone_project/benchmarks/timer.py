from __future__ import annotations

import time
import functools
from typing import Callable


def timed(func: Callable) -> Callable:
   
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000  # seconds → ms
        print(f"[timed] {func.__name__} completed in {elapsed_ms:.3f} ms")
        return result

    return wrapper


def benchmark(fn: Callable, *args, repeats: int = 5, **kwargs) -> float:
    
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    times: list[float] = []

    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    mean_ms = sum(times) / len(times)
    return mean_ms


def benchmark_report(label: str,fn: Callable,*args,repeats: int = 5,**kwargs,) -> dict:
   
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    times: list[float] = []

    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    mean_ms = sum(times) / len(times)
    variance = sum((t - mean_ms) ** 2 for t in times) / len(times)
    std_ms = variance ** 0.5

    return {
        "label":    label,
        "mean_ms":  round(mean_ms, 3),
        "min_ms":   round(min(times), 3),
        "max_ms":   round(max(times), 3),
        "std_ms":   round(std_ms, 3),
        "repeats":  repeats,
    }