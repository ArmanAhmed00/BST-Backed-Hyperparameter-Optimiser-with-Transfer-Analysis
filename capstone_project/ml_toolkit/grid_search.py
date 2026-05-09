from __future__ import annotations

import itertools
from typing import Callable

from tqdm import tqdm

from capstone_project.bst_toolkit.registry import HyperparamRegistry


def grid_search(
    param_grid: dict,       # e.g. {"n_estimators": [50, 100], "max_depth": [3, 5]}
    evaluate_fn: Callable,  # function(params: dict, dataset) -> float
    dataset,                # anything your evaluate_fn understands
    verbose: bool = True,   # show tqdm progress bar
) -> HyperparamRegistry:
    """
    Exhaustive hyperparameter search over every combination in param_grid.

    Algorithm
    ---------
    Uses itertools.product (brute-force cartesian product — Session 1) to
    enumerate all combinations, evaluates each one, and stores results in a
    HyperparamRegistry (BST keyed by accuracy score).

    Parameters
    ----------
    param_grid   : dict mapping parameter names to lists of candidate values.
    evaluate_fn  : callable(params, dataset) -> float — must return a score
                   already rounded to 6 decimals to avoid BST collisions.
    dataset      : passed through to evaluate_fn unchanged.
    verbose      : if True, display a tqdm progress bar with trial info.

    Returns
    -------
    HyperparamRegistry populated with one node per unique score.
    """
    registry = HyperparamRegistry()

    keys = list(param_grid.keys())
    value_lists = list(param_grid.values())

    # Total number of combinations for the progress bar
    combos = list(itertools.product(*value_lists))

    iterator = tqdm(combos, desc="Grid Search", unit="trial") if verbose else combos

    for values in iterator:
        params = dict(zip(keys, values))

        score = evaluate_fn(params, dataset)
        # Defensive rounding — evaluate_fn should already round, but just in case
        score = round(float(score), 6)

        registry.add_trial(score, params)

        if verbose:
            iterator.set_postfix(score=f"{score:.4f}", params=str(params))

    return registry

