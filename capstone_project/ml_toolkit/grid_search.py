from __future__ import annotations

import itertools
from typing import Callable

from tqdm import tqdm

from capstone_project.bst_toolkit.registry import HyperparamRegistry


def grid_search(param_grid: dict,evaluate_fn: Callable,dataset,verbose: bool = True,) -> HyperparamRegistry:
   
    registry = HyperparamRegistry()

    keys = list(param_grid.keys())
    value_lists = list(param_grid.values())

    
    combos = list(itertools.product(*value_lists))

    iterator = tqdm(combos, desc="Grid Search", unit="trial") if verbose else combos

    for values in iterator:
        params = dict(zip(keys, values))

        score = evaluate_fn(params, dataset)
        
        score = round(float(score), 6)

        registry.add_trial(score, params)

        if verbose:
            iterator.set_postfix(score=f"{score:.4f}", params=str(params))

    return registry

