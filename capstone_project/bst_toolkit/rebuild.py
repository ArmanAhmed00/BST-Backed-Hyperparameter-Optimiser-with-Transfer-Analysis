from __future__ import annotations
from typing import List, Callable, Tuple
from .registry import HyperparamRegistry
from .node import TrialNode
import random


def rebuild_naive(registry: HyperparamRegistry,
                  evaluate_fn: Callable,
                  new_dataset) -> HyperparamRegistry:
    """
    Strategy 1 — Re-score every trial and insert one by one. 
    WARNING: all_trials() returns nodes in sorted (ascending) order. 
    Inserting a sorted sequence into a BST produces a degenerate tree (like a linked list), giving O(n²) total time instead of O(n log n). 
    This is intentional — you will measure and demonstrate this problem.
    """
    
    new_registry = HyperparamRegistry()

    # 1. Iterate over registry.all_trials()
    for node in registry.all_trials():
        new_score = evaluate_fn(node.params, new_dataset)     # 2. For each node, call evaluate_fn(node.params, new_dataset)
        new_registry.add_trial(new_score, node.params)       # 3. Insert into a new HyperparamRegistry

    return new_registry       # 4. Return the new registry


def rebuild_shuffled(registry: HyperparamRegistry,
                     evaluate_fn: Callable,
                     new_dataset) -> HyperparamRegistry:
    """
    Strategy 2 — Shuffle trials before re-inserting. 
    Breaking the sorted order prevents degenerate insertion. 
    Expected O(n log n) but the resulting tree is not guaranteed balanced.
    """

    new_registry = HyperparamRegistry()

    # 1. Get all trials and shuffle them (random.shuffle)
    trials = registry.all_trials()
    random.shuffle(trials)

    # 2. Re-score and insert into a new registry
    for node in trials:
        new_score = evaluate_fn(node.params, new_dataset)
        new_registry.add_trial(new_score, node.params)

    return new_registry


def rebuild_balanced(registry: HyperparamRegistry,
                     evaluate_fn: Callable,
                     new_dataset) -> HyperparamRegistry:
    """
    Strategy 3 — Build a perfectly balanced BST using divide & conquer. 
    Steps: 
    1. Re-score all trials. 
    2. Sort by new score — O(n log n). 
    3. Call _build_from_sorted() — O(n), guarantees height = floor(log2 n). 
    This is the recommended strategy for Phase 2.
    """

    # 1. re-score all trials
    rescored = []
    for node in registry.all_trials():
        new_score = evaluate_fn(node.params, new_dataset)
        rescored.append((new_score, node.params))

    # 2. sort by score
    rescored.sort(key=lambda x: x[0])

    # 3. build balanced BST
    new_registry = HyperparamRegistry()
    new_registry._bst.root = _build_from_sorted(rescored)

    # Update size manually
    new_registry._bst._size = len(rescored)

    return new_registry


def _build_from_sorted(sorted_trials: List[Tuple[float, dict]]):
    """
    Recursively build a balanced BST from a sorted list of (score, params). 
    Algorithm (Divide & Conquer — same structure as merge sort): 
    - Base case: empty list → return None 
    - Find the middle element → make it the root 
    - Recurse on the left half → left subtree 
    - Recurse on the right half → right subtree 
    Complexity: O(n) time, O(log n) stack space.
    """

    if not sorted_trials:
        return None

    mid = len(sorted_trials) // 2
    score, params = sorted_trials[mid]

    root = TrialNode(score, params)

    # Build left and right recursively
    root.left = _build_from_sorted(sorted_trials[:mid])
    root.right = _build_from_sorted(sorted_trials[mid + 1:])

    return root