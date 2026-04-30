from __future__ import annotations
from typing import List, Optional

from .bst import BST
from .node import TrialNode

class HyperparamRegistry:
    """
    High-level interface around BST for managing hyperparameter trials.
    Provides range queries, top-k retrieval, pruning, and summaries.
    """

    def __init__(self) -> None:
        # TODO: create self._bst = BST() and self._history = []

        self._bst = BST()
        self._history = []

    def add_trial(self, score: float, params: dict) -> None:
        """
        Record a new trial: insert into BST and append to history log.
        Collision handling: if a trial with exact score already exists
        in the BST, the existing params are kept (first-inserted wins) and 
        this call is ignored silently. Round scores to 6 decimals before 
        calling this method to minimise accidental collisions from floating-point noise.
        """
        before_size = len(self._bst)

        self._bst.insert(score, params)

        after_size = len(self._bst)

        if after_size > before_size:
            self._history.append((score, params))

    def best(self) -> Optional[TrialNode]:
        """Return the highest-scoring trial. Uses BST find_max()."""
        return self._bst.find_max()

    def worst(self) -> Optional[TrialNode]:
        """Return the lowest-scoring trial. Uses BST find_min"""
        return self._bst.find_min()

    def top_k(self, k: int) -> List[TrialNode]:
        """
        Return the k highest-scoring trials in descending order.
        Hint: use a reverse in-order traversal (Right → Node → Left). 
        Complexity: O(k + h).

        """
        result = []
        self._reverse_inorder(self._bst.root, result, k)
        return result

    def range_query(self, lo: float, hi: float) -> List[TrialNode]:
        """
        Return all trials with lo <= score <= hi, sorted ascending.
        Hint: use BST pruning — don't explore a subtree if it can't contain values in the range.
        Complexity: O(k + h) where k = number of results.

        """
        result = []
        self._range(self._bst.root, lo, hi, result)
        return result

    def prune_below(self, threshold: float) -> int:
        """
        Delete all trials with score < threshold.
        Returns the number of deleted nodes.
        Hint: first collect all scores to delete via inorder (),
        then delete them one by one.

        """
        nodes_to_delete = []

        for node in self._bst.inorder():
            if node.score < threshold:
                nodes_to_delete.append(node.score)

        for score in nodes_to_delete:
            self._bst.delete(score)

        return len(nodes_to_delete)

    def all_trials(self) -> List[TrialNode]:
        """Return all trials sorted ascending by score. (in-order)."""
        return self._bst.inorder()

    def summary(self) -> dict:
        """
        Return a dict with: count, best score, worst score, mean score, tree height, is_balanced.
        """
        trials = self.all_trials()

        if len(trials) == 0:
            return {
                "count": 0,
                "best_score": None,
                "worst_score": None,
                "mean_score": None,
                "tree_height": self._bst.height(),
                "is_balanced": self._bst.is_balanced(),
            }

        scores = [node.score for node in trials]

        return {
            "count": len(trials),
            "best_score": self.best().score,
            "worst_score": self.worst().score,
            "mean_score": sum(scores) / len(scores),
            "tree_height": self._bst.height(),
            "is_balanced": self._bst.is_balanced(),
        }

    def _reverse_inorder(self, node, result, k):
        """
        Right → Node → Left traversal. stops when len(result) == k.
        """
        if node is None:
            return

        if len(result) >= k:
            return

        self._reverse_inorder(node.right, result, k)

        if len(result) < k:
            result.append(node)

        self._reverse_inorder(node.left, result, k)

    def _range(self, node, lo, hi, result):
        """
        Collect all nodes with lo <= score <= hi.
        Prune: only go left if node.score > lo;
               only go right if node.score < hi.

        """
        if node is None:
            return

        if node.score > lo:
            self._range(node.left, lo, hi, result)

        if lo <= node.score <= hi:
            result.append(node)

        if node.score < hi:
            self._range(node.right, lo, hi, result)


#Test
if __name__ == "__main__":
    from bst_toolkit.registry import HyperparamRegistry

    reg = HyperparamRegistry()

    reg.add_trial(0.90, {"model": "A"})
    reg.add_trial(0.95, {"model": "B"})
    reg.add_trial(0.85, {"model": "C"})
    reg.add_trial(0.92, {"model": "D"})

    print("All trials:", reg.all_trials())
    print("Best:", reg.best())
    print("Worst:", reg.worst())

    print("Top 2:", reg.top_k(2))
    print("Range 0.88 → 0.95:", reg.range_query(0.88, 0.95))

    print("Summary:", reg.summary())