from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrialNode:
    """
    A single node in the hyperparameter BST.
    The BST is keyed by `score`.

    Attributes
    ----------
    score : the evaluation metric for this trial (e.g. accuracy)
    params : the hyperparameter dictionary used in this trial
    left : left child node (score strictly less than this node)
    right : right child node (score strictly greater than this node)
    """

    score: float
    params: dict
    left: Optional["TrialNode"] = field(default=None, repr=False)
    right: Optional["TrialNode"] = field(default=None, repr=False)

    def __lt__(self, other: "TrialNode") -> bool:
        """
        Compare nodes based on score (BST ordering).
        """
        # TODO: return True if this node's score is less than the other's
        return self.score < other.score

    def __repr__(self) -> str:
        # TODO: return a readable string
        return f"TrialNode(score={self.score:.6f}, params={self.params})"
    

