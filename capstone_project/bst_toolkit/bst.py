from __future__ import annotations
from typing import Optional, List
from .node import TrialNode

class BST:
    

    def __init__(self) -> None:
        self.root: Optional[TrialNode] = None
        self._size: int = 0

    def insert(self, score: float, params: dict) -> None:
        
        # Round to 6 decimals to minimize floating-point collisions
        score = round(score, 6)
        new_root = self._insert(self.root, score, params)
        if self.root is None or self._size_changed:
            self.root = new_root

    def delete(self, score: float) -> None:
        
        self.root, was_deleted = self._delete(self.root, score)
        if was_deleted:
            self._size -= 1

    def search(self, score: float) -> Optional[TrialNode]:
        """Return the node with this score, or None if not found."""
        return self._search(self.root, score)

    def find_min(self) -> Optional[TrialNode]:
        """Return the node with the lowest score (leftmost node)."""
        if not self.root: return None
        return self._find_min(self.root)

    def find_max(self) -> Optional[TrialNode]:
        """Return the node with the highest score (rightmost node)."""
        curr = self.root
        while curr and curr.right:
            curr = curr.right
        return curr

    def height(self) -> int:
        """Return the height of the tree (0 for an empty tree)."""
        return self._height(self.root)

    def is_balanced(self) -> bool:
        """Return True if the tree is balanced (|h_left - h_right| <= 1)."""
        return self._check_balanced(self.root) != -1

    def __len__(self) -> int:
        return self._size

    # --- Traversals ---

    def inorder(self) -> List[TrialNode]:
        """Left -> Node -> Right (Sorted ASCENDING)."""
        result = []
        self._inorder(self.root, result)
        return result

    def preorder(self) -> List[TrialNode]:
        """Node -> Left -> Right (Used for serialization)."""
        result = []
        self._preorder(self.root, result)
        return result

    def postorder(self) -> List[TrialNode]:
        """Left -> Right -> Node (Used for safe deletion)."""
        result = []
        self._postorder(self.root, result)
        return result

    # --- Private Helpers ---

    def _insert(self, node: Optional[TrialNode], score: float, params: dict) -> TrialNode:
        self._size_changed = False
        if node is None:
            self._size += 1
            self._size_changed = True
            return TrialNode(score=score, params=params)
        
        if score < node.score:
            node.left = self._insert(node.left, score, params)
        elif score > node.score:
            node.right = self._insert(node.right, score, params)
        # If score == node.score, do nothing (first-inserted wins)
        return node

    def _delete(self, node: Optional[TrialNode], score: float) -> tuple[Optional[TrialNode], bool]:
        if node is None:
            return None, False
        
        if score < node.score:
            node.left, deleted = self._delete(node.left, score)
            return node, deleted
        elif score > node.score:
            node.right, deleted = self._delete(node.right, score)
            return node, deleted
        else:
            # Case 1 & 2: Leaf or One Child
            if node.left is None: return node.right, True
            if node.right is None: return node.left, True
            
            # Case 3: Two Children (Successor)
            successor = self._find_min(node.right)
            node.score = successor.score
            node.params = successor.params
            node.right, _ = self._delete(node.right, successor.score)
            return node, True

    def _search(self, node: Optional[TrialNode], score: float) -> Optional[TrialNode]:
        if node is None or node.score == score:
            return node
        return self._search(node.left, score) if score < node.score else self._search(node.right, score)

    def _find_min(self, node: TrialNode) -> TrialNode:
        curr = node
        while curr.left:
            curr = curr.left
        return curr

    def _height(self, node: Optional[TrialNode]) -> int:
        if node is None: return 0
        return 1 + max(self._height(node.left), self._height(node.right))

    def _check_balanced(self, node: Optional[TrialNode]) -> int:
        if node is None: return 0
        lh = self._check_balanced(node.left)
        if lh == -1: return -1
        rh = self._check_balanced(node.right)
        if rh == -1: return -1
        
        if abs(lh - rh) > 1: return -1
        return 1 + max(lh, rh)

    def _inorder(self, node: Optional[TrialNode], result: list):
        if node:
            self._inorder(node.left, result)
            result.append(node)
            self._inorder(node.right, result)

    def _preorder(self, node: Optional[TrialNode], result: list):
        if node:
            result.append(node)
            self._preorder(node.left, result)
            self._preorder(node.right, result)

    def _postorder(self, node: Optional[TrialNode], result: list):
        if node:
            self._postorder(node.left, result)
            self._postorder(node.right, result)
            result.append(node)