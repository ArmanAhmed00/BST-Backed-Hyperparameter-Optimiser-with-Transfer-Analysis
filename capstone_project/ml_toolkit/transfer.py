from __future__ import annotations

from typing import List

from capstone_project.bst_toolkit.registry import HyperparamRegistry


def analyse_transfer(
    registry_a: HyperparamRegistry,
    registry_b: HyperparamRegistry,
    good_drift_threshold: int = 0,
) -> List[dict]:
    """
    Compare hyperparameter rankings between two registries and return a
    ranked transfer report.

    The report answers: for every configuration that was evaluated on both
    Dataset A and Dataset B, did it improve, stay stable, or degrade?

    Algorithm
    ---------
    1. In-order traversal of both BSTs → sorted lists ascending by score  O(n)
       (Session 5 — BST traversal property).
    2. Build a rank lookup dict from each traversal using a hash table     O(n)
       (Session 1 — hash maps).  Rank 1 = best (highest-scoring).
    3. For each config in registry_A, look up its score in registry_B.
    4. Compute drift = rank_A − rank_B.  Positive drift means the config
       moved UP in ranking (improved) when transferred.
    5. Sort the report by drift descending — use Python's sorted() with a
       key (Session 3 — comparison-based sorting).

    Parameters
    ----------
    registry_a            : HyperparamRegistry built on Dataset A.
    registry_b            : HyperparamRegistry built on Dataset B.
    good_drift_threshold  : drift >= this value is labelled "✓ good"
                            (default 0 — any improvement or no change).

    Returns
    -------
    List of dicts, each containing:
        params      – hyperparameter dict
        score_A     – accuracy on Dataset A
        score_B     – accuracy on Dataset B  (None if not found)
        rank_A      – rank on Dataset A  (1 = best)
        rank_B      – rank on Dataset B  (None if not found)
        drift       – rank_A − rank_B    (None if rank_B is None)
        transfer    – "✓ good" or "✗ poor"
    Sorted by drift descending (best improvers first), unmatched configs last.
    """

    # ── Step 1: in-order traversals (ascending score) ──────────────────────
    nodes_a = registry_a.all_trials()   # ascending list of TrialNode
    nodes_b = registry_b.all_trials()

    n_a = len(nodes_a)
    n_b = len(nodes_b)

    # ── Step 2: build rank lookup dicts ────────────────────────────────────
    # Rank 1 = best, so rank of node at index i (0-based, ascending) = n - i
    # We key by a frozenset of the params dict items for O(1) lookup.

    def params_key(params: dict) -> frozenset:
        """Hashable representation of a params dict."""
        return frozenset(params.items())

    # rank_map_a: params_key -> (rank, score)
    rank_map_a: dict[frozenset, tuple[int, float]] = {}
    for i, node in enumerate(nodes_a):
        rank = n_a - i          # rank 1 = last element (highest score)
        rank_map_a[params_key(node.params)] = (rank, node.score)

    # rank_map_b: params_key -> (rank, score)
    rank_map_b: dict[frozenset, tuple[int, float]] = {}
    for i, node in enumerate(nodes_b):
        rank = n_b - i
        rank_map_b[params_key(node.params)] = (rank, node.score)

    # ── Step 3 & 4: build report ────────────────────────────────────────────
    report: list[dict] = []

    for node in nodes_a:
        key = params_key(node.params)
        rank_a, score_a = rank_map_a[key]

        if key in rank_map_b:
            rank_b, score_b = rank_map_b[key]
            drift = rank_a - rank_b          # positive = moved up = improved
            transfer = "✓ good" if drift >= good_drift_threshold else "✗ poor"
        else:
            rank_b = None
            score_b = None
            drift = None
            transfer = "✗ poor"              # not present in B → couldn't transfer

        report.append({
            "params":   node.params,
            "score_A":  score_a,
            "score_B":  score_b,
            "rank_A":   rank_a,
            "rank_B":   rank_b,
            "drift":    drift,
            "transfer": transfer,
        })

    # ── Step 5: sort by drift descending (best improvers first) ────────────
    # Configs missing from B (drift=None) are pushed to the end.
    report = sorted(
        report,
        key=lambda r: (r["drift"] is None, -(r["drift"] or 0)),
    )

    return report


def transfer_summary(report: List[dict]) -> dict:
    """
    Convenience helper: aggregate statistics from an analyse_transfer report.

    Returns
    -------
    dict with keys:
        total          – number of configs in the report
        matched        – configs found in both registries
        good           – configs labelled "✓ good"
        poor           – configs labelled "✗ poor"
        mean_drift     – mean drift across matched configs
        top_improvers  – top 3 dicts sorted by drift descending
        top_decliners  – bottom 3 dicts sorted by drift ascending
    """
    matched = [r for r in report if r["drift"] is not None]
    drifts = [r["drift"] for r in matched]

    return {
        "total":         len(report),
        "matched":       len(matched),
        "good":          sum(1 for r in report if r["transfer"] == "✓ good"),
        "poor":          sum(1 for r in report if r["transfer"] == "✗ poor"),
        "mean_drift":    round(sum(drifts) / len(drifts), 3) if drifts else None,
        "top_improvers": sorted(matched, key=lambda r: -r["drift"])[:3],
        "top_decliners": sorted(matched, key=lambda r:  r["drift"])[:3],
    }