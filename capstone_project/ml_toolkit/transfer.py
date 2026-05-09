from __future__ import annotations

from typing import List

from capstone_project.bst_toolkit.registry import HyperparamRegistry


def analyse_transfer(registry_a: HyperparamRegistry,registry_b: HyperparamRegistry,good_drift_threshold: int = 0,) -> List[dict]:


    
    nodes_a = registry_a.all_trials()   # ascending list of TrialNode
    nodes_b = registry_b.all_trials()

    n_a = len(nodes_a)
    n_b = len(nodes_b)

    

    def params_key(params: dict) -> frozenset:
        """Hashable representation of a params dict."""
        return frozenset(params.items())

    
    rank_map_a: dict[frozenset, tuple[int, float]] = {}
    for i, node in enumerate(nodes_a):
        rank = n_a - i          
        rank_map_a[params_key(node.params)] = (rank, node.score)

    
    rank_map_b: dict[frozenset, tuple[int, float]] = {}
    for i, node in enumerate(nodes_b):
        rank = n_b - i
        rank_map_b[params_key(node.params)] = (rank, node.score)

    
    report: list[dict] = []

    for node in nodes_a:
        key = params_key(node.params)
        rank_a, score_a = rank_map_a[key]

        if key in rank_map_b:
            rank_b, score_b = rank_map_b[key]
            drift = rank_a - rank_b         
            transfer = "✓ good" if drift >= good_drift_threshold else "✗ poor"
        else:
            rank_b = None
            score_b = None
            drift = None
            transfer = "✗ poor"             

        report.append({
            "params":   node.params,
            "score_A":  score_a,
            "score_B":  score_b,
            "rank_A":   rank_a,
            "rank_B":   rank_b,
            "drift":    drift,
            "transfer": transfer,
        })

    
    report = sorted(
        report,
        key=lambda r: (r["drift"] is None, -(r["drift"] or 0)),
    )

    return report


def transfer_summary(report: List[dict]) -> dict:
    
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