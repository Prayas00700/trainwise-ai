"""Fairness checks: performance parity across passenger segments and recommender exposure."""
from __future__ import annotations

import math

import pandas as pd

from trainwise.data.network import PREMIUM_TYPES


def segment_gap(y_true, y_pred, groups, metric: str = "accuracy", min_group: int = 40, allowed=None) -> dict:
    """Largest gap in `metric` between segments (accuracy, or false-positive rate for 'fpr')."""
    df = pd.DataFrame({"y": list(y_true), "p": list(y_pred), "g": [str(g) for g in groups]})
    if allowed:
        df = df[df["g"].isin([str(a) for a in allowed])]
    rows = []
    for g, sub in df.groupby("g"):
        if metric == "fpr":
            neg = sub[sub["y"].astype(int) == 0]
            if len(neg) < min_group:
                continue
            val = float((neg["p"].astype(int) == 1).mean())
        else:
            if len(sub) < min_group:
                continue
            val = float((sub["y"].astype(str) == sub["p"].astype(str)).mean())
        rows.append({"segment": g, "n": int(len(sub)), metric: round(val, 4)})
    vals = [r[metric] for r in rows]
    return {"metric": metric, "segments": rows, "gap": round(max(vals) - min(vals), 4) if len(vals) >= 2 else 0.0,
            "evaluated": len(vals) >= 2}


def exposure_shift(entries: list, baseline, k: int = 3) -> dict:
    """Premium-train exposure of the live ranker vs the approved baseline ranker on the SAME candidate sets.

    Position-weighted (1/log2(rank+1)) share of top-k slots given to premium trains. A counterfactual
    comparison isolates the effect of the ranking policy from changes in what is available.
    """
    weights = [1 / math.log2(i + 2) for i in range(k)]
    live = base = total = 0.0
    violations = lists = 0
    for e in entries:
        cands = e.get("candidates") or []
        if not cands:
            continue
        types = {c["option_id"]: c["train_type"] for c in cands}
        fares = {c["option_id"]: c["fare"] for c in cands}
        live_top = e["ranked_ids"][:k]
        base_top = [o["option_id"] for o in baseline.score(cands, e.get("prefs"), e.get("profile"))][:k]
        live += sum(w * (types[o] in PREMIUM_TYPES) for w, o in zip(weights, live_top))
        base += sum(w * (types[o] in PREMIUM_TYPES) for w, o in zip(weights, base_top))
        total += sum(weights[:len(live_top)])
        lists += 1
        if e.get("budget") and live_top and fares[live_top[0]] > e["budget"]:
            violations += 1
    if not lists:
        return {"lists": 0, "live_premium_share": 0.0, "baseline_premium_share": 0.0, "shift": 0.0,
                "budget_violation_rate": 0.0}
    ls, bs = live / total, base / total
    return {"lists": lists, "live_premium_share": round(ls, 4), "baseline_premium_share": round(bs, 4),
            "shift": round(ls - bs, 4), "budget_violation_rate": round(violations / lists, 4)}
