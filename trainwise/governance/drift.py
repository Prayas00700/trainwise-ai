"""Drift detection.

* Data drift: Population Stability Index (PSI) per feature, plus a KS test for numeric features.
* Concept drift: Page-Hinkley change detection on the stream of per-prediction errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi_categorical(ref, cur) -> float:
    ref, cur = pd.Series(ref).astype(str), pd.Series(cur).astype(str)
    cats = sorted(set(ref) | set(cur))
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_numeric(ref, cur, bins: int = 10) -> float:
    ref = pd.to_numeric(pd.Series(ref), errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(pd.Series(cur), errors="coerce").dropna().to_numpy(dtype=float)
    if len(ref) < 10 or len(cur) < 10:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant or binary feature: compare value frequencies instead
        return psi_categorical(np.round(ref, 3), np.round(cur, 3))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def data_drift(reference: pd.DataFrame, current: pd.DataFrame, specs, policy: dict) -> dict:
    seasonal = set(policy.get("seasonal_features", []))
    rows = []
    for s in specs:
        if s.kind == "text" or s.name not in reference.columns or s.name not in current.columns:
            continue
        if s.kind == "num":
            psi = psi_numeric(reference[s.name], current[s.name])
            a = pd.to_numeric(reference[s.name], errors="coerce").dropna()
            b = pd.to_numeric(current[s.name], errors="coerce").dropna()
            ks_p = float(ks_2samp(a, b).pvalue) if len(a) > 10 and len(b) > 10 else None
        else:
            psi, ks_p = psi_categorical(reference[s.name], current[s.name]), None
        if s.name in seasonal:
            status = "seasonal"
        else:
            status = ("critical" if psi >= policy["psi_critical"]
                      else "warning" if psi >= policy["psi_warning"] else "stable")
        rows.append({"feature": s.name, "pipeline": s.pipeline, "psi": round(psi, 4),
                     "ks_pvalue": None if ks_p is None else round(ks_p, 6), "status": status})
    rows.sort(key=lambda r: -r["psi"])
    judged = [r for r in rows if r["status"] != "seasonal"]
    worst = judged[0]["psi"] if judged else 0.0
    return {"features": rows, "max_psi": worst, "top": judged[:3],
            "drifted": [r["feature"] for r in judged if r["status"] != "stable"],
            "status": ("critical" if worst >= policy["psi_critical"]
                       else "warning" if worst >= policy["psi_warning"] else "stable"),
            "n_reference": int(len(reference)), "n_current": int(len(current))}


class PageHinkley:
    """Sequential change detector on an error stream in [0, 1]: alarms when the mean error jumps."""

    def __init__(self, delta: float = 0.005, threshold: float = 25.0):
        self.delta, self.threshold = delta, threshold
        self.n, self.mean, self.cum, self.min_cum = 0, 0.0, 0.0, 0.0

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.cum += x - self.mean - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        return (self.cum - self.min_cum) > self.threshold


def detect_change(errors, delta: float, threshold: float):
    """Index of the first alarm in the error stream, or None."""
    ph = PageHinkley(delta, threshold)
    for i, e in enumerate(errors):
        if ph.update(float(e)):
            return i
    return None
