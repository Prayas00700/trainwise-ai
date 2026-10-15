"""Shared model plumbing: feature specs, preprocessing, local explanations and live metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder

from trainwise.config import SEED


@dataclass(frozen=True)
class FeatureSpec:
    """Declares a model input. Watchtower uses it for data-quality gates, drift checks and RCA."""
    name: str
    kind: str                        # "num" | "cat" | "text"
    lo: Optional[float] = None       # hard bounds: values outside are rejected before inference
    hi: Optional[float] = None
    categories: Optional[tuple] = None
    pipeline: str = "core"           # upstream data pipeline that produces this feature


def make_preprocessor(features: Sequence[FeatureSpec]) -> ColumnTransformer:
    cats = [f.name for f in features if f.kind == "cat"]
    nums = [f.name for f in features if f.kind == "num"]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
        ("num", "passthrough", nums),
    ])


def to_py(v):
    """numpy scalars -> plain Python (JSON friendly)."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def live_metric(metric: str, y_true, y_pred) -> float:
    """Metric computed on (ground truth, logged prediction) pairs from production."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if metric == "accuracy":
        return float(np.mean(y_true == y_pred))
    if metric == "mae":
        return float(np.mean(np.abs(y_true.astype(float) - y_pred.astype(float))))
    if metric == "recall":
        pos = y_true.astype(int) == 1
        return float(np.mean(y_pred[pos].astype(int) == 1)) if pos.any() else float("nan")
    raise ValueError(metric)


def per_sample_error(metric: str, y_true, y_pred, scale: float = 1.0) -> np.ndarray:
    """Error stream in [0, 1] used by the Page-Hinkley change detector."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if metric == "mae":
        return np.clip(np.abs(y_true.astype(float) - y_pred.astype(float)) / max(scale, 1e-9) / 5.0, 0, 1)
    if metric == "recall":
        pos = y_true.astype(int) == 1
        return (y_pred[pos].astype(int) == 0).astype(float)
    return (y_true != y_pred).astype(float)


class BaseModel:
    name = "base"
    task = "classification"
    target = ""
    primary_metric = "accuracy"
    features: tuple = ()
    output_bounds: Optional[tuple] = None   # sanity range for the monitored output

    def __init__(self):
        self.pipeline = None
        self.reference_: dict = {}
        self.importance_: dict = {}
        self.trained_rows_ = 0

    @property
    def feature_names(self) -> list:
        return [f.name for f in self.features]

    def frame(self, records) -> pd.DataFrame:
        if isinstance(records, pd.DataFrame):
            df = records
        elif isinstance(records, dict):
            df = pd.DataFrame([records])
        else:
            df = pd.DataFrame(list(records))
        df = df.reindex(columns=self.feature_names).reset_index(drop=True)
        for f in self.features:
            if f.kind == "num":
                df[f.name] = pd.to_numeric(df[f.name], errors="coerce").astype(float)
            else:
                df[f.name] = df[f.name].astype(object)
        return df

    def _set_reference(self, X: pd.DataFrame) -> None:
        ref = {}
        for f in self.features:
            col = X[f.name]
            ref[f.name] = float(col.median()) if f.kind == "num" else to_py(col.mode().iloc[0])
        self.reference_ = ref

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Scalar per row that explanations decompose (e.g. P(confirm), expected delay)."""
        raise NotImplementedError

    def explain(self, record: dict, top_k: int = 5) -> list:
        """Occlusion explanation: how much each input moves the score vs a typical value.

        For every feature we swap the passenger's value for the training median/mode and
        re-score. Model-agnostic, cheap (one batched predict) and easy to explain to a user.
        """
        base = self.frame(record)
        variants = [base]
        for f in self.feature_names:
            v = base.copy()
            v[f] = self.reference_[f]
            variants.append(v)
        s = self.score(self.frame(pd.concat(variants, ignore_index=True)))
        out = []
        for i, f in enumerate(self.feature_names, start=1):
            effect = float(s[0] - s[i])
            if abs(effect) > 1e-6:
                out.append({"feature": f, "value": to_py(base.iloc[0][f]), "typical": to_py(self.reference_[f]),
                            "effect": round(effect, 4)})
        out.sort(key=lambda c: abs(c["effect"]), reverse=True)
        return out[:top_k]

    def _permutation_importance(self, df: pd.DataFrame, y, scoring: str) -> dict:
        X = self.frame(df)
        r = permutation_importance(self.pipeline, X, y, n_repeats=3, random_state=SEED, scoring=scoring)
        imp = {k: max(float(v), 0.0) for k, v in zip(X.columns, r.importances_mean)}
        total = sum(imp.values()) or 1.0
        return {k: round(v / total, 4) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])}
