"""Module 1 - Intelligent ticket-confirmation predictor (WL / RAC -> CNF, RAC or WL at charting)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from trainwise.config import SEED
from trainwise.data.generators import QUOTAS, WL_PRIORITY
from trainwise.data.network import CLASS_INFO, TRAIN_TYPES
from trainwise.models.base import BaseModel, FeatureSpec, make_preprocessor

STATUSES = ("CNF", "RAC", "WL")


def expected_calibration_error(prob, outcome, bins: int = 10) -> float:
    """Average gap between predicted probability and observed frequency, weighted by bin size."""
    prob, outcome = np.asarray(prob, dtype=float), np.asarray(outcome, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & ((prob < hi) if hi < 1 else (prob <= hi))
        if mask.any():
            ece += mask.mean() * abs(prob[mask].mean() - outcome[mask].mean())
    return float(ece)


class ConfirmationPredictor(BaseModel):
    name = "confirmation_predictor"
    task = "classification"
    target = "final_status"
    primary_metric = "accuracy"
    features = (
        FeatureSpec("train_type", "cat", categories=tuple(TRAIN_TYPES), pipeline="prs_bookings"),
        FeatureSpec("travel_class", "cat", categories=tuple(CLASS_INFO), pipeline="prs_bookings"),
        FeatureSpec("quota", "cat", categories=QUOTAS, pipeline="prs_bookings"),
        FeatureSpec("wl_type", "cat", categories=tuple(WL_PRIORITY), pipeline="prs_bookings"),
        FeatureSpec("current_status", "cat", categories=("RAC", "WL"), pipeline="prs_bookings"),
        FeatureSpec("position", "num", 1, 2000, pipeline="prs_bookings"),
        FeatureSpec("days_before_departure", "num", 0, 120, pipeline="prs_bookings"),
        FeatureSpec("day_of_week", "num", 0, 6, pipeline="calendar"),
        FeatureSpec("month", "num", 1, 12, pipeline="calendar"),
        FeatureSpec("is_festival_window", "num", 0, 1, pipeline="calendar"),
        FeatureSpec("route_popularity", "num", 0, 1, pipeline="prs_inventory"),
        FeatureSpec("hist_cancellation_rate", "num", 0, 1, pipeline="prs_bookings"),
    )

    def fit(self, df: pd.DataFrame) -> "ConfirmationPredictor":
        X, y = self.frame(df), df[self.target].astype(str).to_numpy()
        self.pipeline = Pipeline([
            ("pre", make_preprocessor(self.features)),
            ("clf", HistGradientBoostingClassifier(max_iter=220, learning_rate=0.07, max_leaf_nodes=31,
                                                   l2_regularization=0.1, random_state=SEED)),
        ]).fit(X, y)
        self._set_reference(X)
        self.trained_rows_ = len(df)
        sample = df.sample(min(len(df), 1500), random_state=SEED)
        self.importance_ = self._permutation_importance(sample, sample[self.target].astype(str).to_numpy(), "accuracy")
        return self

    def proba(self, X) -> pd.DataFrame:
        X = self.frame(X)
        p = pd.DataFrame(self.pipeline.predict_proba(X), columns=list(self.pipeline.classes_))
        for s in STATUSES:
            if s not in p:
                p[s] = 0.0
        p = p[list(STATUSES)]
        rac = (X["current_status"] == "RAC").to_numpy()
        if rac.any():  # domain rule: an RAC passenger cannot drop back to the waitlist
            tot = (p.loc[rac, "CNF"] + p.loc[rac, "RAC"]).replace(0, 1.0)
            p.loc[rac, "CNF"] = p.loc[rac, "CNF"] / tot
            p.loc[rac, "RAC"] = p.loc[rac, "RAC"] / tot
            p.loc[rac, "WL"] = 0.0
        return p

    def score(self, X) -> np.ndarray:
        return self.proba(X)["CNF"].to_numpy()

    def predict_frame(self, X) -> list:
        out = []
        for cnf, rac, wl in self.proba(X).to_numpy():
            probs = {"CNF": float(cnf), "RAC": float(rac), "WL": float(wl)}
            final = max(probs, key=probs.get)
            out.append({
                "confirmation_probability": round(probs["CNF"], 4),
                "travel_probability": round(probs["CNF"] + probs["RAC"], 4),   # RAC passengers can board
                "final_status_probabilities": {k: round(v, 4) for k, v in probs.items()},
                "predicted_final_status": final,
                "confidence": round(max(probs.values()), 4),
                "_monitor_value": final,
            })
        return out

    def evaluate(self, df: pd.DataFrame) -> dict:
        y = df[self.target].astype(str).to_numpy()
        P = self.proba(df)
        pred = np.array([r["predicted_final_status"] for r in self.predict_frame(df)])
        is_cnf = (y == "CNF").astype(int)
        m = {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
             "brier_cnf": brier_score_loss(is_cnf, P["CNF"]), "ece_cnf": expected_calibration_error(P["CNF"], is_cnf),
             "mean_predicted_cnf": float(P["CNF"].mean()), "actual_cnf_rate": float(is_cnf.mean())}
        if 0 < is_cnf.sum() < len(is_cnf):
            m["roc_auc_cnf"] = roc_auc_score(is_cnf, P["CNF"])
        m = {k: round(float(v), 4) for k, v in m.items()}
        m["n"] = int(len(df))
        return m
