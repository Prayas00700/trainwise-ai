"""Module 7 - Fraud, security and bot detection.

Hybrid risk score: a supervised gradient-boosting model (known attack patterns) blended with an
Isolation Forest trained on genuine sessions only (catches unfamiliar behaviour). Every alert
carries plain-language reason codes so an analyst can see *why* a session was flagged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from trainwise.config import SEED, model_policy
from trainwise.models.base import BaseModel, FeatureSpec, make_preprocessor

DEFAULT_COSTS = {"false_challenge": 15, "false_block": 400, "missed_fraud": 250}

NUMERIC = ["requests_per_min", "avg_inter_click_ms", "form_fill_seconds", "captcha_solve_seconds",
           "accounts_on_device", "ips_last_24h", "bookings_last_24h", "tatkal_window_hit",
           "payment_failures_24h", "passengers_per_booking", "account_age_days", "headless_signal"]
LABELS = {
    "requests_per_min": "request rate", "avg_inter_click_ms": "time between clicks",
    "form_fill_seconds": "form-fill time", "captcha_solve_seconds": "CAPTCHA solve time",
    "accounts_on_device": "accounts on this device", "ips_last_24h": "IP addresses in 24h",
    "bookings_last_24h": "bookings in 24h", "tatkal_window_hit": "booked in the Tatkal opening minute",
    "payment_failures_24h": "payment failures in 24h", "passengers_per_booking": "passengers per booking",
    "account_age_days": "account age (days)", "headless_signal": "headless-browser signal",
}


class FraudDetector(BaseModel):
    name = "fraud_detector"
    task = "classification"
    target = "is_fraud"
    primary_metric = "recall"
    challenge_at, block_at = 0.40, 0.75
    features = tuple(
        [FeatureSpec(c, "num", 0, 1e6, pipeline="session_telemetry") for c in NUMERIC]
        + [FeatureSpec("device_type", "cat", categories=("mobile", "desktop"), pipeline="session_telemetry"),
           FeatureSpec("region_segment", "cat", categories=("metro", "tier2", "tier3"), pipeline="session_telemetry")])

    def fit(self, df: pd.DataFrame) -> "FraudDetector":
        order = np.random.default_rng(SEED).permutation(len(df))
        n_cal = max(200, int(0.2 * len(df)))
        train, cal = df.iloc[order[n_cal:]], df.iloc[order[:n_cal]]
        X, y = self.frame(train), train[self.target].astype(int).to_numpy()
        self.pipeline = Pipeline([("pre", make_preprocessor(self.features)),
                                  ("clf", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                                                         class_weight="balanced", random_state=SEED))]).fit(X, y)
        legit = X[y == 0][NUMERIC]
        self.iforest = IsolationForest(n_estimators=150, random_state=SEED).fit(legit)
        raw = -self.iforest.score_samples(legit)
        self._anom_lo, self._anom_hi = np.quantile(raw, 0.50), np.quantile(raw, 0.995)
        self._median = legit.median()
        q75, q25 = legit.quantile(0.75), legit.quantile(0.25)
        self._scale = ((q75 - q25) / 1.349).replace(0, np.nan).fillna(legit.std().replace(0, 1.0)) + 1e-6
        self._set_reference(X)
        self.trained_rows_ = len(df)
        self._choose_thresholds(cal)
        sample = train.sample(min(len(train), 1500), random_state=SEED)
        self.importance_ = self._permutation_importance(sample, sample[self.target].astype(int).to_numpy(), "recall")
        return self

    def _choose_thresholds(self, cal: pd.DataFrame) -> None:
        """Cost-sensitive thresholds: minimise expected cost per session on held-out data.

        A false challenge costs friction; a false block costs a genuine booking; a missed fraud costs
        hoarded seats and investigation. The costs live in governance policy, so the business, not
        the data scientist, owns the trade-off.
        """
        costs = {**DEFAULT_COSTS, **(model_policy(self.name).get("decision_costs") or {})}
        risk, y = self.risk(cal), cal[self.target].astype(int).to_numpy()
        grid = np.round(np.linspace(0.15, 0.95, 81), 4)

        def expected_cost(t: float, fp_cost: float) -> float:
            flagged = risk >= t
            fp = float(np.mean(flagged & (y == 0)))
            fn = float(np.mean(~flagged & (y == 1)))
            return fp_cost * fp + costs["missed_fraud"] * fn

        # Where costs tie (separable data), challenge as early as possible and block as late as possible.
        c_cost = np.array([expected_cost(t, costs["false_challenge"]) for t in grid])
        self.challenge_at = float(grid[c_cost <= c_cost.min() + 1e-9].min())
        b_grid = grid[grid >= self.challenge_at]
        b_cost = np.array([expected_cost(t, costs["false_block"]) for t in b_grid])
        self.block_at = float(b_grid[b_cost <= b_cost.min() + 1e-9].max())
        self.threshold_report_ = {"costs": costs, "challenge_at": self.challenge_at, "block_at": self.block_at,
                                  "expected_cost_per_session": round(expected_cost(self.challenge_at, costs["false_challenge"]), 3),
                                  "calibration_rows": int(len(cal))}

    def risk(self, X) -> np.ndarray:
        X = self.frame(X)
        sup = self.pipeline.predict_proba(X)[:, 1]
        raw = -self.iforest.score_samples(X[NUMERIC])
        anom = np.clip((raw - self._anom_lo) / (self._anom_hi - self._anom_lo + 1e-9), 0, 1)
        return np.clip(0.8 * sup + 0.2 * anom, 0, 1)

    def score(self, X) -> np.ndarray:
        return self.risk(X)

    def reasons(self, row: pd.Series, top: int = 3) -> list:
        z = ((row[NUMERIC].astype(float) - self._median) / self._scale).abs().sort_values(ascending=False)
        for threshold in (3.0, 1.5):    # subtle patterns: fall back to the strongest moderate deviations
            out = [f"{LABELS[c]} = {row[c]:g} (typical {self._median[c]:g})" for c, v in z.items() if v >= threshold][:top]
            if out:
                return out
        return []

    @staticmethod
    def pattern(row: pd.Series) -> str:
        if row["requests_per_min"] > 40 and row["form_fill_seconds"] < 8:
            return "bulk booking bot"
        if row["accounts_on_device"] >= 4:
            return "multi-account tout"
        if row["payment_failures_24h"] >= 5:
            return "card testing"
        if row["tatkal_window_hit"] and row["passengers_per_booking"] >= 3 and row["bookings_last_24h"] >= 2:
            return "Tatkal-window hoarding"
        if row["ips_last_24h"] >= 4 and row["payment_failures_24h"] >= 2:
            return "possible account takeover"
        return "anomalous behaviour"

    def predict_frame(self, X) -> list:
        X = self.frame(X)
        out = []
        for (_, row), r in zip(X.iterrows(), self.risk(X)):
            decision = "block" if r >= self.block_at else "challenge" if r >= self.challenge_at else "allow"
            flagged = decision != "allow"
            out.append({"risk_score": round(float(r) * 100, 1), "decision": decision,
                        "action": {"block": "Block session and hold bookings for review",
                                   "challenge": "Step-up verification (OTP / CAPTCHA) before payment",
                                   "allow": "Proceed"}[decision],
                        "suspected_pattern": self.pattern(row) if flagged else None,
                        "reasons": self.reasons(row) if flagged else [],
                        "_monitor_value": int(flagged)})
        return out

    def evaluate(self, df: pd.DataFrame) -> dict:
        y = df[self.target].astype(int).to_numpy()
        r = self.risk(df)
        pred = (r >= self.challenge_at).astype(int)
        neg = y == 0
        m = {"recall": recall_score(y, pred, zero_division=0), "precision": precision_score(y, pred, zero_division=0),
             "f1": f1_score(y, pred, zero_division=0), "fpr": float(pred[neg].mean()) if neg.any() else 0.0,
             "pr_auc": average_precision_score(y, r) if y.any() else 0.0}
        m = {k: round(float(v), 4) for k, v in m.items()}
        m.update(n=int(len(df)), challenge_at=self.challenge_at, block_at=self.block_at)
        return m
