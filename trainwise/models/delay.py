"""Module 3 - Train delay & disruption prediction (median + 80% prediction interval)."""
from __future__ import annotations

from math import erf, sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from trainwise.config import SEED
from trainwise.data.network import TRAIN_TYPES
from trainwise.models.base import BaseModel, FeatureSpec, make_preprocessor


def prob_below(x: float, p10: float, p50: float, p90: float) -> float:
    """P(delay <= x) from three quantiles, using a split-normal approximation."""
    lo = max((p50 - p10) / 1.2816, 1.0)
    hi = max((p90 - p50) / 1.2816, 1.0)
    s = lo if x < p50 else hi
    return 0.5 * (1 + erf((x - p50) / s / sqrt(2)))


class DelayPredictor(BaseModel):
    name = "delay_predictor"
    task = "regression"
    target = "arrival_delay_min"
    primary_metric = "mae"
    output_bounds = (-30.0, 900.0)
    features = (
        FeatureSpec("train_type", "cat", categories=tuple(TRAIN_TYPES), pipeline="timetable"),
        FeatureSpec("distance_km", "num", 10, 5000, pipeline="timetable"),
        FeatureSpec("num_stops", "num", 0, 300, pipeline="timetable"),
        FeatureSpec("departure_hour", "num", 0, 23, pipeline="timetable"),
        FeatureSpec("month", "num", 1, 12, pipeline="calendar"),
        FeatureSpec("day_of_week", "num", 0, 6, pipeline="calendar"),
        FeatureSpec("northern_route", "num", 0, 1, pipeline="timetable"),
        FeatureSpec("fog_index", "num", 0, 1, pipeline="weather_feed"),
        FeatureSpec("rain_index", "num", 0, 1, pipeline="weather_feed"),
        FeatureSpec("route_congestion", "num", 0, 1, pipeline="ntes_running_status"),
        FeatureSpec("upstream_delay_min", "num", 0, 1440, pipeline="ntes_running_status"),
        FeatureSpec("maintenance_block", "num", 0, 1, pipeline="engineering_blocks"),
    )

    def _hgb(self, **kw) -> Pipeline:
        return Pipeline([("pre", make_preprocessor(self.features)),
                         ("reg", HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06,
                                                               random_state=SEED, **kw))])

    COVERAGE = 0.80

    def fit(self, df: pd.DataFrame) -> "DelayPredictor":
        """Fit median and quantile models, then conformalise the interval on a held-out calibration slice.

        Conformalised quantile regression (Romano et al., 2019): widen or shrink the raw quantile
        interval by the (1 - alpha) quantile of calibration scores, which guarantees the stated
        coverage on exchangeable data instead of relying on the quantile models being right.
        """
        order = np.random.default_rng(SEED).permutation(len(df))
        n_cal = max(100, int(0.2 * len(df)))
        train, cal = df.iloc[order[n_cal:]], df.iloc[order[:n_cal]]
        X, y = self.frame(train), train[self.target].astype(float).to_numpy()
        self.pipeline = self._hgb(loss="absolute_error").fit(X, y)
        self.q_lo = self._hgb(loss="quantile", quantile=0.1).fit(X, y)
        self.q_hi = self._hgb(loss="quantile", quantile=0.9).fit(X, y)
        Xc, yc = self.frame(cal), cal[self.target].astype(float).to_numpy()
        scores = np.maximum(self.q_lo.predict(Xc) - yc, yc - self.q_hi.predict(Xc))
        level = min(1.0, np.ceil((len(scores) + 1) * self.COVERAGE) / len(scores))
        self.conformal_margin_ = float(np.quantile(scores, level))
        self._set_reference(X)
        self.trained_rows_ = len(df)
        sample = train.sample(min(len(train), 1500), random_state=SEED)
        self.importance_ = self._permutation_importance(sample, sample[self.target].astype(float).to_numpy(),
                                                        "neg_mean_absolute_error")
        return self

    def quantiles(self, X) -> np.ndarray:
        X = self.frame(X)
        margin = getattr(self, "conformal_margin_", 0.0)
        q = np.column_stack([self.q_lo.predict(X) - margin, self.pipeline.predict(X), self.q_hi.predict(X) + margin])
        return np.sort(q, axis=1)

    def score(self, X) -> np.ndarray:
        return self.pipeline.predict(self.frame(X))

    def predict_frame(self, X) -> list:
        out = []
        for p10, p50, p90 in self.quantiles(X):
            risk = "low" if p50 < 15 else "moderate" if p50 < 45 else "high" if p50 < 90 else "severe"
            out.append({
                "expected_delay_min": round(float(p50), 1),
                "delay_interval_min": [round(float(p10), 1), round(float(p90), 1)],
                "quantiles": {"p10": float(p10), "p50": float(p50), "p90": float(p90)},
                "prob_on_time_15min": round(prob_below(15, p10, p50, p90), 4),
                "prob_delay_over_60min": round(1 - prob_below(60, p10, p50, p90), 4),
                "risk_level": risk,
                "confidence": round(1 / (1 + (p90 - p10) / 60), 4),   # narrower interval -> higher confidence
                "_monitor_value": float(p50),
            })
        return out

    def evaluate(self, df: pd.DataFrame) -> dict:
        y = df[self.target].astype(float).to_numpy()
        q = self.quantiles(df)
        m = {"mae": mean_absolute_error(y, q[:, 1]), "r2": r2_score(y, q[:, 1]),
             "interval_coverage_80": float(np.mean((y >= q[:, 0]) & (y <= q[:, 2]))),
             "interval_width_min": float(np.mean(q[:, 2] - q[:, 0])),
             "conformal_margin_min": float(getattr(self, "conformal_margin_", 0.0))}
        m = {k: round(float(v), 4) for k, v in m.items()}
        m["n"] = int(len(df))
        return m
