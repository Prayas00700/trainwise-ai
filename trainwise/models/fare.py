"""Module 5 - Dynamic fare prediction.

Design: machine learning forecasts *demand* (share of seats sold by day), and a deterministic,
auditable tariff turns demand into a fare (flexi-fare style steps for dynamic-priced trains).
Keeping the money logic rule-based makes every rupee explainable and testable.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from trainwise.config import SEED
from trainwise.data.inventory import demand_index, festival_multiplier
from trainwise.data.network import CLASS_INFO, TRAIN_TYPES, fare_breakdown, flexi_multiplier
from trainwise.models.base import BaseModel, FeatureSpec, make_preprocessor


class FareDemandForecaster(BaseModel):
    name = "fare_demand_forecaster"
    task = "regression"
    target = "sold_fraction"
    primary_metric = "mae"
    output_bounds = (0.0, 1.9)
    features = (
        FeatureSpec("train_type", "cat", categories=tuple(TRAIN_TYPES), pipeline="prs_inventory"),
        FeatureSpec("travel_class", "cat", categories=tuple(CLASS_INFO), pipeline="prs_inventory"),
        FeatureSpec("days_before_departure", "num", 0, 120, pipeline="prs_inventory"),
        FeatureSpec("month", "num", 1, 12, pipeline="calendar"),
        FeatureSpec("day_of_week", "num", 0, 6, pipeline="calendar"),
        FeatureSpec("demand_index", "num", 0.3, 3.0, pipeline="calendar"),
        FeatureSpec("route_popularity", "num", 0, 1, pipeline="prs_inventory"),
    )

    def fit(self, df: pd.DataFrame) -> "FareDemandForecaster":
        X, y = self.frame(df), df[self.target].astype(float).to_numpy()
        self.pipeline = Pipeline([("pre", make_preprocessor(self.features)),
                                  ("reg", HistGradientBoostingRegressor(max_iter=250, learning_rate=0.07,
                                                                        random_state=SEED))]).fit(X, y)
        self._set_reference(X)
        self.trained_rows_ = len(df)
        sample = df.sample(min(len(df), 1500), random_state=SEED)
        self.importance_ = self._permutation_importance(sample, sample[self.target].astype(float).to_numpy(),
                                                        "neg_mean_absolute_error")
        return self

    def score(self, X) -> np.ndarray:
        return self.pipeline.predict(self.frame(X))

    def predict_frame(self, X) -> list:
        return [{"sold_fraction": round(float(v), 4), "_monitor_value": float(v)} for v in self.score(X)]

    def evaluate(self, df: pd.DataFrame) -> dict:
        y = df[self.target].astype(float).to_numpy()
        p = self.score(df)
        return {"mae": round(float(mean_absolute_error(y, p)), 4), "r2": round(float(r2_score(y, p)), 4),
                "n": int(len(df))}


class FareEngine:
    """Quotes, forecasts and booking-window advice. `platform` provides network, inventory, services."""

    def __init__(self, platform):
        self.p = platform

    def _features(self, train, cls: str, journey: date, days_before: int) -> dict:
        return {"train_type": train.ttype, "travel_class": cls, "days_before_departure": days_before,
                "month": journey.month, "day_of_week": journey.weekday(), "demand_index": demand_index(journey),
                "route_popularity": train.popularity}

    def quote(self, train, cls: str, journey: date, as_of: date | None = None, tatkal: bool = False) -> dict:
        as_of = as_of or self.p.today()
        sold = self.p.inventory.sold_fraction(train, cls, journey, as_of)
        mult = flexi_multiplier(sold, train.dynamic_pricing)
        parts = fare_breakdown(train.distance_km, cls, train.ttype, mult, tatkal=tatkal)
        return {"train": train.number, "class": cls, "journey_date": journey.isoformat(), "fare": parts["total"],
                "flexi_multiplier": mult, "dynamic_pricing": train.dynamic_pricing, "breakdown": parts,
                "sold_fraction": round(sold, 3)}

    def forecast(self, train, cls: str, journey: date, as_of: date | None = None, record_override: dict | None = None) -> dict:
        as_of = as_of or self.p.today()
        days_now = (journey - as_of).days
        if days_now < 0:
            return {"status": "error", "message": "Journey date is in the past."}
        current = self.quote(train, cls, journey, as_of)
        record = {**self._features(train, cls, journey, days_now), **(record_override or {})}
        gov = self.p.services["fare_demand_forecaster"].predict(record, outcome=current["sold_fraction"])
        if gov["status"] != "ok":
            return {"status": "degraded", "message": "Demand forecast unavailable: input failed validation. "
                    "Showing today's fare only.", "issues": gov.get("issues", []), "current": current}
        model = self.p.services["fare_demand_forecaster"].model
        days = list(range(days_now, -1, -1))
        X = pd.DataFrame([{**record, "days_before_departure": d} for d in days])
        raw = model.score(X) + (current["sold_fraction"] - gov["sold_fraction"])   # anchor to observed sales
        path = np.maximum.accumulate(raw)                                          # sales never "un-sell"
        dips = float(np.max(path - raw)) if len(raw) else 0.0
        fares = [fare_breakdown(train.distance_km, cls, train.ttype, flexi_multiplier(s, train.dynamic_pricing))["total"]
                 for s in path]
        trajectory = [{"date": (journey - timedelta(days=d)).isoformat(), "days_before": d,
                       "sold_fraction": round(float(s), 3), "fare": f} for d, s, f in zip(days, path, fares)]
        sellout = next((t for t in trajectory if t["sold_fraction"] >= 1.0), None)
        next_step = next((t for t in trajectory if t["fare"] > current["fare"]), None)
        if train.dynamic_pricing and next_step:
            advice = (f"Book now: the fare is forecast to rise from ₹{current['fare']} to ₹{next_step['fare']} "
                      f"around {next_step['date']} as seats sell.")
            window = {"book_by": trajectory[0]["date"], "reason": "next flexi-fare step"}
        elif train.dynamic_pricing:
            advice = "The fare is unlikely to change before departure; booking now or later costs the same."
            window = {"book_by": None, "reason": "no further fare steps forecast"}
        elif sellout:
            advice = (f"The fare is fixed on this train, but seats are forecast to run out around {sellout['date']}; "
                      "book before then to avoid the waitlist.")
            window = {"book_by": sellout["date"], "reason": "forecast sell-out"}
        else:
            advice = "The fare is fixed and seats are expected to remain available."
            window = {"book_by": None, "reason": "no sell-out forecast"}
        dem = demand_index(journey)
        drivers = []
        if festival_multiplier(journey) > 1.0:
            drivers.append(f"festive/holiday window (x{festival_multiplier(journey):.2f} demand)")
        if journey.weekday() in (4, 6):
            drivers.append("Friday/Sunday travel peak")
        if train.popularity > 0.8:
            drivers.append("high-demand train")
        surge = {"surge": bool(dem >= 1.2 or path[-1] >= 1.05), "demand_index": dem,
                 "forecast_final_sold_fraction": round(float(path[-1]), 3), "drivers": drivers}
        return {"status": "ok", "train": train.number, "class": cls, "journey_date": journey.isoformat(),
                "current": current, "trajectory": trajectory, "advice": advice, "booking_window": window,
                "demand_surge": surge, "savings": self.savings(train, cls, journey, as_of, current["fare"]),
                "explanation": {"fare_components": current["breakdown"], "demand_drivers": drivers,
                                "flexi_multiplier_now": current["flexi_multiplier"]},
                "pricing_anomaly": {"non_monotonic_correction": round(dips, 4), "flagged": dips > 0.05},
                "model": gov["model"], "version": gov["version"]}

    def savings(self, train, cls: str, journey: date, as_of: date, fare_now: int) -> list:
        alts = []
        for delta in (-3, -2, -1, 1, 2, 3):
            d = journey + timedelta(days=delta)
            if d < as_of or not train.runs_on(d):
                continue
            q = self.quote(train, cls, d, as_of)
            if q["fare"] < fare_now:
                alts.append({"option": f"Same train on {d.isoformat()}", "fare": q["fare"], "saving": fare_now - q["fare"]})
        for t in self.p.network.trains_between([train.origin], [train.destination], journey):
            if t.number != train.number and cls in t.classes:
                q = self.quote(t, cls, journey, as_of)
                if q["fare"] < fare_now:
                    alts.append({"option": f"{t.number} {t.name}", "fare": q["fare"], "saving": fare_now - q["fare"]})
        return sorted(alts, key=lambda a: -a["saving"])[:4]
