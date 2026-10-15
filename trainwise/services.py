"""Governed model serving: every prediction passes through Watchtower's hooks.

data-quality gate -> degraded-mode imputation (only for quarantined feeds) -> inference ->
latency / SLA telemetry -> output sanity check -> optional explanation.
"""
from __future__ import annotations

import time

import numpy as np

from trainwise.data.network import stable_seed
from trainwise.governance.data_quality import check_record
from trainwise.models.recommender import TrainRecommender

_SLIM_KEYS = ("option_id", "train_number", "train_type", "class", "fare", "duration_min", "travel_probability",
              "prob_on_time_15min", "departure_hour", "expected_delay_min", "availability")


class ModelService:
    def __init__(self, name: str, platform):
        self.name, self.p = name, platform
        self._rng = np.random.default_rng(stable_seed("svc", name))

    @property
    def version(self) -> str:
        return self.p.registry.champion(self.name)

    @property
    def model(self):
        return self.p.registry.model(self.name, self.version)

    def _prepare(self, record: dict, model):
        record = {f.name: record.get(f.name) for f in model.features}
        dq = check_record(record, model.features)
        degraded = False
        if not dq.ok:
            quarantined = {f.name for f in model.features if self.p.feed_status.get(f.pipeline) == "quarantined"}
            if set(dq.bad_features) <= quarantined:
                record = {**record, **{f: model.reference_[f] for f in dq.bad_features}}
                dq = check_record(record, model.features)
                degraded = dq.ok
        return record, dq, degraded

    def _overhead_ms(self) -> float:
        injected = self.p.monitor.injected_latency(self.name)
        if not injected:
            return 0.0
        replicas = max(self.p.deployments[self.name]["replicas"], 1)
        return injected * 2.0 / replicas * float(self._rng.uniform(0.85, 1.15))

    @staticmethod
    def _anomalous(model, value) -> bool:
        if model.output_bounds is None or not isinstance(value, (int, float)):
            return False
        lo, hi = model.output_bounds
        return not lo <= value <= hi

    def predict(self, record: dict, explain: bool = False, outcome=None) -> dict:
        model, version = self.model, self.version
        rec, dq, degraded = self._prepare(dict(record), model)
        if not dq.ok:
            rid = self.p.monitor.log_inference(self.name, version, 0.0, "blocked", features=rec, issues=dq.issues)
            return {"status": "rejected", "model": self.name, "version": version, "request_id": rid,
                    "issues": dq.issues, "message": "Input failed data-quality validation, so no prediction was made."}
        t0 = time.perf_counter()
        try:
            out, status = model.predict_frame(model.frame(rec))[0], "ok"
        except Exception as exc:  # pragma: no cover - surfaced through the error-rate monitor
            out, status = {"error": f"{type(exc).__name__}: {exc}"}, "error"
        latency = (time.perf_counter() - t0) * 1000 + self._overhead_ms()
        value = out.pop("_monitor_value", None)
        rid = self.p.monitor.log_inference(self.name, version, latency, status, features=rec, prediction=value,
                                           degraded=degraded, output_anomaly=self._anomalous(model, value))
        if outcome is not None:
            self.p.monitor.log_outcome(rid, outcome)
        out.update(status=status, model=self.name, version=version, request_id=rid, latency_ms=round(latency, 2))
        if dq.warnings:
            out["data_quality_warnings"] = dq.warnings
        if degraded:
            out["degraded_mode"] = "quarantined feed: invalid inputs replaced with last-known-good values"
        if explain and status == "ok":
            out["explanation"] = model.explain(rec)
        return out

    def predict_batch(self, records: list, outcomes: list | None = None) -> list:
        """Vectorised path used by traffic simulation; logs every request individually."""
        model, version = self.model, self.version
        prepared = [self._prepare(dict(r), model) for r in records]
        ok_idx = [i for i, (_, dq, _) in enumerate(prepared) if dq.ok]
        results: list = [None] * len(records)
        t0 = time.perf_counter()
        preds = model.predict_frame(model.frame([prepared[i][0] for i in ok_idx])) if ok_idx else []
        per_row = (time.perf_counter() - t0) * 1000 / max(len(ok_idx), 1)
        for j, i in enumerate(ok_idx):
            rec, _, degraded = prepared[i]
            out = preds[j]
            value = out.pop("_monitor_value", None)
            rid = self.p.monitor.log_inference(self.name, version, per_row + self._overhead_ms(), "ok", features=rec,
                                               prediction=value, degraded=degraded,
                                               output_anomaly=self._anomalous(model, value))
            if outcomes is not None and outcomes[i] is not None:
                self.p.monitor.log_outcome(rid, outcomes[i])
            results[i] = {**out, "status": "ok", "request_id": rid, "version": version}
        for i, (rec, dq, _) in enumerate(prepared):
            if not dq.ok:
                rid = self.p.monitor.log_inference(self.name, version, 0.0, "blocked", features=rec, issues=dq.issues)
                results[i] = {"status": "rejected", "issues": dq.issues, "request_id": rid, "version": version}
        return results


class RecommenderService:
    """Serves the champion ranking policy and logs every ranked list for bias monitoring."""
    name = "train_recommender"

    def __init__(self, platform):
        self.p = platform

    @property
    def version(self) -> str:
        return self.p.registry.champion(self.name)

    def ranker(self, version: str | None = None) -> TrainRecommender:
        return TrainRecommender(self.p.registry.model(self.name, version or self.version))

    def baseline_ranker(self) -> TrainRecommender:
        return self.ranker(self.p.registry.first_tagged(self.name, "fairness_approved") or self.version)

    def rank(self, options: list, prefs: list | None = None, user_id: str | None = None, budget=None,
             log: bool = True, context: str = "api", synthetic: bool = False) -> list:
        profile = self.p.users.get(user_id) if user_id else None
        t0 = time.perf_counter()
        ranked = self.ranker().score(options, prefs, profile)
        if log and options:
            self.p.monitor.log("recommendations", {
                "version": self.version, "context": context, "synthetic": synthetic,
                "candidates": [{k: o.get(k) for k in _SLIM_KEYS} for o in options],
                "ranked_ids": [r["option_id"] for r in ranked], "prefs": list(prefs or []), "profile": profile,
                "budget": budget, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)})
        return ranked

    def recommend(self, origins, destinations, journey, prefs=None, user_id=None, budget=None, classes=None,
                  top_k: int = 5, synthetic: bool = False) -> dict:
        trains = self.p.tools.search_trains(origins, destinations, journey)
        options = self.p.tools.build_options(trains, journey, classes)
        if budget:
            options = [o for o in options if o["fare"] <= budget]
        ranked = self.rank(options, prefs, user_id, budget, context="recommend", synthetic=synthetic)
        return {"journey_date": journey.isoformat(), "candidates": len(options), "recommendations": ranked[:top_k],
                "ranker_version": self.version, "personalised": bool(user_id and user_id in self.p.users)}

    def learn_from_feedback(self, events: list, actor: str = "watchtower") -> dict:
        """Fit new weights from chosen-vs-skipped feedback; registered as a candidate, never auto-deployed."""
        policy = self.ranker().learn_from_feedback(events)
        version = self.p.registry.register(self.name, policy, metrics={"feedback_events": len(events)},
                                           parent=self.version, notes="weights learned from passenger feedback",
                                           actor=actor)
        return {"candidate_version": version, "weights": policy.weights}
