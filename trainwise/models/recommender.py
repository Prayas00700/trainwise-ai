"""Module 2 - Smart train recommendation engine.

Multi-objective scoring over candidate (train, class) options. Components are normalised within
the candidate set so every score is relative to what is actually available for that search:

  cost         cheaper is better
  duration     faster is better
  availability probability of travelling (AVL, RAC, or a waitlist that clears)
  punctuality  model-predicted probability of arriving within 15 minutes of schedule
  personal     match with the passenger's history (class, train types, departure time)

Weights are a versioned *ranking policy* held in the model registry, so a weight change is a
deployment that Watchtower can audit, evaluate for bias, and roll back.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from trainwise.data.network import CLASS_INFO, PREMIUM_TYPES

COMPONENTS = ("cost", "duration", "availability", "punctuality", "personal")
DEFAULT_WEIGHTS = {"cost": 0.26, "duration": 0.16, "availability": 0.26, "punctuality": 0.16, "personal": 0.16}
SEATED_CLASSES = ("2S", "CC", "EC")
PREF_BOOST = {"cheapest": "cost", "fastest": "duration", "confirmed": "availability", "punctual": "punctuality"}


@dataclass
class RankingPolicy:
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    premium_boost: float = 0.0          # >0 = commercial nudge toward premium trains (should be 0)
    note: str = ""


def personal_score(opt: dict, profile: dict | None) -> float:
    if not profile:
        return 0.5
    s = 0.4 if opt["class"] == profile.get("preferred_class") else 0.1
    s += 0.3 * min(1.0, 2 * profile.get("type_affinity", {}).get(opt["train_type"], 0.0))
    s += 0.3 * math.exp(-((opt["departure_hour"] - profile.get("preferred_departure_hour", 12)) / 4) ** 2)
    return round(min(s, 1.0), 4)


def _minmax_inverse(values: list) -> list:
    lo, hi = min(values), max(values)
    return [1.0 if hi == lo else (hi - v) / (hi - lo) for v in values]


class TrainRecommender:
    name = "train_recommender"

    def __init__(self, policy: RankingPolicy | None = None):
        self.policy = policy or RankingPolicy()

    def weights_for(self, prefs: list | None) -> dict:
        w = dict(self.policy.weights)
        for p in prefs or []:
            if p in PREF_BOOST:
                w[PREF_BOOST[p]] *= 2.2
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    def components(self, options: list, profile: dict | None) -> list:
        cost = _minmax_inverse([o["fare"] for o in options])
        dur = _minmax_inverse([o["duration_min"] for o in options])
        out = []
        for o, c, d in zip(options, cost, dur):
            out.append({"cost": round(c, 4), "duration": round(d, 4),
                        "availability": round(float(o.get("travel_probability", 0.5)), 4),
                        "punctuality": round(float(o.get("prob_on_time_15min", 0.5)), 4),
                        "personal": personal_score(o, profile)})
        return out

    def score(self, options: list, prefs: list | None = None, profile: dict | None = None) -> list:
        if not options:
            return []
        w = self.weights_for(prefs)
        ranked = []
        for o, comp in zip(options, self.components(options, profile)):
            total = sum(w[k] * comp[k] for k in COMPONENTS)
            if o["duration_min"] > 600 and o["class"] in SEATED_CLASSES:
                total -= 0.15          # suitability: a seat (not a berth) for an overnight journey
            if o["train_type"] in PREMIUM_TYPES:
                total += self.policy.premium_boost
            ranked.append({**o, "score": round(total, 4), "components": comp,
                           "rationale": self._rationale(o, comp, w, options)})
        ranked.sort(key=lambda r: -r["score"])
        for i, r in enumerate(ranked, start=1):
            r["rank"] = i
        return ranked

    @staticmethod
    def _rationale(o: dict, comp: dict, w: dict, options: list) -> list:
        reasons = []
        contrib = sorted(COMPONENTS, key=lambda k: -w[k] * comp[k])
        for k in contrib[:2]:
            if comp[k] < 0.5:
                continue
            if k == "cost":
                reasons.append(f"{'lowest' if comp[k] == 1.0 else 'competitive'} fare in this search (₹{o['fare']})")
            elif k == "duration":
                reasons.append(f"{'fastest' if comp[k] == 1.0 else 'quick'} journey ({o['duration_min'] // 60}h {o['duration_min'] % 60:02d}m)")
            elif k == "availability":
                reasons.append(f"{round(100 * o['travel_probability'])}% chance of travelling ({o['availability']})")
            elif k == "punctuality":
                reasons.append(f"{round(100 * o['prob_on_time_15min'])}% chance of arriving within 15 min of schedule")
            elif k == "personal":
                reasons.append(f"matches your usual choices ({CLASS_INFO[o['class']]['label']}, {o['train_type']})")
        if comp["availability"] < 0.4:
            reasons.append(f"risk: only {round(100 * o['travel_probability'])}% chance of a seat ({o['availability']})")
        if o["duration_min"] > 600 and o["class"] in SEATED_CLASSES:
            reasons.append("risk: seated class for a journey longer than 10 hours")
        if comp["punctuality"] < 0.4:
            reasons.append(f"risk: expected delay {o['expected_delay_min']:.0f} min")
        return reasons or ["balanced option on cost, time and availability"]

    def learn_from_feedback(self, events: list) -> RankingPolicy:
        """Continuous optimisation: pairwise logistic regression on chosen-vs-skipped options.

        Each event = {"chosen": components dict, "skipped": [components dicts]}. The fitted
        coefficients become the new (normalised) weights; the result is registered as a new
        policy version and must pass Watchtower's gates like any model.
        """
        X, y = [], []
        for e in events:
            for s in e["skipped"]:
                diff = [e["chosen"][k] - s[k] for k in COMPONENTS]
                X += [diff, [-v for v in diff]]
                y += [1, 0]
        if len(set(y)) < 2:
            return self.policy
        coef = LogisticRegression(fit_intercept=False, C=1.0).fit(np.array(X), np.array(y)).coef_[0]
        coef = np.clip(coef, 0.02, None)
        new = {k: round(float(v / coef.sum()), 4) for k, v in zip(COMPONENTS, coef)}
        return RankingPolicy(weights=new, premium_boost=self.policy.premium_boost, note="learned from feedback")
