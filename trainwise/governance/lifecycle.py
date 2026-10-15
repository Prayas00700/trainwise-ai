"""Automated model lifecycle: retrain on recent labelled data, champion vs challenger, promotion gates."""
from __future__ import annotations

import pandas as pd

from trainwise.config import SEED, model_policy
from trainwise.governance.fairness import segment_gap
from trainwise.models.base import live_metric


class LifecycleManager:
    def __init__(self, platform):
        self.p = platform

    def retrain(self, name: str, reason: str, actor: str = "watchtower") -> dict:
        reg = self.p.registry
        champ_v = reg.champion(name)
        bundle = reg.load(name, champ_v)
        champion = bundle["model"]
        # only the window where the degradation was measured, i.e. data from the new regime
        window = int(self.p.policies["performance"]["window"])
        recent = self.p.monitor.labelled_frame(name, champion.target, version=champ_v).tail(window)
        if len(recent) < 120:
            return {"status": "insufficient_data", "rows": int(len(recent))}
        recent = recent.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        cut = int(len(recent) * 0.7)
        r_train, r_hold = recent.iloc[:cut], recent.iloc[cut:].reset_index(drop=True)
        pol, lc = model_policy(name), self.p.policies["lifecycle"]
        train_df = pd.concat([bundle["train_sample"]] + [r_train] * int(lc["recent_weight"]), ignore_index=True)
        challenger = type(champion)().fit(train_df)
        metric, hib = pol["primary_metric"], pol.get("higher_is_better", True)
        champ_m, chall_m = champion.evaluate(r_hold), challenger.evaluate(r_hold)
        uplift = (chall_m[metric] - champ_m[metric]) if hib else (champ_m[metric] - chall_m[metric])
        hist = bundle.get("valid_sample")
        hist_check = None
        if isinstance(hist, pd.DataFrame) and len(hist):
            hist_check = {"champion": champion.evaluate(hist)[metric], "challenger": challenger.evaluate(hist)[metric]}
        guards = {k: bool(chall_m.get(k, 0.0) <= v) for k, v in (pol.get("guard_metrics") or {}).items()}
        fairness = self.fairness_gate(name, challenger, r_hold, hist)
        passes = uplift >= lc["min_uplift"] and all(guards.values()) and fairness["passed"]
        version = reg.register(
            name, challenger, metrics=chall_m,
            reference=challenger.frame(train_df.sample(min(len(train_df), 5000), random_state=SEED)),
            train_sample=train_df.sample(min(len(train_df), 8000), random_state=SEED).reset_index(drop=True),
            valid_sample=r_hold, parent=champ_v, notes=f"challenger: {reason}", actor=actor)
        result = {"status": "trained", "model": name, "champion": champ_v, "challenger": version, "metric": metric,
                  "champion_score": champ_m[metric], "challenger_score": chall_m[metric],
                  "uplift": round(float(uplift), 4), "historical_check": hist_check, "guards": guards,
                  "fairness": fairness, "passes_gates": bool(passes), "recent_rows": int(len(recent)),
                  "holdout_rows": int(len(r_hold)), "trained_rows": int(len(train_df))}
        self.p.audit.record(actor, "challenger_evaluated", name, result)
        return result

    def fairness_gate(self, name: str, model, recent: pd.DataFrame, hist) -> dict:
        pol, fp = model_policy(name), self.p.policies["fairness"]
        seg = pol.get("fairness_segment")
        if not seg or seg not in recent.columns:
            return {"checked": False, "passed": True}
        data = pd.concat([d for d in (hist, recent) if isinstance(d, pd.DataFrame) and seg in d.columns],
                         ignore_index=True)
        preds = [r["_monitor_value"] for r in model.predict_frame(data)]
        res = segment_gap(data[model.target], preds, data[seg], pol.get("fairness_metric", "accuracy"),
                          fp["min_group_size"], pol.get("fairness_groups"))
        return {**res, "checked": True, "passed": res["gap"] <= fp["max_parity_gap"]}

    def shadow_compare(self, name: str, other_version: str):
        """Score another version on exactly the traffic the champion just served (counterfactual check)."""
        reg, pol = self.p.registry, model_policy(name)
        cur_v = reg.champion(name)
        current, other = reg.model(name, cur_v), reg.model(name, other_version)
        window = int(self.p.policies["performance"]["window"])
        df = self.p.monitor.labelled_frame(name, current.target, version=cur_v).tail(window)
        if len(df) < 50:
            return None
        y = df[current.target].tolist()

        def score(model) -> float:
            return round(live_metric(pol["primary_metric"], y, [r["_monitor_value"] for r in model.predict_frame(df)]), 4)

        return {"metric": pol["primary_metric"], "current_version": cur_v, "current": score(current),
                "other_version": other_version, "other": score(other), "n": int(len(df)),
                "higher_is_better": pol.get("higher_is_better", True)}

    def canary(self, name: str, challenger_version: str, df: pd.DataFrame) -> dict:
        """Score champion and challenger on the same fresh traffic before anything is promoted."""
        reg, pol = self.p.registry, model_policy(name)
        metric, hib = pol["primary_metric"], pol.get("higher_is_better", True)
        a = reg.model(name).evaluate(df)[metric]
        b = reg.model(name, challenger_version).evaluate(df)[metric]
        uplift = (b - a) if hib else (a - b)
        return {"metric": metric, "champion": a, "challenger": b, "uplift": round(float(uplift), 4),
                "rows": int(len(df)), "passes": bool(uplift >= self.p.policies["lifecycle"]["min_uplift"] - 1e-9)}

    def promote(self, name: str, version: str, actor: str, reason: str) -> None:
        self.p.registry.promote(name, version, actor=actor, reason=reason)
        self.p.monitor.reset_window(name)

    def rollback(self, name: str, actor: str, reason: str) -> str:
        version = self.p.registry.rollback(name, actor=actor, reason=reason)
        self.p.monitor.reset_window(name)
        return version
