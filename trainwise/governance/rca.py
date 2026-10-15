"""Autonomous root-cause analysis over the AI dependency graph.

Each incident is explained by competing hypotheses scored from the evidence the governance agents
collected (data-quality failures, drift, recent deployments, upstream incidents, resource load).
The top hypothesis selects the remediation; the full ranked list is kept for the audit trail.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

DEPENDENCIES = {
    "travel_planner": ["train_recommender", "confirmation_predictor", "delay_predictor", "fare_demand_forecaster",
                       "fraud_detector"],
    "support_assistant": ["intent_classifier", "policy_kb", "travel_planner", "confirmation_predictor"],
    "train_recommender": ["confirmation_predictor", "delay_predictor", "fare_demand_forecaster"],
    "confirmation_predictor": ["pipeline:prs_bookings", "pipeline:prs_inventory", "pipeline:calendar"],
    "delay_predictor": ["pipeline:timetable", "pipeline:weather_feed", "pipeline:ntes_running_status",
                        "pipeline:engineering_blocks"],
    "fare_demand_forecaster": ["pipeline:prs_inventory", "pipeline:calendar"],
    "fraud_detector": ["pipeline:session_telemetry"],
    "intent_classifier": ["pipeline:chat_channel"],
}


def _bfs(start: str, graph: dict) -> list:
    seen, q, out = {start}, deque([start]), []
    while q:
        for nxt in graph.get(q.popleft(), []):
            if nxt not in seen:
                seen.add(nxt)
                out.append(nxt)
                q.append(nxt)
    return out


def upstream(node: str) -> list:
    return _bfs(node, DEPENDENCIES)


def downstream(node: str) -> list:
    reverse: dict = {}
    for parent, deps in DEPENDENCIES.items():
        for d in deps:
            reverse.setdefault(d, []).append(parent)
    return _bfs(node, reverse)


@dataclass
class Hypothesis:
    cause: str
    confidence: float
    evidence: list = field(default_factory=list)
    remediation: list = field(default_factory=list)


class RootCauseAnalyzer:
    def __init__(self, platform):
        self.p = platform

    def analyze(self, incident: dict, troubled: set) -> dict:
        target = incident["target"]
        by_kind = {f.kind: f for f in incident["findings"]}
        psi_critical = self.p.policies["drift"]["psi_critical"]
        deploy = self.p.registry.recent_deployment(target) if target in self.p.registry.index else None
        hyps = []
        if "data_quality" in by_kind:
            f = by_kind["data_quality"]
            hyps.append(Hypothesis(
                f"Upstream feed fault: pipeline '{f.evidence.get('pipeline')}' is sending invalid "
                f"'{f.evidence.get('top_bad_feature')}' values", 0.9,
                [f.summary, f"sample issues: {f.evidence.get('top_issues')}"], ["quarantine_feed"]))
        if "performance_degradation" in by_kind:
            perf, drift = by_kind["performance_degradation"], by_kind.get("data_drift")
            if deploy:   # before blaming the deployment, replay the same traffic through the previous version
                cmp = self.p.lifecycle.shadow_compare(target, deploy["replaced"])
                if cmp is None:
                    hyps.append(Hypothesis(f"Possible regression from deployment {deploy['version']}", 0.6,
                                           [perf.summary], ["rollback"]))
                else:
                    better = cmp["other"] > cmp["current"] if cmp["higher_is_better"] else cmp["other"] < cmp["current"]
                    gap = abs(cmp["other"] - cmp["current"]) / max(abs(cmp["current"]), 1e-9)
                    shadow = (f"shadow test on the same {cmp['n']} requests: previous {cmp['other_version']} "
                              f"{cmp['metric']} {cmp['other']} vs current {cmp['current_version']} {cmp['current']}")
                    if better and gap > 0.02:
                        hyps.append(Hypothesis(f"Regression introduced by deployment {deploy['version']} ({shadow})",
                                               0.85, [perf.summary, shadow], ["rollback"]))
                    else:
                        hyps.append(Hypothesis(f"Not a deployment regression ({shadow}); the validated baseline "
                                               "was optimistic", 0.65, [perf.summary, shadow], ["notify"]))
            if drift and drift.evidence.get("max_psi", 0) >= psi_critical:
                top = drift.evidence["top"][0]
                shifted = ", ".join(f"{t['feature']} (PSI {t['psi']:.2f}, {t['pipeline']})" for t in drift.evidence["top"])
                hyps.append(Hypothesis(
                    f"Covariate shift: production inputs moved outside the training distribution: {shifted}",
                    round(min(0.95, 0.6 + top["psi"] / 4), 2), [perf.summary, drift.summary], ["retrain"]))
            else:
                hyps.append(Hypothesis(
                    "Concept drift: outcomes changed for inputs that still look like the training data "
                    "(the input-to-outcome relationship moved)", 0.7,
                    [perf.summary] + ([drift.summary] if drift else ["no input drift above the critical PSI"]),
                    ["retrain"]))
        elif "data_drift" in by_kind:
            hyps.append(Hypothesis("Input distribution shift without measurable accuracy loss yet", 0.5,
                                   [by_kind["data_drift"].summary], ["notify"]))
        if "bias" in by_kind:
            b = by_kind["bias"]
            if deploy:
                hyps.append(Hypothesis(f"Change shipped in deployment {deploy['version']} skews outcomes "
                                       f"(by {deploy.get('promoted_by')})", 0.85, [b.summary], ["rollback"]))
            else:
                hyps.append(Hypothesis("Bias emerging from data or feedback loops", 0.6, [b.summary], ["notify"]))
        if "hallucination" in by_kind:
            hyps.append(Hypothesis("Generative replies are not grounded in retrieved evidence", 0.85,
                                   [by_kind["hallucination"].summary], ["tighten_guardrail"]))
        if "agent_anomaly" in by_kind:
            hyps.append(Hypothesis("Adversarial prompts are trying to make the agent book without approval "
                                   "(prompt injection)", 0.8, [by_kind["agent_anomaly"].summary],
                                   ["tighten_guardrail", "notify"]))
        if "sla_breach" in by_kind:
            s = by_kind["sla_breach"]
            if deploy:
                hyps.append(Hypothesis(f"Latency regression from deployment {deploy['version']}", 0.6, [s.summary],
                                       ["rollback"]))
            hyps.append(Hypothesis("Capacity saturation: p95 latency above SLA at current load", 0.7,
                                   [s.summary, f"replicas: {s.evidence.get('replicas')}"], ["scale_out"]))
        for kind in ("low_csat", "escalation", "agent_performance", "errors"):
            if kind in by_kind:
                hyps.append(Hypothesis(f"Quality signal needs human review: {by_kind[kind].summary}", 0.4,
                                       [by_kind[kind].summary], ["notify"]))
        for dep in upstream(target):
            if dep in troubled:
                hyps.append(Hypothesis(f"Upstream dependency degraded: {dep}", 0.5,
                                       [f"{dep} has an open incident in this cycle"], ["notify"]))
        hyps.sort(key=lambda h: -h.confidence)
        best = hyps[0] if hyps else Hypothesis("Undetermined", 0.0, [], ["notify"])
        return {"target": target, "root_cause": best.cause, "confidence": best.confidence,
                "remediation": best.remediation, "hypotheses": [asdict(h) for h in hyps],
                "blast_radius": downstream(target), "upstream": upstream(target)}
