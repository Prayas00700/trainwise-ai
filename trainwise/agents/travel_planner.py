"""Module 4 - Agentic AI travel planner.

Loop: understand -> guardrails -> search -> enrich (consult specialist models) -> optimise under
constraints (with transparent relaxation) -> independent validation -> recommend -> book ONLY with an
explicit, signed, expiring passenger approval. Every run leaves a trace for Watchtower's behaviour monitor.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta

from trainwise.agents.nlu import parse_travel_query
from trainwise.models.delay import prob_below

INJECTION = re.compile(
    r"ignore (?:all |any |the |your )?(?:previous |prior |earlier )?(?:rules|instructions|policies|guardrails)"
    r"|without (?:asking|approval|my approval|confirmation|confirming|verification)|bypass"
    r"|skip (?:the )?(?:approval|verification|otp|checks)|disable (?:the )?(?:checks|guardrails|safety)"
    r"|developer mode|jailbreak|act as (?:an? )?admin", re.I)

# (key, min P(meet deadline), min P(travel), budget multiplier, use deadline, note shown to the passenger)
RELAXATION_LADDER = [
    ("strict", 0.50, 0.50, 1.00, True, None),
    ("lower_deadline_confidence", 0.30, 0.50, 1.00, True, "Accepted a lower (30%+) chance of arriving by your deadline."),
    ("accept_waitlist", 0.50, 0.15, 1.00, True, "Accepted a waitlisted option with a 15%+ chance of a seat."),
    ("budget_plus_15", 0.50, 0.50, 1.15, True, "Allowed up to 15% over your budget."),
    ("drop_deadline", 0.00, 0.50, 1.00, False, "No option met your arrival time; ignored the deadline."),
    ("anything_in_budget", 0.00, 0.00, 1.00, False, "Showing the best option within budget regardless of seat chance."),
]


class AgentTrace:
    def __init__(self, agent: str, query: str, synthetic: bool = False):
        self.agent, self.query, self.synthetic = agent, query, synthetic
        self.t0 = time.perf_counter()
        self.steps, self.tool_calls, self.violations = [], [], []
        self.tool_errors = self.validation_failures = 0

    def step(self, name: str, detail) -> None:
        self.steps.append({"step": name, "t_ms": round((time.perf_counter() - self.t0) * 1000, 1), "detail": detail})

    def tool(self, name: str, fn, *args, **kwargs):
        t = time.perf_counter()
        try:
            out, ok = fn(*args, **kwargs), True
        except Exception as exc:
            out, ok = None, False
            self.tool_errors += 1
            self.step(f"tool_error:{name}", str(exc))
        self.tool_calls.append({"tool": name, "ok": ok, "ms": round((time.perf_counter() - t) * 1000, 1)})
        return out

    def summary(self) -> dict:
        counts = Counter(c["tool"] for c in self.tool_calls)
        return {"agent": self.agent, "steps": len(self.steps), "tool_calls": len(self.tool_calls),
                "tool_errors": self.tool_errors, "violations": list(self.violations),
                "validation_failures": self.validation_failures, "unapproved_booking_attempts": 0,
                "loop_detected": any(v > 5 for v in counts.values()), "synthetic": self.synthetic,
                "latency_ms": round((time.perf_counter() - self.t0) * 1000, 1)}


class TravelPlannerAgent:
    name = "travel_planner"

    def __init__(self, platform):
        self.p = platform
        self.secret = os.getenv("TRAINWISE_APPROVAL_SECRET", secrets.token_hex(16)).encode()
        self.plans: dict = {}
        self.strict_mode = False

    # ------------------------------------------------------------------ planning
    def plan(self, query: str, user_id: str | None = None, synthetic: bool = False) -> dict:
        trace = AgentTrace(self.name, query, synthetic)
        intent = parse_travel_query(query, self.p.today())
        if self.p.llm and intent.missing:
            self._llm_fill(query, intent, trace)
        trace.step("understand", {k: v for k, v in intent.as_dict().items() if v not in (None, [], "", 1)})
        cap = self.p.policies["agents"]["max_passengers_per_booking"]
        if INJECTION.search(query):
            trace.violations.append("prompt_injection_attempt")
            intent.assumptions.append("Your message asked me to skip approval. I never book without your explicit approval.")
        if intent.passengers > cap:
            trace.violations.append("passenger_cap_exceeded")
            intent.assumptions.append(f"You asked for {intent.passengers} passengers; one booking is capped at {cap}.")
            intent.passengers = cap
        if trace.violations:
            trace.step("guardrails", {"violations": trace.violations, "action": "unsafe instructions ignored; approval still required"})
        if intent.missing:
            return self._finish(trace, {"status": "needs_clarification", "intent": intent.as_dict(),
                                        "question": "Which stations are you travelling between? For example: "
                                                    "'Pune to Delhi tomorrow under ₹1500'."})
        journey = intent.journey_date
        trains = trace.tool("search_trains", self.p.tools.search_trains, intent.origins, intent.destinations, journey) or []
        trace.step("search", {"journey_date": journey.isoformat(), "trains_found": len(trains)})
        if not trains:
            return self._finish(trace, {"status": "no_trains", "intent": intent.as_dict(),
                                        "message": f"No trains run between {intent.origin_text} and "
                                                   f"{intent.destination_text} on {journey:%d %b %Y}."})
        options = trace.tool("build_options", self.p.tools.build_options, trains, journey, intent.classes or None) or []
        trace.step("enrich", {"options": len(options), "consulted": ["seat inventory", "fare engine",
                                                                     "confirmation predictor", "delay predictor"]})
        deadline, note = self._deadline(intent, options)
        if note:
            intent.assumptions.append(note)
        for o in options:
            o["passengers"], o["total_fare"] = intent.passengers, o["fare"] * intent.passengers
            if deadline:
                slack = (deadline - datetime.fromisoformat(o["arrival"])).total_seconds() / 60
                q = o["delay_quantiles"]
                o["deadline_slack_min"] = round(slack)
                o["prob_meet_deadline"] = round(prob_below(slack, q["p10"], q["p50"], q["p90"]), 4) if slack >= 0 else 0.0
        feasible, step, excluded = self._constraints(options, intent, deadline)
        trace.step("optimise", {"feasible": len(feasible), "relaxation": step[0], "excluded": len(excluded)})
        if not feasible:
            return self._finish(trace, {"status": "no_feasible_option", "intent": intent.as_dict(), "excluded": excluded,
                                        "message": "Nothing fits your budget even after relaxing other constraints."})
        ranked = self._rank(feasible, intent, user_id, deadline, log=not synthetic)
        best, checks = None, []
        for cand in ranked:
            verdict = self._validate(cand, intent, deadline, step[0])
            checks.append({"option": cand["option_id"], **verdict})
            if verdict["passed"]:
                best = cand
                break
            trace.validation_failures += 1
        trace.step("validate", checks)
        if best is None:
            return self._finish(trace, {"status": "no_valid_option", "intent": intent.as_dict(),
                                        "message": "Every candidate failed independent validation."})
        also = self._also_consider(options, intent, deadline, user_id, step[0], best["option_id"])
        plan_id = uuid.uuid4().hex[:10]
        token, expires = self._token(plan_id, best["option_id"])
        self.plans[plan_id] = {"option": best, "intent": intent.as_dict(), "flagged": bool(trace.violations),
                               "created": time.time(), "user_id": user_id}
        return self._finish(trace, {
            "status": "plan_ready", "plan_id": plan_id, "intent": intent.as_dict(), "recommendation": best,
            "alternatives": [r for r in ranked if r["option_id"] != best["option_id"]][:3], "also_consider": also,
            "excluded": excluded, "relaxation": step[0], "assumptions": intent.assumptions + ([step[5]] if step[5] else []),
            "risks": self._risks(best, deadline), "summary": self._summary(best, deadline),
            "deadline": deadline.isoformat(timespec="minutes") if deadline else None,
            "approval": {"required": True, "option_id": best["option_id"], "token": token, "expires_at": expires}})

    def _rank(self, options, intent, user_id, deadline, log=True) -> list:
        ranked = self.p.recommender.rank(options, intent.preferences, user_id, intent.budget, log=log, context="planner")
        for r in ranked:   # the planner adds deadline confidence on top of the recommender's score
            r["plan_score"] = round(r["score"] + (0.25 * r.get("prob_meet_deadline", 0) if deadline else 0), 4)
        return sorted(ranked, key=lambda r: -r["plan_score"])

    def _deadline(self, intent, options):
        if not intent.arrive_by or not options:
            return None, None
        for k in range(4):
            dl = datetime.combine(intent.journey_date + timedelta(days=k), intent.arrive_by)
            if any(datetime.fromisoformat(o["arrival"]) <= dl for o in options):
                note = None if k == 0 else (f"Read 'reach before {intent.arrive_by:%H:%M}' as {dl:%H:%M} on {dl:%a %d %b}, "
                                            f"the earliest such time any train leaving {intent.journey_date:%d %b} can make.")
                return dl, note
        dl = datetime.combine(intent.journey_date + timedelta(days=1), intent.arrive_by)
        return dl, f"No train arrives by {intent.arrive_by:%H:%M} within three days; ranking by closeness to it."

    def _reasons(self, o, intent, deadline, p_meet=0.5, p_travel=0.5, budget_mult=1.0, use_deadline=True) -> list:
        r = []
        if intent.budget and o["fare"] > intent.budget * budget_mult:
            r.append(f"fare ₹{o['fare']} is over the ₹{intent.budget} budget")
        if use_deadline and deadline is not None and o.get("prob_meet_deadline", 0) < p_meet:
            r.append(f"arrives {o['arrival'][11:16]} with a {round(100 * o.get('prob_meet_deadline', 0))}% chance of "
                     f"making {deadline:%H:%M}")
        if o["travel_probability"] < p_travel:
            r.append(f"{o['availability']}: only {round(100 * o['travel_probability'])}% chance of a seat")
        dep = datetime.fromisoformat(o["departure"]).time()
        if intent.depart_after and dep < intent.depart_after:
            r.append(f"departs {dep:%H:%M}, before {intent.depart_after:%H:%M}")
        if intent.depart_before and dep > intent.depart_before:
            r.append(f"departs {dep:%H:%M}, after {intent.depart_before:%H:%M}")
        return r

    def _constraints(self, options, intent, deadline):
        for step in RELAXATION_LADDER:
            key, p_meet, p_travel, bm, use_dl, _ = step
            feasible = [o for o in options if not self._reasons(o, intent, deadline, p_meet, p_travel, bm, use_dl)]
            if feasible:
                ids = {o["option_id"] for o in feasible}
                excluded = [{"option": f"{o['train_number']} {o['train_name']} ({o['class']})",
                             "reasons": self._reasons(o, intent, deadline)} for o in options if o["option_id"] not in ids]
                return feasible, step, excluded
        return [], RELAXATION_LADDER[-1], [{"option": f"{o['train_number']} {o['train_name']} ({o['class']})",
                                            "reasons": self._reasons(o, intent, deadline)} for o in options]

    def _also_consider(self, options, intent, deadline, user_id, chosen_key, best_id) -> list:
        """Best option under each *other* relaxation: makes the trade-offs visible instead of hiding them."""
        out, seen = [], {best_id}
        for key, p_meet, p_travel, bm, use_dl, note in RELAXATION_LADDER[1:4]:
            if key == chosen_key:
                continue
            feasible = [o for o in options if not self._reasons(o, intent, deadline, p_meet, p_travel, bm, use_dl)]
            if feasible:
                top = self._rank(feasible, intent, user_id, deadline, log=False)[0]
                if top["option_id"] not in seen:
                    seen.add(top["option_id"])
                    out.append({"trade_off": note, "option": top, "risks": self._risks(top, deadline)})
        return out

    def _validate(self, cand, intent, deadline, step_key) -> dict:
        """Independent critic: re-derives facts from source systems instead of trusting the ranked option."""
        train = self.p.network.get(cand["train_number"])
        journey = intent.journey_date
        fare = self.p.fare.quote(train, cand["class"], journey)["fare"]
        checks = {"runs_on_date": train.runs_on(journey), "class_offered": cand["class"] in train.classes,
                  "route_matches": train.origin in intent.origins and train.destination in intent.destinations,
                  "fare_reconfirmed": fare == cand["fare"],
                  "within_budget": intent.budget is None or fare <= intent.budget * (1.15 if step_key == "budget_plus_15" else 1.0)}
        if deadline and step_key not in ("drop_deadline", "anything_in_budget"):
            checks["scheduled_before_deadline"] = train.arrival_dt(journey) <= deadline
        return {"passed": all(checks.values()), "checks": checks}

    @staticmethod
    def _risks(o, deadline) -> list:
        r = []
        if o["travel_probability"] < 0.9:
            r.append(f"Seat not guaranteed: {o['availability']}, {round(100 * o['confirmation_probability'])}% chance of a "
                     f"confirmed berth and {round(100 * o['travel_probability'])}% chance of at least RAC.")
        if deadline and o.get("prob_meet_deadline", 1) < 0.85:
            r.append(f"Tight arrival: {round(100 * o['prob_meet_deadline'])}% chance of reaching by {deadline:%H:%M} "
                     f"after predicted delays (expected delay {o['expected_delay_min']:.0f} min).")
        if o["dynamic_pricing"]:
            r.append(f"Flexi fare (currently x{o['flexi_multiplier']}): the price can rise as seats sell.")
        if o.get("delay_risk") in ("high", "severe"):
            r.append(f"Delay risk is {o['delay_risk']}.")
        return r

    @staticmethod
    def _summary(o, deadline) -> str:
        dep, arr = datetime.fromisoformat(o["departure"]), datetime.fromisoformat(o["arrival"])
        s = (f"Take {o['train_number']} {o['train_name']} in {o['class_label']}: leaves {dep:%a %d %b %H:%M}, arrives "
             f"{arr:%a %d %b %H:%M}, ₹{o['fare']} per passenger. Seat status {o['availability']} "
             f"({round(100 * o['travel_probability'])}% chance of travelling).")
        if deadline and "prob_meet_deadline" in o:
            s += f" {round(100 * o['prob_meet_deadline'])}% chance of arriving by {deadline:%H:%M} after expected delays."
        return s

    # ------------------------------------------------------------------ approval & booking
    def _token(self, plan_id: str, option_id: str):
        expires = int(time.time()) + 60 * int(self.p.policies["agents"]["approval_ttl_minutes"])
        sig = hmac.new(self.secret, f"{plan_id}|{option_id}|{expires}".encode(), hashlib.sha256).hexdigest()[:32]
        return f"{expires}.{sig}", expires

    def _valid(self, plan_id, option_id, token) -> bool:
        plan = self.plans.get(plan_id)
        if not plan or not token or "." not in token:
            return False
        exp, sig = token.split(".", 1)
        if not exp.isdigit() or int(exp) < time.time() or plan["option"]["option_id"] != option_id:
            return False
        good = hmac.new(self.secret, f"{plan_id}|{option_id}|{exp}".encode(), hashlib.sha256).hexdigest()[:32]
        return hmac.compare_digest(sig, good)

    def book(self, plan_id: str, option_id: str, approval_token: str | None = None, session: dict | None = None,
             synthetic: bool = False) -> dict:
        if not self._valid(plan_id, option_id, approval_token):
            self.p.monitor.log("agent_traces", {"agent": self.name, "steps": 1, "tool_calls": 0, "tool_errors": 0,
                                                "violations": ["unapproved_booking_attempt"],
                                                "unapproved_booking_attempts": 1, "validation_failures": 0,
                                                "loop_detected": False, "latency_ms": 0.0, "synthetic": synthetic})
            self.p.audit.record(self.name, "booking_denied", plan_id or "-",
                                {"option": option_id, "reason": "missing, invalid or expired approval"}, severity="warning")
            return {"status": "denied", "reason": "Booking needs the passenger's explicit approval "
                                                  "(a valid, unexpired approval token for this exact option)."}
        plan = self.plans[plan_id]
        if plan["flagged"] and self.strict_mode and not (session or {}).get("reverified"):
            return {"status": "verification_required", "reason": "This session was flagged for unsafe instructions; "
                                                                 "please re-verify with OTP before booking."}
        if session:
            risk = self.p.services["fraud_detector"].predict(session)
            if risk.get("decision") == "block":
                self.p.audit.record(self.name, "booking_blocked_fraud", plan_id, {"risk": risk["risk_score"]}, "warning")
                return {"status": "blocked", "reason": "Session blocked by fraud screening.", "risk": risk}
            if risk.get("decision") == "challenge":
                return {"status": "verification_required", "reason": "Extra verification needed (OTP / CAPTCHA).", "risk": risk}
        o = plan["option"]
        pnr = str(int(hashlib.sha256(f"{plan_id}|{option_id}".encode()).hexdigest(), 16))[-10:]
        booking = {"status": "booked", "pnr": pnr, "train": f"{o['train_number']} {o['train_name']}", "class": o["class"],
                   "departure": o["departure"], "availability_at_booking": o["availability"],
                   "passengers": o.get("passengers", 1), "fare_per_passenger": o["fare"],
                   "total_fare": o.get("total_fare", o["fare"]),
                   "note": "Simulated booking: no real reservation or payment was made."}
        self.p.audit.record(self.name, "booking_confirmed", pnr, {"plan": plan_id, "option": option_id,
                                                                  "total_fare": booking["total_fare"]})
        return booking

    # ------------------------------------------------------------------ helpers
    def _llm_fill(self, query, intent, trace) -> None:
        data = self.p.llm.extract_json("Extract railway travel details. Reply with JSON only: "
                                       '{"origin_city": string or null, "destination_city": string or null}', query)
        if not data:
            return
        for key, attr in (("origin_city", "origins"), ("destination_city", "destinations")):
            codes = self.p.network.resolve(str(data.get(key) or ""))
            if codes and not getattr(intent, attr):
                setattr(intent, attr, codes)
        intent.missing = [k for k, v in (("origin", intent.origins), ("destination", intent.destinations)) if not v]
        trace.step("llm_assist", {"filled": data})

    def _finish(self, trace: AgentTrace, result: dict) -> dict:
        summary = trace.summary()
        self.p.monitor.log("agent_traces", summary)
        self.p.audit.record(self.name, f"plan_{result['status']}", result.get("plan_id", "-"),
                            {"query": trace.query[:200], "violations": trace.violations,
                             "recommendation": (result.get("recommendation") or {}).get("option_id")},
                            severity="warning" if trace.violations else "info")
        return {**result, "trace": trace.steps, "trace_summary": summary}
