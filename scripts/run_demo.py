"""End-to-end demo: passenger journeys, then a chaos drill that Watchtower detects, diagnoses and fixes.

Usage:  python scripts/run_demo.py            (about a minute; uses a fresh artifacts/demo directory)
Set TRAINWISE_TODAY=YYYY-MM-DD to pin the date used for "today" / "tomorrow".
"""
import os
import shutil
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainwise.config import ROOT_DIR  # noqa: E402
from trainwise.data.generators import make_confirmation_data  # noqa: E402
from trainwise.governance.audit import AuditLog  # noqa: E402
from trainwise.governance.reports import governance_report, save_report  # noqa: E402
from trainwise.platform import Platform  # noqa: E402
from trainwise.simulation import SCENARIOS  # noqa: E402

LOW_AND_SLOW = {"requests_per_min": 7, "avg_inter_click_ms": 2100, "form_fill_seconds": 41, "captcha_solve_seconds": 7,
                "accounts_on_device": 1, "ips_last_24h": 1, "bookings_last_24h": 2, "tatkal_window_hit": 1,
                "payment_failures_24h": 0, "passengers_per_booking": 5, "account_age_days": 380, "headless_signal": 0,
                "device_type": "mobile", "region_segment": "tier2"}


def h(title: str) -> None:
    print("\n" + "=" * 88 + f"\n{title}\n" + "=" * 88)


def pct(x) -> str:
    return f"{round(100 * x)}%"


def main() -> None:
    t0 = time.perf_counter()
    today = date.fromisoformat(os.environ["TRAINWISE_TODAY"]) if os.getenv("TRAINWISE_TODAY") else None
    h("0. Bootstrap: synthetic data -> train 5 models -> register v1 champions")
    p = Platform(artifact_dir=ROOT_DIR / "artifacts" / "demo", fresh=True, today=today)
    print(f"Trained in {p.bootstrap_report['seconds']} s (today = {p.today()})")
    for row in p.registry.table():
        print(f"  {row['model']:<24} {row['version']}  {row['metrics']}")

    h("1. Agentic travel planner (Module 4) on the brief's example query")
    query = "I want to travel from Pune to Delhi tomorrow under ₹1500 and reach before 10:00 AM."
    plan = p.planner.plan(query, user_id="U001")
    it = plan["intent"]
    print(f"Query : {query}")
    print(f"Parsed: {it['origins']} -> {it['destinations']}, {it['journey_date']}, budget ₹{it['budget']}, "
          f"arrive by {it['arrive_by']}")
    print("Agent steps:", " -> ".join(s["step"] for s in plan["trace"]))
    best = plan["recommendation"]
    print(f"\nRecommendation: {plan['summary']}")
    for r in best["rationale"]:
        print(f"  {'risk' if r.startswith('risk:') else 'why '}: {r.replace('risk: ', '')}")
    for r in plan["risks"]:
        print(f"  risk: {r}")
    for a in plan["assumptions"]:
        print(f"  note: {a}")
    for alt in plan["alternatives"]:
        print(f"  alternative: {alt['train_number']} {alt['train_name']} {alt['class']} ₹{alt['fare']} "
              f"{alt['availability']}, {pct(alt.get('prob_meet_deadline', 0))} on-time for the deadline")
    for alt in plan["also_consider"]:
        o = alt["option"]
        print(f"  trade-off ({alt['trade_off']}): {o['train_number']} {o['class']} ₹{o['fare']} {o['availability']}")
    print(f"  excluded {len(plan['excluded'])} options, e.g. {plan['excluded'][0]['option']}: "
          f"{'; '.join(plan['excluded'][0]['reasons'])}")
    print("\nBooking without approval :",
          p.planner.book(plan["plan_id"], best["option_id"], None, synthetic=True)["status"])
    booking = p.planner.book(plan["plan_id"], best["option_id"], plan["approval"]["token"])
    print(f"Booking with approval    : {booking['status']}, PNR {booking['pnr']} ({booking['note']})")

    h("2. Ticket-confirmation predictor (Module 1) with explanation")
    train = p.network.get("92201")
    journey = p.today() + timedelta(days=3)
    snap = {"kind": "WL", "position": 18, "days_before": 3}
    v1_pred = p.tools.confirmation_for(train, "SL", journey, snap, explain=True)
    print(f"92201 SL, GNWL 18, {journey}: confirm {pct(v1_pred['confirmation_probability'])}, "
          f"at least RAC {pct(v1_pred['travel_probability'])}, likely final status {v1_pred['predicted_final_status']} "
          f"(model {v1_pred['version']}, confidence {v1_pred['confidence']})")
    for e in v1_pred["explanation"][:4]:
        print(f"  {e['feature']:<24} = {e['value']!s:<8} (typical {e['typical']!s:<6}) moves P(confirm) by {e['effect']:+.1%}")

    h("3. Delay & disruption prediction (Module 3)")
    out = p.tools.delay_outlook(p.network.get("92205"), p.today() + timedelta(days=1))
    print(f"92205 Deccan–Delhi Express: expected delay {out['expected_delay_min']:.0f} min "
          f"(80% interval {out['delay_interval_min']}), on time within 15 min: {pct(out['prob_on_time_15min'])}, "
          f"risk {out['risk_level']}")
    for c in out["congestion"]:
        print(f"  congestion at {c['station_name']} around {c['time'][11:]}: {c['level']} ({c['congestion_index']})")
    for a in out["alerts"]:
        print(f"  alert: {a}")
    for a in out["alternatives"]:
        print(f"  better-running alternative: {a['train']} (expected delay {a['expected_delay_min']:.0f} min)")

    h("4. Dynamic fare prediction (Module 5)")
    f = p.fare.forecast(p.network.get("92209"), "3A", p.today() + timedelta(days=45))
    print(f"92209 Rajdhani 3A in 45 days: ₹{f['current']['fare']} now (flexi x{f['current']['flexi_multiplier']}). {f['advice']}")
    print(f"  demand surge: {f['demand_surge']['surge']} (index {f['demand_surge']['demand_index']}; "
          f"drivers: {', '.join(f['demand_surge']['drivers']) or 'none'})")
    for s in f["savings"][:2]:
        print(f"  save ₹{s['saving']}: {s['option']} at ₹{s['fare']}")

    h("5. Multilingual support assistant (Module 6) with groundedness guard")
    for msg in ["What is the status of PNR 4521873690?", "मेरा पीएनआर 6203914785 का स्टेटस क्या है",
                "What are the cancellation charges?", "kal pune se delhi jana hai 1500 ke andar"]:
        r = p.support.chat(msg, session_id="demo")
        print(f"\n> {msg}\n  [{r['language']}, intent {r['intent']} {r['confidence']:.2f}, groundedness "
              f"{r['guard']['groundedness']}]\n  {r['reply'][:330]}")

    h("6. Fraud, security & bot detection (Module 7)")
    bot = {**LOW_AND_SLOW, "requests_per_min": 120, "avg_inter_click_ms": 120, "form_fill_seconds": 3,
           "captcha_solve_seconds": 1.5, "headless_signal": 1, "device_type": "desktop", "bookings_last_24h": 12}
    for label, session in (("bulk booking bot", bot), ("new low-and-slow bot", LOW_AND_SLOW)):
        r = p.services["fraud_detector"].predict(session)
        print(f"  {label:<22} risk {r['risk_score']:>5} -> {r['decision']:<9} {r['suspected_pattern'] or ''} {r['reasons'][:2]}")
    print("  (the low-and-slow pattern is a blind spot of v1: it was never in the training data)")

    h("7. Normal production traffic -> governance cycle")
    p.simulator.normal()
    c = p.supervisor.run_cycle()
    print(c["summary"])

    h("8. Chaos drill: eight real-world failures injected at once")
    for name in SCENARIOS:
        r = p.simulator.inject(name)
        print(f"  {SCENARIOS[name]['label']:<22} -> {r['target']:<24} {r['description']}")

    h("9. Watchtower autonomous cycle: observe -> prioritise -> root-cause -> remediate -> verify -> audit")
    c = p.supervisor.run_cycle()
    print(f"{len(c['incidents'])} incidents, {c['autonomous_actions']} autonomous actions, {c['duration_s']} s\n")
    for inc in c["incidents"]:
        print(f"{inc['priority']} {inc['target']} [{inc['status']}]")
        for fnd in inc["findings"]:
            print(f"   found by {fnd['agent']}: {fnd['summary']}")
        print(f"   root cause ({pct(inc['rca']['confidence'])}): {inc['rca']['root_cause']}")
        for a in inc["actions"]:
            print(f"   action {a['action']}: {a['status']}" + (f" | verified: {a['verification']}" if a.get("verification") else ""))
        if inc["rca"]["blast_radius"]:
            print(f"   downstream exposure: {', '.join(inc['rca']['blast_radius'])}")
        print()

    h("10. Human in the loop: the high-risk fraud model needs an approval")
    for a in [a for a in p.approvals if a["status"] == "pending"]:
        ev = a["evidence"] or {}
        print(f"{a['approval_id']}: {a['action']} {a['target']} {a['version']} ({a['why']})")
        print(f"  evidence: hold-out recall {ev.get('champion_score')} -> {ev.get('challenger_score')}, guards {ev.get('guards')}")
        d = p.supervisor.decide(a["approval_id"], True, "fraud-ops-lead", "reviewed evidence and sample alerts")
        print(f"  approved by fraud-ops-lead; post-deployment check: {d['post_decision_check']['summary']}")
    r = p.services["fraud_detector"].predict(LOW_AND_SLOW)
    print(f"  same low-and-slow session on {r['version']}: risk {r['risk_score']} -> {r['decision']} ({r['suspected_pattern']})")

    h("11. Closed loop: 400 unseen festive-season tickets, before vs after autonomous retraining")
    test = make_confirmation_data(400, 4242, regime="festival_surge")
    actual = (test["final_status"] == "CNF").mean()
    for v in dict.fromkeys(["v1", p.registry.champion("confirmation_predictor")]):
        m = p.registry.model("confirmation_predictor", v)
        print(f"  {v}: accuracy {m.evaluate(test)['accuracy']:.3f}, average predicted P(confirm) "
              f"{m.proba(test)['CNF'].mean():.0%} vs actual confirmation rate {actual:.0%}")

    h("12. Follow-up governance cycle")
    print(p.supervisor.run_cycle()["summary"])

    h("13. Audit trail: hash-chained and tamper-evident")
    print("Live audit log:", p.audit.verify())
    tmp = Path(tempfile.mkdtemp()) / "audit_copy.jsonl"
    shutil.copy(p.audit.path, tmp)
    lines = tmp.read_text(encoding="utf-8").splitlines()
    lines[5] = lines[5].replace('"severity": "info"', '"severity": "debug"', 1)
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Copy with one edited line:", AuditLog(tmp).verify())

    path = save_report(p, governance_report(p))
    print(f"\nGovernance report: {path.relative_to(ROOT_DIR)}")
    print(f"Demo finished in {time.perf_counter() - t0:.0f} s.")


if __name__ == "__main__":
    main()
