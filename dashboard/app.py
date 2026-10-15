"""TrainWise AI dashboard: passenger services and the Watchtower governance console.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainwise.governance.reports import governance_report  # noqa: E402
from trainwise.platform import get_platform  # noqa: E402
from trainwise.simulation import SCENARIOS  # noqa: E402

st.set_page_config(page_title="TrainWise AI", page_icon="🚆", layout="wide")
st.markdown("""<style>
.lamp{display:inline-block;width:.78rem;height:.78rem;border-radius:50%;margin-right:.2rem;vertical-align:-1px;
      box-shadow:0 0 0 2px rgba(20,32,43,.28) inset}
table.board{width:100%;border-collapse:collapse;font-size:.92rem}
table.board th{text-align:left;font-weight:600;border-bottom:2px solid rgba(20,32,43,.22);padding:.45rem .5rem}
table.board td{border-bottom:1px solid rgba(20,32,43,.09);padding:.5rem;vertical-align:middle}
.brand{font-size:1.85rem;font-weight:700;letter-spacing:-.015em;margin:0;color:#1D4E89}
.brand-sub{opacity:.75;margin:.1rem 0 1rem 0;line-height:1.35}
</style>""", unsafe_allow_html=True)

ASPECTS = {"green": (["#1a9e5c"], "Clear"), "double_yellow": (["#e3a008", "#e3a008"], "Attention"),
           "yellow": (["#e3a008"], "Caution"), "red": (["#c8102e"], "Danger")}
PRESETS = {
    "Typical passenger": dict(requests_per_min=6, avg_inter_click_ms=2200, form_fill_seconds=45, captcha_solve_seconds=7,
                              accounts_on_device=1, ips_last_24h=1, bookings_last_24h=0, tatkal_window_hit=0,
                              payment_failures_24h=0, passengers_per_booking=2, account_age_days=700, headless_signal=0,
                              device_type="mobile", region_segment="metro"),
    "Bulk booking bot": dict(requests_per_min=120, avg_inter_click_ms=120, form_fill_seconds=3, captcha_solve_seconds=1.5,
                             accounts_on_device=2, ips_last_24h=3, bookings_last_24h=12, tatkal_window_hit=1,
                             payment_failures_24h=0, passengers_per_booking=6, account_age_days=20, headless_signal=1,
                             device_type="desktop", region_segment="metro"),
    "Multi-account tout": dict(requests_per_min=18, avg_inter_click_ms=700, form_fill_seconds=15, captcha_solve_seconds=4,
                               accounts_on_device=9, ips_last_24h=2, bookings_last_24h=8, tatkal_window_hit=1,
                               payment_failures_24h=0, passengers_per_booking=6, account_age_days=120, headless_signal=0,
                               device_type="desktop", region_segment="tier2"),
    "Low-and-slow bot ring": dict(requests_per_min=7, avg_inter_click_ms=2100, form_fill_seconds=41, captcha_solve_seconds=7,
                                  accounts_on_device=1, ips_last_24h=1, bookings_last_24h=2, tatkal_window_hit=1,
                                  payment_failures_24h=0, passengers_per_booking=5, account_age_days=380, headless_signal=0,
                                  device_type="mobile", region_segment="tier2"),
}


def lamp(aspect: str) -> str:
    colors, label = ASPECTS[aspect]
    return "".join(f'<span class="lamp" style="background:{c}"></span>' for c in colors) + f" {label}"


@st.cache_resource(show_spinner="Training models on synthetic data (first run only, about 30 seconds)...")
def platform():
    return get_platform()


P = platform()
PAGES = ["Plan a journey", "Confirmation chances", "Train suggestions", "Delays and disruptions", "Fare forecast",
         "Support chat", "Fraud and bots", "Watchtower governance"]

with st.sidebar:
    st.markdown('<p class="brand">TrainWise AI</p><p class="brand-sub">Railway travel assistants, watched over by '
                'Watchtower, an autonomous AI-governance layer.</p>', unsafe_allow_html=True)
    page = st.radio("Go to", PAGES, label_visibility="collapsed")
    st.caption(f"Today: {P.today():%d %b %Y}. Trains, fares, passengers and incidents are synthetic.")


def train_picker(key: str, trains=None, default: str = "92201"):
    trains = trains or sorted(P.network.trains.values(), key=lambda t: t.number)
    idx = next((i for i, t in enumerate(trains) if t.number == default), 0)
    return st.selectbox("Train", trains, index=idx, format_func=lambda t: f"{t.number} {t.name}", key=key)


def when(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%a %d %b, %H:%M")


# ---------------------------------------------------------------- pages
def page_planner():
    st.header("Plan a journey")
    st.write("Describe the trip in your own words. The agent checks seats, delays and fares, explains its choice, "
             "and books only after you approve.")
    query = st.text_area("Your trip", "I want to travel from Pune to Delhi tomorrow under ₹1500 and reach before 10:00 AM.",
                         height=80)
    user = st.selectbox("Passenger profile (uses travel history)", ["Guest"] + list(P.users)[:8], index=1)
    if st.button("Plan my journey", type="primary"):
        st.session_state["plan"] = P.planner.plan(query, user_id=None if user == "Guest" else user)
        st.session_state.pop("booking", None)
    plan = st.session_state.get("plan")
    if not plan:
        return
    if plan["status"] != "plan_ready":
        st.warning(plan.get("question") or plan.get("message"))
        return
    best = plan["recommendation"]
    st.subheader(f"{best['train_number']} {best['train_name']}, {best['class_label']}")
    c = st.columns(4)
    c[0].metric("Fare per passenger", f"₹{best['fare']:,}")
    c[1].metric("Departs", when(best["departure"]))
    c[2].metric("Arrives", when(best["arrival"]))
    c[3].metric("Chance of travelling", f"{best['travel_probability']:.0%}", help=f"Current status {best['availability']}")
    st.write(plan["summary"])
    for r in best["rationale"]:
        st.markdown(f"- {r}")
    for r in plan["risks"]:
        st.warning(r)
    for a in plan["assumptions"]:
        st.info(a)
    rows = [{"Train": f"{o['train_number']} {o['train_name']}", "Class": o["class"], "Fare (₹)": o["fare"],
             "Seat": o["availability"], "Chance of travelling": f"{o['travel_probability']:.0%}", "Trade-off": ""}
            for o in plan["alternatives"]]
    rows += [{"Train": f"{a['option']['train_number']} {a['option']['train_name']}", "Class": a["option"]["class"],
              "Fare (₹)": a["option"]["fare"], "Seat": a["option"]["availability"],
              "Chance of travelling": f"{a['option']['travel_probability']:.0%}", "Trade-off": a["trade_off"]}
             for a in plan["also_consider"]]
    if rows:
        st.markdown("**Other options**")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    with st.expander(f"Options ruled out ({len(plan['excluded'])})"):
        st.dataframe(pd.DataFrame([{"Option": e["option"], "Why": "; ".join(e["reasons"])} for e in plan["excluded"]]),
                     hide_index=True)
    with st.expander("How the agent worked"):
        st.json(plan["trace"], expanded=False)
    c1, c2 = st.columns(2)
    if c1.button("Approve and book", type="primary"):
        st.session_state["booking"] = P.planner.book(plan["plan_id"], best["option_id"], plan["approval"]["token"])
    if c2.button("Try to book without approval"):
        st.session_state["booking"] = P.planner.book(plan["plan_id"], best["option_id"], None)
    b = st.session_state.get("booking")
    if b:
        if b["status"] == "booked":
            st.success(f"Booked. PNR {b['pnr']}, total ₹{b['total_fare']:,}. {b['note']}")
        else:
            st.error(f"Booking {b['status']}: {b['reason']}")


def page_confirmation():
    st.header("Will my waitlisted ticket confirm?")
    c1, c2, c3 = st.columns(3)
    with c1:
        train = train_picker("conf_train")
    cls = c2.selectbox("Class", list(train.classes), key="conf_cls")
    days = c3.slider("Days until the journey", 0, 60, 3)
    c4, c5, c6 = st.columns(3)
    status = c4.radio("Current status", ["WL", "RAC"], horizontal=True)
    pos = c5.number_input("Waitlist or RAC number", 1, 500, 24)
    quota = c6.selectbox("Quota", ["GN", "TQ", "LD", "SS"], format_func=lambda q: {
        "GN": "General", "TQ": "Tatkal", "LD": "Ladies", "SS": "Lower berth / senior citizen"}[q])
    if st.button("Predict", type="primary"):
        journey = P.today() + timedelta(days=days)
        st.session_state["conf"] = P.tools.confirmation_for(
            train, cls, journey, {"kind": status, "position": int(pos), "days_before": days}, quota=quota, explain=True)
    r = st.session_state.get("conf")
    if not r:
        return
    if r.get("status") != "ok":
        st.error(f"No prediction: {r.get('issues')}")
        return
    c = st.columns(4)
    c[0].metric("Chance of a confirmed berth", f"{r['confirmation_probability']:.0%}")
    c[1].metric("Chance of at least RAC", f"{r['travel_probability']:.0%}")
    c[2].metric("Likely status at charting", r["predicted_final_status"])
    c[3].metric("Model confidence", f"{r['confidence']:.0%}")
    st.bar_chart(pd.Series(r["final_status_probabilities"], name="probability"))
    st.markdown("**What drives this estimate**")
    st.dataframe(pd.DataFrame([{"Factor": e["feature"], "Your value": e["value"], "Typical value": e["typical"],
                                "Effect on P(confirm)": f"{e['effect']:+.1%}"} for e in r["explanation"]]), hide_index=True)
    st.caption(f"Model {r['model']} {r['version']}; request {r['request_id']} is logged for monitoring.")


def page_recommend():
    st.header("Which train should I take?")
    cities = P.network.cities()
    c1, c2, c3 = st.columns(3)
    origin = c1.selectbox("From", cities, index=cities.index("Pune"))
    dest = c2.selectbox("To", cities, index=cities.index("Delhi"))
    days = c3.slider("Days from today", 1, 60, 1)
    c4, c5, c6 = st.columns(3)
    prefs = c4.multiselect("What matters most", ["cheapest", "fastest", "punctual", "confirmed"])
    user = c5.selectbox("Passenger profile", ["Guest"] + list(P.users)[:8], key="rec_user")
    budget = c6.number_input("Budget per passenger (₹, 0 for none)", 0, 20000, 0, step=100)
    if st.button("Suggest trains", type="primary"):
        st.session_state["recs"] = P.recommender.recommend(
            P.network.stations_for_city(origin), P.network.stations_for_city(dest), P.today() + timedelta(days=days),
            prefs, None if user == "Guest" else user, budget or None)
    res = st.session_state.get("recs")
    if not res:
        return
    if not res["recommendations"]:
        st.info("No trains match. Try another date, route or a higher budget.")
        return
    st.dataframe(pd.DataFrame([{"Rank": r["rank"], "Train": f"{r['train_number']} {r['train_name']}", "Class": r["class"],
                                "Fare (₹)": r["fare"], "Seat": r["availability"],
                                "Chance of travelling": f"{r['travel_probability']:.0%}",
                                "Expected delay (min)": round(r["expected_delay_min"]), "Score": r["score"],
                                "Why": "; ".join(r["rationale"])} for r in res["recommendations"]]), hide_index=True)
    st.caption(f"Ranking policy {res['ranker_version']}, {res['candidates']} options considered"
               f"{', personalised to the passenger history' if res['personalised'] else ''}.")


def page_delay():
    st.header("Delays and disruptions")
    c1, c2 = st.columns([2, 1])
    with c1:
        train = train_picker("delay_train", default="92205")
    days = c2.slider("Days from today", 0, 60, 1, key="delay_days")
    if st.button("Forecast delay", type="primary"):
        st.session_state["delay"] = P.tools.delay_outlook(train, P.today() + timedelta(days=days))
    out = st.session_state.get("delay")
    if not out:
        return
    c = st.columns(4)
    c[0].metric("Expected arrival delay", f"{out['expected_delay_min']:.0f} min")
    c[1].metric("80% range", f"{out['delay_interval_min'][0]:.0f} to {out['delay_interval_min'][1]:.0f} min")
    c[2].metric("On time (within 15 min)", f"{out['prob_on_time_15min']:.0%}")
    c[3].metric("Risk level", out["risk_level"])
    for a in out["alerts"]:
        st.warning(a)
    st.dataframe(pd.DataFrame([{"Station": c["station_name"], "Around": c["time"][11:], "Congestion": c["level"],
                                "Index": c["congestion_index"]} for c in out["congestion"]]), hide_index=True)
    if out["alternatives"]:
        st.markdown("**Better-running alternatives on this route**")
        st.dataframe(pd.DataFrame(out["alternatives"]), hide_index=True)
    if out.get("explanation"):
        st.markdown("**What drives the forecast**")
        st.dataframe(pd.DataFrame([{"Factor": e["feature"], "Value": e["value"], "Typical": e["typical"],
                                    "Effect (min)": round(e["effect"], 1)} for e in out["explanation"]]), hide_index=True)


def page_fare():
    st.header("Fare forecast and the best time to book")
    trains = sorted(P.network.trains.values(), key=lambda t: (not t.dynamic_pricing, t.number))
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        train = train_picker("fare_train", trains=trains, default="92209")
    cls = c2.selectbox("Class", list(train.classes), key="fare_cls")
    days = c3.slider("Days until the journey", 1, 60, 45, key="fare_days")
    if st.button("Forecast fare", type="primary"):
        st.session_state["fare"] = P.fare.forecast(train, cls, P.today() + timedelta(days=days))
    f = st.session_state.get("fare")
    if not f:
        return
    if f["status"] != "ok":
        st.error(f["message"])
        return
    c = st.columns(3)
    c[0].metric("Fare today", f"₹{f['current']['fare']:,}")
    c[1].metric("Flexi multiplier", f"x{f['current']['flexi_multiplier']}")
    c[2].metric("Demand index", f["demand_surge"]["demand_index"])
    st.success(f["advice"])
    if f["demand_surge"]["surge"]:
        st.warning("Demand surge expected: " + (", ".join(f["demand_surge"]["drivers"]) or "high forecast occupancy"))
    st.line_chart(pd.DataFrame(f["trajectory"]).set_index("date")[["fare"]])
    if f["savings"]:
        st.markdown("**Ways to save**")
        st.dataframe(pd.DataFrame(f["savings"]), hide_index=True)
    st.markdown("**Fare breakdown**")
    st.dataframe(pd.DataFrame([f["explanation"]["fare_components"]]), hide_index=True)


def page_support():
    st.header("Support chat")
    st.caption("Ask in English, Hindi or Hinglish. Every reply is checked against the evidence it was built from "
               "before you see it.")
    history = st.session_state.setdefault("chat", [])
    examples = ["Status of PNR 4521873690", "मेरा पीएनआर 6203914785 का स्टेटस क्या है", "What are the cancellation charges?",
                "Where is my refund for PNR 2718281828?", "kal pune se delhi jana hai 1500 ke andar"]
    clicked = None
    for col, ex in zip(st.columns(len(examples)), examples):
        if col.button(ex, key=f"ex-{ex}"):
            clicked = ex
    msg = st.chat_input("Type your question") or clicked
    if msg:
        r = P.support.chat(msg, session_id="dashboard")
        history += [("user", msg, None), ("assistant", r["reply"], r)]
    for role, text, meta in history:
        with st.chat_message(role):
            st.write(text)
            if meta:
                st.caption(f"Detected {meta['language']}, intent {meta['intent']} ({meta['confidence']:.0%}), "
                           f"groundedness {meta['guard']['groundedness']:.2f}"
                           + (", handed to a human agent" if meta["escalated"] else ""))
    if history:
        rating = st.radio("How helpful was this?", [1, 2, 3, 4, 5], index=4, horizontal=True)
        if st.button("Send rating"):
            P.support.feedback("dashboard", rating)
            st.success("Thanks, your rating was recorded.")


def page_fraud():
    st.header("Fraud and bot detection")
    preset = st.selectbox("Start from a session profile", list(PRESETS))
    vals = dict(PRESETS[preset])
    numeric = [k for k in vals if k not in ("device_type", "region_segment")]
    cols = st.columns(4)
    for i, k in enumerate(numeric):
        vals[k] = cols[i % 4].number_input(k.replace("_", " "), value=float(vals[k]), key=f"{k}-{preset}")
    c1, c2 = st.columns(2)
    vals["device_type"] = c1.selectbox("device type", ["mobile", "desktop"],
                                       index=["mobile", "desktop"].index(vals["device_type"]), key=f"dev-{preset}")
    vals["region_segment"] = c2.selectbox("region", ["metro", "tier2", "tier3"],
                                          index=["metro", "tier2", "tier3"].index(vals["region_segment"]), key=f"reg-{preset}")
    if st.button("Score session", type="primary"):
        st.session_state["fraud"] = P.services["fraud_detector"].predict(vals)
    r = st.session_state.get("fraud")
    if not r:
        return
    c = st.columns(3)
    c[0].metric("Risk score", r["risk_score"])
    c[1].metric("Decision", r["decision"])
    c[2].metric("Suspected pattern", r["suspected_pattern"] or "none")
    st.write(r["action"])
    for reason in r["reasons"]:
        st.markdown(f"- {reason}")
    st.caption(f"Model {r['model']} {r['version']}. The low-and-slow ring is a blind spot of v1: inject the novel bot "
               "attack on the governance page, approve the retrained model, then score it again.")


def page_governance():
    st.header("Watchtower: autonomous AI governance")
    st.write("Seven governance agents watch every model and agent. Each cycle they find problems, trace root causes, "
             "fix what policy allows, verify each fix, and ask a person to approve the rest.")
    html = ['<table class="board"><tr><th>Service</th><th>Signal</th><th>Version or mode</th><th>Requests</th>'
            '<th>p95 ms</th><th>Live metric</th><th>Input drift</th></tr>']
    for r in P.supervisor.health_board():
        html.append(f"<tr><td>{r['service']}</td><td>{lamp(r['aspect'])}</td><td>{r['version']}</td><td>{r['requests']}</td>"
                    f"<td>{r['p95_ms'] if r['p95_ms'] is not None else '-'}</td><td>{r['metric']}</td><td>{r['drift']}</td></tr>")
    st.markdown("".join(html) + "</table>", unsafe_allow_html=True)
    st.caption("Signal aspects borrow from Indian Railways colour-light signals: green clear, double yellow attention, "
               "yellow caution, red danger.")
    st.subheader("Run the drill")
    c1, c2, c3 = st.columns([1, 2, 1])
    if c1.button("Send normal traffic"):
        with st.spinner("Sending normal traffic..."):
            P.simulator.normal()
        st.toast("Normal traffic sent")
    labels = {SCENARIOS[k]["label"]: k for k in SCENARIOS}
    chosen = c2.multiselect("Failures to inject", list(labels), default=list(labels)[:4])
    if c2.button("Inject selected failures"):
        with st.spinner("Injecting failures..."):
            for label in chosen:
                P.simulator.inject(labels[label])
        st.toast(f"Injected {len(chosen)} failure(s)")
    if c3.button("Run governance cycle", type="primary"):
        with st.spinner("Governance agents at work..."):
            st.session_state["cycle"] = P.supervisor.run_cycle(actor="dashboard")
    cycle = st.session_state.get("cycle")
    if cycle:
        st.markdown(f"**Cycle {cycle['cycle_id']}** ({cycle['duration_s']} s): {cycle['summary']}")
        for inc in cycle["incidents"]:
            with st.expander(f"{inc['priority']}  {inc['target']}  ({inc['status']})", expanded=inc["priority"] == "P1"):
                st.markdown(f"**Root cause** ({inc['rca']['confidence']:.0%} confidence): {inc['rca']['root_cause']}")
                st.dataframe(pd.DataFrame([{"Agent": f["agent"], "Finding": f["summary"], "Severity": f["severity"]}
                                           for f in inc["findings"]]), hide_index=True)
                for a in inc["actions"]:
                    st.markdown(f"- **{a['action']}**: {a['status']}"
                                + (f" (verified: {a['verification']})" if a.get("verification") else ""))
                st.caption("Hypotheses considered: " + " | ".join(
                    f"{h['cause']} ({h['confidence']:.0%})" for h in inc["rca"]["hypotheses"]))
                if inc["rca"]["blast_radius"]:
                    st.caption("Downstream services exposed: " + ", ".join(inc["rca"]["blast_radius"]))
    st.subheader("Waiting for a person")
    pending = [a for a in P.approvals if a["status"] == "pending"]
    if not pending:
        st.caption("No approvals pending.")
    for a in pending:
        st.markdown(f"**{a['approval_id']}**: {a['action']} {a['target']} {a.get('version') or ''}. {a['why']}.")
        ev = a.get("evidence") or {}
        if ev:
            st.caption(f"Hold-out {ev.get('metric')}: {ev.get('champion_score')} now vs {ev.get('challenger_score')} "
                       f"for the candidate; guard metrics {ev.get('guards')}")
        b1, b2 = st.columns(2)
        if b1.button("Approve", key=f"ok-{a['approval_id']}"):
            P.supervisor.decide(a["approval_id"], True, "dashboard-reviewer", "approved in dashboard")
            st.rerun()
        if b2.button("Reject", key=f"no-{a['approval_id']}"):
            P.supervisor.decide(a["approval_id"], False, "dashboard-reviewer", "rejected in dashboard")
            st.rerun()
    tabs = st.tabs(["Drift", "Fairness and GenAI quality", "Model registry", "Audit trail", "Report"])
    with tabs[0]:
        if not P.last_drift:
            st.caption("Run a governance cycle to compute drift.")
        for name, rep in P.last_drift.items():
            st.markdown(f"**{name}**: {rep['status']} (max PSI {rep['max_psi']:.2f}; calendar features are reported "
                        "but exempt from alarms)")
            st.bar_chart(pd.DataFrame(rep["features"]).set_index("feature")["psi"])
    with tabs[1]:
        for name, res in P.last_fairness.items():
            st.markdown(f"**{name}**")
            st.json(res, expanded=False)
        if P.genai_stats:
            st.markdown("**support_assistant**")
            st.json(P.genai_stats, expanded=False)
        if P.agent_stats:
            st.markdown("**travel_planner**")
            st.json(P.agent_stats, expanded=False)
    with tabs[2]:
        st.dataframe(pd.DataFrame(P.registry.table()), hide_index=True)
    with tabs[3]:
        if st.button("Verify hash chain"):
            v = P.audit.verify()
            if v["valid"]:
                st.success(f"{v['entries']} entries, chain intact.")
            else:
                st.error(f"Chain broken at entry {v['first_broken_seq']}.")
        st.dataframe(pd.DataFrame([{"#": e["seq"], "Time (UTC)": e["ts"], "Actor": e["actor"], "Action": e["action"],
                                    "Resource": e["resource"], "Severity": e["severity"]}
                                   for e in reversed(P.audit.tail(40))]), hide_index=True)
    with tabs[4]:
        report = governance_report(P)
        st.download_button("Download report (Markdown)", report, file_name="watchtower_governance_report.md")
        st.markdown(report)


{"Plan a journey": page_planner, "Confirmation chances": page_confirmation, "Train suggestions": page_recommend,
 "Delays and disruptions": page_delay, "Fare forecast": page_fare, "Support chat": page_support,
 "Fraud and bots": page_fraud, "Watchtower governance": page_governance}[page]()
