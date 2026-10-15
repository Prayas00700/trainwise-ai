"""TrainWise AI REST API.

Run:   uvicorn trainwise.api.main:app --reload        Interactive docs: http://127.0.0.1:8000/docs
Auth:  send header X-API-Key (demo keys: demo-passenger-key, demo-ops-key, demo-admin-key)
"""
from __future__ import annotations

import json
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from trainwise.api.security import rate_limit, require_role
from trainwise.data.network import STATIONS
from trainwise.governance.reports import governance_report
from trainwise.platform import get_platform
from trainwise.simulation import SCENARIOS

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"

app = FastAPI(title="TrainWise AI", version="1.0.0",
              description="Intelligent railway travel services governed by Watchtower, an autonomous AI-governance "
                          "layer. All data is synthetic.")
_gov_lock = threading.Lock()
LIMITED = [Depends(rate_limit)]
PASSENGER, OPS, ADMIN = Depends(require_role("passenger")), Depends(require_role("ops")), Depends(require_role("admin"))


def P():
    return get_platform()


def out(obj):
    """Make any result JSON-safe (numpy scalars, dates)."""
    return json.loads(json.dumps(obj, default=str))


def _train(number: str):
    try:
        return P().network.get(number)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown train {number}")


def _journey(d: Optional[date]) -> date:
    return d or P().today() + timedelta(days=1)


class PlanIn(BaseModel):
    query: str = Field(..., examples=["I want to travel from Pune to Delhi tomorrow under ₹1500 and reach before 10:00 AM."])
    user_id: Optional[str] = "U001"


class BookIn(BaseModel):
    plan_id: str
    option_id: str
    approval_token: Optional[str] = None
    session: Optional[dict] = None


class ConfirmationIn(BaseModel):
    train_number: str = "92201"
    travel_class: str = "SL"
    journey_date: Optional[date] = None
    status: str = Field("GNWL 24", description="Current status, e.g. 'GNWL 24' or 'RAC 6'")
    quota: str = "GN"


class RecommendIn(BaseModel):
    origin: str = "Pune"
    destination: str = "Delhi"
    journey_date: Optional[date] = None
    preferences: List[str] = []
    user_id: Optional[str] = None
    budget: Optional[int] = None
    classes: List[str] = []


class TrainDateIn(BaseModel):
    train_number: str = "92201"
    journey_date: Optional[date] = None


class FareIn(TrainDateIn):
    travel_class: str = "SL"


class ChatIn(BaseModel):
    message: str
    session_id: str = "api"
    user_id: Optional[str] = None


class FeedbackIn(BaseModel):
    session_id: str
    rating: int = Field(..., ge=1, le=5)


class SessionIn(BaseModel):
    requests_per_min: float = 6
    avg_inter_click_ms: float = 2200
    form_fill_seconds: float = 45
    captcha_solve_seconds: float = 7
    accounts_on_device: int = 1
    ips_last_24h: int = 1
    bookings_last_24h: int = 0
    tatkal_window_hit: int = 0
    payment_failures_24h: int = 0
    passengers_per_booking: int = 2
    account_age_days: float = 700
    headless_signal: int = 0
    device_type: str = "mobile"
    region_segment: str = "metro"


class DecisionIn(BaseModel):
    approve: bool
    reviewer: str = "api-admin"
    note: str = ""


class SimulateIn(BaseModel):
    scenario: str = Field("normal", description="'normal' or one of: " + ", ".join(SCENARIOS))


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/", include_in_schema=False)
def website():
    """The passenger website and Watchtower console (web/index.html)."""
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "champions": {n: P().registry.champion(n) for n in P().services}}


@app.get("/v1/trains", dependencies=LIMITED)
def trains(role: str = PASSENGER):
    p = P()
    cities = p.network.cities()
    return {"today": p.today().isoformat(), "cities": cities,
            "stations": {c: p.network.stations_for_city(c) for c in cities},
            "station_names": {code: st.name for code, st in STATIONS.items()},
            "trains": [{**t.summary(), "dynamic_pricing": t.dynamic_pricing}
                       for t in sorted(p.network.trains.values(), key=lambda t: t.number)]}


# ---------------------------------------------------------------- passenger services
@app.post("/v1/planner/plan", dependencies=LIMITED)
def plan(body: PlanIn, role: str = PASSENGER):
    return out(P().planner.plan(body.query, body.user_id))


@app.post("/v1/planner/book", dependencies=LIMITED)
def book(body: BookIn, role: str = PASSENGER):
    return out(P().planner.book(body.plan_id, body.option_id, body.approval_token, body.session))


@app.post("/v1/confirmation/predict", dependencies=LIMITED)
def confirmation(body: ConfirmationIn, role: str = PASSENGER):
    p, train, journey = P(), _train(body.train_number), _journey(body.journey_date)
    if body.travel_class not in train.classes:
        raise HTTPException(status_code=422, detail=f"Train {train.number} has no {body.travel_class} class")
    try:
        position = int(body.status.split()[-1])
    except ValueError:
        raise HTTPException(status_code=422, detail="status must look like 'GNWL 24' or 'RAC 6'")
    snap = {"kind": "RAC" if body.status.upper().startswith("RAC") else "WL", "position": position,
            "days_before": max((journey - p.today()).days, 0)}
    return out(p.tools.confirmation_for(train, body.travel_class, journey, snap, quota=body.quota, explain=True))


@app.post("/v1/recommendations", dependencies=LIMITED)
def recommendations(body: RecommendIn, role: str = PASSENGER):
    p = P()
    origins, dests = p.network.resolve(body.origin), p.network.resolve(body.destination)
    if not origins or not dests:
        raise HTTPException(status_code=422, detail="Unknown origin or destination")
    return out(p.recommender.recommend(origins, dests, _journey(body.journey_date), body.preferences, body.user_id,
                                       body.budget, body.classes or None))


@app.post("/v1/delay/predict", dependencies=LIMITED)
def delay(body: TrainDateIn, role: str = PASSENGER):
    return out(P().tools.delay_outlook(_train(body.train_number), _journey(body.journey_date)))


@app.post("/v1/fare/forecast", dependencies=LIMITED)
def fare(body: FareIn, role: str = PASSENGER):
    train = _train(body.train_number)
    if body.travel_class not in train.classes:
        raise HTTPException(status_code=422, detail=f"Train {train.number} has no {body.travel_class} class")
    return out(P().fare.forecast(train, body.travel_class, _journey(body.journey_date)))


@app.post("/v1/support/chat", dependencies=LIMITED)
def chat(body: ChatIn, role: str = PASSENGER):
    return out(P().support.chat(body.message, body.session_id, body.user_id))


@app.post("/v1/support/feedback", dependencies=LIMITED)
def feedback(body: FeedbackIn, role: str = PASSENGER):
    P().support.feedback(body.session_id, body.rating)
    return {"status": "recorded"}


@app.post("/v1/fraud/score", dependencies=LIMITED)
def fraud(body: SessionIn, role: str = OPS):
    return out(P().services["fraud_detector"].predict(body.model_dump(), explain=True))


# ---------------------------------------------------------------- Watchtower governance
@app.get("/v1/governance/health")
def governance_health(role: str = OPS):
    return out(P().supervisor.health_board())


@app.get("/v1/governance/scenarios")
def governance_scenarios(role: str = OPS):
    return [{"id": k, **v} for k, v in SCENARIOS.items()]


@app.get("/v1/governance/models")
def governance_models(role: str = OPS):
    return out(P().registry.table())


@app.post("/v1/governance/cycle")
def governance_cycle(role: str = OPS):
    with _gov_lock:
        return out(P().supervisor.run_cycle(actor=f"api:{role}"))


@app.get("/v1/governance/approvals")
def governance_approvals(role: str = OPS):
    return out([{k: v for k, v in a.items() if k != "evidence"} for a in P().approvals])


@app.post("/v1/governance/approvals/{approval_id}")
def governance_decide(approval_id: str, body: DecisionIn, role: str = ADMIN):
    try:
        return out(P().supervisor.decide(approval_id, body.approve, body.reviewer, body.note))
    except StopIteration:
        raise HTTPException(status_code=404, detail="Unknown approval id")


@app.post("/v1/governance/simulate")
def governance_simulate(body: SimulateIn, role: str = ADMIN):
    p = P()
    if body.scenario == "normal":
        return p.simulator.normal()
    if body.scenario not in SCENARIOS:
        raise HTTPException(status_code=422, detail=f"Unknown scenario; choose from {', '.join(SCENARIOS)}")
    return p.simulator.inject(body.scenario)


@app.get("/v1/governance/audit")
def governance_audit(n: int = 50, role: str = OPS):
    return out(P().audit.tail(n))


@app.get("/v1/governance/audit/verify")
def governance_audit_verify(role: str = OPS):
    return P().audit.verify()


@app.get("/v1/governance/report", response_class=PlainTextResponse)
def governance_report_md(role: str = OPS):
    return governance_report(P())


@app.get("/metrics", response_class=PlainTextResponse)
def metrics(role: str = OPS):
    """Prometheus text format; scrape with `Authorization: Bearer <ops key>` (see ops/prometheus.yml)."""
    p = P()
    aspects = {"green": 0, "double_yellow": 1, "yellow": 2, "red": 3}
    lines = ["# HELP trainwise_requests_total Requests observed in the current monitoring window",
             "# TYPE trainwise_requests_total gauge",
             "# HELP trainwise_signal_level 0 clear, 1 attention, 2 caution, 3 danger",
             "# TYPE trainwise_signal_level gauge"]
    for r in p.supervisor.health_board():
        s = r["service"]
        lines.append(f'trainwise_requests_total{{service="{s}"}} {r["requests"]}')
        lines.append(f'trainwise_signal_level{{service="{s}"}} {aspects[r["aspect"]]}')
        lines.append(f'trainwise_replicas{{service="{s}"}} {r["replicas"]}')
        if r["p95_ms"] is not None:
            lines.append(f'trainwise_latency_p95_ms{{service="{s}"}} {r["p95_ms"]}')
        if r["blocked_rate"] is not None:
            lines.append(f'trainwise_blocked_rate{{service="{s}"}} {r["blocked_rate"]}')
    for name, lm in p.live_metrics.items():
        lines.append(f'trainwise_live_metric{{service="{name}",metric="{lm["metric"]}"}} {lm["live"]}')
        lines.append(f'trainwise_validated_metric{{service="{name}",metric="{lm["metric"]}"}} {lm["baseline"]}')
    lines.append(f"trainwise_pending_approvals {sum(a['status'] == 'pending' for a in p.approvals)}")
    return "\n".join(lines) + "\n"
