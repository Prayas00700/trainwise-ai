import os

import pytest
from fastapi.testclient import TestClient

PASSENGER, OPS, ADMIN = ({"X-API-Key": k} for k in ("demo-passenger-key", "demo-ops-key", "demo-admin-key"))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["TRAINWISE_ARTIFACT_DIR"] = str(tmp_path_factory.mktemp("api_artifacts"))
    os.environ["TRAINWISE_TODAY"] = "2026-10-15"
    from trainwise import platform as plat
    plat._PLATFORM = None
    from trainwise.api.main import app
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_auth_and_roles(client):
    assert client.post("/v1/support/chat", json={"message": "hello"}).status_code == 401
    assert client.get("/v1/governance/health", headers=PASSENGER).status_code == 403
    assert client.get("/v1/governance/health", headers=OPS).status_code == 200


def test_plan_and_book_flow(client):
    plan = client.post("/v1/planner/plan", json={"query": "Pune to Delhi tomorrow under 1500"}, headers=PASSENGER).json()
    assert plan["status"] == "plan_ready"
    body = {"plan_id": plan["plan_id"], "option_id": plan["recommendation"]["option_id"]}
    assert client.post("/v1/planner/book", json=body, headers=PASSENGER).json()["status"] == "denied"
    booked = client.post("/v1/planner/book", json={**body, "approval_token": plan["approval"]["token"]},
                         headers=PASSENGER).json()
    assert booked["status"] == "booked"


def test_prediction_endpoints(client):
    assert client.post("/v1/confirmation/predict", json={"status": "GNWL 12"}, headers=PASSENGER).json()["status"] == "ok"
    assert "expected_delay_min" in client.post("/v1/delay/predict", json={"train_number": "92205"}, headers=PASSENGER).json()
    fare = client.post("/v1/fare/forecast", json={"train_number": "92209", "travel_class": "3A"}, headers=PASSENGER).json()
    assert fare["status"] == "ok"
    assert client.post("/v1/fraud/score", json={}, headers=OPS).json()["decision"] == "allow"


def test_governance_changes_need_admin(client):
    assert client.post("/v1/governance/simulate", json={"scenario": "normal"}, headers=OPS).status_code == 403
    assert client.post("/v1/governance/simulate", json={"scenario": "corrupted_feed"}, headers=ADMIN).status_code == 200
    cycle = client.post("/v1/governance/cycle", headers=OPS).json()
    assert any(i["target"] == "fare_demand_forecaster" for i in cycle["incidents"])
    assert client.get("/v1/governance/audit/verify", headers=OPS).json()["valid"]
    assert "trainwise_requests_total" in client.get("/metrics", headers=OPS).text


def test_website_and_lookup_endpoints(client):
    home = client.get("/")
    assert home.status_code == 200 and "TrainWise" in home.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
    data = client.get("/v1/trains", headers=PASSENGER).json()
    assert data["trains"] and "Pune" in data["cities"]
    assert len(client.get("/v1/governance/scenarios", headers=OPS).json()) == 8
