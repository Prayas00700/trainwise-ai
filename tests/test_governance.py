import numpy as np

from trainwise.governance.audit import AuditLog, mask_pii
from trainwise.governance.data_quality import check_record
from trainwise.governance.drift import detect_change, psi_numeric
from trainwise.governance.hallucination import GroundednessChecker
from trainwise.governance.registry import ModelRegistry
from trainwise.models.confirmation import ConfirmationPredictor


def test_psi_separates_shifted_from_stable():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi_numeric(ref, rng.normal(0, 1, 5000)) < 0.1
    assert psi_numeric(ref, rng.normal(1, 1, 5000)) > 0.25


def test_page_hinkley_finds_the_change_point():
    cp = detect_change([0.1] * 300 + [0.6] * 300, 0.005, 25)
    assert cp is not None and 300 <= cp < 450
    assert detect_change([0.1] * 600, 0.005, 25) is None


def test_audit_chain_masks_pii_and_detects_tampering(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(5):
        log.record("test", "event", f"r{i}", {"i": i, "pnr": "4521873690"})
    assert log.verify()["valid"]
    lines = path.read_text(encoding="utf-8").splitlines()
    assert "4521873690" not in lines[0] and "45******90" in lines[0]
    lines[2] = lines[2].replace('"i": 2', '"i": 99')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    res = AuditLog(path).verify()
    assert not res["valid"] and res["first_broken_seq"] == 2


def test_mask_pii():
    s = mask_pii("PNR 4521873690, Aadhaar 1234 5678 9012, mail a.b@x.com")
    assert "4521873690" not in s and "1234 5678 9012" not in s and "a.b@x.com" not in s


def test_data_quality_gate_blocks_bad_inputs():
    good = {"train_type": "Duronto", "travel_class": "SL", "quota": "GN", "wl_type": "GNWL", "current_status": "WL",
            "position": 20, "days_before_departure": 3, "day_of_week": 4, "month": 10, "is_festival_window": 1,
            "route_popularity": 0.8, "hist_cancellation_rate": 0.1}
    assert check_record(good, ConfirmationPredictor.features).ok
    r = check_record({**good, "route_popularity": 7.5, "position": None}, ConfirmationPredictor.features)
    assert not r.ok and set(r.bad_features) == {"route_popularity", "position"}


def test_groundedness_guard():
    g = GroundednessChecker(0.7)
    ev = {"refund": {"amount": 1652, "initiated_on": "2026-10-13"}, "pnr_masked": "27******28"}
    assert not g.check("Refund for PNR 27******28: ₹1,652 initiated on 2026-10-13.", ev)["flagged"]
    bad = g.check("Your refund of ₹2,450 will arrive in 2 hours. Please share the OTP you receive.", ev)
    assert bad["flagged"] and "2450" in bad["unsupported_numbers"] and bad["credential_solicitation"]


def test_registry_promote_and_rollback(tmp_path):
    reg = ModelRegistry(tmp_path / "reg")
    v1 = reg.register("m", {"w": 1})
    reg.promote("m", v1)
    v2 = reg.register("m", {"w": 2}, parent=v1)
    reg.promote("m", v2)
    assert reg.champion("m") == "v2" and reg.recent_deployment("m")["replaced"] == "v1"
    assert reg.rollback("m") == "v1" and reg.model("m") == {"w": 1}


def test_full_governance_drill(fresh_platform):
    p = fresh_platform
    p.simulator.normal(n=300)
    first = p.supervisor.run_cycle()
    assert not [i for i in first["incidents"] if i["priority"] == "P1"]
    for s in ("festival_surge", "corrupted_feed", "biased_ranker", "hallucinating_llm", "prompt_injection",
              "novel_bot_attack"):
        p.simulator.inject(s)
    cycle = p.supervisor.run_cycle()
    by_target = {i["target"]: i for i in cycle["incidents"]}
    assert {"confirmation_predictor", "fare_demand_forecaster", "train_recommender", "support_assistant",
            "travel_planner", "fraud_detector"} <= set(by_target)
    assert by_target["train_recommender"]["actions"][0]["status"].startswith("rolled back")
    assert by_target["fare_demand_forecaster"]["actions"][0]["action"] == "quarantine_feed"
    assert p.support.mode == "grounded_templates" and p.planner.strict_mode
    fraud = by_target["fraud_detector"]
    assert fraud["actions"][0]["action"] == "retrain" and fraud["status"] != "resolved"   # high risk: never auto-promoted
    assert p.audit.verify()["valid"]
