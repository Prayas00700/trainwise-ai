import sys
from pathlib import Path

from trainwise.data.generators import make_delay_data
from trainwise.governance.monitoring import Monitor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def test_conformal_interval_coverage_is_near_target(platform):
    model = platform.registry.model("delay_predictor")
    coverage = model.evaluate(make_delay_data(1500, 123))["interval_coverage_80"]
    assert 0.72 <= coverage <= 0.95
    assert hasattr(model, "conformal_margin_")


def test_fraud_thresholds_come_from_policy_costs(platform):
    model = platform.registry.model("fraud_detector")
    assert 0.15 <= model.challenge_at <= model.block_at <= 0.95
    assert set(model.threshold_report_["costs"]) == {"false_challenge", "false_block", "missed_fraud"}


def test_telemetry_checkpoint_roundtrip(tmp_path):
    m = Monitor()
    rid = m.log_inference("x", "v1", 1.5, "ok", features={"a": 1}, prediction="CNF")
    m.log_outcome(rid, "CNF")
    m.log("agent_traces", {"steps": 3, "violations": []})
    assert m.save(tmp_path / "t.sqlite")["inferences"] == 1
    restored = Monitor()
    info = restored.load(tmp_path / "t.sqlite")
    assert info["inferences"] == 1 and restored.outcomes[rid] == "CNF"
    assert restored.stream("agent_traces")[0]["steps"] == 3 and restored.health("x")["requests"] == 1


def test_canary_runs_before_autonomous_promotion(fresh_platform):
    p = fresh_platform
    p.simulator.normal(n=300)
    p.simulator.inject("winter_fog")
    cycle = p.supervisor.run_cycle()
    incident = next(i for i in cycle["incidents"] if i["target"] == "delay_predictor")
    retrain = next(a for a in incident["actions"] if a["action"] == "retrain")
    assert "canary" in retrain["verification"] and retrain["detail"]["canary"]["rows"] == 200
    assert p.telemetry_path.exists()


def test_model_cards_cover_every_registered_model(platform, tmp_path):
    from model_cards import write_model_cards
    files = write_model_cards(platform, tmp_path)
    assert {f.stem for f in files} == set(platform.registry.index)
    text = (tmp_path / "fraud_detector.md").read_text(encoding="utf-8")
    assert "## Limitations" in text and "Risk tier | high" in text
