from datetime import timedelta

import pytest

from trainwise.config import model_policy
from trainwise.models.recommender import TrainRecommender

ML = ["confirmation_predictor", "delay_predictor", "fare_demand_forecaster", "fraud_detector", "intent_classifier"]
BOT = dict(requests_per_min=120, avg_inter_click_ms=120, form_fill_seconds=3, captcha_solve_seconds=1.5,
           accounts_on_device=2, ips_last_24h=3, bookings_last_24h=12, tatkal_window_hit=1, payment_failures_24h=0,
           passengers_per_booking=6, account_age_days=20, headless_signal=1, device_type="desktop", region_segment="metro")
PASSENGER = dict(requests_per_min=6, avg_inter_click_ms=2200, form_fill_seconds=45, captcha_solve_seconds=7,
                 accounts_on_device=1, ips_last_24h=1, bookings_last_24h=0, tatkal_window_hit=0, payment_failures_24h=0,
                 passengers_per_booking=2, account_age_days=700, headless_signal=0, device_type="mobile",
                 region_segment="tier3")


@pytest.mark.parametrize("name", ML)
def test_bootstrapped_models_meet_policy_floor(platform, name):
    pol = model_policy(name)
    value = platform.registry.info(name)["metrics"][pol["primary_metric"]]
    assert value >= pol["floor"] if pol["higher_is_better"] else value <= pol["floor"]


def test_confirmation_prediction_is_explained(platform):
    train, journey = platform.network.get("92201"), platform.today() + timedelta(days=3)
    r = platform.tools.confirmation_for(train, "SL", journey, {"kind": "WL", "position": 30, "days_before": 3}, explain=True)
    assert r["status"] == "ok"
    assert abs(sum(r["final_status_probabilities"].values()) - 1) < 1e-6
    assert r["explanation"] and {"feature", "value", "typical", "effect"} <= set(r["explanation"][0])


def test_rac_ticket_never_falls_back_to_waitlist(platform):
    train, journey = platform.network.get("92201"), platform.today() + timedelta(days=2)
    r = platform.tools.confirmation_for(train, "SL", journey, {"kind": "RAC", "position": 5, "days_before": 2})
    assert r["final_status_probabilities"]["WL"] == 0 and r["travel_probability"] == pytest.approx(1.0)


def test_delay_interval_is_ordered(platform):
    r = platform.tools.delay_for(platform.network.get("92205"), platform.today() + timedelta(days=1))
    lo, hi = r["delay_interval_min"]
    assert lo <= r["expected_delay_min"] <= hi and 0 <= r["prob_on_time_15min"] <= 1


def test_fare_forecast_is_monotonic_and_actionable(platform):
    f = platform.fare.forecast(platform.network.get("92209"), "3A", platform.today() + timedelta(days=45))
    sold = [t["sold_fraction"] for t in f["trajectory"]]
    assert f["status"] == "ok" and sold == sorted(sold) and f["advice"] and f["current"]["fare"] > 0


def test_fraud_detector_blocks_bot_and_allows_passenger(platform):
    svc = platform.services["fraud_detector"]
    assert svc.predict(BOT)["decision"] == "block"
    assert svc.predict(PASSENGER)["decision"] == "allow"


def test_recommender_preferences_and_feedback_learning():
    rec = TrainRecommender()
    w = rec.weights_for(["cheapest"])
    assert w["cost"] == max(w.values())
    chosen = {"cost": 1.0, "duration": 0.2, "availability": 0.9, "punctuality": 0.5, "personal": 0.5}
    skipped = {"cost": 0.2, "duration": 1.0, "availability": 0.9, "punctuality": 0.5, "personal": 0.5}
    new = rec.learn_from_feedback([{"chosen": chosen, "skipped": [skipped]}] * 10)
    assert abs(sum(new.weights.values()) - 1) < 1e-3 and new.weights["cost"] > new.weights["duration"]
