"""Tools the agents call: train search, availability, fares, confirmation and delay predictions.

Each tool wraps a governed service, so agent traffic is validated, monitored and audited exactly
like direct API traffic.
"""
from __future__ import annotations

from datetime import date, timedelta

from trainwise.data.inventory import festival_multiplier
from trainwise.data.network import CLASS_INFO, STATIONS

_FALLBACK_DELAY = {"expected_delay_min": 30.0, "quantiles": {"p10": 0.0, "p50": 30.0, "p90": 120.0},
                   "prob_on_time_15min": 0.4, "risk_level": "unknown", "delay_interval_min": [0.0, 120.0]}


class RailTools:
    def __init__(self, platform):
        self.p = platform

    def search_trains(self, origins, destinations, journey: date) -> list:
        return self.p.network.trains_between(origins, destinations, journey)

    def delay_for(self, train, journey: date, explain: bool = False) -> dict:
        res = self.p.services["delay_predictor"].predict(self.p.conditions.for_trip(train, journey), explain=explain)
        return res if res["status"] == "ok" else {**_FALLBACK_DELAY, "status": "fallback", "issues": res.get("issues")}

    def confirmation_for(self, train, cls: str, journey: date, snap: dict, quota: str = "GN",
                         explain: bool = False) -> dict:
        if snap["kind"] == "AVL":
            return {"status": "certain", "confirmation_probability": 1.0, "travel_probability": 1.0,
                    "predicted_final_status": "CNF", "version": None}
        rec = {"train_type": train.ttype, "travel_class": cls, "quota": quota,
               "wl_type": "TQWL" if quota == "TQ" else "GNWL", "current_status": snap["kind"],
               "position": snap["position"], "days_before_departure": snap["days_before"],
               "day_of_week": journey.weekday(), "month": journey.month,
               "is_festival_window": int(festival_multiplier(journey) > 1.0), "route_popularity": train.popularity,
               "hist_cancellation_rate": train.hist_cancellation_rate}
        res = self.p.services["confirmation_predictor"].predict(rec, explain=explain)
        if res["status"] != "ok":
            return {"status": "fallback", "confirmation_probability": 0.3, "travel_probability": 0.5,
                    "predicted_final_status": "unknown", "version": None, "issues": res.get("issues")}
        return res

    def build_options(self, trains: list, journey: date, classes=None, as_of: date | None = None) -> list:
        as_of = as_of or self.p.today()
        options = []
        for t in trains:
            delay = self.delay_for(t, journey)
            for cls in t.classes:
                if classes and cls not in classes:
                    continue
                snap = self.p.inventory.snapshot(t, cls, journey, as_of)
                quote = self.p.fare.quote(t, cls, journey, as_of)
                conf = self.confirmation_for(t, cls, journey, snap)
                options.append({
                    "option_id": f"{t.number}-{cls}", "train_number": t.number, "train_name": t.name,
                    "train_type": t.ttype, "class": cls, "class_label": CLASS_INFO[cls]["label"],
                    "origin": t.origin, "destination": t.destination,
                    "departure": t.departure_dt(journey).isoformat(timespec="minutes"),
                    "arrival": t.arrival_dt(journey).isoformat(timespec="minutes"),
                    "departure_hour": t.departure.hour, "duration_min": t.duration_min, "fare": quote["fare"],
                    "flexi_multiplier": quote["flexi_multiplier"], "dynamic_pricing": t.dynamic_pricing,
                    "availability": snap["status"], "availability_kind": snap["kind"],
                    "confirmation_probability": conf["confirmation_probability"],
                    "travel_probability": conf["travel_probability"],
                    "predicted_final_status": conf.get("predicted_final_status"),
                    "expected_delay_min": delay["expected_delay_min"], "delay_quantiles": delay["quantiles"],
                    "prob_on_time_15min": delay["prob_on_time_15min"], "delay_risk": delay["risk_level"],
                    "model_versions": {"confirmation": conf.get("version"), "delay": delay.get("version")}})
        return options

    def delay_outlook(self, train, journey: date) -> dict:
        """Delay forecast + station congestion + proactive alerts + better-running alternatives."""
        pred = self.delay_for(train, journey, explain=True)
        dep = train.departure_dt(journey)
        arr = train.arrival_dt(journey) + timedelta(minutes=pred["expected_delay_min"])
        congestion = [self.p.conditions.station_congestion(train.origin, dep),
                      self.p.conditions.station_congestion(train.destination, arr,
                                                           {train.number: pred["expected_delay_min"]})]
        alerts = []
        if pred.get("prob_delay_over_60min", 0) >= 0.3:
            alerts.append(f"{round(100 * pred['prob_delay_over_60min'])}% risk of arriving more than 60 minutes late.")
        if pred["risk_level"] in ("high", "severe"):
            alerts.append(f"Delay risk is {pred['risk_level']}: expected {pred['expected_delay_min']:.0f} minutes.")
        for c in congestion:
            if c["level"] in ("High", "Severe"):
                alerts.append(f"{c['level']} congestion expected at {c['station_name']} around {c['time'][11:16]}.")
        alternatives = []
        if alerts:
            origins = self.p.network.stations_for_city(STATIONS[train.origin].city)
            dests = self.p.network.stations_for_city(STATIONS[train.destination].city)
            for t in self.p.network.trains_between(origins, dests, journey):
                if t.number == train.number:
                    continue
                d = self.delay_for(t, journey)
                if d["expected_delay_min"] < pred["expected_delay_min"]:
                    alternatives.append({"train": f"{t.number} {t.name}", "departure": t.departure.strftime("%H:%M"),
                                         "expected_delay_min": d["expected_delay_min"],
                                         "prob_on_time_15min": d["prob_on_time_15min"]})
            alternatives = sorted(alternatives, key=lambda a: a["expected_delay_min"])[:3]
        return {**pred, "train": train.summary(), "journey_date": journey.isoformat(),
                "scheduled_arrival": train.arrival_dt(journey).isoformat(timespec="minutes"),
                "expected_arrival": arr.isoformat(timespec="minutes"), "congestion": congestion, "alerts": alerts,
                "alternatives": alternatives}
