"""Seat-inventory and operating-conditions simulators.

These stand in for the PRS availability API, the NTES running-status feed, a weather feed and
the engineering-blocks schedule. All values are synthetic but deterministic per (train, date),
so the same query returns the same answer within a day.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np

from trainwise.data.network import CLASS_INFO, STATIONS, RailNetwork, Train, stable_seed

ARP_DAYS = 60  # advance reservation period used by the simulator (days)

# Approximate festive / vacation demand windows (month, day) -> demand multiplier. Configurable.
FESTIVE_WINDOWS = [
    ((10, 1), (11, 20), 1.35),   # Navratri / Durga Puja / Diwali / Chhath season
    ((3, 1), (3, 25), 1.20),     # Holi
    ((5, 1), (6, 25), 1.25),     # summer vacations
    ((12, 20), (12, 31), 1.20),  # year-end holidays
    ((1, 1), (1, 5), 1.15),
]
DOW_FACTOR = {0: 1.00, 1: 0.93, 2: 0.92, 3: 0.95, 4: 1.12, 5: 1.03, 6: 1.10}
CLASS_DEMAND = {"SL": 1.06, "3A": 1.08, "2A": 0.96, "1A": 0.85, "CC": 1.00, "EC": 0.90, "2S": 1.02}


def festival_multiplier(d: date) -> float:
    for (sm, sd), (em, ed), mult in FESTIVE_WINDOWS:
        if (sm, sd) <= (d.month, d.day) <= (em, ed):
            return mult
    return 1.0


def demand_index(d: date) -> float:
    return round(festival_multiplier(d) * DOW_FACTOR[d.weekday()], 3)


def sold_fraction_curve(days_before: float, popularity: float, demand: float) -> float:
    """Share of capacity sold `days_before` departure. >1 means the excess is on RAC/waitlist."""
    final = 0.35 + 0.55 * popularity * demand
    k = 1.2 + 2.6 * popularity                       # popular trains sell out earlier
    progress = min(max((ARP_DAYS - days_before) / ARP_DAYS, 0.0), 1.0)
    return final * (1 - math.exp(-k * progress)) / (1 - math.exp(-k))


def seasonal_fog(month: int) -> float:
    return {12: 0.80, 1: 0.85, 11: 0.35, 2: 0.40}.get(month, 0.03)


def seasonal_rain(month: int) -> float:
    return {7: 0.70, 8: 0.70, 6: 0.40, 9: 0.40}.get(month, 0.05)


class InventorySimulator:
    """Current seat availability per (train, class, journey date) as seen on `as_of`."""

    def __init__(self, network: RailNetwork):
        self.network = network

    def sold_fraction(self, train: Train, cls: str, journey: date, as_of: date) -> float:
        days_before = max((journey - as_of).days, 0)
        pop = min(train.popularity * CLASS_DEMAND.get(cls, 1.0), 1.0)
        base = sold_fraction_curve(days_before, pop, demand_index(journey))
        noise = np.random.default_rng(stable_seed(train.number, cls, journey.toordinal())).normal(0, 0.04)
        return max(0.0, base + noise * (1 - days_before / (ARP_DAYS + 1)))

    def snapshot(self, train: Train, cls: str, journey: date, as_of: date) -> dict:
        sold = self.sold_fraction(train, cls, journey, as_of)
        info = CLASS_INFO[cls]
        cap = info["capacity"]
        rac_slots = int(cap * 0.05) if info["rac"] else 0
        if sold < 1.0:
            kind, position = "AVL", max(1, int(cap * (1 - sold)))
            status = f"AVL {position}"
        else:
            overflow = int((sold - 1.0) * cap) + 1
            if overflow <= rac_slots:
                kind, position, status = "RAC", overflow, f"RAC {overflow}"
            else:
                kind, position = "WL", overflow - rac_slots
                status = f"GNWL {position}"
        return {"train": train.number, "class": cls, "journey_date": journey.isoformat(),
                "days_before": max((journey - as_of).days, 0), "sold_fraction": round(sold, 3),
                "status": status, "kind": kind, "position": position, "capacity": cap,
                "rac_slots": rac_slots, "demand_index": demand_index(journey)}


class ConditionsFeed:
    """Synthetic weather, congestion, maintenance and incoming-rake delay for a trip."""

    def __init__(self, network: RailNetwork):
        self.network = network

    def for_trip(self, train: Train, journey: date) -> dict:
        rng = np.random.default_rng(stable_seed("cond", train.number, journey.toordinal()))
        fog = seasonal_fog(journey.month) * (1.0 if train.northern else 0.15) * rng.uniform(0.6, 1.1)
        rain = seasonal_rain(journey.month) * rng.uniform(0.5, 1.1)
        congestion = float(np.clip(0.25 + 0.35 * (demand_index(journey) - 0.9) + rng.normal(0, 0.08), 0, 1))
        return {
            "train_type": train.ttype, "distance_km": train.distance_km, "num_stops": train.stops,
            "departure_hour": train.departure.hour, "month": journey.month, "day_of_week": journey.weekday(),
            "northern_route": int(train.northern), "fog_index": round(min(fog, 1.0), 3),
            "rain_index": round(min(rain, 1.0), 3), "route_congestion": round(congestion, 3),
            "upstream_delay_min": round(float(rng.exponential(6) + 40 * fog * rng.random()), 1),
            "maintenance_block": int(rng.random() < 0.08),
        }

    def station_congestion(self, station: str, when: datetime, predicted_delays: dict | None = None) -> dict:
        """Heuristic congestion index (0-100) for a station around a given time.

        Combines a diurnal utilisation profile, festive demand, scheduled movements of
        synthetic trains in a +/-60 min window, and model-predicted delays (knock-on effect).
        """
        st = STATIONS[station]
        hour = when.hour + when.minute / 60
        diurnal = 0.5 + 0.5 * math.sin((hour - 6) / 24 * 2 * math.pi) ** 2
        movements = 0
        delay_pressure = 0.0
        for t in self.network.trains.values():
            for d in (when.date() - timedelta(days=1), when.date()):
                if not t.runs_on(d):
                    continue
                for stn, ts in ((t.origin, t.departure_dt(d)), (t.destination, t.arrival_dt(d))):
                    if stn == station and abs((ts - when).total_seconds()) <= 3600:
                        movements += 1
                        if predicted_delays and t.number in predicted_delays:
                            delay_pressure += predicted_delays[t.number] / 120
        util = 0.30 + 0.25 * diurnal + 0.20 * (demand_index(when.date()) - 0.9) + movements / max(st.capacity, 1) * 0.6
        index = float(np.clip(100 * (util + 0.1 * delay_pressure), 0, 100))
        level = "Low" if index < 45 else "Moderate" if index < 65 else "High" if index < 80 else "Severe"
        return {"station": station, "station_name": st.name, "time": when.isoformat(timespec="minutes"),
                "congestion_index": round(index, 1), "level": level, "scheduled_movements_2h": movements}
