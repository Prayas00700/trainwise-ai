"""Synthetic Indian Railways network: stations, trains, timetable and tariff.

Station codes and city names are real. Every train, timetable, capacity and fare in this
module is SYNTHETIC: it exists so the platform can run end-to-end without access to
PRS (reservations) or NTES (running status) data, which are not publicly available.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np

from trainwise.config import SEED


@dataclass(frozen=True)
class Station:
    code: str
    name: str
    city: str
    lat: float
    lon: float
    northern: bool   # Indo-Gangetic plain: exposed to winter fog
    capacity: int    # synthetic train-handling capacity per hour


STATIONS: dict[str, Station] = {s.code: s for s in [
    Station("NDLS", "New Delhi", "Delhi", 28.643, 77.219, True, 18),
    Station("NZM", "Hazrat Nizamuddin", "Delhi", 28.588, 77.254, True, 9),
    Station("PUNE", "Pune Junction", "Pune", 18.529, 73.874, False, 8),
    Station("CSMT", "Mumbai CSMT", "Mumbai", 18.940, 72.835, False, 16),
    Station("BCT", "Mumbai Central", "Mumbai", 18.969, 72.819, False, 10),
    Station("HWH", "Howrah Junction", "Kolkata", 22.583, 88.342, False, 20),
    Station("SBC", "KSR Bengaluru", "Bengaluru", 12.978, 77.569, False, 10),
    Station("MAS", "Chennai Central", "Chennai", 13.083, 80.275, False, 14),
    Station("CTC", "Cuttack", "Cuttack", 20.464, 85.880, False, 6),
    Station("BBS", "Bhubaneswar", "Bhubaneswar", 20.266, 85.843, False, 8),
    Station("AGC", "Agra Cantt", "Agra", 27.158, 77.990, True, 7),
    Station("LKO", "Lucknow Charbagh", "Lucknow", 26.832, 80.923, True, 10),
    Station("PNBE", "Patna Junction", "Patna", 25.603, 85.137, True, 10),
    Station("ADI", "Ahmedabad Junction", "Ahmedabad", 23.026, 72.601, False, 11),
    Station("JP", "Jaipur Junction", "Jaipur", 26.919, 75.788, True, 8),
    Station("BPL", "Bhopal Junction", "Bhopal", 23.268, 77.412, False, 7),
    Station("SC", "Secunderabad", "Hyderabad", 17.434, 78.501, False, 10),
]}

# City names (English, common old names, Hindi) -> station codes serving that city.
CITY_ALIASES: dict[str, list[str]] = {
    "delhi": ["NDLS", "NZM"], "new delhi": ["NDLS"], "nizamuddin": ["NZM"], "dilli": ["NDLS", "NZM"],
    "दिल्ली": ["NDLS", "NZM"],
    "pune": ["PUNE"], "poona": ["PUNE"], "पुणे": ["PUNE"],
    "mumbai": ["CSMT", "BCT"], "bombay": ["CSMT", "BCT"], "मुंबई": ["CSMT", "BCT"],
    "kolkata": ["HWH"], "calcutta": ["HWH"], "howrah": ["HWH"], "कोलकाता": ["HWH"],
    "bengaluru": ["SBC"], "bangalore": ["SBC"], "बेंगलुरु": ["SBC"],
    "chennai": ["MAS"], "madras": ["MAS"], "चेन्नई": ["MAS"],
    "cuttack": ["CTC"], "कटक": ["CTC"],
    "bhubaneswar": ["BBS"], "भुवनेश्वर": ["BBS"],
    "agra": ["AGC"], "आगरा": ["AGC"],
    "lucknow": ["LKO"], "लखनऊ": ["LKO"],
    "patna": ["PNBE"], "पटना": ["PNBE"],
    "ahmedabad": ["ADI"], "अहमदाबाद": ["ADI"],
    "jaipur": ["JP"], "जयपुर": ["JP"],
    "bhopal": ["BPL"], "भोपाल": ["BPL"],
    "hyderabad": ["SC"], "secunderabad": ["SC"], "हैदराबाद": ["SC"],
}
for _code in STATIONS:
    CITY_ALIASES.setdefault(_code.lower(), [_code])

# Train categories. speed = average km/h, fare_factor vs a plain mail/express,
# dynamic = flexi-fare style pricing, delay_bias = minutes added to typical delay.
TRAIN_TYPES: dict[str, dict] = {
    "Rajdhani":     dict(speed=78, fare_factor=1.30, dynamic=True,  delay_bias=-10, day_train=False, classes=("1A", "2A", "3A")),
    "Duronto":      dict(speed=76, fare_factor=1.25, dynamic=True,  delay_bias=-6,  day_train=False, classes=("1A", "2A", "3A", "SL")),
    "Shatabdi":     dict(speed=82, fare_factor=1.25, dynamic=True,  delay_bias=-12, day_train=True,  classes=("EC", "CC")),
    "Vande Bharat": dict(speed=88, fare_factor=1.40, dynamic=False, delay_bias=-14, day_train=True,  classes=("EC", "CC")),
    "Garib Rath":   dict(speed=66, fare_factor=0.68, dynamic=False, delay_bias=8,   day_train=False, classes=("3A",)),
    "Superfast":    dict(speed=62, fare_factor=1.00, dynamic=False, delay_bias=10,  day_train=False, classes=("2A", "3A", "SL", "2S")),
    "Express":      dict(speed=52, fare_factor=0.95, dynamic=False, delay_bias=22,  day_train=False, classes=("2A", "3A", "SL", "2S")),
}
PREMIUM_TYPES = {"Rajdhani", "Duronto", "Shatabdi", "Vande Bharat"}

# ratio = fare relative to Sleeper per km; capacity = berths/seats per train (synthetic)
CLASS_INFO: dict[str, dict] = {
    "1A": dict(ratio=6.0, ac=True,  rac=False, capacity=24,  label="First AC"),
    "2A": dict(ratio=3.6, ac=True,  rac=True,  capacity=96,  label="AC 2-Tier"),
    "3A": dict(ratio=2.6, ac=True,  rac=True,  capacity=320, label="AC 3-Tier"),
    "SL": dict(ratio=1.0, ac=False, rac=True,  capacity=560, label="Sleeper"),
    "CC": dict(ratio=1.9, ac=True,  rac=False, capacity=480, label="AC Chair Car"),
    "EC": dict(ratio=3.8, ac=True,  rac=False, capacity=96,  label="Executive Chair Car"),
    "2S": dict(ratio=0.6, ac=False, rac=False, capacity=600, label="Second Sitting"),
}

# Illustrative tariff parameters (NOT the official Indian Railways fare table).
RESERVATION_CHARGE = {"1A": 60, "2A": 50, "3A": 40, "SL": 20, "CC": 40, "EC": 60, "2S": 15}
SUPERFAST_CHARGE = {"1A": 75, "2A": 45, "3A": 45, "SL": 30, "CC": 45, "EC": 75, "2S": 15}
TARIFF = {"flexi_step": 0.10, "flexi_cap": 1.5, "tatkal_rate": 0.30, "tatkal_rate_2s": 0.10, "gst_ac": 0.05}


def stable_seed(*parts) -> int:
    """Deterministic seed from arbitrary parts (same inputs -> same synthetic world)."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def rail_distance(a: str, b: str) -> int:
    """Great-circle distance x 1.28 detour factor: a rough proxy for rail kilometres."""
    s1, s2 = STATIONS[a], STATIONS[b]
    p1, p2 = math.radians(s1.lat), math.radians(s2.lat)
    dphi, dlam = p2 - p1, math.radians(s2.lon - s1.lon)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return int(round(2 * 6371 * math.asin(math.sqrt(h)) * 1.28 + 15))


def flexi_multiplier(sold_fraction: float, dynamic: bool) -> float:
    """Flexi-fare style step pricing: +10% for every 10% of berths sold, capped (illustrative)."""
    if not dynamic:
        return 1.0
    steps = math.floor(min(max(sold_fraction, 0.0), 1.0) * 10 + 1e-9)
    return round(min(TARIFF["flexi_cap"], 1.0 + TARIFF["flexi_step"] * steps), 2)


def fare_breakdown(distance_km: int, travel_class: str, train_type: str,
                   multiplier: float = 1.0, tatkal: bool = False) -> dict:
    """Itemised synthetic fare for one passenger."""
    info, tt = CLASS_INFO[travel_class], TRAIN_TYPES[train_type]
    base = distance_km ** 0.9 * info["ratio"] * tt["fare_factor"]
    flexi = base * (multiplier - 1.0)
    rate = TARIFF["tatkal_rate_2s"] if travel_class == "2S" else TARIFF["tatkal_rate"]
    tatkal_amt = base * rate if tatkal else 0.0
    res = RESERVATION_CHARGE[travel_class]
    sf = 0 if train_type == "Express" else SUPERFAST_CHARGE[travel_class]
    gst = TARIFF["gst_ac"] * (base + flexi + tatkal_amt + res + sf) if info["ac"] else 0.0
    parts = {"base_fare": base, "flexi_surcharge": flexi, "tatkal_premium": tatkal_amt,
             "reservation_charge": res, "superfast_charge": sf, "gst": gst}
    parts = {k: int(round(v)) for k, v in parts.items()}
    parts["total"] = int(sum(parts.values()))
    return parts


@dataclass(frozen=True)
class Train:
    number: str
    name: str
    ttype: str
    origin: str
    destination: str
    departure: time
    duration_min: int
    distance_km: int
    classes: tuple
    days: tuple            # weekdays it runs on, 0 = Monday
    stops: int
    popularity: float      # 0..1 synthetic demand index
    hist_cancellation_rate: float

    @property
    def dynamic_pricing(self) -> bool:
        return TRAIN_TYPES[self.ttype]["dynamic"]

    @property
    def northern(self) -> bool:
        return STATIONS[self.origin].northern or STATIONS[self.destination].northern

    def runs_on(self, d: date) -> bool:
        return d.weekday() in self.days

    def departure_dt(self, d: date) -> datetime:
        return datetime.combine(d, self.departure)

    def arrival_dt(self, d: date) -> datetime:
        return self.departure_dt(d) + timedelta(minutes=self.duration_min)

    def summary(self) -> dict:
        return {"number": self.number, "name": self.name, "type": self.ttype,
                "origin": self.origin, "destination": self.destination,
                "departure": self.departure.strftime("%H:%M"),
                "duration": f"{self.duration_min // 60}h {self.duration_min % 60:02d}m",
                "distance_km": self.distance_km, "classes": list(self.classes),
                "runs_on": [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d] for d in self.days]}


# Hand-specified Pune <-> Delhi corridor used by the showcase query in the brief.
# number, name, return-train name, type, origin, destination, departure, duration (min), classes,
# popularity, cancellation rate
SHOWCASE_TRAINS = [
    ("92201", "Pune–Nizamuddin Duronto", "Nizamuddin–Pune Duronto", "Duronto", "PUNE", "NZM", "11:10", 1185,
     ("1A", "2A", "3A", "SL"), 0.86, 0.11),
    ("92203", "Pune–New Delhi Superfast", "New Delhi–Pune Superfast", "Superfast", "PUNE", "NDLS", "13:30", 1370,
     ("2A", "3A", "SL", "2S"), 0.74, 0.13),
    ("92205", "Deccan–Delhi Express", "Delhi–Deccan Express", "Express", "PUNE", "NDLS", "05:55", 1660,
     ("2A", "3A", "SL", "2S"), 0.78, 0.14),
    ("92207", "Pune–Nizamuddin Garib Rath", "Nizamuddin–Pune Garib Rath", "Garib Rath", "PUNE", "NZM", "08:40", 1440,
     ("3A",), 0.92, 0.09),
    ("92209", "Pune–Nizamuddin Rajdhani", "Nizamuddin–Pune Rajdhani", "Rajdhani", "PUNE", "NZM", "16:15", 1195,
     ("1A", "2A", "3A"), 0.80, 0.10),
]

CORRIDORS = [
    ("CSMT", "NDLS"), ("BCT", "NDLS"), ("HWH", "NDLS"), ("SBC", "NDLS"), ("MAS", "NDLS"), ("BBS", "NDLS"),
    ("SC", "NDLS"), ("PNBE", "NDLS"), ("LKO", "NDLS"), ("JP", "NDLS"), ("AGC", "NDLS"), ("BPL", "NDLS"),
    ("CTC", "HWH"), ("BBS", "HWH"), ("CSMT", "HWH"), ("PUNE", "SBC"), ("PUNE", "SC"), ("ADI", "CSMT"),
    ("MAS", "SBC"), ("CTC", "SBC"), ("ADI", "NDLS"), ("PUNE", "HWH"),
]

_STOP_DENSITY = {"Express": 1 / 55, "Superfast": 1 / 85, "Garib Rath": 1 / 110, "Duronto": 1 / 400,
                 "Rajdhani": 1 / 220, "Shatabdi": 1 / 120, "Vande Bharat": 1 / 140}


def _parse_time(hhmm: str) -> time:
    h, m = hhmm.split(":")
    return time(int(h), int(m))


class RailNetwork:
    """In-memory timetable. Deterministic for a given seed."""

    def __init__(self, seed: int = SEED):
        self.trains: dict[str, Train] = {}
        rng = np.random.default_rng(seed)
        for num, name, rev_name, ttype, o, d, dep, dur, classes, pop, canc in SHOWCASE_TRAINS:
            dist = rail_distance(o, d)
            stops = max(1, int(dist * _STOP_DENSITY[ttype]))
            self._add(Train(num, name, ttype, o, d, _parse_time(dep), dur, dist, classes,
                            tuple(range(7)), stops, pop, canc))
            back = (datetime.combine(date(2026, 1, 1), _parse_time(dep)) + timedelta(hours=9, minutes=5)).time()
            self._add(Train(str(int(num) + 1), rev_name, ttype, d, o, back, dur + 10, dist, classes,
                            tuple(range(7)), stops, round(pop - 0.04, 2), canc))
        counter = 93001
        for a, b in CORRIDORS:
            dist = rail_distance(a, b)
            pool = (["Shatabdi", "Vande Bharat", "Superfast", "Express"] if dist < 700
                    else ["Rajdhani", "Duronto", "Superfast", "Express", "Garib Rath"])
            for ttype in rng.choice(pool, size=int(rng.integers(3, 5)), replace=False):
                for o, d in ((a, b), (b, a)):
                    self._add(self._make_train(str(counter), o, d, str(ttype), dist, rng))
                    counter += 1

    def _add(self, train: Train) -> None:
        self.trains[train.number] = train

    @staticmethod
    def _make_train(number: str, o: str, d: str, ttype: str, dist: int, rng) -> Train:
        cfg = TRAIN_TYPES[ttype]
        stops = max(1, int(dist * _STOP_DENSITY[ttype]))
        duration = int(dist / cfg["speed"] * 60 + stops * 3)
        if cfg["day_train"]:
            minutes = int(rng.integers(60, 100)) * 5          # 05:00 - 08:15
        else:
            minutes = int(rng.integers(0, 288)) * 5
        dep = time(minutes // 60, minutes % 60)
        if rng.random() < 0.6:
            days = tuple(range(7))
        else:
            days = tuple(sorted(int(x) for x in rng.choice(7, size=int(rng.integers(3, 6)), replace=False)))
        name = f"{STATIONS[o].name}–{STATIONS[d].name} {ttype}"
        return Train(number, name, ttype, o, d, dep, duration, dist, cfg["classes"], days, stops,
                     float(np.clip(rng.beta(5, 2), 0.2, 0.98)), float(rng.uniform(0.07, 0.18)))

    def get(self, number: str) -> Train:
        return self.trains[number]

    def resolve(self, place: str) -> list[str]:
        return list(CITY_ALIASES.get(place.strip().lower(), []))

    def trains_between(self, origins, destinations, d: date | None = None) -> list[Train]:
        out = [t for t in self.trains.values() if t.origin in origins and t.destination in destinations
               and (d is None or t.runs_on(d))]
        return sorted(out, key=lambda t: t.departure)

    def cities(self) -> list[str]:
        return sorted({s.city for s in STATIONS.values()})

    def stations_for_city(self, city: str) -> list[str]:
        return [s.code for s in STATIONS.values() if s.city == city]
