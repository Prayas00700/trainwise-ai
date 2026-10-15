"""Synthetic training / production data with explicit ground-truth processes.

Each generator encodes a plausible causal story (documented inline) and supports *regimes*,
which is how the platform simulates real-world change for the governance layer:

* confirmation  -- ``historical`` vs ``festival_surge`` (cancellations collapse: concept drift)
* delay         -- ``historical`` (Mar-Oct) vs ``winter_fog`` (inputs never seen in training)
* sessions      -- ``historical`` vs ``novel_attack`` (a new low-and-slow bot pattern)

Nothing here is real passenger data.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

from trainwise.config import SEED
from trainwise.data.inventory import (CLASS_DEMAND, demand_index, festival_multiplier, seasonal_fog,
                                   seasonal_rain, sold_fraction_curve)
from trainwise.data.network import CLASS_INFO, TRAIN_TYPES, RailNetwork, stable_seed

QUOTAS = ("GN", "TQ", "LD", "SS")
WL_PRIORITY = {"GNWL": 1.0, "RLWL": 0.55, "PQWL": 0.40, "TQWL": 0.30}  # how fast each waitlist type clears
_TYPES = list(TRAIN_TYPES)
_TYPE_P = np.array([0.12, 0.12, 0.08, 0.06, 0.08, 0.30, 0.24])
_TYPE_P = _TYPE_P / _TYPE_P.sum()


def _rand_date(rng, start: date, end: date) -> date:
    return start + timedelta(days=int(rng.integers(0, (end - start).days + 1)))


# --------------------------------------------------------------------------- confirmation
def make_confirmation_data(n: int = 12000, seed: int = SEED, regime: str = "historical") -> pd.DataFrame:
    """Waitlisted / RAC bookings and their final chart status (CNF, RAC or WL).

    Ground truth: berths freed before charting ~ Poisson(capacity x cancellation-rate x time-left
    + quota releases). A booking confirms if its queue position (scaled by waitlist-type priority)
    is covered by freed berths. Festivals suppress cancellations; the ``festival_surge`` regime
    suppresses them far more than history shows -> the learned mapping becomes wrong.
    """
    rng = np.random.default_rng(seed)
    if regime == "festival_surge":
        start, end, cancel_mult, intensity = date(2026, 10, 10), date(2026, 11, 15), 0.40, 2.0
    else:
        start, end, cancel_mult, intensity = date(2025, 1, 1), date(2026, 9, 30), 1.0, 1.0
    rows = []
    for _ in range(n):
        ttype = _TYPES[rng.choice(len(_TYPES), p=_TYPE_P)]
        cls = str(rng.choice(TRAIN_TYPES[ttype]["classes"]))
        info = CLASS_INFO[cls]
        cap = info["capacity"]
        rac_slots = int(cap * 0.05) if info["rac"] else 0
        d = _rand_date(rng, start, end)
        festival = int(festival_multiplier(d) > 1.0)
        quota = str(rng.choice(QUOTAS, p=[0.78, 0.14, 0.04, 0.04]))
        if quota == "TQ":
            wl_type, days_before = "TQWL", int(rng.integers(0, 2))
        else:
            wl_type = str(rng.choice(["GNWL", "RLWL", "PQWL"], p=[0.70, 0.18, 0.12]))
            days_before = int(min(60, rng.exponential(14)))
        popularity = float(np.clip(rng.beta(5, 2), 0.2, 0.98))
        hist_cancel = float(rng.uniform(0.06, 0.20))
        if rac_slots and rng.random() < 0.18:
            status, position = "RAC", int(rng.integers(1, rac_slots + 1))
            queue = position
        else:
            status = "WL"
            position = int(min(1 + rng.exponential(cap * 0.09 * (0.6 + popularity)), cap * 0.8))
            queue = rac_slots + position
        cancel_rate = max(0.005, hist_cancel * cancel_mult * (1 - 0.30 * festival * intensity))
        time_factor = 0.45 + 0.55 * min(days_before, 60) / 60
        freed = rng.poisson(cap * cancel_rate * time_factor + cap * 0.015)
        eff = queue / WL_PRIORITY[wl_type] if status == "WL" else queue
        if quota in ("LD", "SS"):
            eff *= 0.8
        final = "CNF" if eff <= freed else ("RAC" if eff <= freed + rac_slots else "WL")
        rows.append({"train_type": ttype, "travel_class": cls, "quota": quota, "wl_type": wl_type,
                     "current_status": status, "position": position, "days_before_departure": days_before,
                     "day_of_week": d.weekday(), "month": d.month, "is_festival_window": festival,
                     "route_popularity": round(popularity, 3), "hist_cancellation_rate": round(hist_cancel, 3),
                     # what a real-time feed would report: the cancellation rate actually seen this week
                     "recent_cancellation_rate": round(float(cancel_rate * rng.uniform(0.85, 1.15)), 4),
                     "final_status": final})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- delay
_DENSITY = {"Express": 1 / 55, "Superfast": 1 / 85, "Garib Rath": 1 / 110, "Duronto": 1 / 400,
            "Rajdhani": 1 / 220, "Shatabdi": 1 / 120, "Vande Bharat": 1 / 140}


def make_delay_data(n: int = 15000, seed: int = SEED, regime: str = "historical") -> pd.DataFrame:
    """Train runs and their arrival delay (minutes).

    Ground truth: delay = 0.75 x incoming delay + train-type bias + fog x 120 + rain x 30
    + congestion x 45 + stops x 0.6 + engineering block x 35 + night penalty + gamma noise.
    History covers March-October only, so winter fog is never observed in training.
    """
    rng = np.random.default_rng(seed)
    months = [12, 1] if regime == "winter_fog" else [3, 4, 5, 6, 7, 8, 9, 10]
    rows = []
    for _ in range(n):
        ttype = _TYPES[rng.choice(len(_TYPES), p=_TYPE_P)]
        cfg = TRAIN_TYPES[ttype]
        month = int(rng.choice(months))
        dist = float(rng.uniform(150, 750) if cfg["day_train"] else rng.uniform(250, 2300))
        northern = int(rng.random() < 0.55)
        fog = float(np.clip(seasonal_fog(month) * northern * rng.uniform(0.5, 1.2), 0, 1))
        rain = float(np.clip(seasonal_rain(month) * rng.uniform(0.4, 1.2), 0, 1))
        congestion = float(rng.beta(2, 3))
        stops = max(1, int(dist * _DENSITY[ttype]))
        dep_hour = int(rng.integers(0, 24))
        upstream = float(rng.exponential(10) + 50 * fog * rng.random())
        maint = int(rng.random() < 0.08)
        night = 6 if (dep_hour >= 22 or dep_hour <= 4) else 0
        delay = (0.75 * upstream + cfg["delay_bias"] + 120 * fog + 30 * rain + 45 * congestion
                 + 0.6 * stops + 35 * maint + night + rng.gamma(2.0, 7.0) - 10)
        rows.append({"train_type": ttype, "distance_km": round(dist), "num_stops": stops,
                     "departure_hour": dep_hour, "month": month, "day_of_week": int(rng.integers(0, 7)),
                     "northern_route": northern, "fog_index": round(fog, 3), "rain_index": round(rain, 3),
                     "route_congestion": round(congestion, 3), "upstream_delay_min": round(upstream, 1),
                     "maintenance_block": maint, "arrival_delay_min": round(max(-10.0, delay), 1)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- demand / fare
def make_demand_data(n: int = 10000, seed: int = SEED, regime: str = "historical") -> pd.DataFrame:
    """Share of seats sold at a given number of days before departure (drives flexi fares)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        ttype = _TYPES[rng.choice(len(_TYPES), p=_TYPE_P)]
        cls = str(rng.choice(TRAIN_TYPES[ttype]["classes"]))
        d = _rand_date(rng, date(2025, 1, 1), date(2026, 9, 30))
        days_before = int(rng.integers(0, 61))
        pop = float(np.clip(rng.beta(5, 2), 0.2, 0.98))
        dem = demand_index(d)
        sold = sold_fraction_curve(days_before, min(pop * CLASS_DEMAND[cls], 1.0), dem) + rng.normal(0, 0.035)
        rows.append({"train_type": ttype, "travel_class": cls, "days_before_departure": days_before,
                     "month": d.month, "day_of_week": d.weekday(), "demand_index": dem,
                     "route_popularity": round(pop, 3), "sold_fraction": round(max(sold, 0.0), 4)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- fraud / bots
def _sessions(rng, k: int, kind: str) -> pd.DataFrame:
    def ln(mu, s):
        return rng.lognormal(np.log(mu), s, k)

    region = rng.choice(["metro", "tier2", "tier3"], size=k, p=[0.45, 0.35, 0.20])
    if kind == "legit":
        d = dict(requests_per_min=ln(6, 0.5), avg_inter_click_ms=np.clip(rng.normal(2200, 700, k), 300, None),
                 form_fill_seconds=ln(45, 0.4), captcha_solve_seconds=ln(7, 0.35),
                 accounts_on_device=1 + rng.poisson(0.12, k),
                 ips_last_24h=1 + rng.poisson(np.where(region == "tier3", 0.9, 0.4)),  # mobile-data IP churn
                 bookings_last_24h=rng.poisson(0.4, k), tatkal_window_hit=rng.random(k) < 0.08,
                 payment_failures_24h=rng.poisson(0.12, k),
                 passengers_per_booking=rng.choice([1, 2, 3, 4, 5, 6], size=k, p=[0.42, 0.28, 0.12, 0.10, 0.05, 0.03]),
                 account_age_days=ln(700, 1.0), headless_signal=rng.random(k) < 0.004,
                 device_type=rng.choice(["mobile", "desktop"], size=k, p=[0.68, 0.32]))
    elif kind == "bulk_bot":
        d = dict(requests_per_min=ln(90, 0.5), avg_inter_click_ms=np.clip(rng.normal(150, 60, k), 40, None),
                 form_fill_seconds=ln(4, 0.4), captcha_solve_seconds=ln(1.8, 0.3),
                 accounts_on_device=1 + rng.poisson(1.0, k), ips_last_24h=1 + rng.poisson(1.5, k),
                 bookings_last_24h=1 + rng.poisson(9, k), tatkal_window_hit=rng.random(k) < 0.92,
                 payment_failures_24h=rng.poisson(0.6, k), passengers_per_booking=rng.choice([4, 5, 6], size=k),
                 account_age_days=ln(40, 1.0), headless_signal=rng.random(k) < 0.6,
                 device_type=rng.choice(["mobile", "desktop"], size=k, p=[0.1, 0.9]))
    elif kind == "tout_multi_account":
        d = dict(requests_per_min=ln(20, 0.5), avg_inter_click_ms=np.clip(rng.normal(700, 250, k), 100, None),
                 form_fill_seconds=ln(15, 0.4), captcha_solve_seconds=ln(4, 0.3),
                 accounts_on_device=5 + rng.poisson(6, k), ips_last_24h=1 + rng.poisson(2, k),
                 bookings_last_24h=2 + rng.poisson(6, k), tatkal_window_hit=rng.random(k) < 0.8,
                 payment_failures_24h=rng.poisson(0.3, k), passengers_per_booking=rng.choice([4, 5, 6], size=k),
                 account_age_days=ln(120, 1.0), headless_signal=rng.random(k) < 0.1,
                 device_type=rng.choice(["mobile", "desktop"], size=k, p=[0.3, 0.7]))
    elif kind == "account_takeover":
        d = dict(requests_per_min=ln(10, 0.5), avg_inter_click_ms=np.clip(rng.normal(1200, 400, k), 200, None),
                 form_fill_seconds=ln(25, 0.4), captcha_solve_seconds=ln(6, 0.35),
                 accounts_on_device=1 + rng.poisson(0.3, k), ips_last_24h=4 + rng.poisson(4, k),
                 bookings_last_24h=1 + rng.poisson(3, k), tatkal_window_hit=rng.random(k) < 0.3,
                 payment_failures_24h=1 + rng.poisson(2.5, k), passengers_per_booking=rng.choice([1, 2, 3, 4], size=k),
                 account_age_days=ln(900, 0.8), headless_signal=rng.random(k) < 0.05,
                 device_type=rng.choice(["mobile", "desktop"], size=k, p=[0.5, 0.5]))
    elif kind == "card_testing":
        d = dict(requests_per_min=ln(25, 0.5), avg_inter_click_ms=np.clip(rng.normal(600, 200, k), 100, None),
                 form_fill_seconds=ln(10, 0.4), captcha_solve_seconds=ln(3, 0.3),
                 accounts_on_device=1 + rng.poisson(2, k), ips_last_24h=1 + rng.poisson(3, k),
                 bookings_last_24h=rng.poisson(1, k), tatkal_window_hit=rng.random(k) < 0.1,
                 payment_failures_24h=6 + rng.poisson(6, k), passengers_per_booking=np.ones(k, dtype=int),
                 account_age_days=ln(15, 1.0), headless_signal=rng.random(k) < 0.3,
                 device_type=rng.choice(["mobile", "desktop"], size=k, p=[0.2, 0.8]))
    elif kind == "low_and_slow":   # novel: human-like timings, one account per residential IP, Tatkal hoarding
        d = dict(requests_per_min=ln(7, 0.4), avg_inter_click_ms=np.clip(rng.normal(2000, 500, k), 400, None),
                 form_fill_seconds=ln(40, 0.3), captcha_solve_seconds=ln(7, 0.3),
                 accounts_on_device=np.ones(k, dtype=int), ips_last_24h=1 + rng.poisson(0.6, k),
                 bookings_last_24h=1 + rng.poisson(1.0, k), tatkal_window_hit=rng.random(k) < 0.95,
                 payment_failures_24h=rng.poisson(0.1, k), passengers_per_booking=rng.choice([3, 4, 5, 6], size=k),
                 account_age_days=ln(400, 0.8), headless_signal=np.zeros(k, dtype=bool),
                 device_type=rng.choice(["mobile", "desktop"], size=k, p=[0.65, 0.35]))
    else:
        raise ValueError(kind)
    df = pd.DataFrame(d)
    df["region_segment"] = region
    for col in ("tatkal_window_hit", "headless_signal"):
        df[col] = df[col].astype(int)
    for col in ("requests_per_min", "avg_inter_click_ms", "form_fill_seconds", "captcha_solve_seconds", "account_age_days"):
        df[col] = df[col].round(2)
    df["attack_type"] = "none" if kind == "legit" else kind
    df["is_fraud"] = int(kind != "legit")
    return df


def make_sessions(n: int = 12000, seed: int = SEED, regime: str = "historical") -> pd.DataFrame:
    """Booking sessions (telemetry) labelled as genuine or fraudulent/bot."""
    rng = np.random.default_rng(seed)
    if regime == "novel_attack":
        mix = {"legit": 0.84, "low_and_slow": 0.12, "bulk_bot": 0.015, "tout_multi_account": 0.01,
               "account_takeover": 0.01, "card_testing": 0.005}
    else:
        mix = {"legit": 0.94, "bulk_bot": 0.02, "tout_multi_account": 0.015, "account_takeover": 0.015,
               "card_testing": 0.01}
    parts = [_sessions(rng, max(1, int(round(n * share))), kind) for kind, share in mix.items()]
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- support intents
_INTENT_TEMPLATES = {
    "pnr_status": [
        "what is the status of pnr {pnr}", "check my pnr {pnr}", "pnr {pnr} status please",
        "is my ticket {pnr} confirmed", "has my waitlisted ticket {pnr} got confirmed",
        "current status of booking {pnr}", "will my wl ticket get confirmed", "my ticket is waitlisted, any chance?",
        "मेरा पीएनआर {pnr} का स्टेटस क्या है", "पीएनआर {pnr} कन्फर्म हुआ क्या", "टिकट {pnr} की स्थिति बताइए",
        "mera pnr {pnr} confirm hua kya", "pnr {pnr} ka status batao", "{pnr} wala ticket confirm hai kya",
        "waiting ticket confirm hoga kya",
    ],
    "refund_status": [
        "where is my refund for pnr {pnr}", "refund not received for cancelled ticket {pnr}", "track refund {pnr}",
        "when will i get my money back for {pnr}", "refund status of my cancelled ticket",
        "money not credited after cancellation", "मेरा रिफंड अभी तक नहीं आया", "पीएनआर {pnr} का रिफंड कब मिलेगा",
        "रद्द टिकट का पैसा कब वापस आएगा", "refund kab aayega {pnr}", "mera refund abhi tak nahi aaya",
        "cancel ticket ka paisa kab milega",
    ],
    "cancellation_policy": [
        "what are the cancellation charges", "how much will be deducted if i cancel my ticket",
        "cancellation rules for sleeper class", "can i cancel a tatkal ticket",
        "what happens if i cancel 10 hours before departure", "is there a refund for waitlisted tickets",
        "refund rules if train is late", "टिकट रद्द करने पर कितना पैसा कटेगा", "कैंसिलेशन चार्ज क्या है",
        "तत्काल टिकट रद्द कर सकते हैं क्या", "ticket cancel karne pe kitna katega", "cancellation charge kya hai",
        "tatkal ticket cancel ho sakta hai kya", "train late hai to refund milega kya",
    ],
    "booking_help": [
        "how do i book a tatkal ticket", "what time does tatkal booking open", "how many days in advance can i book",
        "how to book ticket for senior citizen", "how to add passenger details", "when is the chart prepared",
        "तत्काल टिकट कैसे बुक करें", "तत्काल बुकिंग कितने बजे खुलती है", "कितने दिन पहले टिकट बुक कर सकते हैं",
        "tatkal booking kitne baje khulti hai", "ticket kaise book kare", "kitne din pehle booking hoti hai",
        "chart kab banta hai",
    ],
    "travel_recommendation": [
        "which is the best train from {a} to {b}", "suggest a fast train to {b}", "most punctual train from {a} to {b}",
        "cheapest train from {a} to {b}", "recommend a train for {a} to {b} in ac", "best train for {b} this week",
        "{a} से {b} के लिए सबसे अच्छी ट्रेन कौन सी है", "{b} जाने के लिए सस्ती ट्रेन बताइए",
        "{a} se {b} ke liye best train konsi hai", "{b} ke liye sasti train batao",
    ],
    "journey_planning": [
        "plan my trip from {a} to {b} tomorrow under 1500", "i want to travel from {a} to {b} on friday and reach before 10 am",
        "help me plan a journey to {b} this weekend", "need to reach {b} by tomorrow morning from {a}",
        "i need to go from {a} to {b} tomorrow in sleeper under 900", "plan a trip {a} to {b} day after tomorrow",
        "कल {a} से {b} जाना है 1500 के अंदर", "मुझे {a} से {b} की यात्रा की योजना बनानी है",
        "kal {a} se {b} jana hai 1500 ke andar", "{a} se {b} trip plan karo kal subah tak pahunchna hai",
    ],
    "greeting": [
        "hi", "hello", "good morning", "hey there", "thanks", "thank you", "hello, can you help me",
        "नमस्ते", "धन्यवाद", "शुभ प्रभात", "namaste", "shukriya", "hello ji",
    ],
    "complaint": [
        "the coach was very dirty", "ac not working in my coach", "staff misbehaved with me", "i want to file a complaint",
        "food quality was bad", "my luggage was stolen", "the toilet had no water", "कोच बहुत गंदा था",
        "एसी काम नहीं कर रहा है", "मुझे शिकायत दर्ज करनी है", "coach bahut ganda tha", "ac kaam nahi kar raha",
        "complaint darj karni hai",
    ],
}
_CITIES_EN = ["pune", "delhi", "mumbai", "kolkata", "bengaluru", "chennai", "cuttack", "agra", "jaipur", "patna"]
_CITIES_HI = ["पुणे", "दिल्ली", "मुंबई", "कोलकाता", "पटना", "आगरा", "जयपुर"]


def _lang(text: str) -> str:
    return "hi" if any("\u0900" <= ch <= "\u097f" for ch in text) else "en"


def make_intent_data(seed: int = SEED, variants: int = 6) -> pd.DataFrame:
    """Multilingual (English / Hindi / Hinglish) support utterances with intent labels."""
    rnd = random.Random(seed)
    rows = []
    for intent, templates in _INTENT_TEMPLATES.items():
        for tpl in templates:
            for _ in range(variants if "{" in tpl else 2):
                cities = _CITIES_HI if _lang(tpl) == "hi" else _CITIES_EN
                a, b = rnd.sample(cities, 2)
                text = tpl.format(pnr="".join(rnd.choice("0123456789") for _ in range(10)), a=a, b=b)
                rows.append({"text": text, "intent": intent, "lang": _lang(text)})
    return pd.DataFrame(rows).drop_duplicates("text").reset_index(drop=True)


# --------------------------------------------------------------------------- users & PNRs
def make_user_profiles(seed: int = SEED, n: int = 40) -> dict:
    rng = np.random.default_rng(seed)
    users = {}
    for i in range(1, n + 1):
        cls = str(rng.choice(["SL", "3A", "2A", "CC"], p=[0.45, 0.35, 0.12, 0.08]))
        affinity = dict(zip(_TYPES, np.round(rng.dirichlet(np.ones(len(_TYPES)) * 0.7), 3).tolist()))
        users[f"U{i:03d}"] = {"user_id": f"U{i:03d}", "preferred_class": cls, "type_affinity": affinity,
                              "preferred_departure_hour": int(rng.integers(5, 23)),
                              "price_sensitivity": round(float(rng.uniform(0.2, 0.9)), 2),
                              "trips_last_year": int(rng.integers(1, 25))}
    users["U001"].update({"preferred_class": "SL", "preferred_departure_hour": 11, "price_sensitivity": 0.75,
                          "type_affinity": {**{t: 0.03 for t in _TYPES}, "Duronto": 0.45, "Superfast": 0.37}})
    return users


def make_pnr_db(network: RailNetwork, inventory, today: date, seed: int = SEED) -> dict:
    """Mock PNR records (current bookings and cancelled tickets with refunds)."""
    rng = np.random.default_rng(seed)
    db: dict[str, dict] = {}
    trains = list(network.trains.values())

    def add(pnr: str, train, cls: str, journey: date, status: str, cancelled: bool = False, refund: dict | None = None):
        fare = int(round(train.distance_km ** 0.9 * CLASS_INFO[cls]["ratio"] * TRAIN_TYPES[train.ttype]["fare_factor"]))
        db[pnr] = {"pnr": pnr, "train_number": train.number, "train_name": train.name, "class": cls,
                   "from": train.origin, "to": train.destination, "journey_date": journey.isoformat(),
                   "departure": train.departure.strftime("%H:%M"), "passengers": 1, "booking_status": status,
                   "current_status": status, "chart_prepared": False, "fare_paid": fare,
                   "cancelled": cancelled, "refund": refund}

    duronto = network.get("92201")
    add("4521873690", duronto, "SL", today + timedelta(days=3), "GNWL 18")
    add("6203914785", network.get("92207"), "3A", today + timedelta(days=5), "RAC 9")
    add("2718281828", network.get("92203"), "3A", today + timedelta(days=9), "CNF B2 41", cancelled=True,
        refund={"status": "initiated", "amount": 1652, "initiated_on": (today - timedelta(days=2)).isoformat(),
                "mode": "original payment method (UPI)"})
    for _ in range(25):
        t = trains[int(rng.integers(0, len(trains)))]
        cls = str(rng.choice(t.classes))
        journey = today + timedelta(days=int(rng.integers(1, 30)))
        snap = inventory.snapshot(t, cls, journey, today)
        pnr = str(stable_seed("pnr", t.number, cls, journey.toordinal()))[-10:].rjust(10, "4")
        add(pnr, t, cls, journey, snap["status"] if snap["kind"] != "AVL" else f"CNF S{int(rng.integers(1, 12))} {int(rng.integers(1, 72))}")
    return db
