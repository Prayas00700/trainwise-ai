"""Deterministic language understanding for travel and support queries (English, Hindi, Hinglish).

Rule-based on purpose: it is fast, testable, works offline and never invents a station. An LLM,
when configured, only fills fields these rules could not find.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, time, timedelta

from trainwise.data.network import CITY_ALIASES

_DEV = "\u0900-\u097f"
WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
            "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
CLASS_RULES = [
    (r"\b(?:1a|1ac|first ac|first class ac)\b", ["1A"]), (r"\b(?:2a|2ac|2 ac|second ac|two tier|2 tier)\b", ["2A"]),
    (r"\b(?:3a|3ac|3 ac|third ac|three tier|3 tier)\b", ["3A"]), (r"\b(?:sleeper|sl)\b|स्लीपर", ["SL"]),
    (r"\b(?:chair car|cc)\b", ["CC"]), (r"\b(?:executive|ec)\b", ["EC"]), (r"\b(?:second sitting|2s)\b", ["2S"]),
]
PREFERENCES = {"fastest": r"fastest|quickest|\bfast\b|jaldi|जल्दी", "cheapest": r"cheapest|\bcheap\b|lowest fare|sasti|sasta|सस्त",
               "punctual": r"punctual|on[- ]time|reliable|no delay", "confirmed": r"confirm(?:ed)? (?:seat|berth|ticket)|sure seat|guaranteed"}
HINGLISH = {"kya", "hai", "mera", "meri", "kab", "kaise", "kitna", "kitne", "nahi", "karo", "batao", "hua", "jana",
            "chahiye", "aayega", "milega", "paisa", "andar", "wala", "kar", "ho", "sakta", "khulti", "katega"}
_TIME = r"(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?"


@dataclass
class TravelIntent:
    origins: list = field(default_factory=list)
    destinations: list = field(default_factory=list)
    origin_text: str | None = None
    destination_text: str | None = None
    journey_date: date | None = None
    budget: int | None = None
    arrive_by: time | None = None
    depart_after: time | None = None
    depart_before: time | None = None
    classes: list = field(default_factory=list)
    passengers: int = 1
    preferences: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    missing: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("journey_date", "arrive_by", "depart_after", "depart_before"):
            if d[k] is not None:
                d[k] = d[k].isoformat() if k == "journey_date" else d[k].strftime("%H:%M")
        return d


def _to_time(h, m, ampm) -> time:
    h, m = int(h), int(m or 0)
    ampm = (ampm or "").replace(".", "")
    if ampm == "pm" and h < 12:
        h += 12
    if ampm == "am" and h == 12:
        h = 0
    return time(h % 24, min(m, 59))


def _places(t: str) -> list:
    found, taken = [], []
    for alias in sorted(CITY_ALIASES, key=len, reverse=True):
        for m in re.finditer(rf"(?<![\w{_DEV}]){re.escape(alias)}(?![\w{_DEV}])", t):
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), m.end(), alias, CITY_ALIASES[alias]))
    return sorted(found)


def _date(t: str, today: date):
    if re.search(r"day after tomorrow|\bparso\b|परसों", t):
        return today + timedelta(days=2)
    if re.search(r"\btomorrow\b|\bkal\b|(?<![\w" + _DEV + r"])कल(?![\w" + _DEV + r"])", t):
        return today + timedelta(days=1)
    if re.search(r"\btoday\b|\btonight\b|\baaj\b|आज", t):
        return today
    m = re.search(r"\b(20\d\d)-(\d{2})-(\d{2})\b", t)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*", t) or \
        re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?\b", t)
    if m:
        a, b = m.group(1), m.group(2)
        day, mon = (int(a), MONTHS[b]) if a.isdigit() else (int(b), MONTHS[a])
        d = date(today.year, mon, day)
        return d if d >= today else date(today.year + 1, mon, day)
    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", t)
    if m:
        year = int(m.group(3)) if m.group(3) else today.year
        year = year + 2000 if year < 100 else year
        return date(year, int(m.group(2)), int(m.group(1)))
    if "weekend" in t:
        return today + timedelta(days=(5 - today.weekday()) % 7 or 7)
    for name, wd in WEEKDAYS.items():
        if re.search(rf"\b{name}\b", t):
            ahead = (wd - today.weekday()) % 7
            return today + timedelta(days=ahead or (7 if "next" in t else 0))
    return None


def parse_travel_query(text: str, today: date) -> TravelIntent:
    it = TravelIntent()
    t = " " + text.lower().replace("₹", " rs ").replace(",", "") + " "
    # places: role from neighbouring words ("from X", "to Y", "X se Y", "Y jana")
    origin = dest = None
    places = _places(t)
    for s, e, alias, codes in places:
        before, after = t[max(0, s - 14):s], t[e:e + 12]
        if origin is None and re.search(r"(?:from|frm|ex)\s*$", before):
            origin = (alias, codes)
        elif dest is None and re.search(r"(?:to|towards|for|till|until)\s*$", before):
            dest = (alias, codes)
        elif origin is None and re.match(r"\s*(?:se|से)(?=\s|$)", after):
            origin = (alias, codes)
        elif dest is None and re.match(r"\s*(?:tak|को|ko|jana|jaana|जाना|ke liye|के लिए|की)", after):
            dest = (alias, codes)
    for _, _, alias, codes in places:
        if (alias, codes) in (origin, dest):
            continue
        if origin is None:
            origin = (alias, codes)
        elif dest is None:
            dest = (alias, codes)
    if origin:
        it.origin_text, it.origins = origin[0], list(origin[1])
    if dest:
        it.destination_text, it.destinations = dest[0], list(dest[1])
    it.missing = [k for k, v in (("origin", it.origins), ("destination", it.destinations)) if not v]
    # date
    d = _date(t, today)
    if d is None:
        d = today + timedelta(days=1)
        it.assumptions.append("No travel date given; assuming tomorrow.")
    it.journey_date = d
    # budget
    m = (re.search(r"(?:under|below|within|less than|upto|up to|max(?:imum)?|budget(?: of| is)?|not more than)\s*"
                   r"(?:rs\.?|inr|rupees)?\s*(\d+(?:\.\d+)?)\s*(k\b)?", t)
         or re.search(r"(\d+(?:\.\d+)?)\s*(k\b)?\s*(?:rs\.?|rupees|rupaye|रुपये)?\s*(?:ke andar|ke under|se kam|tak ka|के अंदर|से कम)", t))
    if m:
        it.budget = int(float(m.group(1)) * (1000 if m.group(2) else 1))
    # arrival deadline
    m = re.search(r"(?:reach|arrive|arrival|get there|be there|pahunch\w*|पहुँच\w*|पहुंच\w*)[^0-9]{0,25}?" + _TIME, t)
    if m:
        it.arrive_by = _to_time(*m.groups())
    else:
        m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(?:baje|बजे)\s*(?:tak|से पहले)?\s*(?:pahunch|पहुँच|पहुंच)", t)
        if m:
            it.arrive_by = _to_time(m.group(1), m.group(2), "pm" if re.search(r"shaam|raat|शाम|रात", t) else "am")
        else:
            m = re.search(r"\bby\s+" + r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", t)
            if m and not re.search(r"(?:leave|depart|start)\w*\s+by", t):
                it.arrive_by = _to_time(*m.groups())
    m = re.search(r"(?:leave|leaving|depart\w*|start\w*)\s+(?:after|from)\s+" + _TIME, t)
    if m:
        it.depart_after = _to_time(*m.groups())
    m = re.search(r"(?:leave|leaving|depart\w*|start\w*)\s+(?:before|by)\s+" + _TIME, t)
    if m:
        it.depart_before = _to_time(*m.groups())
    # class, passengers, preferences
    for pattern, codes in CLASS_RULES:
        if re.search(pattern, t):
            it.classes += [c for c in codes if c not in it.classes]
    if not it.classes and re.search(r"(?<![\w-])ac(?![\w-])|एसी", t):
        it.classes = ["1A", "2A", "3A", "CC", "EC"]
    m = re.search(r"(\d+)\s*(?:people|persons|passengers|adults|pax|travellers|travelers|tickets?|log|लोग|यात्री)\b", t) \
        or re.search(r"family of (\d+)", t)
    if m:
        it.passengers = max(1, int(m.group(1)))
    it.preferences = [k for k, pat in PREFERENCES.items() if re.search(pat, t)]
    if it.budget and it.passengers == 1:
        pass
    elif it.budget:
        it.assumptions.append(f"Treating ₹{it.budget} as the budget per passenger.")
    return it


def detect_language(text: str) -> str:
    if re.search(rf"[{_DEV}]", text):
        return "hi"
    words = set(re.findall(r"[a-z]+", text.lower()))
    return "hinglish" if len(words & HINGLISH) >= 2 else "en"


def extract_entities(text: str) -> dict:
    pnr = re.search(r"(?<!\d)(\d{10})(?!\d)", text)
    train = re.search(r"(?<!\d)(\d{5})(?!\d)", text)
    return {"pnr": pnr.group(1) if pnr else None, "train_number": train.group(1) if train else None}
