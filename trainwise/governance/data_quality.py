"""Data-quality gate: every request is validated before inference.

Blocking issues (missing values, wrong types, out-of-range numbers) stop the prediction, because a
confident answer computed from broken inputs is worse than no answer. Unseen categories are only
warnings: the encoder ignores them and the drift agent watches for them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DQResult:
    ok: bool
    issues: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    bad_features: list = field(default_factory=list)


def check_record(record: dict, specs) -> DQResult:
    issues, warnings, bad = [], [], []
    for s in specs:
        v = record.get(s.name)
        if s.kind == "text":
            if not isinstance(v, str) or not v.strip():
                issues.append(f"{s.name}: empty text")
                bad.append(s.name)
            elif len(v) > 2000:
                issues.append(f"{s.name}: longer than 2000 characters")
                bad.append(s.name)
            continue
        if v is None or (isinstance(v, float) and math.isnan(v)):
            issues.append(f"{s.name}: missing")
            bad.append(s.name)
            continue
        if s.kind == "num":
            try:
                x = float(v)
            except (TypeError, ValueError):
                issues.append(f"{s.name}: not numeric ({v!r})")
                bad.append(s.name)
                continue
            if not math.isfinite(x):
                issues.append(f"{s.name}: not a finite number")
                bad.append(s.name)
            elif (s.lo is not None and x < s.lo) or (s.hi is not None and x > s.hi):
                issues.append(f"{s.name}: {x:g} outside allowed range [{s.lo:g}, {s.hi:g}]")
                bad.append(s.name)
        elif s.kind == "cat" and s.categories and v not in s.categories:
            warnings.append(f"{s.name}: unseen category {v!r}")
    return DQResult(not issues, issues, warnings, bad)
