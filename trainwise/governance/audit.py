"""Tamper-evident audit trail.

Append-only JSON-lines file where every entry stores the SHA-256 hash of the previous entry.
Editing or deleting any past line breaks the chain, which `verify()` detects. Personal data
(PNR / phone numbers, Aadhaar-like numbers, e-mails) is masked before anything is written.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

_PII = [
    (re.compile(r"\b(\d{2})\d{6}(\d{2})\b"), r"\1******\2"),          # 10-digit PNR / mobile number
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "XXXX XXXX XXXX"),      # 12-digit Aadhaar-like number
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<email>"),
]


def mask_pii(obj):
    if isinstance(obj, str):
        for pattern, repl in _PII:
            obj = pattern.sub(repl, obj)
        return obj
    if isinstance(obj, dict):
        return {k: mask_pii(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [mask_pii(v) for v in obj]
    return obj


def _canonical(entry: dict) -> str:
    return json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)


class AuditLog:
    GENESIS = "0" * 64

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last, self._seq = self.GENESIS, 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = json.loads(line)
                    self._last, self._seq = e["hash"], e["seq"] + 1

    def record(self, actor: str, action: str, resource: str, details: dict | None = None,
               severity: str = "info") -> dict:
        with self._lock:
            entry = {"seq": self._seq, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "actor": actor, "action": action, "resource": resource, "severity": severity,
                     "details": json.loads(_canonical(mask_pii(details or {}))), "prev_hash": self._last}
            entry["hash"] = hashlib.sha256(_canonical(entry).encode("utf-8")).hexdigest()
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._last, self._seq = entry["hash"], self._seq + 1
            return entry

    def verify(self) -> dict:
        prev, n = self.GENESIS, 0
        if not self.path.exists():
            return {"valid": True, "entries": 0, "first_broken_seq": None}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            stored = e.pop("hash")
            if e.get("prev_hash") != prev or hashlib.sha256(_canonical(e).encode("utf-8")).hexdigest() != stored:
                return {"valid": False, "entries": n, "first_broken_seq": e.get("seq")}
            prev, n = stored, n + 1
        return {"valid": True, "entries": n, "first_broken_seq": None}

    def tail(self, n: int = 50) -> list:
        if not self.path.exists():
            return []
        lines = [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return [json.loads(ln) for ln in lines[-n:]]
