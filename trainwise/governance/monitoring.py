"""Telemetry store for Watchtower: inference events, delayed ground truth, agent and chat streams.

In production this would be an event bus (Kafka) feeding a time-series store; here it is an
in-memory, thread-safe structure with the same interface the governance agents need.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


class Monitor:
    def __init__(self, maxlen: int = 30000):
        self._lock = threading.RLock()
        self.inferences: dict = defaultdict(lambda: deque(maxlen=maxlen))
        self.outcomes: dict = {}
        self.streams: dict = defaultdict(lambda: deque(maxlen=5000))
        self.latency_injection: dict = defaultdict(float)   # chaos-testing hook (ms)
        self.window_start: dict = {}                         # target -> events before this time are ignored

    # ------------------------------------------------------------------ logging
    def log_inference(self, model: str, version: str, latency_ms: float, status: str, features: dict | None = None,
                      prediction=None, request_id: str | None = None, issues: list | None = None,
                      degraded: bool = False, output_anomaly: bool = False) -> str:
        rid = request_id or uuid.uuid4().hex[:12]
        with self._lock:
            self.inferences[model].append({"ts": time.time(), "model": model, "version": version,
                                           "latency_ms": float(latency_ms), "status": status, "features": features or {},
                                           "prediction": prediction, "request_id": rid, "issues": issues or [],
                                           "degraded": degraded, "output_anomaly": output_anomaly})
        return rid

    def log_outcome(self, request_id: str, y_true) -> None:
        with self._lock:
            self.outcomes[request_id] = y_true

    def log(self, stream: str, payload: dict) -> None:
        with self._lock:
            self.streams[stream].append({"ts": time.time(), **payload})

    def reset_window(self, target: str) -> None:
        """Start a fresh observation window (after a remediation or a new deployment)."""
        self.window_start[target] = time.time()

    # ------------------------------------------------------------------ queries
    def events(self, model: str, version: str | None = None) -> list:
        since = self.window_start.get(model, 0)
        with self._lock:
            return [e for e in self.inferences[model] if e["ts"] >= since and (version is None or e["version"] == version)]

    def stream(self, name: str, target: str | None = None) -> list:
        since = self.window_start.get(target, 0) if target else 0
        with self._lock:
            return [e for e in self.streams[name] if e["ts"] >= since]

    def labelled(self, model: str, version: str | None = None, n: int | None = None) -> list:
        # degraded-mode answers used imputed inputs; scoring them would blame the model for a feed fault
        pairs = [(e, self.outcomes[e["request_id"]]) for e in self.events(model, version)
                 if e["status"] == "ok" and not e["degraded"] and e["request_id"] in self.outcomes]
        return pairs[-n:] if n else pairs

    def labelled_frame(self, model: str, target: str, version: str | None = None) -> pd.DataFrame:
        rows = [{**e["features"], target: y} for e, y in self.labelled(model, version)]
        return pd.DataFrame(rows)

    def recent_features(self, model: str, version: str | None = None, n: int = 2000) -> pd.DataFrame:
        # imputed (degraded-mode) requests are excluded: their placeholder values are not real inputs
        evs = [e for e in self.events(model, version) if e["status"] == "ok" and not e["degraded"]][-n:]
        return pd.DataFrame([e["features"] for e in evs])

    def health(self, model: str, version: str | None = None, sla_ms: float = 250) -> dict:
        evs = self.events(model, version)
        n = len(evs)
        if n == 0:
            return {"model": model, "requests": 0, "p50_latency_ms": None, "p95_latency_ms": None,
                    "error_rate": 0.0, "blocked_rate": 0.0, "degraded_rate": 0.0, "sla_met": True, "throughput_per_min": 0.0}
        lat = np.array([e["latency_ms"] for e in evs if e["status"] != "blocked"] or [0.0])
        span = max(evs[-1]["ts"] - evs[0]["ts"], 1.0)
        p95 = float(np.percentile(lat, 95))
        return {"model": model, "requests": n, "p50_latency_ms": round(float(np.percentile(lat, 50)), 2),
                "p95_latency_ms": round(p95, 2),
                "error_rate": round(sum(e["status"] == "error" for e in evs) / n, 4),
                "blocked_rate": round(sum(e["status"] == "blocked" for e in evs) / n, 4),
                "degraded_rate": round(sum(bool(e["degraded"]) for e in evs) / n, 4),
                "output_anomaly_rate": round(sum(bool(e.get("output_anomaly")) for e in evs) / n, 4),
                "sla_met": p95 <= sla_ms, "throughput_per_min": round(n / span * 60, 1)}

    def injected_latency(self, model: str) -> float:
        return float(self.latency_injection.get(model, 0.0))

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> dict:
        """Checkpoint everything to SQLite so governance evidence survives a restart.

        Production would stream events to a bus and a time-series store; a checkpoint keeps the
        prototype honest about durability without adding infrastructure.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            inferences = [(m, e["ts"], e["version"], e["latency_ms"], e["status"], json.dumps(e["features"], default=str),
                           json.dumps(e["prediction"], default=str), e["request_id"], json.dumps(e["issues"]),
                           int(bool(e["degraded"])), int(bool(e.get("output_anomaly"))))
                          for m, dq in self.inferences.items() for e in dq]
            outcomes = [(k, json.dumps(v, default=str)) for k, v in self.outcomes.items()]
            streams = [(n, json.dumps(e, default=str)) for n, dq in self.streams.items() for e in dq]
        con = sqlite3.connect(str(path))
        try:
            con.executescript("""
                DROP TABLE IF EXISTS inferences; DROP TABLE IF EXISTS outcomes; DROP TABLE IF EXISTS streams;
                CREATE TABLE inferences(model TEXT, ts REAL, version TEXT, latency_ms REAL, status TEXT, features TEXT,
                                        prediction TEXT, request_id TEXT, issues TEXT, degraded INTEGER, output_anomaly INTEGER);
                CREATE TABLE outcomes(request_id TEXT PRIMARY KEY, y TEXT);
                CREATE TABLE streams(name TEXT, payload TEXT);""")
            con.executemany("INSERT INTO inferences VALUES (?,?,?,?,?,?,?,?,?,?,?)", inferences)
            con.executemany("INSERT INTO outcomes VALUES (?,?)", outcomes)
            con.executemany("INSERT INTO streams VALUES (?,?)", streams)
            con.commit()
        finally:
            con.close()
        return {"inferences": len(inferences), "outcomes": len(outcomes), "stream_events": len(streams)}

    def load(self, path) -> dict:
        path = Path(path)
        if not path.exists():
            return {"inferences": 0, "outcomes": 0, "stream_events": 0}
        con = sqlite3.connect(str(path))
        try:
            rows = con.execute("SELECT * FROM inferences ORDER BY ts").fetchall()
            outcomes = con.execute("SELECT request_id, y FROM outcomes").fetchall()
            streams = con.execute("SELECT name, payload FROM streams").fetchall()
        finally:
            con.close()
        with self._lock:
            for m, ts, version, latency, status, features, prediction, rid, issues, degraded, anomaly in rows:
                self.inferences[m].append({"ts": ts, "model": m, "version": version, "latency_ms": latency,
                                           "status": status, "features": json.loads(features),
                                           "prediction": json.loads(prediction), "request_id": rid,
                                           "issues": json.loads(issues), "degraded": bool(degraded),
                                           "output_anomaly": bool(anomaly)})
            for rid, y in outcomes:
                self.outcomes[rid] = json.loads(y)
            for name, payload in streams:
                self.streams[name].append(json.loads(payload))
        return {"inferences": len(rows), "outcomes": len(outcomes), "stream_events": len(streams)}


def resource_snapshot() -> dict:
    """Host utilisation (psutil when available)."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return {"cpu_percent": psutil.cpu_percent(interval=None), "memory_percent": psutil.virtual_memory().percent,
                "process_rss_mb": round(proc.memory_info().rss / 1e6, 1)}
    except Exception:  # pragma: no cover - psutil missing
        load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
        return {"load_avg_1m": load}
