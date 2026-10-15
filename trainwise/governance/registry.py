"""Model registry: versions, lineage, champion/challenger status, promotion and rollback.

A lightweight, file-based stand-in for MLflow's registry. Each version is a joblib bundle with
the model plus the reference data Watchtower needs for drift checks and retraining.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import joblib
import pandas as pd


def fingerprint(df) -> str | None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:12]


class ModelRegistry:
    def __init__(self, root: Path, audit=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        self.index: dict = json.loads(self.index_path.read_text()) if self.index_path.exists() else {}
        self.audit = audit
        self._cache: dict = {}
        self._lock = threading.RLock()

    def _save(self) -> None:
        self.index_path.write_text(json.dumps(self.index, indent=2, default=str))

    def _log(self, action: str, name: str, details: dict, actor: str = "registry") -> None:
        if self.audit:
            self.audit.record(actor, action, name, details)

    def register(self, name: str, obj, metrics: dict | None = None, reference=None, train_sample=None,
                 valid_sample=None, parent: str | None = None, notes: str = "", tags: list | None = None,
                 actor: str = "registry") -> str:
        with self._lock:
            entry = self.index.setdefault(name, {"champion": None, "previous": [], "versions": {}})
            version = f"v{len(entry['versions']) + 1}"
            path = self.root / name / f"{version}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": obj, "reference": reference, "train_sample": train_sample,
                         "valid_sample": valid_sample}, path, compress=3)
            entry["versions"][version] = {
                "created_at": time.time(), "metrics": metrics or {}, "status": "candidate", "parent": parent,
                "notes": notes, "tags": tags or [], "artifact": str(path.relative_to(self.root)),
                "data_fingerprint": fingerprint(train_sample),
                "trained_rows": int(len(train_sample)) if isinstance(train_sample, pd.DataFrame) else None}
            self._save()
        self._log("model_registered", name, {"version": version, "parent": parent, "notes": notes,
                                             "metrics": metrics or {}}, actor)
        return version

    def promote(self, name: str, version: str, actor: str = "system", reason: str = "") -> None:
        with self._lock:
            entry = self.index[name]
            old = entry["champion"]
            if old == version:
                return
            if old:
                entry["previous"].append(old)
                entry["versions"][old]["status"] = "archived"
            entry["champion"] = version
            v = entry["versions"][version]
            v.update(status="champion", promoted_at=time.time(), promoted_by=actor)
            self._save()
        self._log("model_promoted", name, {"version": version, "replaced": old, "reason": reason}, actor)

    def rollback(self, name: str, actor: str = "system", reason: str = "") -> str:
        with self._lock:
            entry = self.index[name]
            if not entry["previous"]:
                raise ValueError(f"{name}: no previous version to roll back to")
            current, target = entry["champion"], entry["previous"].pop()
            entry["versions"][current]["status"] = "rolled_back"
            entry["champion"] = target
            entry["versions"][target].update(status="champion", promoted_at=time.time(), promoted_by=actor)
            self._save()
        self._log("model_rolled_back", name, {"from": current, "to": target, "reason": reason}, actor)
        return target

    def reject(self, name: str, version: str, actor: str, reason: str) -> None:
        with self._lock:
            self.index[name]["versions"][version]["status"] = "rejected"
            self._save()
        self._log("model_rejected", name, {"version": version, "reason": reason}, actor)

    def set_baseline(self, name: str, version: str, metric: str, value: float, source: str, actor: str = "watchtower") -> None:
        """Replace the validated baseline with a post-deployment burn-in measurement.

        A hold-out of a few hundred rows is a noisy yardstick; the first live window after promotion
        is what later windows should be compared with. The original figure is kept for the record.
        """
        with self._lock:
            metrics = self.index[name]["versions"][version]["metrics"]
            metrics.setdefault(f"holdout_{metric}", metrics.get(metric))
            metrics[metric] = round(float(value), 4)
            metrics["baseline_source"] = source
            self._save()
        self._log("baseline_reset", name, {"version": version, "metric": metric, "value": round(float(value), 4),
                                           "source": source}, actor)

    def champion(self, name: str) -> str | None:
        return self.index.get(name, {}).get("champion")

    def info(self, name: str, version: str | None = None) -> dict:
        version = version or self.champion(name)
        return {"name": name, "version": version, **self.index[name]["versions"][version]}

    def versions(self, name: str) -> dict:
        return self.index.get(name, {}).get("versions", {})

    def load(self, name: str, version: str | None = None) -> dict:
        version = version or self.champion(name)
        key = (name, version)
        if key not in self._cache:
            self._cache[key] = joblib.load(self.root / self.index[name]["versions"][version]["artifact"])
        return self._cache[key]

    def model(self, name: str, version: str | None = None):
        return self.load(name, version)["model"]

    def recent_deployment(self, name: str, within_s: float = 3600) -> dict | None:
        """The current champion if it replaced an earlier version recently (a regression suspect)."""
        entry = self.index.get(name)
        if not entry or not entry["previous"] or not entry["champion"]:
            return None
        v = entry["versions"][entry["champion"]]
        if time.time() - v.get("promoted_at", 0) <= within_s:
            return {"version": entry["champion"], "replaced": entry["previous"][-1], "promoted_by": v.get("promoted_by")}
        return None

    def first_tagged(self, name: str, tag: str) -> str | None:
        for version, meta in self.versions(name).items():
            if tag in meta.get("tags", []):
                return version
        return None

    def table(self) -> list:
        rows = []
        for name, entry in self.index.items():
            for version, meta in entry["versions"].items():
                rows.append({"model": name, "version": version, "status": meta["status"], "parent": meta.get("parent"),
                             "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(meta["created_at"])),
                             "notes": meta.get("notes", ""),
                             "metrics": ", ".join(f"{k}={v}" for k, v in meta.get("metrics", {}).items()
                                                  if k in ("accuracy", "mae", "recall", "fpr", "macro_f1", "r2"))})
        return rows
