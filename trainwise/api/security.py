"""API security: role-based access with API keys, and per-key rate limiting (first line of bot defence).

Demo keys are for local use only. Set TRAINWISE_API_KEYS="role:key,role:key" in any real deployment.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import Header, HTTPException, Request

ROLE_RANK = {"passenger": 1, "ops": 2, "admin": 3}
DEMO_KEYS = "passenger:demo-passenger-key,ops:demo-ops-key,admin:demo-admin-key"


def load_keys() -> dict:
    keys = {}
    for pair in os.getenv("TRAINWISE_API_KEYS", DEMO_KEYS).split(","):
        if ":" in pair:
            role, key = pair.split(":", 1)
            keys[key.strip()] = role.strip()
    return keys


KEYS = load_keys()
_hits: dict = defaultdict(deque)
_lock = threading.Lock()


def _key(x_api_key: Optional[str], authorization: Optional[str]) -> str:
    if x_api_key:
        return x_api_key
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return ""


def require_role(min_role: str):
    def dependency(x_api_key: Optional[str] = Header(default=None),
                   authorization: Optional[str] = Header(default=None)) -> str:
        role = KEYS.get(_key(x_api_key, authorization))
        if role is None:
            raise HTTPException(status_code=401, detail="Missing or unknown X-API-Key header")
        if ROLE_RANK[role] < ROLE_RANK[min_role]:
            raise HTTPException(status_code=403, detail=f"This endpoint needs the '{min_role}' role")
        return role
    return dependency


def rate_limit(request: Request, x_api_key: Optional[str] = Header(default=None),
               authorization: Optional[str] = Header(default=None)) -> None:
    limit = int(os.getenv("TRAINWISE_RATE_LIMIT_PER_MIN", "120"))
    key = _key(x_api_key, authorization) or (request.client.host if request.client else "anonymous")
    now = time.time()
    with _lock:
        hits = _hits[key]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded; slow down")
        hits.append(now)
