"""Composition root: wires data, models, agents and the Watchtower governance layer together."""
from __future__ import annotations

import os
import shutil
import threading
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

from sklearn.model_selection import train_test_split

from trainwise.agents.llm import get_llm
from trainwise.agents.support_assistant import KnowledgeBase, SupportAssistant
from trainwise.agents.tools import RailTools
from trainwise.agents.travel_planner import TravelPlannerAgent
from trainwise.config import DEFAULT_ARTIFACT_DIR, KB_PATH, SEED, load_policies
from trainwise.data.generators import (make_confirmation_data, make_delay_data, make_demand_data, make_intent_data,
                                    make_pnr_db, make_sessions, make_user_profiles)
from trainwise.data.inventory import ConditionsFeed, InventorySimulator
from trainwise.data.network import RailNetwork
from trainwise.governance.audit import AuditLog
from trainwise.governance.lifecycle import LifecycleManager
from trainwise.governance.monitoring import Monitor
from trainwise.governance.registry import ModelRegistry
from trainwise.governance.supervisor import GovernanceSupervisor
from trainwise.models.confirmation import ConfirmationPredictor
from trainwise.models.delay import DelayPredictor
from trainwise.models.fare import FareDemandForecaster, FareEngine
from trainwise.models.fraud import FraudDetector
from trainwise.models.intent import IntentClassifier
from trainwise.models.recommender import RankingPolicy
from trainwise.services import ModelService, RecommenderService
from trainwise.simulation import TrafficSimulator

MODEL_CLASSES = {c.name: c for c in (ConfirmationPredictor, DelayPredictor, FareDemandForecaster, FraudDetector,
                                     IntentClassifier)}
GENERATORS = {"confirmation_predictor": make_confirmation_data, "delay_predictor": make_delay_data,
              "fare_demand_forecaster": make_demand_data, "fraud_detector": make_sessions}
SCALES = {
    "full": {"confirmation_predictor": 12000, "delay_predictor": 15000, "fare_demand_forecaster": 10000,
             "fraud_detector": 12000},
    "small": {"confirmation_predictor": 3000, "delay_predictor": 3500, "fare_demand_forecaster": 3000,
              "fraud_detector": 4000},
}


class Platform:
    def __init__(self, artifact_dir=None, scale: str = "full", fresh: bool = False, today: date | None = None):
        self.artifact_dir = Path(artifact_dir or os.getenv("TRAINWISE_ARTIFACT_DIR") or DEFAULT_ARTIFACT_DIR)
        if fresh and self.artifact_dir.exists():
            shutil.rmtree(self.artifact_dir)
        self.scale = scale
        self.policies = load_policies()
        self._today = today
        self.audit = AuditLog(self.artifact_dir / "audit" / "audit_log.jsonl")
        self.registry = ModelRegistry(self.artifact_dir / "registry", self.audit)
        self.monitor = Monitor()
        self.network = RailNetwork(SEED)
        self.inventory = InventorySimulator(self.network)
        self.conditions = ConditionsFeed(self.network)
        self.feed_status: dict = {}
        self.deployments = defaultdict(lambda: {"replicas": 2})
        self.approvals: list = []
        self.notifications: list = []
        self.live_metrics: dict = {}
        self.last_drift: dict = {}
        self.last_fairness: dict = {}
        self.genai_stats: dict = {}
        self.agent_stats: dict = {}
        self.services = {name: ModelService(name, self) for name in MODEL_CLASSES}
        self.bootstrap_report = self.bootstrap()
        self.recommender = RecommenderService(self)
        self.fare = FareEngine(self)
        self.tools = RailTools(self)
        self.kb = KnowledgeBase(KB_PATH)
        self.users = make_user_profiles(SEED)
        self.pnr_db = make_pnr_db(self.network, self.inventory, self.today(), SEED)
        self.llm = get_llm()
        self.planner = TravelPlannerAgent(self)
        self.support = SupportAssistant(self)
        self.lifecycle = LifecycleManager(self)
        self.supervisor = GovernanceSupervisor(self)
        self.simulator = TrafficSimulator(self)
        self.telemetry_path = self.artifact_dir / "telemetry.sqlite"
        self.telemetry_restored = self.monitor.load(self.telemetry_path)

    def today(self) -> date:
        return self._today or date.today()

    def checkpoint(self) -> dict:
        """Persist telemetry so monitoring evidence and the approval queue's context survive restarts."""
        return self.monitor.save(self.telemetry_path)

    def bootstrap(self, force: bool = False) -> dict:
        names = list(MODEL_CLASSES) + ["train_recommender"]
        if not force and all(self.registry.champion(n) for n in names):
            return {"status": "loaded"}
        t0 = time.perf_counter()
        sizes, report = SCALES[self.scale], {}
        for name, cls in MODEL_CLASSES.items():
            df = make_intent_data(SEED) if name == "intent_classifier" else GENERATORS[name](sizes[name], SEED)
            strat = df[cls.target] if cls.task == "classification" else None
            train, valid = train_test_split(df, test_size=0.2, random_state=SEED, stratify=strat)
            model = cls().fit(train)
            metrics = model.evaluate(valid)
            v = self.registry.register(
                name, model, metrics=metrics,
                reference=model.frame(train.sample(min(len(train), 5000), random_state=SEED)),
                train_sample=train.sample(min(len(train), 8000), random_state=SEED).reset_index(drop=True),
                valid_sample=valid.reset_index(drop=True), notes="bootstrap: trained on synthetic historical data",
                tags=["fairness_approved"], actor="bootstrap")
            self.registry.promote(name, v, actor="bootstrap", reason="initial deployment")
            report[name] = metrics
        v = self.registry.register("train_recommender", RankingPolicy(note="default multi-objective weights"),
                                   notes="default multi-objective weights", tags=["fairness_approved"], actor="bootstrap")
        self.registry.promote("train_recommender", v, actor="bootstrap", reason="initial deployment")
        seconds = round(time.perf_counter() - t0, 1)
        self.audit.record("bootstrap", "platform_bootstrapped", "all", {"seconds": seconds, "scale": self.scale})
        return {"status": "trained", "seconds": seconds, "metrics": report}


_PLATFORM: Platform | None = None
_LOCK = threading.Lock()


def get_platform(reset: bool = False) -> Platform:
    """Process-wide platform used by the API and the dashboard."""
    global _PLATFORM
    with _LOCK:
        if _PLATFORM is None or reset:
            today = os.getenv("TRAINWISE_TODAY")
            _PLATFORM = Platform(scale=os.getenv("TRAINWISE_SCALE", "full"), fresh=reset,
                                 today=date.fromisoformat(today) if today else None)
        return _PLATFORM
