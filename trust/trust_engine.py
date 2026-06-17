from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Set

import yaml

from drift.behavioral_drift import BehavioralDriftModule
from graph.propagation_graph import PropagationGraph


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


@dataclass
class TrustUpdateResult:
    agent: str
    sender: str
    previous_trust: float
    new_trust: float
    ms: float
    bd: float
    recovery: float
    propagation_drop: float
    weight: float
    compromised: bool


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TrustEngine:
    """
    T_a(t) = rho*T_a(t-1) - w_m*MS - w_b*BD + eta*H
    Then: T_B <- T_B * (1 - mu * w_AB * (1 - T_A))
    Then: T_a = min(1, max(0, T_a))
  """

    def __init__(self, graph: PropagationGraph, drift_module: BehavioralDriftModule):
        cfg = _load_config()
        self.graph = graph
        self.drift = drift_module
        self.mu = cfg.get("mu", 0.3)
        self.w_m = cfg.get("w_m", 0.5)
        self.w_b = cfg.get("w_b", 0.5)
        self.rho = cfg.get("rho", 0.95)
        self.eta = cfg.get("eta", 0.05)
        self.delta = cfg.get("delta", 0.4)
        self.k = cfg.get("persistence_window", cfg.get("k", 3))
        self.trust_scores: Dict[str, float] = {}
        self.below_delta_counts: Dict[str, int] = {}
        self.compromised: Set[str] = set()
        self.compromise_timestamps: Dict[str, float] = {}
        self.last_update: Optional[TrustUpdateResult] = None
        for node in self.graph.get_all_nodes():
            self._init_agent(node)

    def _init_agent(self, name: str) -> None:
        if name not in self.trust_scores:
            self.trust_scores[name] = 1.0
            self.below_delta_counts[name] = 0

    @staticmethod
    def _bound(value: float) -> float:
        return min(1.0, max(0.0, value))

    def update(
        self,
        receiver: str,
        sender: str,
        ms: float,
        bd: float,
        current_output: str,
        timestamp: Optional[float] = None,
    ) -> TrustUpdateResult:
        self._init_agent(receiver)
        self._init_agent(sender)

        T_A = self.trust_scores[sender]
        T_B = self.trust_scores[receiver]
        w_AB = self.graph.get_edge_weight(sender, receiver)
        propagation_drop = self.mu * w_AB * (1.0 - T_A)
        H = T_A
        recovery = self.eta * H

        T_intermediate = self.rho * T_B - self.w_m * ms - self.w_b * bd + recovery
        T_new = T_intermediate * (1.0 - propagation_drop)
        T_new = self._bound(T_new)

        self.trust_scores[receiver] = T_new
        self.graph.set_trust(receiver, T_new)
        self._check_compromise(receiver, T_new, timestamp or time.time())

        result = TrustUpdateResult(
            agent=receiver,
            sender=sender,
            previous_trust=T_B,
            new_trust=T_new,
            ms=ms,
            bd=bd,
            recovery=recovery,
            propagation_drop=propagation_drop,
            weight=w_AB,
            compromised=receiver in self.compromised,
        )
        self.last_update = result
        return result

    def _check_compromise(self, agent: str, trust: float, timestamp: float) -> None:
        if trust < self.delta:
            self.below_delta_counts[agent] = self.below_delta_counts.get(agent, 0) + 1
        else:
            self.below_delta_counts[agent] = 0
            self.compromised.discard(agent)

        trust_floor = trust <= 0.05
        persistent = self.below_delta_counts.get(agent, 0) >= self.k
        if trust_floor or persistent:
            if agent not in self.compromised:
                self.compromise_timestamps[agent] = timestamp
            self.compromised.add(agent)

    def is_compromised(self, agent: str) -> bool:
        return agent in self.compromised

    def get_all_compromised(self) -> Set[str]:
        return set(self.compromised)

    def get_compromise_timestamp(self, agent: str) -> Optional[float]:
        return self.compromise_timestamps.get(agent)

    def recover_agent(self, agent: str) -> float:
        self.trust_scores[agent] = 1.0
        self.below_delta_counts[agent] = 0
        self.compromised.discard(agent)
        self.compromise_timestamps.pop(agent, None)
        self.graph.set_trust(agent, 1.0)
        self.graph.reset_compromise_count(agent)
        return 1.0

    def recover_gradually(self, agent: str) -> None:
        if agent not in self.compromised:
            T = self.trust_scores.get(agent, 1.0)
            T_new = self._bound(T + self.rho * (1.0 - T))
            self.trust_scores[agent] = T_new
            self.graph.set_trust(agent, T_new)
