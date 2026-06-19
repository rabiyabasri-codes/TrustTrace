from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import yaml
import config

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


@dataclass
class MultilayerResult:
    agent: str
    compromised: bool
    irs: float
    ms: float
    bd: float
    lambda_threshold: float
    delta: float
    gamma: float
    irs_pass: bool
    ms_pass: bool
    bd_pass: bool
    consecutive: int
    required_k: int


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TrustEngine:
    """
    T_a(t) = rho*T_a(t-1) - w_m*MS - w_b*BD + eta*H
    Then: T_B <- T_B * (1 - mu * w_AB * (1 - T_A))
    Compromise requires multi-layer: IRS > lambda, MS > delta, BD > gamma (k consecutive).
    """

    EXCLUDED_AGENTS = {"TrustTrace", "System"}

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
        self.gamma = cfg.get("gamma", 0.25)
        self.k = cfg.get("persistence_window", cfg.get("k", 3))
        self.trust_scores: Dict[str, float] = {}
        self.multilayer_counts: Dict[str, int] = {}
        self.compromised: Set[str] = set()
        self.compromise_timestamps: Dict[str, float] = {}
        self.recovery_timestamps: Dict[str, float] = {}
        self.last_multilayer: Dict[str, MultilayerResult] = {}
        self.last_update: Optional[TrustUpdateResult] = None
        self._debug = config.DEBUG
        for node in self.graph.get_all_nodes():
            self._init_agent(node)

    def _init_agent(self, name: str) -> None:
        if name in self.EXCLUDED_AGENTS:
            return
        if name not in self.trust_scores:
            self.trust_scores[name] = 1.0
            self.multilayer_counts[name] = 0

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
        if receiver in self.EXCLUDED_AGENTS:
            return TrustUpdateResult(
                agent=receiver,
                sender=sender,
                previous_trust=1.0,
                new_trust=1.0,
                ms=ms,
                bd=bd,
                recovery=0.0,
                propagation_drop=0.0,
                weight=0.0,
                compromised=False,
            )
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

        if self._debug:
            print(f"[BD Debug] Agent: {receiver} | BD: {bd:.3f} | Trust Before: {T_B:.3f} | Trust After: {T_new:.3f}")

        self.trust_scores[receiver] = T_new
        self.graph.set_trust(receiver, T_new)

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

    def evaluate_multilayer(
        self,
        agent: str,
        irs: float,
        ms: float,
        bd: float,
        lambda_threshold: float,
        timestamp: Optional[float] = None,
    ) -> MultilayerResult:
        """Compromise only when IRS > lambda AND MS > delta AND BD > gamma for k consecutive evaluations."""
        if agent in self.EXCLUDED_AGENTS:
            return MultilayerResult(
                agent=agent,
                compromised=False,
                irs=irs,
                ms=ms,
                bd=bd,
                lambda_threshold=lambda_threshold,
                delta=self.delta,
                gamma=self.gamma,
                irs_pass=False,
                ms_pass=False,
                bd_pass=False,
                consecutive=0,
                required_k=self.k,
            )
        self._init_agent(agent)
        ts = timestamp or time.time()

        irs_pass = irs > lambda_threshold
        ms_pass = ms > self.delta
        bd_pass = bd > self.gamma
        all_pass = irs_pass and ms_pass and bd_pass

        if all_pass:
            self.multilayer_counts[agent] = self.multilayer_counts.get(agent, 0) + 1
            if agent not in self.compromised:
                self.compromise_timestamps[agent] = ts
            self.compromised.add(agent)
        else:
            self.multilayer_counts[agent] = 0

        consecutive = self.multilayer_counts.get(agent, 0)

        result = MultilayerResult(
            agent=agent,
            compromised=agent in self.compromised,
            irs=irs,
            ms=ms,
            bd=bd,
            lambda_threshold=lambda_threshold,
            delta=self.delta,
            gamma=self.gamma,
            irs_pass=irs_pass,
            ms_pass=ms_pass,
            bd_pass=bd_pass,
            consecutive=consecutive,
            required_k=self.k,
        )
        self.last_multilayer[agent] = result

        if (self._debug or config.EXPERIMENT_MODE) and not getattr(config, "SUPPRESS_MULTILAYER", False):
            status = "COMPROMISED" if result.compromised else "healthy"
            failed = []
            if not irs_pass:
                failed.append(f"IRS {irs:.3f} <= lambda {lambda_threshold:.3f}")
            if not ms_pass:
                failed.append(f"MS {ms:.3f} <= delta {self.delta:.3f}")
            if not bd_pass:
                failed.append(f"BD {bd:.3f} <= gamma {self.gamma:.3f}")
            fail_msg = "; ".join(failed) if failed else "all layers passed"
            print(
                f"[Multilayer] {agent} | {status} | IRS={irs:.3f} MS={ms:.3f} BD={bd:.3f} | "
                f"k={consecutive}/{self.k} | {fail_msg}"
            )

        return result

    def compromise(self, agent: str, timestamp: Optional[float] = None) -> None:
        """Explicitly mark agent compromised (multi-layer confirmed)."""
        self._init_agent(agent)
        ts = timestamp or time.time()
        self.compromised.add(agent)
        self.compromise_timestamps[agent] = ts
        self.multilayer_counts[agent] = self.k

    def is_compromised(self, agent: str) -> bool:
        return agent in self.compromised

    def get_all_compromised(self) -> Set[str]:
        return set(self.compromised)

    def get_compromise_timestamp(self, agent: str) -> Optional[float]:
        return self.compromise_timestamps.get(agent)

    def get_recovery_timestamp(self, agent: str) -> Optional[float]:
        return self.recovery_timestamps.get(agent)

    def recover_agent(self, agent: str) -> float:
        ts = time.time()
        self.trust_scores[agent] = 1.0
        self.multilayer_counts[agent] = 0
        self.compromised.discard(agent)
        self.compromise_timestamps.pop(agent, None)
        self.recovery_timestamps[agent] = ts
        self.graph.set_trust(agent, 1.0)
        self.graph.reset_compromise_count(agent)
        return 1.0

    def recover_gradually(self, agent: str) -> None:
        if agent not in self.compromised:
            T = self.trust_scores.get(agent, 1.0)
            T_new = self._bound(T + self.rho * (1.0 - T))
            self.trust_scores[agent] = T_new
            self.graph.set_trust(agent, T_new)
