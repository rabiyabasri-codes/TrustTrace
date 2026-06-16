import os
from typing import Dict, Set

import yaml

from graph.propagation_graph import PropagationGraph
from drift.behavioral_drift import BehavioralDriftModule


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TrustEngine:
    """
    Maintains and updates trust scores for all pipeline components.

    Requires:
      - PropagationGraph (for w_AB edge weights)
      - BehavioralDriftModule (for BD scores)

    Change 9: Trust bounded to [0, 1] after every update.
    Change 10: Agent marked compromised only after k consecutive drops below delta.
    """

    def __init__(self, graph: PropagationGraph, drift_module: BehavioralDriftModule):
        cfg = _load_config()
        self.graph = graph
        self.drift = drift_module

        # Parameters (all tunable in config.yaml / Bayesian search)
        self.mu: float = cfg.get("mu", 0.3)
        self.w_m: float = cfg.get("w_m", 0.5)  # message suspicion weight
        self.w_b: float = cfg.get("w_b", 0.5)  # behavioural drift weight
        self.rho: float = cfg.get("rho", 0.95)  # trust retention factor
        self.eta: float = cfg.get("eta", 0.05)  # trust recovery coefficient
        self.delta: float = cfg.get("delta", 0.4)  # compromise threshold
        self.k: int = cfg.get("persistence_window", cfg.get("k", 3))  # persistence window (Change 10)

        # Per-agent trust state
        self.trust_scores: Dict[str, float] = {}
        self.below_delta_counts: Dict[str, int] = {}
        self.compromised: Set[str] = set()

        # Initialise all known agents from the graph
        for node in self.graph.get_all_nodes():
            self._init_agent(node)

    def _init_agent(self, name: str) -> None:
        if name not in self.trust_scores:
            self.trust_scores[name] = 1.0
            self.below_delta_counts[name] = 0

    @staticmethod
    def _bound(value: float) -> float:
        """
        Change 9: Enforce T_a = min(1, max(0, T_a)) after every update.
        """
        return min(1.0, max(0.0, value))

    def update(
        self,
        receiver: str,
        sender: str,
        suspicion_score: float,
        current_output: str,
    ) -> float:
        """
        Full trust update for one interaction.
        Called after each inter-agent message is logged and scored.

        Returns the new trust score for receiver.
        """
        self._init_agent(receiver)
        self._init_agent(sender)

        T_A = self.trust_scores[sender]
        T_B = self.trust_scores[receiver]
        w_AB = self.graph.get_edge_weight(sender, receiver)

        # Propagation component: how much sender's compromise contaminates receiver
        propagation_drop = self.mu * w_AB * (1.0 - T_A)

        # Behavioral drift component
        BD = self.drift.compute_drift(receiver, current_output)

        # Suspicion component (from scanner)
        S = suspicion_score

        # H = healthy signal from sender (η H term in paper)
        H = T_A

        # Paper equation:
        #   T_a(t) = ρ T_a(t−1) − w_m MS − w_b BD + η H
        T_new = (
            self.rho * T_B
            - self.w_m * S
            - self.w_b * BD
            + self.eta * H
        )

        # Propagation influence applied separately (graph-level contamination)
        T_new = T_new * (1.0 - propagation_drop)

        # Change 9: Apply trust bounding immediately
        T_new = self._bound(T_new)
        self.trust_scores[receiver] = T_new
        self.graph.set_trust(receiver, T_new)

        # Change 10: Compromise confirmation via persistence window
        self._check_compromise(receiver, T_new)

        return T_new

    def _check_compromise(self, agent: str, trust: float) -> None:
        """
        Change 10: Mark agent compromised only if trust < delta
        for k consecutive interactions.
        """
        if trust < self.delta:
            self.below_delta_counts[agent] = self.below_delta_counts.get(agent, 0) + 1
        else:
            self.below_delta_counts[agent] = 0
            self.compromised.discard(agent)

        if self.below_delta_counts.get(agent, 0) >= self.k:
            self.compromised.add(agent)

    def is_compromised(self, agent: str) -> bool:
        return agent in self.compromised

    def get_all_compromised(self) -> Set[str]:
        return set(self.compromised)

    def recover_agent(self, agent: str) -> None:
        """
        Called by RecoveryManager after successful rollback and health check.
        """
        self.trust_scores[agent] = 1.0
        self.below_delta_counts[agent] = 0
        self.compromised.discard(agent)
        self.graph.set_trust(agent, 1.0)
        self.graph.reset_compromise_count(agent)

    def recover_gradually(self, agent: str) -> None:
        """
        Incremental trust recovery for clean interactions.
        """
        if agent not in self.compromised:
            T = self.trust_scores.get(agent, 1.0)
            T_new = self._bound(T + self.rho * (1.0 - T))
            self.trust_scores[agent] = T_new
            self.graph.set_trust(agent, T_new)

