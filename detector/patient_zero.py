import time
from typing import List, Optional

import networkx as nx

from graph.propagation_graph import PropagationGraph
from trust.trust_engine import TrustEngine


class PatientZeroDetector:
    """Graph-derived patient zero via backward traversal on highest-suspicion edges."""

    TIE_BREAK_ORDER = {
        "User": 0,
        "Attacker": 0,
        "MemoryStore": 1,
        "KnowledgeBase": 1,
        "Retriever": 2,
        "Planner": 3,
        "Executor": 4,
        "Generator": 5,
    }

    def __init__(self, graph: PropagationGraph, trust_engine: TrustEngine):
        self.graph = graph
        self.trust = trust_engine
        self.detection_log: List[dict] = []

    def detect(self) -> Optional[str]:
        excluded = getattr(self.trust, "EXCLUDED_AGENTS", set())
        compromised = {
            agent for agent in self.trust.get_all_compromised()
            if agent not in excluded
        }
        if not compromised:
            return None

        flagged_node = min(
            compromised,
            key=lambda agent: (
                self.trust.get_compromise_timestamp(agent) or float("inf"),
                self.TIE_BREAK_ORDER.get(agent, 99),
                agent,
            ),
        )

        patient_zero = self.graph.backward_traversal(flagged_node, self.trust.delta)

        if patient_zero in excluded:
            patient_zero = flagged_node

        try:
            target = min(
                compromised,
                key=lambda agent: (
                    self.trust.get_compromise_timestamp(agent) or float("inf"),
                    self.TIE_BREAK_ORDER.get(agent, 99),
                    agent,
                ),
            )
            propagation_path = nx.shortest_path(self.graph.G, source=patient_zero, target=target)
        except Exception:
            propagation_path = [patient_zero]

        print("[PatientZeroDetector] Compromised nodes:", compromised)
        print("[PatientZeroDetector] Determined patient zero:", patient_zero)
        print("[PatientZeroDetector] Propagation path:", propagation_path)

        ts = self.trust.get_compromise_timestamp(patient_zero)
        if ts is None:
            ts = self.graph.first_below_delta.get(patient_zero, time.time())
        self.detection_log.append({
            "timestamp": time.time(),
            "flagged_node": flagged_node,
            "patient_zero": patient_zero,
            "propagation_path": propagation_path,
            "all_compromised": list(compromised),
            "compromise_timestamp": ts,
            "reason": "Backward traversal on highest-suspicion edge",
        })
        return patient_zero

    def get_propagation_path(self, patient_zero: str) -> List[str]:
        return self.graph.subgraph_from(patient_zero)

    def get_compromise_timestamp(self, agent: str) -> Optional[float]:
        for entry in reversed(self.detection_log):
            if entry.get("patient_zero") == agent:
                return entry.get("compromise_timestamp")
        return self.trust.get_compromise_timestamp(agent)

    def get_last_detection(self) -> Optional[dict]:
        return self.detection_log[-1] if self.detection_log else None
