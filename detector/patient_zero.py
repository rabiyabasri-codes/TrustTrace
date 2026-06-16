import time
from typing import List, Optional

from graph.propagation_graph import PropagationGraph
from trust.trust_engine import TrustEngine


class PatientZeroDetector:
    """
    When any agent is confirmed compromised, traverse the propagation graph
    backward to find the original infection point (Patient Zero).
    """

    def __init__(self, graph: PropagationGraph, trust_engine: TrustEngine):
        self.graph = graph
        self.trust = trust_engine
        self.detection_log: List[dict] = []

    def detect(self) -> Optional[str]:
        """
        Check all agents. If any are compromised, find Patient Zero.
        Returns the Patient Zero agent name, or None if no compromise detected.
        """
        compromised = self.trust.get_all_compromised()
        if not compromised:
            return None

        flagged = list(compromised)[0]
        patient_zero = self.graph.backward_traversal(flagged, self.trust.delta)

        self.detection_log.append(
            {
                "timestamp": time.time(),
                "flagged_node": flagged,
                "patient_zero": patient_zero,
                "all_compromised": list(compromised),
            }
        )

        return patient_zero

    def get_propagation_path(self, patient_zero: str) -> list:
        """Return the downstream infection path from Patient Zero."""
        return self.graph.subgraph_from(patient_zero)

    def get_compromise_timestamp(self, agent: str) -> Optional[float]:
        """Estimate when this agent became compromised from the detection log."""
        for entry in reversed(self.detection_log):
            if agent in entry.get("all_compromised", []):
                return entry["timestamp"]
        return None

