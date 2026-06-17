import time
from typing import List, Optional
import networkx as nx

from graph.propagation_graph import PropagationGraph
from trust.trust_engine import TrustEngine


class PatientZeroDetector:
    """Graph-derived patient zero: PZ = argmin_a(t_a)."""

    def __init__(self, graph: PropagationGraph, trust_engine: TrustEngine):
        self.graph = graph
        self.trust = trust_engine
        self.detection_log: List[dict] = []

    def detect(self) -> Optional[str]:
        compromised = self.trust.get_all_compromised()
        if not compromised:
            return None

        patient_zero = self.graph.find_patient_zero(self.trust.delta)
        propagation_path = []
        if not patient_zero:
            # backward_traversal now returns a path list from patient zero to flagged node
            path = self.graph.backward_traversal(list(compromised)[0], self.trust.delta)
            if isinstance(path, list) and len(path) > 0:
                patient_zero = path[0]
                propagation_path = path
            else:
                patient_zero = list(compromised)[0]
                propagation_path = [patient_zero]
        else:
            # If patient zero is found directly, compute forward path to the flagged node
            try:
                propagation_path = nx.shortest_path(self.graph.G, source=patient_zero, target=list(compromised)[0])
            except Exception:
                propagation_path = [patient_zero]
        ts = self.trust.get_compromise_timestamp(patient_zero)
        if ts is None:
            ts = self.graph.first_below_delta.get(patient_zero, time.time())
        self.detection_log.append({
            "timestamp": time.time(),
            "flagged_node": list(compromised)[0],
            "patient_zero": patient_zero,
            "propagation_path": propagation_path,
            "all_compromised": list(compromised),
            "compromise_timestamp": ts,
            "reason": "First node below threshold (argmin t_a)",
        })
        return patient_zero

    def get_propagation_path(self, patient_zero: str) -> list:
        return self.graph.subgraph_from(patient_zero)

    def get_compromise_timestamp(self, agent: str) -> Optional[float]:
        for entry in reversed(self.detection_log):
            if entry.get("patient_zero") == agent:
                return entry.get("compromise_timestamp")
        return self.trust.get_compromise_timestamp(agent)

    def get_last_detection(self) -> Optional[dict]:
        return self.detection_log[-1] if self.detection_log else None
