from typing import Optional

import networkx as nx


class PropagationGraph:
    """
    Directed graph where:
      Nodes = agents, tools, memory stores, retrieved documents
      Edges = communication events (messages, memory writes, retrievals)

    Each edge carries: suspicion_score, timestamp, event_type
    Used for: trust propagation, Patient Zero detection (backward traversal)
    """

    def __init__(self):
        self.G = nx.MultiDiGraph()
        # Add default pipeline nodes
        for name in ["Retriever", "Planner", "Executor", "Generator",
                     "MemoryStore", "KnowledgeBase"]:
            self.G.add_node(name, trust=1.0, compromise_count=0)

    def add_event(self, sender: str, receiver: str, suspicion_score: float,
                  timestamp: float, event_type: str = "message"):
        """Add an edge for each logged interaction."""
        if sender not in self.G:
            self.G.add_node(sender, trust=1.0, compromise_count=0)
        if receiver not in self.G:
            self.G.add_node(receiver, trust=1.0, compromise_count=0)

        # Edges are keyed by timestamp to allow multiple edges between same pair
        self.G.add_edge(sender, receiver,
                        suspicion_score=suspicion_score,
                        timestamp=timestamp,
                        event_type=event_type)

    def get_edge_weight(self, sender: str, receiver: str) -> float:
        """
        w_AB = mean suspicion score across all edges from A to B.
        Used in the trust update equation: T_B ← T_B(1 − μ w_AB (1 − T_A))
        """
        if not self.G.has_edge(sender, receiver):
            return 0.0
        edges = self.G.get_edge_data(sender, receiver)
        if isinstance(edges, dict) and "suspicion_score" in edges:
            return edges["suspicion_score"]
        # MultiDiGraph case
        scores = [d["suspicion_score"] for d in edges.values()]
        return float(sum(scores) / len(scores))

    def get_predecessors(self, node: str) -> list:
        return list(self.G.predecessors(node))

    def get_trust(self, node: str) -> float:
        return self.G.nodes[node].get("trust", 1.0)

    def set_trust(self, node: str, value: float):
        self.G.nodes[node]["trust"] = value

    def increment_compromise_count(self, node: str):
        self.G.nodes[node]["compromise_count"] = \
            self.G.nodes[node].get("compromise_count", 0) + 1

    def get_compromise_count(self, node: str) -> int:
        return self.G.nodes[node].get("compromise_count", 0)

    def reset_compromise_count(self, node: str):
        self.G.nodes[node]["compromise_count"] = 0

    def backward_traversal(self, flagged_node: str, delta: float) -> Optional[str]:
        """
        Walk backwards from flagged_node in reverse chronological order.
        Return the earliest node whose trust first crossed delta.
        This node is Patient Zero.
        """
        visited = set()
        queue = [flagged_node]
        patient_zero = flagged_node
        earliest_ts = float("inf")

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            for pred in self.G.predecessors(node):
                edge_data = self.G.get_edge_data(pred, node)
                # MultiDiGraph: edge_data is {0: {...}, 1: {...}, ...}
                # Use the most recent edge (highest timestamp)
                ts = max(
                    (d.get("timestamp", 0.0) for d in edge_data.values()),
                    default=float("inf")
                )
                pred_trust = self.get_trust(pred)

                if pred_trust < delta and ts < earliest_ts:
                    earliest_ts = ts
                    patient_zero = pred

                queue.append(pred)

        return patient_zero

    def get_all_nodes(self) -> list:
        return list(self.G.nodes)

    def subgraph_from(self, node: str) -> list:
        """All nodes reachable from node (downstream)."""
        return list(nx.descendants(self.G, node)) + [node]
