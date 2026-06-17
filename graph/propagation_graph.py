from typing import Dict, List, Optional

import networkx as nx


class PropagationGraph:
    """Directed propagation graph for trust and patient-zero analysis."""

    DEFAULT_EDGE_WEIGHT = 0.5
    PIPELINE_EDGES = [
        ("KnowledgeBase", "Retriever"),
        ("Retriever", "Planner"),
        ("Planner", "Executor"),
        ("Executor", "Generator"),
        ("MemoryStore", "Retriever"),
    ]

    def __init__(self):
        self.G = nx.MultiDiGraph()
        for name in [
            "Retriever", "Planner", "Executor", "Generator",
            "MemoryStore", "KnowledgeBase",
        ]:
            self.G.add_node(name, trust=1.0, compromise_count=0)
        self._seed_pipeline_edges()
        self.first_below_delta: Dict[str, float] = {}

    def _seed_pipeline_edges(self) -> None:
        """Ensure pipeline edges exist with non-zero default weights."""
        for sender, receiver in self.PIPELINE_EDGES:
            if not self.G.has_edge(sender, receiver):
                self.G.add_edge(
                    sender,
                    receiver,
                    suspicion_score=self.DEFAULT_EDGE_WEIGHT,
                    timestamp=0.0,
                    event_type="structural",
                )

    def add_event(
        self,
        sender: str,
        receiver: str,
        suspicion_score: float,
        timestamp: float,
        event_type: str = "message",
    ):
        if sender not in self.G:
            self.G.add_node(sender, trust=1.0, compromise_count=0)
        if receiver not in self.G:
            self.G.add_node(receiver, trust=1.0, compromise_count=0)
        self.G.add_edge(
            sender,
            receiver,
            suspicion_score=suspicion_score,
            timestamp=timestamp,
            event_type=event_type,
        )

    def record_trust_crossing(self, node: str, trust: float, delta: float, timestamp: float):
        if trust < delta and node not in self.first_below_delta:
            self.first_below_delta[node] = timestamp

    def get_edge_weight(self, sender: str, receiver: str) -> float:
        if not self.G.has_edge(sender, receiver):
            for s, r in self.PIPELINE_EDGES:
                if s == sender and r == receiver:
                    return self.DEFAULT_EDGE_WEIGHT
            return 0.0
        edges = self.G.get_edge_data(sender, receiver)
        if isinstance(edges, dict) and "suspicion_score" in edges:
            return edges["suspicion_score"]
        scores = [d["suspicion_score"] for d in edges.values()]
        return float(sum(scores) / len(scores))

    def get_predecessors(self, node: str) -> list:
        return list(self.G.predecessors(node))

    def get_trust(self, node: str) -> float:
        return self.G.nodes[node].get("trust", 1.0)

    def set_trust(self, node: str, value: float):
        self.G.nodes[node]["trust"] = value

    def increment_compromise_count(self, node: str):
        self.G.nodes[node]["compromise_count"] = (
            self.G.nodes[node].get("compromise_count", 0) + 1
        )

    def get_compromise_count(self, node: str) -> int:
        return self.G.nodes[node].get("compromise_count", 0)

    def reset_compromise_count(self, node: str):
        self.G.nodes[node]["compromise_count"] = 0

    def find_patient_zero(self, delta: float) -> Optional[str]:
        """
        Return the earliest node whose trust first crossed below the given delta.
        Uses the timestamps recorded in `first_below_delta` which capture the moment
        each node first fell below the threshold.
        """
        # If no node has ever crossed the threshold, there is no patient zero.
        if not self.first_below_delta:
            return None
        # Select the node with the smallest (earliest) timestamp.
        return min(self.first_below_delta, key=self.first_below_delta.get)


    def backward_traversal(self, flagged_node: str, delta: float) -> Optional[list]:
        """Return the forward propagation path from patient zero to the flagged node.

        If a patient zero is already identified via `find_patient_zero`, we compute the
        shortest directed path from that node to the flagged node.  If no patient zero
        exists yet, we perform a backward BFS to locate the earliest predecessor that
        crossed the threshold and then compute the forward path from that node.
        """
        # Attempt to locate a known patient zero.
        pz = self.find_patient_zero(delta)
        if pz:
            # Compute forward path from patient zero to flagged node.
            try:
                return nx.shortest_path(self.G, source=pz, target=flagged_node)
            except nx.NetworkXNoPath:
                return [pz]
        # No patient zero recorded yet – perform backward search.
        visited = set()
        queue = [flagged_node]
        candidate_pz = flagged_node
        earliest_ts = float("inf")
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for pred in self.G.predecessors(node):
                edge_data = self.G.get_edge_data(pred, node)
                ts = max(
                    (d.get("timestamp", 0.0) for d in edge_data.values()),
                    default=float("inf"),
                )
                pred_trust = self.get_trust(pred)
                if pred_trust < delta and ts < earliest_ts:
                    earliest_ts = ts
                    candidate_pz = pred
                queue.append(pred)
        # Return forward path from the discovered candidate patient zero.
        try:
            return nx.shortest_path(self.G, source=candidate_pz, target=flagged_node)
        except nx.NetworkXNoPath:
            return [candidate_pz]


    def get_propagation_chain(self, start_label: str, agent_order: List[str]) -> List[dict]:
        chain = [{"name": start_label, "label": start_label, "trust": 1.0}]
        for agent in agent_order:
            chain.append({
                "name": agent,
                "label": agent,
                "trust": self.get_trust(agent),
            })
        return chain

    def get_all_nodes(self) -> list:
        return list(self.G.nodes)

    def subgraph_from(self, node: str) -> list:
        return list(nx.descendants(self.G, node)) + [node]
