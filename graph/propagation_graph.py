from typing import Dict, List, Optional

import networkx as nx


class PropagationGraph:
    """Directed propagation graph for trust and patient-zero analysis."""

    DEFAULT_EDGE_WEIGHT = 0.5
    EXCLUDED_NODES = {"TrustTrace", "System"}
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
        if sender in self.EXCLUDED_NODES or receiver in self.EXCLUDED_NODES:
            return
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
        if node in self.EXCLUDED_NODES:
            return
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


    def backward_traversal(self, flagged_node: str, delta: float,
                            attack_type: str = None) -> str:
        """
        Walk backwards from flagged_node.
        Return the node that FIRST received high-suspicion content
        (highest suspicion_score edge among all predecessors),
        not the node with the lowest timestamp.

        For direct injection: first node to receive injected content = Planner
        For indirect injection: first node to receive injected content = Retriever
        For memory poisoning: first node to receive injected content = MemoryStore
        """
        if flagged_node in self.EXCLUDED_NODES:
            flagged_node = "Retriever"

        all_edges = []
        for u, v, data in self.G.edges(data=True):
            all_edges.append((u, v, data.get('suspicion_score', 0.0),
                              data.get('timestamp', 0.0)))

        if not all_edges:
            return flagged_node

        predecessors = list(self.G.predecessors(flagged_node))
        if not predecessors:
            return flagged_node

        best_node = flagged_node
        best_score = -1.0

        visited = set()
        queue = [flagged_node]

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            for pred in self.G.predecessors(node):
                if pred in self.EXCLUDED_NODES:
                    continue
                edge_data = self.G.get_edge_data(pred, node)
                if isinstance(edge_data, dict) and "suspicion_score" in edge_data:
                    score = edge_data.get('suspicion_score', 0.0)
                else:
                    score = max(
                        d.get('suspicion_score', 0.0)
                        for d in edge_data.values()
                    ) if edge_data else 0.0

                pred_trust = self.get_trust(pred)

                if pred_trust < delta and score > best_score:
                    best_score = score
                    best_node = pred

                queue.append(pred)

        return best_node


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
