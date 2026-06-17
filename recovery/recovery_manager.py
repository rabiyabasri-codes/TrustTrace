import json
import os
import time
from typing import List, Set

from detector.patient_zero import PatientZeroDetector
from display import terminal_display as display
from memory.memory_manager import MemoryManager
from trust.trust_engine import TrustEngine
from victim_pipeline.agents import run_pipeline_stepwise


REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "reports")


class RecoveryManager:
    """Quarantine → rollback → restore with real state changes."""

    def __init__(
        self,
        memory: MemoryManager,
        trust: TrustEngine,
        detector: PatientZeroDetector,
    ):
        self.memory = memory
        self.trust = trust
        self.detector = detector
        self.quarantined: Set[str] = set()
        self.held_tasks: List[dict] = []
        os.makedirs(REPORT_DIR, exist_ok=True)

    def quarantine(self, compromised_agents: Set[str]) -> None:
        self.quarantined.update(compromised_agents)

    def is_quarantined(self, agent: str) -> bool:
        return agent in self.quarantined

    def hold_task(self, task: dict) -> None:
        self.held_tasks.append(task)

    def rollback_memory(self, compromise_timestamp: float) -> int:
        return self.memory.rollback_after(compromise_timestamp, self.quarantined)

    def restore(self) -> tuple:
        restored_agents = list(self.quarantined)
        for agent in restored_agents:
            restored_trust = self.trust.recover_agent(agent)
            ckpt = self.memory.checkpoint()
            display.show_recovery(ckpt, agent, restored_trust)

        self.quarantined.clear()
        replayed_count = 0
        for task in self.held_tasks:
            query = task.get("query", "")
            if query:
                run_pipeline_stepwise(query)
                replayed_count += 1
        held_count = len(self.held_tasks)
        self.held_tasks.clear()
        return held_count, replayed_count

    def execute_recovery(
        self,
        patient_zero: str,
        attack_source: str,
        propagation_path: list,
        compromise_timestamp: float,
        run_id: str,
    ) -> float:
        start = time.time()
        compromised = self.trust.get_all_compromised()
        self.quarantine(compromised)
        rolled_back = self.rollback_memory(compromise_timestamp)
        held_count, replayed_count = self.restore()
        recovery_time = time.time() - start
        self._generate_report(
            run_id=run_id,
            attack_source=attack_source,
            patient_zero=patient_zero,
            propagation_path=propagation_path,
            compromised_agents=list(compromised),
            held_tasks=held_count,
            replayed_tasks=replayed_count,
            rolled_back_count=rolled_back,
            recovery_time_s=recovery_time,
            compromise_timestamp=compromise_timestamp,
        )
        return recovery_time

    def recovery_complete(self, trust_engine: TrustEngine) -> bool:
        return all(
            trust_engine.trust_scores.get(agent, 1.0) > trust_engine.delta
            for agent in trust_engine.trust_scores
        )

    def _generate_report(self, **kwargs) -> None:
        report_path = os.path.join(REPORT_DIR, f"incident_{kwargs['run_id']}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(kwargs, f, indent=2, default=str)
