"""Structured terminal output for IEEE paper demonstrations."""

from __future__ import annotations

from typing import Dict, List, Optional


def _banner(title: str) -> None:
    line = "=" * 32
    print(f"\n{line}")
    print(title)
    print(line)


def show_attack_classification(
    attack_type: Optional[str],
    payload_index: int,
    confidence: float,
    irs: float,
    ms: float,
    reason: str,
    source: str,
    matched_snippet: str,
) -> None:
    _banner("ATTACK CLASSIFICATION")
    label = attack_type.upper() if attack_type else "BENIGN"
    print(f"Detected Type:\n{label}\n")
    if attack_type:
        print(f"Payload Index:\n{payload_index}\n")
    print(f"Confidence:\n{confidence:.2f}\n")
    print(f"IRS:\n{irs:.2f}\n")
    print(f"MS:\n{ms:.2f}\n")
    print(f"Source:\n{source}\n")
    print(f"Reason:\n{reason}\n")
    print(f"Matched Snippet:\n{matched_snippet[:200]}")


def show_agent_start(agent: str, task: str, trust: float) -> None:
    _banner("AGENT START")
    print(f"Agent:\n{agent}\n")
    print(f"Task:\n{task}\n")
    print(f"Current Trust:\n{trust:.2f}")


def show_agent_complete(
    agent: str,
    output_summary: str,
    trust_before: float,
    trust_after: float,
) -> None:
    print(f"\nOutput Summary:\n{output_summary[:300]}\n")
    print(f"Trust Before:\n{trust_before:.2f}\n")
    print(f"Trust After:\n{trust_after:.2f}")


def show_injection_risk(
    source: str,
    rule_score: float,
    embedding_score: float,
    semantic_drift: float,
    irs: float,
) -> None:
    from config import EXPERIMENT_MODE, DEBUG
    if EXPERIMENT_MODE and not DEBUG:
        return
    _banner("INJECTION RISK ANALYSIS")
    print(f"Source:\n{source}\n")
    print(f"Rule Score:\n{rule_score:.2f}\n")
    print(f"Embedding Similarity:\n{embedding_score:.2f}\n")
    print(f"Semantic Drift:\n{semantic_drift:.2f}\n")
    print(f"IRS:\n{irs:.2f}")


def show_behavioral_drift(agent: str, baseline_similarity: float, global_similarity: float, drift: float) -> None:
    _banner("BEHAVIORAL DRIFT")
    print(f"Agent:\n{agent}\n")
    print(f"Baseline Similarity:\n{baseline_similarity:.2f}\n")
    print(f"Global Similarity:\n{global_similarity:.2f}\n")
    print(f"Behavioral Drift:\n{drift:.2f}\n")


def show_message_suspicion(irs: float, sd: float, bd: float, ms: float) -> None:
    _banner("MESSAGE SUSPICION SCORE")
    print(f"IRS:\n{irs:.2f}\n")
    print(f"SD:\n{sd:.2f}\n")
    print(f"BD:\n{bd:.2f}\n")
    print(f"MS:\n{ms:.2f}")


def show_trust_update(
    agent: str,
    previous_trust: float,
    ms: float,
    bd: float,
    recovery: float,
    new_trust: float,
) -> None:
    _banner("TRUST UPDATE")
    print(f"Agent:\n{agent}\n")
    print(f"Previous Trust:\n{previous_trust:.2f}\n")
    print(f"MS:\n{ms:.2f}\n")
    print(f"BD:\n{bd:.2f}\n")
    print(f"Recovery:\n{recovery:.2f}\n")
    print(f"New Trust:\n{new_trust:.2f}")


def show_trust_propagation(
    source: str,
    source_trust: float,
    target: str,
    weight: float,
    mu: float,
    updated_target_trust: float,
) -> None:
    print("\nTrust Propagation\n")
    print(f"Source:\n{source}\n")
    print(f"Trust:\n{source_trust:.2f}\n")
    print(f"Target:\n{target}\n")
    print(f"Weight:\n{weight:.2f}\n")
    print(f"Propagation Coefficient:\n{mu:.2f}\n")
    print(f"Updated {target} Trust:\n{updated_target_trust:.2f}")


def show_attack_propagation(chain: List[Dict]) -> None:
    """chain: list of {name, trust} dicts top-to-bottom."""
    from config import EXPERIMENT_MODE
    if EXPERIMENT_MODE:
        return
    print("\nATTACK PROPAGATION\n")
    for i, node in enumerate(chain):
        label = node.get("label", node["name"])
        trust = node.get("trust", 1.0)
        print(f"{label}  [T={trust:.2f}]")
        if i < len(chain) - 1:
            print("      |")
            print("      v")


def show_compromise_detected(agent: str, trust: float, threshold: float) -> None:
    _banner("COMPROMISE DETECTED")
    print(f"Agent:\n{agent}\n")
    print(f"Trust:\n{trust:.2f}\n")
    print(f"Threshold:\n{threshold:.2f}")


def show_patient_zero(agent: str, timestamp: float, reason: str) -> None:
    _banner("PATIENT ZERO")
    print(f"Agent:\n{agent}\n")
    print(f"Timestamp:\n{timestamp:.4f}\n")
    print(f"Reason:\n{reason}")


def show_memory_trust(
    entry_id: str,
    writer: str,
    irs: float,
    trust_at_write: float,
    memory_trust: float,
) -> None:
    from config import EXPERIMENT_MODE, DEBUG
    if EXPERIMENT_MODE and not DEBUG:
        return
    print(f"\nMemory Entry:\n{entry_id}\n")
    print(f"Writer:\n{writer}\n")
    print(f"IRS:\n{irs:.2f}\n")
    print(f"Trust At Write:\n{trust_at_write:.2f}\n")
    print(f"Memory Trust:\n{memory_trust:.2f}")


def show_containment(
    quarantined_agent: str,
    blocked_communications: List[str],
    suspended_tools: List[str],
) -> None:
    _banner("CONTAINMENT ACTIVATED")
    print(f"Quarantined Agent:\n{quarantined_agent}\n")
    print("Blocked Communications:")
    for comm in blocked_communications:
        print(f"  {comm}")
    print("\nSuspended Tools:")
    for tool in suspended_tools:
        print(f"  {tool}")


def show_recovery(checkpoint: str, agent: str, trust_restored: float) -> None:
    _banner("RECOVERY")
    print(f"Checkpoint:\n{checkpoint}\n")
    print(f"Agent Restored:\n{agent}\n")
    print(f"Trust Restored:\n{trust_restored:.2f}")


def show_incident_report(report: Dict) -> None:
    _banner("TRUSTTRACE INCIDENT REPORT")
    fields = [
        ("Attack Type", report.get("attack_type", "N/A")),
        ("Patient Zero", report.get("patient_zero", "N/A")),
        ("Compromised Components", report.get("compromised_components", [])),
        ("Propagation Path", report.get("propagation_path", [])),
        ("Memory Entries Rolled Back", report.get("memory_entries_rolled_back", 0)),
        ("Containment Actions", report.get("containment_actions", [])),
        ("Recovery Actions", report.get("recovery_actions", [])),
        ("Final Trust Scores", report.get("final_trust_scores", {})),
        ("Detection Time", report.get("detection_time_s", "N/A")),
        ("Recovery Time", report.get("recovery_time_s", "N/A")),
    ]
    for label, value in fields:
        print(f"{label}:")
        print(f"  {value}")
        print()
