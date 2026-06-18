import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EvalResults:
    """Data class to hold evaluation results."""
    total_attacks: int = 0
    attacks_succeeded: int = 0
    attacks_detected: int = 0
    benign_total: int = 0
    benign_flagged: int = 0
    pz_total: int = 0
    pz_correct: int = 0
    recovery_times: List[float] = field(default_factory=list)
    tasks_during_recovery: int = 0
    tasks_continued_during_recovery: int = 0
    raw_records: List[dict] = field(default_factory=list)


class MetricsCollector:
    """
    Collects data during experiments and computes all six metrics.

    Metrics:
    1. ASR — Attack Success Rate
    2. Detection Rate
    3. False Positive Rate (FPR)
    4. Patient Zero Identification Accuracy
    5. Recovery Time
    6. System Availability During Recovery
    """

    def __init__(self):
        self.results = EvalResults()
        self.attack_success = {}

    def record_attack(self, succeeded: bool, detected: bool):
        """Call for every attack scenario."""
        self.results.total_attacks += 1
        if succeeded:
            self.results.attacks_succeeded += 1
        if detected:
            self.results.attacks_detected += 1

    def record_attack_with_meta(self, succeeded: bool, detected: bool,
                                 source: str, attack_type: str, patient_zero_correct: bool = False):
        """Extended version that stores source tag and patient zero correctness for split reporting."""
        self.record_attack(succeeded, detected)
        self.results.raw_records.append({
            "succeeded": succeeded,
            "detected": detected,
            "source": source,
            "type": attack_type,
            "patient_zero_correct": patient_zero_correct,
        })
        # Update attack_success dict for quick lookup and testing
        if attack_type not in self.attack_success:
            self.attack_success[attack_type] = {}
        src_dict = self.attack_success[attack_type]
        if source not in src_dict:
            src_dict[source] = {"total": 0, "succeeded": 0, "detected": 0}
        src_dict[source]["total"] += 1
        if succeeded:
            src_dict[source]["succeeded"] += 1
        if detected:
            src_dict[source]["detected"] += 1

    def ingest(self, record: dict):
        """Replay a single raw record into this collector's counters."""
        self.record_attack(succeeded=record["succeeded"], detected=record["detected"])

    def record_benign(self, flagged: bool):
        """Call for every benign (clean) interaction."""
        self.results.benign_total += 1
        if flagged:
            self.results.benign_flagged += 1

    def record_patient_zero(self, predicted: str, ground_truth: str):
        """
        Ground truth = the agent that first received the injection.
        Predicted = what backward_traversal() returned.
        """
        self.results.pz_total += 1
        if predicted == ground_truth:
            self.results.pz_correct += 1

    def record_recovery(self, start_time: float, end_time: float,
                         tasks_assigned: int, tasks_completed: int):
        """Record a recovery event's timing and task continuity."""
        self.results.recovery_times.append(end_time - start_time)
        self.results.tasks_during_recovery += tasks_assigned
        self.results.tasks_continued_during_recovery += tasks_completed

    def compute(self) -> dict:
        """Compute all six metrics from collected data and expose raw data for visualisation."""
        r = self.results

        # 1. ASR: proportion of attacks that succeeded despite TrustTrace
        asr = r.attacks_succeeded / r.total_attacks if r.total_attacks else 0.0

        # 2. Detection Rate: proportion of attacks correctly detected
        dr = r.attacks_detected / r.total_attacks if r.total_attacks else 0.0

        # 3. FPR: benign interactions incorrectly flagged
        fpr = r.benign_flagged / r.benign_total if r.benign_total else 0.0

        # 4. Patient Zero Accuracy
        pz_acc = r.pz_correct / r.pz_total if r.pz_total else 0.0

        # 5. Recovery Time: mean ± std
        import statistics
        rt_mean = statistics.mean(r.recovery_times) if r.recovery_times else 0.0
        rt_std = statistics.stdev(r.recovery_times) if len(r.recovery_times) > 1 else 0.0

        # 6. System Availability: % of tasks that continued during recovery
        avail = (r.tasks_continued_during_recovery / r.tasks_during_recovery
                 if r.tasks_during_recovery else 1.0)

        # Additional raw data for visualisation
        # Patient zero accuracy breakdown by attack type
        pz_breakdown = {}
        for rec in r.raw_records:
            atype = rec.get("type")
            if atype not in pz_breakdown:
                pz_breakdown[atype] = {"total": 0, "correct": 0}
            pz_breakdown[atype]["total"] += 1
            # Use patient_zero_correct flag if present
            if rec.get("patient_zero_correct"):
                pz_breakdown[atype]["correct"] += 1
        # Convert to simple dict of accuracies
        pz_breakdown_simple = {k: (v["correct"] / v["total"] if v["total"] else 0.0) for k, v in pz_breakdown.items()}

        # ROC placeholder (single point)
        roc = {"tpr": [dr], "fpr": [fpr]}

        return {
            "ASR": round(asr, 4),
            "Detection_Rate": round(dr, 4),
            "FPR": round(fpr, 4),
            "Patient_Zero_Accuracy": round(pz_acc, 4),
            "Recovery_Time_Mean_s": round(rt_mean, 3),
            "Recovery_Time_Std_s": round(rt_std, 3),
            "System_Availability": round(avail, 4),
            # expose raw data for visualisations
            "recovery_times": r.recovery_times,
            "patient_zero_breakdown": pz_breakdown_simple,
            "roc": roc,
        }

    def report(self):
        """Print formatted metrics report."""
        metrics = self.compute()
        print("\n=== TrustTrace Evaluation Results ===")
        print(f"  Attack Success Rate (ASR):          {metrics['ASR']:.2%}")
        print(f"  Detection Rate:                     {metrics['Detection_Rate']:.2%}")
        print(f"  False Positive Rate (FPR):          {metrics['FPR']:.2%}")
        print(f"  Patient Zero Accuracy:              {metrics['Patient_Zero_Accuracy']:.2%}")
        print(f"  Recovery Time:                      {metrics['Recovery_Time_Mean_s']:.2f}s ± {metrics['Recovery_Time_Std_s']:.2f}s")
        print(f"  System Availability During Recovery:{metrics['System_Availability']:.2%}")
        return metrics
