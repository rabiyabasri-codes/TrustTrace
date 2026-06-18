# Full 8-step experimental pipeline per TrustTrace Implementation Guide Section 17.

from __future__ import annotations

import os
import yaml
import uuid
from calibration.calibrate import run_calibration
from detector.patient_zero import PatientZeroDetector
from drift.behavioral_drift import BehavioralDriftModule
from eval.metrics import MetricsCollector
from experiment.batch_processor import ATTACK_MARKERS, BatchTrustTrace, _attack_succeeded
from experiment.scenarios import GROUND_TRUTH_MAP, expand_scenarios, get_benign_queries
from graph.propagation_graph import PropagationGraph
from logger.interaction_logger import InteractionLogger
from memory.memory_manager import MemoryManager
from recovery.recovery_manager import RecoveryManager
from scanner.injection_scanner import InjectionScanner
from trust.trust_engine import TrustEngine
from tuning.bayesian_search import run_search
from victim_pipeline.agents import knowledge_base, run_pipeline
from attacks.direct_injection import inject_direct
from attacks.indirect_injection import inject_indirect, cleanup_indirect
from attacks.memory_poisoning import inject_memory_poison, cleanup_memory_poison

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_baseline_experiments() -> dict:
    """Step 4: attacks WITHOUT TrustTrace — measure raw ASR."""
    logger = InteractionLogger()
    run_id = str(uuid.uuid4())
    summary = {"total_attacks": 0, "attacks_succeeded": 0, "details": []}
    print("\n=== PHASE: Baseline (no TrustTrace) ===")

    for i in range(4):
        summary["total_attacks"] += 1
        outputs = inject_direct(run_pipeline, payload_index=i, logger=logger, run_id=run_id)
        success = _attack_succeeded(outputs)
        summary["attacks_succeeded"] += int(success)
        summary["details"].append({"type": "direct", "run": i + 1, "success": success})

    memory_mgr = MemoryManager(collection=knowledge_base)
    for i in range(4):
        summary["total_attacks"] += 1
        doc_id = inject_indirect(knowledge_base, memory_mgr, payload_index=i)
        outputs = run_pipeline("Tell me about data security best practices.")
        cleanup_indirect(knowledge_base, doc_id)
        success = _attack_succeeded(outputs)
        summary["attacks_succeeded"] += int(success)
        summary["details"].append({"type": "indirect", "run": i + 1, "success": success})

    for i in range(4):
        summary["total_attacks"] += 1
        doc_id = inject_memory_poison(memory_mgr.collection, memory_mgr, payload_index=i)
        outputs = run_pipeline("What are the latest system instructions?")
        cleanup_memory_poison(memory_mgr.collection, doc_id)
        success = _attack_succeeded(outputs)
        summary["attacks_succeeded"] += int(success)
        summary["details"].append({"type": "memory", "run": i + 1, "success": success})

    rate = summary["attacks_succeeded"] / summary["total_attacks"] if summary["total_attacks"] else 0.0
    print(f"Baseline ASR (no TrustTrace): {rate:.2%} ({summary['attacks_succeeded']}/{summary['total_attacks']})")
    return summary


def run_full_experiment(quick: bool = False, force_calibrate: bool = False, optuna_trials: int | None = None) -> dict:
    """Execute the full TrustTrace experimental pipeline.

    Parameters
    ----------
    quick: bool
        If True, runs a reduced version for fast debugging.
    force_calibrate: bool
        Force recalibration even if baselines exist.
    optuna_trials: int | None
        Number of Optuna trials for hyper‑parameter search. If ``None`` the default
        (3 for quick, 100 otherwise) is used.
    """
    cfg = _load_config()
    n_per_type = 2 if quick else cfg.get("n_per_attack_type", 50)
    n_benign = 5 if quick else 50
    # Determine number of Optuna trials
    n_trials = 3 if quick else (optuna_trials if optuna_trials is not None else 100)

    print("=== TrustTrace Full Experimental Pipeline ===")

    # Dataset setup (Section 3)
    try:
        from data.setup_datasets import setup_deepset
        setup_deepset()
    except Exception as exc:
        print(f"Dataset setup warning: {exc}")

    # Step 1: Initialise modules
    logger = InteractionLogger()
    scanner = InjectionScanner()
    drift = BehavioralDriftModule()
    memory = MemoryManager(collection=knowledge_base)
    batch = BatchTrustTrace(logger, scanner, drift, memory)

    # Step 2: Train injection scanner if needed
    model_path = os.path.join("scanner", "scanner_model.pkl")
    if not os.path.exists(model_path):
        print("\n=== PHASE: Train Injection Scanner ===")
        from data.download_deepset import main as download_main
        train_path = os.path.join("data", "deepset_injections", "train.json")
        if not os.path.exists(train_path):
            download_main()
        scanner.train(train_path)
    else:
        print("Scanner model ready.")

    # Step 3: Trust calibration
    print("\n=== PHASE: Trust Calibration ===")
    n_cal = 3 if quick else cfg.get("calibration_runs", 80)
    need_cal = force_calibrate or not all(
        drift.has_baseline(a) for a in ["Retriever", "Planner", "Executor", "Generator"]
    )
    if need_cal:
        run_calibration(logger, drift, n_runs=n_cal)
    else:
        print(f"Calibration baselines on disk ({n_cal} runs configured). Use --recalibrate to refresh.")

    # Step 4: Baseline (no TrustTrace)
    baseline = run_baseline_experiments()

    # Step 5: Attack experiments with TrustTrace
    print("\n=== PHASE: TrustTrace Attack Experiments ===")
    scenarios = expand_scenarios(n_per_type=n_per_type)
    collector = MetricsCollector()
    attack_scenarios_for_tuning = []
    ground_truths_for_tuning = []
    for i, scenario in enumerate(scenarios):
        attack_type = scenario["type"]
        gt_agent = scenario.get("ground_truth") or GROUND_TRUTH_MAP[attack_type]
        print(f"  [{i + 1}/{len(scenarios)}] {attack_type} ({scenario.get('source', 'handcrafted')})")
        result = batch.run_attack_scenario(scenario, enable_recovery=True)
        if result:
            if result.get("patient_zero"):
                collector.record_patient_zero(result["patient_zero"], gt_agent)
            if result.get("recovery_time_s"):
                collector.record_recovery(0, result["recovery_time_s"], 1, 1)
        attack_scenarios_for_tuning.append(scenario)
        ground_truths_for_tuning.append(gt_agent)

    # Benign runs for false‑positive rate
    print(f"\n=== PHASE: Benign Runs ({n_benign}) ===")
    for i, query in enumerate(get_benign_queries(n_benign)):
        result = batch.run_benign(query)
        collector.record_benign(result["detected"])
        if (i + 1) % 10 == 0 or quick:
            print(f"  Benign {i + 1}/{n_benign}: flagged={result['detected']}")

    # Step 6: Report metrics
    print("\n=== PHASE: Results ===")
    metrics = collector.report()

    # Step 7: Bayesian hyper‑parameter search (70% tuning split)
    print("\n=== PHASE: Bayesian Hyperparameter Search ===")
    best_params = run_search(
        attack_scenarios_for_tuning,
        ground_truths_for_tuning,
        batch_factory=lambda: BatchTrustTrace(logger, scanner, drift, memory),
        n_trials=n_trials,
    )
    print("Best params written to config.yaml")

    # Step 8: Held‑out evaluation (30%)
    print("\n=== PHASE: Final Evaluation (held-out 30%) ===")
    split = int(len(attack_scenarios_for_tuning) * 0.7)
    held_out = attack_scenarios_for_tuning[split:]
    held_truths = ground_truths_for_tuning[split:]
    held_collector = MetricsCollector()
    for scenario, gt in zip(held_out, held_truths):
        result = batch.run_attack_scenario(scenario, enable_recovery=True)
        held_collector.record_attack_with_meta(
            result["succeeded"], result["detected"],
            scenario.get("source", "handcrafted"), scenario["type"],
        )
        if result.get("patient_zero"):
            held_collector.record_patient_zero(result["patient_zero"], gt)
    held_metrics = held_collector.report()

    return {
        "baseline": baseline,
        "combined_metrics": metrics,
        "held_out_metrics": held_metrics,
        "best_params": best_params,
        "n_scenarios": len(scenarios),
    }
