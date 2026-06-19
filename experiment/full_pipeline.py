# Full 8-step experimental pipeline per TrustTrace Implementation Guide Section 17.

from __future__ import annotations

import os
import uuid

import yaml

import config
from calibration.calibrate import run_calibration
from drift.behavioral_drift import BehavioralDriftModule
from eval.metrics import MetricsCollector
from experiment.batch_processor import ATTACK_MARKERS, BatchTrustTrace, _attack_succeeded
from experiment.scenarios import GROUND_TRUTH_MAP, expand_scenarios, get_benign_queries
from logger.interaction_logger import InteractionLogger
from memory.memory_manager import MemoryManager
from scanner.injection_scanner import InjectionScanner
from tuning.bayesian_search import run_search
from victim_pipeline.agents import knowledge_base, run_pipeline
from attacks.direct_injection import inject_direct
from attacks.indirect_injection import inject_indirect, cleanup_indirect
from attacks.memory_poisoning import inject_memory_poison, cleanup_memory_poison

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
DEFAULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "config.defaults.yaml")


def _restore_config_defaults() -> None:
    with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
        defaults = yaml.safe_load(f)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(defaults, f, default_flow_style=False)


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
    summary["asr_pct"] = rate * 100
    print(f"Baseline ASR (no TrustTrace): {rate:.2%} ({summary['attacks_succeeded']}/{summary['total_attacks']})")
    return summary


def _print_final_summary(results: list, baseline_asr_pct: float) -> None:
    total = len(results)
    if total == 0:
        return
    detected = sum(1 for r in results if r.get("detected"))
    blocked_l1 = sum(1 for r in results if r.get("blocked_at") == "Layer1")
    missed = total - detected
    tt_asr_count = sum(1 for r in results if r.get("succeeded"))
    tt_asr = 100 * tt_asr_count / total
    goal_achieved_count = sum(1 for r in results if r.get("goal_achieved"))

    print("\n" + "=" * 50)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total attacks:          {total}")
    print(f"Detected (total):       {detected} ({100 * detected / total:.1f}%)")
    print(f"  Blocked at Layer 1:   {blocked_l1}")
    print(f"  Caught by TrustTrace: {detected - blocked_l1}")
    print(f"Missed:                 {missed} ({100 * missed / total:.1f}%)")
    print(f"Malicious goal achieved (raw): {goal_achieved_count} ({100 * goal_achieved_count / total:.1f}%)")
    print(f"Baseline ASR:           {baseline_asr_pct:.1f}%")
    print(f"TrustTrace ASR:         {tt_asr:.1f}%")
    print(f"ASR Reduction:          {baseline_asr_pct - tt_asr:.1f}%")
    print("=" * 50)


def _print_asr_audit_table(scenarios: list, attack_results: list) -> None:
    """Print per-attack ASR audit table and split summaries."""
    print("\n=== ASR Audit Table ===")
    header = (
        f"{'Attack ID':<10} | {'Type':<10} | {'Source':<12} | "
        f"{'Detected':<9} | {'Recovered':<10} | {'Goal Achieved':<14} | {'ASR Success':<11}"
    )
    print(header)
    print("-" * len(header))

    for i, (scenario, result) in enumerate(zip(scenarios, attack_results), start=1):
        attack_id = f"A{i:03d}"
        detected = result.get("detected", False)
        recovered = result.get("recovered", False)
        goal = result.get("goal_achieved", False)
        asr_success = result.get("succeeded", False)
        attack_type = scenario.get("type", result.get("attack_type", "unknown"))
        source = scenario.get("source", "handcrafted")
        print(
            f"{attack_id:<10} | {attack_type:<10} | {source:<12} | "
            f"{str(detected):<9} | {str(recovered):<10} | {str(goal):<14} | {str(asr_success):<11}"
        )

    for source in sorted({s.get("source", "handcrafted") for s in scenarios}):
        indices = [i for i, s in enumerate(scenarios) if s.get("source", "handcrafted") == source]
        subset = [attack_results[i] for i in indices]
        total = len(subset)
        if total == 0:
            continue
        asr_successes = sum(1 for r in subset if r.get("succeeded"))
        goals = sum(1 for r in subset if r.get("goal_achieved"))
        print(f"\n--- ASR by source: {source.upper()} ({total} attacks) ---")
        print(f"  Raw goal achieved:  {goals / total:.1%}")
        print(f"  TrustTrace ASR:     {asr_successes / total:.1%}")


def run_full_experiment(quick: bool = False, force_calibrate: bool = False, optuna_trials: int | None = None) -> dict:
    config.EXPERIMENT_MODE = True
    if quick:
        _restore_config_defaults()
    cfg = _load_config()
    n_per_type = 2 if quick else cfg.get("n_per_attack_type", 50)
    n_benign = 5 if quick else 50
    n_trials = 3 if quick else (optuna_trials if optuna_trials is not None else 100)

    print("=== TrustTrace Full Experimental Pipeline ===")

    try:
        from data.setup_datasets import setup_deepset
        setup_deepset()
    except Exception as exc:
        print(f"Dataset setup warning: {exc}")

    logger = InteractionLogger()
    scanner = InjectionScanner()
    drift = BehavioralDriftModule()
    memory = MemoryManager(collection=knowledge_base)
    batch = BatchTrustTrace(logger, scanner, drift, memory)

    model_path = os.path.join("scanner", "scanner_model.pkl")
    train_path = os.path.join("data", "deepset_injections", "train.json")
    if quick or not os.path.exists(model_path):
        print("\n=== PHASE: Train Injection Scanner ===")
        if not os.path.exists(train_path):
            from data.download_deepset import main as download_main
            download_main()
        scanner.train(train_path)
    else:
        print("Scanner model ready.")

    print("\n=== PHASE: Trust Calibration ===")
    n_cal = 15 if quick else cfg.get("calibration_runs", 80)
    need_cal = force_calibrate or not all(
        drift.has_baseline(a) for a in ["Retriever", "Planner", "Executor", "Generator"]
    )
    if need_cal:
        run_calibration(logger, drift, n_runs=n_cal)
    else:
        print(f"Calibration baselines ready ({n_cal} runs configured). Use --recalibrate to refresh.")

    baseline = run_baseline_experiments()
    baseline_asr_pct = baseline.get("asr_pct", 0.0)

    print("\n=== PHASE: TrustTrace Attack Experiments ===")
    scenarios = expand_scenarios(n_per_type=n_per_type)
    collector = MetricsCollector()
    attack_results: list = []

    for i, scenario in enumerate(scenarios):
        attack_type = scenario["type"]
        gt_agent = scenario.get("ground_truth") or GROUND_TRUTH_MAP.get(attack_type, "Retriever")
        if config.EXPERIMENT_MODE:
            print(f"  [{i + 1}/{len(scenarios)}]", end="")
        result = batch.run_attack_scenario(scenario, enable_recovery=True)
        attack_results.append(result)

        collector.record_attack_with_source(
            succeeded=result["succeeded"],
            detected=result["detected"],
            source=scenario.get("source", "handcrafted"),
            attack_type=attack_type,
            pz_predicted=result.get("patient_zero"),
            pz_ground_truth=gt_agent,
            recovery_time_s=result.get("recovery_time_s"),
            chroma_size=result.get("chroma_size", 0),
            retrieval_hits=result.get("retrieval_hits", 0),
        )
        if result.get("recovery_time_s"):
            collector.record_recovery(0, result["recovery_time_s"], 1, 1)

    _print_asr_audit_table(scenarios, attack_results)
    _print_final_summary(attack_results, baseline_asr_pct)

    print(f"\n=== PHASE: Benign Runs ({n_benign}) ===")
    benign_fps = []
    for i, query in enumerate(get_benign_queries(n_benign)):
        result = batch.run_benign(query)
        collector.record_benign(result["detected"], query=query)
        if result["detected"]:
            benign_fps.append(query)
        if (i + 1) % 10 == 0 or quick:
            print(f"  Benign {i + 1}/{n_benign}: flagged={result['detected']}")

    print("\n=== PHASE: Results ===")
    metrics = collector.report()
    collector.report_split()

    print("\n" + "=" * 50)
    print("FINAL VALIDATION REPORT")
    print("=" * 50)
    print(f"Detection Rate:           {metrics['Detection_Rate']:.2%}")
    print(f"ASR:                      {metrics['ASR']:.2%}")
    print(f"FPR:                      {metrics['FPR']:.2%}")
    print(f"Patient Zero Accuracy:    {metrics['Patient_Zero_Accuracy']:.2%}")
    print(f"Mean Recovery Time:       {metrics['Recovery_Time_Mean_s']:.3f}s")
    print(f"Median Recovery Time:     {metrics['Recovery_Time_Median_s']:.3f}s")
    print(f"Availability (Recovery):  {metrics['System_Availability']:.2%}")
    print(f"Chroma Collection Size:   {metrics['Chroma_Collection_Size_Mean']:.1f}")
    print(f"Retrieval Hits (mean):    {metrics['Retrieval_Hits_Mean']:.1f}")
    print(f"Benign False Positives:   {metrics['Benign_False_Positive_Count']}")
    if benign_fps:
        print("  Flagged benign queries:")
        for q in benign_fps[:5]:
            print(f"    - {q[:70]}")
    print("=" * 50)

    if not quick:
        print("\n=== PHASE: Bayesian Hyperparameter Search ===")
        best_params = run_search(
            scenarios,
            [s.get("ground_truth", "Retriever") for s in scenarios],
            batch_factory=lambda: BatchTrustTrace(logger, scanner, drift, memory),
            n_trials=n_trials,
        )
        print("Best params written to config.yaml")
    else:
        best_params = {}

    return {
        "baseline": baseline,
        "attack_results": attack_results,
        "combined_metrics": metrics,
        "best_params": best_params,
        "n_scenarios": len(scenarios),
        "benign_false_positives": benign_fps,
    }
