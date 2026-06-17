"""Verify TrustTrace implementation against the guide checklist."""

from __future__ import annotations

import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODULES = [
    "victim_pipeline.agents",
    "logger.interaction_logger",
    "scanner.injection_scanner",
    "graph.propagation_graph",
    "drift.behavioral_drift",
    "trust.trust_engine",
    "memory.memory_manager",
    "detector.patient_zero",
    "recovery.recovery_manager",
    "calibration.calibrate",
    "attacks.direct_injection",
    "attacks.indirect_injection",
    "attacks.memory_poisoning",
    "eval.metrics",
    "tuning.bayesian_search",
    "experiment.full_pipeline",
    "experiment.batch_processor",
    "experiment.scenarios",
    "irs.injection_risk",
    "runtime.trusttrace_runtime",
]

FILES = [
    "config.yaml",
    "config.defaults.yaml",
    "requirements.txt",
    "data/download_deepset.py",
    "data/setup_datasets.py",
    "data/deepset_injections/train.json",
    "scanner/scanner_model.pkl",
    "calibration/baselines/Retriever.json",
    "calibration/baselines/Planner.json",
    "calibration/baselines/Executor.json",
    "calibration/baselines/Generator.json",
]


def main():
    print("=== TrustTrace Compliance Check ===\n")
    ok = True

    print("Modules:")
    for mod in MODULES:
        try:
            importlib.import_module(mod)
            print(f"  [OK] {mod}")
        except Exception as exc:
            print(f"  [FAIL] {mod}: {exc}")
            ok = False

    print("\nFiles:")
    for path in FILES:
        full = os.path.join(ROOT, path)
        if os.path.exists(full):
            print(f"  [OK] {path}")
        else:
            print(f"  [MISSING] {path}")
            if path not in ("data/agentdojo", "data/promptbench"):
                ok = False

    print("\nGuide Section 17 entry points:")
    for flag, desc in [
        ("python main.py", "Interactive mode"),
        ("python main.py --validate", "Validation suite"),
        ("python main.py --experiment --quick", "Full pipeline (quick)"),
        ("python main.py --experiment", "Full pipeline (150+ scenarios)"),
        ("python main.py --restore-config", "Restore guide defaults"),
        ("python data/setup_datasets.py", "Dataset setup"),
    ]:
        print(f"  {desc}: {flag}")

    print(f"\nOverall: {'PASS' if ok else 'ISSUES FOUND'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
