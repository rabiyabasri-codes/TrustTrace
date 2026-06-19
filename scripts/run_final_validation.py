"""Run full validation and print final report."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from experiment.full_pipeline import run_full_experiment


def main():
    config.EXPERIMENT_MODE = True
    print("Running TrustTrace final validation (quick mode)...")
    report = run_full_experiment(quick=True, force_calibrate=False)
    metrics = report["combined_metrics"]
    print("\nValidation complete.")
    print(f"  FPR: {metrics['FPR']:.2%}")
    print(f"  PZ Accuracy: {metrics['Patient_Zero_Accuracy']:.2%}")
    print(f"  Chroma size: {metrics['Chroma_Collection_Size_Mean']:.0f}")
    return 0 if metrics["FPR"] <= 0.20 and metrics["Chroma_Collection_Size_Mean"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
