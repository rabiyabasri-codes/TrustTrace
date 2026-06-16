#!/usr/bin/env python3
"""
TrustTrace Setup & Initialization Script

This script handles:
1. Verifying environment setup
2. Downloading Phase 1 datasets (deepset)
3. Training the injection scanner
4. Running quick calibration test
"""

import os
import sys
import json
from pathlib import Path

def check_environment():
    """Verify Python environment is properly configured."""
    print("=" * 60)
    print("🔍 ENVIRONMENT CHECK")
    print("=" * 60)
    
    checks = {
        "Python version": sys.version.split()[0],
        "Working directory": os.getcwd(),
        "Virtual env active": hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix),
    }
    
    for check, result in checks.items():
        status = "✓" if result else "✗"
        print(f"{status} {check}: {result}")
    
    return True


def check_dependencies():
    """Verify required packages are installed."""
    print("\n" + "=" * 60)
    print("📦 DEPENDENCY CHECK")
    print("=" * 60)
    
    required_packages = [
        "crewai",
        "langchain",
        "chromadb",
        "networkx",
        "sentence_transformers",
        "sklearn",
        "optuna",
        "datasets",
        "openai",
        "numpy",
        "pandas",
        "yaml",
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (MISSING)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def check_project_structure():
    """Verify project folder structure."""
    print("\n" + "=" * 60)
    print("📁 PROJECT STRUCTURE CHECK")
    print("=" * 60)
    
    required_dirs = [
        "victim_pipeline",
        "logger",
        "scanner",
        "graph",
        "drift",
        "trust",
        "memory",
        "detector",
        "recovery",
        "calibration",
        "attacks",
        "eval",
        "tuning",
        "data",
        "logs",
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ (MISSING)")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\n⚠️  Missing directories: {', '.join(missing_dirs)}")
        return False
    
    print("\n✓ Project structure complete!")
    return True


def download_deepset_dataset():
    """Download deepset/prompt-injections dataset."""
    print("\n" + "=" * 60)
    print("📥 DOWNLOADING DEEPSET DATASET")
    print("=" * 60)
    
    train_path = Path("data/deepset_injections/train.json")
    test_path = Path("data/deepset_injections/test.json")
    
    if train_path.exists() and test_path.exists():
        train_size = train_path.stat().st_size / (1024 * 1024)
        test_size = test_path.stat().st_size / (1024 * 1024)
        print(f"✓ deepset dataset already exists!")
        print(f"  - train.json: {train_size:.1f} MB")
        print(f"  - test.json: {test_size:.1f} MB")
        return True
    
    try:
        from datasets import load_dataset
        print("Fetching from Hugging Face...")
        ds = load_dataset("deepset/prompt-injections")
        
        os.makedirs("data/deepset_injections", exist_ok=True)
        
        print(f"Saving {len(ds['train'])} training samples...")
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(list(ds["train"]), f, indent=2)
        
        print(f"Saving {len(ds['test'])} test samples...")
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(list(ds["test"]), f, indent=2)
        
        print(f"✓ Dataset downloaded successfully!")
        print(f"  - train.json: {len(ds['train'])} samples")
        print(f"  - test.json: {len(ds['test'])} samples")
        return True
    
    except ImportError:
        print("✗ 'datasets' package not installed")
        print("Run: pip install datasets")
        return False
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def verify_core_modules():
    """Verify core TrustTrace modules load correctly."""
    print("\n" + "=" * 60)
    print("✓ CORE MODULE VERIFICATION")
    print("=" * 60)
    
    modules = {
        "Victim Pipeline": "victim_pipeline.agents",
        "Logger": "logger.interaction_logger",
        "Scanner": "scanner.injection_scanner",
        "Graph": "graph.propagation_graph",
        "Drift": "drift.behavioral_drift",
        "Trust Engine": "trust.trust_engine",
        "Memory Manager": "memory.memory_manager",
        "Patient Zero": "detector.patient_zero",
        "Recovery": "recovery.recovery_manager",
        "Calibration": "calibration.calibrate",
        "Metrics": "eval.metrics",
        "Bayesian Search": "tuning.bayesian_search",
    }
    
    failed = []
    for name, module_path in modules.items():
        try:
            __import__(module_path)
            print(f"✓ {name}")
        except Exception as e:
            print(f"✗ {name}: {str(e)[:50]}")
            failed.append((name, e))
    
    if failed:
        print(f"\n⚠️  {len(failed)} modules failed to import")
        return False
    
    print("\n✓ All core modules imported successfully!")
    return True


def main():
    """Run all setup checks and initialization."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  TRUSTTRACE SETUP & INITIALIZATION SCRIPT  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = {
        "Environment": check_environment(),
        "Dependencies": check_dependencies(),
        "Structure": check_project_structure(),
        "Modules": verify_core_modules(),
        "Dataset": download_deepset_dataset(),
    }
    
    print("\n" + "=" * 60)
    print("📊 SETUP SUMMARY")
    print("=" * 60)
    
    for check, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ SETUP COMPLETE - Ready to run TrustTrace!")
        print("\nNext steps:")
        print("1. Train scanner: python -c \"from scanner.injection_scanner import InjectionScanner; s=InjectionScanner(); s.train()\"")
        print("2. Run calibration: python -c \"from calibration.calibrate import run_calibration; ...\"")
        print("3. Run pipeline: python main.py")
    else:
        print("✗ SETUP INCOMPLETE - Fix errors above before proceeding")
        return 1
    
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
