#!/usr/bin/env python3
"""
Scanner training runner - Trains injection detector separately.

Usage:
    python train_scanner.py

This script:
1. Checks for deepset dataset
2. Trains the injection scanner
3. Saves the model
"""

import os
import time
from scanner.injection_scanner import InjectionScanner

def main():
    print("=" * 70)
    print("🔍 TRUSTTRACE INJECTION SCANNER TRAINER")
    print("=" * 70)
    print()
    
    # Check dataset
    train_path = "data/deepset_injections/train.json"
    if not os.path.exists(train_path):
        print(f"✗ Dataset not found: {train_path}")
        print("Run: python data/download_deepset.py")
        return 1
    
    print(f"✓ Dataset found: {train_path}")
    print()
    
    # Train scanner
    try:
        print("Initializing scanner...")
        scanner = InjectionScanner()
        
        print("Training on deepset/prompt-injections...")
        start = time.time()
        scanner.train(train_path)
        elapsed = time.time() - start
        
        print()
        print("=" * 70)
        print(f"✓ SCANNER TRAINING COMPLETE ({elapsed:.1f}s)")
        print("=" * 70)
        print(f"Model saved to: scanner/scanner_model.pkl")
        print(f"Scaler saved to: scanner/scanner_scaler.pkl")
        print()
        print("Next: Run calibration or full pipeline")
        print("=" * 70)
        
        return 0
    
    except KeyboardInterrupt:
        print("\n✗ Training cancelled by user")
        return 1
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
