"""
Phase 2 dataset setup per Implementation Guide Section 3.

Phase 1 (required):
  - deepset/prompt-injections  (via data/download_deepset.py)
  - agentdojo                  (pip install agentdojo OR git clone)

Phase 2 (optional validation):
  - promptbench                (git clone)
  - jailbreakbench             (pip, downloads on first use)
"""

import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def setup_deepset():
    train = os.path.join(DATA, "deepset_injections", "train.json")
    if os.path.exists(train):
        print("deepset: already downloaded")
        return
    sys.path.insert(0, ROOT)
    from data.download_deepset import main as download_main
    download_main()


def setup_agentdojo():
    target = os.path.join(DATA, "agentdojo")
    if os.path.isdir(target):
        print("agentdojo: directory exists at data/agentdojo")
        return
    print("agentdojo: cloning repository...")
    os.makedirs(DATA, exist_ok=True)
    subprocess.run(
        ["git", "clone", "https://github.com/ethz-spylab/agentdojo.git", target],
        check=False,
    )
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", target],
            check=False,
        )
        print("agentdojo: installed")
    except Exception as exc:
        print(f"agentdojo: pip install -e failed ({exc}). Try: pip install agentdojo")


def setup_promptbench():
    target = os.path.join(DATA, "promptbench")
    if os.path.isdir(target):
        print("promptbench: directory exists")
        return
    print("promptbench: cloning repository...")
    os.makedirs(DATA, exist_ok=True)
    subprocess.run(
        ["git", "clone", "https://github.com/microsoft/promptbench.git", target],
        check=False,
    )


def setup_jailbreakbench():
    try:
        import jailbreakbench as jbb
        jbb.read_dataset()
        print("jailbreakbench: dataset ready")
    except Exception as exc:
        print(f"jailbreakbench: {exc}")


def main():
    print("=== TrustTrace Dataset Setup ===")
    setup_deepset()
    setup_agentdojo()
    setup_promptbench()
    setup_jailbreakbench()
    print("=== Setup complete ===")


if __name__ == "__main__":
    main()
