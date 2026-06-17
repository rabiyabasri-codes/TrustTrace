# TrustTrace Guide Compliance Status

Last verified: compliance check **PASS** (`python scripts/compliance_check.py`)

## Guide Section Status

| Section | Status | Notes |
|---------|--------|-------|
| §1 Environment Setup | ✅ | `requirements.txt`, `config.yaml`, `config.defaults.yaml` |
| §2 Folder Structure | ✅ | All core packages + `experiment/`, `irs/`, `runtime/`, `display/` |
| §3 Dataset Download | ✅ | `data/download_deepset.py`, `data/setup_datasets.py` |
| §4 Victim Pipeline | ✅ | 4 CrewAI agents, ChromaDB, stepwise + batch runners |
| §5 Interaction Logger (C1) | ✅ | SQLite, all event types |
| §6 Injection Scanner | ✅ | Trained on deepset, `scanner_model.pkl` |
| §7 Propagation Graph (C2) | ✅ | `w_AB`, backward traversal, patient zero timestamps |
| §8 Behavioral Drift (C3) | ✅ | BD equation, 4 agent baselines on disk |
| §9–10 Trust Engine (C9–C10) | ✅ | Bounding, k-consecutive compromise, propagation |
| §11 Memory Manager | ✅ | Write/read/rollback, memory trust MT |
| §12 Patient Zero Detector | ✅ | Graph-derived PZ = argmin(t_a) |
| §13 Recovery Manager | ✅ | Quarantine → rollback → restore + JSON reports |
| §14 Attacks (C5) | ✅ | Direct, indirect, memory poisoning |
| §15 Eval Metrics (C6) | ✅ | All 6 metrics in `eval/metrics.py` |
| §16 Bayesian Search (C7) | ✅ | `run_search()`, 70/30 split, Optuna TPE |
| §17 Full Pipeline | ✅ | `python main.py --experiment` (8 steps) |
| §18 Dependency Map | ✅ | Followed |

## Optional / Phase 2 (guide)

| Item | Status |
|------|--------|
| AgentDojo clone in `data/agentdojo/` | ⚠️ Optional — auto-loaded via pip if installed |
| PromptBench clone | ⚠️ Optional — `data/setup_datasets.py` |
| JailbreakBench in pipeline | ⚠️ Optional — setup script only |

## Run Commands

```powershell
# Compliance check
.venv\Scripts\python.exe scripts\compliance_check.py

# Interactive (dynamic user input)
.venv\Scripts\python.exe main.py

# 4-scenario validation
.venv\Scripts\python.exe main.py --validate

# Full guide pipeline — quick (6 attacks + 5 benign + 3 tuning trials)
.venv\Scripts\python.exe main.py --experiment --quick

# Full guide pipeline — paper scale (150+ attacks, 50 benign, 100 tuning trials)
.venv\Scripts\python.exe main.py --experiment

# Force recalibration (80 runs)
.venv\Scripts\python.exe main.py --experiment --recalibrate

# Restore guide config defaults
.venv\Scripts\python.exe main.py --restore-config

# Download Phase 1/2 datasets
.venv\Scripts\python.exe data\setup_datasets.py
```

## Section 17 Pipeline Steps

| Step | Implementation |
|------|----------------|
| 1. Initialise modules | `experiment/full_pipeline.py` |
| 2. Train scanner | Auto if `scanner_model.pkl` missing |
| 3. Calibration (80 runs) | `calibration/calibrate.py` |
| 4. Baseline ASR (no TrustTrace) | `run_baseline_experiments()` |
| 5. Attacks WITH TrustTrace | `BatchTrustTrace.run_attack_scenario()` |
| 6. Report 6 metrics | Hand-crafted / AgentDojo / Combined |
| 7. Bayesian search (70%) | `tuning/bayesian_search.run_search()` |
| 8. Held-out eval (30%) | Re-runs held-out scenarios |

## Extensions Beyond Guide

- `irs/injection_risk.py` — IRS, MS, Memory Trust equations
- `runtime/trusttrace_runtime.py` — Real-time monitoring + terminal display
- `display/terminal_display.py` — IEEE demo output
