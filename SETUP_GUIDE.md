# TrustTrace Setup & Execution Guide

## 🚀 Quick Start

### Step 1: Install Dependencies
```powershell
cd "c:\Users\Rabiya Basri\OneDrive\Desktop\TrustTrace"
.\.venv\Scripts\pip.exe install -r requirements.txt
```
**Estimated time:** 5-10 minutes (first run)

### Step 2: Download Datasets (Phase 1)

#### a) Download deepset/prompt-injections
```powershell
.\.venv\Scripts\python.exe data/download_deepset.py
```
**Output:** Creates `data/deepset_injections/train.json` and `data/deepset_injections/test.json`  
**Size:** ~2-5 MB  
**Time:** 1-2 minutes

#### b) Clone AgentDojo (Optional but Recommended)
```powershell
cd data
git clone https://github.com/ethz-spylab/agentdojo.git agentdojo
cd agentdojo
pip install -e .
cd ..
cd ..
```
**Size:** ~50 MB  
**Time:** 2-3 minutes  
**Purpose:** Realistic multi-agent attack scenarios (primary dataset)

### Step 3: Verify Installation

#### Test Import
```powershell
.\.venv\Scripts\python.exe -c "from victim_pipeline.agents import run_pipeline; print('✓ Imports OK')"
```

#### List Datasets
```powershell
ls data\
```
Expected output:
```
chroma_db/
deepset_injections/
download_deepset.py
agentdojo/  (if cloned)
```

---

## 📊 Execution Phases

### Phase 1: Scanner Training (5 minutes)
Trains the injection detector on deepset data
```powershell
.\.venv\Scripts\python.exe -c "
from scanner.injection_scanner import InjectionScanner
scanner = InjectionScanner()
scanner.train('data/deepset_injections/train.json')
print('✓ Scanner trained')
"
```

### Phase 2: Trust Calibration (10-15 minutes)
Collects 80 benign baseline runs
```powershell
.\.venv\Scripts\python.exe -c "
from logger.interaction_logger import InteractionLogger
from drift.behavioral_drift import BehavioralDriftModule
from calibration.calibrate import run_calibration

logger = InteractionLogger()
drift = BehavioralDriftModule()
run_calibration(logger, drift, n_runs=80)
print('✓ Calibration complete')
"
```

### Phase 3: Full Experimental Pipeline (20-30 minutes)
Main attack simulation & evaluation
```powershell
.\.venv\Scripts\python.exe main.py
```

---

## 🔍 Troubleshooting

### Issue: `ModuleNotFoundError`
**Solution:** Reinstall dependencies
```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt --upgrade
```

### Issue: `No such file or directory: data/deepset_injections/train.json`
**Solution:** Run dataset download
```powershell
.\.venv\Scripts\python.exe data/download_deepset.py
```

### Issue: Slow performance on first run
**Normal:** First run downloads ML models (~500 MB for sentence-transformers)

### Issue: OPENAI_API_KEY not set
**Normal:** Project uses mock LLM when key is missing  
To use real GPT-4o:
```powershell
$env:OPENAI_API_KEY = "YOUR_API_KEY_HERE"
```

---

## 📈 Expected Results

After full pipeline execution:

```
=== TrustTrace Performance Summary ===
Precision: 0.85-0.95
Recall: 0.80-0.95
F1-Score: 0.82-0.94
False Positive Rate: 0.05-0.15
Avg Recovery Time: 1.5-3.0s
Confusion Matrix: TP=22-28, FP=1-4, TN=8-9, FN=0-2
```

---

## 📁 Output Files

After execution, check:
- `logs/interactions.db` — All logged interactions
- `logs/reports/` — Recovery incident reports (JSON)
- `calibration/baselines/` — Baseline embeddings (per-agent)
- `scanner/scanner_model.pkl` — Trained injection scanner

---

## 🧪 Development Workflow

### Test a Single Module
```powershell
# Test trust engine
.\.venv\Scripts\python.exe -c "
from graph.propagation_graph import PropagationGraph
from drift.behavioral_drift import BehavioralDriftModule
from trust.trust_engine import TrustEngine

graph = PropagationGraph()
drift = BehavioralDriftModule()
engine = TrustEngine(graph, drift)
print(f'✓ Trust engine initialized')
print(f'  Agents: {engine.trust_scores}')
"
```

### Run Calibration Only
```powershell
# Useful for testing without attack scenarios
.\.venv\Scripts\python.exe -c "
from logger.interaction_logger import InteractionLogger
from drift.behavioral_drift import BehavioralDriftModule
from calibration.calibrate import run_calibration

logger = InteractionLogger()
drift = BehavioralDriftModule()
run_calibration(logger, drift, n_runs=10)  # Quick 10-run test
"
```

### Inspect Results
```powershell
# View metrics
$reports = Get-ChildItem logs\reports\ -Filter "*.json"
$reports | ForEach-Object { Write-Host $_.Name; Get-Content $_.FullName | ConvertFrom-Json }
```

---

## ⚙️ Configuration

Edit `config.yaml` to tune parameters:

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `delta` | 0.4 | [0.2, 0.7] | Compromise threshold |
| `k` | 3 | [1, 5] | Persistence window |
| `mu` | 0.3 | [0.1, 0.5] | Propagation influence |
| `rho` | 0.95 | [0.8, 0.98] | Trust retention |
| `eta` | 0.05 | [0.01, 0.15] | Recovery coefficient |
| `w_m` | 0.5 | [0.2, 0.8] | Message weight |
| `w_b` | 0.5 | [0.2, 0.8] | Behavior weight |

---

## 📚 Documentation Structure

- `IMPLEMENTATION_GUIDE.md` (from guide) — Full architecture & design
- `SETUP_GUIDE.md` (this file) — Setup & execution steps
- `config.yaml` — Runtime parameters
- `main.py` — Full pipeline orchestration
- Individual module docstrings — Detailed component docs

---

**Next:** Run `main.py` after dependencies install and datasets download.
