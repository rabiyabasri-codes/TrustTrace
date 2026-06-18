# TrustTrace - Quick Start Guide

**Status**: All 16 core modules implemented. Ready for Phase 1 execution.

## 📋 What is TrustTrace?

TrustTrace is a real-time trust propagation monitoring and attack recovery system for multi-agent LLM pipelines. It detects compromised agents, identifies attack origin (Patient Zero), and executes recovery procedures.

---

## 🚀 Quick Start

### Step 1: Verify Setup
```powershell
# Run comprehensive environment check
.\.venv\Scripts\python.exe setup.py
```

This checks:
- ✓ Python environment
- ✓ Required dependencies
- ✓ Project structure
- ✓ Core module imports
- ✓ Deepset dataset availability

### Step 2: Download Deepset Dataset (Phase 1)

**Status**: Pip dependencies still installing in background...

Once dependencies complete:
```powershell
.\.venv\Scripts\python.exe data/download_deepset.py
```

This downloads ~4000 prompt injection samples for scanner training.

### Step 3: Train Injection Scanner

```powershell
.\.venv\Scripts\python.exe train_scanner.py
```

Output:
- `scanner/scanner_model.pkl` - Trained LogisticRegression model
- `scanner/scanner_scaler.pkl` - Feature scaler

### Step 4: Run Trust Calibration

**Quick test (10 runs)**:
```powershell
.\.venv\Scripts\python.exe run_calibration_quick.py 10
```

**Full calibration (80 runs - recommended)**:
```powershell
.\.venv\Scripts\python.exe run_calibration_quick.py 80
```

Output: Baseline embeddings for all agents saved

### Step 5: Run Full Attack Simulation

```powershell
.\.venv\Scripts\python.exe main.py
```

Executes full pipeline:
1. Benign runs (baseline collection)
2. Direct injection attacks
3. Indirect (RAG) injection attacks
4. Memory poisoning attacks
5. Attack detection & metrics calculation
6. Hyperparameter optimization

Duration: ~20-30 minutes

### Step 6: Analyze Results

```powershell
.\.venv\Scripts\python.exe analyze_results.py
```

Displays:
- Interaction statistics
- Incident reports
- Performance metrics (Precision, Recall, F1, FPR)
- Recovery times

---

## 📁 Project Structure

```
TrustTrace/
├── victim_pipeline/          # 4-agent CrewAI pipeline (target system)
├── logger/                   # SQLite interaction logging middleware
├── scanner/                  # Injection detection (sentence-transformers)
├── graph/                    # Trust propagation graph (NetworkX)
├── drift/                    # Behavioral drift computation
├── trust/                    # Trust engine (core update equations)
├── memory/                   # ChromaDB memory manager with rollback
├── detector/                 # Patient Zero detector (graph traversal)
├── recovery/                 # Recovery manager (3-phase: quarantine→rollback→restore)
├── calibration/              # Baseline collection (80 clean runs)
├── attacks/                  # 3 attack types (direct, indirect, memory poison)
├── eval/                     # Metrics calculation (Precision, Recall, F1, FPR)
├── tuning/                   # Bayesian hyperparameter optimization (Optuna)
├── data/                     # Datasets directory
├── logs/                     # Runtime logs & incident reports
├── main.py                   # Full pipeline orchestrator
├── config.yaml               # 13 tunable hyperparameters
├── requirements.txt          # Python dependencies
└── [NEW] Utility Scripts:
    ├── setup.py                    # Environment verification
    ├── train_scanner.py            # Scanner training wrapper
    ├── run_calibration_quick.py    # Calibration runner
    └── analyze_results.py          # Results analyzer
```

---

## ⚙️ Configuration

Edit `config.yaml` to adjust hyperparameters:

```yaml
# Trust engine parameters
delta: 0.4              # Trust threshold for compromise detection
k: 3                    # Consecutive low-trust violations trigger alert
mu: 0.3                 # Memory degradation rate
rho: 0.95               # Trust decay coefficient
eta: 0.05               # Trust recovery rate
w_m: 0.5                # Injection weight
w_b: 0.5                # Behavioral drift weight
```

---

## 🔍 What Gets Executed

### Main Pipeline (main.py)

1. **Calibration Phase** (80 runs)
   - Collects baseline embeddings for all agents
   - Trains behavioral drift module

2. **Benign Runs** (20 runs)
   - Normal operation without attacks
   - Validates system baseline

3. **Direct Injection Attacks** (6 variants)
   - Prompt injection targeting Planner agent
   - Tests detection latency

4. **Indirect (RAG) Injection Attacks** (4 variants)
   - Malicious documents in knowledge base
   - Tests memory contamination detection

5. **Memory Poisoning Attacks** (4 variants)
   - MINJA/MemoryGraft-style attacks
   - Persistent memory corruption

6. **Metrics Calculation**
   - Precision, Recall, F1-Score
   - False Positive Rate
   - Recovery Time
   - System Availability

7. **Bayesian Optimization** (Optuna)
   - Hyperparameter tuning
   - Optimization across all 7 parameters

---

## 📊 Key Outputs

### During Execution
- `logs/interactions.db` - SQLite database with all inter-agent events
- `logs/reports/incident_*.json` - Recovery incident details
- `calibration/baselines/*.json` - Baseline embeddings per agent
- `scanner/scanner_model.pkl` - Trained injection detector

### After Execution
- `logs/metrics_final.json` - Final performance metrics
- `logs/optimized_params.yaml` - Best hyperparameters found
- Console output - Real-time progress & attack summaries

---

## 🛠️ Troubleshooting

### "datasets" module not found
```powershell
# Wait for pip install to complete, then:
.\.venv\Scripts\pip.exe install datasets --quiet
```

### Memory issues during training
```powershell
# Reduce calibration runs for testing
.\.venv\Scripts\python.exe run_calibration_quick.py 20
```

### OPENAI_API_KEY not set
- System runs with MockChatOpenAI (deterministic mode)
- Results are valid but not calling real LLM
- Set env var to use real GPT-4o:
```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY_HERE"
```

### Port conflicts (ChromaDB/vector store)
```python
# Modify in memory/memory_manager.py if needed:
client = chromadb.HttpClient(host="localhost", port=8001)
```

---

## 📝 Next Steps

1. **Wait for pip install to complete** (currently running in background)
2. **Run `setup.py`** to verify all components
3. **Follow Quick Start steps** 1-6 above
4. **View results** with `analyze_results.py`

---

## 📖 Implementation Guide Reference

This project follows the comprehensive 18-section implementation guide:
1. Environment Setup ✓
2. Project Structure ✓
3. Victim Pipeline ✓
4. Logger ✓
5. Scanner ✓
6. Graph ✓
7. Drift ✓
8. Trust Engine ✓
9. Memory Manager ✓
10. Detector ✓
11. Recovery ✓
12. Calibration ✓
13. Attacks ✓
14. Evaluation ✓
15. Tuning ✓
16. Main Pipeline ✓
17. Execution ← **You are here**
18. Validation

---

## 🎯 Success Criteria

Pipeline is successful when:
- ✓ All 22 attack scenarios execute without errors
- ✓ Precision > 0.95 (low false positives)
- ✓ Recall > 0.90 (catches most attacks)
- ✓ Recovery Time < 5 seconds (average)
- ✓ Patient Zero identification = 100% accuracy
- ✓ Memory rollback removes all poisoned entries

---

**Created**: During Phase 1 (Dataset Download) initialization  
**Status**: Awaiting pip dependencies → Ready to execute
