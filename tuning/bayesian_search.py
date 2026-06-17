"""
Bayesian hyperparameter search (Change 7).
Optuna TPE, 70/30 split, tunes trust-engine parameters.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional

import optuna
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_config(params: dict) -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.update(params)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)


def objective(
    trial: optuna.Trial,
    attack_scenarios: list,
    ground_truths: list,
    batch_factory: Callable,
) -> float:
    """Maximise Detection Rate − FPR."""
    params = {
        "lambda_direct": trial.suggest_float("lambda_direct", 0.4, 0.9),
        "lambda_indirect": trial.suggest_float("lambda_indirect", 0.35, 0.85),
        "lambda_memory": trial.suggest_float("lambda_memory", 0.3, 0.8),
        "w_m": trial.suggest_float("w_m", 0.3, 0.7),
        "w_b": trial.suggest_float("w_b", 0.3, 0.7),
        "rho": trial.suggest_float("rho", 0.8, 0.99),
        "eta": trial.suggest_float("eta", 0.01, 0.15),
        "mu": trial.suggest_float("mu", 0.1, 0.6),
        "delta": trial.suggest_float("delta", 0.2, 0.6),
        "k": trial.suggest_int("k", 1, 6),
        "persistence_window": trial.suggest_int("persistence_window", 1, 6),
    }
    params["persistence_window"] = params["k"]

    if params["w_m"] + params["w_b"] > 1.0:
        return -1.0

    _write_config(params)

    from eval.metrics import MetricsCollector
    collector = MetricsCollector()

    for scenario, _truth in zip(attack_scenarios, ground_truths):
        batch = batch_factory()
        result = batch.run_attack_scenario(scenario, enable_recovery=False)
        collector.record_attack(succeeded=result["succeeded"], detected=result["detected"])

    metrics = collector.compute()
    return metrics["Detection_Rate"] - metrics["FPR"]


def run_search(
    attack_scenarios: list,
    ground_truths: list,
    batch_factory: Callable,
    n_trials: int = 100,
) -> dict:
    """Run Bayesian search on 70% tuning split. Returns best parameters."""
    split = max(1, int(len(attack_scenarios) * 0.7))
    tune_scenarios = attack_scenarios[:split]
    tune_truths = ground_truths[:split]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective(trial, tune_scenarios, tune_truths, batch_factory),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    best["persistence_window"] = best.get("k", 3)
    print(f"\nBest params after {n_trials} trials:")
    for k, v in best.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    _write_config(best)
    return best


def run_tuning(scanner, drift, n_trials: int = 20) -> dict:
    """Backward-compatible wrapper used by legacy scripts."""
    from experiment.batch_processor import BatchTrustTrace
    from experiment.scenarios import ATTACK_SCENARIOS_HANDCRAFTED
    from logger.interaction_logger import InteractionLogger
    from memory.memory_manager import MemoryManager
    from victim_pipeline.agents import knowledge_base

    logger = InteractionLogger()
    memory = MemoryManager(collection=knowledge_base)
    scenarios = ATTACK_SCENARIOS_HANDCRAFTED
    truths = [s.get("ground_truth", "Retriever") for s in scenarios]

    def factory():
        return BatchTrustTrace(logger, scanner, drift, memory)

    return run_search(scenarios, truths, factory, n_trials=n_trials)
