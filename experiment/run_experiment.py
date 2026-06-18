import datetime
import os
import yaml

import plotly.express as px
import plotly.io as pio

from experiment.full_pipeline import run_full_experiment

def main():
    # By default run the full experiment (not quick). Use env var QUICK=1 for a fast run.
    quick = os.getenv("QUICK", "0") == "1"
    # Optuna trial count: default 200 unless overridden by OPTUNA_TRIALS env var
    optuna_trials = int(os.getenv("OPTUNA_TRIALS", "200")) if not quick else 3
    print(f"Running TrustTrace experiment (quick={quick}, optuna_trials={optuna_trials}) ...")
    results = run_full_experiment(quick=quick, optuna_trials=optuna_trials)

    # Prepare a timestamped report filename
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, f"experiment_{ts}.md")

    # Simple Markdown report summarising key sections
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# TrustTrace Experimental Run Report\n\n")
        f.write(f"**Timestamp:** {ts}\n\n")
        f.write("## Summary of Results\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        # Baseline ASR
        baseline = results.get("baseline", {})
        asr = baseline.get("attacks_succeeded", 0) / baseline.get("total_attacks", 1)
        f.write(f"| Baseline ASR | {asr:.2%} |\n")
        # Combined metrics (already include percentages)
        combined = results.get("combined_metrics", {})
        for key, val in combined.items():
            if isinstance(val, float):
                f.write(f"| {key} | {val:.2%} |\n")
        # Held‑out metrics
        held = results.get("held_out_metrics", {})
        for key, val in held.items():
            if isinstance(val, float):
                f.write(f"| Held‑out {key} | {val:.2%} |\n")
        f.write("\n---\n\n")
        f.write("**Best hyper‑parameters** (written to `config.yaml`):\n\n")
        for k, v in results.get("best_params", {}).items():
            f.write(f"- {k}: {v}\n")
        f.write("\n---\n\n")
        f.write("*Run details:*\n")
        f.write(f"- Quick mode: {quick}\n")
        f.write(f"- Total scenarios: {results.get('n_scenarios', 'N/A')}\n")
        f.write("\nReport saved to: " + report_path + "\n")
        f.write("\n## Visualisations\n")
        # Prepare markdown with proper URIs
        figure_uri = figures_dir.replace('\\', '/')
        f.write(f"![Recovery Time Histogram](file:///{figure_uri}/recovery_hist.png)\n")
        f.write(f"![Patient Zero Accuracy](file:///{figure_uri}/pz_accuracy.png)\n")
        f.write(f"![Detection ROC](file:///{figure_uri}/roc_curve.png)\n")


        # Generate visualisations using Plotly
        recovery_times = results.get("combined_metrics", {}).get("recovery_times", [])
        if recovery_times:
            fig = px.histogram(recovery_times, nbins=20, title="Recovery Time Distribution (s)")
            fig.write_image(os.path.join(figures_dir, "recovery_hist.png"))
        pz_data = results.get("combined_metrics", {}).get("patient_zero_breakdown", {})
        if pz_data:
            fig = px.bar(x=list(pz_data.keys()), y=list(pz_data.values()), title="Patient Zero Accuracy by Attack Type")
            fig.write_image(os.path.join(figures_dir, "pz_accuracy.png"))
        roc = results.get("roc", {})
        if roc:
            fig = px.line(
                x=roc.get("fpr", []),
                y=roc.get("tpr", []),
                title="ROC Curve",
                labels={"x": "False Positive Rate", "y": "True Positive Rate"}
            )
            fig.write_image(os.path.join(figures_dir, "roc_curve.png"))


    print("Experiment complete. Report written to:", report_path)

if __name__ == "__main__":
    main()
