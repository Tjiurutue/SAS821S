"""
06_simulation.py
-----------------
C7: Simulation - compare at least three control/triage scenarios using a
repeatable simulation, run >=1,000 iterations (Monte Carlo).

Scenarios:
  1. Manual/FIFO triage - analyst works alerts in arrival order, no scoring.
  2. Risk-scored triage - analyst always works the highest tp_probability
     alert next.
  3. Risk-scored + auto-suppress - low-risk alerts below a threshold are
     auto-suppressed (not queued for the analyst at all), analyst works the
     remaining risk-scored queue.

Each simulated "shift" has a fixed analyst capacity (alerts reviewable per
shift). We measure, per scenario, across many randomised shifts:
  - mean time-to-review for TRUE POSITIVE alerts (lower = better, in "alert
    positions" since arrival order is randomised per iteration)
  - % of true positives reviewed within the shift capacity
  - analyst hours spent on false positives (wasted effort)
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # T13_SME_Security/
PROC = str(ROOT / "02_data" / "processed")
OUT = str(ROOT / "08_outputs")
SEED = 821
rng = np.random.default_rng(SEED)

alerts = pd.read_csv(f"{PROC}/alerts_scored.csv")

SHIFT_CAPACITY = 40          # alerts an analyst can realistically review per shift
MINUTES_PER_ALERT = 6         # avg minutes to review one alert
SUPPRESS_THRESHOLD = 0.40     # tp_probability below this -> auto-suppressed in scenario 3
                               # (calibrated to the 25th-30th percentile of the scored
                               # alert population so suppression has a measurable effect)
N_ITERATIONS = 1500
SHIFT_SIZE = 60                # alerts arriving in a simulated shift window

def run_shift(shift_alerts, strategy):
    """Return (pct_TP_reviewed, mean_TP_review_position, wasted_minutes_on_FP)."""
    n = len(shift_alerts)
    if strategy == "manual_fifo":
        order = shift_alerts.sample(frac=1, random_state=None).reset_index(drop=True)  # arrival order = random
        queue = order
    elif strategy == "risk_scored":
        queue = shift_alerts.sort_values("tp_probability", ascending=False).reset_index(drop=True)
    elif strategy == "risk_scored_suppress":
        kept = shift_alerts[shift_alerts["tp_probability"] >= SUPPRESS_THRESHOLD]
        queue = kept.sort_values("tp_probability", ascending=False).reset_index(drop=True)
    else:
        raise ValueError(strategy)

    reviewed = queue.head(SHIFT_CAPACITY).reset_index(drop=True)
    reviewed["position"] = np.arange(1, len(reviewed) + 1)

    tp_total = (shift_alerts["label"] == "TP").sum()
    tp_reviewed = reviewed[reviewed["label"] == "TP"]
    pct_tp_reviewed = len(tp_reviewed) / tp_total if tp_total else np.nan
    mean_tp_position = tp_reviewed["position"].mean() if len(tp_reviewed) else np.nan
    wasted_minutes_fp = (reviewed["label"] == "FP").sum() * MINUTES_PER_ALERT

    return pct_tp_reviewed, mean_tp_position, wasted_minutes_fp

results = {s: {"pct_tp_reviewed": [], "mean_tp_position": [], "wasted_minutes_fp": []}
           for s in ["manual_fifo", "risk_scored", "risk_scored_suppress"]}

for i in range(N_ITERATIONS):
    shift_sample = alerts.sample(n=SHIFT_SIZE, random_state=int(rng.integers(0, 1_000_000)))
    for strat in results:
        pct, pos, waste = run_shift(shift_sample, strat)
        results[strat]["pct_tp_reviewed"].append(pct)
        results[strat]["mean_tp_position"].append(pos)
        results[strat]["wasted_minutes_fp"].append(waste)

summary_rows = []
for strat, vals in results.items():
    summary_rows.append({
        "strategy": strat,
        "mean_pct_tp_reviewed": round(np.nanmean(vals["pct_tp_reviewed"]) * 100, 1),
        "mean_tp_review_position": round(np.nanmean(vals["mean_tp_position"]), 2),
        "mean_wasted_minutes_on_fp": round(np.nanmean(vals["wasted_minutes_fp"]), 1),
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{OUT}/simulation_summary.csv", index=False)
print(f"=== C7: Simulation ({N_ITERATIONS} iterations, shift capacity={SHIFT_CAPACITY}) ===")
print(summary_df.to_string(index=False))

# Sensitivity test: vary shift capacity to see how strategy gap changes
sensitivity_rows = []
for cap in [20, 40, 60]:
    for strat in ["manual_fifo", "risk_scored", "risk_scored_suppress"]:
        pcts = []
        for i in range(300):
            shift_sample = alerts.sample(n=SHIFT_SIZE, random_state=int(rng.integers(0, 1_000_000)))
            SHIFT_CAPACITY_TMP = cap
            n = len(shift_sample)
            if strat == "manual_fifo":
                queue = shift_sample.sample(frac=1).reset_index(drop=True)
            elif strat == "risk_scored":
                queue = shift_sample.sort_values("tp_probability", ascending=False).reset_index(drop=True)
            else:
                kept = shift_sample[shift_sample["tp_probability"] >= SUPPRESS_THRESHOLD]
                queue = kept.sort_values("tp_probability", ascending=False).reset_index(drop=True)
            reviewed = queue.head(cap)
            tp_total = (shift_sample["label"] == "TP").sum()
            pct = (reviewed["label"] == "TP").sum() / tp_total if tp_total else np.nan
            pcts.append(pct)
        sensitivity_rows.append({"shift_capacity": cap, "strategy": strat, "mean_pct_tp_reviewed": round(np.nanmean(pcts)*100,1)})

sens_df = pd.DataFrame(sensitivity_rows)
sens_df.to_csv(f"{OUT}/simulation_sensitivity.csv", index=False)
print("\nSensitivity (shift capacity vs % TP reviewed):")
print(sens_df.pivot(index="shift_capacity", columns="strategy", values="mean_pct_tp_reviewed").to_string())

# chart
fig, ax = plt.subplots(figsize=(7,4))
for strat in ["manual_fifo","risk_scored","risk_scored_suppress"]:
    sub = sens_df[sens_df["strategy"]==strat]
    ax.plot(sub["shift_capacity"], sub["mean_pct_tp_reviewed"], marker="o", label=strat)
ax.set_xlabel("Analyst shift capacity (alerts/shift)")
ax.set_ylabel("% of true positives reviewed")
ax.set_title("Triage Strategy Comparison Across Shift Capacities")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/simulation_comparison_chart.png", dpi=150)
plt.close()

with open(f"{OUT}/simulation_assumptions.json", "w") as f:
    json.dump({
        "n_iterations": N_ITERATIONS,
        "shift_capacity_default": SHIFT_CAPACITY,
        "shift_size_alerts_arriving": SHIFT_SIZE,
        "minutes_per_alert_review": MINUTES_PER_ALERT,
        "suppress_threshold_tp_probability": SUPPRESS_THRESHOLD,
        "limitations": "Review time assumed constant per alert; does not model analyst fatigue, "
                        "interruptions, or partial reviews. Suppression threshold fixed rather than adaptive.",
    }, f, indent=2)

print(f"\nSaved simulation_summary.csv, simulation_sensitivity.csv, simulation_comparison_chart.png")
