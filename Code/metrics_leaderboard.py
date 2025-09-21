import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ------------------ paths ------------------
BASE_DIR   = Path("Results/Regression")
CSV_DIR    = BASE_DIR / "CSV"
PLOT_DIR   = BASE_DIR / "Plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ load & combine ------------------
csv_files = sorted(CSV_DIR.glob("*_metrics_*.csv"))
if not csv_files:
    raise RuntimeError(f"No metrics CSVs found in {CSV_DIR}. "
                       "Run your model scripts (RF/GRBT/SVR/LASSO/RIDGE) first.")

rows = []
for f in csv_files:
    try:
        df = pd.read_csv(f)
        # Expect columns: target, model, pearson, spearman, r2, adj_r2, mse, mae
        missing = {"model","pearson","spearman"} - set(df.columns)
        if missing:
            print(f"[WARN] Skipping {f.name} (missing columns: {missing})")
            continue
        # take only the first row; your files have a single summary row
        row = df.iloc[0].to_dict()
        rows.append(row)
    except Exception as e:
        print(f"[WARN] Could not read {f.name}: {e}")

if not rows:
    raise RuntimeError("No valid metrics rows were found.")

metrics = pd.DataFrame(rows)

# Clean & sort
num_cols = ["pearson","spearman","r2","adj_r2","mse","mae"]
for c in num_cols:
    if c in metrics.columns:
        metrics[c] = pd.to_numeric(metrics[c], errors="coerce")

# order by pearson desc, then spearman desc
metrics_sorted = metrics.sort_values(
    by=[c for c in ["pearson","spearman"] if c in metrics.columns],
    ascending=[False, False]
).reset_index(drop=True)

# Save combined
out_csv = BASE_DIR / "combined_metrics.csv"
metrics_sorted.to_csv(out_csv, index=False)
print("Saved combined metrics ->", out_csv)

# ------------------ bar chart: Pearson & Spearman by model ------------------
if {"pearson","spearman","model"} <= set(metrics_sorted.columns):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(metrics_sorted))
    width = 0.38

    ax.bar(x - width/2, metrics_sorted["pearson"], width, label="Pearson")
    ax.bar(x + width/2, metrics_sorted["spearman"], width, label="Spearman")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_sorted["model"].astype(str), rotation=0, ha="center")
    title_target = ""
    if "target" in metrics_sorted.columns and metrics_sorted["target"].notna().any():
        # if all targets same, include it
        uniq_t = metrics_sorted["target"].dropna().unique().tolist()
        if len(uniq_t) == 1:
            title_target = f" — Target: {uniq_t[0]}"
    ax.set_title(f"Model Leaderboard: Pearson & Spearman{title_target}")
    ax.set_ylabel("Correlation")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    bar_path = PLOT_DIR / "leaderboard_bar_pearson_spearman.png"
    fig.savefig(bar_path, dpi=300)
    plt.close(fig)
    print("Saved bar plot ->", bar_path)
else:
    print("[INFO] Skipping bar plot (needed columns missing).")

# ------------------ scatter: Pearson vs Spearman ------------------
if {"pearson","spearman","model"} <= set(metrics_sorted.columns):
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(metrics_sorted["pearson"], metrics_sorted["spearman"], alpha=0.85, edgecolors="k")

    # annotate each point with the model name
    for _, r in metrics_sorted.iterrows():
        ax.annotate(str(r["model"]), (r["pearson"], r["spearman"]),
                    textcoords="offset points", xytext=(6,5), fontsize=9)

    ax.set_xlabel("Pearson Correlation")
    ax.set_ylabel("Spearman Correlation")
    ax.set_title("Model Comparison: Pearson vs Spearman")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    sc_path = PLOT_DIR / "leaderboard_scatter_pearson_vs_spearman.png"
    fig.savefig(sc_path, dpi=300)
    plt.close(fig)
    print("Saved scatter plot ->", sc_path)
else:
    print("[INFO] Skipping scatter plot (needed columns missing).")

print("\nDone. Open:")
print(" ", out_csv)
print(" ", bar_path if 'bar_path' in locals() else "(no bar plot)")
print(" ", sc_path  if 'sc_path'  in locals() else "(no scatter plot)")
