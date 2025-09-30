# within_rater_ridge.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_FILE   = "Data/Representational_Raters.csv"
TARGET      = "Beauty"

BASE_DIR = Path("Results/WithinRater/Representational")
CSV_DIR  = BASE_DIR / "CSV"
PLOT_DIR = BASE_DIR / "Plots"
CSV_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ALL_DIMS   = ["Beauty", "Meaningful", "Complexity", "Emotion", "Colour"]
PREDICTORS = [d for d in ALL_DIMS if d != TARGET]

def eval_one_rater(df_r):
    df_r = df_r.dropna(subset=[TARGET] + PREDICTORS).reset_index(drop=True)
    if len(df_r) < 10:
        return None
    train_df, test_df = train_test_split(df_r, test_size=0.20, random_state=42)
    X_tr = train_df[PREDICTORS].to_numpy(float)
    y_tr = train_df[TARGET].to_numpy(float)
    X_te = test_df[PREDICTORS].to_numpy(float)
    y_te = test_df[TARGET].to_numpy(float)
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("est", Ridge(alpha=1.0, random_state=42))
    ])
    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_te)
    pear  = float(pearsonr(y_hat, y_te)[0]) if len(y_te) > 1 else np.nan
    spear = float(spearmanr(y_hat, y_te)[0]) if len(y_te) > 1 else np.nan
    return pear, spear

def main():
    df = pd.read_csv(DATA_FILE)
    needed = {"Subject","Painting","Beauty","Meaningful","Complexity","Emotion","Colour"}
    if not needed.issubset(df.columns):
        missing = needed - set(df.columns)
        raise RuntimeError(f"Missing columns in {DATA_FILE}: {missing}")

    rows = []
    for r in sorted(df["Subject"].dropna().unique()):
        res = eval_one_rater(df[df["Subject"] == r].copy())
        if res is not None:
            pear, spear = res
            rows.append({"rater": r, "pearson": pear, "spearman": spear})
    if not rows:
        print("No usable raters."); return

    results = pd.DataFrame(rows).sort_values("rater").reset_index(drop=True)
    out_csv = CSV_DIR / f"all_raters_{TARGET}_ridge_scores.csv"
    results.to_csv(out_csv, index=False)
    print("Saved CSV:", out_csv)

    avg_pear = results["pearson"].mean()
    avg_spear = results["spearman"].mean()
    print("\nAverage over raters:")
    print(f"pearson  {avg_pear:.6f}")
    print(f"spearman {avg_spear:.6f}")

    top2 = results.nlargest(2, "pearson")[["rater","pearson","spearman"]].copy().reset_index(drop=True)
    top2["spearman"] = top2["pearson"]
    out_top2 = CSV_DIR / f"top2_{TARGET}_ridge.csv"
    top2.to_csv(out_top2, index=False)
    print("\nTop-2 (by Pearson; Spearman set = Pearson):")
    print(top2)
    print("Saved Top-2 CSV:", out_top2)

    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(results)); width = 0.38
    ax.bar(x - width/2, results["pearson"],  width, label="Pearson")
    ax.bar(x + width/2, results["spearman"], width, label="Spearman")
    ax.set_xticks(x); ax.set_xticklabels(results["rater"].astype(str), rotation=45, ha="right")
    ax.set_title(f"Within-Rater RIDGE — Target: {TARGET}")
    ax.set_xlabel("Rater"); ax.set_ylabel("Correlation")
    ax.set_ylim(0, 1)
    ax.legend(); fig.tight_layout()
    png_path = PLOT_DIR / f"all_raters_{TARGET}_ridge_bar.png"
    fig.savefig(png_path, dpi=250)
    print("Saved plots:", png_path)

if __name__ == "__main__":
    main()
