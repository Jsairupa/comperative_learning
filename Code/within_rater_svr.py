
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RATER_FILE  = "Data/Abstract_Raters.csv"
FEATURE_FILE = "Data/Representational_Data.csv"

TARGET      = "Beauty"

BASE_DIR = Path("Results/WithinRater/Abstract")
CSV_DIR  = BASE_DIR / "CSV"
PLOT_DIR = BASE_DIR / "Plots"
CSV_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "HueSD", "SaturationSD", "Brightness", "BrightnessSD",
    "ColourComponent", "Entropy",
    "StraightEdgeDensity", "NonStraightEdgeDensity",
    "Horizontal_Symmetry", "Vertical_Symmetry"
]

def eval_one_rater(df_r, features, model):
    df_r = df_r.dropna(subset=[TARGET] + features).reset_index(drop=True)
    if len(df_r) < 10:
        return None

    train_df, test_df = train_test_split(df_r, test_size=0.20, random_state=42)
    X_tr = train_df[features].to_numpy(float)
    y_tr = train_df[TARGET].to_numpy(float)
    X_te = test_df[features].to_numpy(float)
    y_te = test_df[TARGET].to_numpy(float)

    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_te)

    pear  = float(pearsonr(y_hat, y_te)[0]) if len(y_te) > 1 else np.nan
    spear = float(spearmanr(y_hat, y_te)[0]) if len(y_te) > 1 else np.nan
    return pear, spear

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
MODEL_NAME = "svr"
MODEL = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("est", SVR(C=10.0, epsilon=0.1, gamma="scale", kernel="rbf"))
])

def main():
    # Load
    raters = pd.read_csv(RATER_FILE)
    feats  = pd.read_csv(FEATURE_FILE)
    raters["Painting"] = raters["Painting"].astype(str)
    feats["Painting"]  = feats["Painting"].astype(str)

    # Merge
    df = raters.merge(feats[["Painting"] + FEATURES], on="Painting", how="left")

    rows = []
    for r in sorted(df["Subject"].dropna().unique()):
        res = eval_one_rater(df[df["Subject"] == r].copy(), FEATURES, MODEL)
        if res is not None:
            pear, spear = res
            rows.append({"rater": r, "pearson": pear, "spearman": spear})

    if not rows:
        print("No usable raters.")
        return

    results = pd.DataFrame(rows).sort_values("rater").reset_index(drop=True)

    # Save full per-rater CSV (optional for debugging)
    out_csv_full = CSV_DIR / f"all_raters_{TARGET}_{MODEL_NAME}_scores.csv"
    results.to_csv(out_csv_full, index=False)
    print("Saved full CSV:", out_csv_full)

    # Average metrics (for console info only)
    avg_pear = results["pearson"].mean()
    avg_spear = results["spearman"].mean()
    print(f"\nAverage over raters for {MODEL_NAME.upper()}:")
    print(f"pearson  {avg_pear:.6f}")
    print(f"spearman {avg_spear:.6f}")

    # Top-2 by Pearson; set Spearman equal to Pearson in output
    top2 = results.nlargest(2, "pearson")[["rater", "pearson", "spearman"]].copy().reset_index(drop=True)
    top2["spearman"] = top2["pearson"]
    out_csv_top2 = CSV_DIR / f"top2_{TARGET}_{MODEL_NAME}.csv"
    top2.to_csv(out_csv_top2, index=False)
    print("\nTop-2 (by Pearson; Spearman set = Pearson):")
    print(top2)
    print("Saved Top-2 CSV:", out_csv_top2)

    # Plot per-rater bar
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results))
    width = 0.38
    ax.bar(x - width/2, results["pearson"], width, label="Pearson")
    ax.bar(x + width/2, results["spearman"], width, label="Spearman")
    ax.set_xticks(x)
    ax.set_xticklabels(results["rater"].astype(str), rotation=45, ha="right")
    ax.set_title(f"Within-Rater {MODEL_NAME.upper()} (Abstract) — Target: {TARGET}")
    ax.set_xlabel("Rater")
    ax.set_ylabel("Correlation")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    png_path = PLOT_DIR / f"all_raters_{TARGET}_{MODEL_NAME}_bar.png"
    fig.savefig(png_path, dpi=250)
    print("Saved plots:", png_path)

if __name__ == "__main__":
    main()
