import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr
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


def eval_one_rater(df_r, features):
    df_r = df_r.dropna(subset=[TARGET] + features).reset_index(drop=True)
    if len(df_r) < 10:
        return None

    train_df, test_df = train_test_split(df_r, test_size=0.20, random_state=42)
    X_tr = train_df[features].to_numpy(float)
    y_tr = train_df[TARGET].to_numpy(float)
    X_te = test_df[features].to_numpy(float)
    y_te = test_df[TARGET].to_numpy(float)

    rf = RandomForestRegressor(
        n_estimators=600, max_depth=16,
        min_samples_split=4, min_samples_leaf=3,
        max_features=0.7, bootstrap=True, max_samples=0.85,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    y_hat = rf.predict(X_te)

    pear  = float(pearsonr(y_hat, y_te)[0]) if len(y_te) > 1 else np.nan
    spear = float(spearmanr(y_hat, y_te)[0]) if len(y_te) > 1 else np.nan
    return pear, spear

def main():
    # Load rater file + features
    raters = pd.read_csv(RATER_FILE)
    feats  = pd.read_csv(FEATURE_FILE)

    # Normalize Painting IDs to str
    raters["Painting"] = raters["Painting"].astype(str)
    feats["Painting"]  = feats["Painting"].astype(str)

    # Merge features into raters
    df = raters.merge(feats[["Painting"] + FEATURES], on="Painting", how="left")

    rows = []
    for r in sorted(df["Subject"].dropna().unique()):
        res = eval_one_rater(df[df["Subject"] == r].copy(), FEATURES)
        if res is not None:
            pear, spear = res
            rows.append({"rater": r, "pearson": pear, "spearman": spear})

    if not rows:
        print("No usable raters.")
        return

    results = pd.DataFrame(rows).sort_values("rater").reset_index(drop=True)

    
    out_csv = CSV_DIR / f"all_raters_{TARGET}_rf_scores.csv"
    results.to_csv(out_csv, index=False)
    print("Saved CSV:", out_csv)


    avg_pear = results["pearson"].mean()
    avg_spear = results["spearman"].mean()
    print("\nAverage over raters:")
    print(f"pearson  {avg_pear:.6f}")
    print(f"spearman {avg_spear:.6f}")

    avg_csv = CSV_DIR / f"summary_avg_{TARGET}_rf.csv"
    pd.DataFrame([{"target": TARGET, "model": "rf",
                   "avg_pearson": avg_pear, "avg_spearman": avg_spear}]
                 ).to_csv(avg_csv, index=False)
    print("Saved summary:", avg_csv)

  
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results))
    width = 0.38
    ax.bar(x - width/2, results["pearson"], width, label="Pearson")
    ax.bar(x + width/2, results["spearman"], width, label="Spearman")
    ax.set_xticks(x)
    ax.set_xticklabels(results["rater"].astype(str), rotation=45, ha="right")
    ax.set_title(f"Within-Rater RF (Abstract) — Target: {TARGET}")
    ax.set_xlabel("Rater")
    ax.set_ylabel("Correlation")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    png_path = PLOT_DIR / f"all_raters_{TARGET}_rf_bar.png"
    fig.savefig(png_path, dpi=250)
    print("Saved plots:", png_path)

if __name__ == "__main__":
    main()
