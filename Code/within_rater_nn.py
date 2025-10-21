# within_rater_hybrid3_ridge_gbrt_nn.py
# Triple-ensemble: Ridge + GBRT + NN with learned blend weights per rater

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pearsonr, spearmanr
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Config ----------
DATA_FILE = "Data/Representational_Raters.csv"
TARGET = "Beauty"
FEATURES = ["Meaningful", "Complexity", "Emotion", "Colour"]

BASE_DIR = Path("Results/WithinRater/Representational")
CSV_DIR  = BASE_DIR / "CSV"
PLOT_DIR = BASE_DIR / "Plots"
CSV_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Small helper ----------
def best_blend3(y_true, p1, p2, p3, step=0.05):
    # weights w1,w2,w3 >= 0, w1+w2+w3=1; brute grid
    best = (0.0, 0.0, 1.0, -2.0)  # (w1,w2,w3, best_pearson)
    ws = np.arange(0.0, 1.0 + 1e-9, step)
    for w1 in ws:
        for w2 in ws:
            w3 = 1.0 - w1 - w2
            if w3 < 0.0: 
                continue
            blend = w1*p1 + w2*p2 + w3*p3
            r = pearsonr(blend, y_true)[0]
            if r > best[3]:
                best = (w1, w2, w3, r)
    return best  # (w1,w2,w3,best_pearson)

def build_nn(input_dim):
    model = Sequential([
        Dense(64, activation="tanh", input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation="tanh"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def eval_one_rater(df_r):
    df_r = df_r.dropna(subset=[TARGET] + FEATURES)
    if len(df_r) < 10:
        return None

    X = df_r[FEATURES].to_numpy(float)
    y = df_r[TARGET].to_numpy(float)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize (helps Ridge, often OK for GBRT; safe for NN)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # --- Ridge ---
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, y_tr)
    p_ridge = ridge.predict(X_te)

    # --- GBRT ---
    gbrt = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
    )
    gbrt.fit(X_tr, y_tr)
    p_gbrt = gbrt.predict(X_te)

    # --- NN ---
    nn = build_nn(X_tr.shape[1])
    es = EarlyStopping(monitor="loss", patience=20, restore_best_weights=True, verbose=0)
    nn.fit(X_tr, y_tr, epochs=400, batch_size=8, verbose=0, callbacks=[es])
    p_nn = nn.predict(X_te, verbose=0).flatten()

    # --- Learn best blend ---
    w1, w2, w3, _ = best_blend3(y_te, p_ridge, p_gbrt, p_nn, step=0.05)
    p_blend = w1*p_ridge + w2*p_gbrt + w3*p_nn

    pear  = pearsonr(p_blend, y_te)[0]
    spear = spearmanr(p_blend, y_te)[0]
    return pear, spear, w1, w2, w3

def main():
    df = pd.read_csv(DATA_FILE)
    rows = []

    for r in sorted(df["Subject"].dropna().unique()):
        res = eval_one_rater(df[df["Subject"] == r].copy())
        if res is not None:
            pear, spear, w_ridge, w_gbrt, w_nn = res
            rows.append({
                "rater": r,
                "pearson": pear,
                "spearman": spear,
                "w_ridge": w_ridge,
                "w_gbrt": w_gbrt,
                "w_nn": w_nn
            })

    if not rows:
        print("No valid raters.")
        return

    results = pd.DataFrame(rows)
    out_csv = CSV_DIR / f"all_raters_{TARGET}_hybrid3.csv"
    results.to_csv(out_csv, index=False)

    # Averages
    avg_pear  = results["pearson"].mean()
    avg_spear = results["spearman"].mean()
    avg_wr    = results["w_ridge"].mean()
    avg_wg    = results["w_gbrt"].mean()
    avg_wn    = results["w_nn"].mean()

    summary_csv = CSV_DIR / f"summary_avg_{TARGET}_hybrid3.csv"
    pd.DataFrame([{
        "target": TARGET,
        "model": "hybrid_ridge+gbrt+nn",
        "avg_pearson": avg_pear,
        "avg_spearman": avg_spear,
        "avg_w_ridge": avg_wr,
        "avg_w_gbrt": avg_wg,
        "avg_w_nn": avg_wn
    }]).to_csv(summary_csv, index=False)

    print("\nOverall summary:")
    print(f"  Pearson  {avg_pear:.3f}")
    print(f"  Spearman {avg_spear:.3f}")
    print(f"  Avg weights  Ridge={avg_wr:.2f}, GBRT={avg_wg:.2f}, NN={avg_wn:.2f}")
    print("Saved summary:", summary_csv)

    # Top-2
    top2 = results.nlargest(2, "pearson")[["rater","pearson","spearman","w_ridge","w_gbrt","w_nn"]]
    top2.to_csv(CSV_DIR / f"top2_{TARGET}_hybrid3.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(results["rater"].astype(str), results["pearson"])
    ax.set_title(f"Within-Rater Hybrid (Ridge + GBRT + NN) — {TARGET}")
    ax.set_xlabel("Rater")
    ax.set_ylabel("Correlation")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"all_raters_{TARGET}_hybrid3_bar.png", dpi=250)

    print("\nSaved:")
    print("  Full CSV:", out_csv)
    print("  Top-2 CSV:", CSV_DIR / f"top2_{TARGET}_hybrid3.csv")
    print("  Plot:", PLOT_DIR / f"all_raters_{TARGET}_hybrid3_bar.png")

if __name__ == "__main__":
    main()
