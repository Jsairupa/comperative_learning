import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

DATA_FILE   = "Data/Representational_Data.csv"
TRAIN_SPLIT = "Results/train_paintings.csv"  
TEST_SPLIT  = "Results/test_paintings.csv"

RESULTS_DIR = Path("Results/Regression")
CSV_DIR  = RESULTS_DIR / "CSV"
PLOT_DIR = RESULTS_DIR / "Plots"
CSV_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


TARGET = "Liking_M" 
FEATURES = [
    "HueSD","SaturationSD","Brightness","BrightnessSD",
    "ColourComponent","Entropy",
    "StraightEdgeDensity","NonStraightEdgeDensity",
    "Horizontal_Symmetry","Vertical_Symmetry",
]


raw = pd.read_csv(DATA_FILE)
train_items = pd.read_csv(TRAIN_SPLIT)
test_items  = pd.read_csv(TEST_SPLIT)


if "Representational" in raw.columns:
    raw = raw[raw["Representational"] == 1].copy()


id_col = "Painting" if "Painting" in raw.columns else ("item_id" if "item_id" in raw.columns else None)
if id_col is None:
    raise RuntimeError("Could not find an ID column ('Painting' or 'item_id') in Representational_Data.csv")

for df in (raw, train_items, test_items):
    if id_col not in df.columns:
        raise RuntimeError(f"ID column '{id_col}' not found in one of the split files. Check {TRAIN_SPLIT} / {TEST_SPLIT}.")
    df[id_col] = df[id_col].astype(str).str.strip()

needed_cols = [id_col, TARGET] + FEATURES
missing = set(needed_cols) - set(raw.columns)
if missing:
    raise RuntimeError(f"Missing columns in {DATA_FILE}: {missing}")
raw_small = raw[needed_cols].copy()

train_df = train_items[[id_col]].merge(raw_small, on=id_col, how="left").dropna(subset=[TARGET] + FEATURES)
test_df  = test_items[[id_col]].merge(raw_small, on=id_col, how="left").dropna(subset=[TARGET] + FEATURES)

if len(train_df) == 0 or len(test_df) == 0:
    raise RuntimeError(f"After merging, empty train/test. Train n={len(train_df)}, Test n={len(test_df)}. "
                       f"Check that your split IDs match the Painting IDs in {DATA_FILE}.")

X_train = train_df[FEATURES].to_numpy(float)
y_train = train_df[TARGET].to_numpy(float)
X_test  = test_df[FEATURES].to_numpy(float)
y_test  = test_df[TARGET].to_numpy(float)

print(f"Train n={len(y_train)} | Test n={len(y_test)} | Features={len(FEATURES)}")

model = GradientBoostingRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.9,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

pear   = pearsonr(y_pred, y_test)[0]
spear  = spearmanr(y_pred, y_test)[0]
r2     = r2_score(y_test, y_pred)
n, p   = len(y_test), X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
mse    = mean_squared_error(y_test, y_pred)
mae    = mean_absolute_error(y_test, y_pred)

print(f"[GRBT] Target={TARGET}")
print(f"Pearson : {pear:.4f}")
print(f"Spearman: {spear:.4f}")
print(f"R²      : {r2:.4f}")
print(f"Adj R²  : {adj_r2:.4f}")
print(f"MSE     : {mse:.4f}")
print(f"MAE     : {mae:.4f}")


metrics_df = pd.DataFrame([{
    "target": TARGET, "model": "GRBT",
    "pearson": pear, "spearman": spear,
    "r2": r2, "adj_r2": adj_r2,
    "mse": mse, "mae": mae
}])
metrics_path = CSV_DIR / f"GRBT_metrics_{TARGET}.csv"
metrics_df.to_csv(metrics_path, index=False)
print("Saved metrics CSV ->", metrics_path)


plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color="steelblue", edgecolors="k")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, 'r--', lw=2)
plt.xlim(lims); plt.ylim(lims)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title(f"True vs Predicted — GRBT ({TARGET})")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plot_path = PLOT_DIR / f"GRBT_true_vs_pred_{TARGET}.png"
plt.savefig(plot_path, dpi=300)
plt.close()
print("Saved plot ->", plot_path)
