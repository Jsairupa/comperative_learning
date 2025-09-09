import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr, spearmanr

RESULTS_DIR = Path("Results")
DATA_DIR    = Path("Data")

# load train/test + raw data

train_items = pd.read_csv("Results/items_train_scaled.csv")
test_items  = pd.read_csv("Results/items_test_scaled.csv")
raw = pd.read_csv("Data/Representational_Data.csv")

# filter representational if column exists
if "Representational" in raw.columns:
    raw = raw[raw["Representational"] == 1].copy()
    
Objective_predictors = [
    "HueSD", "SaturationSD", "Brightness", "BrightnessSD",
    "ColourComponent", "Entropy",
    "StraightEdgeDensity", "NonStraightEdgeDensity",
    "Horizontal_Symmetry", "Vertical_Symmetry"
]

TARGET = "Liking_M"  



# id column
id_col = "Painting" if "Painting" in train_items.columns else "item_id"
train_items[id_col] = train_items[id_col].astype(str)
test_items[id_col]  = test_items[id_col].astype(str)
raw[id_col]         = raw[id_col].astype(str)

# merge target into train
train_df = train_items.merge(raw[[id_col, TARGET]], on=id_col, how="left")

X_train = train_df[Objective_predictors].to_numpy(float)
y_train = train_df[TARGET].to_numpy(float)

X_test = test_items[Objective_predictors].to_numpy(float)
y_test = test_items[TARGET].to_numpy(float)


model = RidgeCV(alphas=np.logspace(-3, 3, 13))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


pear = pearsonr(y_pred, y_test)[0]
spear = spearmanr(y_pred, y_test)[0]

print(f"[Ridge Regression] Pearson:  {pear:.4f}")
print(f"[Ridge Regression] Spearman: {spear:.4f}")


out = pd.DataFrame({
    id_col: test_items[id_col],
    "true": y_test,
    "pred": y_pred
})
out.to_csv(f"Results/predictions_ridge_{TARGET}.csv", index=False)

