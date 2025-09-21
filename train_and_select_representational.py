

import numpy as np
import pandas as pd
import statsmodels.api as sm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr, spearmanr


# Load dataset
data = pd.read_csv("Data/Representational_Data.csv")

# Filter only representational paintings
data = data[data["Representational"] == 1].copy()

objective_predictors = ["HueSD", "SaturationSD", "Brightness", "BrightnessSD", 
                        "ColourComponent", "Entropy", "StraightEdgeDensity", "NonStraightEdgeDensity",
                        "Horizontal_Symmetry", "Vertical_Symmetry"]


data.replace(["#NULL!", np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

targets = ["Beauty"]

# 1) Standardize features (do this first)
scaler = StandardScaler()
data[objective_predictors] = scaler.fit_transform(data[objective_predictors])

# 2) (Optional but useful) save scaled items for later steps
from pathlib import Path
id_col = "Painting" if "Painting" in data.columns else "item_id"
if id_col not in data.columns:
    data[id_col] = np.arange(len(data))

items = data[[id_col] + objective_predictors].copy()
items.columns = [id_col] + [f"{c}_z" for c in objective_predictors]
Path("Data").mkdir(parents=True, exist_ok=True)
items.to_csv("Data/items_scaled_representational.csv", index=False)

# 3) Create 20 seed A–B pairs to hand-label (winner = 'i' or 'j')
from itertools import combinations
rng = np.random.default_rng(42)

ids = data[id_col].astype(str).tolist()
pairs = list(combinations(ids, 2))              # all unordered pairs
k = min(20, len(pairs))                         # change 20 if you want a different seed size
seed_idx = rng.permutation(len(pairs))[:k]

seed = pd.DataFrame([pairs[i] for i in seed_idx], columns=["i_id","j_id"])
seed.insert(0, "pair_id", [f"p{i:03d}" for i in range(1, len(seed)+1)])
seed["winner"] = ""                              # you will fill with 'i' or 'j'
seed.to_csv("Data/seed_pairs_for_labeling_representational.csv", index=False)

print("Wrote:",
      "Data/items_scaled_representational.csv,",
      "Data/seed_pairs_for_labeling_representational.csv")

