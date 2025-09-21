
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import random

RESULTS_DIR = Path("Results")  # outputs will be written here

NUM_SEED_PAIRS = 50
RANDOM_SEED = 42 

#Load dataset
df = pd.read_csv("Data/Representational_Data.csv")

if "Representational" in df.columns:
    df = df[df["Representational"] == 1].copy()
    
Objective_predictors = [
    "HueSD", "SaturationSD", "Brightness", "BrightnessSD",
    "ColourComponent", "Entropy",
    "StraightEdgeDensity", "NonStraightEdgeDensity",
    "Horizontal_Symmetry", "Vertical_Symmetry"
]


TARGET = ["Liking_M"]


if "Painting" in df.columns:
    id_col = "Painting"
elif "item_id" in df.columns:
    id_col = "item_id"
else:
    id_col = "item_id"
    df[id_col] = np.arange(len(df))


df = df.replace(["#NULL!", np.inf, -np.inf], np.nan)
df = df.dropna(subset=Objective_predictors).reset_index(drop=True)



TARGET = ["Liking_M"]


#Train/Test split
all_ids = df[id_col].astype(str).unique()
train_X, test_y = train_test_split(
    all_ids, test_size=0.20, random_state=42, shuffle=True
)
train_df = df[df[id_col].astype(str).isin(train_X)].copy()
test_df  = df[df[id_col].astype(str).isin(test_y)].copy()

#Scale features -on only trianing dataset
scaler = StandardScaler().fit(train_df[Objective_predictors].values)

train_df[Objective_predictors] = scaler.transform(train_df[Objective_predictors].values)
test_df[Objective_predictors]  = scaler.transform(test_df[Objective_predictors].values)



#Saving train/test CSV files
train_out = train_df[[id_col] + Objective_predictors].copy()
train_out[id_col] = train_out[id_col].astype(str)
train_out.to_csv("Results/train_paintings.csv", index=False)

present_TARGERT = [i for i in TARGET if i in test_df.columns]
test_out = test_df[[id_col] + Objective_predictors + present_TARGERT].copy()
test_out[id_col] = test_out[id_col].astype(str)
test_out.to_csv("Results/test_paintings.csv", index=False)

#For Generating the candidate pairs only on the training data
train_id_list = train_out[id_col].astype(str).tolist()
all_pairs = list(combinations(sorted(train_id_list), 2))

pd.DataFrame(all_pairs, columns=["painting1", "painting2"]).to_csv(
    "Results/train_candidate_pairs.csv", index=False
)

#Sample seed pairss
print(f"Sampling {NUM_SEED_PAIRS} seed pairs...")
rng = np.random.default_rng(RANDOM_SEED)
num_seed = min(NUM_SEED_PAIRS, len(all_pairs))
seed_idx = random.sample(range(len(all_pairs)), num_seed)


seed_pairs = pd.DataFrame([all_pairs[i] for i in seed_idx], columns=["painting1", "painting2"])
seed_pairs["pair_id"] = ["p" + str(i) for i in range(1, len(seed_pairs) + 1)]
seed_pairs["winner"] = ""  
seed_pairs.to_csv("Results/train_seed_pairs.csv", index=False)

#Results
print("Preprocessing complete. Files written:")
print("  Results/train_paintings.csv")
print("  Results/test_paintings.csv")
print("  Results/train_candidate_pairs.csv")
print("  Results/train_seed_pairs.csv")
