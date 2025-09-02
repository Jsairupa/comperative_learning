import pandas as pd
import numpy as np
from itertools import permutations
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Subtract
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# Load dataset
data = pd.read_csv("Data/Abstract_Data.csv")

# Filter only representational paintings
data = data[data["Representational"] == 0].copy()

# Define predictors and target variable
objective_predictors = ["HueSD", "SaturationSD", "Brightness", "BrightnessSD", 
                        "ColourComponent", "Entropy", "StraightEdgeDensity", "NonStraightEdgeDensity",
                        "Horizontal_Symmetry", "Vertical_Symmetry"]

data.replace(["#NULL!", np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

target = "Liking_M"  # Using Liking_M as the preference measure

pairwise_data = []
num_pairs = 5  # Start with n * (1)
for idx1, row1 in data.iterrows():
    sampled_rows = data.sample(n=num_pairs, replace=False, random_state=42)  # Sample fewer comparisons per row
    for _, row2 in sampled_rows.iterrows():
        pairwise_data.append((row1[objective_predictors].values, row2[objective_predictors].values, 
                              1 if row1[target] > row2[target] else -1, f"{row1['Painting']},{row2['Painting']}"))

# Convert to DataFrame
pairwise_df = pd.DataFrame(pairwise_data, columns=["X1", "X2", "Preference", "Pair"])

# Ensure Preference is numeric
pairwise_df["Preference"] = pairwise_df["Preference"].astype(float)
pairwise_df["Pair"] = pairwise_df["Pair"].astype(str)  # Explicitly cast Pair to string

# Save pairwise dataset to CSV
pairwise_df.to_csv("Data/pairwise_painting_data.csv", index=False)

# Standardize features
scaler = StandardScaler()
X1 = np.vstack(pairwise_df["X1"].values)
X2 = np.vstack(pairwise_df["X2"].values)
X1 = scaler.fit_transform(X1)
X2 = scaler.transform(X2)
y = pairwise_df["Preference"].values.astype(float)

# Split into train and test sets
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

# Define encoder (shared network)
def build_encoder(input_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(1024, activation='relu', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    return Model(inputs, x)

# Build encoder
encoder = build_encoder(X1.shape[1])

# Define inputs
input_A = Input(shape=(X1.shape[1],))
input_B = Input(shape=(X2.shape[1],))

# Encode inputs
encoded_A = encoder(input_A)
encoded_B = encoder(input_B)

# Compute difference
diff = Subtract()([encoded_A, encoded_B])
output = Dense(1, activation='tanh')(diff)

# Define model
model = Model(inputs=[input_A, input_B], outputs=output)

# Compile model with SGD optimizer
model.compile(optimizer=SGD(learning_rate=0.005, momentum=0.98, nesterov=True), loss='hinge', metrics=['mae'])

# Early Stopping with stricter patience
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

# Train the model with increased batch size
history = model.fit([X1_train, X2_train], y_train, epochs=200, batch_size=64, validation_data=([X1_test, X2_test], y_test), 
                    verbose=1, callbacks=[early_stopping])

# Evaluate the model
test_loss = model.evaluate([X1_test, X2_test], y_test, verbose=0)
y_pred_proba = model.predict([X1_test, X2_test]).flatten()
y_pred = np.sign(y_pred_proba)  # Convert to -1 or 1
roc_auc = roc_auc_score(y_test, y_pred_proba)
r2 = r2_score(y_test, y_pred_proba)
adj_r2 = 1 - ((1 - r2) * (len(y_test) - 1) / (len(y_test) - X1.shape[1] - 1))

# Calculate additional metrics
mae = mean_absolute_error(y_test, y_pred_proba)
mse = mean_squared_error(y_test, y_pred_proba)
pearson_corr, _ = pearsonr(y_test, y_pred_proba)
spearman_corr, _ = spearmanr(y_test, y_pred_proba)

print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Pearson Correlation: {pearson_corr:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f}")
print(f"r2: {r2:.4f}")
print(f"adj_r2: {adj_r2:.4f}")