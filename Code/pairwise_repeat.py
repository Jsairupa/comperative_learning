import pandas as pd
import numpy as np
from itertools import permutations
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Subtract
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Data/Abstract_Data.csv")

# Filter only representational paintings
data = data[data["Representational"] == 0].copy()

# Define predictors and target variable
objective_predictors = ["HueSD", "SaturationSD", "Brightness", "BrightnessSD", 
                        "ColourComponent", "Entropy", "StraightEdgeDensity", "NonStraightEdgeDensity",
                        "Horizontal_Symmetry", "Vertical_Symmetry"]

# Remove rows with non-numeric values
data.replace("#NULL!", np.nan, inplace=True)
data.dropna(inplace=True)

target = "Liking_M"  # Using Liking_M as the preference measure

num_repeats = 10
results = []
pairwise_data_all = []

for num_pairs in range(1, 16):  # Iterate over n from 1 to 15
    pearson_list = []
    spearman_list = []
    adj_r2_list = []
    mae_list = []
    mse_list = []
    
    for _ in range(num_repeats):  # Repeat the experiment 10 times
        pairwise_data = []
        for idx1, row1 in data.iterrows():
            sampled_rows = data.sample(n=num_pairs, replace=False)  # Sample fewer comparisons per row
            for _, row2 in sampled_rows.iterrows():
                pair_info = f"{row1['Painting']},{row2['Painting']}"
                pairwise_data.append((row1[objective_predictors].values, row2[objective_predictors].values, 
                                      1 if row1[target] > row2[target] else -1, pair_info))

        # Convert to DataFrame
        pairwise_df = pd.DataFrame(pairwise_data, columns=["X1", "X2", "Preference", "Pair"])
        pairwise_data_all.append(pairwise_df)
        
        # Standardize features
        scaler = StandardScaler()
        X1 = np.vstack(pairwise_df["X1"].values)
        X2 = np.vstack(pairwise_df["X2"].values)
        X1 = scaler.fit_transform(X1)
        X2 = scaler.transform(X2)
        y = pairwise_df["Preference"].values.astype(float)
        
        # Split into train and test sets
        X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2)
        
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
        model.compile(optimizer=SGD(learning_rate=0.005, momentum=0.98, nesterov=True), loss='hinge', metrics=['accuracy'])
        
        # Early Stopping with stricter patience
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
        
        # Train the model
        model.fit([X1_train, X2_train], y_train, epochs=200, batch_size=64, validation_data=([X1_test, X2_test], y_test), 
                  verbose=0, callbacks=[early_stopping])
        
        # Evaluate the model
        y_pred_proba = model.predict([X1_test, X2_test]).flatten()
        pearson_corr, _ = pearsonr(y_test, y_pred_proba)
        spearman_corr, _ = spearmanr(y_test, y_pred_proba)
        r2 = r2_score(y_test, y_pred_proba)
        adj_r2 = 1 - ((1 - r2) * (len(y_test) - 1) / (len(y_test) - X1.shape[1] - 1))
        mae = mean_absolute_error(y_test, y_pred_proba)
        mse = mean_squared_error(y_test, y_pred_proba)
        
        pearson_list.append(pearson_corr)
        spearman_list.append(spearman_corr)
        adj_r2_list.append(adj_r2)
        mae_list.append(mae)
        mse_list.append(mse)
    
    # Store mean results
    results.append([num_pairs, np.mean(pearson_list), np.mean(spearman_list), np.mean(adj_r2_list), np.mean(mae_list), np.mean(mse_list)])

# Save results to CSV
results_df = pd.DataFrame(results, columns=["n_pairs", "Mean_Pearson", "Mean_Spearman", "Mean_Adj_R2", "Mean_MAE", "Mean_MSE"])
results_df.to_csv("n_vs_mean_correlation.csv", index=False)

# Save pairwise data to CSV
pairwise_all_df = pd.concat(pairwise_data_all, ignore_index=True)
pairwise_all_df.to_csv("pairwise_painting_data.csv", index=False)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(results_df["n_pairs"], results_df["Mean_Pearson"], marker='o', label="Mean Pearson Correlation")
plt.plot(results_df["n_pairs"], results_df["Mean_Spearman"], marker='s', label="Mean Spearman Correlation")
plt.plot(results_df["n_pairs"], results_df["Mean_Adj_R2"], marker='^', label="Mean Adjusted R2")
plt.xlabel("Number of Pairs (n)")
plt.ylabel("Metric Value")
plt.title("Elbow Point Analysis: Mean n vs Correlations & Adjusted R2")
plt.legend()
plt.grid()
plt.show()
