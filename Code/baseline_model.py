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

# Define predictors and target variables
objective_predictors = ["HueSD", "SaturationSD", "Brightness", "BrightnessSD", 
                        "ColourComponent", "Entropy", "StraightEdgeDensity", "NonStraightEdgeDensity",
                        "Horizontal_Symmetry", "Vertical_Symmetry"]

data.replace(["#NULL!", np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

targets = ["Beauty"]

# Standardize objective predictors
scaler = StandardScaler()
data[objective_predictors] = scaler.fit_transform(data[objective_predictors])

# Define stepwise regression function
def stepwise_selection(X, y, threshold_in=0.05, threshold_out=0.1):
    included = []
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        if excluded:
            new_pval = pd.Series(index=excluded, dtype=float)
            for col in excluded:
                model = sm.OLS(y, sm.add_constant(X[included + [col]])).fit()
                new_pval[col] = model.pvalues[col]
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                included.append(new_pval.idxmin())
                changed = True
        
        if included:
            model = sm.OLS(y, sm.add_constant(X[included])).fit()
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                included.remove(pvalues.idxmax())
                changed = True
        
        if not changed:
            break
    return included

# Run stepwise regression
X = data[objective_predictors]
y_beauty = data["Beauty"]
selected_features_beauty = stepwise_selection(X, y_beauty)

if not selected_features_beauty:
    print("No features selected by stepwise regression.")
else:
    final_model_beauty = sm.OLS(y_beauty, sm.add_constant(X[selected_features_beauty])).fit()
    
    # Save regression summary
    with open("results/regression_summary.txt", "w") as f:
        f.write(final_model_beauty.summary().as_text())
    
    # Calculate and save VIF values
    X_filtered = X[selected_features_beauty]
    if X_filtered.shape[1] > 1:
        vif_data = pd.DataFrame({
            "Feature": selected_features_beauty,
            "VIF": [variance_inflation_factor(X_filtered.values, i) for i in range(X_filtered.shape[1])]
        })
        vif_data.to_csv("results/vif_values.csv", index=False)
    
    # Compute and save correlation metrics
    adjusted_r2 = final_model_beauty.rsquared_adj
    bic = final_model_beauty.bic
    pearson_corr, pearson_pval = pearsonr(y_beauty, final_model_beauty.fittedvalues)
    spearman_corr, spearman_pval = spearmanr(y_beauty, final_model_beauty.fittedvalues)
    correlation_data = pd.DataFrame({
        "Metric": ["Pearson", "Spearman", "Adjusted R²", "BIC"],
        "Correlation": [pearson_corr, spearman_corr, adjusted_r2, bic],
        "P-Value": [pearson_pval, spearman_pval, None, None]
    })
    correlation_data.to_csv("results/correlation_metrics.csv", index=False)

                    #Disregard following code
"""
    # Train deep learning model
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_beauty, test_size=0.2, random_state=42)
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_filtered.shape[1],)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), verbose=1)

    # Evaluate and save deep learning results
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict(X_test).flatten()
    deep_pearson_corr, deep_pearson_pval = pearsonr(y_test, predictions)
    deep_spearman_corr, deep_spearman_pval = spearmanr(y_test, predictions)
    deep_adjusted_r2 = 1 - ((1 - np.corrcoef(y_test, predictions)[0, 1] ** 2) * (len(y_test) - 1) / (len(y_test) - X_filtered.shape[1] - 1))

    deep_results = pd.DataFrame({
        "Metric": ["MSE", "MAE", "Adjusted R²", "Pearson Correlation", "Spearman Correlation"],
        "Value": [test_loss, test_mae, deep_adjusted_r2, deep_pearson_corr, deep_spearman_corr],
        "P-Value": [None, None, None, deep_pearson_pval, deep_spearman_pval]
    })
    deep_results.to_csv("results/deep_learning_results.csv", index=False)

    print("Results saved in 'results' folder.")"""
