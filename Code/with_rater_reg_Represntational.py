import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

df = pd.read_csv("Data/Representational_Raters.csv")


rater_id = 1
df = df[df["Subject"] == rater_id].copy()

target = "Beauty"


features = ["Meaningful", "Complexity", "Emotion", "Colour"]

df = df.dropna(subset=[target] + features)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

X_train, y_train = train_df[features], train_df[target]
X_test,  y_test  = test_df[features],  test_df[target]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


model = RidgeCV(alphas=[0.1, 1.0, 10.0])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


pear = pearsonr(y_pred, y_test)[0]
spear = spearmanr(y_pred, y_test)[0]
r2 = r2_score(y_test, y_pred)

n = len(y_test)       
p = X_test.shape[1]    
adj_r2 = 1 - (1-r2) * (n-1) / (n-p-1) if n > p+1 else None

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)


print(f"Rater {rater_id} predicting {target}")
print("Features:", features)
print(f"Pearson    : {pear:.4f}")
print(f"Spearman   : {spear:.4f}")
print(f"MSE        : {mse:.4f}")
print(f"MAE        : {mae:.4f}")
