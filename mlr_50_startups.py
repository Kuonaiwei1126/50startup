import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

os.makedirs("images", exist_ok=True)

# ---------- Data Preparation ----------
df = pd.read_csv("50_Startups.csv")
print(df.head())
print(df.isnull().sum())

df = pd.get_dummies(df, columns=["State"], dtype=int)   # One-Hot Encoding
X = df.drop(columns=["Profit"])
y = df["Profit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ---------- Forward Selection ----------
def evaluate(features):
    model = LinearRegression().fit(X_train[features], y_train)
    pred = model.predict(X_test[features])
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    return mse, rmse, r2

remaining = list(X.columns)
selected = []
records = []

while remaining:
    best = None
    for f in remaining:
        mse, rmse, r2 = evaluate(selected + [f])
        if best is None or rmse < best[2]:
            best = (f, mse, rmse, r2)
    f, mse, rmse, r2 = best
    selected.append(f)
    remaining.remove(f)
    records.append({
        "Number of Features": len(selected),
        "Selected Features": list(selected),
        "MSE": mse,
        "RMSE": rmse,
        "R-squared": r2,
    })

result = pd.DataFrame(records)
print(result.to_string(index=False))

# ---------- Top 3 ----------
top3 = result.loc[2, "Selected Features"]
print("Top 3 features:", top3)

final = LinearRegression().fit(X_train[top3], y_train)
print("Intercept:", final.intercept_)
print("Coefficients:", dict(zip(top3, final.coef_)))
print("Test MSE / RMSE / R2:", evaluate(top3))

# ---------- 貼圖：MSE / RMSE / R² by Number of Features ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
for a, col in zip(ax, ["MSE", "RMSE", "R-squared"]):
    a.plot(result["Number of Features"], result[col], marker="o")
    a.axvline(3, color="gray", linestyle="--", linewidth=1)
    a.set_title(f"{col} by Number of Features")
    a.set_xlabel("Number of Features"); a.set_ylabel(col)
plt.tight_layout()
plt.savefig("images/rmse_r2_by_features.png", dpi=150)
plt.show()

# 另存 HTML 表格，方便貼進報告
result.to_html("images/feature_selection_table.html", index=False)
