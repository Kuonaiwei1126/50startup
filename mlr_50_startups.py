import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score

os.makedirs("images", exist_ok=True)

# ---------- Data Preparation ----------
df = pd.read_csv("50_Startups.csv")
print(df.head())
print(df.isnull().sum())

# One-Hot Encoding，drop_first 避免 dummy variable trap（California 為基準）→ 5 個候選特徵
df = pd.get_dummies(df, columns=["State"], drop_first=True, dtype=int)
X = df.drop(columns=["Profit"])
y = df["Profit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

def evaluate(features):
    model = LinearRegression().fit(X_train[features], y_train)
    pred = model.predict(X_test[features])
    mse = mean_squared_error(y_test, pred)
    return mse, np.sqrt(mse), r2_score(y_test, pred)

def make_table(scheme, feature_sets):
    rows = []
    for k, features in enumerate(feature_sets, start=1):
        mse, rmse, r2 = evaluate(features)
        rows.append({"Scheme": scheme, "Number of Features": k,
                     "Selected Features": list(features),
                     "MSE": mse, "RMSE": rmse, "R-squared": r2})
    return pd.DataFrame(rows)

n = X.shape[1]

# ---------- Scheme 1: Forward Selection（每步加入使測試 RMSE 最小的特徵）----------
remaining, selected, forward_sets = list(X.columns), [], []
while remaining:
    best = min(remaining, key=lambda f: evaluate(selected + [f])[1])
    selected.append(best)
    remaining.remove(best)
    forward_sets.append(list(selected))

# ---------- Scheme 2: Backward Elimination（每步移除使測試 RMSE 最小的特徵）----------
current, backward_sets = list(X.columns), [list(X.columns)]
while len(current) > 1:
    worst = min(current, key=lambda f: evaluate([c for c in current if c != f])[1])
    current.remove(worst)
    backward_sets.insert(0, list(current))

# ---------- Scheme 3: RFE（依係數大小遞迴刪除，特徵先標準化）----------
X_std = (X_train - X_train.mean()) / X_train.std()
rfe_sets = []
for k in range(1, n + 1):
    rfe = RFE(LinearRegression(), n_features_to_select=k).fit(X_std, y_train)
    rfe_sets.append(list(X.columns[rfe.support_]))

# ---------- Scheme 4: SelectKBest（f_regression 單變量 F 檢定）----------
kbest_sets = []
for k in range(1, n + 1):
    kb = SelectKBest(f_regression, k=k).fit(X_train, y_train)
    kbest_sets.append(list(X.columns[kb.get_support()]))

tables = {
    "Forward Selection": make_table("Forward Selection", forward_sets),
    "Backward Elimination": make_table("Backward Elimination", backward_sets),
    "RFE": make_table("RFE", rfe_sets),
    "SelectKBest (f_regression)": make_table("SelectKBest (f_regression)", kbest_sets),
}
for name, t in tables.items():
    print(f"\n=== {name} ===")
    print(t.drop(columns="Scheme").to_string(index=False))

result = tables["Forward Selection"]

# ---------- 貼圖 1：Forward Selection 的 RMSE / R² vs. 特徵數 ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for a, col in zip(ax, ["RMSE", "R-squared"]):
    a.plot(result["Number of Features"], result[col], marker="o")
    a.set_title(f"{col} by Number of Features")
    a.set_xlabel("Number of Features"); a.set_ylabel(col)
plt.tight_layout()
plt.savefig("images/rmse_r2_by_features.png", dpi=150)

# ---------- 貼圖 2：四種方案比較 ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for name, t in tables.items():
    ax[0].plot(t["Number of Features"], t["RMSE"], marker="o", label=name)
    ax[1].plot(t["Number of Features"], t["R-squared"], marker="o", label=name)
for a, col in zip(ax, ["RMSE", "R-squared"]):
    a.set_title(f"{col} by Number of Features (all schemes)")
    a.set_xlabel("Number of Features"); a.set_ylabel(col)
    a.set_xticks(range(1, n + 1)); a.legend(fontsize=8)
plt.tight_layout()
plt.savefig("images/scheme_comparison.png", dpi=150)
plt.show()

# 另存 HTML 表格，方便貼進報告
result[["Number of Features", "Selected Features", "RMSE", "R-squared"]].to_html(
    "images/feature_selection_table.html", index=False)
pd.concat(tables.values()).to_csv("images/all_schemes.csv", index=False)
