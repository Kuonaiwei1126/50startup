# sklearn 多元線性迴歸（MLR）— Kaggle 50_Startups

> 作業要求（白板）：1. CRISP-DM　2. Feature selection（Top 3）並 apply　3. 貼圖（MSE / RMSE / R² vs. 特徵數）
> 資料集：Kaggle `50_Startups.csv`（欄位：R&D Spend, Administration, Marketing Spend, State, Profit）
> 環境：VS Code + Python 3.x（`pandas`, `numpy`, `scikit-learn`, `matplotlib`）

---

## 1. CRISP-DM

| 階段 | 本作業對應內容 |
|---|---|
| **Business Understanding** | 預測新創公司的 **Profit**，找出哪些支出項目（R&D / Administration / Marketing）及所在州對獲利影響最大，作為資源配置依據。 |
| **Data Understanding** | 50 筆樣本、4 個自變數（3 個數值、1 個類別 State）、1 個目標 Profit。以 `df.describe()`、相關係數矩陣、散佈圖初探：R&D Spend 與 Profit 相關最高。 |
| **Data Preparation** | 1) 檢查缺失值 2) State 做 One-Hot Encoding（`State_California / State_Florida / State_New York`）3) 切分訓練 / 測試集（80 / 20）。 |
| **Modeling** | `sklearn.linear_model.LinearRegression`，搭配 **Forward Selection** 逐步加入特徵，記錄每個特徵數下的 RMSE 與 R²。 |
| **Evaluation** | 以測試集 **RMSE**（越小越好）與 **R²**（越大越好）評估，畫出「特徵數 vs. RMSE / R²」曲線，選出 Top 3 特徵。 |
| **Deployment** | 以 Top 3 特徵重新訓練最終模型，輸出迴歸係數與預測，整理成本報告（Markdown + HTML 圖表）。 |

---

## 2. Feature Selection（Top 3）與套用

### 2.1 完整程式碼（`mlr_50_startups.py`）

```python
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
```

> 程式會自動建立 `images/` 資料夾。資料來源：Kaggle 50_Startups（50 筆）。

### 2.2 Forward Selection 過程（測試集，random_state=0）

<table>
  <thead>
    <tr><th>Number of Features</th><th>Selected Features</th><th>MSE</th><th>RMSE</th><th>R-squared</th></tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>[R&amp;D Spend]</td><td>68,473,440</td><td>8274.87</td><td>0.9465</td></tr>
    <tr><td>2</td><td>[R&amp;D Spend, Marketing Spend]</td><td>67,220,280</td><td>8198.80</td><td>0.9474</td></tr>
    <tr><td>3</td><td>[R&amp;D Spend, Marketing Spend, State_California]</td><td>67,229,760</td><td>8199.38</td><td>0.9474</td></tr>
    <tr><td>4</td><td>[R&amp;D Spend, Marketing Spend, State_California, State_New York]</td><td>70,726,700</td><td>8409.92</td><td>0.9447</td></tr>
    <tr><td>5</td><td>[R&amp;D Spend, Marketing Spend, State_California, State_New York, State_Florida]</td><td>70,726,700</td><td>8409.92</td><td>0.9447</td></tr>
    <tr><td>6</td><td>[R&amp;D Spend, Marketing Spend, State_California, State_New York, State_Florida, Administration]</td><td>83,502,860</td><td>9137.99</td><td>0.9347</td></tr>
  </tbody>
</table>

### 2.3 Top 3 特徵與最終模型

- **Top 3**：`R&D Spend`、`Marketing Spend`、`State_California`
- 最終模型：`Profit = 46335.26 + 0.7886·(R&D Spend) + 0.0326·(Marketing Spend) + 20.32·(State_California)`
- 測試集 MSE = `67,229,760`，RMSE = `8199.38`，R² = `0.9474`

---

## 3. 貼圖

<p align="center">
  <img src="images/rmse_r2_by_features.png" alt="RMSE and R-squared by Number of Features" width="800">
</p>

**圖表說明**

- MSE / RMSE 在特徵數 = 2～3 時達到最低（RMSE ≈ 8199）、R² 最高（≈ 0.9474），之後再加入特徵誤差反而上升 → 典型的 **overfitting** U 型曲線（白板上的 MSE 圖）。
- 第 3 個特徵 `State_California` 的係數僅約 20，對 RMSE 幾乎無影響（8198.80 → 8199.38），顯示 State 對 Profit 幾乎沒有解釋力。
- 特徵數 4 與 5 結果完全相同：三個 State dummy 加總恆為 1，與截距完全共線（**dummy variable trap**），第三個 dummy 不提供新資訊。
- 最後加入的 `Administration` 使 RMSE 跳升至 9138，是最不重要的特徵。
- 依作業要求取 **Top 3**（虛線處）作為最終特徵集合，R² = 0.947。

---

## 附錄：資料集欄位

| 欄位 | 型態 | 說明 |
|---|---|---|
| R&D Spend | float | 研發支出 |
| Administration | float | 行政支出 |
| Marketing Spend | float | 行銷支出 |
| State | category | New York / California / Florida（One-Hot 後 3 欄） |
| Profit | float | **目標變數** |
