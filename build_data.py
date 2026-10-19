"""
產生 index.html 所需的 data.json：
5 種特徵選擇方法 × k = 1~5 的測試集指標、OLS 係數統計、VIF、殘差。
"""
import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, lars_path, enet_path
from sklearn.model_selection import train_test_split

df = pd.read_csv("50_Startups.csv")
df = pd.get_dummies(df, columns=["State"], drop_first=True, dtype=int)
X = df.drop(columns=["Profit"])
y = df["Profit"]
FEATURES = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
n_train, n_feat = X_train.shape


def ols_stats(features):
    """手刻 OLS：係數、標準誤、t、p、95% CI、F 檢定、VIF（訓練集）。"""
    A = np.column_stack([np.ones(n_train), X_train[features].values])
    yt = y_train.values
    p = A.shape[1]
    beta = np.linalg.lstsq(A, yt, rcond=None)[0]
    resid = yt - A @ beta
    dof = n_train - p
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    pval = 2 * stats.t.sf(np.abs(t), dof)
    tcrit = stats.t.ppf(0.975, dof)
    ss_res = resid @ resid
    ss_tot = ((yt - yt.mean()) ** 2).sum()
    f_stat = ((ss_tot - ss_res) / (p - 1)) / (ss_res / dof)
    f_p = stats.f.sf(f_stat, p - 1, dof)

    vif = []
    for f in features:
        others = [o for o in features if o != f]
        if not others:
            vif.append(1.0)
            continue
        r2 = LinearRegression().fit(X_train[others], X_train[f]).score(X_train[others], X_train[f])
        vif.append(1 / (1 - r2) if r2 < 1 else float("inf"))

    coefs = [{"name": "Intercept", "est": beta[0], "se": se[0], "t": t[0], "p": pval[0],
              "ci": [beta[0] - tcrit * se[0], beta[0] + tcrit * se[0]], "vif": None}]
    for i, f in enumerate(features, start=1):
        coefs.append({"name": f, "est": beta[i], "se": se[i], "t": t[i], "p": pval[i],
                      "ci": [beta[i] - tcrit * se[i], beta[i] + tcrit * se[i]], "vif": vif[i - 1]})
    return coefs, f_stat, f_p


def evaluate(features):
    model = LinearRegression().fit(X_train[features], y_train)
    pred = model.predict(X_test[features])
    err = y_test.values - pred
    n, k = len(y_test), len(features)
    rmse = float(np.sqrt((err ** 2).mean()))
    r2 = float(1 - (err ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum())
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    mae = float(np.abs(err).mean())
    coefs, f_stat, f_p = ols_stats(features)
    residuals = [{"id": int(idx), "actual": float(a), "pred": float(pr), "resid": float(a - pr)}
                 for idx, a, pr in zip(X_test.index, y_test.values, pred)]
    residuals.sort(key=lambda r: -abs(r["resid"]))
    return {"features": list(features), "rmse": rmse, "r2": r2, "adj_r2": adj_r2, "mae": mae,
            "f_stat": float(f_stat), "f_p": float(f_p), "coefs": coefs, "residuals": residuals}


def rmse_of(features):
    return evaluate(features)["rmse"]


# ---- 1. Lasso（L1 路徑進入順序排名）----
X_std = (X_train - X_train.mean()) / X_train.std()
_, _, lars_coefs = lars_path(X_std.values, y_train.values, method="lasso")
lasso_order = []
for step in range(lars_coefs.shape[1]):
    for i in np.flatnonzero(lars_coefs[:, step]):
        if FEATURES[i] not in lasso_order:
            lasso_order.append(FEATURES[i])
lasso_sets = [lasso_order[:k] for k in range(1, n_feat + 1)]

# ---- 2. Forward SFS（貪婪，每步加入使測試 RMSE 最小的特徵）----
remaining, selected, sfs_sets = list(FEATURES), [], []
while remaining:
    best = min(remaining, key=lambda f: rmse_of(selected + [f]))
    selected.append(best); remaining.remove(best)
    sfs_sets.append(list(selected))

# ---- 3. RFE（標準化後依係數遞迴刪除）----
rfe_sets = []
for k in range(1, n_feat + 1):
    rfe = RFE(LinearRegression(), n_features_to_select=k).fit(X_std, y_train)
    rfe_sets.append([f for f in FEATURES if rfe.support_[FEATURES.index(f)]])

# ---- 4. ElasticNet（L1+L2 路徑進入順序）----
_, enet_coefs, _ = enet_path(X_std.values, y_train.values, l1_ratio=0.5, n_alphas=300)
enet_order = []
for step in range(enet_coefs.shape[1]):
    for i in np.flatnonzero(enet_coefs[:, step]):
        if FEATURES[i] not in enet_order:
            enet_order.append(FEATURES[i])
for f in FEATURES:
    if f not in enet_order:
        enet_order.append(f)
enet_sets = [enet_order[:k] for k in range(1, n_feat + 1)]

# ---- 5. Exhaustive Search（每個 k 的最佳子集）----
exh_sets = [list(min(combinations(FEATURES, k), key=lambda c: rmse_of(list(c))))
            for k in range(1, n_feat + 1)]

METHODS = [
    {"id": "lasso", "name": "Lasso (L1 Ranking)", "tag": "L1 路徑", "sets": lasso_sets,
     "desc": "以 LARS-Lasso 路徑中特徵進入模型的先後順序排名，取前 k 個。"},
    {"id": "sfs", "name": "Forward SFS", "tag": "Greedy", "sets": sfs_sets,
     "desc": "從空集合開始，每一步加入使測試 RMSE 下降最多的特徵。"},
    {"id": "rfe", "name": "RFE", "tag": "Backward", "sets": rfe_sets,
     "desc": "標準化後依係數絕對值遞迴刪除最不重要的特徵，直到剩下 k 個。"},
    {"id": "enet", "name": "ElasticNet", "tag": "L1 + L2", "sets": enet_sets,
     "desc": "以 ElasticNet（l1_ratio = 0.5）正則化路徑中特徵進入順序排名。"},
    {"id": "exh", "name": "Exhaustive Search", "tag": "Best subset", "sets": exh_sets,
     "desc": "枚舉所有 C(5, k) 個子集，取測試 RMSE 最小者；是其他方法的上界。"},
]

out = {
    "features": FEATURES,
    "train_means": {f: float(X_train[f].mean()) for f in FEATURES},
    "n_train": int(n_train), "n_test": int(len(y_test)),
    "methods": [],
}
for m in METHODS:
    results = [evaluate(s) for s in m["sets"]]
    out["methods"].append({k: v for k, v in m.items() if k != "sets"} | {"results": results})
    print(f"\n=== {m['name']} ===")
    for r in results:
        print(f"k={len(r['features'])}  RMSE={r['rmse']:.2f}  R2={r['r2']:.4f}  {r['features']}")

with open("data.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
# data.js 讓 index.html 直接以 <script> 載入，離線開啟也能用
with open("data.js", "w", encoding="utf-8") as fh:
    fh.write("window.DATA = " + json.dumps(out, ensure_ascii=False) + ";\n")
print("\nwrote data.json / data.js")
