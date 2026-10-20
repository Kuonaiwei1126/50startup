"""
產生 index.html 所需的 data.js / data.json：
每個資料集 × 5 種特徵選擇方法 × k = 1~n 的測試集指標、OLS 係數統計、VIF、殘差。
"""
import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, lars_path, enet_path
from sklearn.model_selection import train_test_split


def analyze(X, y, meta):
    FEATURES = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    n_train, n_feat = X_train.shape
    X_std = (X_train - X_train.mean()) / X_train.std()

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
                vif.append(1.0); continue
            r2 = LinearRegression().fit(X_train[others], X_train[f]).score(X_train[others], X_train[f])
            vif.append(1 / (1 - r2) if r2 < 1 else 1e9)
        coefs = [{"name": "Intercept", "est": beta[0], "se": se[0], "t": t[0], "p": pval[0],
                  "ci": [beta[0] - tcrit * se[0], beta[0] + tcrit * se[0]], "vif": None}]
        for i, f in enumerate(features, start=1):
            coefs.append({"name": f, "est": beta[i], "se": se[i], "t": t[i], "p": pval[i],
                          "ci": [beta[i] - tcrit * se[i], beta[i] + tcrit * se[i]], "vif": vif[i - 1]})
        return coefs, f_stat, f_p

    def evaluate(features, full=True):
        model = LinearRegression().fit(X_train[features], y_train)
        pred = model.predict(X_test[features])
        err = y_test.values - pred
        rmse = float(np.sqrt((err ** 2).mean()))
        if not full:
            return rmse
        n, k = len(y_test), len(features)
        r2 = float(1 - (err ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum())
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        mae = float(np.abs(err).mean())
        coefs, f_stat, f_p = ols_stats(features)
        residuals = [{"id": int(idx), "actual": float(a), "pred": float(pr), "resid": float(a - pr)}
                     for idx, a, pr in zip(X_test.index, y_test.values, pred)]
        residuals.sort(key=lambda r: -abs(r["resid"]))
        return {"features": list(features), "rmse": rmse, "r2": r2, "adj_r2": adj_r2, "mae": mae,
                "f_stat": float(f_stat), "f_p": float(f_p), "coefs": coefs, "residuals": residuals[:10]}

    rmse_of = lambda feats: evaluate(list(feats), full=False)

    def path_order(coef_path):
        order = []
        for step in range(coef_path.shape[1]):
            for i in np.flatnonzero(coef_path[:, step]):
                if FEATURES[i] not in order:
                    order.append(FEATURES[i])
        return order + [f for f in FEATURES if f not in order]

    # 1. Lasso（LARS-Lasso 路徑進入順序）
    _, _, lars_coefs = lars_path(X_std.values, y_train.values, method="lasso")
    lasso_order = path_order(lars_coefs)
    lasso_sets = [lasso_order[:k] for k in range(1, n_feat + 1)]

    # 2. Forward SFS（貪婪）
    remaining, selected, sfs_sets = list(FEATURES), [], []
    while remaining:
        best = min(remaining, key=lambda f: rmse_of(selected + [f]))
        selected.append(best); remaining.remove(best)
        sfs_sets.append(list(selected))

    # 3. RFE
    rfe_sets = []
    for k in range(1, n_feat + 1):
        rfe = RFE(LinearRegression(), n_features_to_select=k).fit(X_std, y_train)
        rfe_sets.append([f for f in FEATURES if rfe.support_[FEATURES.index(f)]])

    # 4. ElasticNet 路徑進入順序
    _, enet_coefs, _ = enet_path(X_std.values, y_train.values, l1_ratio=0.5, n_alphas=300)
    enet_order = path_order(enet_coefs)
    enet_sets = [enet_order[:k] for k in range(1, n_feat + 1)]

    # 5. Exhaustive Search（每個 k 的最佳子集）
    exh_sets = [list(min(combinations(FEATURES, k), key=rmse_of)) for k in range(1, n_feat + 1)]

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
         "desc": f"枚舉所有 C({n_feat}, k) 個子集，取測試 RMSE 最小者；是其他方法的上界。"},
    ]
    out = dict(meta)
    out.update({"features": FEATURES, "n_train": int(n_train), "n_test": int(len(y_test)), "methods": []})
    for m in METHODS:
        results = [evaluate(s) for s in m["sets"]]
        out["methods"].append({k: v for k, v in m.items() if k != "sets"} | {"results": results})
        print(f"  [{m['name']}] best k = {min(range(n_feat), key=lambda i: results[i]['rmse']) + 1}, "
              f"RMSE = {min(r['rmse'] for r in results):.3f}")
    return out


def numeric_inputs(X, cols, step_map=None):
    """即時預測器的欄位：min / max / step / 預設值（中位數）。"""
    inputs = []
    for c in cols:
        lo, hi, med = float(X[c].min()), float(X[c].max()), float(X[c].median())
        span = hi - lo
        step = (step_map or {}).get(c, 10 ** (np.floor(np.log10(span)) - 2) if span > 0 else 1)
        default = round(round(med / step) * step, 6)
        inputs.append({"type": "number", "feature": c, "label": c, "min": lo, "max": hi,
                       "step": float(step), "default": default})
    return inputs


datasets = []

# ============ 1. 50 Startups ============
print("50 Startups")
df = pd.read_csv("50_Startups.csv")
df = pd.get_dummies(df, columns=["State"], drop_first=True, dtype=int)
X, y = df.drop(columns=["Profit"]), df["Profit"]
inputs = numeric_inputs(X, ["R&D Spend", "Marketing Spend", "Administration"],
                        {"R&D Spend": 1000, "Marketing Spend": 1000, "Administration": 1000})
inputs.append({"type": "select", "label": "State Location", "options": [
    {"label": "California (Reference)", "dummies": {"State_Florida": 0, "State_New York": 0}},
    {"label": "Florida", "dummies": {"State_Florida": 1, "State_New York": 0}},
    {"label": "New York", "dummies": {"State_Florida": 0, "State_New York": 1}},
]})
datasets.append(analyze(X, y, {
    "id": "startups", "name": "50 Startups", "target": "Profit", "unit_prefix": "$", "unit_suffix": "",
    "decimals": 2, "sample_label": "Startup", "n_rows": int(len(df)),
    "title": "Feature Selection & Parsimony Analysis",
    "blurb": "預測新創公司的 <strong>Profit</strong>。特徵：R&D Spend、Administration、Marketing Spend 與 State（One-Hot，California 為基準）。",
    "source": "Kaggle 50 Startups（n = 50）",
    "prep": "train_test_split(test_size=0.2, random_state=0)；State One-Hot（drop_first）",
    "inputs": inputs,
    "feature_desc": {"R&D Spend": "研發支出", "Administration": "行政支出", "Marketing Spend": "行銷支出",
                     "State_Florida": "州 = Florida", "State_New York": "州 = New York"},
}))

# ============ 2. Boston Housing ============
print("Boston Housing")
bh = pd.read_csv("BostonHousing.csv")
bh["chas"] = bh["chas"].astype(int)
X, y = bh.drop(columns=["medv"]), bh["medv"]
desc = {"crim": "人均犯罪率", "zn": "大面積住宅用地比例", "indus": "非零售商業用地比例",
        "chas": "是否臨 Charles 河（1 = 是）", "nox": "一氧化氮濃度", "rm": "每戶平均房間數",
        "age": "1940 年前自用住宅比例", "dis": "到五個就業中心的加權距離", "rad": "公路可達性指數",
        "tax": "每萬元財產稅率", "ptratio": "師生比", "b": "1000(Bk − 0.63)²，Bk 為黑人比例", "lstat": "低收入人口比例"}
datasets.append(analyze(X, y, {
    "id": "boston", "name": "Boston Housing", "target": "MEDV", "unit_prefix": "$", "unit_suffix": "k",
    "decimals": 2, "sample_label": "House", "n_rows": int(len(bh)),
    "title": "Boston Housing — Feature Selection Analysis",
    "blurb": "預測波士頓各區房價中位數 <strong>MEDV</strong>（單位：千美元）。13 個特徵，k = 1～13；Exhaustive Search 枚舉全部 8,191 個子集。",
    "source": "Boston Housing（Harrison & Rubinfeld, 1978；n = 506）",
    "prep": "train_test_split(test_size=0.2, random_state=0)；無缺失值，chas 為 0/1",
    "inputs": numeric_inputs(X, list(X.columns), {"chas": 1, "rad": 1, "tax": 1, "zn": 1}),
    "feature_desc": desc,
}))

out = {"datasets": datasets}
with open("data.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
with open("data.js", "w", encoding="utf-8") as fh:
    fh.write("window.DATA = " + json.dumps(out, ensure_ascii=False) + ";\n")
print("\nwrote data.json / data.js")
