# 多元線性迴歸（MLR）特徵選擇探索器 — 50 Startups & Boston Housing

互動式探索器 `index.html`：頁面頂端下拉選單切換資料集（**50 Startups**：5 特徵 → Profit；**Boston Housing**：13 特徵 → MEDV），
每個資料集都有 5 種特徵選擇方法（Lasso / Forward SFS / RFE / ElasticNet / Exhaustive Search）× k = 1～n 的 RMSE / R² 曲線、
係數統計（SE、t、p、95% CI、VIF）、即時預測器與殘差診斷。
直接開啟 Boston：https://kuonaiwei1126.github.io/50startup/#boston
靜態報告：`report.html` / [MLR_50_Startups_作業.md](MLR_50_Startups_作業.md)。

```bash
pip install pandas numpy scikit-learn matplotlib
python mlr_50_startups.py   # 產生圖表
python build_data.py        # 產生 index.html 用的 data.js / data.json
```

[![互動式探索器畫面](images/explorer_screenshot.png)](https://kuonaiwei1126.github.io/50startup/)

![RMSE and R-squared](images/rmse_r2_by_features.png)

線上報告（GitHub Pages）：https://kuonaiwei1126.github.io/50startup/
