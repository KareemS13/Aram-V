"""
Model Overfitting / Underfitting Diagnostics
============================================

Standalone script — does NOT affect the dashboard or pipeline outputs.

Run:
    python diagnostics.py

Saves to: outputs/diagnostics/
    - 01_learning_curve.png
    - 02_residuals_over_time.png
    - 03_insample_vs_outsample.png
    - 04_feature_importance.png
    - diagnostics_report.txt
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dotenv import load_dotenv

load_dotenv()

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)

STYLE = {
    "train_color":  "#1565c0",
    "test_color":   "#e53935",
    "resid_color":  "#455a64",
    "zero_color":   "rgba(0,0,0,0.25)",
    "fig_bg":       "#f8fafc",
    "card_bg":      "#ffffff",
}

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.facecolor":     STYLE["card_bg"],
    "figure.facecolor":   STYLE["fig_bg"],
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
})


def load_data():
    from data.loader import build_master_df
    from features.engineering import build_feature_matrix, get_exog_for_sarima
    from config import SARIMA_EXOG_COLS

    fred_key = os.environ.get("FRED_API_KEY", "")
    master_df = build_master_df(fred_api_key=fred_key)
    X, y = build_feature_matrix(master_df)
    exog_sarima = get_exog_for_sarima(X, SARIMA_EXOG_COLS)
    return X, y, exog_sarima


# ==========================================================================
# 1. LEARNING CURVE
#    Train on expanding window, compute both in-sample and out-of-sample MAE
#    as a function of training size. Healthy = gap closes; overfit = gap stays.
# ==========================================================================

def learning_curve(y, X, exog_sarima, min_train=36, step=6):
    from models.sarima_model import SARIMAXModel
    from models.gbm_model import GBMForecaster
    import pmdarima as pm
    from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX

    n = len(y)
    train_sizes = list(range(min_train, n - step, step))

    # Run auto_arima once on the smallest window to find order for all folds
    print("  Learning curve: finding SARIMA order once...")
    _init_endog = y.iloc[:min_train]
    _init_exog  = exog_sarima.iloc[:min_train]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _auto = pm.auto_arima(
            _init_endog, exogenous=_init_exog, m=12, stepwise=True,
            information_criterion="aic", max_p=3, max_q=3, max_P=1, max_Q=1,
            max_d=2, max_D=1, seasonal=True, error_action="ignore", suppress_warnings=True,
        )
    _fixed_order    = _auto.order
    _fixed_seasonal = _auto.seasonal_order
    print(f"  Fixed order: {_fixed_order} x {_fixed_seasonal}")

    records = []
    for ts in train_sizes:
        y_tr   = y.iloc[:ts]
        y_te   = y.iloc[ts:ts + step]
        X_tr   = X.iloc[:ts]
        X_te   = X.iloc[ts:ts + step]
        ex_tr  = exog_sarima.iloc[:ts]
        ex_te  = exog_sarima.iloc[ts:ts + step]

        # --- SARIMA (fixed order, no auto_arima) ---
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _model = _SARIMAX(y_tr, exog=ex_tr, order=_fixed_order,
                                  seasonal_order=_fixed_seasonal,
                                  enforce_stationarity=False, enforce_invertibility=False)
                _fitted = _model.fit(disp=False)
            sar = SARIMAXModel.__new__(SARIMAXModel)
            sar._fitted = _fitted
            sar._order = _fixed_order
            sar._seasonal_order = _fixed_seasonal
            sar.m = 12

            # in-sample: residuals = actual - fitted (already computed by statsmodels)
            resid      = sar._fitted.resid.dropna()
            in_mae_sar = float(np.abs(resid).mean())

            # out-of-sample
            fc_sar      = sar.forecast(steps=len(y_te), exog_future=ex_te)
            out_mae_sar = float(np.abs(y_te.values - fc_sar.point.values).mean())
        except Exception as e:
            print(f"    SARIMA failed at ts={ts}: {e}")
            in_mae_sar = out_mae_sar = np.nan

        # --- GBM ---
        try:
            gbm = GBMForecaster()
            gbm.fit(y_tr, exog=X_tr)

            # in-sample: use only the selected features subset
            sel = gbm._selected_features
            X_tr_gbm   = gbm.get_training_matrix(y_tr, exog=X_tr[sel])
            in_preds   = gbm.forecaster.estimator.predict(X_tr_gbm)
            in_mae_gbm = float(np.abs(y_tr.iloc[len(y_tr) - len(in_preds):].values - in_preds).mean())

            # out-of-sample: apply same feature selection to test exog
            exog_fut = X_te[[c for c in sel if c in X_te.columns]].copy()
            fc_gbm       = gbm.forecast(steps=len(y_te), exog_future=exog_fut)
            out_mae_gbm  = float(np.abs(y_te.values - fc_gbm.point.values).mean())
        except Exception as e:
            print(f"    GBM failed at ts={ts}: {e}")
            in_mae_gbm = out_mae_gbm = np.nan

        records.append({
            "train_size":    ts,
            "in_mae_sarima": in_mae_sar,
            "out_mae_sarima": out_mae_sar,
            "in_mae_gbm":    in_mae_gbm,
            "out_mae_gbm":   out_mae_gbm,
        })
        print(f"  train_size={ts:3d} | SARIMA in={in_mae_sar:.3f} out={out_mae_sar:.3f} "
              f"| GBM in={in_mae_gbm:.3f} out={out_mae_gbm:.3f}")

    return pd.DataFrame(records)


def plot_learning_curve(lc_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle("Learning Curve — In-Sample vs Out-of-Sample MAE", fontsize=14, fontweight="bold", y=1.01)

    for ax, model, label in zip(axes, ["sarima", "gbm"], ["SARIMA", "GBM"]):
        in_col  = f"in_mae_{model}"
        out_col = f"out_mae_{model}"
        df = lc_df.dropna(subset=[in_col, out_col])

        ax.plot(df["train_size"], df[in_col],  color=STYLE["train_color"],
                lw=2, marker="o", markersize=4, label="In-sample MAE (train)")
        ax.plot(df["train_size"], df[out_col], color=STYLE["test_color"],
                lw=2, marker="s", markersize=4, label="Out-of-sample MAE (test)")

        gap = (df[out_col] - df[in_col]).mean()
        ax.fill_between(df["train_size"], df[in_col], df[out_col],
                        alpha=0.10, color=STYLE["test_color"])

        ax.set_title(f"{label}  (avg gap = {gap:+.3f} pp)", fontsize=12)
        ax.set_xlabel("Training size (months)")
        ax.set_ylabel("MAE (percentage points)")
        ax.legend(fontsize=10)

        # Annotation
        verdict = "Likely overfitting" if gap > 0.15 else ("Likely underfitting" if df[out_col].mean() > 0.6 else "Reasonable fit")
        ax.annotate(verdict, xy=(0.97, 0.06), xycoords="axes fraction",
                    ha="right", fontsize=10,
                    color=STYLE["test_color"] if "over" in verdict else STYLE["train_color"],
                    fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "01_learning_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ==========================================================================
# 2. RESIDUALS OVER TIME
#    For each model, plot actual - fitted (in-sample) and actual - predicted
#    (out-of-sample, from walk-forward CV). Patterns = model misspecification.
# ==========================================================================

def compute_residuals(y, X, exog_sarima, eval_start="2023-01-01"):
    from models.sarima_model import SARIMAXModel
    from models.gbm_model import GBMForecaster
    from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold

    eval_ts   = pd.Timestamp(eval_start)
    train_mask = y.index < eval_ts
    y_tr = y[train_mask];  y_te = y[~train_mask]
    X_tr = X[train_mask];  X_te = X[~train_mask]
    ex_tr = exog_sarima[train_mask]

    # SARIMA in-sample residuals (on training data)
    sar = SARIMAXModel()
    sar.fit(y_tr, exog=ex_tr)
    sar_insample = sar._fitted.resid.dropna()

    # SARIMA out-of-sample residuals via walk-forward CV (h=1)
    sar_full = SARIMAXModel()
    sar_full.fit(y_tr, exog=ex_tr)
    sar_cv   = sar_full.walk_forward_cv(y, exog=exog_sarima,
                                         eval_start=eval_start, horizons=[1])
    sar_cv_h1 = sar_cv[sar_cv["horizon"] == 1].set_index("date")
    sar_outsample = sar_cv_h1["error"]

    # GBM in-sample residuals
    gbm = GBMForecaster()
    gbm.fit(y_tr, exog=X_tr)
    sel = gbm._selected_features
    X_tr_gbm  = gbm.get_training_matrix(y_tr, exog=X_tr[sel])
    gbm_preds = gbm.forecaster.estimator.predict(X_tr_gbm)
    gbm_insample = pd.Series(
        y_tr.iloc[len(y_tr) - len(gbm_preds):].values - gbm_preds,
        index=y_tr.index[len(y_tr) - len(gbm_preds):]
    )

    # GBM out-of-sample residuals via walk-forward CV (h=1)
    gbm_full = GBMForecaster()
    gbm_full.fit(y_tr, exog=X_tr)
    gbm_cv   = gbm_full.walk_forward_cv(y, exog=X, eval_start=eval_start, horizons=[1])
    gbm_cv_h1     = gbm_cv[gbm_cv["horizon"] == 1].set_index("date")
    gbm_outsample = gbm_cv_h1["error"]

    return {
        "sar_in":  sar_insample,
        "sar_out": sar_outsample,
        "gbm_in":  gbm_insample,
        "gbm_out": gbm_outsample,
    }


def plot_residuals(resids, eval_start="2023-01-01"):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle("Residuals Over Time — In-Sample (train) vs Out-of-Sample (test)",
                 fontsize=14, fontweight="bold")

    pairs = [
        (axes[0, 0], resids["sar_in"],  "SARIMA — In-Sample Residuals (train)",  STYLE["train_color"]),
        (axes[0, 1], resids["sar_out"], "SARIMA — Out-of-Sample Residuals (test)", STYLE["test_color"]),
        (axes[1, 0], resids["gbm_in"],  "GBM — In-Sample Residuals (train)",      STYLE["train_color"]),
        (axes[1, 1], resids["gbm_out"], "GBM — Out-of-Sample Residuals (test)",   STYLE["test_color"]),
    ]

    for ax, series, title, color in pairs:
        ax.axhline(0, color="black", lw=1, alpha=0.4)
        ax.bar(series.index, series.values, color=color, alpha=0.55, width=20)
        ax.plot(series.index, series.values, color=color, lw=1, alpha=0.8)

        std  = series.std()
        mean = series.mean()
        ax.axhline(mean, color=color, lw=1.5, ls="--", alpha=0.7, label=f"Mean={mean:+.3f}")
        ax.fill_between(series.index, -std, std, alpha=0.08, color=color, label=f"±1 std={std:.3f}")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Residual (pp)")
        ax.legend(fontsize=9)

        # Structural breaks annotation
        ax.axvline(pd.Timestamp("2020-03-01"), color="orange", lw=1, ls=":", alpha=0.7)
        ax.axvline(pd.Timestamp("2022-02-01"), color="purple", lw=1, ls=":", alpha=0.7)
        ax.text(pd.Timestamp("2020-04-01"), ax.get_ylim()[1] * 0.85, "COVID", fontsize=7, color="orange")
        ax.text(pd.Timestamp("2022-03-01"), ax.get_ylim()[1] * 0.85, "Ukraine", fontsize=7, color="purple")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "02_residuals_over_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ==========================================================================
# 3. IN-SAMPLE VS OUT-OF-SAMPLE TABLE + BAR CHART
# ==========================================================================

def insample_vs_outsample(y, X, exog_sarima, eval_start="2023-01-01"):
    from models.sarima_model import SARIMAXModel
    from models.gbm_model import GBMForecaster

    eval_ts    = pd.Timestamp(eval_start)
    train_mask = y.index < eval_ts
    y_tr = y[train_mask]; X_tr = X[train_mask]; ex_tr = exog_sarima[train_mask]
    y_te = y[~train_mask]; X_te = X[~train_mask]; ex_te = exog_sarima[~train_mask]

    records = []

    # SARIMA
    sar = SARIMAXModel()
    sar.fit(y_tr, exog=ex_tr)
    sar_in_mae  = float(np.abs(sar._fitted.resid.dropna()).mean())
    sar_in_rmse = float(np.sqrt((sar._fitted.resid.dropna()**2).mean()))
    fc_sar      = sar.forecast(steps=len(y_te), exog_future=ex_te)
    sar_out_mae  = float(np.abs(y_te.values - fc_sar.point.values).mean())
    sar_out_rmse = float(np.sqrt(((y_te.values - fc_sar.point.values)**2).mean()))
    records += [
        {"model": "SARIMA", "split": "In-sample",     "MAE": sar_in_mae,  "RMSE": sar_in_rmse},
        {"model": "SARIMA", "split": "Out-of-sample",  "MAE": sar_out_mae, "RMSE": sar_out_rmse},
    ]

    # GBM
    gbm = GBMForecaster()
    gbm.fit(y_tr, exog=X_tr)
    sel = gbm._selected_features
    X_tr_gbm   = gbm.get_training_matrix(y_tr, exog=X_tr[sel])
    gbm_preds  = gbm.forecaster.estimator.predict(X_tr_gbm)
    actual_tr  = y_tr.iloc[len(y_tr) - len(gbm_preds):].values
    gbm_in_mae  = float(np.abs(actual_tr - gbm_preds).mean())
    gbm_in_rmse = float(np.sqrt(((actual_tr - gbm_preds)**2).mean()))

    exog_fut = X_te[[c for c in sel if c in X_te.columns]].copy()
    fc_gbm      = gbm.forecast(steps=len(y_te), exog_future=exog_fut)
    gbm_out_mae  = float(np.abs(y_te.values - fc_gbm.point.values).mean())
    gbm_out_rmse = float(np.sqrt(((y_te.values - fc_gbm.point.values)**2).mean()))
    records += [
        {"model": "GBM", "split": "In-sample",    "MAE": gbm_in_mae,  "RMSE": gbm_in_rmse},
        {"model": "GBM", "split": "Out-of-sample", "MAE": gbm_out_mae, "RMSE": gbm_out_rmse},
    ]

    df = pd.DataFrame(records)
    df["Overfit ratio"] = df.apply(
        lambda r: df.loc[(df.model == r.model) & (df.split == "Out-of-sample"), "MAE"].values[0]
                  / df.loc[(df.model == r.model) & (df.split == "In-sample"),   "MAE"].values[0]
        if r.split == "In-sample" else np.nan, axis=1
    )
    return df


def plot_insample_vs_outsample(df):
    models = ["SARIMA", "GBM"]
    colors = {"In-sample": STYLE["train_color"], "Out-of-sample": STYLE["test_color"]}
    x = np.arange(len(models))
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("In-Sample vs Out-of-Sample Error — Overfitting Check",
                 fontsize=14, fontweight="bold")

    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        for i, (split, color) in enumerate(colors.items()):
            vals = [df.loc[(df.model == m) & (df.split == split), metric].values[0] for m in models]
            bars = ax.bar(x + (i - 0.5) * width, vals, width, label=split, color=color, alpha=0.80)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_title(f"{metric} (percentage points)", fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
        ax.set_ylabel(metric)
        ax.legend(fontsize=10)

        # Ratio annotation
        for i, m in enumerate(models):
            ratio = df.loc[(df.model == m) & (df.split == "In-sample"), "Overfit ratio"].values[0]
            if not np.isnan(ratio):
                color = STYLE["test_color"] if ratio > 1.5 else "green"
                ax.text(i, ax.get_ylim()[1] * 0.05,
                        f"ratio {ratio:.2f}x", ha="center", fontsize=9,
                        color=color, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "03_insample_vs_outsample.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ==========================================================================
# 4. GBM FEATURE IMPORTANCE
#    High importance on a few features + large train/test gap = overfitting
# ==========================================================================

def plot_feature_importance(y, X, eval_start="2023-01-01"):
    from models.gbm_model import GBMForecaster

    eval_ts    = pd.Timestamp(eval_start)
    train_mask = y.index < eval_ts
    y_tr = y[train_mask]; X_tr = X[train_mask]

    gbm = GBMForecaster()
    gbm.fit(y_tr, exog=X_tr)

    feat_names = gbm.get_feature_names()
    importances = gbm.forecaster.estimator.feature_importances_
    if len(feat_names) != len(importances):
        feat_names = [f"f{i}" for i in range(len(importances))]

    fi = pd.Series(importances, index=feat_names).sort_values(ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f"GBM Feature Importance ({len(feat_names)} selected features)\n"
                 "High concentration on few features may indicate overfitting",
                 fontsize=13, fontweight="bold")

    colors_bar = [STYLE["test_color"] if v > fi.quantile(0.80) else STYLE["train_color"]
                  for v in fi.values]
    ax.barh(fi.index, fi.values, color=colors_bar, alpha=0.75)
    ax.set_xlabel("Importance score")
    ax.set_title("Red = top 20% most important features", fontsize=10, color=STYLE["test_color"])

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "04_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ==========================================================================
# 5. TEXT REPORT
# ==========================================================================

def write_report(lc_df, iso_df):
    lines = []
    lines.append("=" * 62)
    lines.append("ARMENIA CPI — MODEL FIT DIAGNOSTICS REPORT")
    lines.append("=" * 62)
    lines.append("")

    lines.append("IN-SAMPLE vs OUT-OF-SAMPLE MAE")
    lines.append("-" * 40)
    for _, row in iso_df.iterrows():
        lines.append(f"  {row['model']:8s} {row['split']:16s}  MAE={row['MAE']:.4f}  RMSE={row['RMSE']:.4f}")
    lines.append("")

    lines.append("OVERFIT RATIO  (out-of-sample MAE / in-sample MAE)")
    lines.append("-" * 40)
    lines.append("  Ratio > 2.0 -> likely overfitting")
    lines.append("  Ratio < 1.1 -> possible underfitting (model too simple)")
    lines.append("  Ratio 1.1-2.0 -> reasonable generalisation")
    lines.append("")
    for model in ["SARIMA", "GBM"]:
        in_mae  = iso_df.loc[(iso_df.model == model) & (iso_df.split == "In-sample"),  "MAE"].values[0]
        out_mae = iso_df.loc[(iso_df.model == model) & (iso_df.split == "Out-of-sample"), "MAE"].values[0]
        ratio   = out_mae / in_mae
        if ratio > 2.0:
            verdict = "OVERFITTING"
        elif out_mae > 0.6:
            verdict = "UNDERFITTING"
        else:
            verdict = "OK"
        lines.append(f"  {model:8s}  ratio={ratio:.2f}x  -> {verdict}")
    lines.append("")

    lines.append("LEARNING CURVE SUMMARY")
    lines.append("-" * 40)
    lines.append("  (avg gap = mean(out_MAE - in_MAE) across all training sizes)")
    for model in ["sarima", "gbm"]:
        df = lc_df.dropna(subset=[f"in_mae_{model}", f"out_mae_{model}"])
        if len(df):
            gap = (df[f"out_mae_{model}"] - df[f"in_mae_{model}"]).mean()
            closing = df[f"out_mae_{model}"].iloc[-1] - df[f"out_mae_{model}"].iloc[0]
            lines.append(f"  {model.upper():8s}  avg gap={gap:+.4f}  "
                         f"OOS trend={'improving' if closing < 0 else 'worsening'} ({closing:+.4f})")
    lines.append("")

    lines.append("INTERPRETATION GUIDE")
    lines.append("-" * 40)
    lines.append("  Learning curve: if in-sample MAE << out-of-sample MAE")
    lines.append("  with a persistent gap that does not close as training grows,")
    lines.append("  the model is memorizing training data (overfitting).")
    lines.append("")
    lines.append("  Residual plots: random scatter around zero = good.")
    lines.append("  Patterns (trending, clustered errors) = misspecification.")
    lines.append("  Spike in out-of-sample residuals = overfitting or regime change.")
    lines.append("")
    lines.append("  Feature importance: if 1-2 features dominate heavily,")
    lines.append("  the model may be latching onto spurious correlations.")
    lines.append("=" * 62)

    path = os.path.join(OUT_DIR, "diagnostics_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {path}")
    print()
    print("\n".join(lines))
    return path


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("Loading data...")
    X, y, exog_sarima = load_data()
    print(f"Dataset: {len(y)} obs, {X.shape[1]} features\n")

    print("=" * 50)
    print("1/4  Learning curve (this takes a few minutes)...")
    print("=" * 50)
    lc_df = learning_curve(y, X, exog_sarima)
    plot_learning_curve(lc_df)

    print()
    print("=" * 50)
    print("2/4  Residuals over time...")
    print("=" * 50)
    resids = compute_residuals(y, X, exog_sarima)
    plot_residuals(resids)

    print()
    print("=" * 50)
    print("3/4  In-sample vs out-of-sample comparison...")
    print("=" * 50)
    iso_df = insample_vs_outsample(y, X, exog_sarima)
    plot_insample_vs_outsample(iso_df)

    print()
    print("=" * 50)
    print("4/4  GBM feature importance...")
    print("=" * 50)
    plot_feature_importance(y, X)

    print()
    print("=" * 50)
    print("Writing report...")
    print("=" * 50)
    write_report(lc_df, iso_df)

    print()
    print(f"All diagnostics saved to: {OUT_DIR}")
