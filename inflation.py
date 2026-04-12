"""
Armenia CPI Inflation Forecasting Pipeline
==========================================

Usage
-----
    python inflation.py --horizon 12 --eval-start 2023-01-01 --fred-key YOUR_KEY

    # Or with FRED_API_KEY in .env file:
    python inflation.py --horizon 12

Arguments
---------
  --horizon      Months to forecast (default: 12)
  --eval-start   Walk-forward CV start date (default: 2023-01-01)
  --fred-key     FRED API key (overrides .env file)
  --no-tune      Skip GBM hyperparameter tuning (faster)
  --no-cv        Skip walk-forward CV (faster, no model comparison chart)
"""

import argparse
import os
import sys
import warnings

import pandas as pd

# Suppress verbose warnings from statsmodels / pmdarima
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Armenia CPI inflation forecasting pipeline"
    )
    parser.add_argument("--horizon",     type=int, default=12,
                        help="Months to forecast ahead (default: 12)")
    parser.add_argument("--eval-start",  default="2023-01-01",
                        help="Walk-forward CV start date (default: 2023-01-01)")
    parser.add_argument("--fred-key",    default=None,
                        help="FRED API key (or set FRED_API_KEY in .env)")
    parser.add_argument("--no-tune",     action="store_true",
                        help="Skip GBM hyperparameter tuning")
    parser.add_argument("--no-cv",       action="store_true",
                        help="Skip walk-forward cross-validation")
    return parser.parse_args()


def run_pipeline(args):
    from config import (
        Config, SARIMA_EXOG_COLS, FORECAST_HORIZON,
        TABLES_DIR, FIGURES_DIR,
    )
    from data.loader import build_master_df
    from features.engineering import build_feature_matrix, get_exog_for_sarima
    from models.sarima_model import SARIMAXModel
    from models.gbm_model import GBMForecaster
    from models.ensemble import ForecastEnsemble
    from explain.shap_decompose import InflationDecomposer
    from viz import charts

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fred_key = args.fred_key or os.environ.get("FRED_API_KEY", "")
    horizon = args.horizon

    # ==========================================================================
    # 1. LOAD DATA
    # ==========================================================================
    print("\n" + "="*60)
    print("STEP 1: Loading data")
    print("="*60)

    comtrade_key = os.environ.get("COMTRADE_API_KEY", "")
    master_df = build_master_df(fred_api_key=fred_key, comtrade_api_key=comtrade_key)

    # ==========================================================================
    # 2. BUILD FEATURE MATRIX
    # ==========================================================================
    print("\n" + "="*60)
    print("STEP 2: Building feature matrix")
    print("="*60)

    X, y = build_feature_matrix(master_df)

    # Save CPI history for dashboard
    hist_df = y.reset_index()
    hist_df.columns = ["date", "value"]
    hist_df.to_csv(os.path.join(TABLES_DIR, "cpi_history.csv"), index=False)

    # SARIMA exogenous features (subset of X)
    exog_sarima = get_exog_for_sarima(X, SARIMA_EXOG_COLS)

    # ==========================================================================
    # 3. TRAIN/TEST SPLIT
    # ==========================================================================
    eval_start = pd.Timestamp(args.eval_start)
    train_mask = X.index < eval_start

    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    exog_train = exog_sarima[train_mask]
    exog_test  = exog_sarima[~train_mask]

    print(f"\nTrain: {y_train.index.min().strftime('%Y-%m')} -> "
          f"{y_train.index.max().strftime('%Y-%m')} ({len(y_train)} obs)")
    print(f"Test:  {y_test.index.min().strftime('%Y-%m')} -> "
          f"{y_test.index.max().strftime('%Y-%m')} ({len(y_test)} obs)")

    # ==========================================================================
    # 4. FIT SARIMA
    # ==========================================================================
    print("\n" + "="*60)
    print("STEP 4: Fitting SARIMA model")
    print("="*60)

    sarima = SARIMAXModel()
    sarima.fit(y_train, exog=exog_train)

    # ==========================================================================
    # 5. FIT GBM
    # ==========================================================================
    print("\n" + "="*60)
    print("STEP 5: Fitting LightGBM model")
    print("="*60)

    gbm = GBMForecaster()

    if not args.no_tune:
        print("Tuning GBM hyperparameters (this may take a minute)...")
        gbm.tune(y_train, exog=X_train)
    else:
        gbm.fit(y_train, exog=X_train)

    # ==========================================================================
    # 6. WALK-FORWARD CROSS-VALIDATION
    # ==========================================================================
    sarima_cv, gbm_cv = None, None

    if not args.no_cv:
        print("\n" + "="*60)
        print("STEP 6: Walk-forward cross-validation")
        print("="*60)

        print("Running SARIMA CV...")
        sarima_cv = sarima.walk_forward_cv(
            y, exog=exog_sarima, eval_start=args.eval_start,
            horizons=[1, 3, 6, 12],
        )

        print("Running GBM CV...")
        gbm_cv = gbm.walk_forward_cv(
            y, exog=X, eval_start=args.eval_start,
            horizons=[1, 3, 6, 12],
        )

        # Save CV metrics
        ensemble_obj_tmp = ForecastEnsemble()
        sarima_metrics = ensemble_obj_tmp.compute_cv_metrics(sarima_cv, "SARIMA")
        gbm_metrics    = ensemble_obj_tmp.compute_cv_metrics(gbm_cv,    "GBM")
        all_metrics = pd.concat([sarima_metrics, gbm_metrics], ignore_index=True)
        metrics_path = os.path.join(TABLES_DIR, "model_cv_metrics.csv")
        all_metrics.to_csv(metrics_path, index=False)
        print(f"\nCV metrics saved to {metrics_path}")
        print(all_metrics.to_string(index=False))

    # ==========================================================================
    # 7. FORECAST
    # ==========================================================================
    print("\n" + "="*60)
    print(f"STEP 7: Generating {horizon}-month forecast")
    print("="*60)

    # Refit both models on the full dataset so that last_window aligns with
    # the most recent data and the future exog index is contiguous.
    print("Refitting SARIMA on full dataset...")
    sarima.fit(y, exog=exog_sarima)

    print("Refitting GBM on full dataset...")
    gbm.fit(y, exog=X)

    last_date = y.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    # Future exog for SARIMA: carry forward last known values
    last_exog = exog_sarima.iloc[-1:]
    exog_future_sarima = pd.concat(
        [last_exog] * horizon, ignore_index=True
    )
    exog_future_sarima.index = future_dates
    for col in ["covid_2020", "ukraine_2022", "rub_crisis_2014"]:
        if col in exog_future_sarima.columns:
            exog_future_sarima[col] = 0

    # Future exog for GBM
    last_X = X.iloc[-1:]
    exog_future_gbm = pd.concat(
        [last_X] * horizon, ignore_index=True
    )
    exog_future_gbm.index = future_dates
    for col in ["covid_2020", "ukraine_2022", "rub_crisis_2014"]:
        if col in exog_future_gbm.columns:
            exog_future_gbm[col] = 0

    sarima_fc = sarima.forecast(steps=horizon, exog_future=exog_future_sarima)
    gbm_fc    = gbm.forecast(steps=horizon, exog_future=exog_future_gbm)

    # ==========================================================================
    # 8. ENSEMBLE
    # ==========================================================================
    ensemble = ForecastEnsemble()
    result = ensemble.combine(
        sarima_result=sarima_fc,
        gbm_point=gbm_fc.point,
        sarima_cv=sarima_cv,
        gbm_cv=gbm_cv,
    )

    # Save forecast CSV
    fc_df = pd.DataFrame({
        "date":     future_dates,
        "point":    result.point.values,
        "lower_50": result.lower_50.values,
        "upper_50": result.upper_50.values,
        "lower_95": result.lower_95.values,
        "upper_95": result.upper_95.values,
    })
    fc_path = os.path.join(TABLES_DIR, "forecast_point_ci.csv")
    fc_df.to_csv(fc_path, index=False)
    print(f"Forecast saved to {fc_path}")

    # ==========================================================================
    # 9. SHAP ATTRIBUTION
    # ==========================================================================
    print("\n" + "="*60)
    print("STEP 9: Computing SHAP driver attribution")
    print("="*60)

    decomposer = InflationDecomposer(gbm)
    # Use full-dataset training matrix for SHAP background (model was refit on all data)
    X_full_gbm = gbm.get_training_matrix(y, exog=X)
    decomposer.fit_explainer(X_full_gbm)

    # SHAP values for the last `horizon` rows of the training matrix
    # (these correspond to the most recent observations the model "sees" when
    #  making recursive predictions — closest proxy to what drives the forecast)
    X_forecast_gbm = X_full_gbm.iloc[-horizon:] if len(X_full_gbm) >= horizon else X_full_gbm

    explanation = decomposer.explain_forecast(X_forecast_gbm)
    monthly_contrib = decomposer.monthly_contributions(explanation)
    driver_summary = decomposer.driver_summary(explanation)

    shap_path = os.path.join(TABLES_DIR, "shap_driver_summary.csv")
    driver_summary.to_csv(shap_path, index=False)
    print(f"SHAP driver summary saved to {shap_path}")
    print("\nTop drivers:")
    print(driver_summary.head(7).to_string(index=False))

    # ==========================================================================
    # 10. GENERATE CHARTS
    # ==========================================================================
    print("\n" + "="*60)
    print("STEP 10: Generating charts")
    print("="*60)

    charts.plot_history_forecast(y, result)
    charts.plot_coicop_heatmap(master_df)

    try:
        charts.plot_shap_waterfall(explanation, step=0)
    except Exception as e:
        print(f"Waterfall chart skipped: {e}")

    try:
        charts.plot_shap_contribution(monthly_contrib, explanation.base_value)
    except Exception as e:
        print(f"SHAP contribution chart skipped: {e}")

    charts.plot_sarima_diagnostics(sarima)
    charts.plot_sarima_coef(sarima)

    if sarima_cv is not None and gbm_cv is not None:
        ensemble_cv = pd.concat([
            ForecastEnsemble.compute_cv_metrics(sarima_cv, "SARIMA"),
            ForecastEnsemble.compute_cv_metrics(gbm_cv, "GBM"),
        ])
        charts.plot_model_comparison(ensemble_cv)

    # ==========================================================================
    # 11. PRINT SUMMARY
    # ==========================================================================
    print("\n" + "="*60)
    print("FORECAST SUMMARY")
    print("="*60)
    print(f"Last known CPI MoM%: {y.iloc[-1]:.2f}% ({y.index[-1].strftime('%B %Y')})")
    print(f"\n{horizon}-month forecast (ensemble):")
    for date, pt, lo, hi in zip(
        future_dates,
        result.point,
        result.lower_95,
        result.upper_95,
    ):
        print(f"  {date.strftime('%b %Y')}: {pt:+.2f}%  "
              f"[95% CI: {lo:+.2f}% to {hi:+.2f}%]")

    print(f"\nTop 3 inflation drivers (SHAP):")
    for _, row in driver_summary.head(3).iterrows():
        arrow = "up" if row["direction"] == "inflationary" else "down"
        print(f"  [{arrow}] {row['group']}: avg |SHAP| = {row['mean_abs_shap']:.4f} pp")

    print(f"\nAll outputs saved to:")
    print(f"  Tables:  {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print("="*60)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
