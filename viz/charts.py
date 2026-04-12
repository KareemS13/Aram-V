"""
All visualization functions for the Armenia CPI forecasting pipeline.

Each function:
  - Returns a matplotlib Figure
  - Optionally saves to outputs/figures/ if save_path is provided
  - Is self-contained (no shared state)

Output files:
  01_history_forecast.png     Fan chart: history + 12m ensemble forecast
  02_coicop_heatmap.png       COICOP sub-index heatmap (last 24 months)
  03_shap_waterfall_m1.png    SHAP waterfall for month+1 forecast
  04_shap_contribution_12m.png Stacked bar driver attribution over horizon
  05_sarima_diagnostics.png   SARIMA residual diagnostics
  06_model_comparison.png     Walk-forward CV: MAE by model and horizon
  07_sarima_coefficients.png  SARIMAX coefficient table
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns

from config import COICOP_LABELS, FIGURES_DIR


def _save(fig: plt.Figure, filename: str, save_path: str | None = None) -> None:
    """Save figure to outputs/figures/ directory."""
    if save_path is None:
        save_path = FIGURES_DIR
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 1. History + Forecast Fan Chart
# ---------------------------------------------------------------------------

def plot_history_forecast(
    history: pd.Series,
    ensemble_result,
    title: str = "Armenia Headline CPI — Month-on-Month % Change",
    save: bool = True,
) -> plt.Figure:
    """
    Fan chart showing historical CPI MoM% and 12-month ensemble forecast
    with 50% and 95% confidence bands.

    Parameters
    ----------
    history        : pd.Series, historical CPI MoM% (DatetimeIndex)
    ensemble_result: EnsembleResult from models/ensemble.py
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    # Historical series (last 48 months for readability)
    hist_plot = history.iloc[-48:]
    ax.plot(hist_plot.index, hist_plot.values,
            color="#455A64", linewidth=1.8, label="Historical CPI MoM%")

    # Forecast
    fc = ensemble_result
    ax.plot(fc.point.index, fc.point.values,
            color="#1976D2", linewidth=2.0, label="Ensemble Forecast")

    # 50% CI
    ax.fill_between(fc.point.index, fc.lower_50, fc.upper_50,
                    alpha=0.35, color="#1976D2", label="50% CI")
    # 95% CI
    ax.fill_between(fc.point.index, fc.lower_95, fc.upper_95,
                    alpha=0.15, color="#1976D2", label="95% CI")

    # Forecast start line
    forecast_start = fc.point.index[0]
    ax.axvline(forecast_start, color="#E53935", linestyle="--",
               linewidth=1.2, alpha=0.7)
    ax.text(forecast_start, ax.get_ylim()[1] * 0.95, "  Forecast start",
            color="#E53935", fontsize=9, va="top")

    # Zero line
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylabel("MoM % Change", fontsize=11)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b\n%Y"))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # Annotate model weights
    ax.text(
        0.99, 0.04,
        f"SARIMA w={fc.sarima_weight:.2f} | GBM w={fc.gbm_weight:.2f}",
        transform=ax.transAxes, fontsize=8, ha="right",
        color="#777777",
    )

    plt.tight_layout()
    if save:
        _save(fig, "01_history_forecast.png")
    return fig


# ---------------------------------------------------------------------------
# 2. COICOP Heatmap
# ---------------------------------------------------------------------------

def plot_coicop_heatmap(
    cpi_df: pd.DataFrame,
    months: int = 24,
    save: bool = True,
) -> plt.Figure:
    """
    Heatmap: rows = 12 COICOP divisions, columns = last N months.
    Cell values = MoM% change. RdYlGn color scale (red = high inflation).

    Parameters
    ----------
    cpi_df : DataFrame with COICOP columns (from master_df or ingest_cpi)
    months : number of recent months to display
    """
    coicop_cols = [c for c in COICOP_LABELS if c in cpi_df.columns and c != "cpi_headline"]
    plot_df = cpi_df[coicop_cols].iloc[-months:].T

    # Rename rows to human-readable labels
    plot_df.index = [COICOP_LABELS.get(c, c) for c in plot_df.index]

    vmax = max(abs(plot_df.values[~np.isnan(plot_df.values)]).max(), 1.0)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        plot_df,
        ax=ax,
        cmap="RdYlGn_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 7},
        linewidths=0.4,
        cbar_kws={"label": "MoM % Change", "shrink": 0.6},
    )
    ax.set_title("Armenia CPI by COICOP Division (MoM% Change)", fontsize=12, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(
        [pd.Timestamp(t.get_text()).strftime("%b\n%Y")
         if pd.notna(pd.Timestamp(t.get_text())) else t.get_text()
         for t in ax.get_xticklabels()],
        fontsize=8,
    )
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, rotation=0)

    plt.tight_layout()
    if save:
        _save(fig, "02_coicop_heatmap.png")
    return fig


# ---------------------------------------------------------------------------
# 3. SHAP Waterfall (Month +1)
# ---------------------------------------------------------------------------

def plot_shap_waterfall(
    explanation,
    step: int = 0,
    save: bool = True,
) -> plt.Figure:
    """
    SHAP waterfall chart for one forecast step.
    Delegates to InflationDecomposer.waterfall_chart().
    """
    from explain.shap_decompose import InflationDecomposer

    date = explanation.forecast_dates[step]
    title = f"SHAP Driver Attribution — {date.strftime('%B %Y')} Forecast"

    import shap
    shap_vals = explanation.shap_values[step]
    base = explanation.base_value

    expl_obj = shap.Explanation(
        values=shap_vals,
        base_values=base,
        feature_names=explanation.feature_names,
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(expl_obj, max_display=15, show=False)
    plt.title(title, fontsize=12, pad=12)
    plt.tight_layout()

    if save:
        _save(fig, "03_shap_waterfall_m1.png")
    return fig


# ---------------------------------------------------------------------------
# 4. SHAP Stacked Contribution Bar (12-month horizon)
# ---------------------------------------------------------------------------

def plot_shap_contribution(
    monthly_contrib: pd.DataFrame,
    base_value: float,
    title: str = "Inflation Forecast: Driver Contributions (12-Month Horizon)",
    save: bool = True,
) -> plt.Figure:
    """
    Stacked bar chart of SHAP driver contributions over the forecast horizon.
    Positive = inflationary, negative = disinflationary.
    """
    from config import SHAP_GROUP_COLORS

    fig, ax = plt.subplots(figsize=(13, 6))

    dates = monthly_contrib.index
    x = np.arange(len(dates))

    pos_bottom = np.zeros(len(dates))
    neg_bottom = np.zeros(len(dates))
    handles = []

    for group in monthly_contrib.columns:
        vals = monthly_contrib[group].values
        color = SHAP_GROUP_COLORS.get(group, "#888888")

        pos_vals = np.where(vals > 0, vals, 0)
        neg_vals = np.where(vals < 0, vals, 0)

        bar = ax.bar(x, pos_vals, bottom=pos_bottom, color=color,
                     alpha=0.85, width=0.65)
        ax.bar(x, neg_vals, bottom=neg_bottom, color=color,
               alpha=0.85, width=0.65)

        pos_bottom = pos_bottom + pos_vals
        neg_bottom = neg_bottom + neg_vals

        handles.append(mpatches.Patch(color=color, label=group))

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%b\n%Y") for d in dates], fontsize=9)
    ax.set_ylabel("SHAP contribution (pp MoM%)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        _save(fig, "04_shap_contribution_12m.png")
    return fig


# ---------------------------------------------------------------------------
# 5. SARIMA Residual Diagnostics
# ---------------------------------------------------------------------------

def plot_sarima_diagnostics(
    sarima_model,
    save: bool = True,
) -> plt.Figure:
    """Wrapper: calls statsmodels plot_diagnostics on the fitted SARIMAX."""
    fig = sarima_model.plot_diagnostics()
    if save:
        _save(fig, "05_sarima_diagnostics.png")
    return fig


# ---------------------------------------------------------------------------
# 6. Model Comparison (Walk-forward CV MAE)
# ---------------------------------------------------------------------------

def plot_model_comparison(
    metrics_df: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """
    Line chart comparing walk-forward CV MAE across models and horizons.

    Parameters
    ----------
    metrics_df : pd.DataFrame with columns: model, horizon, MAE, RMSE, MAPE
                 (output of ForecastEnsemble.compute_cv_metrics, concatenated)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    models = metrics_df["model"].unique()
    colors = ["#1976D2", "#E53935", "#2E7D32"]
    markers = ["o", "s", "^"]

    for metric, ax in zip(["MAE", "RMSE"], axes):
        for i, model in enumerate(models):
            sub = metrics_df[metrics_df["model"] == model].sort_values("horizon")
            ax.plot(sub["horizon"], sub[metric],
                    label=model, color=colors[i % len(colors)],
                    marker=markers[i % len(markers)], linewidth=1.8,
                    markersize=7)
        ax.set_title(f"Walk-Forward CV — {metric} by Forecast Horizon", fontsize=11)
        ax.set_xlabel("Horizon (months)", fontsize=10)
        ax.set_ylabel(f"{metric} (pp)", fontsize=10)
        ax.set_xticks([1, 3, 6, 12])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("Model Comparison: Armenia CPI Forecasting", fontsize=12, y=1.02)
    plt.tight_layout()
    if save:
        _save(fig, "06_model_comparison.png")
    return fig


# ---------------------------------------------------------------------------
# 7. SARIMA Coefficient Table
# ---------------------------------------------------------------------------

def plot_sarima_coef(
    sarima_model,
    save: bool = True,
) -> plt.Figure:
    """
    Render SARIMAX coefficient table as a matplotlib figure.
    Useful for presenting to economists/policymakers.
    """
    coef_df = sarima_model.coef_table()

    fig, ax = plt.subplots(figsize=(10, max(3, len(coef_df) * 0.45 + 1)))
    ax.axis("off")

    col_labels = coef_df.columns.tolist()
    cell_text = coef_df.values.tolist()

    # Color significant rows (p-value < 0.05) in light green
    cell_colors = []
    p_idx = col_labels.index("p-value")
    for row in cell_text:
        try:
            p = float(row[p_idx])
            bg = "#E8F5E9" if p < 0.05 else "white"
        except (ValueError, TypeError):
            bg = "white"
        cell_colors.append([bg] * len(col_labels))

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Header styling
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    order = sarima_model.order
    seasonal = sarima_model.seasonal_order
    ax.set_title(
        f"SARIMAX Coefficients — Order {order} x Seasonal {seasonal}\n"
        f"AIC = {sarima_model.aic:.1f}   (green = p < 0.05)",
        fontsize=10, pad=12,
    )

    plt.tight_layout()
    if save:
        _save(fig, "07_sarima_coefficients.png")
    return fig
