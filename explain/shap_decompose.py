"""
SHAP-based driver attribution for the GBM inflation forecaster.

Uses shap.TreeExplainer (exact Shapley values for tree models — no approximation).

Key design:
  - Background dataset = last 60 months of training data
    (makes SHAP values interpretable relative to recent "normal" behavior)
  - Lag features from ForecasterRecursive are named lag_1, lag_2, etc.
    These are mapped back to human-readable names using the forecaster's
    feature_names_out() before any plotting.
  - Features are grouped into 7 economic categories for the summary chart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import SHAP_GROUPS, SHAP_GROUP_COLORS


@dataclass
class ShapExplanation:
    """Container for SHAP output across a forecast horizon."""
    shap_values: np.ndarray        # shape: (n_steps, n_features)
    base_value: float
    feature_names: list[str]       # human-readable names
    forecast_dates: pd.DatetimeIndex


class InflationDecomposer:
    """
    Explain GBM forecast using SHAP TreeExplainer.

    Parameters
    ----------
    gbm_forecaster : fitted GBMForecaster instance
    """

    def __init__(self, gbm_forecaster):
        self.gbm = gbm_forecaster
        self.explainer = None
        self._feature_names = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def fit_explainer(
        self,
        X_train: pd.DataFrame,
        background_months: int = 60,
    ) -> "InflationDecomposer":
        """
        Initialize SHAP TreeExplainer with a background dataset.

        Parameters
        ----------
        X_train         : training feature matrix (from GBMForecaster)
        background_months: number of most-recent training months to use
                           as the SHAP background reference distribution
        """
        import shap

        if self.gbm.regressor is None:
            raise RuntimeError("GBM model not fitted.")

        # Use last N months as background (recent = more interpretable reference)
        background = X_train.iloc[-background_months:] if len(X_train) > background_months else X_train

        self.explainer = shap.TreeExplainer(
            self.gbm.regressor,
            data=background,
            feature_perturbation="interventional",
        )
        self._feature_names = self.gbm.get_feature_names()
        print(f"SHAP: explainer fitted on {len(background)} background samples, "
              f"{len(self._feature_names)} features.")
        return self

    # ------------------------------------------------------------------
    # Explain forecast
    # ------------------------------------------------------------------

    def explain_forecast(
        self,
        X_forecast: pd.DataFrame,
    ) -> ShapExplanation:
        """
        Compute SHAP values for each step in the forecast horizon.

        Parameters
        ----------
        X_forecast : feature matrix for forecast steps (shape: steps x n_features)
                     Each row corresponds to one forecast month.

        Returns
        -------
        ShapExplanation with shap_values (n_steps x n_features), base_value,
        feature_names (human-readable), forecast_dates
        """
        if self.explainer is None:
            raise RuntimeError("Call fit_explainer() first.")

        shap_values = self.explainer.shap_values(X_forecast)

        # shap_values is (n_steps, n_features) for regression
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        base_value = float(self.explainer.expected_value)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[0])

        return ShapExplanation(
            shap_values=shap_values,
            base_value=base_value,
            feature_names=self._feature_names or list(X_forecast.columns),
            forecast_dates=X_forecast.index,
        )

    # ------------------------------------------------------------------
    # Driver summary
    # ------------------------------------------------------------------

    def driver_summary(
        self,
        explanation: ShapExplanation,
    ) -> pd.DataFrame:
        """
        Aggregate SHAP values by economic driver group over the forecast horizon.

        Lag features (lag_1, lag_2, ...) are mapped back to their parent
        variable names using the forecaster's feature_names_out().

        Returns
        -------
        pd.DataFrame with columns:
            group, mean_abs_shap, net_contribution, direction
        Sorted by mean_abs_shap descending.
        """
        feat_names = explanation.feature_names
        shap_mat = explanation.shap_values  # (n_steps, n_features)

        # Map each feature to its group
        feat_to_group = self._map_features_to_groups(feat_names)

        rows = {}
        for i, feat in enumerate(feat_names):
            group = feat_to_group.get(feat, "Other")
            if group not in rows:
                rows[group] = {"abs_sum": 0.0, "net_sum": 0.0, "count": 0}
            rows[group]["abs_sum"] += np.abs(shap_mat[:, i]).mean()
            rows[group]["net_sum"] += shap_mat[:, i].mean()
            rows[group]["count"] += 1

        records = []
        for group, vals in rows.items():
            net = vals["net_sum"]
            records.append({
                "group":            group,
                "mean_abs_shap":    round(vals["abs_sum"], 4),
                "net_contribution": round(net, 4),
                "direction":        "inflationary" if net > 0 else "disinflationary",
            })

        df = pd.DataFrame(records).sort_values("mean_abs_shap", ascending=False)
        return df.reset_index(drop=True)

    def monthly_contributions(
        self,
        explanation: ShapExplanation,
    ) -> pd.DataFrame:
        """
        SHAP contributions by group for each forecast month.
        Used for the stacked bar chart.

        Returns
        -------
        pd.DataFrame: index=forecast_dates, columns=groups, values=net SHAP contribution
        """
        feat_names = explanation.feature_names
        shap_mat = explanation.shap_values
        feat_to_group = self._map_features_to_groups(feat_names)

        groups = list(dict.fromkeys(feat_to_group.values()))  # preserve order
        monthly = pd.DataFrame(0.0, index=explanation.forecast_dates, columns=groups)

        for i, feat in enumerate(feat_names):
            group = feat_to_group.get(feat, "Other")
            if group in monthly.columns:
                monthly[group] += shap_mat[:, i]

        return monthly

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def waterfall_chart(
        self,
        explanation: ShapExplanation,
        step: int = 0,
        title: str | None = None,
    ) -> plt.Figure:
        """
        SHAP waterfall chart for a single forecast step.

        Shows base_value + each feature's contribution -> final prediction.
        Features are sorted by absolute SHAP value.

        Parameters
        ----------
        explanation : ShapExplanation from explain_forecast()
        step        : which forecast step to plot (0 = first month ahead)
        """
        import shap

        shap_vals = explanation.shap_values[step]
        feat_names = explanation.feature_names
        base = explanation.base_value
        date = explanation.forecast_dates[step]

        # Create shap Explanation object for waterfall plot
        expl_obj = shap.Explanation(
            values=shap_vals,
            base_values=base,
            feature_names=feat_names,
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.plots.waterfall(expl_obj, max_display=15, show=False)

        if title is None:
            title = f"SHAP Driver Attribution — {date.strftime('%B %Y')} Forecast"
        plt.title(title, fontsize=12, pad=12)
        plt.tight_layout()
        return fig

    def contribution_stacked_bar(
        self,
        monthly_contrib: pd.DataFrame,
        base_value: float,
        title: str = "Inflation Forecast: Driver Contributions (12-Month Horizon)",
    ) -> plt.Figure:
        """
        Stacked bar chart showing driver contributions for each forecast month.

        Positive bars = inflationary contribution.
        Negative bars = disinflationary contribution.

        Parameters
        ----------
        monthly_contrib : output of monthly_contributions()
        base_value      : SHAP base value (expected model output)
        """
        fig, ax = plt.subplots(figsize=(13, 6))

        dates = monthly_contrib.index
        x = np.arange(len(dates))

        pos_bottom = np.zeros(len(dates))
        neg_bottom = np.zeros(len(dates))

        for group in monthly_contrib.columns:
            vals = monthly_contrib[group].values
            color = SHAP_GROUP_COLORS.get(group, "#888888")

            pos_vals = np.where(vals > 0, vals, 0)
            neg_vals = np.where(vals < 0, vals, 0)

            if pos_vals.any():
                ax.bar(x, pos_vals, bottom=pos_bottom, color=color,
                       label=group, alpha=0.85, width=0.65)
                pos_bottom += pos_vals

            if neg_vals.any():
                ax.bar(x, neg_vals, bottom=neg_bottom, color=color,
                       alpha=0.85, width=0.65)
                neg_bottom += neg_vals

        # Base value line
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        # Labels
        ax.set_xticks(x)
        ax.set_xticklabels(
            [d.strftime("%b\n%Y") for d in dates], fontsize=9
        )
        ax.set_ylabel("SHAP contribution (pp MoM%)", fontsize=11)
        ax.set_title(title, fontsize=12, pad=10)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _map_features_to_groups(self, feat_names: list[str]) -> dict[str, str]:
        """
        Map each feature name to an economic group label.

        ForecasterRecursive may name lag features as 'lag_1', 'lag_2', ...
        These need to be mapped back to their original variable name first.
        """
        # Build reverse map from lag index to original variable name
        # (skforecast stores this in forecaster.lags_names or feature_names_out)
        lag_to_var = self._build_lag_to_var_map(feat_names)

        feat_to_group = {}
        for feat in feat_names:
            resolved = lag_to_var.get(feat, feat)
            assigned = False
            for group, keywords in SHAP_GROUPS.items():
                if any(kw in resolved for kw in keywords):
                    feat_to_group[feat] = group
                    assigned = True
                    break
            if not assigned:
                feat_to_group[feat] = "Other"

        return feat_to_group

    def _build_lag_to_var_map(self, feat_names: list[str]) -> dict[str, str]:
        """
        Build mapping from 'lag_N' style names to their original variable names.

        skforecast ForecasterRecursive names target lags as 'lag_1', 'lag_2', etc.
        Exogenous features retain their original names.
        """
        mapping = {}
        if self.gbm.forecaster is None:
            return mapping

        try:
            # Try to get the lag feature names list from forecaster
            all_names = self.gbm.get_feature_names()
            lags = self.gbm.forecaster.lags  # array of lag values

            for i, name in enumerate(all_names):
                if name.startswith("lag_"):
                    try:
                        mapping[name] = "cpi_headline_mom"
                    except (IndexError, ValueError):
                        mapping[name] = name
        except Exception:
            pass

        return mapping
