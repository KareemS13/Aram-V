"""
Ensemble combiner for SARIMA and GBM forecasts.

Uses inverse-MAE weighting derived from walk-forward cross-validation results.
SARIMA typically outperforms at h=1–3; GBM at h=6–12.
If CV results are unavailable, falls back to default weights [0.4, 0.6].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.sarima_model import ForecastResult


@dataclass
class EnsembleResult:
    point: pd.Series
    lower_95: pd.Series
    upper_95: pd.Series
    lower_50: pd.Series
    upper_50: pd.Series
    sarima_weight: float
    gbm_weight: float


class ForecastEnsemble:
    """
    Combine SARIMA and GBM forecasts using inverse-MAE weights.

    Parameters
    ----------
    sarima_weight : fixed SARIMA weight (None = derive from CV)
    gbm_weight    : fixed GBM weight (None = derive from CV)
    """

    DEFAULT_SARIMA_WEIGHT = 0.4
    DEFAULT_GBM_WEIGHT = 0.6

    def __init__(
        self,
        sarima_weight: float | None = None,
        gbm_weight: float | None = None,
    ):
        self._sarima_w = sarima_weight
        self._gbm_w = gbm_weight

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def compute_weights_from_cv(
        self,
        sarima_cv: pd.DataFrame,
        gbm_cv: pd.DataFrame,
        horizon: int | None = None,
    ) -> tuple[float, float]:
        """
        Compute inverse-MAE weights from walk-forward CV results.

        Parameters
        ----------
        sarima_cv : DataFrame with columns: date, horizon, actual, predicted
        gbm_cv    : same structure for GBM
        horizon   : if provided, compute weights for that specific horizon only

        Returns
        -------
        (sarima_weight, gbm_weight) — normalized to sum to 1
        """
        def get_mae(cv_df, h):
            if h is not None:
                cv_df = cv_df[cv_df["horizon"] == h]
            if cv_df.empty:
                return float("nan")
            return (cv_df["actual"] - cv_df["predicted"]).abs().mean()

        mae_sarima = get_mae(sarima_cv, horizon)
        mae_gbm = get_mae(gbm_cv, horizon)

        if np.isnan(mae_sarima) or np.isnan(mae_gbm) or mae_sarima == 0 or mae_gbm == 0:
            print("Ensemble: using default weights (CV data insufficient).")
            return self.DEFAULT_SARIMA_WEIGHT, self.DEFAULT_GBM_WEIGHT

        inv_sarima = 1.0 / mae_sarima
        inv_gbm = 1.0 / mae_gbm
        total = inv_sarima + inv_gbm

        w_sarima = round(inv_sarima / total, 3)
        w_gbm = round(inv_gbm / total, 3)

        print(f"Ensemble: MAE_SARIMA={mae_sarima:.4f}, MAE_GBM={mae_gbm:.4f} "
              f"-> weights SARIMA={w_sarima:.3f}, GBM={w_gbm:.3f}")

        return w_sarima, w_gbm

    # ------------------------------------------------------------------
    # Combining forecasts
    # ------------------------------------------------------------------

    def combine(
        self,
        sarima_result: ForecastResult,
        gbm_point: pd.Series,
        sarima_cv: pd.DataFrame | None = None,
        gbm_cv: pd.DataFrame | None = None,
    ) -> EnsembleResult:
        """
        Combine SARIMA and GBM forecasts into an ensemble.

        Confidence intervals come from SARIMA (parametric), widened by
        the additional uncertainty from the GBM residual std in CV.

        Parameters
        ----------
        sarima_result : ForecastResult from SARIMAXModel.forecast()
        gbm_point     : pd.Series point forecast from GBMForecaster.forecast()
        sarima_cv     : optional CV results for inverse-MAE weighting
        gbm_cv        : optional CV results for inverse-MAE weighting

        Returns
        -------
        EnsembleResult
        """
        # Determine weights
        if self._sarima_w is not None and self._gbm_w is not None:
            w_sarima, w_gbm = self._sarima_w, self._gbm_w
        elif sarima_cv is not None and gbm_cv is not None:
            w_sarima, w_gbm = self.compute_weights_from_cv(sarima_cv, gbm_cv)
            self._sarima_w = w_sarima
            self._gbm_w = w_gbm
        else:
            w_sarima = self.DEFAULT_SARIMA_WEIGHT
            w_gbm = self.DEFAULT_GBM_WEIGHT
            print(f"Ensemble: no CV data provided — using defaults "
                  f"(SARIMA={w_sarima}, GBM={w_gbm}).")

        # Align indices
        idx = sarima_result.point.index
        gbm_aligned = gbm_point.reindex(idx)

        # Weighted point forecast
        point = w_sarima * sarima_result.point + w_gbm * gbm_aligned

        # CI: use SARIMA's parametric CI, optionally widen by GBM disagreement
        gbm_spread = (gbm_aligned - sarima_result.point).abs()
        extra_uncertainty = gbm_spread * w_gbm

        lower_95 = sarima_result.lower_95 - extra_uncertainty
        upper_95 = sarima_result.upper_95 + extra_uncertainty
        lower_50 = sarima_result.lower_50 - extra_uncertainty * 0.5
        upper_50 = sarima_result.upper_50 + extra_uncertainty * 0.5

        return EnsembleResult(
            point=point,
            lower_95=lower_95,
            upper_95=upper_95,
            lower_50=lower_50,
            upper_50=upper_50,
            sarima_weight=w_sarima,
            gbm_weight=w_gbm,
        )

    # ------------------------------------------------------------------
    # CV metrics summary
    # ------------------------------------------------------------------

    @staticmethod
    def compute_cv_metrics(cv_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Compute MAE, RMSE, MAPE per forecast horizon.

        Parameters
        ----------
        cv_df      : DataFrame with columns: date, horizon, actual, predicted
        model_name : label for this model (e.g. 'SARIMA', 'GBM', 'Ensemble')

        Returns
        -------
        pd.DataFrame with columns: model, horizon, MAE, RMSE, MAPE
        """
        rows = []
        for h, grp in cv_df.groupby("horizon"):
            errors = grp["actual"] - grp["predicted"]
            mae  = errors.abs().mean()
            rmse = np.sqrt((errors ** 2).mean())
            # MAPE: skip if actual has zeros
            actual_nonzero = grp["actual"][grp["actual"] != 0]
            if len(actual_nonzero) > 0:
                preds_nonzero = grp.loc[actual_nonzero.index, "predicted"]
                mape = ((actual_nonzero - preds_nonzero).abs() / actual_nonzero.abs()).mean() * 100
            else:
                mape = float("nan")

            rows.append({
                "model":   model_name,
                "horizon": h,
                "MAE":     round(mae, 4),
                "RMSE":    round(rmse, 4),
                "MAPE":    round(mape, 2),
            })

        return pd.DataFrame(rows)
