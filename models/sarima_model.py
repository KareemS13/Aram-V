"""
SARIMAX model for Armenia CPI forecasting.

Workflow:
  1. pmdarima.auto_arima -> finds best (p,d,q)(P,D,Q,12) order by AIC
  2. statsmodels SARIMAX -> refit with found order for full diagnostics,
     coefficient table, and confidence intervals

Key design choices:
  - Always include ukraine_2022 and covid_2020 as exogenous dummies to
    prevent false non-stationarity detection in 2022
  - Max 5 exogenous variables to avoid overfitting on ~120 obs
  - Seasonal period m=12 (monthly data)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class ForecastResult:
    point: pd.Series
    lower_95: pd.Series
    upper_95: pd.Series
    lower_50: pd.Series
    upper_50: pd.Series


class SARIMAXModel:
    """
    Wrapper around pmdarima auto_arima + statsmodels SARIMAX.

    Parameters
    ----------
    seasonal_period : int, seasonal period (12 for monthly data)
    max_p, max_q    : max AR/MA order to search
    max_P, max_Q    : max seasonal AR/MA order to search
    """

    def __init__(
        self,
        seasonal_period: int = 12,
        max_p: int = 3,
        max_q: int = 3,
        max_P: int = 1,
        max_Q: int = 1,
    ):
        self.m = seasonal_period
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self._order = None
        self._seasonal_order = None
        self._fitted = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        endog: pd.Series,
        exog: pd.DataFrame | None = None,
    ) -> "SARIMAXModel":
        """
        Fit the SARIMAX model.

        Step 1: pmdarima.auto_arima to find optimal order.
        Step 2: statsmodels SARIMAX refit with that order.

        Parameters
        ----------
        endog : pd.Series, target variable (CPI MoM%)
        exog  : pd.DataFrame, exogenous variables aligned to endog index
        """
        import pmdarima as pm
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        endog_clean = endog.dropna()
        exog_clean = exog.loc[endog_clean.index] if exog is not None else None

        print("SARIMA: running auto_arima to find optimal order...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auto = pm.auto_arima(
                endog_clean,
                exogenous=exog_clean,
                m=self.m,
                stepwise=True,
                information_criterion="aic",
                max_p=self.max_p,
                max_q=self.max_q,
                max_P=self.max_P,
                max_Q=self.max_Q,
                max_d=2,
                max_D=1,
                seasonal=True,
                error_action="ignore",
                suppress_warnings=True,
            )

        self._order = auto.order
        self._seasonal_order = auto.seasonal_order
        print(f"SARIMA: selected order {self._order} x {self._seasonal_order}")

        # Refit with statsmodels for full inference output
        model = SARIMAX(
            endog_clean,
            exog=exog_clean,
            order=self._order,
            seasonal_order=self._seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fitted = model.fit(disp=False)

        print(f"SARIMA: fitted. AIC={self._fitted.aic:.1f}")
        return self

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(
        self,
        steps: int,
        exog_future: pd.DataFrame | None = None,
    ) -> ForecastResult:
        """
        Generate multi-step forecast with confidence intervals.

        Parameters
        ----------
        steps       : number of months to forecast
        exog_future : future exogenous values (required if model has exog)

        Returns
        -------
        ForecastResult with point, lower_95, upper_95, lower_50, upper_50
        """
        if self._fitted is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        pred = self._fitted.get_forecast(
            steps=steps,
            exog=exog_future,
        )
        mean = pred.predicted_mean
        ci_95 = pred.conf_int(alpha=0.05)
        ci_50 = pred.conf_int(alpha=0.50)

        return ForecastResult(
            point=mean,
            lower_95=ci_95.iloc[:, 0],
            upper_95=ci_95.iloc[:, 1],
            lower_50=ci_50.iloc[:, 0],
            upper_50=ci_50.iloc[:, 1],
        )

    # ------------------------------------------------------------------
    # Walk-forward cross-validation
    # ------------------------------------------------------------------

    def walk_forward_cv(
        self,
        endog: pd.Series,
        exog: pd.DataFrame | None = None,
        eval_start: str = "2023-01-01",
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Expanding-window walk-forward cross-validation.

        For each month from eval_start to the end of the series:
          - Fit on all data up to that month
          - Forecast 1, 3, 6, 12 steps ahead
          - Record actual vs. predicted

        Returns
        -------
        pd.DataFrame with columns: date, horizon, actual, predicted, error
        """
        if horizons is None:
            horizons = [1, 3, 6, 12]

        eval_idx = endog.index.get_loc(
            endog.index[endog.index >= pd.Timestamp(eval_start)][0]
        )

        records = []
        n = len(endog)

        # Run auto_arima ONCE on the initial training window to find the order.
        # All subsequent CV folds reuse this fixed order — avoids 38+ auto_arima
        # calls and makes CV 10-20x faster without materially affecting results.
        import pmdarima as pm
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        init_endog = endog.iloc[:eval_idx]
        init_exog  = exog.iloc[:eval_idx] if exog is not None else None
        print("SARIMA CV: finding order on initial training window...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auto = pm.auto_arima(
                init_endog,
                exogenous=init_exog,
                m=self.m,
                stepwise=True,
                information_criterion="aic",
                max_p=self.max_p, max_q=self.max_q,
                max_P=self.max_P, max_Q=self.max_Q,
                max_d=2, max_D=1,
                seasonal=True,
                error_action="ignore",
                suppress_warnings=True,
            )
        fixed_order = auto.order
        fixed_seasonal = auto.seasonal_order
        print(f"SARIMA CV: fixed order {fixed_order} x {fixed_seasonal}")

        for t in range(eval_idx, n):
            train_endog = endog.iloc[:t]
            train_exog = exog.iloc[:t] if exog is not None else None

            try:
                model = SARIMAX(
                    train_endog,
                    exog=train_exog,
                    order=fixed_order,
                    seasonal_order=fixed_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = model.fit(disp=False)

                # Wrap in a minimal object so forecast() works
                m = SARIMAXModel.__new__(SARIMAXModel)
                m._fitted = fitted
                m._order = fixed_order
                m._seasonal_order = fixed_seasonal
                m.m = self.m

                for h in horizons:
                    if t + h > n:
                        continue
                    exog_h = exog.iloc[t: t + h] if exog is not None else None
                    fc = m.forecast(steps=h, exog_future=exog_h)
                    pred_val = fc.point.iloc[-1]
                    actual_val = endog.iloc[t + h - 1]
                    records.append({
                        "date":      endog.index[t + h - 1],
                        "horizon":   h,
                        "actual":    actual_val,
                        "predicted": pred_val,
                        "error":     actual_val - pred_val,
                    })
            except Exception as e:
                warnings.warn(f"SARIMA CV: failed at t={t}: {e}")

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Diagnostics and reporting
    # ------------------------------------------------------------------

    def plot_diagnostics(self) -> plt.Figure:
        """Residual diagnostics: plot, ACF, PACF, QQ."""
        if self._fitted is None:
            raise RuntimeError("Model not fitted.")
        fig = self._fitted.plot_diagnostics(figsize=(12, 8))
        fig.suptitle("SARIMAX Residual Diagnostics", fontsize=13)
        plt.tight_layout()
        return fig

    def coef_table(self) -> pd.DataFrame:
        """
        Return coefficient summary as a clean DataFrame.

        Columns: Variable, Coefficient, Std Error, t-stat, p-value
        """
        if self._fitted is None:
            raise RuntimeError("Model not fitted.")
        summary = self._fitted.summary()
        # Extract the parameter table (table index 1 in statsmodels)
        params = self._fitted.params
        bse = self._fitted.bse
        tvalues = self._fitted.tvalues
        pvalues = self._fitted.pvalues

        df = pd.DataFrame({
            "Variable":    params.index,
            "Coefficient": params.values.round(4),
            "Std Error":   bse.values.round(4),
            "t-stat":      tvalues.values.round(3),
            "p-value":     pvalues.values.round(4),
        })
        return df

    def residuals(self) -> pd.Series:
        """Return model residuals."""
        if self._fitted is None:
            raise RuntimeError("Model not fitted.")
        return self._fitted.resid

    @property
    def aic(self) -> float:
        return self._fitted.aic if self._fitted else float("nan")

    @property
    def order(self) -> tuple:
        return self._order

    @property
    def seasonal_order(self) -> tuple:
        return self._seasonal_order
