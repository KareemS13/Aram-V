"""
LightGBM forecaster using skforecast's ForecasterRecursive.

ForecasterRecursive handles:
  - Recursive multi-step forecasting (predicted values fed back as lag features)
  - Look-ahead-safe lag encoding
  - Integration with backtesting_forecaster for walk-forward CV

Key regularization settings for short series (~120 obs, ~40 features):
  - num_leaves=15 (shallow trees)
  - min_child_samples=10
  - lambda_l1, lambda_l2 regularization
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class GBMForecastResult:
    point: pd.Series
    feature_names: list[str]


class GBMForecaster:
    """
    LightGBM recursive multi-step forecaster.

    Parameters
    ----------
    lags        : list of lag integers for the target variable
    n_estimators: number of boosting rounds
    """

    DEFAULT_LAGS = [1, 2, 12]

    def __init__(
        self,
        lags: list[int] | None = None,
        n_estimators: int = 100,
        num_leaves: int = 8,
        learning_rate: float = 0.05,
        min_child_samples: int = 15,
        max_features: int = 8,
    ):
        self.lags = lags or self.DEFAULT_LAGS
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.max_features = max_features
        self.forecaster = None
        self._best_params = None
        self._selected_features: list[str] = []   # populated by _select_features

    def _build_forecaster(self, lags=None, **lgbm_kwargs):
        """Instantiate a fresh ForecasterRecursive."""
        from skforecast.recursive import ForecasterRecursive
        from lightgbm import LGBMRegressor

        lgbm_params = {
            "n_estimators":     self.n_estimators,
            "learning_rate":    self.learning_rate,
            "num_leaves":       self.num_leaves,
            "min_child_samples":self.min_child_samples,
            "max_depth":        4,       # cap tree depth — was unlimited
            "subsample":        0.7,     # was 0.8
            "colsample_bytree": 0.7,     # was 0.8
            "lambda_l1":        1.0,
            "lambda_l2":        1.0,
            "min_gain_to_split": 0.01,   # require meaningful gain to split
            "verbose":         -1,
            **lgbm_kwargs,
        }

        return ForecasterRecursive(
            estimator=LGBMRegressor(**lgbm_params),
            lags=lags or self.lags,
        )

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    def _select_features(
        self,
        y: pd.Series,
        exog: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Keep only the top-K exog columns most correlated with y.

        With ~66-104 training observations and 36 engineered features,
        using all features gives a ~1.6 obs/feature ratio which causes
        severe overfitting. Keeping max_features (default 12) improves
        the ratio to ~5-8x, within acceptable range for tree models.

        Selection criterion: absolute Pearson correlation with target.
        Structural break dummies are always kept regardless of correlation.
        """
        if exog is None or exog.empty:
            return exog

        # Align y and exog on common index
        common = y.index.intersection(exog.index)
        y_aligned   = y.loc[common]
        exog_aligned = exog.loc[common]

        # Always keep structural dummies — needed for correct prediction
        always_keep = [c for c in exog_aligned.columns
                       if any(k in c for k in ["covid", "ukraine", "rub_crisis"])]

        # Rank remaining features by |correlation| with target
        other_cols = [c for c in exog_aligned.columns if c not in always_keep]
        corrs = (
            exog_aligned[other_cols]
            .corrwith(y_aligned)
            .abs()
            .sort_values(ascending=False)
        )

        n_free = max(1, self.max_features - len(always_keep))
        selected = list(corrs.head(n_free).index) + always_keep

        self._selected_features = selected
        return exog[selected]

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        y: pd.Series,
        exog: pd.DataFrame | None = None,
    ) -> "GBMForecaster":
        """
        Fit the ForecasterRecursive.

        Parameters
        ----------
        y    : pd.Series, target variable (CPI MoM%), DatetimeIndex MS freq
        exog : pd.DataFrame, exogenous features aligned to y's index
        """
        exog_sel = self._select_features(y, exog) if exog is not None else None
        self.forecaster = self._build_forecaster()
        self.forecaster.fit(y=y, exog=exog_sel)
        print(f"GBM: fitted on {len(y)} obs, "
              f"{len(self.get_feature_names())} features "
              f"(selected {len(self._selected_features)} from {exog.shape[1] if exog is not None else 0}).")
        return self

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------

    def tune(
        self,
        y: pd.Series,
        exog: pd.DataFrame | None = None,
        n_splits: int = 5,
    ) -> dict:
        """
        Grid search over key hyperparameters using walk-forward CV.

        Searches over:
          - lags: [1,2,3,6,12] vs [1,2,3,6,12,24] (if enough data)
          - n_estimators: [100, 200, 300]
          - num_leaves: [10, 15, 31]

        Returns best params dict and updates self.
        """
        from skforecast.model_selection import grid_search_forecaster
        from sklearn.model_selection import TimeSeriesSplit
        from lightgbm import LGBMRegressor

        # Lag options constrained to small sets — avoids feature explosion
        lag_options = [[1, 2, 12], [1, 12]]

        # Regularisation-first grid — no deep trees or large ensembles
        param_grid = {
            "n_estimators": [50, 100, 150],
            "num_leaves":   [4, 8, 15],
        }

        best_mae = float("inf")
        best_params = {}

        for lags in lag_options:
            forecaster = self._build_forecaster(lags=lags)

            # Calculate initial training size for CV
            initial_train_size = max(int(len(y) * 0.6), max(lags) + 5)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    results = grid_search_forecaster(
                        forecaster=forecaster,
                        y=y,
                        exog=exog,
                        param_grid=param_grid,
                        steps=3,
                        metric="mean_absolute_error",
                        initial_train_size=initial_train_size,
                        refit=False,
                        verbose=False,
                    )

                    if results is not None and len(results) > 0:
                        best_row = results.iloc[0]
                        if best_row["mean_absolute_error"] < best_mae:
                            best_mae = best_row["mean_absolute_error"]
                            best_params = {
                                "lags": lags,
                                "n_estimators": best_row["n_estimators"],
                                "num_leaves":   best_row["num_leaves"],
                            }
                except Exception as e:
                    warnings.warn(f"GBM tuning failed for lags={lags}: {e}")

        if best_params:
            self.lags = best_params.pop("lags")
            self.n_estimators = best_params.get("n_estimators", self.n_estimators)
            self.num_leaves = best_params.get("num_leaves", self.num_leaves)
            self._best_params = best_params
            print(f"GBM tuning: best MAE={best_mae:.4f}, "
                  f"lags={self.lags}, params={best_params}")
        else:
            print("GBM tuning: no improvement found, using defaults.")

        # Refit with best params
        self.fit(y, exog)
        return self._best_params or {}

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(
        self,
        steps: int,
        exog_future: pd.DataFrame | None = None,
    ) -> GBMForecastResult:
        """
        Generate recursive multi-step forecast.

        Parameters
        ----------
        steps       : number of months to forecast
        exog_future : future exogenous values (shape: steps x n_exog_cols)

        Returns
        -------
        GBMForecastResult with point forecast as pd.Series
        """
        if self.forecaster is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Apply same feature selection used during fit
        if exog_future is not None and self._selected_features:
            cols = [c for c in self._selected_features if c in exog_future.columns]
            exog_future = exog_future[cols]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self.forecaster.predict(steps=steps, exog=exog_future)

        return GBMForecastResult(
            point=pred,
            feature_names=self.get_feature_names(),
        )

    # ------------------------------------------------------------------
    # Walk-forward cross-validation
    # ------------------------------------------------------------------

    def walk_forward_cv(
        self,
        y: pd.Series,
        exog: pd.DataFrame | None = None,
        eval_start: str = "2023-01-01",
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Walk-forward CV using skforecast's backtesting_forecaster.

        Returns
        -------
        pd.DataFrame with columns: date, horizon, actual, predicted, error
        """
        from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold

        if horizons is None:
            horizons = [1, 3, 6, 12]

        eval_idx = y.index.get_loc(
            y.index[y.index >= pd.Timestamp(eval_start)][0]
        )
        initial_train_size = eval_idx

        # Apply feature selection on the training portion only,
        # then pass the reduced exog for the full CV run
        exog_cv = exog
        if exog is not None:
            y_train_cv = y.iloc[:initial_train_size]
            exog_cv    = self._select_features(y_train_cv, exog)

        records_all = []
        for h in horizons:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    cv = TimeSeriesFold(
                        steps=h,
                        initial_train_size=initial_train_size,
                        refit=False,
                    )
                    _, preds = backtesting_forecaster(
                        forecaster=self._build_forecaster(),
                        y=y,
                        cv=cv,
                        exog=exog_cv,
                        metric="mean_absolute_error",
                        verbose=False,
                        show_progress=False,
                        suppress_warnings=True,
                    )
                    # preds has DatetimeIndex and column 'pred'
                    for date, row in preds.iterrows():
                        pred_val = row["pred"]
                        if date in y.index:
                            records_all.append({
                                "date":      date,
                                "horizon":   h,
                                "actual":    y[date],
                                "predicted": pred_val,
                                "error":     y[date] - pred_val,
                            })
                except Exception as e:
                    warnings.warn(f"GBM CV: failed for h={h}: {e}")

        return pd.DataFrame(records_all)

    # ------------------------------------------------------------------
    # Feature access (needed for SHAP)
    # ------------------------------------------------------------------

    def get_feature_names(self) -> list[str]:
        """
        Return feature names in the order the model sees them.
        Combines lag feature names + exogenous feature names.
        """
        if self.forecaster is None:
            return []
        try:
            return list(self.forecaster.X_train_features_names_out_)
        except AttributeError:
            pass
        try:
            return list(self.forecaster.estimator.feature_names_in_)
        except AttributeError:
            pass
        return []

    def get_training_matrix(
        self,
        y: pd.Series,
        exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Get the training feature matrix X that the model was trained on.
        Used as background dataset for SHAP TreeExplainer.
        """
        if self.forecaster is None:
            raise RuntimeError("Model not fitted.")
        X_train, _ = self.forecaster.create_train_X_y(y=y, exog=exog)
        return X_train

    @property
    def regressor(self):
        """Access the underlying fitted LightGBM regressor."""
        if self.forecaster is None:
            return None
        return self.forecaster.estimator
