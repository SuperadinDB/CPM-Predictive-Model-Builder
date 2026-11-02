"""Direct multi-step forecasting for a single regularly spaced time series."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from cpm.models import _estimator


FREQUENCIES = {
    "Hourly": "h", "Daily": "D", "Weekly (Monday)": "W-MON",
    "Monthly": "MS", "Quarterly": "QS", "Yearly": "YS",
}


def parse_positive_ints(raw: str, *, allow_zero: bool = False) -> list[int]:
    """Parse comma-separated lag/window values and reject ambiguous input."""
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("Enter comma-separated whole numbers.") from exc
    if not values:
        raise ValueError("Enter at least one whole number.")
    minimum = 0 if allow_zero else 1
    if any(value < minimum for value in values):
        raise ValueError(f"Values must be {minimum} or greater.")
    return sorted(set(values))


def prepare_series(frame: pd.DataFrame, time_column: str, target_column: str,
                   frequency: str | None, aggregation: str,
                   missing_policy: str) -> tuple[pd.Series, str]:
    """Parse, order, aggregate duplicate periods, regularize, and fill gaps."""
    if time_column == target_column:
        raise ValueError("Choose different columns for time and target.")
    if time_column not in frame or target_column not in frame:
        raise ValueError("The selected time or target column is missing.")
    working = pd.DataFrame({
        "time": pd.to_datetime(frame[time_column], errors="coerce"),
        "value": pd.to_numeric(frame[target_column], errors="coerce"),
    }).dropna(subset=["time", "value"])
    if working.empty:
        raise ValueError("No valid date and numeric target pairs were found.")
    working = working.sort_values("time")
    series = working.groupby("time", sort=True)["value"].agg(aggregation)
    if len(series) < 3:
        raise ValueError("At least three valid time points are required.")
    if frequency is None:
        frequency = pd.infer_freq(series.index)
        if not frequency:
            raise ValueError("FREQUENCY_REQUIRED")
    try:
        series = series.resample(frequency).agg(aggregation).astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Could not regularize this series: {exc}") from exc
    if missing_policy == "zero":
        series = series.fillna(0.0)
    elif missing_policy == "forward_fill":
        series = series.ffill()
    return series.rename(target_column), frequency


def _features_for_lead(series: pd.Series, frequency: str, lags: list[int],
                       windows: list[int], lead: int) -> pd.DataFrame:
    """Build leakage-safe lag/rolling features and known future calendar signals."""
    features = pd.DataFrame(index=series.index)
    for lag in lags:
        features[f"lag_{lag}"] = series.shift(lag)
    shifted = series.shift(1)
    for window in windows:
        rolling = shifted.rolling(window=window, min_periods=window)
        features[f"rolling_mean_{window}"] = rolling.mean()
        features[f"rolling_std_{window}"] = rolling.std().fillna(0.0)

    offset = pd.tseries.frequencies.to_offset(frequency)
    target_dates = pd.date_range(series.index[0], periods=len(series) + lead,
                                 freq=offset)[lead:]
    features["trend_index"] = np.arange(len(series), dtype=float) + lead
    month_angle = 2 * np.pi * (target_dates.month - 1) / 12
    weekday_angle = 2 * np.pi * target_dates.day_of_week / 7
    year_angle = 2 * np.pi * (target_dates.day_of_year - 1) / 365.25
    features["month_sin"], features["month_cos"] = np.sin(month_angle), np.cos(month_angle)
    features["weekday_sin"], features["weekday_cos"] = np.sin(weekday_angle), np.cos(weekday_angle)
    features["year_day_sin"], features["year_day_cos"] = np.sin(year_angle), np.cos(year_angle)
    return features


def _forecast_pipeline(model_name: str, cfg: dict[str, Any]) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    scaler = {"StandardScaler": StandardScaler, "RobustScaler": RobustScaler}.get(cfg.get("scaler"))
    if scaler:
        steps.append(("scaler", scaler()))
    steps.append(("model", _estimator(model_name, "Regression", cfg)))
    return Pipeline(steps)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = np.abs(actual) + np.abs(predicted)
    smape = np.mean(np.divide(200 * np.abs(actual - predicted), denominator,
                              out=np.zeros_like(actual), where=denominator != 0))
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "sMAPE (%)": float(smape),
        "R²": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
    }


def train_forecast(frame: pd.DataFrame, *, time_column: str, target_column: str,
                   frequency: str | None, aggregation: str, missing_policy: str,
                   horizon: int, lags: list[int], windows: list[int],
                   test_size: float, model_name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a direct model per forecast lead, then refit each lead on all history."""
    if horizon < 1:
        raise ValueError("The forecast horizon must be at least one period.")
    series, resolved_frequency = prepare_series(
        frame, time_column, target_column, frequency, aggregation, missing_policy)
    if max(lags + windows) >= len(series):
        raise ValueError("The largest lag or rolling window must be smaller than the series length.")
    n_rows = len(series)
    test_origins = max(1, int(round((n_rows - horizon) * test_size)))
    test_start = n_rows - horizon - test_origins
    min_history = max(max(lags), max(windows)) + 8
    if test_start <= min_history:
        raise ValueError("TOO_LITTLE_HISTORY")

    holdout_parts: list[pd.DataFrame] = []
    horizon_scores: list[dict[str, Any]] = []
    future_models: dict[int, Pipeline] = {}
    future_rows: list[dict[str, Any]] = []
    future_dates = pd.date_range(series.index[-1], periods=horizon + 1,
                                 freq=resolved_frequency)[1:]

    for lead in range(1, horizon + 1):
        X = _features_for_lead(series, resolved_frequency, lags, windows, lead)
        y = series.shift(-lead)
        valid = X.notna().all(axis=1) & y.notna()
        origin_positions = np.arange(n_rows)
        train_mask = valid.to_numpy() & ((origin_positions + lead) < test_start)
        test_mask = valid.to_numpy() & (origin_positions >= test_start) & (
            origin_positions < n_rows - horizon)
        train_indices = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(test_mask)
        if len(train_indices) < 8 or not len(test_indices):
            raise ValueError("TOO_LITTLE_HISTORY")

        evaluation_model = _forecast_pipeline(model_name, cfg)
        # y is already shifted to the label at origin + lead.
        evaluation_model.fit(X.iloc[train_indices], y.iloc[train_indices].to_numpy())
        predictions = evaluation_model.predict(X.iloc[test_indices])
        actual = series.iloc[test_indices + lead].to_numpy(dtype=float)
        naive = series.iloc[test_indices].to_numpy(dtype=float)
        target_times = series.index[test_indices + lead]
        origins = series.index[test_indices]
        holdout_parts.append(pd.DataFrame({
            "forecast_origin": origins, "target_time": target_times, "horizon": lead,
            "actual": actual, "prediction": predictions, "last_value_baseline": naive,
        }))
        horizon_scores.append({"horizon": lead, **_metrics(actual, predictions),
            "baseline_MAE": float(mean_absolute_error(actual, naive)),
            "baseline_RMSE": float(np.sqrt(mean_squared_error(actual, naive)))})

        # Refit each lead on every labeled point available before the forecast origin.
        final_indices = np.flatnonzero(valid.to_numpy())
        final_model = _forecast_pipeline(model_name, cfg)
        final_model.fit(X.iloc[final_indices], y.iloc[final_indices].to_numpy())
        final_features = _features_for_lead(series, resolved_frequency, lags, windows, lead).iloc[[-1]]
        if final_features.isna().any(axis=None):
            raise ValueError("LATEST_PERIOD_HAS_GAPS")
        future_prediction = float(final_model.predict(final_features)[0])
        future_models[lead] = final_model
        future_rows.append({"target_time": future_dates[lead - 1], "horizon": lead,
                            "prediction": future_prediction})

    holdout = pd.concat(holdout_parts, ignore_index=True)
    overall = _metrics(holdout["actual"].to_numpy(), holdout["prediction"].to_numpy())
    baseline = _metrics(holdout["actual"].to_numpy(),
                        holdout["last_value_baseline"].to_numpy())
    return {
        "model_name": model_name, "time_column": time_column, "target": target_column,
        "frequency": resolved_frequency, "aggregation": aggregation,
        "missing_policy": missing_policy, "horizon": horizon, "lags": list(lags),
        "windows": list(windows), "series": series, "models_by_lead": future_models,
        "future": pd.DataFrame(future_rows), "holdout": holdout,
        "horizon_metrics": pd.DataFrame(horizon_scores),
        "metrics": overall, "baseline_metrics": baseline,
        "test_origins": int(test_origins), "train_cutoff": series.index[test_start],
        "config": dict(cfg),
    }


def predict_forecast_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Generate a fresh future path from a compatible history file."""
    series, frequency = prepare_series(frame, bundle["time_column"], bundle["target"],
        bundle["frequency"], bundle["aggregation"], bundle["missing_policy"])
    if max(bundle["lags"] + bundle["windows"]) >= len(series):
        raise ValueError("The uploaded history is shorter than the model's largest lag/window.")
    horizon = int(bundle["horizon"])
    future_dates = pd.date_range(series.index[-1], periods=horizon + 1, freq=frequency)[1:]
    rows = []
    for lead, model in bundle["models_by_lead"].items():
        features = _features_for_lead(series, frequency, bundle["lags"], bundle["windows"], int(lead)).iloc[[-1]]
        rows.append({"target_time": future_dates[int(lead) - 1],
                     "horizon": int(lead), "prediction": float(model.predict(features)[0])})
    return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)
