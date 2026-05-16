import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_WINDOW_SIZE = 100
DEFAULT_MIN_WINDOW_SIZE = 30
DEFAULT_LOW_QUANTILE = 0.01
DEFAULT_HIGH_QUANTILE = 0.99
DEFAULT_SAMPLE_ALERT_THRESHOLD = 0.15
DEFAULT_WINDOW_ALERT_THRESHOLD = 0.15


def _to_float(value):
    if pd.isna(value):
        return None
    return float(value)


def _finite_or_default(value, default):
    if value is None or not np.isfinite(value):
        return default
    return float(value)


def _normalise_numeric_frame(df):
    return df.apply(pd.to_numeric, errors="coerce").astype(np.float32)


def build_drift_reference(
    reference_df,
    output_path,
    *,
    source="validation_normal",
    window_size=DEFAULT_WINDOW_SIZE,
    min_window_size=DEFAULT_MIN_WINDOW_SIZE,
    low_quantile=DEFAULT_LOW_QUANTILE,
    high_quantile=DEFAULT_HIGH_QUANTILE,
):
    """Build a simple unsupervised normal-profile reference for streaming drift checks."""
    if reference_df.empty:
        raise ValueError("No hay filas de referencia para construir el perfil de deriva")

    numeric_df = _normalise_numeric_frame(reference_df)
    features = numeric_df.columns.tolist()
    if not features:
        raise ValueError("No hay columnas numericas para construir el perfil de deriva")

    stats = {}
    lower_bounds = []
    upper_bounds = []

    for feature in features:
        series = numeric_df[feature].dropna()
        if series.empty:
            mean = median = q_low = q_high = 0.0
            std = iqr = 1.0
        else:
            mean = _finite_or_default(series.mean(), 0.0)
            median = _finite_or_default(series.median(), mean)
            std = _finite_or_default(series.std(ddof=0), 0.0)
            q_low = _finite_or_default(series.quantile(low_quantile), mean)
            q_high = _finite_or_default(series.quantile(high_quantile), mean)
            q25 = _finite_or_default(series.quantile(0.25), median)
            q75 = _finite_or_default(series.quantile(0.75), median)
            iqr = max(q75 - q25, 1e-6)
            std = max(std, 1e-6)

        if q_high < q_low:
            q_low, q_high = q_high, q_low

        stats[feature] = {
            "mean": mean,
            "median": median,
            "std": std,
            "iqr": iqr,
            "q_low": q_low,
            "q_high": q_high,
        }
        lower_bounds.append(q_low)
        upper_bounds.append(q_high)

    values = numeric_df[features].to_numpy(dtype=np.float32)
    lower = np.array(lower_bounds, dtype=np.float32)
    upper = np.array(upper_bounds, dtype=np.float32)
    out_of_range = (values < lower) | (values > upper)
    sample_scores = out_of_range.mean(axis=1)

    if len(sample_scores) >= window_size:
        window_scores = [
            float(sample_scores[start : start + window_size].mean())
            for start in range(0, len(sample_scores) - window_size + 1, window_size)
        ]
    else:
        window_scores = [float(sample_scores.mean())]

    sample_threshold = max(
        float(np.quantile(sample_scores, 0.99)),
        DEFAULT_SAMPLE_ALERT_THRESHOLD,
    )
    window_threshold = max(
        float(np.quantile(window_scores, 0.99)),
        DEFAULT_WINDOW_ALERT_THRESHOLD,
    )

    reference = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "normal_profile_quantile_window",
        "source": source,
        "reference_rows": int(len(numeric_df)),
        "features_count": int(len(features)),
        "features": features,
        "low_quantile": low_quantile,
        "high_quantile": high_quantile,
        "window_size": int(window_size),
        "min_window_size": int(min_window_size),
        "sample_alert_threshold": sample_threshold,
        "window_alert_threshold": window_threshold,
        "baseline": {
            "sample_score_mean": float(sample_scores.mean()),
            "sample_score_p95": float(np.quantile(sample_scores, 0.95)),
            "sample_score_p99": float(np.quantile(sample_scores, 0.99)),
            "window_score_p99": float(np.quantile(window_scores, 0.99)),
        },
        "feature_stats": stats,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(reference, f, indent=2)

    return reference


class DriftMonitor:
    def __init__(self, reference_path):
        self.reference_path = Path(reference_path)
        self.enabled = self.reference_path.is_file()
        self.total_observations = 0
        self.scores = deque()
        self.feature_flags = deque()
        self.last_sample_score = None
        self.last_alert_active = False

        if not self.enabled:
            self.reference = {}
            self.features = []
            self.lower = np.array([], dtype=np.float32)
            self.upper = np.array([], dtype=np.float32)
            self.window_size = DEFAULT_WINDOW_SIZE
            self.min_window_size = DEFAULT_MIN_WINDOW_SIZE
            self.sample_threshold = DEFAULT_SAMPLE_ALERT_THRESHOLD
            self.window_threshold = DEFAULT_WINDOW_ALERT_THRESHOLD
            return

        with self.reference_path.open("r", encoding="utf-8") as f:
            self.reference = json.load(f)

        self.features = self.reference["features"]
        stats = self.reference["feature_stats"]
        self.lower = np.array([stats[feature]["q_low"] for feature in self.features], dtype=np.float32)
        self.upper = np.array([stats[feature]["q_high"] for feature in self.features], dtype=np.float32)
        self.window_size = int(self.reference.get("window_size", DEFAULT_WINDOW_SIZE))
        self.min_window_size = int(self.reference.get("min_window_size", DEFAULT_MIN_WINDOW_SIZE))
        self.sample_threshold = float(
            self.reference.get("sample_alert_threshold", DEFAULT_SAMPLE_ALERT_THRESHOLD)
        )
        self.window_threshold = float(
            self.reference.get("window_alert_threshold", DEFAULT_WINDOW_ALERT_THRESHOLD)
        )

    def observe(self, df):
        if not self.enabled:
            return self.status()

        numeric_df = _normalise_numeric_frame(df.reindex(columns=self.features, fill_value=0))
        values = numeric_df.to_numpy(dtype=np.float32)
        flags = (values < self.lower) | (values > self.upper)
        sample_scores = flags.mean(axis=1)

        for score, flag_row in zip(sample_scores, flags):
            self.scores.append(float(score))
            self.feature_flags.append(flag_row.astype(bool))
            if len(self.scores) > self.window_size:
                self.scores.popleft()
                self.feature_flags.popleft()
            self.total_observations += 1
            self.last_sample_score = float(score)

        return self.status()

    def reset(self):
        self.total_observations = 0
        self.scores.clear()
        self.feature_flags.clear()
        self.last_sample_score = None
        self.last_alert_active = False
        return self.status()

    def status(self):
        if not self.enabled:
            return {
                "enabled": False,
                "reference_path": str(self.reference_path),
                "detail": "No existe referencia de deriva",
            }

        window_observations = len(self.scores)
        rolling_score = float(np.mean(self.scores)) if self.scores else 0.0
        last_score = float(self.last_sample_score) if self.last_sample_score is not None else 0.0
        sample_alert = last_score >= self.sample_threshold
        window_alert = (
            window_observations >= self.min_window_size
            and rolling_score >= self.window_threshold
        )
        alert_active = bool(sample_alert or window_alert)

        if self.feature_flags:
            flag_matrix = np.vstack(self.feature_flags)
            ratios = flag_matrix.mean(axis=0)
            top_indices = np.argsort(ratios)[::-1][:5]
            top_features = [
                {
                    "feature": self.features[index],
                    "out_of_range_ratio": float(ratios[index]),
                }
                for index in top_indices
                if ratios[index] > 0
            ]
        else:
            top_features = []

        reason = None
        if sample_alert:
            reason = "ultima_muestra_fuera_de_perfil"
        if window_alert:
            reason = "ventana_deslizante_fuera_de_perfil"

        return {
            "enabled": True,
            "reference_path": str(self.reference_path),
            "method": self.reference.get("method"),
            "source": self.reference.get("source"),
            "reference_rows": self.reference.get("reference_rows"),
            "features_count": self.reference.get("features_count"),
            "window_size": self.window_size,
            "min_window_size": self.min_window_size,
            "total_observations": self.total_observations,
            "window_observations": window_observations,
            "last_sample_score": last_score,
            "rolling_drift_score": rolling_score,
            "sample_alert_threshold": self.sample_threshold,
            "window_alert_threshold": self.window_threshold,
            "alert_active": alert_active,
            "alert_reason": reason,
            "top_features": top_features,
        }
