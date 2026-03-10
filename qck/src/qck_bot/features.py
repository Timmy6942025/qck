from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_15",
    "range_1",
    "close_vs_vwap",
    "volume_z_20",
    "trade_count_z_20",
    "volatility_15",
    "volatility_60",
    "rsi_14",
    "trend_10_30",
]


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window).mean()
    avg_loss = losses.rolling(window).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_feature_frame(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    data = frame.copy()

    data["ret_1"] = data["close"].pct_change(1)
    data["ret_3"] = data["close"].pct_change(3)
    data["ret_5"] = data["close"].pct_change(5)
    data["ret_15"] = data["close"].pct_change(15)
    data["range_1"] = (data["high"] - data["low"]) / data["close"].replace(0, np.nan)
    data["vwap"] = data["quote_volume"] / data["volume"].replace(0, np.nan)
    data["close_vs_vwap"] = (data["close"] / data["vwap"]) - 1
    data["volume_z_20"] = _zscore(data["volume"], 20)
    data["trade_count_z_20"] = _zscore(data["trade_count"], 20)
    data["volatility_15"] = data["ret_1"].rolling(15).std()
    data["volatility_60"] = data["ret_1"].rolling(60).std()
    data["rsi_14"] = _rsi(data["close"], 14) / 100.0
    data["trend_10_30"] = (
        data["close"].rolling(10).mean() / data["close"].rolling(30).mean()
    ) - 1

    data["future_return"] = data["close"].shift(-horizon) / data["close"] - 1
    return data
