from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from qck_bot.config import MODEL_DIR
from qck_bot.features import FEATURE_COLUMNS


@dataclass(frozen=True)
class WalkForwardConfig:
    horizon: int
    train_bars: int
    test_bars: int
    fee_bps: float
    slippage_bps: float


def trade_cost_threshold(fee_bps: float, slippage_bps: float) -> float:
    total_bps = (fee_bps + slippage_bps) * 2
    return total_bps / 10_000


def make_training_frame(frame: pd.DataFrame, config: WalkForwardConfig) -> pd.DataFrame:
    data = frame.copy()
    threshold = trade_cost_threshold(config.fee_bps, config.slippage_bps)
    data["target"] = (data["future_return"] > threshold).astype(int)
    data = data.dropna(subset=FEATURE_COLUMNS + ["future_return", "target"]).reset_index(drop=True)
    return data


def walk_forward_predictions(frame: pd.DataFrame, config: WalkForwardConfig) -> pd.DataFrame:
    if len(frame) < config.train_bars + config.test_bars:
        raise ValueError("not enough data for the requested train/test windows")

    chunks: list[pd.DataFrame] = []
    start = config.train_bars
    while start < len(frame):
        train_start = start - config.train_bars
        train_end = start
        test_end = min(start + config.test_bars, len(frame))

        train = frame.iloc[train_start:train_end]
        test = frame.iloc[start:test_end]
        if test.empty:
            break

        model = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=250,
            min_samples_leaf=200,
            random_state=7,
        )
        model.fit(train[FEATURE_COLUMNS], train["target"])

        scored = test[["open_time", "close", "future_return", "target"]].copy()
        scored["prob_up"] = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        chunks.append(scored)
        start = test_end

    if not chunks:
        raise ValueError("walk-forward evaluation produced no test chunks")

    return pd.concat(chunks, ignore_index=True)


def fit_model(train: pd.DataFrame) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=250,
        min_samples_leaf=200,
        random_state=7,
    )
    model.fit(train[FEATURE_COLUMNS], train["target"])
    return model


def train_final_model(frame: pd.DataFrame) -> HistGradientBoostingClassifier:
    return fit_model(frame)


def save_model_artifact(
    model: HistGradientBoostingClassifier,
    metadata: dict,
    path: str | Path | None = None,
) -> Path:
    output_path = Path(path) if path is not None else MODEL_DIR / "btc_model.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump({"model": model, "metadata": metadata}, handle)
    return output_path


def load_model_artifact(path: str | Path) -> tuple[HistGradientBoostingClassifier, dict]:
    with Path(path).open("rb") as handle:
        artifact = pickle.load(handle)
    return artifact["model"], artifact["metadata"]
