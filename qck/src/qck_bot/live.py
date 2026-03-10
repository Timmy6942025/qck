from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from qck_bot.config import REPORTS_DIR
from qck_bot.data import load_klines, update_local_klines
from qck_bot.features import FEATURE_COLUMNS, build_feature_frame
from qck_bot.model import load_model_artifact


@dataclass(frozen=True)
class LiveConfig:
    model_path: str
    input_path: str
    symbol: str
    interval: str
    refresh_limit: int
    poll_seconds: int
    iterations: int


def score_latest_bar(model_path: str, input_path: str) -> dict:
    model, metadata = load_model_artifact(model_path)
    frame = load_klines(input_path)
    features = build_feature_frame(frame, horizon=metadata["horizon"]).dropna(subset=FEATURE_COLUMNS)
    latest = features.iloc[-1]
    probability = float(model.predict_proba(features[FEATURE_COLUMNS].tail(1))[:, 1][0])
    signal = "buy" if probability >= metadata["entry_probability"] else "hold"
    return {
        "open_time": str(latest["open_time"]),
        "close": float(latest["close"]),
        "prob_up": probability,
        "signal": signal,
        "entry_probability": metadata["entry_probability"],
        "horizon": metadata["horizon"],
    }


def sync_and_score(model_path: str, input_path: str, symbol: str, interval: str, refresh_limit: int) -> dict:
    updated_path, merged = update_local_klines(
        path=input_path,
        symbol=symbol,
        interval=interval,
        limit=refresh_limit,
    )
    result = score_latest_bar(model_path=model_path, input_path=str(updated_path))
    result["rows"] = int(len(merged))
    result["data_path"] = str(updated_path)
    return result


def _append_signal_log(record: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORTS_DIR / "live_signals.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return log_path


def run_live_loop(config: LiveConfig) -> None:
    remaining = config.iterations
    while remaining != 0:
        result = sync_and_score(
            model_path=config.model_path,
            input_path=config.input_path,
            symbol=config.symbol,
            interval=config.interval,
            refresh_limit=config.refresh_limit,
        )
        log_path = _append_signal_log(result)
        print(json.dumps({**result, "log_path": str(log_path)}, indent=2), flush=True)

        if remaining > 0:
            remaining -= 1
            if remaining == 0:
                break
        time.sleep(config.poll_seconds)
