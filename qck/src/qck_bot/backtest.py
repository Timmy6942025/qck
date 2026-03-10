from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from qck_bot.config import REPORTS_DIR


@dataclass(frozen=True)
class BacktestConfig:
    horizon: int
    entry_probability: float
    fee_bps: float
    slippage_bps: float


def _round_trip_cost(fee_bps: float, slippage_bps: float) -> float:
    return ((fee_bps + slippage_bps) * 2) / 10_000


def run_backtest(predictions: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, dict]:
    trades: list[dict] = []
    costs = _round_trip_cost(config.fee_bps, config.slippage_bps)

    last_exit = -1
    for index, row in predictions.iterrows():
        if index <= last_exit:
            continue
        if row["prob_up"] < config.entry_probability:
            continue

        exit_index = index + config.horizon
        if exit_index >= len(predictions):
            break

        exit_row = predictions.iloc[exit_index]
        gross_return = (exit_row["close"] / row["close"]) - 1
        net_return = gross_return - costs
        trades.append(
            {
                "entry_time": row["open_time"],
                "exit_time": exit_row["open_time"],
                "entry_price": row["close"],
                "exit_price": exit_row["close"],
                "prob_up": row["prob_up"],
                "gross_return": gross_return,
                "net_return": net_return,
            }
        )
        last_exit = exit_index

    trade_frame = pd.DataFrame(trades)
    if trade_frame.empty:
        metrics = {
            "trade_count": 0,
            "win_rate": 0.0,
            "average_net_return": 0.0,
            "compound_return": 0.0,
            "max_drawdown": 0.0,
        }
        return trade_frame, metrics

    trade_frame["equity_curve"] = (1 + trade_frame["net_return"]).cumprod()
    rolling_peak = trade_frame["equity_curve"].cummax()
    drawdown = trade_frame["equity_curve"] / rolling_peak - 1

    metrics = {
        "trade_count": int(len(trade_frame)),
        "win_rate": float((trade_frame["net_return"] > 0).mean()),
        "average_net_return": float(trade_frame["net_return"].mean()),
        "compound_return": float(trade_frame["equity_curve"].iloc[-1] - 1),
        "max_drawdown": float(drawdown.min()),
    }
    return trade_frame, metrics


def save_backtest_outputs(
    predictions: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: dict,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(REPORTS_DIR / "predictions.parquet", index=False)
    trades.to_csv(REPORTS_DIR / "trades.csv", index=False)
    (REPORTS_DIR / "backtest_metrics.json").write_text(json.dumps(metrics, indent=2))
