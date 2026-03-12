from __future__ import annotations

import argparse
import json

from qck_bot.backtest import BacktestConfig, run_backtest, save_backtest_outputs
from qck_bot.data import DownloadSpec, download_klines, load_klines
from qck_bot.features import FEATURE_COLUMNS, build_feature_frame
from qck_bot.data import DownloadSpec, download_klines
from qck_bot.features import FEATURE_COLUMNS
from qck_bot.live import LiveConfig, run_live_loop, score_latest_bar, sync_and_score
from qck_bot.model import (
    WalkForwardConfig,
    load_model_artifact,
    make_training_frame,
    save_model_artifact,
    train_final_model,
    walk_forward_predictions,
)


def download_command(args: argparse.Namespace) -> None:
    output_path = download_klines(
        DownloadSpec(
            market=args.market,
            symbol=args.symbol,
            interval=args.interval,
            start_month=args.start_month,
            end_month=args.end_month,
        )
    )
    print(f"saved: {output_path}")


def backtest_command(args: argparse.Namespace) -> None:
    frame = load_klines(args.input)
    features = build_feature_frame(frame, horizon=args.horizon)
    model_frame = make_training_frame(
        features,
        WalkForwardConfig(
            horizon=args.horizon,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
        ),
    )
    predictions = walk_forward_predictions(
        model_frame,
        WalkForwardConfig(
            horizon=args.horizon,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
        ),
    )
    trades, metrics = run_backtest(
        predictions,
        BacktestConfig(
            horizon=args.horizon,
            entry_probability=args.entry_probability,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
        ),
    )
    save_backtest_outputs(predictions, trades, metrics)
    print(json.dumps(metrics, indent=2))


def train_command(args: argparse.Namespace) -> None:
    frame = load_klines(args.input)
    features = build_feature_frame(frame, horizon=args.horizon)
    config = WalkForwardConfig(
        horizon=args.horizon,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )
    model_frame = make_training_frame(features, config)
    if args.exclude_tail_bars > 0:
        train_frame = model_frame.iloc[:-args.exclude_tail_bars].copy()
    else:
        train_frame = model_frame

    model = train_final_model(train_frame)
    metadata = {
        "symbol": args.symbol,
        "market": args.market,
        "interval": args.interval,
        "horizon": args.horizon,
        "entry_probability": args.entry_probability,
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "feature_columns": FEATURE_COLUMNS,
        "train_rows": int(len(train_frame)),
        "train_start": str(train_frame["open_time"].iloc[0]),
        "train_end": str(train_frame["open_time"].iloc[-1]),
    }
    output_path = save_model_artifact(model, metadata, args.output)
    print(json.dumps({"saved_model": str(output_path), **metadata}, indent=2))


def predict_command(args: argparse.Namespace) -> None:
    print(json.dumps(score_latest_bar(model_path=args.model, input_path=args.input), indent=2))


def sync_command(args: argparse.Namespace) -> None:
    result = sync_and_score(
        model_path=args.model,
        input_path=args.input,
        symbol=args.symbol,
        interval=args.interval,
        refresh_limit=args.refresh_limit,
    )
    print(json.dumps(result, indent=2))


def live_command(args: argparse.Namespace) -> None:
    run_live_loop(
        LiveConfig(
            model_path=args.model,
            input_path=args.input,
            symbol=args.symbol,
            interval=args.interval,
            refresh_limit=args.refresh_limit,
            poll_seconds=args.poll_seconds,
            iterations=args.iterations,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qck-bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download free BTC kline history")
    download.add_argument("--market", default="spot", choices=["spot", "futures"])
    download.add_argument("--symbol", default="BTCUSDT")
    download.add_argument("--interval", default="1m")
    download.add_argument("--start-month", required=True)
    download.add_argument("--end-month", required=True)
    download.set_defaults(func=download_command)

    backtest = subparsers.add_parser("backtest", help="train and backtest on local data")
    backtest.add_argument("--input", required=True)
    backtest.add_argument("--horizon", type=int, default=5)
    backtest.add_argument("--entry-probability", type=float, default=0.58)
    backtest.add_argument("--train-bars", type=int, default=180_000)
    backtest.add_argument("--test-bars", type=int, default=30_000)
    backtest.add_argument("--fee-bps", type=float, default=4.0)
    backtest.add_argument("--slippage-bps", type=float, default=2.0)
    backtest.set_defaults(func=backtest_command)

    train = subparsers.add_parser("train", help="train final model and save artifact")
    train.add_argument("--input", required=True)
    train.add_argument("--symbol", default="BTCUSDT")
    train.add_argument("--market", default="spot")
    train.add_argument("--interval", default="1m")
    train.add_argument("--horizon", type=int, default=5)
    train.add_argument("--train-bars", type=int, default=180_000)
    train.add_argument("--test-bars", type=int, default=30_000)
    train.add_argument("--entry-probability", type=float, default=0.58)
    train.add_argument("--fee-bps", type=float, default=4.0)
    train.add_argument("--slippage-bps", type=float, default=2.0)
    train.add_argument("--exclude-tail-bars", type=int, default=0)
    train.add_argument("--output", default="models/btc_model.pkl")
    train.set_defaults(func=train_command)

    predict = subparsers.add_parser("predict", help="score the latest bar using a saved model")
    predict.add_argument("--model", required=True)
    predict.add_argument("--input", required=True)
    predict.set_defaults(func=predict_command)

    sync = subparsers.add_parser("sync", help="refresh local klines from Binance REST and score latest bar")
    sync.add_argument("--model", required=True)
    sync.add_argument("--input", required=True)
    sync.add_argument("--symbol", default="BTCUSDT")
    sync.add_argument("--interval", default="1m")
    sync.add_argument("--refresh-limit", type=int, default=1000)
    sync.set_defaults(func=sync_command)

    live = subparsers.add_parser("live", help="run a polling live signal loop")
    live.add_argument("--model", required=True)
    live.add_argument("--input", required=True)
    live.add_argument("--symbol", default="BTCUSDT")
    live.add_argument("--interval", default="1m")
    live.add_argument("--refresh-limit", type=int, default=1000)
    live.add_argument("--poll-seconds", type=int, default=60)
    live.add_argument("--iterations", type=int, default=-1)
    live.set_defaults(func=live_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
