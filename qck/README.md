# qck-bot

Local BTC short-term trading research bot built around free Binance historical data.

This does not promise profits and it will not produce `99%` accuracy. What it does give you is:

- a free data pipeline for long-run BTCUSDT minute candles
- a feature generator for short-term price and volume behavior
- walk-forward model training so evaluation stays out of sample
- a simple long-only backtest with fees and slippage

## Why this shape

Short-term BTC prediction is noisy. The useful question is not "was the next candle green", it is:

`does the predicted move exceed trading costs by enough to justify a trade?`

That is the target used here.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Download free BTC data

The downloader pulls monthly Binance Vision klines and stores one local Parquet file.

```bash
qck-bot download \
  --market spot \
  --symbol BTCUSDT \
  --interval 1m \
  --start-month 2020-01 \
  --end-month 2026-02
```

Output:

`data/processed/spot_BTCUSDT_1m.parquet`

Use complete UTC months. For example, on March 9, 2026, `2026-02` is safe but `2026-03` may still be incomplete.

## Run a backtest

```bash
qck-bot backtest \
  --input data/processed/spot_BTCUSDT_1m.parquet \
  --horizon 5 \
  --entry-probability 0.58 \
  --train-bars 180000 \
  --test-bars 30000 \
  --fee-bps 4 \
  --slippage-bps 2
```

Outputs:

- `reports/backtest_metrics.json`
- `reports/trades.csv`
- `reports/predictions.parquet`

## Train the final model artifact

```bash
qck-bot train \
  --input data/processed/spot_BTCUSDT_1m.parquet \
  --symbol BTCUSDT \
  --market spot \
  --interval 1m \
  --horizon 5 \
  --train-bars 180000 \
  --test-bars 30000 \
  --entry-probability 0.58 \
  --fee-bps 4 \
  --slippage-bps 2 \
  --output models/btc_model.pkl
```

## Score the latest bar

```bash
qck-bot predict \
  --model models/btc_model.pkl \
  --input data/processed/spot_BTCUSDT_1m.parquet
```

## Refresh to current market data and score

```bash
qck-bot sync \
  --model models/btc_model.pkl \
  --input data/processed/spot_BTCUSDT_1m.parquet \
  --symbol BTCUSDT \
  --interval 1m \
  --refresh-limit 1000
```

This pulls the latest candles from Binance REST, merges them into the local Parquet file, and scores the newest complete bar.

## Run the live signal loop

```bash
qck-bot live \
  --model models/btc_model.pkl \
  --input data/processed/spot_BTCUSDT_1m.parquet \
  --symbol BTCUSDT \
  --interval 1m \
  --refresh-limit 1000 \
  --poll-seconds 60
```

Each iteration:

- refreshes local BTCUSDT 1 minute candles
- scores the latest bar with the trained model
- appends JSON lines to `reports/live_signals.jsonl`

## Notes

- The model is a baseline `HistGradientBoostingClassifier`.
- Signals are walk-forward only: each chunk is trained on older data and tested on newer data.
- The backtest is intentionally conservative compared to typical social media examples.
- If this produces no edge after costs, that is useful information. Most strategies die there.
- Binance monthly kline archives use millisecond timestamps in older files and microseconds in newer files; the downloader handles both.
