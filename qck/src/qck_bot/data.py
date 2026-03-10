from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from qck_bot.config import KLINE_COLUMNS, PROCESSED_DIR, RAW_DIR

BASE_URL = "https://data.binance.vision"
REST_BASE_URL = "https://api.binance.com"


@dataclass(frozen=True)
class DownloadSpec:
    market: str
    symbol: str
    interval: str
    start_month: str
    end_month: str


def month_range(start_month: str, end_month: str) -> list[str]:
    start = datetime.strptime(start_month, "%Y-%m")
    end = datetime.strptime(end_month, "%Y-%m")
    if start > end:
        raise ValueError("start_month must be before or equal to end_month")

    months: list[str] = []
    current = start
    while current <= end:
        months.append(current.strftime("%Y-%m"))
        year = current.year + (current.month // 12)
        month = 1 if current.month == 12 else current.month + 1
        current = current.replace(year=year, month=month)
    return months


def market_prefix(market: str) -> str:
    normalized = market.lower()
    if normalized == "spot":
        return "data/spot/monthly/klines"
    if normalized in {"futures", "um"}:
        return "data/futures/um/monthly/klines"
    raise ValueError("market must be 'spot' or 'futures'")


def build_monthly_url(market: str, symbol: str, interval: str, month: str) -> str:
    prefix = market_prefix(market)
    file_name = f"{symbol}-{interval}-{month}.zip"
    return f"{BASE_URL}/{prefix}/{symbol}/{interval}/{file_name}"


def _read_kline_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        csv_name = archive.namelist()[0]
        with archive.open(csv_name) as handle:
            frame = pd.read_csv(handle, header=None, names=KLINE_COLUMNS)

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Binance monthly klines switched from millisecond to microsecond timestamps in 2025.
    # Detect the epoch scale from the data so older and newer archives parse consistently.
    open_unit = "us" if frame["open_time"].dropna().max() >= 10**15 else "ms"
    close_unit = "us" if frame["close_time"].dropna().max() >= 10**15 else "ms"
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit=open_unit, utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit=close_unit, utc=True)
    return frame.drop(columns=["ignore"])


def _download_month(url: str, destination: Path) -> bytes:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return response.content


def download_klines(spec: DownloadSpec) -> Path:
    symbol = spec.symbol.upper()
    interval = spec.interval
    output_path = PROCESSED_DIR / f"{spec.market.lower()}_{symbol}_{interval}.parquet"

    monthly_frames: list[pd.DataFrame] = []
    for month in month_range(spec.start_month, spec.end_month):
        url = build_monthly_url(spec.market, symbol, interval, month)
        raw_path = RAW_DIR / spec.market.lower() / symbol / interval / f"{month}.zip"
        try:
            content = _download_month(url, raw_path)
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            raise RuntimeError(f"failed to download {url} (status {status_code})") from exc
        monthly_frames.append(_read_kline_zip(content))

    frame = pd.concat(monthly_frames, ignore_index=True)
    frame = frame.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return output_path


def load_klines(path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True)
    return frame.sort_values("open_time").reset_index(drop=True)


def fetch_recent_klines(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    response = requests.get(
        f"{REST_BASE_URL}/api/v3/klines",
        params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    frame = pd.DataFrame(payload, columns=KLINE_COLUMNS)

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    return frame.drop(columns=["ignore"]).sort_values("open_time").reset_index(drop=True)


def merge_klines(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    frame = pd.concat([existing, incoming], ignore_index=True)
    frame = frame.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")
    return frame.reset_index(drop=True)


def update_local_klines(path: str | Path, symbol: str, interval: str, limit: int = 1000) -> tuple[Path, pd.DataFrame]:
    output_path = Path(path)
    existing = load_klines(output_path)
    incoming = fetch_recent_klines(symbol=symbol, interval=interval, limit=limit)
    merged = merge_klines(existing, incoming)
    merged.to_parquet(output_path, index=False)
    return output_path, merged
