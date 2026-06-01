"""Leveraged ETF fragility sub-index.

The sub-index combines public proxies for swap-funding stress and observable
leveraged ETF behavior. It is intentionally independent from the core ULSI
calculation so it can be merged into the broader factor set without changing
the existing ULSI formula.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta
import logging
from importlib import resources
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .data import fetch_series

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_RESOURCE = "lev_etf_fragility.yaml"
BUSINESS_DAY_FREQ = "B"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load leveraged ETF fragility YAML config."""

    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    config_ref = resources.files("usd_liquidity_monitor.configs").joinpath(DEFAULT_CONFIG_RESOURCE)
    with config_ref.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _calendar_index(start: date, end: date) -> pd.DatetimeIndex:
    return pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(end), freq=BUSINESS_DAY_FREQ)


def _limited_ffill(frame: pd.DataFrame | pd.Series, limit: int) -> pd.DataFrame | pd.Series:
    return frame.sort_index().ffill(limit=limit)


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.rolling(window=window, min_periods=window).mean()
    std = values.rolling(window=window, min_periods=window).std(ddof=0)
    zscore = (values - mean) / std
    zero_std = std.eq(0) & values.notna() & mean.notna()
    zscore = zscore.mask(zero_std, 0.0)
    return zscore.clip(-5.0, 5.0)


def fetch_fred_inputs(series_names: list[str], start: date, end: date, *, max_ffill_days: int) -> pd.DataFrame:
    """Fetch FRED/NY Fed series and align them to the FRED business-day calendar."""

    calendar = _calendar_index(start, end)
    out = pd.DataFrame(index=calendar)

    for series_name in series_names:
        try:
            frame = fetch_series(series_name, start=start, end=end)
            if frame.empty:
                LOGGER.warning("FRED series %s returned no rows; factor inputs will be NaN.", series_name)
                out[series_name] = np.nan
                continue

            series = (
                frame.assign(date=lambda x: pd.to_datetime(x["date"], errors="coerce"))
                .dropna(subset=["date"])
                .set_index("date")["value"]
                .sort_index()
            )
            out[series_name] = _limited_ffill(series.reindex(calendar), max_ffill_days)
        except Exception as exc:  # noqa: BLE001 - degraded factor availability is expected.
            LOGGER.warning("Failed to fetch FRED series %s; using NaN. Error: %s", series_name, exc)
            out[series_name] = np.nan

    return out


def _load_yfinance() -> Any:
    try:
        import yfinance as yf  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("yfinance is required for leveraged ETF factors") from exc
    return yf


def fetch_yfinance_prices(tickers: list[str], start: date, end: date, *, max_ffill_days: int) -> pd.DataFrame:
    """Fetch daily adjusted closes and volumes for yfinance tickers."""

    calendar = _calendar_index(start, end)
    columns = pd.MultiIndex.from_product([["close", "volume"], tickers], names=["field", "ticker"])
    empty = pd.DataFrame(index=calendar, columns=columns, dtype=float)
    if not tickers:
        return empty

    try:
        yf = _load_yfinance()
        raw = yf.download(
            tickers=tickers,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=False,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to fetch yfinance price data; ETF factors will be NaN. Error: %s", exc)
        return empty

    if raw.empty:
        LOGGER.warning("yfinance returned no rows for tickers=%s; ETF factors will be NaN.", tickers)
        return empty

    out = empty.copy()
    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw[("Close", ticker)] if ("Close", ticker) in raw.columns else raw[("Adj Close", ticker)]
                volume = raw[("Volume", ticker)]
            else:
                close = raw["Close"] if "Close" in raw.columns else raw["Adj Close"]
                volume = raw["Volume"]
            out[("close", ticker)] = _limited_ffill(pd.to_numeric(close, errors="coerce").reindex(calendar), max_ffill_days)
            out[("volume", ticker)] = _limited_ffill(pd.to_numeric(volume, errors="coerce").reindex(calendar), max_ffill_days)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not normalize yfinance data for %s; using NaN. Error: %s", ticker, exc)

    return out


def fetch_yfinance_shares(tickers: list[str]) -> dict[str, float | None]:
    """Fetch current shares outstanding for AUM proxy construction."""

    shares: dict[str, float | None] = {}
    try:
        yf = _load_yfinance()
    except RuntimeError as exc:
        LOGGER.warning("%s; AUM proxy will fall back to turnover where possible.", exc)
        return {ticker: None for ticker in tickers}

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).fast_info
            value = getattr(info, "shares", None)
            if value is None and isinstance(info, Mapping):
                value = info.get("shares")
            shares[ticker] = float(value) if value is not None and float(value) > 0 else None
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not fetch sharesOutstanding for %s; using turnover proxy. Error: %s", ticker, exc)
            shares[ticker] = None
    return shares


def compute_funding_factors(fred: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Compute bank funding pressure proxies.

    SOFR-EFFR captures secured-funding tightness. SOFR deviation versus its
    20-day mean captures calendar-related funding jumps that can precede ETF
    swap and hedge friction.
    """

    factors = pd.DataFrame(index=fred.index)
    factor_cfg = config["factors"]

    spread_cfg = factor_cfg["funding_sofr_effr_spread"]["params"]
    left = spread_cfg["left"]
    right = spread_cfg["right"]
    factors["funding_sofr_effr_spread"] = fred.get(left, pd.Series(np.nan, index=fred.index)) - fred.get(
        right, pd.Series(np.nan, index=fred.index)
    )

    deviation_cfg = factor_cfg["funding_sofr_20d_deviation"]["params"]
    series_name = deviation_cfg["series"]
    ma_days = int(deviation_cfg["moving_average_days"])
    base = fred.get(series_name, pd.Series(np.nan, index=fred.index))
    factors["funding_sofr_20d_deviation"] = base - base.rolling(ma_days, min_periods=ma_days).mean()
    return factors


def compute_tracking_gap_factor(price_data: pd.DataFrame, config: Mapping[str, Any]) -> pd.Series:
    """Compute nominal-minus-realized leverage gap.

    Leveraged ETF total return swaps require dealer hedging. When realized
    leverage falls below nominal leverage, the gap can point to financing,
    rebalancing, or hedge execution friction.
    """

    pairs = config["factors"]["leveraged_etf_tracking_gap"]["params"]["pairs"]
    min_abs_return = float(config["sub_index"].get("min_abs_benchmark_return", 0.0005))
    pair_gaps: list[pd.Series] = []

    for pair in pairs:
        leveraged = pair["leveraged"]
        benchmark = pair["benchmark"]
        leverage = float(pair["leverage"])
        try:
            lev_return = price_data[("close", leveraged)].pct_change(fill_method=None)
            bench_return = price_data[("close", benchmark)].pct_change(fill_method=None)
            realized_leverage = lev_return / bench_return.where(bench_return.abs() >= min_abs_return)
            pair_gaps.append((leverage - realized_leverage).rename(f"{leveraged}_{benchmark}_tracking_gap"))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Tracking gap pair %s/%s unavailable; using NaN. Error: %s", leveraged, benchmark, exc)

    if not pair_gaps:
        LOGGER.warning("No tracking gap pairs were available; factor will be NaN.")
        return pd.Series(np.nan, index=price_data.index, name="leveraged_etf_tracking_gap")

    return pd.concat(pair_gaps, axis=1).mean(axis=1, skipna=True).rename("leveraged_etf_tracking_gap")


def compute_aum_growth_factor(price_data: pd.DataFrame, config: Mapping[str, Any]) -> pd.Series:
    """Compute leveraged ETF AUM growth proxy.

    yfinance does not provide historical AUM. The preferred proxy is current
    sharesOutstanding times close; if shares are unavailable, the function
    falls back to close times volume as a turnover proxy and logs that choice.
    """

    tickers = config["factors"]["leveraged_etf_aum_growth"]["params"]["tickers"]
    change_days = int(config["sub_index"].get("aum_rolling_change_days", 20))
    shares = fetch_yfinance_shares(tickers)
    proxy_changes: list[pd.Series] = []

    for ticker in tickers:
        close = price_data.get(("close", ticker), pd.Series(np.nan, index=price_data.index))
        volume = price_data.get(("volume", ticker), pd.Series(np.nan, index=price_data.index))

        if shares.get(ticker) is not None:
            LOGGER.info("AUM proxy for %s uses close * current sharesOutstanding.", ticker)
            proxy_level = close * float(shares[ticker])
        else:
            LOGGER.warning("AUM proxy for %s uses close * volume turnover because sharesOutstanding is unavailable.", ticker)
            proxy_level = close * volume

        proxy_changes.append(proxy_level.pct_change(periods=change_days, fill_method=None).rename(f"{ticker}_aum_growth"))

    if not proxy_changes:
        LOGGER.warning("No AUM proxy tickers were configured; factor will be NaN.")
        return pd.Series(np.nan, index=price_data.index, name="leveraged_etf_aum_growth")

    return pd.concat(proxy_changes, axis=1).mean(axis=1, skipna=True).rename("leveraged_etf_aum_growth")


def compute_raw_factors(
    start: date,
    end: date,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Fetch public inputs and compute raw leveraged ETF fragility factors."""

    cfg = dict(config or load_config())
    max_ffill_days = int(cfg["sub_index"].get("max_ffill_days", 5))
    fred_inputs = ["sofr", "effr"]
    fred = fetch_fred_inputs(fred_inputs, start, end, max_ffill_days=max_ffill_days)

    pairs = cfg["factors"]["leveraged_etf_tracking_gap"]["params"]["pairs"]
    aum_tickers = cfg["factors"]["leveraged_etf_aum_growth"]["params"]["tickers"]
    tickers = sorted({item["leveraged"] for item in pairs} | {item["benchmark"] for item in pairs} | set(aum_tickers))
    prices = fetch_yfinance_prices(tickers, start, end, max_ffill_days=max_ffill_days)

    factors = compute_funding_factors(fred, cfg)
    factors["leveraged_etf_tracking_gap"] = compute_tracking_gap_factor(prices, cfg)
    factors["leveraged_etf_aum_growth"] = compute_aum_growth_factor(prices, cfg)
    return factors


def standardize_factors(raw_factors: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Apply rolling z-score and explicit YAML signs to all configured factors."""

    window = int(config["sub_index"].get("zscore_window", 252))
    zscores = pd.DataFrame(index=raw_factors.index)

    for factor_name, factor_cfg in config["factors"].items():
        sign = int(factor_cfg["sign"])
        if sign not in (-1, 1):
            raise ValueError(f"Factor {factor_name} has invalid sign={sign}; expected +1 or -1.")
        if factor_name not in raw_factors.columns:
            LOGGER.warning("Configured factor %s is missing from raw factors; using NaN.", factor_name)
            zscores[factor_name] = np.nan
            continue
        zscores[factor_name] = _rolling_z(raw_factors[factor_name], window=window) * sign

    return zscores


def compose_subindex(zscores: pd.DataFrame, weights: Mapping[str, float] | None = None) -> pd.Series:
    """Compose signed factor z-scores into the baseline sub-index.

    The optional weights argument is intentionally a simple factor-name mapping
    so it can receive precomputed IC weights from the existing ULSI weighting
    module without reimplementing IC estimation here.
    """

    if zscores.empty:
        return pd.Series(dtype=float, name="lev_etf_fragility")

    if weights is None:
        weight_series = pd.Series(1.0, index=zscores.columns, dtype=float)
    else:
        missing = set(zscores.columns) - set(weights.keys())
        if missing:
            raise ValueError(f"weights missing configured factors: {sorted(missing)}")
        weight_series = pd.Series({name: float(weights[name]) for name in zscores.columns}, dtype=float)

    weighted = zscores.multiply(weight_series, axis=1)
    observed_weights = zscores.notna().multiply(weight_series.abs(), axis=1).sum(axis=1)
    subindex = weighted.sum(axis=1, min_count=1) / observed_weights.replace(0.0, np.nan)
    return subindex.rename("lev_etf_fragility")


def compute_lev_etf_fragility(
    start: date,
    end: date,
    *,
    config_path: str | Path | None = None,
    weights: Mapping[str, float] | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Return sub-index, signed factor z-scores, and raw factor values."""

    config = load_config(config_path)
    raw_factors = compute_raw_factors(start, end, config)
    zscores = standardize_factors(raw_factors, config)
    subindex = compose_subindex(zscores, weights=weights)
    return subindex, zscores, raw_factors


def _latest_valid_row(frame: pd.DataFrame) -> pd.Series:
    non_empty = frame.dropna(how="all")
    if non_empty.empty:
        return pd.Series(dtype=float)
    return non_empty.iloc[-1]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    end = date.today()
    start = end - timedelta(days=365 * 2)

    subindex, zscores, raw = compute_lev_etf_fragility(start, end)
    latest_raw = _latest_valid_row(raw)
    latest_z = _latest_valid_row(zscores)
    latest_subindex = subindex.dropna().tail(1)

    print("Latest raw factors:")
    print(latest_raw.to_string() if not latest_raw.empty else "No valid raw factors.")
    print("\nLatest signed z-score factors:")
    print(latest_z.to_string() if not latest_z.empty else "No valid z-score factors.")
    print("\nLatest lev_etf_fragility:")
    print(latest_subindex.to_string() if not latest_subindex.empty else "No valid sub-index value.")

    output_path = Path("lev_etf_fragility.png").resolve()
    fig, ax = plt.subplots(figsize=(11, 5))
    subindex.dropna().plot(ax=ax, color="black", linewidth=1.6)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title("Leveraged ETF Fragility Sub-Index")
    ax.set_ylabel("signed factor z-score composite")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved chart to: {output_path}")


if __name__ == "__main__":
    main()
