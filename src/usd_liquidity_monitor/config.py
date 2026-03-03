"""Project-level constants and defaults."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class SeriesSpec:
    """Configuration for an input time series."""

    source: str
    series_id: str
    fallback_source: str | None = None
    fallback_series_id: str | None = None
    required: bool = True


DEFAULT_START_DATE = date(2010, 1, 1)

# Canonical names are used throughout feature engineering and dashboard code.
SERIES_SPECS: dict[str, SeriesSpec] = {
    "sofr": SeriesSpec(source="nyfed", series_id="SOFR", fallback_source="fred", fallback_series_id="SOFR"),
    "effr": SeriesSpec(source="fred", series_id="DFF"),
    "iorb": SeriesSpec(source="fred", series_id="IORB"),
    "on_rrp": SeriesSpec(source="fred", series_id="RRPONTSYD"),
    "reserves": SeriesSpec(source="fred", series_id="WRESBAL"),
    "fed_assets": SeriesSpec(source="fred", series_id="WALCL"),
    "tga": SeriesSpec(source="fred", series_id="WTREGEN"),
    "t_bill_3m": SeriesSpec(source="fred", series_id="TB3MS"),
    "yield_3m": SeriesSpec(source="fred", series_id="DGS3MO"),
    "yield_2y": SeriesSpec(source="fred", series_id="DGS2"),
    "yield_10y": SeriesSpec(source="fred", series_id="DGS10"),
    "cp_3m": SeriesSpec(source="fred", series_id="CPF3M"),
    "ig_oas": SeriesSpec(source="fred", series_id="BAMLC0A0CM"),
    "hy_oas": SeriesSpec(source="fred", series_id="BAMLH0A0HYM2"),
    "dxy": SeriesSpec(source="fred", series_id="DTWEXBGS"),
    "vix": SeriesSpec(source="fred", series_id="VIXCLS"),
    "nasdaq_composite": SeriesSpec(source="fred", series_id="NASDAQCOM", required=False),
    "nasdaq100": SeriesSpec(source="fred", series_id="NASDAQ100", required=False),
    # Public proxy for rates-vol stress in place of MOVE for MVP.
    "move_proxy": SeriesSpec(source="fred", series_id="NFCI", required=False),
}

COMPONENT_WEIGHTS: dict[str, float] = {
    "funding": 0.35,
    "fiscal": 0.25,
    "reserves": 0.20,
    "credit": 0.20,
}

# Used by legacy fixed alert rule (`ULSI > 1.5 for 3 days`) while regime labels are now
# computed from rolling quantiles in `metrics.compute_ulsi`.
REGIME_THRESHOLDS: tuple[float, float, float] = (0.5, 1.5, 2.5)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
NYFED_SOFR_URL = "https://markets.newyorkfed.org/api/rates/secured/sofr/search.json"
