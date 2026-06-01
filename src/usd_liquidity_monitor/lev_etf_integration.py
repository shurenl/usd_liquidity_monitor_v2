"""Diagnostic integration analysis for the leveraged ETF fragility sub-index.

This module deliberately keeps analysis separate from the production ULSI path.
It answers whether `lev_etf_fragility` is redundant, predictive, and stable
before any decision to include it in the main index.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from importlib import resources
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .data import fetch_all_series
from .lev_etf_fragility import compute_lev_etf_fragility
from .metrics import compute_features, compute_ulsi

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_RESOURCE = "lev_etf_integration.yaml"
BUSINESS_DAY_FREQ = "B"
ULSI_FACTOR_COLUMNS = ["z_F", "z_G", "z_R", "z_C"]


@dataclass(frozen=True)
class OLSResult:
    r2: float
    residual: pd.Series
    coefficients: pd.Series


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the integration diagnostic YAML config."""

    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    config_ref = resources.files("usd_liquidity_monitor.configs").joinpath(DEFAULT_CONFIG_RESOURCE)
    with config_ref.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _business_index(start: date, end: date) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=BUSINESS_DAY_FREQ)


def _align_series(series: pd.Series, index: pd.DatetimeIndex, max_ffill_days: int) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").sort_index().reindex(index)
    if max_ffill_days <= 0:
        return out
    return out.ffill(limit=max_ffill_days)


def prepare_base_inputs(start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Fetch public inputs and return ULSI factor z-scores, ULSI, and raw data."""

    raw_df, _ = fetch_all_series(start=start, end=end)
    features_df = compute_features(raw_df)
    ulsi_df = compute_ulsi(features_df)

    ulsi_work = ulsi_df.copy()
    ulsi_work["date"] = pd.to_datetime(ulsi_work["date"], errors="coerce")
    ulsi_work = ulsi_work.dropna(subset=["date"]).set_index("date").sort_index()

    factor_cols = [col for col in ULSI_FACTOR_COLUMNS if col in ulsi_work.columns]
    ulsi_factors = ulsi_work[factor_cols].apply(pd.to_numeric, errors="coerce")
    ulsi = pd.to_numeric(ulsi_work["ulsi"], errors="coerce").rename("ulsi_base")
    return raw_df, ulsi_factors, ulsi


def build_forward_target(
    raw_df: pd.DataFrame,
    *,
    series_name: str,
    horizon: int,
    start: date,
    end: date,
    max_ffill_days: int,
) -> pd.Series:
    """Build a strictly forward target.

    At date t, the target is `series[t + horizon] - series[t]`. The negative
    shift moves the future realization back to t, so factor values at t only
    predict information after t.
    """

    if raw_df.empty:
        return pd.Series(dtype=float, name=f"forward_{series_name}_change_{horizon}d")

    calendar = _business_index(start, end)
    working = raw_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    wide = (
        working.dropna(subset=["date"])
        .pivot_table(index="date", columns="series_name", values="value", aggfunc="last")
        .sort_index()
    )
    if series_name not in wide.columns:
        LOGGER.warning("Target series %s is unavailable; target will be NaN.", series_name)
        return pd.Series(np.nan, index=calendar, name=f"forward_{series_name}_change_{horizon}d")

    target_base = _align_series(wide[series_name], calendar, max_ffill_days=max_ffill_days)
    target = target_base.shift(-horizon) - target_base
    return target.rename(f"forward_{series_name}_change_{horizon}d")


def spearman_ic(signal: pd.Series, target: pd.Series, *, min_observations: int = 40) -> float:
    """Compute Spearman rank IC with a minimum overlap guard."""

    pairs = pd.concat([signal.rename("signal"), target.rename("target")], axis=1).dropna()
    if pairs.shape[0] < min_observations:
        return float("nan")
    if pairs["signal"].nunique() < 2 or pairs["target"].nunique() < 2:
        return float("nan")
    ranked = pairs.rank(method="average")
    return float(ranked["signal"].corr(ranked["target"]))


def rolling_spearman_ic(
    signal: pd.Series,
    target: pd.Series,
    *,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Rolling Spearman IC for stability diagnostics."""

    pairs = pd.concat([signal.rename("signal"), target.rename("target")], axis=1).sort_index()
    values: list[float] = []
    idx = pairs.index
    for end_pos in range(len(pairs)):
        start_pos = max(0, end_pos - window + 1)
        sample = pairs.iloc[start_pos : end_pos + 1].dropna()
        if sample.shape[0] < min_periods or sample["signal"].nunique() < 2 or sample["target"].nunique() < 2:
            values.append(float("nan"))
        else:
            ranked = sample.rank(method="average")
            values.append(float(ranked["signal"].corr(ranked["target"])))
    return pd.Series(values, index=idx, name=f"rolling_ic_{signal.name or 'signal'}")


def fit_ols_residual(y: pd.Series, x: pd.DataFrame) -> OLSResult:
    """Regress y on x with an intercept and return R² plus residual series."""

    y_named = y.rename("y")
    pairs = pd.concat([y_named, x], axis=1).dropna()
    residual = pd.Series(np.nan, index=y.index, name="lev_etf_fragility_resid")
    if pairs.shape[0] <= x.shape[1] + 1:
        return OLSResult(r2=float("nan"), residual=residual, coefficients=pd.Series(dtype=float))

    y_arr = pairs["y"].to_numpy(dtype=float)
    x_arr = pairs.drop(columns=["y"]).to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x_arr)), x_arr])
    beta, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    fitted = design @ beta
    resid = y_arr - fitted
    ss_total = float(np.sum((y_arr - y_arr.mean()) ** 2))
    ss_resid = float(np.sum(resid**2))
    r2 = float(1.0 - ss_resid / ss_total) if ss_total > 0 else float("nan")
    residual.loc[pairs.index] = resid
    coefficients = pd.Series(beta, index=["intercept", *pairs.drop(columns=["y"]).columns], dtype=float)
    return OLSResult(r2=r2, residual=residual, coefficients=coefficients)


def full_sample_correlations(ulsi_factors: pd.DataFrame, lev: pd.Series) -> pd.DataFrame:
    combined = pd.concat([ulsi_factors, lev.rename("lev_etf_fragility")], axis=1)
    return combined.rank(method="average").corr()


def rolling_correlations(ulsi_factors: pd.DataFrame, lev: pd.Series, *, window: int) -> pd.DataFrame:
    out = pd.DataFrame(index=ulsi_factors.index)
    for col in ulsi_factors.columns:
        out[col] = ulsi_factors[col].rolling(window=window, min_periods=max(20, window // 3)).corr(lev)
    return out


def _plot_heatmap(matrix: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    data = matrix.to_numpy(dtype=float)
    image = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            if pd.notna(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_lines(frame: pd.DataFrame, title: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in frame.columns:
        series = pd.to_numeric(frame[col], errors="coerce").dropna()
        if not series.empty:
            ax.plot(series.index, series, label=col, linewidth=1.4)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_series(series: pd.Series, title: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if not clean.empty:
        ax.plot(clean.index, clean, color="black", linewidth=1.5)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compute_ic_table(
    signals: Mapping[str, pd.Series],
    targets: Mapping[int, pd.Series],
    *,
    min_observations: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon, target in targets.items():
        for name, signal in signals.items():
            ic = spearman_ic(signal, target, min_observations=min_observations)
            rows.append(
                {
                    "horizon": horizon,
                    "signal": name,
                    "ic": ic,
                    "sign": "positive" if pd.notna(ic) and ic > 0 else "negative" if pd.notna(ic) and ic < 0 else "NA",
                }
            )
    return pd.DataFrame(rows)


def make_subsample_periods(index: pd.DatetimeIndex, periods_config: list[Mapping[str, str]]) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    clean_index = pd.DatetimeIndex(index).dropna().sort_values().unique()
    if len(clean_index) == 0:
        return []
    if periods_config:
        return [
            (item["name"], pd.Timestamp(item["start"]), pd.Timestamp(item["end"]))
            for item in periods_config
        ]

    splits = np.array_split(clean_index, 3)
    periods: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for idx, split in enumerate(splits, start=1):
        if len(split) > 0:
            periods.append((f"sample_{idx}", pd.Timestamp(split[0]), pd.Timestamp(split[-1])))
    return periods


def compute_subsample_ic_table(
    signals: Mapping[str, pd.Series],
    target: pd.Series,
    periods: list[tuple[str, pd.Timestamp, pd.Timestamp]],
    *,
    min_observations: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period_name, start, end in periods:
        mask = (target.index >= start) & (target.index <= end)
        target_slice = target.loc[mask]
        for signal_name, signal in signals.items():
            signal_slice = signal.reindex(target_slice.index)
            rows.append(
                {
                    "period": period_name,
                    "start": start.date().isoformat(),
                    "end": end.date().isoformat(),
                    "signal": signal_name,
                    "ic": spearman_ic(signal_slice, target_slice, min_observations=min_observations),
                }
            )
    return pd.DataFrame(rows)


def compute_trimmed_ic(
    signal: pd.Series,
    target: pd.Series,
    *,
    trim_pct: float,
    min_observations: int,
) -> float:
    pairs = pd.concat([signal.rename("signal"), target.rename("target")], axis=1).dropna()
    if pairs.empty:
        return float("nan")
    cutoff = pairs["signal"].abs().quantile(1.0 - trim_pct)
    trimmed = pairs[pairs["signal"].abs() <= cutoff]
    return spearman_ic(trimmed["signal"], trimmed["target"], min_observations=min_observations)


def compose_candidate_ulsi_with_lev(
    ulsi_base: pd.Series,
    lev: pd.Series,
    *,
    lev_sign: int,
    lev_weight: float,
    external_weights: Mapping[str, float] | None = None,
) -> pd.Series:
    """Create an analysis-only A/B candidate.

    If external IC weights are supplied, expected keys are `ulsi_base` and
    `lev_etf_fragility`. Otherwise, use the configured diagnostic blend.
    """

    if external_weights is not None:
        base_weight = float(external_weights["ulsi_base"])
        lev_weight = float(external_weights["lev_etf_fragility"])
    else:
        base_weight = 1.0 - lev_weight

    candidate = base_weight * ulsi_base + lev_weight * lev_sign * lev
    return candidate.rename("ulsi_with_lev")


def _direction_from_ic(ic_value: float) -> tuple[int, str]:
    if pd.isna(ic_value):
        return 1, "IC is unavailable; keep configured sign +1 for diagnostics only."
    if ic_value < 0:
        return -1, "IC is negative: treat lev_etf_fragility as a contrarian signal and flip sign for candidate blending."
    return 1, "IC is positive: keep lev_etf_fragility sign +1 for candidate blending."


def _format_table(frame: pd.DataFrame, float_cols: list[str] | None = None) -> str:
    if frame.empty:
        return "_No data._"
    out = frame.copy()
    for col in float_cols or []:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "NA" if pd.isna(x) else f"{float(x):.4f}")
    out = out.fillna("NA").astype(str)
    columns = list(out.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for _, row in out.iterrows()]
    return "\n".join([header, separator, *rows])


def _write_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostic(
    *,
    start: date,
    end: date,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    external_ab_weights: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Run the full integration diagnostic and write markdown + PNG artifacts."""

    config = load_config(config_path)
    max_ffill_days = int(config["analysis"].get("max_ffill_days", 5))
    out_dir = Path(output_dir or config["analysis"]["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df, ulsi_factors, ulsi_base = prepare_base_inputs(start, end)
    lev, _, _ = compute_lev_etf_fragility(start, end)
    lev = lev.reindex(ulsi_base.index).rename("lev_etf_fragility")

    corr_matrix = full_sample_correlations(ulsi_factors, lev)
    rolling_corr = rolling_correlations(
        ulsi_factors,
        lev,
        window=int(config["redundancy"]["rolling_corr_window"]),
    )
    ols = fit_ols_residual(lev, ulsi_factors)

    target_cfg = config["target"]
    horizons = [int(item) for item in target_cfg["horizons"]]
    targets = {
        horizon: build_forward_target(
            raw_df,
            series_name=target_cfg["series_name"],
            horizon=horizon,
            start=start,
            end=end,
            max_ffill_days=max_ffill_days,
        ).reindex(ulsi_base.index)
        for horizon in horizons
    }
    primary_horizon = int(target_cfg["primary_horizon"])
    primary_target = targets[primary_horizon]

    signals = {
        "lev_etf_fragility": lev,
        "lev_etf_fragility_resid": ols.residual,
        "ulsi_base": ulsi_base,
    }
    ic_table = compute_ic_table(signals, targets, min_observations=int(config["ic"]["min_observations"]))
    primary_resid_ic = ic_table[
        (ic_table["horizon"] == primary_horizon) & (ic_table["signal"] == "lev_etf_fragility_resid")
    ]["ic"].iloc[0]
    primary_lev_ic = ic_table[
        (ic_table["horizon"] == primary_horizon) & (ic_table["signal"] == "lev_etf_fragility")
    ]["ic"].iloc[0]

    lev_sign, sign_comment = _direction_from_ic(primary_resid_ic if pd.notna(primary_resid_ic) else primary_lev_ic)
    ulsi_with_lev = compose_candidate_ulsi_with_lev(
        ulsi_base,
        lev,
        lev_sign=lev_sign,
        lev_weight=float(config["ab_test"]["lev_weight"]),
        external_weights=external_ab_weights,
    )
    ab_table = compute_ic_table(
        {"ulsi_base": ulsi_base, "ulsi_with_lev": ulsi_with_lev},
        {primary_horizon: primary_target},
        min_observations=int(config["ic"]["min_observations"]),
    )
    ab_corr = float(pd.concat([ulsi_base, ulsi_with_lev], axis=1).dropna().rank(method="average").corr().iloc[0, 1])

    rolling_ic_frame = pd.DataFrame(
        {
            "lev_etf_fragility": rolling_spearman_ic(
                lev,
                primary_target,
                window=int(config["ic"]["rolling_window"]),
                min_periods=int(config["ic"]["rolling_min_periods"]),
            ),
            "lev_etf_fragility_resid": rolling_spearman_ic(
                ols.residual,
                primary_target,
                window=int(config["ic"]["rolling_window"]),
                min_periods=int(config["ic"]["rolling_min_periods"]),
            ),
        }
    )
    rolling_sign_share = float((np.sign(rolling_ic_frame["lev_etf_fragility_resid"].dropna()) == np.sign(primary_resid_ic)).mean()) if pd.notna(primary_resid_ic) and not rolling_ic_frame["lev_etf_fragility_resid"].dropna().empty else float("nan")

    periods = make_subsample_periods(primary_target.index, config["subsamples"].get("periods", []))
    subsample_table = compute_subsample_ic_table(
        {"lev_etf_fragility": lev, "lev_etf_fragility_resid": ols.residual},
        primary_target,
        periods,
        min_observations=int(config["ic"]["min_observations"]),
    )
    trimmed_rows = []
    for signal_name, signal in {"lev_etf_fragility": lev, "lev_etf_fragility_resid": ols.residual}.items():
        full_ic = spearman_ic(signal, primary_target, min_observations=int(config["ic"]["min_observations"]))
        trimmed_ic = compute_trimmed_ic(
            signal,
            primary_target,
            trim_pct=float(config["ic"]["trim_extreme_pct"]),
            min_observations=int(config["ic"]["min_observations"]),
        )
        trimmed_rows.append({"signal": signal_name, "full_ic": full_ic, "trimmed_ic": trimmed_ic, "delta": trimmed_ic - full_ic})
    trimmed_table = pd.DataFrame(trimmed_rows)

    heatmap_path = out_dir / "redundancy_correlation_heatmap.png"
    rolling_corr_path = out_dir / "rolling_correlation.png"
    rolling_ic_path = out_dir / "rolling_ic.png"
    residual_path = out_dir / "lev_etf_fragility_residual.png"
    ab_path = out_dir / "ulsi_ab_comparison.png"

    _plot_heatmap(corr_matrix, "Spearman Correlation: ULSI Factors vs Leveraged ETF Fragility", heatmap_path)
    _plot_lines(rolling_corr, "Rolling Correlation: Leveraged ETF Fragility vs Existing Factors", "correlation", rolling_corr_path)
    _plot_lines(rolling_ic_frame, f"Rolling {primary_horizon}D Forward Target IC", "Spearman IC", rolling_ic_path)
    _plot_series(ols.residual, "Leveraged ETF Fragility Residual", "residual z-score", residual_path)
    _plot_lines(pd.concat([ulsi_base, ulsi_with_lev], axis=1), "ULSI Base vs Candidate With Leveraged ETF Fragility", "index value", ab_path)

    base_ic = ab_table.loc[ab_table["signal"] == "ulsi_base", "ic"].iloc[0]
    with_ic = ab_table.loc[ab_table["signal"] == "ulsi_with_lev", "ic"].iloc[0]
    ic_improvement = with_ic - base_ic if pd.notna(with_ic) and pd.notna(base_ic) else float("nan")
    high_r2 = pd.notna(ols.r2) and ols.r2 > float(config["redundancy"]["high_r2_threshold"])
    meaningful_resid = pd.notna(primary_resid_ic) and abs(primary_resid_ic) >= float(config["ic"]["meaningful_abs_ic"])
    stable = pd.notna(rolling_sign_share) and rolling_sign_share >= float(config["ic"]["stable_sign_share"])
    improved = pd.notna(ic_improvement) and ic_improvement >= float(config["ic"]["improvement_threshold"])
    go = (not high_r2) and meaningful_resid and stable and improved

    recommendation = (
        "GO: add lev_etf_fragility only after wiring external IC weights into production."
        if go
        else "NO-GO for production inclusion now: evidence is not strong enough after redundancy, stability, and A/B checks."
    )

    report_lines = [
        "# Leveraged ETF Fragility Integration Diagnostic",
        "",
        "## Configurable Parameters",
        f"- Target variable: `{target_cfg['series_name']}` transformed as `{target_cfg['transform']}`.",
        f"- Horizons: `{horizons}`; primary horizon: `{primary_horizon}` business days.",
        f"- Rolling correlation window: `{config['redundancy']['rolling_corr_window']}` business days.",
        f"- Rolling IC window: `{config['ic']['rolling_window']}` business days.",
        f"- Subsamples: `manual` if configured, otherwise three equal date blocks.",
        f"- Extreme trimming: largest `{float(config['ic']['trim_extreme_pct']) * 100:.1f}%` by absolute signal value.",
        "",
        "## 1. Redundancy",
        f"- OLS R² of `lev_etf_fragility ~ existing ULSI factors`: `{ols.r2:.4f}`.",
        f"- High redundancy threshold: `{float(config['redundancy']['high_r2_threshold']):.2f}`.",
        f"- Interpretation: {'high redundancy' if high_r2 else 'not highly redundant'} by this threshold.",
        "",
        "### Full-Sample Spearman Correlation Matrix",
        _format_table(corr_matrix.reset_index(names="factor"), [col for col in corr_matrix.columns]),
        "",
        f"![Correlation heatmap]({heatmap_path.name})",
        f"![Rolling correlation]({rolling_corr_path.name})",
        "",
        "## 2. Forward IC / Incremental Predictive Value",
        "Forward target construction avoids look-ahead bias: `target_t = VIX_{t+h} - VIX_t`, implemented as `series.shift(-h) - series`.",
        "",
        _format_table(ic_table, ["ic"]),
        "",
        f"- Sign conclusion: {sign_comment}",
        "",
        "## 3. Stability",
        f"- Rolling residual IC sign-consistency share: `{rolling_sign_share:.4f}`.",
        "",
        "### Subsample IC",
        _format_table(subsample_table, ["ic"]),
        "",
        "### Extreme-Value Sensitivity",
        _format_table(trimmed_table, ["full_ic", "trimmed_ic", "delta"]),
        "",
        f"![Rolling IC]({rolling_ic_path.name})",
        f"![Residual series]({residual_path.name})",
        "",
        "## 4. A/B Candidate Comparison",
        _format_table(ab_table, ["ic"]),
        "",
        f"- Spearman correlation between `ulsi_base` and `ulsi_with_lev`: `{ab_corr:.4f}`.",
        f"- IC improvement: `{ic_improvement:.4f}`.",
        f"- Candidate lev sign: `{lev_sign:+d}`.",
        f"- Candidate blend note: this is analysis-only. It uses external weights if provided, otherwise the diagnostic `lev_weight={float(config['ab_test']['lev_weight']):.2f}`.",
        "",
        f"![A/B comparison]({ab_path.name})",
        "",
        "## Conclusion",
        f"- Redundancy: {'high' if high_r2 else 'acceptable / orthogonal enough'} based on OLS R².",
        f"- Incremental value: {'present' if meaningful_resid else 'weak or unavailable'} based on residual IC.",
        f"- Stability: {'stable enough' if stable else 'not stable enough'} based on rolling IC sign consistency.",
        f"- Recommendation: **{recommendation}**",
    ]

    report_path = out_dir / "lev_etf_integration_diagnostic.md"
    _write_report(report_path, report_lines)
    print("\n".join(report_lines))
    print(f"\nArtifacts saved to: {out_dir}")

    return {
        "report_path": report_path,
        "output_dir": out_dir,
        "corr_matrix": corr_matrix,
        "rolling_corr": rolling_corr,
        "ols": ols,
        "ic_table": ic_table,
        "rolling_ic": rolling_ic_frame,
        "subsample_ic": subsample_table,
        "trimmed_ic": trimmed_table,
        "ab_table": ab_table,
        "recommendation": recommendation,
    }


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether leveraged ETF fragility should enter ULSI")
    parser.add_argument("--start", type=_parse_date, default=None, help="Start date in YYYY-MM-DD")
    parser.add_argument("--end", type=_parse_date, default=date.today(), help="End date in YYYY-MM-DD")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for markdown and PNG files")
    args = parser.parse_args()

    config = load_config(args.config)
    start = args.start or (args.end - timedelta(days=365 * int(config["analysis"].get("start_years_back", 5))))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_diagnostic(start=start, end=args.end, config_path=args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
