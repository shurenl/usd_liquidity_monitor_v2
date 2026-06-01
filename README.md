# USD Liquidity Monitor (MVP)

A Python app to monitor USD liquidity stress on a daily basis using public data.

## Scope

- Daily batch update (ET-oriented macro use case)
- Public data first (FRED + NY Fed SOFR API)
- Feature engineering + ULSI composite index
- Streamlit web dashboard + CSV export

## Data Sources

- NY Fed: SOFR (`/api/rates/secured/sofr/search.json`)
- FRED: EFFR (DFF), IORB, ON RRP (RRPONTSYD), reserves (WRESBAL), Fed assets (WALCL),
  TGA (WTREGEN), yields (DGS3MO/DGS2/DGS10), CP proxy (CPF3M), TBill (TB3MS),
  OAS (BAMLC0A0CM/BAMLH0A0HYM2), DXY (DTWEXBGS), VIX (VIXCLS), NFCI

## Core Metrics

- `spread_policy = EFFR - IORB`
- `spread_repo = SOFR - IORB`
- `pressure_reserve = -delta(reserves, 20D)`
- `pressure_tga = delta(TGA, 20D)`
- `pressure_on_rrp = -delta(ON_RRP, 20D)`
- `pressure_fed_assets = -delta(FedAssets, 20D)`
- `pressure_cp = CP - 3M TBill`
- `pressure_credit = HY_OAS / IG_OAS`
- `pressure_curve_inversion = -(10Y - 3M)`
- `pressure_frontend_jump = delta(2Y, 20D)`
- `pressure_market = z(VIX) + z(MOVE_proxy) + z(DXY)`
- `ULSI = sum(weight_i * z_component_i)`

## Leveraged ETF Fragility Sub-Index

The project also includes an optional sub-index named `lev_etf_fragility`.
It is designed to be merged into a broader factor set as a signed z-score
series, without changing the core four-factor ULSI formula.

Configuration:

- `src/usd_liquidity_monitor/configs/lev_etf_fragility.yaml`

Factor groups:

- Bank funding pressure:
  `SOFR - EFFR` and `SOFR - rolling_mean(SOFR, 20B)`.
  These proxy secured funding tightness and calendar-related swap financing stress.
- Leveraged ETF behavior traces:
  tracking gap for configurable `(leveraged ETF, benchmark ETF, leverage)` pairs,
  plus AUM growth proxies for configured leveraged ETF tickers.
- Close-flow concentration:
  reserved hook for future intraday data; not active in the baseline version.

Every factor in YAML has an explicit `sign`:

- `1`: larger raw value means higher fragility.
- `-1`: larger raw value means lower fragility.

AUM approximation is explicit in logs:

- Preferred proxy: `close * sharesOutstanding`.
- Fallback proxy: `close * volume` turnover if shares are unavailable.

Run the minimal example:

```bash
python -m usd_liquidity_monitor.lev_etf_fragility
```

It fetches the latest two years, prints latest raw factors and signed z-scores,
and saves:

```bash
lev_etf_fragility.png
```

### Leveraged ETF Integration Diagnostic

Before adding the sub-index to production ULSI, run the integration diagnostic:

```bash
ulsi-lev-etf-diagnose --start 2021-01-01 --end 2026-06-01 --output-dir /tmp/lev_etf_integration
```

or:

```bash
python -m usd_liquidity_monitor.lev_etf_integration --output-dir /tmp/lev_etf_integration
```

Default configuration:

- `src/usd_liquidity_monitor/configs/lev_etf_integration.yaml`

What it checks:

- Redundancy: full-sample/rolling correlation and OLS R² versus existing ULSI factor z-scores.
- Incremental value: forward Spearman IC versus future VIX changes, including residual IC after removing existing factor exposure.
- Stability: rolling IC, date subsamples, and top-1% absolute signal trimming.
- A/B comparison: `ulsi_base` versus an analysis-only candidate `ulsi_with_lev`.

The forward target avoids look-ahead bias:

```text
target_t = VIX_{t+h} - VIX_t
```

Artifacts:

- `lev_etf_integration_diagnostic.md`
- `redundancy_correlation_heatmap.png`
- `rolling_correlation.png`
- `rolling_ic.png`
- `lev_etf_fragility_residual.png`
- `ulsi_ab_comparison.png`

## Regime Thresholds

- `<0.5`: Normal
- `0.5-1.5`: Watch
- `1.5-2.5`: Tight
- `>2.5`: Stress

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Run

```bash
ulsi-dashboard
```

or

```bash
streamlit run /Users/linshuren/Documents/project1/usd_liquidity_monitor/src/usd_liquidity_monitor/app.py
```

## Sync and Export

```bash
ulsi-sync --start 2015-01-01 --end 2026-02-10 --output /tmp/ulsi_output.csv
```

## Notes

- MOVE is not directly available from FRED in this MVP; NFCI is used as a public proxy.
- SOFR is fetched from NY Fed first, with FRED fallback.

## GitHub Automated Deployment

This repository includes a GitHub Actions workflow at:

- `.github/workflows/ci-cd.yml`

What it does:

1. On `pull_request` to `main`: installs dependencies and runs `pytest`.
2. On `push` to `main`: runs tests, then builds and pushes a Docker image to GHCR.

Published image:

- `ghcr.io/<your-github-username>/usd-liquidity-monitor-v2:latest`

Run the published image:

```bash
docker pull ghcr.io/<your-github-username>/usd-liquidity-monitor-v2:latest
docker run --rm -p 8501:8501 ghcr.io/<your-github-username>/usd-liquidity-monitor-v2:latest
```

Then open:

- `http://localhost:8501`

## GitHub Daily 9:30AM Email Report

This repository also includes:

- `.github/workflows/daily-report.yml`

Schedule:

- Runs daily at **09:30 Asia/Shanghai** (`01:30 UTC`) via GitHub Actions schedule.

Workflow behavior:

1. Installs project dependencies.
2. Generates daily ULSI + Nasdaq impact summary.
3. Generates a PDF with visual charts.
4. Sends report by SMTP email with the PDF attachment.

Required GitHub repository secrets:

- `SMTP_HOST` (e.g. `smtp.gmail.com`)
- `SMTP_PORT` (usually `587`)
- `SMTP_USER` (your SMTP login)
- `SMTP_PASSWORD` (SMTP/app password)
- `REPORT_TO` (recipient email)
- `REPORT_FROM` (optional sender email; defaults to `SMTP_USER`)

Optional GitHub repository variable:

- `REPORT_TIMEZONE` (defaults to `Asia/Shanghai`, used in email subject date)

Manual test run:

1. Open GitHub `Actions` tab.
2. Select `Daily ULSI Email Report`.
3. Click `Run workflow`.

Local dry-run (no email sent):

```bash
ulsi-daily-report --dry-run
```

Local dry-run and save PDF:

```bash
ulsi-daily-report --dry-run --save-pdf /tmp/ulsi_daily_report.pdf
```
