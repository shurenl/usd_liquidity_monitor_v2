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

- `ghcr.io/<your-github-username>/usd-liquidity-monitor:latest`

Run the published image:

```bash
docker pull ghcr.io/<your-github-username>/usd-liquidity-monitor:latest
docker run --rm -p 8501:8501 ghcr.io/<your-github-username>/usd-liquidity-monitor:latest
```

Then open:

- `http://localhost:8501`

## GitHub Daily 9AM Email Report

This repository also includes:

- `.github/workflows/daily-report.yml`

Schedule:

- Runs daily at **09:00 Asia/Shanghai** (`01:00 UTC`) via GitHub Actions schedule.

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
