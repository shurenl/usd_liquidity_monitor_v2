FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN python -m pip install --upgrade pip && \
    python -m pip install .

EXPOSE 8501

CMD ["streamlit", "run", "/app/src/usd_liquidity_monitor/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
