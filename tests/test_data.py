from datetime import date

import pandas as pd
import pytest

from usd_liquidity_monitor import data


def _sample_frame(source: str = "fred") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "value": [1.0, 2.0],
            "source": [source, source],
        }
    )


def test_fetch_series_unknown_name_raises_key_error() -> None:
    with pytest.raises(KeyError):
        data.fetch_series("unknown_series", start=date(2024, 1, 1), end=date(2024, 1, 10))


def test_fetch_series_uses_fallback_when_primary_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_primary(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("primary failed")

    def _fallback(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _sample_frame(source="fred")

    monkeypatch.setattr(data, "_fetch_nyfed_sofr", _raise_primary)
    monkeypatch.setattr(data, "_fetch_fred_series", _fallback)

    out = data.fetch_series("sofr", start=date(2024, 1, 1), end=date(2024, 1, 10))
    assert out.shape[0] == 2
    assert set(out["source"]) == {"fred"}


def test_fetch_all_series_collects_statuses(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_fetch(series_name: str, start: date, end: date) -> pd.DataFrame:
        if series_name == "move_proxy":
            raise RuntimeError("upstream unavailable")
        return _sample_frame(source="fred")

    monkeypatch.setattr(data, "fetch_series", _fake_fetch)

    raw, statuses = data.fetch_all_series(
        start=date(2024, 1, 1),
        end=date(2024, 1, 5),
        series_names=["effr", "move_proxy"],
    )

    assert "effr" in statuses
    assert "move_proxy" in statuses
    assert statuses["effr"].success is True
    assert statuses["move_proxy"].success is False
    assert set(raw["series_name"]) == {"effr"}
