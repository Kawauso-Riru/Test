"""Shared CSV I/O helpers.

netkeiba's jockey_id/trainer_id codes are zero-padded (e.g. "01209"), so they
must be read as strings. Left to pandas' automatic dtype inference, an
all-digit column like this becomes int64 and silently drops the leading
zero (01209 -> 1209) -- which then breaks every merge that joins on these
IDs later, since the freshly-scraped side still has the zero-padded string.
Every place that reads a race-entry CSV (history, shutuba, raw results)
should go through `read_race_csv` rather than a bare `pd.read_csv`.
"""
from __future__ import annotations

import pandas as pd

ID_COLUMN_DTYPES = {"horse_id": str, "jockey_id": str, "trainer_id": str}


def read_race_csv(path, parse_dates=None) -> pd.DataFrame:
    return pd.read_csv(path, dtype=ID_COLUMN_DTYPES, parse_dates=parse_dates)
