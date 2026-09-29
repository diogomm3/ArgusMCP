"""Internal date-normalization utilities shared across indicators modules.

This module is intentionally internal (underscore prefix). All public-facing
indicator functions re-export their own stable API; these helpers are
implementation details that may change without a deprecation cycle.

No network calls, no database access, no side effects.
"""

from __future__ import annotations

import pandas as pd


def extract_norm_dates(df: pd.DataFrame) -> "pd.Series[pd.Timestamp]":
    """Return a Series of normalized (midnight UTC) Timestamps aligned to df's index.

    Accepts DataFrames with either:
    - A ``'date'`` column containing date-like values (strings, date objects,
      Timestamps), or
    - A ``pd.DatetimeIndex`` as the index.

    In both cases the returned Series has the same index as ``df`` and each
    value is a timezone-naive midnight Timestamp (i.e. ``ts.normalize()``).

    Args:
        df: DataFrame whose date information to extract.

    Returns:
        pd.Series of pd.Timestamp, index aligned to df.index.

    Raises:
        ValueError: If ``df`` has no ``'date'`` column and its index is not a
            ``DatetimeIndex``, or if the ``'date'`` column cannot be parsed.
    """
    if "date" in df.columns:
        try:
            return pd.to_datetime(df["date"]).dt.normalize()
        except Exception as exc:
            raise ValueError("Failed to parse 'date' column") from exc
    elif isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index.normalize(), index=df.index)
    else:
        raise ValueError(
            "DataFrame must contain a 'date' column or have a DatetimeIndex"
        )
