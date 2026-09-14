"""Pure chart rendering using mplfinance with in-memory PNG output."""

import io
import logging

import matplotlib

# Headless backend must be configured before importing pyplot or mplfinance
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplfinance as mpf  # noqa: E402
import pandas as pd  # noqa: E402

# Suppress verbose font discovery messages from matplotlib
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


def render_chart(df: pd.DataFrame, title: str | None = None) -> bytes:
    """Render an OHLCV candlestick chart with EMA overlays into PNG bytes.

    Args:
        df: DataFrame with DatetimeIndex and capitalized columns:
            'Open', 'High', 'Low', 'Close', 'Volume'.
            May optionally include 'EMA20', 'EMA50', 'EMA200' columns.
        title: Optional plot title displayed at the top.

    Returns:
        Raw PNG image bytes.
    """
    addplots = []
    if "EMA20" in df.columns and df["EMA20"].notna().any():
        addplots.append(
            mpf.make_addplot(
                df["EMA20"],
                color="#1e88e5",  # Blue
                width=1.0,
            )
        )
    if "EMA50" in df.columns and df["EMA50"].notna().any():
        addplots.append(
            mpf.make_addplot(
                df["EMA50"],
                color="#fb8c00",  # Orange
                width=1.2,
            )
        )
    if "EMA200" in df.columns and df["EMA200"].notna().any():
        addplots.append(
            mpf.make_addplot(
                df["EMA200"],
                color="#8e24aa",  # Purple
                width=1.4,
            )
        )

    # Clean, high-contrast trading style
    marketcolors = mpf.make_marketcolors(
        up="#26a69a",
        down="#ef5350",
        edge="inherit",
        wick="inherit",
        volume="in",
    )
    style = mpf.make_mpf_style(
        base_mpf_style="charles",
        marketcolors=marketcolors,
        gridstyle=":",
        gridcolor="#e0e0e0",
    )

    buf = io.BytesIO()
    plot_kwargs: dict[str, object] = {
        "type": "candle",
        "volume": True,
        "style": style,
        "title": title if title else "",
        "figsize": (10, 6),
        "savefig": dict(fname=buf, format="png", dpi=100, bbox_inches="tight"),
    }
    if addplots:
        plot_kwargs["addplot"] = addplots
    try:
        mpf.plot(df, **plot_kwargs)
        buf.seek(0)
        return buf.getvalue()
    finally:
        # Guarantee all figure handles are closed to prevent memory leaks
        plt.close("all")
        buf.close()
