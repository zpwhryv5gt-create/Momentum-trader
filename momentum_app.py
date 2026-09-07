import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "Momentum Trader — V3 Intraday"
DEFAULT_TICKERS = [
    "VWRP.L",
    "SWDA.L",
    "CSP1.L",
    "EQQQ.L",
    "IITU.L",
    "EMIM.L",
    "IGLN.L",
]
EVAL_EVERY_BARS = 3  # 15 minutes when using five-minute bars


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(
    "Experimental intraday momentum dashboard. Paper/research use only — "
    "no broker connection and no live orders."
)


def download_intraday(tickers, period="60d", interval="5m"):
    import yfinance as yf

    if not tickers:
        raise ValueError("Enter at least one ticker.")

    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
        prepost=False,
        timeout=20,
    )
    if raw.empty:
        raise RuntimeError("No intraday market data returned.")

    if isinstance(raw.columns, pd.MultiIndex):
        fields = raw.columns.get_level_values(0)
        field = "Adj Close" if "Adj Close" in fields else "Close"
        prices = raw[field].copy()
    else:
        field = "Adj Close" if "Adj Close" in raw.columns else "Close"
        prices = raw[[field]].copy()
        prices.columns = [tickers[0]]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.sort_index().dropna(how="all")
    prices = prices.ffill(limit=3).dropna(axis=1, how="all")
    if prices.empty:
        raise RuntimeError("The data feed returned no usable prices.")
    return prices


def signal_components(prices):
    # Five-minute bars: 15m = 3 bars, 60m = 12 bars, one session ~= 78 bars.
    r15 = prices / prices.shift(3) - 1
    r60 = prices / prices.shift(12) - 1
    r1d = prices / prices.shift(78) - 1
    score = 0.50 * r15 + 0.30 * r60 + 0.20 * r1d

    ema20 = prices.ewm(span=20, adjust=False).mean()
    ema60 = prices.ewm(span=60, adjust=False).mean()
    trend_ok = (prices > ema20) & (ema20 > ema60)
    eligible = trend_ok & (r15 > 0) & (r60 > 0) & (score > 0)
    return score, ema20, ema60, eligible, {"15m": r15, "60m": r60, "1d": r1d}


def build_target_weights(prices, top_n=1):
    score, _, _, eligible, _ = signal_components(prices)
    eval_dates = prices.index[::EVAL_EVERY_BARS]
    signals = pd.DataFrame(0.0, index=eval_dates, columns=prices.columns)

    for timestamp in eval_dates:
        valid = (
            score.loc[timestamp]
            .where(eligible.loc[timestamp])
            .dropna()
            .sort_values(ascending=False)
        )
        winners = valid.head(top_n).index.tolist()
        if winners:
            signals.loc[timestamp, winners] = 1.0 / len(winners)

    weights = signals.reindex(prices.index).ffill().fillna(0.0)
    return weights.shift(1).fillna(0.0)


def signal_snapshot(prices, top_n=1):
    score, ema20, ema60, eligible, components = signal_components(prices)
    latest = prices.index[-1]
    data = {"Price": prices.loc[latest]}
    for label, frame in components.items():
        data[label] = frame.loc[latest]
    data["Score"] = score.loc[latest]
    data["EMA20"] = ema20.loc[latest]
    data["EMA60"] = ema60.loc[latest]

    table = pd.DataFrame(data)
    table["Trend OK"] = (table["Price"] > table["EMA20"]) & (
        table["EMA20"] > table["EMA60"]
    )
    table["Eligible"] = eligible.loc[latest]
    ranked = table[table["Eligible"]].sort_values("Score", ascending=False)
    winners = ranked.head(top_n).index.tolist()
    table["Target weight"] = 0.0
    if winners:
        table.loc[winners, "Target weight"] = 1.0 / len(winners)
    return latest, table.sort_values("Score", ascending=False)


def run_backtest(prices, initial=1000.0, top_n=1, cost_bps=8.0):
    weights = build_target_weights(prices, top_n=top_n)
    returns = prices.pct_change().fillna(0.0)
    gross = (weights * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    costs = turnover * (cost_bps / 10000.0)
    net = gross - costs
    equity = initial * (1.0 + net).cumprod()

    daily = pd.DataFrame(
        {"net": net, "turnover": turnover, "action": (turnover > 0).astype(int)}
    )
    try:
        daily.index = daily.index.tz_convert("Europe/London")
    except (TypeError, AttributeError):
        pass
    summary = daily.groupby(daily.index.date).agg(
        return_frac=("net", lambda values: (1 + values).prod() - 1),
        turnover=("turnover", "sum"),
        actions=("action", "sum"),
    )
    return turnover, costs, equity, summary


@st.cache_data(ttl=300, show_spinner=False)
def cached_prices(tickers_tuple):
    return download_intraday(list(tickers_tuple))


with st.sidebar:
    st.header("V3 settings")
    initial = st.number_input(
        "Paper capital (£)", min_value=100.0, value=1000.0, step=100.0
    )
    top_n = st.selectbox("Maximum holdings", [1, 2], index=0)
    cost_bps = st.number_input(
        "Estimated one-way spread + slippage (bps)",
        min_value=0.0,
        value=8.0,
        step=1.0,
        help=(
            "Applied on every unit of portfolio turnover. This is deliberately "
            "non-zero even where broker commission is zero."
        ),
    )
    ticker_text = st.text_input(
        "GBP-listed ETF universe", value=", ".join(DEFAULT_TICKERS)
    )
    tickers = [item.strip().upper() for item in ticker_text.split(",") if item.strip()]


st.subheader("V3 hypothesis")
st.write(
    "Every **15 minutes**, rank the ETF universe using **15-minute, 60-minute "
    "and one-session momentum**. A holding is allowed only when price > EMA20 "
    "> EMA60 and both the 15-minute and 60-minute returns are positive. "
    "Otherwise the model sits in cash."
)
st.write(
    "The default is **one ETF at a time**. This is intentionally a high-turnover "
    "experiment, not a claim that intraday momentum will outperform."
)

col1, col2, col3 = st.columns(3)
backtest_btn = col1.button("Run 60-day Intraday Test", use_container_width=True)
signal_btn = col2.button("Current V3 Signal", use_container_width=True)
paper_btn = col3.button("Record Paper Decision", use_container_width=True)

prices = None
if backtest_btn or signal_btn or paper_btn:
    try:
        with st.spinner("Loading five-minute market data…"):
            prices = cached_prices(tuple(tickers))
    except Exception as error:
        st.error(f"Market data could not be loaded: {error}")
else:
    st.info("Choose an action above. Market data loads only when needed.")


if backtest_btn and prices is not None:
    turnover, costs, equity, daily_summary = run_backtest(
        prices, initial, top_n, cost_bps
    )
    st.subheader("Intraday test results")
    a, b, c, d = st.columns(4)
    a.metric("End value", f"£{equity.iloc[-1]:,.2f}")
    b.metric("Net return", f"{equity.iloc[-1] / initial - 1:.2%}")
    drawdown = (equity / equity.cummax() - 1).min()
    c.metric("Max drawdown", f"{drawdown:.2%}")
    d.metric("Portfolio actions", f"{int((turnover > 0).sum()):,}")
    st.line_chart(equity.rename("V3 paper equity"))
    st.caption(
        f"Cumulative modelled trading-cost drag: {costs.sum():.2%}; "
        f"total turnover: {turnover.sum():.1f}x."
    )
    st.subheader("Daily evaluation")
    display = daily_summary.copy()
    display.index = pd.Index([str(value) for value in display.index], name="Date")
    display["return_frac"] = display["return_frac"].map(lambda value: f"{value:.2%}")
    display["turnover"] = display["turnover"].map(lambda value: f"{value:.2f}x")
    st.dataframe(
        display.rename(
            columns={
                "return_frac": "Net return",
                "turnover": "Turnover",
                "actions": "Actions",
            }
        ),
        use_container_width=True,
    )


if signal_btn and prices is not None:
    timestamp, table = signal_snapshot(prices, top_n=top_n)
    selected = table[table["Target weight"] > 0]["Target weight"]
    st.subheader(f"Current signal — {timestamp}")
    if selected.empty:
        st.success("V3 signal: CASH 100%")
    else:
        st.success(
            "V3 signal: "
            + ", ".join(f"{ticker} {weight:.0%}" for ticker, weight in selected.items())
        )
    display = table.copy()
    for column in ["15m", "60m", "1d", "Score", "Target weight"]:
        display[column] = display[column].map(
            lambda value: f"{value:.2%}" if pd.notna(value) else ""
        )
    st.dataframe(display, use_container_width=True)


if "paper_log_v3" not in st.session_state:
    st.session_state.paper_log_v3 = []

if paper_btn and prices is not None:
    timestamp, table = signal_snapshot(prices, top_n=top_n)
    selected = table[table["Target weight"] > 0]["Target weight"]
    allocation = (
        "CASH 100%"
        if selected.empty
        else ", ".join(
            f"{ticker} {weight:.0%}" for ticker, weight in selected.items()
        )
    )
    st.session_state.paper_log_v3.append(
        {"time": str(timestamp), "capital": float(initial), "allocation": allocation}
    )
    st.success(f"Recorded: {allocation}")

if st.session_state.paper_log_v3:
    st.subheader("Paper decision log")
    st.dataframe(pd.DataFrame(st.session_state.paper_log_v3), use_container_width=True)

st.divider()
st.caption(
    "Important: yfinance intraday data is suitable for experimentation, not "
    "execution-grade trading. This app deliberately has no Trading 212 "
    "connection. Validate the hypothesis forward in paper mode before "
    "considering real capital."
)
