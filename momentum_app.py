import io
import math
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

APP_TITLE = "Momentum Trader"
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "IYR"]

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("A simple research dashboard for the Momentum v1 strategy. Paper/research use only.")


def download_prices(tickers, start, end=None):
    import yfinance as yf
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("No market data returned.")
    if isinstance(raw.columns, pd.MultiIndex):
        field = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        px = raw[field].copy()
    else:
        field = "Adj Close" if "Adj Close" in raw.columns else "Close"
        px = raw[[field]].copy()
        px.columns = [tickers[0]]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.sort_index().dropna(how="all").ffill()
    return px


def signal_snapshot(prices, top_n=2):
    m6 = prices / prices.shift(126) - 1
    m12 = prices / prices.shift(252) - 1
    score = 0.5 * m6 + 0.5 * m12
    sma200 = prices.rolling(200).mean()
    latest = prices.index[-1]
    table = pd.DataFrame({
        "Price": prices.loc[latest],
        "6m": m6.loc[latest],
        "12m": m12.loc[latest],
        "Score": score.loc[latest],
        "SMA200": sma200.loc[latest],
    })
    table["Above 200d"] = table["Price"] > table["SMA200"]
    table["Eligible"] = table["Above 200d"] & (table["Score"] > 0)
    ranked = table[table["Eligible"]].sort_values("Score", ascending=False)
    winners = ranked.head(top_n).index.tolist()
    target = pd.Series(0.0, index=table.index)
    if winners:
        target.loc[winners] = 1.0 / len(winners)
    table["Target weight"] = target
    table = table.sort_values("Score", ascending=False)
    return latest, table


def build_target_weights(prices, top_n=2):
    m6 = prices / prices.shift(126) - 1
    m12 = prices / prices.shift(252) - 1
    score = 0.5 * m6 + 0.5 * m12
    sma200 = prices.rolling(200).mean()
    eligible = (prices > sma200) & (score > 0)
    month_end_dates = prices.groupby(prices.index.to_period("M")).tail(1).index
    signals = pd.DataFrame(0.0, index=month_end_dates, columns=prices.columns)
    for dt in month_end_dates:
        valid = score.loc[dt].where(eligible.loc[dt]).dropna().sort_values(ascending=False)
        winners = valid.head(top_n).index.tolist()
        if winners:
            signals.loc[dt, winners] = 1.0 / len(winners)
    daily = signals.reindex(prices.index).ffill().fillna(0.0).shift(1).fillna(0.0)
    return daily


def backtest(prices, initial=1000.0, top_n=2, cost_bps=10.0):
    weights = build_target_weights(prices, top_n=top_n)
    rets = prices.pct_change().fillna(0.0)
    gross = (weights * rets).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    costs = turnover * (cost_bps / 10000.0)
    net = gross - costs
    equity = initial * (1 + net).cumprod()
    bench_ticker = "SPY" if "SPY" in prices.columns else prices.columns[0]
    bench_ret = prices[bench_ticker].pct_change().fillna(0.0)
    benchmark = initial * (1 + bench_ret).cumprod()
    return weights, net, turnover, costs, equity, benchmark, bench_ticker


def perf_stats(equity, returns):
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    dd = equity / equity.cummax() - 1
    vol = returns.std() * math.sqrt(252)
    sharpe = returns.mean() / returns.std() * math.sqrt(252) if returns.std() else np.nan
    return {
        "End value": equity.iloc[-1],
        "CAGR": cagr,
        "Max drawdown": dd.min(),
        "Annual volatility": vol,
        "Sharpe": sharpe,
    }

with st.sidebar:
    st.header("Settings")
    initial = st.number_input("Starting capital (£)", min_value=100.0, value=1000.0, step=100.0)
    start = st.date_input("Backtest start", value=date(2015, 1, 1))
    top_n = st.selectbox("Number of holdings", [1, 2, 3], index=1)
    cost_bps = st.number_input("Trading cost (bps, one-way)", min_value=0.0, value=10.0, step=1.0)
    tickers_text = st.text_input("Universe", value=", ".join(DEFAULT_TICKERS))
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

if "paper_cash" not in st.session_state:
    st.session_state.paper_cash = 1000.0
if "paper_positions" not in st.session_state:
    st.session_state.paper_positions = {}
if "paper_log" not in st.session_state:
    st.session_state.paper_log = []

@st.cache_data(ttl=3600)
def cached_prices(tickers_tuple, start_iso):
    return download_prices(list(tickers_tuple), start_iso)

prices = None
try:
    prices = cached_prices(tuple(tickers), str(start))
except Exception as e:
    st.warning(f"Market data is not loaded yet: {e}")

col1, col2, col3, col4 = st.columns(4)
run_backtest_btn = col1.button("Run Backtest", use_container_width=True)
current_signal_btn = col2.button("Current Signal", use_container_width=True)
paper_btn = col3.button("Paper Portfolio", use_container_width=True)
performance_btn = col4.button("Performance", use_container_width=True)

if run_backtest_btn:
    if prices is None:
        st.error("No price data available.")
    else:
        weights, net, turnover, costs, equity, benchmark, bench_ticker = backtest(
            prices, initial=initial, top_n=top_n, cost_bps=cost_bps
        )
        stats = perf_stats(equity, net)
        bstats = perf_stats(benchmark, prices[bench_ticker].pct_change().fillna(0.0))
        a, b, c, d = st.columns(4)
        a.metric("Momentum end value", f"£{stats['End value']:,.0f}")
        b.metric("Momentum CAGR", f"{stats['CAGR']:.1%}")
        c.metric("Max drawdown", f"{stats['Max drawdown']:.1%}")
        d.metric("Benchmark end value", f"£{bstats['End value']:,.0f}")
        chart = pd.DataFrame({"Momentum v1": equity, bench_ticker: benchmark})
        st.line_chart(chart)
        st.subheader("Diagnostics")
        st.write(f"Total turnover units: **{turnover.sum():.2f}**")
        st.write(f"Approximate cumulative cost drag: **{costs.sum():.2%}**")
        st.write(f"Time invested: **{(weights.sum(axis=1) > 0).mean():.1%}**")

if current_signal_btn:
    if prices is None:
        st.error("No price data available.")
    else:
        dt, table = signal_snapshot(prices, top_n=top_n)
        st.subheader(f"Current signal — {dt.date()}")
        display = table.copy()
        for c in ["6m", "12m", "Score", "Target weight"]:
            display[c] = display[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
        display["Price"] = display["Price"].map(lambda x: f"{x:,.2f}")
        display["SMA200"] = display["SMA200"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
        st.dataframe(display, use_container_width=True)
        selected = table[table["Target weight"] > 0]["Target weight"]
        if selected.empty:
            st.success("Signal: 100% CASH")
        else:
            st.success("Signal: " + ", ".join(f"{t} {w:.0%}" for t, w in selected.items()))

if paper_btn:
    st.subheader("Paper portfolio")
    st.write("This does **not** place a real trade. It records the current signal as a simulated portfolio action.")
    paper_capital = st.number_input("Paper capital (£)", min_value=100.0, value=float(st.session_state.paper_cash), step=100.0)
    if st.button("Apply Current Signal to Paper Portfolio"):
        if prices is None:
            st.error("No price data available.")
        else:
            dt, table = signal_snapshot(prices, top_n=top_n)
            selected = table[table["Target weight"] > 0]["Target weight"]
            st.session_state.paper_cash = paper_capital
            st.session_state.paper_positions = {t: paper_capital * w for t, w in selected.items()}
            st.session_state.paper_log.append({
                "date": str(dt.date()),
                "capital": paper_capital,
                "allocation": ", ".join(f"{t} {w:.0%}" for t, w in selected.items()) if len(selected) else "CASH 100%",
            })
            st.success("Paper allocation updated.")
    if st.session_state.paper_positions:
        p = pd.DataFrame.from_dict(st.session_state.paper_positions, orient="index", columns=["£ allocated"])
        st.dataframe(p, use_container_width=True)
    else:
        st.write("Current paper allocation: **CASH**")
    if st.session_state.paper_log:
        st.dataframe(pd.DataFrame(st.session_state.paper_log), use_container_width=True)

if performance_btn:
    if prices is None:
        st.error("No price data available.")
    else:
        weights, net, turnover, costs, equity, benchmark, bench_ticker = backtest(
            prices, initial=initial, top_n=top_n, cost_bps=cost_bps
        )
        s = perf_stats(equity, net)
        bs = perf_stats(benchmark, prices[bench_ticker].pct_change().fillna(0.0))
        st.subheader("Performance summary")
        rows = pd.DataFrame({
            "Momentum v1": [s["End value"], s["CAGR"], s["Max drawdown"], s["Annual volatility"], s["Sharpe"]],
            bench_ticker: [bs["End value"], bs["CAGR"], bs["Max drawdown"], bs["Annual volatility"], bs["Sharpe"]],
        }, index=["End value", "CAGR", "Max drawdown", "Annual volatility", "Sharpe"])
        st.dataframe(rows, use_container_width=True)

st.divider()
st.caption("No broker connection is included. The dashboard is intentionally research/paper-trading only.")
