import math
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

APP_TITLE = "Momentum Trader"
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "IYR"]

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Momentum v1 + v2 research dashboard. Paper/research use only.")

def download_prices(tickers, start, end=None):
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False,
                      progress=False, group_by="column", threads=True)
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
    return px.sort_index().dropna(how="all").ffill()

def signal_components(prices, version):
    if version == "V1 Slow":
        m6 = prices / prices.shift(126) - 1
        m12 = prices / prices.shift(252) - 1
        score = 0.5*m6 + 0.5*m12
        trend = prices.rolling(200).mean()
        eligible = (prices > trend) & (score > 0)
        return score, trend, eligible, {"6m":m6, "12m":m12}
    # V2 is frozen as a separate hypothesis: faster 1/3/6m signal,
    # 100-day trend filter, weekly rebalance.
    m1 = prices / prices.shift(21) - 1
    m3 = prices / prices.shift(63) - 1
    m6 = prices / prices.shift(126) - 1
    score = 0.50*m1 + 0.30*m3 + 0.20*m6
    trend = prices.rolling(100).mean()
    eligible = (prices > trend) & (score > 0)
    return score, trend, eligible, {"1m":m1, "3m":m3, "6m":m6}

def rebalance_dates(prices, version):
    if version == "V1 Slow":
        return prices.groupby(prices.index.to_period("M")).tail(1).index
    # last trading day of each week
    return prices.groupby(prices.index.to_period("W-FRI")).tail(1).index

def build_target_weights(prices, version, top_n=2):
    score, trend, eligible, _ = signal_components(prices, version)
    dates = rebalance_dates(prices, version)
    signals = pd.DataFrame(0.0, index=dates, columns=prices.columns)
    for dt in dates:
        valid = score.loc[dt].where(eligible.loc[dt]).dropna().sort_values(ascending=False)
        winners = valid.head(top_n).index.tolist()
        if winners:
            signals.loc[dt, winners] = 1/len(winners)
    # next-day implementation prevents same-close look-ahead
    return signals.reindex(prices.index).ffill().fillna(0).shift(1).fillna(0)

def signal_snapshot(prices, version, top_n=2):
    score, trend, eligible, comps = signal_components(prices, version)
    latest = prices.index[-1]
    data = {"Price": prices.loc[latest]}
    for label, frame in comps.items():
        data[label] = frame.loc[latest]
    data["Score"] = score.loc[latest]
    data["Trend"] = trend.loc[latest]
    table = pd.DataFrame(data)
    table["Above trend"] = table["Price"] > table["Trend"]
    table["Eligible"] = eligible.loc[latest]
    ranked = table[table["Eligible"]].sort_values("Score", ascending=False)
    winners = ranked.head(top_n).index.tolist()
    table["Target weight"] = 0.0
    if winners:
        table.loc[winners, "Target weight"] = 1/len(winners)
    return latest, table.sort_values("Score", ascending=False)

def backtest(prices, version, initial=1000, top_n=2, cost_bps=10):
    weights = build_target_weights(prices, version, top_n)
    rets = prices.pct_change().fillna(0)
    gross = (weights * rets).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    costs = turnover * cost_bps/10000
    net = gross - costs
    equity = initial * (1+net).cumprod()
    bench = "SPY" if "SPY" in prices else prices.columns[0]
    bench_ret = prices[bench].pct_change().fillna(0)
    benchmark = initial * (1+bench_ret).cumprod()
    return weights, net, turnover, costs, equity, benchmark, bench

def stats(equity, rets):
    years = (equity.index[-1]-equity.index[0]).days/365.25
    cagr = (equity.iloc[-1]/equity.iloc[0])**(1/years)-1 if years else np.nan
    dd = equity/equity.cummax()-1
    vol = rets.std()*math.sqrt(252)
    sharpe = rets.mean()/rets.std()*math.sqrt(252) if rets.std() else np.nan
    return equity.iloc[-1], cagr, dd.min(), vol, sharpe

with st.sidebar:
    st.header("Settings")
    initial = st.number_input("Starting capital (£)", min_value=100.0, value=1000.0, step=100.0)
    start = st.date_input("Backtest start", value=date(2015,1,1))
    top_n = st.selectbox("Number of holdings", [1,2,3], index=1)
    cost_bps = st.number_input("Trading cost (bps, one-way)", min_value=0.0, value=10.0, step=1.0)
    txt = st.text_input("Universe", value=", ".join(DEFAULT_TICKERS))
    tickers = [x.strip().upper() for x in txt.split(",") if x.strip()]

@st.cache_data(ttl=3600)
def cached_prices(tickers_tuple, start_iso):
    return download_prices(list(tickers_tuple), start_iso)

try:
    prices = cached_prices(tuple(tickers), str(start))
except Exception as e:
    prices = None
    st.warning(f"Market data is not loaded yet: {e}")

st.subheader("The experiment")
st.write("**V1 Slow:** 6/12-month momentum + 200-day trend, monthly rebalance.")
st.write("**V2 Fast:** 1/3/6-month momentum + 100-day trend, weekly rebalance. V2 is kept separate so V1 is not rewritten after seeing its result.")

if st.button("Compare V1 vs V2 vs SPY", use_container_width=True):
    if prices is None:
        st.error("No price data.")
    else:
        results = {}
        chart = {}
        for version in ["V1 Slow","V2 Fast"]:
            w,r,t,c,e,b,bench = backtest(prices, version, initial, top_n, cost_bps)
            results[version] = (*stats(e,r), t.sum(), c.sum(), (w.sum(axis=1)>0).mean())
            chart[version] = e
        chart["SPY"] = b
        bs = stats(b, prices[bench].pct_change().fillna(0))
        rows = []
        for v in ["V1 Slow","V2 Fast"]:
            end,cagr,dd,vol,sh,turn,cost,invested = results[v]
            rows.append([v,end,cagr,dd,vol,sh,turn,cost,invested])
        rows.append(["SPY",bs[0],bs[1],bs[2],bs[3],bs[4],np.nan,np.nan,1.0])
        df = pd.DataFrame(rows, columns=["Strategy","End £","CAGR","Max DD","Volatility","Sharpe","Turnover","Cost drag","Invested"])
        st.dataframe(df.style.format({"End £":"£{:,.0f}","CAGR":"{:.1%}","Max DD":"{:.1%}",
                                      "Volatility":"{:.1%}","Sharpe":"{:.2f}","Turnover":"{:.1f}",
                                      "Cost drag":"{:.1%}","Invested":"{:.1%}"}), use_container_width=True)
        st.line_chart(pd.DataFrame(chart))

if st.button("Current V1 + V2 Signals", use_container_width=True):
    if prices is None:
        st.error("No price data.")
    else:
        for version in ["V1 Slow","V2 Fast"]:
            dt, table = signal_snapshot(prices, version, top_n)
            selected = table[table["Target weight"]>0]["Target weight"]
            st.subheader(f"{version} — {dt.date()}")
            if selected.empty:
                st.success("Signal: CASH 100%")
            else:
                st.success("Signal: " + ", ".join(f"{t} {w:.0%}" for t,w in selected.items()))
            show = table.copy()
            for c in show.columns:
                if c in ["1m","3m","6m","12m","Score","Target weight"]:
                    show[c] = show[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
            st.dataframe(show, use_container_width=True)

st.divider()
st.caption("No broker connection. No live orders. Backtests can overstate future performance; V2 should be treated as a new hypothesis, not proof of an edge.")
