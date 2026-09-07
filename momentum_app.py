from itertools import product

import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "Momentum Trader — V5 Robust Ensemble Lab"
DEFAULT_TICKERS = [
    "VWRP.L", "SWDA.L", "CSP1.L", "EQQQ.L", "IITU.L", "EMIM.L", "IGLN.L"
]
DEFAULT_PARAMS = {
    "eval_bars": 24,
    "ema_fast": 156,
    "ema_slow": 780,
    "mom_fast": 36,
    "mom_mid": 156,
    "mom_long": 780,
    "min_hold_bars": 390,
    "switch_buffer": 0.008,
    "entry_floor": 0.001,
}


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(
    "Regime-aware, low-turnover momentum research with train/validation/holdout testing. "
    "Paper use only — no broker connection and no live orders."
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
        timeout=25,
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


def components(prices, params):
    r_fast = prices / prices.shift(params["mom_fast"]) - 1.0
    r_mid = prices / prices.shift(params["mom_mid"]) - 1.0
    r_long = prices / prices.shift(params["mom_long"]) - 1.0
    score = 0.20 * r_fast + 0.50 * r_mid + 0.30 * r_long
    ema_fast = prices.ewm(span=params["ema_fast"], adjust=False).mean()
    ema_slow = prices.ewm(span=params["ema_slow"], adjust=False).mean()
    trend = (prices > ema_fast) & (ema_fast > ema_slow)
    eligible = (
        trend
        & (r_fast > params["entry_floor"])
        & (r_mid > 0.0)
        & (r_long > 0.0)
        & (score > 0.0)
    )
    return score, ema_fast, ema_slow, trend, eligible, {
        "1h": r_fast,
        "1d": r_mid,
        "5d": r_long,
    }


def target_weights(prices, params):
    """Stateful one-position engine with hysteresis and a next-bar execution lag."""
    score, _, _, trend, eligible, _ = components(prices, params)
    decisions = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    current = None
    entry_bar = -10**9

    warmup = max(params["ema_slow"], params["mom_long"])
    eval_rows = range(warmup, len(prices), params["eval_bars"])
    for row in eval_rows:
        timestamp = prices.index[row]
        next_position = current
        held_long_enough = row - entry_bar >= params["min_hold_bars"]

        valid = score.loc[timestamp].where(eligible.loc[timestamp]).dropna()
        best = valid.idxmax() if not valid.empty else None

        if current is not None:
            current_trend = bool(trend.loc[timestamp].get(current, False))
            current_mid = prices[current].iloc[row] / prices[current].iloc[
                max(0, row - params["mom_mid"])
            ] - 1.0
            # Risk exits are immediate; rotation requires a minimum hold and a margin.
            if (not current_trend) or (not np.isfinite(current_mid)) or current_mid < -0.002:
                next_position = None
            elif best is not None and best != current and held_long_enough:
                incumbent_score = score.loc[timestamp].get(current, np.nan)
                advantage = valid.loc[best] - incumbent_score
                if np.isfinite(advantage) and advantage > params["switch_buffer"]:
                    next_position = best
        elif best is not None:
            next_position = best

        if next_position != current:
            entry_bar = row
        current = next_position
        decisions.loc[timestamp] = 0.0
        if current is not None:
            decisions.loc[timestamp, current] = 1.0

    weights = decisions.ffill().fillna(0.0)
    # A decision made using a bar close can only be acted on from the next bar.
    return weights.shift(1).fillna(0.0)


def backtest(prices, params, cost_bps):
    weights = target_weights(prices, params)
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    gross = (weights * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * (cost_bps / 10000.0)
    net = gross - costs
    return net, turnover, weights


def segment_stats(net, turnover, start, end):
    segment_net = net.iloc[start:end]
    segment_turnover = turnover.iloc[start:end]
    if segment_net.empty:
        return {
            "return": np.nan, "max_dd": np.nan, "sharpe": np.nan,
            "actions": 0, "turnover": 0.0, "active_days": 0,
        }
    equity = (1.0 + segment_net).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    frame = pd.DataFrame({"net": segment_net})
    try:
        frame.index = frame.index.tz_convert("Europe/London")
    except (TypeError, AttributeError):
        pass
    daily = frame.groupby(frame.index.date)["net"].apply(lambda x: (1.0 + x).prod() - 1.0)
    volatility = daily.std(ddof=1)
    sharpe = (
        float(daily.mean() / volatility * np.sqrt(252.0))
        if len(daily) > 2 and volatility > 0
        else np.nan
    )
    active = segment_turnover.groupby(segment_turnover.index.date).sum() > 0
    return {
        "return": float(equity.iloc[-1] - 1.0),
        "max_dd": float(drawdown.min()),
        "sharpe": sharpe,
        "actions": int((segment_turnover > 1e-12).sum()),
        "turnover": float(segment_turnover.sum()),
        "active_days": int(active.sum()),
    }


def objective(stats):
    if not np.isfinite(stats["return"]):
        return -999.0
    sharpe = stats["sharpe"] if np.isfinite(stats["sharpe"]) else -1.0
    inactivity_penalty = 0.02 if stats["actions"] == 0 else 0.0
    return (
        stats["return"]
        + 0.025 * np.clip(sharpe, -3.0, 3.0)
        + 0.40 * stats["max_dd"]
        - 0.00025 * stats["actions"]
        - inactivity_penalty
    )


def candidate_grid():
    candidates = []
    for eval_bars, ema_pair, mom_fast, min_hold, switch_buffer in product(
        [12, 24, 78],
        [(78, 390), (156, 780)],
        [36, 78],
        [156, 390],
        [0.004, 0.008],
    ):
        candidates.append({
            "eval_bars": eval_bars,
            "ema_fast": ema_pair[0],
            "ema_slow": ema_pair[1],
            "mom_fast": mom_fast,
            "mom_mid": 156,
            "mom_long": 780,
            "min_hold_bars": min_hold,
            "switch_buffer": switch_buffer,
            "entry_floor": 0.001,
        })
    return candidates


@st.cache_data(ttl=300, show_spinner=False)
def cached_prices(tickers_tuple):
    return download_intraday(list(tickers_tuple))


@st.cache_data(ttl=300, show_spinner=False)
def optimise(prices, cost_bps):
    n = len(prices)
    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)
    rows = []
    series = {}
    for candidate_id, params in enumerate(candidate_grid(), start=1):
        net, turnover, weights = backtest(prices, params, cost_bps)
        stressed_net, stressed_turnover, _ = backtest(
            prices, params, cost_bps * 1.5
        )
        train = segment_stats(net, turnover, 0, train_end)
        validation = segment_stats(net, turnover, train_end, validation_end)
        test = segment_stats(net, turnover, validation_end, n)
        stressed_test = segment_stats(
            stressed_net, stressed_turnover, validation_end, n
        )
        rows.append({
            "candidate": candidate_id,
            "train_score": objective(train),
            "validation_score": objective(validation),
            "train_return": train["return"],
            "validation_return": validation["return"],
            "test_return": test["return"],
            "test_max_dd": test["max_dd"],
            "test_actions": test["actions"],
            "stressed_test_return": stressed_test["return"],
            "total_actions": int((turnover > 1e-12).sum()),
            "total_turnover": float(turnover.sum()),
            "params": params,
        })
        series[candidate_id] = (net, turnover, weights)

    results = pd.DataFrame(rows)
    # Select for consistency, not a single lucky patch. Holdout is never used here.
    results["stability"] = (
        results[["train_score", "validation_score"]].min(axis=1)
        - 0.35 * (results["train_return"] - results["validation_return"]).abs()
    )
    robust_pool = results[
        (results["train_return"] > 0.0)
        & (results["validation_return"] > 0.0)
        & (results["total_actions"] <= 80)
    ]
    selection_pool = robust_pool if not robust_pool.empty else results
    winner = selection_pool.nlargest(1, "stability").iloc[0]
    winner_id = int(winner["candidate"])
    net, turnover, weights = series[winner_id]
    stats = {
        "Train": segment_stats(net, turnover, 0, train_end),
        "Validation": segment_stats(net, turnover, train_end, validation_end),
        "Holdout": segment_stats(net, turnover, validation_end, n),
    }
    # A failed gate means the safe model output is cash, not a forced trade.
    holdout = stats["Holdout"]
    passed = (
        not robust_pool.empty
        and stats["Train"]["return"] > 0.0
        and stats["Validation"]["return"] > 0.0
        and holdout["return"] > 0.0
        and winner["stressed_test_return"] > 0.0
        and holdout["max_dd"] > -0.08
        and holdout["actions"] <= 8
        and winner["total_actions"] <= 80
    )
    return results, winner["params"], stats, net, weights, passed


def current_snapshot(prices, params):
    score, ema_fast, ema_slow, trend, eligible, returns = components(prices, params)
    weights = target_weights(prices, params)
    latest = prices.index[-1]
    data = {
        "Price": prices.loc[latest],
        "1h": returns["1h"].loc[latest],
        "1d": returns["1d"].loc[latest],
        "5d": returns["5d"].loc[latest],
        "Score": score.loc[latest],
        "EMA fast": ema_fast.loc[latest],
        "EMA slow": ema_slow.loc[latest],
        "Trend OK": trend.loc[latest],
        "Entry eligible": eligible.loc[latest],
        "Model weight": weights.loc[latest],
    }
    return latest, pd.DataFrame(data).sort_values("Score", ascending=False)


with st.sidebar:
    st.header("V5 settings")
    initial = st.number_input("Paper capital (£)", min_value=100.0, value=1000.0, step=100.0)
    cost_bps = st.number_input(
        "One-way spread + slippage (bps)", min_value=0.0, value=8.0, step=1.0,
        help="Charged on every unit of portfolio turnover; zero commission is not zero cost.",
    )
    ticker_text = st.text_input("GBP-listed ETF universe", value=", ".join(DEFAULT_TICKERS))
    tickers = [item.strip().upper() for item in ticker_text.split(",") if item.strip()]


st.subheader("V5 feedback loop")
st.write(
    "V5 tests 48 slower variants. It selects on the **weaker** of training and "
    "validation performance, then reveals the final 20% only after selection. The "
    "strategy must also survive 1.5× trading costs. A safety gate forces **cash** "
    "unless every stage is positive."
)
st.caption(
    "It evaluates every 1–6.5 hours, uses multi-day momentum, minimum 2–5 day holds "
    "and wider switch hysteresis to suppress noise and excessive trading."
)

left, middle, right = st.columns(3)
optimise_btn = left.button("Run V5 robustness loop", use_container_width=True)
signal_btn = middle.button("Current V5 signal", use_container_width=True)
paper_btn = right.button("Record paper decision", use_container_width=True)

prices = None
if optimise_btn or signal_btn or paper_btn:
    try:
        with st.spinner("Loading and checking five-minute market data…"):
            prices = cached_prices(tuple(tickers))
    except Exception as error:
        st.error(f"Market data could not be loaded: {error}")
else:
    st.info("Start with ‘Run V5 robustness loop’. Market data loads only when requested.")


if optimise_btn and prices is not None:
    with st.spinner("Running the train → validation → holdout feedback loop…"):
        results, params, stats, net, weights, passed = optimise(prices, cost_bps)
    st.session_state.v5_params = params
    st.session_state.v5_passed = passed
    st.session_state.v5_stats = stats

    if passed:
        st.success("V5 safety gate: PASSED. Continue paper testing; this is not live-trading approval.")
    else:
        st.warning("Safety gate: FAILED. V5 will recommend CASH until a robust edge is demonstrated.")

    st.subheader("Selected model: honest split results")
    metric_columns = st.columns(3)
    for column, split in zip(metric_columns, ["Train", "Validation", "Holdout"]):
        item = stats[split]
        with column:
            st.metric(f"{split} return", f"{item['return']:.2%}")
            st.caption(
                f"Max drawdown {item['max_dd']:.2%} · "
                f"actions {item['actions']} · Sharpe {item['sharpe']:.2f}"
            )

    st.subheader("Selected controls")
    st.json(params)
    equity = initial * (1.0 + net).cumprod()
    st.line_chart(equity.rename("V5 modelled equity"))

    st.subheader("Sensitivity across all 48 candidates")
    show = results.drop(columns=["params"]).sort_values("validation_score", ascending=False).copy()
    for column in ["train_return", "validation_return", "test_return", "test_max_dd"]:
        show[column] = show[column].map(lambda value: f"{value:.2%}")
    st.dataframe(show.head(12), use_container_width=True, hide_index=True)


def active_configuration():
    return (
        st.session_state.get("v5_params", DEFAULT_PARAMS),
        bool(st.session_state.get("v5_passed", False)),
    )


if signal_btn and prices is not None:
    params, passed = active_configuration()
    timestamp, table = current_snapshot(prices, params)
    selected = table[table["Model weight"] > 0.0]["Model weight"]
    st.subheader(f"Current V5 signal — {timestamp}")
    if not passed:
        st.warning("V5 safety gate has not passed: CASH 100%")
    elif selected.empty:
        st.success("V5 signal: CASH 100%")
    else:
        st.success(
            "V5 paper signal: "
            + ", ".join(f"{ticker} {weight:.0%}" for ticker, weight in selected.items())
        )
    display = table.copy()
    for column in ["1h", "1d", "5d", "Score", "Model weight"]:
        display[column] = display[column].map(
            lambda value: f"{value:.2%}" if pd.notna(value) else ""
        )
    st.dataframe(display, use_container_width=True)


if "paper_log_v5" not in st.session_state:
    st.session_state.paper_log_v5 = []

if paper_btn and prices is not None:
    params, passed = active_configuration()
    timestamp, table = current_snapshot(prices, params)
    selected = table[table["Model weight"] > 0.0]["Model weight"]
    allocation = "CASH 100%"
    if passed and not selected.empty:
        allocation = ", ".join(
            f"{ticker} {weight:.0%}" for ticker, weight in selected.items()
        )
    record = {"time": str(timestamp), "capital": float(initial), "allocation": allocation}
    duplicate = any(row["time"] == record["time"] for row in st.session_state.paper_log_v5)
    if duplicate:
        st.info("That market timestamp is already recorded; no duplicate was added.")
    else:
        st.session_state.paper_log_v5.append(record)
        st.success(f"Recorded: {allocation}")

if st.session_state.paper_log_v5:
    st.subheader("Paper decision log")
    log = pd.DataFrame(st.session_state.paper_log_v5)
    st.dataframe(log, use_container_width=True, hide_index=True)
    st.download_button(
        "Download paper log (CSV)",
        log.to_csv(index=False).encode("utf-8"),
        file_name="momentum_v5_paper_log.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Research limitation: 60 days of yfinance five-minute data is a small, non-execution-grade "
    "sample. A positive holdout is only permission to continue forward paper testing, never proof "
    "of future profit. V5 deliberately cannot place a real order."
)
