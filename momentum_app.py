from itertools import product

import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "Momentum Trader — V8 Frozen Paper Test"
DEFAULT_TICKERS = [
    "VWRP.L", "SWDA.L", "CSP1.L", "EQQQ.L", "IITU.L", "EMIM.L", "IGLN.L"
]
DEFAULT_PARAMS = {
    # Frozen after V8 development. Do not alter during the forward paper test.
    "eval_bars": 4,
    "ema_fast": 70,
    "ema_slow": 280,
    "mom_fast": 21,
    "mom_mid": 35,
    "mom_long": 280,
    "min_hold_bars": 35,
    "switch_buffer": 0.01,
    "entry_floor": 0.001,
    "target_vol": 0.12,
    "vol_lookback": 140,
}


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(
    "Regime-aware, low-turnover momentum research with train/validation/holdout testing. "
    "Paper use only — no broker connection and no live orders."
)


def download_intraday(tickers, period="2y", interval="1h"):
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
        "1d": r_fast,
        "1w": r_mid,
        "1m": r_long,
    }


def target_weights(prices, params):
    """Stateful one-position engine with hysteresis and a next-bar execution lag."""
    score, _, _, trend, eligible, _ = components(prices, params)
    annual_vol = (
        prices.pct_change(fill_method=None)
        .rolling(
            params.get("vol_lookback", 140),
            min_periods=max(35, params.get("vol_lookback", 140) // 2),
        )
        .std()
        * np.sqrt(252.0 * 7.0)
    )
    decisions = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    current = None
    current_weight = 0.0
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
            if next_position is None:
                current_weight = 0.0
            else:
                observed_vol = annual_vol.loc[timestamp].get(next_position, np.nan)
                current_weight = (
                    float(np.clip(params.get("target_vol", 0.12) / observed_vol, 0.25, 1.0))
                    if np.isfinite(observed_vol) and observed_vol > 0.0
                    else 0.5
                )
        current = next_position
        decisions.loc[timestamp] = 0.0
        if current is not None:
            decisions.loc[timestamp, current] = current_weight

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
            "actions": 0, "turnover": 0.0, "active_days": 0, "days": 0,
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
        "days": int(len(daily)),
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
        [1, 4, 7],
        [(35, 140), (70, 280)],
        [7, 21],
        [14, 35],
        [0.005, 0.01],
    ):
        candidates.append({
            "eval_bars": eval_bars,
            "ema_fast": ema_pair[0],
            "ema_slow": ema_pair[1],
            "mom_fast": mom_fast,
            "mom_mid": 35,
            "mom_long": ema_pair[1],
            "min_hold_bars": min_hold,
            "switch_buffer": switch_buffer,
            "entry_floor": 0.001,
            "target_vol": 0.12,
            "vol_lookback": 140,
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
        # Positions do not change in a cost stress, so reuse them instead of
        # recomputing the entire state machine.
        extra_cost = turnover * (cost_bps * 0.5 / 10000.0)
        stressed_net = net - extra_cost
        stressed_turnover = turnover
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
            "selection_actions": train["actions"] + validation["actions"],
            "selection_days": train["days"] + validation["days"],
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
        & (
            results["selection_actions"]
            / results["selection_days"].clip(lower=1)
            <= 0.5
        )
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
    benchmark_ticker = "VWRP.L" if "VWRP.L" in prices.columns else prices.columns[0]
    benchmark_net = prices[benchmark_ticker].pct_change(fill_method=None).fillna(0.0)
    no_turnover = pd.Series(0.0, index=prices.index)
    benchmark = {
        "ticker": benchmark_ticker,
        "Train": segment_stats(benchmark_net, no_turnover, 0, train_end),
        "Validation": segment_stats(
            benchmark_net, no_turnover, train_end, validation_end
        ),
        "Holdout": segment_stats(benchmark_net, no_turnover, validation_end, n),
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
        and holdout["actions"] / max(holdout["days"], 1) <= 0.5
        and winner["total_actions"] / max(
            stats["Train"]["days"]
            + stats["Validation"]["days"]
            + stats["Holdout"]["days"],
            1,
        ) <= 0.5
    )
    return results, winner["params"], stats, benchmark, net, weights, passed


def validate_frozen(prices, params, cost_bps):
    """Evaluate one preregistered configuration; this function never searches parameters."""
    n = len(prices)
    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)
    net, turnover, weights = backtest(prices, params, cost_bps)
    stressed_net = net - turnover * (cost_bps * 0.5 / 10000.0)
    stats = {
        "Train": segment_stats(net, turnover, 0, train_end),
        "Validation": segment_stats(net, turnover, train_end, validation_end),
        "Holdout": segment_stats(net, turnover, validation_end, n),
    }
    stressed_holdout = segment_stats(stressed_net, turnover, validation_end, n)
    benchmark_ticker = "VWRP.L" if "VWRP.L" in prices.columns else prices.columns[0]
    benchmark_net = prices[benchmark_ticker].pct_change(fill_method=None).fillna(0.0)
    no_turnover = pd.Series(0.0, index=prices.index)
    benchmark = {
        "ticker": benchmark_ticker,
        "Train": segment_stats(benchmark_net, no_turnover, 0, train_end),
        "Validation": segment_stats(
            benchmark_net, no_turnover, train_end, validation_end
        ),
        "Holdout": segment_stats(
            benchmark_net, no_turnover, validation_end, n
        ),
    }
    total_days = sum(stats[split]["days"] for split in stats)
    total_actions = int((turnover > 1e-12).sum())
    holdout = stats["Holdout"]
    passed = (
        stats["Train"]["return"] > 0.0
        and stats["Validation"]["return"] > 0.0
        and holdout["return"] > 0.0
        and stressed_holdout["return"] > 0.0
        and holdout["max_dd"] > -0.08
        and holdout["actions"] / max(holdout["days"], 1) <= 0.5
        and total_actions / max(total_days, 1) <= 0.5
    )
    return stats, benchmark, net, weights, passed


def current_snapshot(prices, params):
    score, ema_fast, ema_slow, trend, eligible, returns = components(prices, params)
    weights = target_weights(prices, params)
    latest = prices.index[-1]
    data = {
        "Price": prices.loc[latest],
        "1d": returns["1d"].loc[latest],
        "1w": returns["1w"].loc[latest],
        "1m": returns["1m"].loc[latest],
        "Score": score.loc[latest],
        "EMA fast": ema_fast.loc[latest],
        "EMA slow": ema_slow.loc[latest],
        "Trend OK": trend.loc[latest],
        "Entry eligible": eligible.loc[latest],
        "Model weight": weights.loc[latest],
    }
    return latest, pd.DataFrame(data).sort_values("Score", ascending=False)


with st.sidebar:
    st.header("V8 settings")
    initial = st.number_input("Paper capital (£)", min_value=100.0, value=1000.0, step=100.0)
    cost_bps = st.number_input(
        "One-way spread + slippage (bps)", min_value=0.0, value=8.0, step=1.0,
        help="Charged on every unit of portfolio turnover; zero commission is not zero cost.",
    )
    ticker_text = st.text_input("GBP-listed ETF universe", value=", ".join(DEFAULT_TICKERS))
    tickers = [item.strip().upper() for item in ticker_text.split(",") if item.strip()]



st.subheader("Audited forward paper account")
@st.cache_data(ttl=60, show_spinner=False)
def load_audited_daily():
    return pd.read_csv(
        "https://raw.githubusercontent.com/zpwhryv5gt-create/Momentum-trader/main/paper/daily.csv"
    )
try:
    audited = load_audited_daily()
    last = audited.iloc[-1]
    st.caption(f"Last price bar: {last['market_time']}. Current-day figures are provisional.")
    a, b, c = st.columns(3)
    a.metric("Paper account", f"£{last['equity_gbp']:,.2f}",
             f"£{last['day_pnl_gbp']:+.2f} today")
    b.metric("Since start", f"{last['return_pct']:+.2f}%")
    c.metric("VWRP since start", f"{last['benchmark_return_pct']:+.2f}%")
    st.line_chart(audited.set_index("london_date")[["equity_gbp", "benchmark_gbp"]])
    st.caption("Starts with £1,000 fresh cash; historical target-only entries are excluded.")
    st.markdown("[Latest report and pending orders](https://github.com/zpwhryv5gt-create/Momentum-trader/blob/main/paper/REPORT.md) · [Runner status](https://github.com/zpwhryv5gt-create/Momentum-trader/actions/workflows/v8-paper.yml)")
    stamp = pd.Timestamp(last["market_time"])
    if pd.Timestamp.now(tz="UTC") - stamp > pd.Timedelta(hours=24):
        st.warning("This valuation is more than 24 hours old. Check runner status and the market calendar.")
except Exception:
    st.info("The audited account has not published an available valuation yet.")
    st.markdown("[Check the scheduled runner](https://github.com/zpwhryv5gt-create/Momentum-trader/actions/workflows/v8-paper.yml)")
st.divider()

st.subheader("Frozen V8 validation")
st.write(
    "This paper-test build evaluates the single configuration selected during V8 "
    "development. It does not search, rank or replace parameters. The historical "
    "split is a safety check, not fresh evidence."
)
st.caption(
    "Two years of hourly data · fixed 8 bps baseline cost · 1.5× cost stress · "
    "12% volatility target · VWRP comparison."
)

left, middle = st.columns(2)
validate_btn = left.button("Validate frozen V8", use_container_width=True)
signal_btn = middle.button("Current frozen signal", use_container_width=True)
paper_btn = False

prices = None
if validate_btn or signal_btn or paper_btn:
    try:
        with st.spinner("Loading and checking hourly market data…"):
            prices = cached_prices(tuple(tickers))
    except Exception as error:
        st.error(f"Market data could not be loaded: {error}")
else:
    st.info("Start with ‘Validate frozen V8’. Market data loads only when requested.")

if validate_btn and prices is not None:
    params = DEFAULT_PARAMS.copy()
    with st.spinner("Validating the fixed configuration…"):
        stats, benchmark, net, weights, passed = validate_frozen(
            prices, params, cost_bps
        )
    st.session_state.v8_params = params
    st.session_state.v8_passed = passed
    st.session_state.v8_stats = stats

    if passed:
        st.success(
            "Frozen V8 safety gate: PASSED. Continue paper testing; "
            "this is not live-trading approval."
        )
    else:
        st.warning(
            "Frozen V8 safety gate: FAILED. V8 will recommend CASH "
            "until the fixed model passes."
        )

    st.subheader("Fixed model: historical split results")
    metric_columns = st.columns(3)
    for column, split in zip(metric_columns, ["Train", "Validation", "Holdout"]):
        item = stats[split]
        with column:
            st.metric(f"{split} return", f"{item['return']:.2%}")
            st.caption(
                f"Max drawdown {item['max_dd']:.2%} · "
                f"actions {item['actions']} · Sharpe {item['sharpe']:.2f}"
            )
    st.caption(
        f"Passive {benchmark['ticker']} comparison — train "
        f"{benchmark['Train']['return']:.2%}, validation "
        f"{benchmark['Validation']['return']:.2%}, holdout "
        f"{benchmark['Holdout']['return']:.2%}."
    )
    st.subheader("Frozen controls")
    st.json(params)
    equity = initial * (1.0 + net).cumprod()
    st.line_chart(equity.rename("V8 modelled equity"))


def active_configuration():
    return (
        st.session_state.get("v8_params", DEFAULT_PARAMS),
        bool(st.session_state.get("v8_passed", False)),
    )


if signal_btn and prices is not None:
    params, passed = active_configuration()
    timestamp, table = current_snapshot(prices, params)
    selected = table[table["Model weight"] > 0.0]["Model weight"]
    st.subheader(f"Current V8 signal — {timestamp}")
    if not passed:
        st.warning("V8 safety gate has not passed: CASH 100%")
    elif selected.empty:
        st.success("V8 signal: CASH 100%")
    else:
        st.success(
            "V8 paper signal: "
            + ", ".join(f"{ticker} {weight:.0%}" for ticker, weight in selected.items())
        )
    display = table.copy()
    for column in ["1d", "1w", "1m", "Score", "Model weight"]:
        display[column] = display[column].map(
            lambda value: f"{value:.2%}" if pd.notna(value) else ""
        )
    st.dataframe(display, use_container_width=True)



st.divider()
st.caption(
    "Research limitation: yfinance hourly data is not execution-grade. The fixed historical split has already informed development. A positive holdout is "
    "only permission to continue forward paper testing, never proof of future profit. V8 "
    "deliberately cannot place a real order."
)
