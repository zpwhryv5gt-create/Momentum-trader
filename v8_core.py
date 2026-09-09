"""Frozen V8 research logic, extracted unchanged from momentum_app.py."""
import numpy as np
import pandas as pd

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


