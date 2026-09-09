"""Audited forward-only V8 paper ledger. No broker interface."""
from pathlib import Path
import json
import os
import traceback
import numpy as np
import pandas as pd
import yfinance as yf
from v8_core import DEFAULT_TICKERS, DEFAULT_PARAMS, target_weights, validate_frozen

ROOT = Path("paper")
COST = 0.0008
CAPITAL = 1000.0

def currency_scale(currency):
    if currency == "GBP":
        return 1.0
    if currency in ("GBp", "GBX"):
        return 0.01
    raise ValueError(f"Unsupported or unverified quote currency: {currency!r}")

def utc(value):
    return pd.Timestamp(value).tz_convert("UTC")

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)

def append(path, rows):
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)

def get_market(now, started=None):
    opens, closes, units = {}, {}, {}
    for symbol in DEFAULT_TICKERS:
        ticker = yf.Ticker(symbol)
        frame = ticker.history(period="2y" if started is None else "1mo",
                               interval="1h", auto_adjust=False, actions=True,
                               prepost=False, raise_errors=True)
        if frame.empty or frame.index.tz is None:
            raise ValueError(f"Missing timestamped data: {symbol}")
        currency = ticker.history_metadata.get("currency")
        scale = 1.0 if currency == "USD" else currency_scale(currency)
        units[symbol] = {"currency": currency, "scale": scale}
        frame.index = frame.index.tz_convert("UTC")
        # Conservative: a bar is final only after a full hour plus five minutes.
        frame = frame.loc[frame.index + pd.Timedelta(minutes=65) <= now]
        if started:
            actions = frame.loc[frame.index >= utc(started)]
            for col in ("Dividends", "Stock Splits"):
                if col in actions and actions[col].fillna(0).ne(0).any():
                    raise ValueError(f"{symbol}: corporate action requires ledger reconciliation")
        if frame.empty:
            raise ValueError(f"No completed bars: {symbol}")
        opens[symbol] = frame["Open"] * scale
        closes[symbol] = frame["Close"] * scale
    o = pd.DataFrame(opens).sort_index()
    c = pd.DataFrame(closes).sort_index()
    # Never forward-fill execution or valuation prices.
    c = c.dropna()
    o = o.reindex(c.index).dropna()
    c = c.reindex(o.index)
    if c.empty or not np.isfinite(c.to_numpy()).all() or (c <= 0).any().any():
        raise ValueError("Incomplete or invalid price set")
    if not np.isfinite(o.to_numpy()).all() or (o <= 0).any().any():
        raise ValueError("Invalid execution prices")
    signal_closes = c.copy()
    usd_symbols = [k for k, u in units.items() if u["currency"] == "USD"]
    if usd_symbols:
        fx = yf.Ticker("GBPUSD=X").history(
            period="2y" if started is None else "1mo", interval="1h",
            auto_adjust=False, prepost=False, raise_errors=True)
        if fx.empty or fx.index.tz is None:
            raise ValueError("Missing GBP/USD conversion data")
        fx.index = fx.index.tz_convert("UTC")
        fx = fx.loc[fx.index + pd.Timedelta(minutes=65) <= now]
        common = c.index.intersection(fx.dropna(subset=["Open", "Close"]).index)
        o, c, fx = o.loc[common].copy(), c.loc[common].copy(), fx.loc[common]
        if c.empty or (fx[["Open", "Close"]] <= 0).any().any():
            raise ValueError("No aligned GBP/USD prices")
        for k in usd_symbols:
            o[k] = o[k] / fx["Open"]
            c[k] = c[k] / fx["Close"]
    latest = c.index[-1].tz_convert("Europe/London")
    local_now = now.tz_convert("Europe/London")
    if latest.date() != local_now.date():
        raise ValueError("No completed bars for today's London session; holiday or stale feed")
    if local_now.hour < 17 and now - c.index[-1] > pd.Timedelta(hours=3):
        raise ValueError("Intraday feed is stale")
    return o, c, units, signal_closes.loc[:c.index[-1]]

def desired_allocation(prices):
    # Original target_weights lags by one row. Appending a dummy row exposes
    # the decision from the last real close without using a future price.
    dummy = prices.iloc[[-1]].copy()
    dummy.index = pd.DatetimeIndex([prices.index[-1] + pd.Timedelta(hours=1)])
    w = target_weights(pd.concat([prices, dummy]), DEFAULT_PARAMS).iloc[-1]
    _, _, _, _, passed = validate_frozen(prices, DEFAULT_PARAMS, 8.0)
    return ({k: float(v) for k, v in w.items() if v > 0} if passed else {}), bool(passed)

def same_target(a, b):
    return set(a) == set(b) and all(abs(a[k] - b[k]) < 1e-10 for k in a)

def value(account, prices):
    return account["cash"] + sum(q * float(prices[k]) for k, q in account["holdings"].items())

def rebalance(account, target, prices, market_time, recorded_at, book, fx_symbols=()):
    rates = {k: COST + (0.0015 if k in fx_symbols else 0.0) for k in prices.index}
    equity = value(account, prices)
    old = account["holdings"].copy()
    # Scale requested buys to available cash including costs; never borrow.
    wanted = {k: equity * w / float(prices[k]) for k, w in target.items()}
    trades = []
    for k in sorted(set(old) | set(wanted)):
        delta = wanted.get(k, 0.0) - old.get(k, 0.0)
        if delta < -1e-10:
            qty = -delta
            notional = qty * float(prices[k])
            fee = notional * rates[k]
            account["cash"] += notional - fee
            account["holdings"][k] = account["holdings"].get(k, 0.0) - qty
            trades.append((k, -qty, notional, fee))
    buys = {k: max(0.0, wanted[k] - account["holdings"].get(k, 0.0)) for k in wanted}
    total = sum(q * float(prices[k]) * (1 + rates[k]) for k, q in buys.items())
    scale = min(1.0, account["cash"] / total) if total else 1.0
    for k, q in buys.items():
        q *= scale
        if q > 1e-10:
            notional = q * float(prices[k])
            fee = notional * rates[k]
            account["cash"] -= notional + fee
            account["holdings"][k] = account["holdings"].get(k, 0.0) + q
            trades.append((k, q, notional, fee))
    if account["cash"] < -1e-7:
        raise ValueError("Ledger would borrow cash")
    account["cash"] = max(0.0, account["cash"])
    account["holdings"] = {k: q for k, q in account["holdings"].items() if q > 1e-10}
    account["costs"] += sum(t[3] for t in trades)
    return [dict(book=book, market_time=str(market_time), recorded_at=str(recorded_at),
                 ticker=k, quantity=q, reference_price_gbp=float(prices[k]),
                 notional_gbp=n, cost_gbp=f, fx_cost_gbp=n*(0.0015 if k in fx_symbols else 0.0), status="SIMULATED_FILL")
            for k, q, n, f in trades]

def eligible_open(opens, pending):
    # An order must have been recorded BEFORE its simulated execution bar.
    after = opens.loc[opens.index > utc(pending["recorded_at"])]
    return None if after.empty else after.index[0]

def run(now=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else utc(now)
    local = now.tz_convert("Europe/London")
    if local.weekday() >= 5 or local.hour < 9 or local.hour > 19:
        dump(ROOT / "health.json", {"status": "OUTSIDE_SESSION", "checked_at": str(now)})
        return
    state_path = ROOT / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    opens, closes, units, signal_closes = get_market(now, state["started_at"] if state else None)
    latest = closes.index[-1]
    if state and state["quote_scales"] != units:
        raise ValueError("Quote currency changed; manual reconciliation required")
    if state:
        archive = pd.read_csv(ROOT / "signal_prices.csv", index_col=0, parse_dates=True)
        archive.index = pd.to_datetime(archive.index, utc=True)
        overlap = archive.index.intersection(signal_closes.index)
        if len(overlap):
            difference = (signal_closes.loc[overlap] / archive.loc[overlap] - 1).abs()
            if difference.max().max() > 0.02:
                raise ValueError("Historical price revision >2%; preserve ledger pending review")
        fresh = signal_closes.loc[signal_closes.index > archive.index[-1]]
        prices = pd.concat([archive, fresh])
        if latest <= utc(state["last_bar"]):
            dump(ROOT / "health.json", {"status": "NO_NEW_BAR", "checked_at": str(now),
                                        "last_bar": state["last_bar"]})
            return
    else:
        prices = signal_closes
        if len(prices) < 1000:
            raise ValueError("Insufficient V8 warmup/validation history")
        state = dict(schema=1, started_at=str(now), initial_capital=CAPITAL,
                     strategy=dict(cash=CAPITAL, holdings={}, costs=0.0),
                     benchmark=dict(cash=CAPITAL, holdings={}, costs=0.0),
                     pending=None, benchmark_pending={"recorded_at": str(now)},
                     last_target=None, last_bar=str(latest), quote_scales=units)
    fills = []
    for book, key in (("strategy", "pending"), ("benchmark", "benchmark_pending")):
        pending = state[key]
        if pending:
            execution = eligible_open(opens, pending)
            if execution is not None:
                # A long outage must not invent a complete execution history.
                if execution - utc(pending["recorded_at"]) > pd.Timedelta(days=7):
                    raise ValueError("Pending order too old; review before resuming")
                target = pending["target"] if book == "strategy" else {"VWRP.L": 1.0}
                fills.extend(rebalance(state[book], target, opens.loc[execution],
                                       execution, now, book,
                                       [k for k, u in units.items() if isinstance(u, dict) and u['currency'] == 'USD']))
                state[key] = None
    target, passed = desired_allocation(prices)
    decisions = []
    if state["last_target"] is None or not same_target(target, state["last_target"]):
        state["pending"] = {"recorded_at": str(now), "signal_bar": str(latest), "target": target}
        state["last_target"] = target
        decisions.append(dict(recorded_at=str(now), signal_bar=str(latest),
                              target=json.dumps(target, sort_keys=True),
                              gate_passed=passed, status="PENDING_NEXT_OBSERVED_OPEN"))
    strategy = value(state["strategy"], closes.loc[latest])
    benchmark = value(state["benchmark"], closes.loc[latest])
    row = dict(recorded_at=str(now), market_time=str(latest),
               london_date=str(latest.tz_convert("Europe/London").date()),
               equity_gbp=strategy, benchmark_gbp=benchmark,
               pnl_gbp=strategy-CAPITAL, return_pct=100*(strategy/CAPITAL-1),
               benchmark_return_pct=100*(benchmark/CAPITAL-1),
               excess_percentage_points=100*(strategy-benchmark)/CAPITAL,
               cash_gbp=state["strategy"]["cash"], costs_gbp=state["strategy"]["costs"],
               holdings=json.dumps(state["strategy"]["holdings"], sort_keys=True))
    state["last_bar"] = str(latest)
    state["checked_at"] = str(now)
    state["gate_passed"] = passed
    # Repository commit is the durable transaction boundary. A failed run must
    # not commit partial ledger changes (workflow enforces this).
    ROOT.mkdir(exist_ok=True)
    prices.to_csv(ROOT / "signal_prices.csv")
    append(ROOT / "fills.csv", fills)
    append(ROOT / "decisions.csv", decisions)
    append(ROOT / "valuations.csv", [row])
    dump(state_path, state)
    history = pd.read_csv(ROOT / "valuations.csv")
    daily = history.groupby("london_date", sort=True).tail(1).copy()
    daily["day_pnl_gbp"] = daily["equity_gbp"].diff()
    daily["day_return_pct"] = daily["equity_gbp"].pct_change() * 100
    daily["benchmark_day_return_pct"] = daily["benchmark_gbp"].pct_change() * 100
    daily.loc[daily.index[0], "day_pnl_gbp"] = daily.iloc[0]["equity_gbp"] - CAPITAL
    daily.loc[daily.index[0], "day_return_pct"] = (daily.iloc[0]["equity_gbp"]/CAPITAL-1)*100
    daily.loc[daily.index[0], "benchmark_day_return_pct"] = (daily.iloc[0]["benchmark_gbp"]/CAPITAL-1)*100
    daily.to_csv(ROOT / "daily.csv", index=False)
    last = daily.iloc[-1]
    report = (
        "# V8 forward paper test\n\n"
        f"Observed at {now}; latest completed price bar starts {latest}.\n\n"
        f"Started {state['started_at']} with £1,000. Simulated fills only.\n\n"
        "| Measure | Value |\n|---|---:|\n"
        f"| Portfolio | £{strategy:.2f} |\n"
        f"| Today, provisional until session data complete | £{last['day_pnl_gbp']:.2f} ({last['day_return_pct']:.2f}%) |\n"
        f"| Since start | £{strategy-CAPITAL:.2f} ({row['return_pct']:.2f}%) |\n"
        f"| VWRP since start | {row['benchmark_return_pct']:.2f}% |\n"
        f"| Difference | {row['excess_percentage_points']:.2f} percentage points |\n"
        f"| Simulated strategy costs | £{state['strategy']['costs']:.2f} |\n\n"
        f"Holdings (units): {json.dumps(state['strategy']['holdings'])}.\n\n"
        f"Cash: £{state['strategy']['cash']:.2f}. Pending target: {json.dumps(state['pending'])}.\n\n"
        "The legacy target-only log is excluded. Both books start together and pay "
        "8 bps per side plus 15 bps FX on USD trades; cash earns zero. Yahoo prices are indicative, not broker quotes.\n"
    )
    (ROOT / "REPORT.md").write_text(report)
    dump(ROOT / "health.json", {"status": "OK", "checked_at": str(now), "last_bar": str(latest),
                               "gate_passed": passed, "fills_this_run": len(fills)})
    print(report)

if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        dump(ROOT / "health.json", {"status": "ERROR", "checked_at": str(pd.Timestamp.now(tz="UTC")),
                                   "error": str(error)})
        traceback.print_exc()
        raise
