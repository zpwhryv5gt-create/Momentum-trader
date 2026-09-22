"""Frozen V9 monthly trend hypothesis; pure functions, no broker or optimiser."""
import hashlib
import json
import numpy as np
import pandas as pd

CONFIG = {
    'version': 'v9-paper-1',
    'budgets': {'VWRP.L': 0.50, 'IGLT.L': 0.25, 'SGLN.L': 0.25},
    'horizons_months': [1, 3, 12],
    'vol_windows': [21, 63], 'vol_ceiling': 0.12,
    'capital_gbp': 1000.0, 'cost_per_side': 0.0008,
    'execution': 'pre-recorded_next_XLON_session_close',
    'units': 'whole_units_round_down', 'cash_interest': 0.0,
    'dividends': 'receivable_on_ex_date_cash_only_on_verified_pay_date',
}
SYMBOLS = list(CONFIG['budgets'])
CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
BOOKS = ['v9', 'vwrp', 'passive_mix', 'passive_scaled', 'trend_unscaled', 'vwrp_scaled', 'cash']


def quote_scale(currency):
    if currency == 'GBP':
        return 1.0
    if currency in ('GBp', 'GBX'):
        return 0.01
    raise ValueError(f'Unverified sterling quote currency: {currency!r}')


def risk_scale(tri, weights):
    if not any(weights.values()):
        return 1.0, 0.0
    returns = tri[SYMBOLS].pct_change(fill_method=None).iloc[-63:]
    if len(returns) != 63 or not np.isfinite(returns.to_numpy()).all():
        raise ValueError('63 complete daily returns required')
    x = returns.mul(pd.Series(weights).reindex(SYMBOLS, fill_value=0)).sum(axis=1)
    vol = max(float(x.tail(n).std(ddof=1)) for n in CONFIG['vol_windows']) * np.sqrt(252)
    if not np.isfinite(vol) or vol <= 0:
        raise ValueError('Invalid or zero estimated volatility')
    return min(1.0, CONFIG['vol_ceiling'] / vol), vol


def signals(tri, session, calendar):
    """Only completed data at or before session may affect targets."""
    session = pd.Timestamp(session).normalize().tz_localize(None)
    tri = tri.loc[:session, SYMBOLS]
    if tri.empty or tri.index[-1] != session:
        raise ValueError('Missing signal-session prices')
    if not np.isfinite(tri.to_numpy()).all() or (tri <= 0).any().any():
        raise ValueError('Invalid total-return history')
    end_period = session.to_period('M')
    month_ends = []
    for period in pd.period_range(end_period - 12, end_period, freq='M'):
        sessions = calendar.sessions_in_range(period.start_time, period.end_time.normalize())
        if sessions.empty:
            raise ValueError('Missing exchange calendar')
        month_ends.append(sessions[-1].tz_localize(None))
    if month_ends[-1] != session or not pd.Index(month_ends).isin(tri.index).all():
        raise ValueError('13 complete exchange month-ends required')
    expected = calendar.sessions_in_range(tri.index[0], session)[-64:].tz_localize(None)
    if len(expected) != 64 or not expected.equals(tri.index[-64:]):
        raise ValueError('Missing daily session in volatility window')
    monthly = tri.loc[month_ends]
    details, q = {}, {}
    for symbol, budget in CONFIG['budgets'].items():
        rs = {str(h): float(monthly[symbol].iloc[-1] / monthly[symbol].iloc[-1-h] - 1)
              for h in CONFIG['horizons_months']}
        votes = sum(v > 0 for v in rs.values())
        q[symbol] = budget * votes / 3
        details[symbol] = {'returns': rs, 'positive_votes': votes, 'preliminary_weight': q[symbol]}
    scale, vol = risk_scale(tri, q)
    pscale, pvol = risk_scale(tri, CONFIG['budgets'])
    escale, evol = risk_scale(tri, {'VWRP.L': 1.0})
    targets = {
        'v9': {k: v * scale for k, v in q.items()},
        'passive_mix': dict(CONFIG['budgets']),
        'passive_scaled': {k: v * pscale for k, v in CONFIG['budgets'].items()},
        'trend_unscaled': q,
        'vwrp_scaled': {'VWRP.L': escale},
    }
    return targets, {'assets': details, 'scale': scale, 'forecast_vol': vol,
                     'passive_forecast_vol': pvol, 'equity_forecast_vol': evol}


def new_account():
    return {'cash': CONFIG['capital_gbp'], 'holdings': {}, 'costs': 0.0, 'receivables': {}}


def value(account, prices):
    return (account['cash'] + sum(q * float(prices[k]) for k, q in account['holdings'].items())
            + sum(r['amount'] for r in account['receivables'].values()))


def rebalance(account, target, prices, decision_id, session, recorded_at, book):
    if any(k not in SYMBOLS or not np.isfinite(w) or w < 0 for k, w in target.items()) or sum(target.values()) > 1 + 1e-10:
        raise ValueError('Invalid long-only cash-funded allocation')
    if any(not np.isfinite(prices[k]) or prices[k] <= 0 for k in SYMBOLS):
        raise ValueError('Invalid execution prices')
    nav = value(account, prices)
    wanted = {k: int(np.floor(nav * w / prices[k])) for k, w in target.items()}
    fills = []
    def trade(symbol, qty):
        notional = abs(qty) * float(prices[symbol])
        fee = notional * CONFIG['cost_per_side']
        account['cash'] -= qty * float(prices[symbol]) + fee
        account['holdings'][symbol] = account['holdings'].get(symbol, 0) + qty
        account['costs'] += fee
        fills.append({'id': f'{decision_id}:{book}:{symbol}', 'decision_id': decision_id,
                      'book': book, 'session': str(session.date()), 'recorded_at': str(recorded_at),
                      'ticker': symbol, 'quantity': qty, 'reference_price_gbp': float(prices[symbol]),
                      'notional_gbp': notional, 'cost_gbp': fee, 'status': 'SIMULATED_FILL'})
    for k in SYMBOLS:
        delta = wanted.get(k, 0) - account['holdings'].get(k, 0)
        if delta < 0:
            trade(k, delta)
    buys = {k: max(0, wanted.get(k, 0) - account['holdings'].get(k, 0)) for k in SYMBOLS}
    need = sum(q * prices[k] * (1 + CONFIG['cost_per_side']) for k, q in buys.items())
    scale = min(1.0, max(0.0, account['cash']) / need) if need else 1.0
    for k in SYMBOLS:
        q = int(np.floor(buys[k] * scale))
        if q:
            trade(k, q)
    account['holdings'] = {k: q for k, q in account['holdings'].items() if q > 0}
    if account['cash'] < -1e-7 or any(q < 0 for q in account['holdings'].values()):
        raise ValueError('Negative cash/position')
    account['cash'] = max(0.0, account['cash'])
    return fills
