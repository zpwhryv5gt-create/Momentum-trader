"""Forward-only daily valuation / monthly decision runner. NO BROKER INTERFACE."""
import copy
import hashlib
import json
import os
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import exchange_calendars as xcals
import yfinance as yf
from v9_core import CONFIG, CONFIG_HASH, SYMBOLS, BOOKS, quote_scale, signals, new_account, value, rebalance

ROOT = Path(os.environ.get('V9_PAPER_DIR', 'paper_v9'))
CAL = xcals.get_calendar('XLON')


def utc(x):
    return pd.Timestamp(x).tz_convert('UTC')


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def append(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode='a', header=not path.exists(), index=False)


def calendar_date(ts):
    return pd.Timestamp(ts).tz_localize(None).normalize()


def latest_completed(now):
    local_date = now.tz_convert('Europe/London').tz_localize(None).normalize()
    sessions = CAL.sessions_in_range(local_date - pd.Timedelta(days=14), local_date)
    done = [calendar_date(s) for s in sessions if CAL.session_close(s) + pd.Timedelta(minutes=90) <= now]
    if not done:
        raise ValueError('No completed exchange session')
    return done[-1]


def recover_historical_day(ticker, symbol, day, frame):
    """Use a complete observed hourly session only when adjustment is unchanged.

    This is an indicative historical signal input, never an execution price.
    Preserve its source explicitly; do not interpolate a daily price.
    """
    before, after = frame.loc[frame.index < day], frame.loc[frame.index > day]
    if before.empty or after.empty:
        return None
    left = float(before.iloc[-1]['Adj Close'] / before.iloc[-1]['Close'])
    right = float(after.iloc[0]['Adj Close'] / after.iloc[0]['Close'])
    if not np.isfinite([left, right]).all() or abs(left / right - 1) > 1e-6:
        return None
    bars = ticker.history(start=str(day.date()), end=str((day + pd.Timedelta(days=1)).date()),
                          interval='1h', auto_adjust=False, actions=True, prepost=False, raise_errors=True)
    if bars.empty or bars.index.tz is None or bars.index.has_duplicates:
        return None
    bars.index = bars.index.tz_convert('UTC')
    expected = pd.date_range(CAL.session_open(day), CAL.session_close(day), freq='1h', inclusive='left')
    if not expected.isin(bars.index).all():
        return None
    bars = bars.loc[expected]
    if not np.isfinite(bars[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy()).all():
        return None
    if (bars[['Open', 'High', 'Low', 'Close']] <= 0).any().any() or bars['Volume'].sum() <= 0:
        return None
    if any(c not in bars or bars[c].fillna(0).ne(0).any() for c in ['Dividends', 'Stock Splits']):
        return None
    row = {'Open': float(bars['Open'].iloc[0]), 'High': float(bars['High'].max()),
           'Low': float(bars['Low'].min()), 'Close': float(bars['Close'].iloc[-1]),
           'Adj Close': float(bars['Close'].iloc[-1]) * right,
           'Volume': float(bars['Volume'].sum()), 'Dividends': 0., 'Stock Splits': 0.,
           'V9 Source': 'complete_observed_hourly_session'}
    print(f'Recovered historical signal day from {len(bars)} observed hourly bars: {symbol} {day.date()}')
    return pd.DataFrame([row], index=pd.DatetimeIndex([day]))


def get_market(now, latest):
    frames, units = {}, {}
    for symbol in SYMBOLS:
        ticker = yf.Ticker(symbol)
        frame = ticker.history(period='2y', interval='1d', auto_adjust=False,
                               actions=True, repair=False, raise_errors=True)
        if frame.empty or frame.index.tz is None:
            raise ValueError(f'Missing timestamped daily data: {symbol}')
        currency = ticker.history_metadata.get('currency')
        scale = quote_scale(currency)
        units[symbol] = {'currency': currency, 'scale': scale}
        frame.index = frame.index.tz_convert('Europe/London').tz_localize(None).normalize()
        if frame.index.has_duplicates:
            raise ValueError(f'Duplicate daily bars: {symbol}')
        frame = frame.loc[:latest].copy()
        frame['V9 Source'] = 'Yahoo_daily'
        # Yahoo's long-range response occasionally omits a valid trading day.
        # Retry required missing sessions as narrow daily requests; accept only
        # actual returned bars, never an interpolated or repeated price.
        recent = CAL.sessions_in_range(latest - pd.Timedelta(days=120), latest)[-64:].tz_localize(None)
        months = pd.period_range(latest.to_period('M') - 13, latest.to_period('M'), freq='M')
        month_ends = [calendar_date(CAL.sessions_in_range(p.start_time, p.end_time.normalize())[-1]) for p in months]
        required = recent.union(pd.DatetimeIndex([d for d in month_ends if d <= latest]))
        for day in required.difference(frame.index):
            try:
                retry = ticker.history(start=str(day.date()), end=str((day + pd.Timedelta(days=1)).date()),
                                       interval='1d', auto_adjust=False, actions=True, repair=False, raise_errors=True)
            except yf.exceptions.YFPricesMissingError:
                retry = pd.DataFrame()
            if not retry.empty and retry.index.tz is not None:
                retry.index = retry.index.tz_convert('Europe/London').tz_localize(None).normalize()
                if day in retry.index:
                    retry['V9 Source'] = 'Yahoo_daily_narrow_request'
                    frame = pd.concat([frame, retry.loc[[day]]]).sort_index()
                    print(f'Recovered observed daily bar with narrow request: {symbol} {day.date()}')
            if day not in frame.index:
                recovered = recover_historical_day(ticker, symbol, day, frame)
                if recovered is not None:
                    frame = pd.concat([frame, recovered]).sort_index()
                else:
                    print(f'Missing required daily bar: {symbol} {day.date()}')
        if frame.empty or frame.index[-1] != latest:
            raise ValueError(f'Stale feed: {symbol}; expected {latest.date()}')
        for col in ['Open', 'Close', 'Adj Close', 'Dividends']:
            if col not in frame:
                raise ValueError(f'Missing {col}: {symbol}')
            frame[col] *= scale
        for col in ['Open', 'Close', 'Adj Close']:
            if not np.isfinite(frame[col]).all() or (frame[col] <= 0).any():
                raise ValueError(f'Invalid {col}: {symbol}')
        if not np.isfinite(frame['Dividends']).all() or (frame['Dividends'] < 0).any():
            raise ValueError(f'Invalid distributions: {symbol}')
        if not np.isfinite(frame['Stock Splits']).all() or (frame['Stock Splits'] < 0).any():
            raise ValueError(f'Invalid splits: {symbol}')
        if frame.loc[latest, 'Volume'] <= 0:
            raise ValueError(f'No trading volume: {symbol}')
        frames[symbol] = frame
    tri = pd.DataFrame({k: f['Adj Close'] for k, f in frames.items()}).sort_index()
    # Vendors can differ on old/non-session rows. Keep observed common XLON
    # sessions only; never fill a price. Required month-ends and the full
    # 64-session risk window are checked explicitly, so dropping an unrelated
    # old row cannot silently shorten a signal horizon.
    expected = CAL.sessions_in_range(max(f.index[0] for f in frames.values()), latest).tz_localize(None)
    tri = tri.reindex(expected)
    missing = tri.index[tri.isna().any(axis=1)]
    if len(missing):
        print('Incomplete historical sessions (not filled):', [str(d.date()) for d in missing])
    tri = tri.dropna()
    if len(expected) < 64 or not expected[-64:].equals(tri.index[-64:]):
        raise ValueError('Missing session in current 64-session risk window')
    return frames, tri, units


def corporate_actions(account, frames, session, payments, book, now):
    events = []
    for symbol in SYMBOLS:
        row = frames[symbol].loc[session]
        split = float(row['Stock Splits'])
        if split and split != 1 and account['holdings'].get(symbol, 0):
            # Split-adjusted vendor historical prices require an explicit audit.
            raise ValueError(f'{symbol}: held split requires reconciliation; ledger preserved')
        dividend = float(row['Dividends'])
        qty = account['holdings'].get(symbol, 0)
        key = f'{symbol}:{session.date()}'
        if dividend and qty:
            if key in account['receivables']:
                raise ValueError('Duplicate dividend entitlement')
            account['receivables'][key] = {'amount': qty * dividend, 'per_unit': dividend,
                                           'quantity': qty, 'ex_date': str(session.date())}
            events.append({'id': f'{book}:{key}:accrue', 'book': book, 'event': 'DIVIDEND_RECEIVABLE',
                           'session': str(session.date()), 'recorded_at': str(now), 'amount_gbp': qty * dividend})
    for key, r in list(account['receivables'].items()):
        pay = payments.get(key)
        if pay and pd.Timestamp(pay['pay_date']) <= session:
            if not pay.get('source_url') or abs(float(pay['per_unit_gbp']) - r['per_unit']) > 1e-7:
                raise ValueError(f'Unverified or mismatched payment: {key}')
            if pd.Timestamp(pay['pay_date']) < pd.Timestamp(r['ex_date']):
                raise ValueError('Payment precedes ex date')
            account['cash'] += r['amount']
            del account['receivables'][key]
            events.append({'id': f'{book}:{key}:pay', 'book': book, 'event': 'DIVIDEND_CASH_RECEIPT',
                           'session': str(session.date()), 'recorded_at': str(now), 'amount_gbp': r['amount']})
    return events


def process(state, frames, tri, units, now, latest, payments):
    """Return a complete proposed transaction; caller persists only after success."""
    state = copy.deepcopy(state)
    fills, decisions, events, valuations = [], [], [], []
    if state is None:
        state = {'schema': 1, 'config': CONFIG, 'config_hash': CONFIG_HASH,
                 'started_at': str(now), 'start_session': str(latest.date()),
                 'last_session': str(latest.date()), 'units': units,
                 'books': {b: new_account() for b in BOOKS}, 'pending': None,
                 'last_signal_month': None, 'first_execution_session': None}
        # Opening balance only, never a retrospective trade.
        sessions = [latest]
    else:
        if state['config_hash'] != CONFIG_HASH:
            raise ValueError('Frozen configuration changed; explicit new experiment required')
        if units != state['units']:
            raise ValueError('Quote units changed; reconcile before resuming')
        last = pd.Timestamp(state['last_session'])
        if latest <= last:
            return state, [], [], [], []
        if latest - last > pd.Timedelta(days=7):
            raise ValueError('Outage over seven days: reconcile before resuming')
        sessions = list(CAL.sessions_in_range(last + pd.Timedelta(days=1), latest).tz_localize(None))
    for session in sessions:
        if any(session not in f.index for f in frames.values()):
            raise ValueError(f'Missing accounting session: {session}')
        prices = pd.Series({k: float(frames[k].loc[session, 'Close']) for k in SYMBOLS})
        if str(session.date()) != state['start_session']:
            for book, account in state['books'].items():
                events.extend(corporate_actions(account, frames, session, payments, book, now))
        pending = state['pending']
        if pending and pd.Timestamp(pending['execution_session']) == session:
            if any('V9 Source' in f and f.loc[session, 'V9 Source'] == 'complete_observed_hourly_session' for f in frames.values()):
                raise ValueError('Recovered historical signal prices cannot be execution prices')
            if utc(pending['recorded_at']) >= CAL.session_close(session):
                raise ValueError('Decision was not recorded before execution')
            for book, target in pending['targets'].items():
                fills.extend(rebalance(state['books'][book], target, prices,
                                       pending['id'], session, now, book))
            if state['first_execution_session'] is None:
                state['first_execution_session'] = str(session.date())
            state['pending'] = None
        elif pending and pd.Timestamp(pending['execution_session']) < session:
            raise ValueError('Missed execution session requires review')
        vals = {b: value(a, prices) for b, a in state['books'].items()}
        row = {'session': str(session.date()), 'market_time': str(CAL.session_close(session)),
               'recorded_at': str(now), **{b + '_gbp': v for b, v in vals.items()},
               'cash_gbp': state['books']['v9']['cash'],
               'receivables_gbp': sum(r['amount'] for r in state['books']['v9']['receivables'].values()),
               'costs_gbp': state['books']['v9']['costs'],
               'holdings': json.dumps(state['books']['v9']['holdings'], sort_keys=True)}
        valuations.append(row)
        state['last_session'] = str(session.date())
    # Only today's observed data can create a new decision, never outage backfills.
    next_session = calendar_date(CAL.next_session(latest))
    is_month_end = next_session.month != latest.month
    signal_month = str(latest.to_period('M'))
    if is_month_end and state['last_signal_month'] != signal_month:
        if now >= CAL.session_close(next_session):
            raise ValueError('Month-end execution window missed; no backdated decision allowed')
        targets, details = signals(tri, latest, CAL)
        if state['first_execution_session'] is None:
            targets['vwrp'] = {'VWRP.L': 1.0}
        if state['pending']:
            raise ValueError('Unfilled previous decision')
        signal_json = tri.loc[:latest].to_csv()
        pending = {'id': f'{CONFIG_HASH[:12]}:{signal_month}', 'recorded_at': str(now),
                   'signal_session': str(latest.date()), 'execution_session': str(next_session.date()),
                   'targets': targets, 'details': details,
                   'data_sha256': hashlib.sha256(signal_json.encode()).hexdigest(),
                   'config_hash': CONFIG_HASH, 'status': 'PENDING_SIMULATED_NEXT_SESSION_CLOSE'}
        state['pending'] = pending
        state['last_signal_month'] = signal_month
        decisions.append(pending)
    state['checked_at'] = str(now)
    return state, fills, decisions, events, valuations


def write_report(root, state, now):
    history = pd.read_csv(root / 'daily.csv')
    latest = history.iloc[-1]
    previous = history.iloc[-2] if len(history) > 1 else None
    capital = CONFIG['capital_gbp']
    status = 'ACTIVE' if state['first_execution_session'] else 'WAITING_FOR_FIRST_MONTH_END_EXECUTION'
    lines = ['# V9 forward paper test', '', f'Status: **{status}**.', '',
             f"Observed {now}; valuation at London session close {latest['market_time']}.", '',
             f"Created {state['started_at']}. Each book starts with £1,000 simulated cash.", '',
             '| Book | Value | Since start | Latest session | Costs |', '|---|---:|---:|---:|---:|']
    for book in BOOKS:
        nav = float(latest[book + '_gbp'])
        prev = float(previous[book + '_gbp']) if previous is not None else capital
        lines.append(f'| {book} | £{nav:.2f} | {100*(nav/capital-1):.2f}% | {100*(nav/prev-1):.2f}% | £{state["books"][book]["costs"]:.2f} |')
    account = state['books']['v9']
    lines += ['', f"V9 holdings (whole units): {json.dumps(account['holdings'])}.",
              f"Cash £{account['cash']:.2f}; dividend receivables £{sum(r['amount'] for r in account['receivables'].values()):.2f}.", '',
              'Pending order: ' + (json.dumps(state['pending']['targets'], sort_keys=True) +
                                   ' for ' + state['pending']['execution_session'] if state['pending'] else 'none'), '',
              'Monthly 1/3/12-month trend; policy budgets 50% VWRP / 25% IGLT / 25% SGLN; estimated 12% volatility ceiling. No leverage.', '',
              'All fills are SIMULATED at a pre-recorded next-session daily close, with 8 bps per side; Yahoo prices are indicative, not broker quotes. Whole units can leave substantial cash at £1,000. Cash interest is zero.', '',
              'Benchmarks start trading at the same first execution as V9. vwrp is buy-and-hold; passive_mix is monthly 50/25/25. Other books isolate trend and risk scaling.', '',
              'Dividend entitlements count in NAV but are not spendable until a verified payment date is supplied. Missing payment dates generate a warning, not invented cash.', '',
              'Check the latest GitHub Actions run and health.json before treating this report as current. V8 and V9 have different inception dates; compare only matching dates, not raw since-start returns.', '',
              f'Frozen configuration SHA256: {CONFIG_HASH}', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))


def run(now=None, root=None, market_loader=get_market):
    now = pd.Timestamp.now(tz='UTC') if now is None else utc(now)
    root = ROOT if root is None else Path(root)
    latest = latest_completed(now)
    state_path = root / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if state and state['config_hash'] != CONFIG_HASH:
        raise ValueError('Frozen configuration changed')
    if state and pd.Timestamp(state['last_session']) >= latest:
        dump(root / 'health.json', {'status': 'NO_NEW_COMPLETED_SESSION', 'checked_at': str(now),
                                   'last_session': state['last_session']})
        return
    frames, tri, units = market_loader(now, latest)
    payments = json.loads(Path('v9_dividend_payments.json').read_text())
    state, fills, decisions, events, rows = process(state, frames, tri, units, now, latest, payments)
    root.mkdir(parents=True, exist_ok=True)
    recovered_days = []
    for symbol, frame in frames.items():
        if 'V9 Source' in frame:
            recovered = frame.loc[frame['V9 Source'] == 'complete_observed_hourly_session']
            if not recovered.empty:
                (root / 'data_recoveries').mkdir(exist_ok=True)
                for day, candle in recovered.iterrows():
                    candle.to_frame().T.to_csv(root / 'data_recoveries' / f'{day.date()}-{symbol}.csv')
                    recovered_days.append(f'{symbol}:{day.date()}')
    for decision in decisions:
        month = decision['signal_session'][:7]
        (root / 'snapshots').mkdir(exist_ok=True)
        tri.loc[:decision['signal_session']].to_csv(root / 'snapshots' / f'{month}-total-return.csv')
        dump(root / 'snapshots' / f'{month}-decision.json', decision)
        for symbol, frame in frames.items():
            frame.loc[:decision['signal_session']].to_csv(root / 'snapshots' / f'{month}-{symbol}.csv')
    append(root / 'fills.csv', fills)
    append(root / 'events.csv', events)
    append(root / 'decisions.csv', [{'id': d['id'], 'recorded_at': d['recorded_at'],
                                   'signal_session': d['signal_session'], 'execution_session': d['execution_session'],
                                   'targets': json.dumps(d['targets'], sort_keys=True),
                                   'config_hash': CONFIG_HASH, 'data_sha256': d['data_sha256']} for d in decisions])
    append(root / 'daily.csv', rows)
    dump(state_path, state)
    warnings = [f'{book}: payment date needed for {key}' for book, a in state['books'].items()
                for key in a['receivables'] if key not in payments]
    status = 'OK_DIVIDEND_PAYMENT_PENDING' if warnings else ('OK' if state['first_execution_session'] else 'WAITING_FOR_FIRST_MONTH_END')
    dump(root / 'health.json', {'status': status, 'checked_at': str(now), 'last_session': state['last_session'],
                               'fills_this_run': len(fills), 'warnings': warnings, 'recovered_historical_signal_days': recovered_days,
                               'config_hash': CONFIG_HASH, 'data_source': 'Yahoo daily; unadjusted execution / adjusted signals'})
    write_report(root, state, now)
    print((root / 'REPORT.md').read_text())


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        dump(ROOT / 'health.json', {'status': 'ERROR', 'checked_at': str(pd.Timestamp.now(tz='UTC')), 'error': str(error)})
        traceback.print_exc()
        raise
