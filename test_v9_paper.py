import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
from v9_core import CONFIG, CONFIG_HASH, SYMBOLS, signals, quote_scale, new_account, value, rebalance
from v9_paper import CAL, process, corporate_actions, run, latest_completed, get_market


def fixture(end='2026-10-02'):
    idx = CAL.sessions_in_range('2024-10-01', end).tz_localize(None)
    rng = np.random.default_rng(911)
    data = np.exp(np.cumsum(rng.normal(0.002, 0.008, (len(idx), 3)), axis=0)) * [100, 10, 40]
    tri = pd.DataFrame(data, index=idx, columns=SYMBOLS)
    frames = {k: pd.DataFrame({'Open': tri[k], 'Close': tri[k], 'Adj Close': tri[k],
                             'Dividends': 0.0, 'Stock Splits': 0.0, 'Volume': 1000}, index=idx) for k in SYMBOLS}
    units = {k: {'currency': 'GBP', 'scale': 1.0} for k in SYMBOLS}
    return frames, tri, units


class V9Tests(unittest.TestCase):
    def test_market_alignment_old_gap_allowed_recent_gap_rejected(self):
        frames, _, _ = fixture(end='2026-09-22')
        class FakeTicker:
            def __init__(self, symbol):
                self.symbol = symbol
                self.history_metadata = {'currency': 'GBP'}
            def history(self, **kwargs):
                f = frames[self.symbol].copy()
                if 'start' in kwargs:
                    f = f.loc[kwargs['start']:].loc[:pd.Timestamp(kwargs['end']) - pd.Timedelta(days=1)]
                f.index = f.index.tz_localize('Europe/London')
                return f
        frames['SGLN.L'] = frames['SGLN.L'].drop(pd.Timestamp('2024-10-02'))
        with patch('v9_paper.yf.Ticker', FakeTicker):
            _, tri, _ = get_market(pd.Timestamp('2026-09-22T19:00Z'), pd.Timestamp('2026-09-22'))
            self.assertNotIn(pd.Timestamp('2024-10-02'), tri.index)
            frames['SGLN.L'] = frames['SGLN.L'].drop(pd.Timestamp('2026-09-21'))
            with self.assertRaises(ValueError):
                get_market(pd.Timestamp('2026-09-22T19:00Z'), pd.Timestamp('2026-09-22'))

    def test_units_and_currency_rejection(self):
        self.assertEqual(quote_scale('GBp') * 10000, 100)
        self.assertEqual(quote_scale('GBP'), 1)
        with self.assertRaises(ValueError): quote_scale('USD')

    def test_signals_ignore_future(self):
        _, tri, _ = fixture()
        a = signals(tri, '2026-09-30', CAL)
        tri.loc['2026-10-01':] *= 100
        self.assertEqual(a, signals(tri, '2026-09-30', CAL))
        targets, details = a
        self.assertGreater(sum(targets['v9'].values()), 0)
        for k, w in targets['v9'].items(): self.assertLessEqual(w, CONFIG['budgets'][k])
        self.assertLessEqual(details['scale'], 1)

    def test_all_negative_and_ties_mean_cash(self):
        _, tri, _ = fixture()
        tri.iloc[:] = np.exp(-np.arange(len(tri))[:, None] * 0.001) * np.array([[100, 10, 40]])
        targets, _ = signals(tri, '2026-09-30', CAL)
        self.assertEqual(sum(targets['v9'].values()), 0)
        # Direct per-asset ties while other assets retain nonzero risk.
        tri['SGLN.L'] = 100
        targets, _ = signals(tri, '2026-09-30', CAL)
        self.assertEqual(targets['v9']['SGLN.L'], 0)

    def test_zero_volatility_is_exception(self):
        _, tri, _ = fixture()
        tri.iloc[:] = 100
        with self.assertRaises(ValueError): signals(tri, '2026-09-30', CAL)

    def test_missing_session_fails(self):
        _, tri, _ = fixture()
        tri = tri.drop(pd.Timestamp('2026-09-29'))
        with self.assertRaises(ValueError): signals(tri, '2026-09-30', CAL)

    def test_accounting_rounding_and_costs(self):
        a = new_account()
        p = pd.Series({'VWRP.L': 101., 'IGLT.L': 10., 'SGLN.L': 40.})
        fills = rebalance(a, {'VWRP.L': 1.}, p, 'a', pd.Timestamp('2026-10-01'), 'now', 'v9')
        self.assertGreaterEqual(a['cash'], 0)
        self.assertAlmostEqual(value(a, p), 1000 - a['costs'])
        self.assertEqual(a['holdings']['VWRP.L'], 9)
        fills += rebalance(a, {'IGLT.L': 0.5, 'SGLN.L': 0.5}, p, 'b', pd.Timestamp('2026-10-02'), 'now', 'v9')
        self.assertGreaterEqual(a['cash'], 0)
        self.assertAlmostEqual(value(a, p), 1000 - a['costs'])
        self.assertTrue(all(q == int(q) for q in a['holdings'].values()))
        self.assertAlmostEqual(a['costs'], sum(f['cost_gbp'] for f in fills))
        with self.assertRaises(ValueError): rebalance(a, {'VWRP.L': 1.1}, p, 'c', pd.Timestamp('2026-10-02'), 'now', 'v9')

    def test_month_end_then_next_close_and_no_duplicate(self):
        f, tri, u = fixture()
        state, fills, decisions, _, _ = process(None, f, tri, u, pd.Timestamp('2026-09-22T18:45Z'), pd.Timestamp('2026-09-22'), {})
        self.assertFalse(fills or decisions)
        # Advance within the allowed seven-day outage limit.
        state, *_ = process(state, f, tri, u, pd.Timestamp('2026-09-28T18:45Z'), pd.Timestamp('2026-09-28'), {})
        state, fills, decisions, _, _ = process(state, f, tri, u, pd.Timestamp('2026-09-30T18:45Z'), pd.Timestamp('2026-09-30'), {})
        self.assertFalse(fills)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(state['pending']['execution_session'], '2026-10-01')
        self.assertEqual(state['books']['vwrp']['cash'], 1000)
        state, fills, decisions, _, rows = process(state, f, tri, u, pd.Timestamp('2026-10-01T18:45Z'), pd.Timestamp('2026-10-01'), {})
        self.assertTrue(fills)
        self.assertFalse(decisions)
        for book in ('vwrp', 'passive_mix'):
            self.assertAlmostEqual(rows[-1][book+'_gbp'], 1000-state['books'][book]['costs'])
        # New allocation gets no return from before its simulated close fill.
        again, *changes = process(state, f, tri, u, pd.Timestamp('2026-10-01T19:45Z'), pd.Timestamp('2026-10-01'), {})
        self.assertEqual(state, again)
        self.assertFalse(any(changes))

    def test_dividend_entitlement_is_not_spendable_or_doubled(self):
        f, _, _ = fixture()
        a = new_account(); a['holdings'] = {'IGLT.L': 20}; a['cash'] = 800
        day = pd.Timestamp('2026-10-01')
        f['IGLT.L'].loc[day, 'Dividends'] = 0.1
        corporate_actions(a, f, day, {}, 'v9', 'now')
        self.assertEqual(a['cash'], 800)
        self.assertEqual(sum(r['amount'] for r in a['receivables'].values()), 2)
        pay = {'IGLT.L:2026-10-01': {'pay_date': '2026-10-02', 'per_unit_gbp': 0.1, 'source_url': 'https://issuer.example/payment'}}
        corporate_actions(a, f, pd.Timestamp('2026-10-02'), pay, 'v9', 'now')
        self.assertEqual(a['cash'], 802)
        self.assertFalse(a['receivables'])
        corporate_actions(a, f, pd.Timestamp('2026-10-02'), pay, 'v9', 'now')
        self.assertEqual(a['cash'], 802)

    def test_held_split_halts(self):
        f, _, _ = fixture(); a = new_account(); a['holdings'] = {'VWRP.L': 3}
        f['VWRP.L'].loc['2026-10-01', 'Stock Splits'] = 2
        with self.assertRaises(ValueError): corporate_actions(a, f, pd.Timestamp('2026-10-01'), {}, 'v9', 'now')

    def test_atomic_failure_and_duplicate_run(self):
        f, tri, u = fixture()
        with tempfile.TemporaryDirectory() as d:
            loader = lambda now, latest: (f, tri.loc[:latest], u)
            run(pd.Timestamp('2026-09-22T18:45Z'), d, loader)
            state = (Path(d)/'state.json').read_bytes()
            ledger = (Path(d)/'daily.csv').read_bytes()
            run(pd.Timestamp('2026-09-22T19:45Z'), d, loader)
            self.assertEqual(ledger, (Path(d)/'daily.csv').read_bytes())
            bad = copy.deepcopy(u); bad['VWRP.L']['scale'] = 0.01
            with self.assertRaises(ValueError):
                run(pd.Timestamp('2026-09-23T18:45Z'), d, lambda now, latest: (f, tri.loc[:latest], bad))
            self.assertEqual(state, (Path(d)/'state.json').read_bytes())
            self.assertEqual(ledger, (Path(d)/'daily.csv').read_bytes())

    def test_calendar_holiday_and_close_delay(self):
        self.assertEqual(str(latest_completed(pd.Timestamp('2026-12-25T19:00Z')).date()), '2026-12-24')
        self.assertEqual(str(latest_completed(pd.Timestamp('2026-09-30T16:00Z')).date()), '2026-09-29')
        self.assertEqual(str(latest_completed(pd.Timestamp('2026-09-30T18:00Z')).date()), '2026-09-30')

    def test_no_retroactive_start(self):
        f, tri, u = fixture()
        state, fills, decisions, _, _ = process(None, f, tri, u, pd.Timestamp('2026-10-02T18:45Z'), pd.Timestamp('2026-10-02'), {})
        self.assertFalse(fills or decisions)
        self.assertIsNone(state['pending'])

    def test_configuration_guard_and_outage(self):
        f, tri, u = fixture()
        s, *_ = process(None, f, tri, u, pd.Timestamp('2026-09-22T18:45Z'), pd.Timestamp('2026-09-22'), {})
        with self.assertRaises(ValueError): process(s, f, tri, u, pd.Timestamp('2026-10-02T18:45Z'), pd.Timestamp('2026-10-02'), {})
        s['config_hash'] = 'changed'
        with self.assertRaises(ValueError): process(s, f, tri, u, pd.Timestamp('2026-09-23T18:45Z'), pd.Timestamp('2026-09-23'), {})


if __name__ == '__main__': unittest.main()
