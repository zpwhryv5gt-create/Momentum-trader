import copy
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
import json
import pandas as pd
import numpy as np
import v8_paper as p

class LedgerTests(unittest.TestCase):
    def test_pence_and_currency_rejection(self):
        self.assertEqual(p.currency_scale("GBp") * 12345, 123.45)
        self.assertEqual(p.currency_scale("GBP"), 1)
        with self.assertRaises(ValueError):
            p.currency_scale("USD")

    def test_costs_cash_and_rotation(self):
        a = dict(cash=1000.0, holdings={}, costs=0.0)
        prices = pd.Series({"A": 10.0, "B": 20.0})
        fills = p.rebalance(a, {"A": 1.0}, prices, "t", "r", "strategy")
        self.assertGreaterEqual(a["cash"], 0)
        self.assertAlmostEqual(p.value(a, prices) + a["costs"], 1000)
        self.assertAlmostEqual(fills[0]["quantity"], 1000 / 10.008)
        p.rebalance(a, {"B": 0.72}, prices, "t2", "r2", "strategy")
        self.assertNotIn("A", a["holdings"])
        self.assertGreater(a["cash"], 270)
        self.assertAlmostEqual(p.value(a, prices) + a["costs"], 1000)
        p.rebalance(a, {}, prices, "t3", "r3", "strategy")
        self.assertFalse(a["holdings"])
        self.assertAlmostEqual(a["cash"] + a["costs"], 1000)

    def test_usd_fx_fee(self):
        a = dict(cash=1000.0, holdings={}, costs=0.0)
        fills = p.rebalance(a, {"GOLD": 1.0}, pd.Series({"GOLD": 50.0}),
                            "t", "r", "strategy", ["GOLD"])
        self.assertAlmostEqual(a["holdings"]["GOLD"], 1000 / (50 * 1.0023))
        self.assertAlmostEqual(fills[0]["fx_cost_gbp"], fills[0]["notional_gbp"] * .0015)
        self.assertAlmostEqual(p.value(a, pd.Series({"GOLD": 50.0})) + a["costs"], 1000)

    def test_never_fill_at_or_before_recording(self):
        ix = pd.date_range("2026-09-09 08:00", periods=4, freq="h", tz="UTC")
        opens = pd.DataFrame({"A": [10, 11, 12, 13]}, index=ix)
        self.assertEqual(p.eligible_open(opens, {"recorded_at": str(ix[1])}), ix[2])
        self.assertIsNone(p.eligible_open(opens, {"recorded_at": str(ix[-1])}))

    def test_forward_run_idempotence_and_benchmark(self):
        ix = pd.date_range(end="2026-09-09 08:00Z", periods=1100, freq="h")
        c = pd.DataFrame(100.0, index=ix, columns=p.DEFAULT_TICKERS)
        scales = {s: 1.0 for s in p.DEFAULT_TICKERS}
        with TemporaryDirectory() as tmp, patch.object(p, "ROOT", Path(tmp)), \
             patch.object(p, "get_market", return_value=(c, c, scales, c)), \
             patch.object(p, "desired_allocation", return_value=({"IITU.L": .72}, True)):
            p.run(pd.Timestamp("2026-09-09 09:15Z"))
            state1 = json.loads((p.ROOT / "state.json").read_text())
            self.assertEqual(state1["strategy"]["holdings"], {})
            self.assertFalse((p.ROOT / "fills.csv").exists())
            p.run(pd.Timestamp("2026-09-09 09:20Z"))
            self.assertEqual(len(pd.read_csv(p.ROOT / "valuations.csv")), 1)
            c2 = pd.concat([c, pd.DataFrame(101.0, index=pd.DatetimeIndex([
                pd.Timestamp("2026-09-09 10:00Z")]), columns=p.DEFAULT_TICKERS)])
            with patch.object(p, "get_market", return_value=(c2, c2, scales, c2)):
                p.run(pd.Timestamp("2026-09-09 11:15Z"))
                state2 = json.loads((p.ROOT / "state.json").read_text())
                self.assertAlmostEqual(state2["strategy"]["holdings"]["IITU.L"], 720 / 101)
                self.assertAlmostEqual(state2["strategy"]["cash"], 280 - 720*.0008)
                self.assertGreater(state2["benchmark"]["holdings"]["VWRP.L"], 0)
                self.assertEqual(len(pd.read_csv(p.ROOT / "fills.csv")), 2)
                p.run(pd.Timestamp("2026-09-09 11:20Z"))
                self.assertEqual(len(pd.read_csv(p.ROOT / "fills.csv")), 2)

    def test_currency_change_preserves_state(self):
        ix = pd.date_range(end="2026-09-09 08:00Z", periods=1100, freq="h")
        c = pd.DataFrame(100.0, index=ix, columns=p.DEFAULT_TICKERS)
        scales = {s: 1.0 for s in p.DEFAULT_TICKERS}
        with TemporaryDirectory() as tmp, patch.object(p, "ROOT", Path(tmp)), \
             patch.object(p, "get_market", return_value=(c, c, scales, c)), \
             patch.object(p, "desired_allocation", return_value=({}, False)):
            p.run(pd.Timestamp("2026-09-09 09:15Z"))
            before = (p.ROOT / "state.json").read_text()
            changed = {**scales, "IITU.L": .01}
            with patch.object(p, "get_market", return_value=(c, c, changed, c)):
                with self.assertRaises(ValueError):
                    p.run(pd.Timestamp("2026-09-09 10:15Z"))
            self.assertEqual(before, (p.ROOT / "state.json").read_text())

if __name__ == "__main__":
    unittest.main()
