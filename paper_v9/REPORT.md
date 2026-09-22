# V9 forward paper test

Status: **WAITING_FOR_FIRST_MONTH_END_EXECUTION**.

Observed 2026-09-22 20:07:16.287124+00:00; valuation at London session close 2026-09-22 15:30:00+00:00.

Created 2026-09-22 20:07:16.287124+00:00. Each book starts with £1,000 simulated cash.

| Book | Value | Since start | Latest session | Costs |
|---|---:|---:|---:|---:|
| v9 | £1000.00 | 0.00% | 0.00% | £0.00 |
| vwrp | £1000.00 | 0.00% | 0.00% | £0.00 |
| passive_mix | £1000.00 | 0.00% | 0.00% | £0.00 |
| passive_scaled | £1000.00 | 0.00% | 0.00% | £0.00 |
| trend_unscaled | £1000.00 | 0.00% | 0.00% | £0.00 |
| vwrp_scaled | £1000.00 | 0.00% | 0.00% | £0.00 |
| cash | £1000.00 | 0.00% | 0.00% | £0.00 |

V9 holdings (whole units): {}.
Cash £1000.00; dividend receivables £0.00.

Pending order: none

Monthly 1/3/12-month trend; policy budgets 50% VWRP / 25% IGLT / 25% SGLN; estimated 12% volatility ceiling. No leverage.

All fills are SIMULATED at a pre-recorded next-session daily close, with 8 bps per side; Yahoo prices are indicative, not broker quotes. Whole units can leave substantial cash at £1,000. Cash interest is zero.

Benchmarks start trading at the same first execution as V9. vwrp is buy-and-hold; passive_mix is monthly 50/25/25. Other books isolate trend and risk scaling.

Dividend entitlements count in NAV but are not spendable until a verified payment date is supplied. Missing payment dates generate a warning, not invented cash.

Check the latest GitHub Actions run and health.json before treating this report as current. V8 and V9 have different inception dates; compare only matching dates, not raw since-start returns.

Frozen configuration SHA256: 833633785ee1d4518addb015664aa4379fc71e8c9e3edd696eae2004e8a679a8
