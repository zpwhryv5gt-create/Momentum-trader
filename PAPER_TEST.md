# V8 audited forward paper test

The scheduled runner is separate from the interactive historical research app.
It runs on GitHub Actions, so no phone, iPad, or open Streamlit session is required.

- Workflow: hourly at :17, 08:00–18:00 UTC, weekdays (London DST is checked in Python).
- A push changing runner code also starts a verification run. Manual Run workflow is available in Actions.
- GitHub schedules can be delayed or dropped. This is an indicative hourly paper test, not low-latency trading infrastructure.
- Start: first successful in-session run after installation, with £1,000 fresh cash in each book.
- Old paper_test_log.csv entries are target-only records, excluded from audited performance.
- Strategy parameters and universe are extracted unchanged into v8_core.py. No optimisation.
- The initial historical sample anchors the evaluation grid and is retained in paper/signal_prices.csv.
- New decisions use completed hourly bars, conservatively delayed by 65 minutes from bar start.
- A decision is recorded before a simulated fill at the first subsequently observed hourly open.
  On an outage, only previously recorded orders can fill; intervening strategy decisions are never invented.
- No continuous rebalancing when the target is unchanged. Fractional units are allowed.
- Both V8 and passive VWRP pay 8 bps one-way costs. Cash earns zero. No borrowing.
- Quote currency must be GBP or GBp/GBX, explicitly normalized to pounds.
- No forward-filled execution prices. Missing/stale data fails closed and preserves the prior ledger.
- Corporate actions after start, currency changes, or material historical revisions halt for reconciliation.
  This is a known limitation, particularly for distributing ETFs; alerts must not call the old valuation current.
- Current-day results are provisional until the final session bar is available.
- The initial benchmark purchase is scheduled when the strategy test starts, even if V8 elects cash.
- Simulated fills are indicative Yahoo-data executions, not verified broker fills.

## Evidence

- paper/REPORT.md: readable latest performance
- paper/daily.csv: daily valuations and returns versus VWRP
- paper/fills.csv: signed units, reference prices, transaction costs and observed times
- paper/decisions.csv: forward-recorded pending targets
- paper/valuations.csv: intraday mark-to-market history
- paper/state.json: cash, units, pending orders and frozen replay state
- paper/health.json: last successful/no-new-bar check
- GitHub Actions status and artifacts: failures (the last committed health file may be older)

Each successful run commits its ledger as one transaction. Failed calculations never commit a partial ledger.
Concurrent runs are serialized. A concurrent external repository edit may reject a push: the artifact preserves
that attempted run but it is not authoritative until reconciled; the reporter must flag the failed run.

## Verification

Run python -m unittest -v test_v8_paper.py. Tests cover no borrowing, cost reconciliation,
pence conversion, fill timing, duplicate-run idempotence, benchmark start, and currency-change rejection.

Market-data package versions for each hosted run are shown in its install logs.
Frozen research calculations still have the limitations described in the original app; this infrastructure
does not establish a profitable edge.

GitHub schedule documentation: https://docs.github.com/actions/using-workflows/events-that-trigger-workflows#schedule
