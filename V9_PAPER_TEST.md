# V9 monthly trend: forward paper implementation

Installed on 22 September 2026. This is an experiment, not a validated investment edge.
No broker connection, real money, leverage, shorting or optimisation.
V8 code and its paper/ ledger are unchanged. V9 owns paper_v9/ exclusively.

## Frozen investment rules

- £1,000 simulated opening cash per book; cash interest zero.
- Global equities VWRP.L: maximum target 50%; UK government bonds IGLT.L: 25%; physical gold SGLN.L: 25%.
- At each XLON month-end, use daily GBP total-return data at the final exchange sessions 1, 3 and 12 calendar months earlier.
- Each strictly positive horizon supplies one third of that asset's policy budget; ties vote zero. Unused allocations stay in cash.
- Estimate risk using the preliminary weights applied to the latest 63 aligned daily total returns. Use the larger of 21- and 63-session sample standard deviations, annualised with sqrt(252).
- Scale all risky targets by min(1, 0.12 / estimated_volatility). All-zero risky targets remain cash. Invalid or zero volatility with nonzero exposure is a data exception.
- Monthly decisions only; daily bookkeeping never changes the signal. No automatic return-driven gate, re-optimisation, ranking or retrospective startup.
- Config hash is persisted and checked. Changing investment rules requires a new explicitly identified experiment.

## Deployment execution protocol — explicit differences from the research proposal

The saved research proposal specified next-session 15:00 bid/ask observations.
This deployment instead uses a **pre-recorded next-XLON-session close** with an
8-basis-point all-in spread/slippage cost on each side. It follows V8's indicative
Yahoo-data simulation style; it is not a live quote or broker execution test.
This protocol is frozen before the first decision. It must not be presented as
an exact replication of the proposal's intraday execution convention.

We use whole units rounded down because fractional support has not been verified.
This can leave material residual cash at £1,000, especially in VWRP. Targets,
filled quantities and residual cash are distinct. Both strategy and controls
obey the same rounding and cost conventions. Sell first, then scale buys to cash
available including costs. No negative cash and no spending dividend receivables.
An existing position retains its price exposure until its sale at the execution
close. A new position receives no earlier session return.

Data are daily Yahoo unadjusted closes for fills/valuation, adjusted closes for
signals and risk. GBP and GBp/GBX normalize to pounds. Other currencies fail.
Accumulating fund income is not credited separately. Missing prices are never
forward-filled. Month-end signal input snapshots are retained with hashes.
This is a single-provider indicative data test, not independent data verification.

## Start and scheduling

GitHub Actions runs weekdays at 17:43 and 18:43 UTC, with holiday/DST/early-close
handling through the XLON exchange calendar. The second run is a retry/check;
no-new-session runs cannot duplicate fills. GitHub may delay or drop schedules.
A code push or manual workflow dispatch also runs verification and bookkeeping.
The closing data are not eligible until 90 minutes after the exchange close.

Start in cash on the first successful hosted run. The first intended signal is
30 September 2026, for simulated execution at the 1 October close, conditional
on complete data and timely successful runs. Do not backdate a September entry.
All controls start trading on that same first eligible execution date, even if
V9's signal chooses cash. Waiting cash is not evidence of strategy performance.

Previously recorded pending orders can be accounted for after a short outage,
but missed intermediate decisions are never invented. An outage over seven
calendar days stops the runner for reconciliation. Late month-end decisions
cannot obtain an execution close that has already happened.

## Dividends and splits

IGLT distributes income. Entitlement accrues using the holdings carried into
the ex-date, before any close trades. Receivables count in NAV but not cash.
The cash payment requires an issuer-verified entry in v9_dividend_payments.json:

    "IGLT.L:YYYY-MM-DD": {
      "pay_date": "YYYY-MM-DD",
      "per_unit_gbp": 0.1234,
      "source_url": "https://issuer-document-url"
    }

Dates and amounts above are placeholders, not actual payment records. Missing
payment dates raise a visible warning and keep the entitlement unspent. If a
payment is verified late, cash is released at the next processed session;
historical cash entries are not silently rewritten. This conservative delay
is a documented limitation until a verified payment feed is added. The daily
reporting task must flag it. Splits in held securities stop for reconciliation
because vendor historical-price restatements need an explicit audit.

## Controls

Every book has its own £1,000 ledger, holdings and costs:

- v9: trend plus one-sided risk ceiling.
- vwrp: buy and hold, with cash dividends retained if any.
- passive_mix: monthly 50/25/25 allocation.
- passive_scaled: passive mix with the same monthly risk ceiling.
- trend_unscaled: trend votes without risk scaling.
- vwrp_scaled: monthly VWRP/cash risk ceiling.
- cash: zero-interest cash.

These controls attribute outcomes; they are not a competition to select a new
rule after seeing results. V8 has an earlier inception. Compare V8 and V9 using
matching completed sessions and rebased NAV, never unmatched since-start returns.
No historical backtest or 12/20 bps stress study has been performed by this
installation; those remain separate research diagnostics.

## Files and reliability

- paper_v9/REPORT.md: readable latest account values, costs, holdings and pending targets.
- paper_v9/state.json: authoritative account state, frozen config and pending decision.
- paper_v9/daily.csv: completed-session values for every book.
- paper_v9/decisions.csv and snapshots/: immutable decision evidence and signal data.
- paper_v9/fills.csv: signed whole units, reference price, cost and SIMULATED_FILL status.
- paper_v9/events.csv: dividend receivable and cash events.
- paper_v9/health.json: latest successful check. On failure, inspect the newest Actions run/artifact; committed health can be older.

No fills/decisions files exist until the first such event. That is expected.
A successful repository commit is the durable transaction boundary. Failed
runs do not commit partial state. Workflow concurrency serializes V9 runs.
Independent V8 commits are rebased before the V9 ledger push; conflicts fail
rather than overwrite either ledger. Artifacts retain attempted-run evidence
for 90 days but are not authoritative unless committed.

Run python -m unittest -v test_v9_paper.py for accounting, timing, calendar,
configuration, data-failure and duplicate-run checks. Core dependency versions
are pinned separately from V8. No secrets or broker credentials are required.
