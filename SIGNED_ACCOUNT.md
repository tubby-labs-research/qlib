# Opt-in signed daily accounting — v1

Use when building, testing or extending this derivative's signed-position mode.
It is a restricted research simulation, not a broker margin model or a claim of
market neutrality. Ordinary Qlib accounts and strategies remain unchanged.

## Identity and patch boundary

- Upstream: Microsoft Qlib v0.9.7, `da920b7f954f48ab1bb64117c976710de198373e`.
- Derivative: `tubby-labs-research/qlib`, version `0.9.7+tubby.2` (tubby.1 plus the
  changes listed under *Versions* below; accounting results are unchanged).
- MIT notices remain in place. This repository contains generic framework code,
  synthetic fixtures and tests, not private strategies, portfolios or market data.
- Core hooks: accept an `Account` instance through public backtest configuration;
  call its compatibility check and pre-decision bar hook. Both hooks are no-ops
  for an ordinary account. No executor algorithm was replaced.
- Opt-in components: `qlib.backtest.signed_position.SignedPosition`,
  `qlib.backtest.signed_exchange.SignedExchange`,
  `qlib.backtest.signed_account.SignedAccount`, and
  `qlib.contrib.strategy.signed_weight.SignedWeightStrategy`.

## Supported contract

Initialize Qlib in the US configuration with no global price-limit threshold.
Use a fresh cash-only SignedAccount, SignedExchange and daily serial
SimulatorExecutor with portfolio metrics enabled and no settlement delay.
Pass the account object to `backtest(account=...)` and the exchange object through
`exchange_kwargs={"exchange": exchange}`. The account's benchmark configuration
is authoritative when an account instance is supplied.

The strategy consumes **already-normalized** signed target weights indexed by
`datetime` and `instrument`. It uses only the previous session's weights.
Gross target weight must not exceed one. Alpha evaluation, normalization,
neutralization and portfolio optimization belong to the caller, not this engine.

Target shares are whole shares rounded toward zero from
`weight * previous_close_equity_after_borrow_charge / current_open`.
Long sales precede short covers; other orders have deterministic symbol order.
Reductions precede increases. Affordability can clip or reject an order, and
unused cash is not redistributed. Actual exposure can differ from targets.
The strategy retains targets, sizing equity, reasons and issued order objects in
`decisions`; filled orders retain actual quantities and skip/partial reasons.
Missing signals mean no rebalance, while explicit zero weights request cash.

Short-sale proceeds are pooled at their original sale values per instrument,
not spendable cash. Covering releases the matching fraction; reversals account
for closing and opening quantities without double charging traded value.
Fees default to 0.0005 of traded value on either side; unequal buy/sell rates,
minimum fees, volume caps and impact are rejected in this first mode.
No cash loan, interest credit or short-proceeds rebate is simulated.

Borrow fees default to `prior_close_short_value * 0.004 * calendar_days / 365`.
They are charged once, before strategy sizing, including empty-order sessions
and weekend gaps. Today's cover does not erase the prior gap's fee. No charge
is booked after the final observation. All stocks are assumed borrowable;
there are no recalls or observed historical borrow rates.

## Outputs and deliberate limitations

Use `net_return` and net `account` equity, with separate
`transaction_cost_amount`, `borrow_cost_amount` and `total_cost_amount`.
For native Qlib compatibility, the raw table still includes `return` and `cost`;
their difference reconciles to `net_return`. Do not present native gross returns
as the primary result or subtract borrow expense a second time.
Free cash, restricted cash, total cash, long value, short liability and unfunded
shortfalls are retained. `funding_feasible` is only a nonnegative-free-cash and
positive-equity check, **not** broker feasibility or a margin approval.

Borrow expenses may create negative cash and losses may exhaust equity.
Balances remain visible; there is no forced liquidation, margin call or invented
financing. Nonpositive equity disables new scaled targets and return ratios.
Missing closing marks carry the last known mark and flag `stale_marks`; their
borrow valuation is consequently stale too. These runs need data-quality review.
Missing opens never fall back to the future close. Daily-close-derived limits
are rejected, including a non-None global Qlib limit configuration. Missing
closing quotes do not decide whether an opening fill is allowed.

Only identity adjustment factors are supported. Splits, dividend obligations,
delistings, point-in-time universe selection, settlement, intraday/nested execution,
broker collateral and shortability are not validated. A factor of one is a unit
convention, **not** proof that data are raw or action-correct.

## Build and validation

With a suitable C++ compiler, the ordinary source build remains available:
`python -m pip wheel . --no-deps --wheel-dir <new-wheel-directory>`.
Build/install in a separate environment, then run the signed tests from outside
the checkout so an uncompiled source directory cannot shadow the installed wheel:
`python -m unittest discover -s <checkout>/tests -p "test_signed_*.py"`.

On Windows without MSVC, `scripts/build_signed_wheel.py` offers a narrowly scoped
binary-preserving build. Download the matching **official pyqlib 0.9.7 wheel**
and verify its SHA256 against PyPI. For CPython 3.12 Windows x64, the verified
base digest is `dfbee9f0f3005fe805798e2a21c73b198272f1341d5f0d7771e127166faac08e`.
Invoke the helper with `--base-wheel`, `--base-sha256`, and `--output-dir`.
Use a clean committed checkout for acceptance builds; `--allow-dirty` is for
development only. Existing output wheels are never overwritten.

This route is **not a fresh native compilation**: it retains the exact released
native binaries, overlays only tracked changed Python files, updates distribution
metadata and RECORD, and embeds `qlib/_tubby_build.json` with source commit,
upstream commit, native hashes, Python hashes and base-wheel identity. It rejects
native/build-source changes. A later native or dependency/ABI change requires a
normal source build and fresh validation, not an extension of this shortcut.
Wheel hashes and the exact tested commit must be recorded by the integrating project.

Tests include six literal accounting paths / 26 event snapshots, affordability,
reversals, integer-share precision, unsupported configurations, timing, costs,
stale prices, negative equity and real Qlib daily-loop tests on tiny synthetic
binary datasets. No market data downloads or private project imports are needed.
Passing these does not certify the complete upstream Qlib test suite or readiness
to trade. Preserve long-only regressions when changing the core hooks.

## Maintenance

Keep framework changes separate from data adapters and user-facing consoles.
Review changes against the pinned upstream baseline. Rebase or remove the hooks
only after equivalent upstream behavior passes these fixtures. Keep old builds
and manifests identifiable; never patch installed site-packages as a hidden fix.
This checkpoint does not install the derivative into any existing application.

## Versions

- `0.9.7+tubby.2` (2026-09-27): `SignedAccount.update_order` measures each fill on the equity
  terms it can change (free cash, that instrument's restricted proceeds and marked value)
  instead of revaluing every holding twice per order, which made a thousand-name session
  quadratic. TOP3000 signed runs are 1.7-2.1x faster with identical ledgers, positions and
  metrics. The wheel builder fixes the ZIP creating-OS field, so Windows and macOS builds of
  the same commit are byte-identical.
- `0.9.7+tubby.1` (2026-09-26): opt-in daily signed accounting and restricted short proceeds.
