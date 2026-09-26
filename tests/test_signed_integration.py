# Copyright (c) Tubby Labs contributors.
# Licensed under the MIT License.
"""Real Qlib daily-loop checks on tiny, entirely synthetic binary datasets."""

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import qlib
from qlib.backtest import backtest
from qlib.backtest.signed_account import SignedAccount
from qlib.backtest.signed_exchange import SignedExchange
from qlib.contrib.strategy.signed_weight import SignedWeightStrategy


DATES = pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"])


class SignedIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_case(self, weights, *, fee=0, monday_open=11, closes=None, cash=100):
        root = self.root
        (root / "calendars").mkdir(exist_ok=True)
        (root / "instruments").mkdir(exist_ok=True)
        (root / "calendars/day.txt").write_text("\n".join(d.strftime("%Y-%m-%d") for d in DATES) + "\n")
        (root / "instruments/all.txt").write_text("S\t2024-01-04\t2024-01-10\n")
        folder = root / "features/s"
        folder.mkdir(parents=True, exist_ok=True)
        fields = {
            "open": [10, 10, monday_open, 8, 10],
            "close": closes if closes is not None else [10, 12, 15, 8, 10],
            "factor": [1] * 5, "volume": [1000000] * 5,
        }
        for name, values in fields.items():
            np.array([0] + values, dtype="<f4").tofile(folder / (name + ".day.bin"))
        qlib.init(provider_uri=str(root), region="us", kernels=1, expression_cache=None, dataset_cache=None)
        series = pd.Series(
            list(weights.values()),
            index=pd.MultiIndex.from_tuples([(pd.Timestamp(day), "S") for day in weights], names=["datetime", "instrument"]),
            dtype=float,
        )
        strategy = SignedWeightStrategy(signal=series)
        account = SignedAccount(init_cash=cash)
        exchange = SignedExchange(start_time=DATES[1], end_time=DATES[3], codes=["S"], open_cost=fee, close_cost=fee)
        result, _ = backtest(
            start_time=DATES[1], end_time=DATES[3], strategy=strategy,
            executor={"class": "qlib.backtest.executor.SimulatorExecutor", "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}},
            account=account, benchmark=None, exchange_kwargs={"exchange": exchange},
        )
        report, positions = result["1day"]
        return account, strategy, report, positions

    def test_weekend_cover_uses_friday_close_before_monday_orders(self):
        account, strategy, report, positions = self.run_case({"2024-01-04": -1, "2024-01-05": 0, "2024-01-08": 0})
        borrow = 120 * 0.004 * 3 / 365
        self.assertEqual(positions[DATES[1]].get_stock_amount("S"), -10)
        self.assertAlmostEqual(report.loc[DATES[1], "account"], 80)
        self.assertAlmostEqual(report.loc[DATES[2], "borrow_cost_amount"], borrow)
        self.assertAlmostEqual(strategy.decisions[1]["sizing_equity"], 80 - borrow)
        self.assertAlmostEqual(report.loc[DATES[2], "account"], 90 - borrow)
        self.assertEqual(report.loc[DATES[3], "borrow_cost_amount"], 0)
        self.assertEqual(account.current_position.get_stock_amount_dict(), {})
        self.assertAlmostEqual(report["net_return"].add(1).prod() * 100, report.iloc[-1]["account"])
        np.testing.assert_allclose(report["return"] - report["cost"], report["net_return"])

    def test_empty_order_session_still_charges_borrow_and_keeps_short(self):
        _, strategy, report, positions = self.run_case({"2024-01-04": -1, "2024-01-08": 0})
        self.assertEqual(strategy.decisions[1]["reasons"]["session"], "missing_signal")
        self.assertEqual(positions[DATES[2]].get_stock_amount("S"), -10)
        self.assertAlmostEqual(report.loc[DATES[3], "borrow_cost_amount"], 150 * 0.004 / 365)
        self.assertAlmostEqual(report.iloc[-1]["account"], 120 - (120 * 3 + 150) * 0.004 / 365)

    def test_costs_net_equity_and_rounding_are_reconciled(self):
        _, _, report, positions = self.run_case({"2024-01-04": -1, "2024-01-05": 0, "2024-01-08": 0}, fee=0.0005)
        self.assertAlmostEqual(report["transaction_cost_amount"].sum(), 0.105)
        self.assertAlmostEqual(report["total_cost_amount"].sum(), report["transaction_cost_amount"].sum() + report["borrow_cost_amount"].sum())
        self.assertAlmostEqual(report.iloc[-1]["account"], 90 - 0.105 - 120 * 0.004 * 3 / 365)
        for day, pos in positions.items():
            self.assertAlmostEqual(pos.calculate_value(), report.loc[day, "account"])
            self.assertGreaterEqual(pos.get_cash(), 0)

    def test_all_long_has_no_borrow_and_cannot_spend_fee_budget_twice(self):
        _, strategy, report, positions = self.run_case({"2024-01-04": 1, "2024-01-05": 0, "2024-01-08": 0}, fee=0.0005)
        self.assertEqual(strategy.decisions[0]["target_quantities"]["S"], 10)
        self.assertEqual(positions[DATES[1]].get_stock_amount("S"), 9)
        self.assertEqual(report["borrow_cost_amount"].sum(), 0)
        self.assertAlmostEqual(report.iloc[-1]["account"], 109 - (90 + 99) * 0.0005)

    def test_missing_open_does_not_fill_at_future_close(self):
        _, strategy, _, positions = self.run_case({"2024-01-04": -1, "2024-01-05": 0, "2024-01-08": 0}, monday_open=float("nan"))
        self.assertEqual(positions[DATES[2]].get_stock_amount("S"), -10)
        self.assertIn("S", strategy.decisions[1]["reasons"])
        self.assertEqual(positions[DATES[3]].get_stock_amount_dict(), {})

    def test_missing_close_does_not_suppress_valid_open_fill(self):
        _, _, report, positions = self.run_case({"2024-01-04": -1, "2024-01-05": 0, "2024-01-08": 0}, closes=[10, float("nan"), 15, 8, 10])
        self.assertEqual(positions[DATES[1]].get_stock_amount("S"), -10)
        self.assertEqual(report.loc[DATES[1], "stale_marks"], ("S",))
        self.assertAlmostEqual(report.loc[DATES[2], "borrow_cost_amount"], 100 * 0.004 * 3 / 365)


if __name__ == "__main__":
    unittest.main()
