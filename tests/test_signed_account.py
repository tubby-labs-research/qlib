"""Public, numeric lifecycle checks for opt-in daily signed accounting."""

import math
import unittest
from unittest.mock import patch

import pandas as pd

from qlib.backtest import create_account_instance
from qlib.backtest.account import Account
from qlib.backtest.decision import Order
from qlib.backtest.executor import NestedExecutor, SimulatorExecutor
from qlib.backtest.position import BasePosition, Position
from qlib.backtest.report import PortfolioMetrics
from qlib.backtest.signed_account import SignedAccount
from qlib.backtest.signed_exchange import SignedExchange
from qlib.backtest.signed_position import SignedPosition


FRIDAY = pd.Timestamp("2024-01-05")
MONDAY = pd.Timestamp("2024-01-08")
TUESDAY = pd.Timestamp("2024-01-09")


def exchange(closes=None):
    """A typed exchange whose close source is visible to the test."""
    result = SignedExchange.__new__(SignedExchange)
    result.close_requests = []
    closes = {} if closes is None else closes

    def get_close(code, start, end):
        result.close_requests.append((code, start))
        return closes.get((start, code))

    result.get_close = get_close
    return result


def fill(account, day, code, signed_shares, price, fee_rate=0.0005):
    value = abs(signed_shares * price)
    order = Order(code, abs(signed_shares),
                  Order.BUY if signed_shares > 0 else Order.SELL, day, day)
    account.update_order(order, value, value * fee_rate, price)


def close_bar(account, day, venue):
    # Indicator calculation needs a strategy decision. Only that unrelated
    # calculation is replaced; position marking, metrics and history run normally.
    with patch.object(account, "update_indicator"):
        account.update_bar_end(day, day, venue, atomic=True,
                               outer_trade_decision=None, trade_info=[])
    return account.get_portfolio_metrics()[0].loc[day]


def simulator(*, frequency="day", trade_type=SimulatorExecutor.TT_SERIAL,
              settlement=BasePosition.ST_NO, metrics=True, venue=None):
    result = SimulatorExecutor.__new__(SimulatorExecutor)
    result.time_per_step = frequency
    result.trade_type = trade_type
    result._settle_type = settlement
    result.generate_portfolio_metrics = metrics
    result._trade_exchange = exchange() if venue is None else venue
    return result


class SignedAccountTests(unittest.TestCase):
    def test_public_account_injection_preserves_instance_and_long_only_default(self):
        signed = SignedAccount(init_cash=100)
        self.assertIs(create_account_instance(FRIDAY, MONDAY, None, signed), signed)
        self.assertIsInstance(signed.current_position, SignedPosition)
        # The legacy constructor maps benchmark=None to {} and then attempts a
        # default CSI300 data lookup. Isolate that lookup, not account creation.
        with patch.object(PortfolioMetrics, "_cal_benchmark", return_value=None) as benchmark_lookup:
            ordinary = create_account_instance(FRIDAY, MONDAY, None, 100)
        benchmark_lookup.assert_called_once()
        self.assertIs(type(ordinary), Account)
        self.assertIs(type(ordinary.current_position), Position)

    def test_rejects_unsupported_executor_shapes(self):
        account = SignedAccount(init_cash=100)
        account.validate_executor(simulator())
        for changes in (
            {"time_per_step": "1min"},
            {"trade_type": SimulatorExecutor.TT_PARAL},
            {"_settle_type": BasePosition.ST_CASH},
            {"generate_portfolio_metrics": False},
            {"_trade_exchange": object()},
        ):
            with self.subTest(changes=changes):
                executor = simulator()
                executor.__dict__.update(changes)
                with self.assertRaisesRegex(ValueError, "daily serial"):
                    account.validate_executor(executor)
        nested = NestedExecutor.__new__(NestedExecutor)
        with self.assertRaisesRegex(ValueError, "daily serial"):
            account.validate_executor(nested)

    def test_friday_to_monday_borrow_precedes_cover_and_is_charged_once(self):
        account = SignedAccount(init_cash=1000)
        venue = exchange({(FRIDAY, "S"): 12, (MONDAY, "S"): 8})
        account.start_bar(FRIDAY, FRIDAY, venue)
        fill(account, FRIDAY, "S", -10, 10)
        friday = close_bar(account, FRIDAY, venue)
        self.assertAlmostEqual(friday.account, 979.95)
        self.assertAlmostEqual(friday.free_cash, 999.95)
        self.assertAlmostEqual(friday.restricted_cash, 100)
        self.assertAlmostEqual(friday.short_liability, 120)
        self.assertAlmostEqual(friday.transaction_cost_amount, 0.05)
        self.assertEqual(friday.borrow_cost_amount, 0)

        account.start_bar(MONDAY, MONDAY, venue)
        borrow = 120 * 0.004 * 3 / 365
        self.assertAlmostEqual(account.current_position.borrow_cost, borrow)
        self.assertAlmostEqual(account.get_cash(), 999.95 - borrow)
        self.assertEqual(venue.close_requests, [("S", FRIDAY)])
        account.start_bar(MONDAY, MONDAY, venue)
        self.assertAlmostEqual(account.current_position.borrow_cost, borrow)
        fill(account, MONDAY, "S", 10, 8)
        monday = close_bar(account, MONDAY, venue)
        self.assertAlmostEqual(monday.account, 1019.91 - borrow)
        self.assertAlmostEqual(monday.free_cash, 1019.91 - borrow)
        self.assertEqual(monday.restricted_cash, 0)
        self.assertEqual(monday.short_liability, 0)
        self.assertAlmostEqual(monday.transaction_cost_amount, 0.04)
        self.assertAlmostEqual(monday.borrow_cost_amount, borrow)
        self.assertAlmostEqual(monday.total_cost_amount, 0.04 + borrow)
        self.assertAlmostEqual(monday.net_return,
                               (1019.91 - borrow - 979.95) / 979.95)
        self.assertAlmostEqual(monday["return"],
                               (1019.91 - borrow - 979.95 + 0.04 + borrow) / 979.95)
        self.assertEqual(venue.close_requests, [("S", FRIDAY)])

    def test_empty_order_session_accrues_from_previous_close(self):
        account = SignedAccount(init_cash=1000)
        venue = exchange({(FRIDAY, "S"): 10, (MONDAY, "S"): 11})
        account.start_bar(FRIDAY, FRIDAY, venue)
        fill(account, FRIDAY, "S", -10, 10, fee_rate=0)
        close_bar(account, FRIDAY, venue)
        account.start_bar(MONDAY, MONDAY, venue)
        monday = close_bar(account, MONDAY, venue)
        borrow = 100 * 0.004 * 3 / 365
        self.assertAlmostEqual(monday.account, 990 - borrow)
        self.assertAlmostEqual(monday.borrow_cost_amount, borrow)
        self.assertEqual(monday.transaction_cost_amount, 0)
        self.assertAlmostEqual(monday.total_cost_amount, borrow)
        self.assertAlmostEqual(monday.net_return, (-10 - borrow) / 1000)
        self.assertAlmostEqual(monday["return"], -10 / 1000)

    def test_per_order_return_equals_full_book_revaluation(self):
        # update_order measures each fill on the terms it can change; over a mixed sequence of opens,
        # adds, partial covers, flips and closes the booked return must equal full revaluations.
        account = SignedAccount(init_cash=100_000)
        venue = exchange()
        account.start_bar(FRIDAY, FRIDAY, venue)
        position = account.current_position
        steps = [("A", 30, 10.0), ("B", -40, 25.0), ("C", 12, 7.5), ("A", 15, 10.5), ("B", 10, 24.0),
                 ("C", -20, 7.0), ("D", -5, 101.0), ("B", 30, 23.0), ("A", -45, 11.0), ("D", 5, 99.0)]
        for code, shares, price in steps:
            with self.subTest(code=code, shares=shares):
                before_return = account.accum_info.get_return
                before_equity = position.calculate_value()
                cost = abs(shares * price) * 0.0005
                fill(account, FRIDAY, code, shares, price)
                self.assertAlmostEqual(account.accum_info.get_return - before_return,
                                       position.calculate_value() - before_equity + cost, places=9)

    def test_mixed_book_reports_long_and_short_values_against_net_equity(self):
        account = SignedAccount(init_cash=1000)
        venue = exchange({(FRIDAY, "L"): 12, (FRIDAY, "S"): 8})
        account.start_bar(FRIDAY, FRIDAY, venue)
        fill(account, FRIDAY, "L", 5, 10)
        fill(account, FRIDAY, "S", -10, 10)
        row = close_bar(account, FRIDAY, venue)
        self.assertAlmostEqual(row.free_cash, 949.925)
        self.assertAlmostEqual(row.restricted_cash, 100)
        self.assertAlmostEqual(row.cash_total, 1049.925)
        self.assertEqual(row.long_value, 60)
        self.assertEqual(row.short_liability, 80)
        self.assertAlmostEqual(row.account, 1029.925)
        self.assertAlmostEqual(row.transaction_cost_amount, 0.075)
        self.assertAlmostEqual(row.total_cost_amount, 0.075)
        self.assertAlmostEqual(row.net_return, 29.925 / 1000)
        self.assertAlmostEqual(row["return"], 30 / 1000)

    def test_nested_bar_close_is_rejected_without_closing_session(self):
        account = SignedAccount(init_cash=100)
        venue = exchange()
        account.start_bar(FRIDAY, FRIDAY, venue)
        with self.assertRaisesRegex(ValueError, "nested signed execution"):
            account.update_bar_end(FRIDAY, FRIDAY, venue, atomic=False,
                                   outer_trade_decision=None)
        self.assertEqual(account._active_session, FRIDAY)
        close_bar(account, FRIDAY, venue)

    def test_stale_close_is_flagged_and_retained_for_next_borrow_gap(self):
        account = SignedAccount(init_cash=1000)
        venue = exchange({(FRIDAY, "S"): 12, (MONDAY, "S"): None,
                          (TUESDAY, "S"): 8})
        account.start_bar(FRIDAY, FRIDAY, venue)
        fill(account, FRIDAY, "S", -10, 10, fee_rate=0)
        close_bar(account, FRIDAY, venue)
        account.start_bar(MONDAY, MONDAY, venue)
        monday = close_bar(account, MONDAY, venue)
        self.assertEqual(monday.stale_marks, ("S",))
        self.assertEqual(monday.short_liability, 120)
        self.assertEqual(account.current_position.get_stock_price("S"), 12)
        account.start_bar(TUESDAY, TUESDAY, venue)
        self.assertAlmostEqual(account.current_position.borrow_cost,
                               120 * 0.004 * 4 / 365)
        self.assertEqual(venue.close_requests, [("S", FRIDAY), ("S", MONDAY)])
        tuesday = close_bar(account, TUESDAY, venue)
        self.assertEqual(tuesday.stale_marks, ())
        self.assertEqual(tuesday.short_liability, 80)

    def test_zero_and_negative_equity_keep_observations_without_ratios(self):
        zero = SignedAccount(init_cash=1)
        zero_venue = exchange({(FRIDAY, "S"): 2})
        zero.start_bar(FRIDAY, FRIDAY, zero_venue)
        fill(zero, FRIDAY, "S", -1, 1, fee_rate=0)
        zero_row = close_bar(zero, FRIDAY, zero_venue)
        self.assertEqual(zero_row.account, 0)
        self.assertFalse(zero_row.metrics_defined)
        self.assertTrue(math.isnan(zero_row.net_return))
        self.assertTrue(math.isnan(zero_row["return"]))

        negative = SignedAccount(init_cash=1)
        adverse = exchange({(FRIDAY, "S"): 100, (MONDAY, "S"): 50})
        negative.start_bar(FRIDAY, FRIDAY, adverse)
        fill(negative, FRIDAY, "S", -100, 1, fee_rate=0)
        first = close_bar(negative, FRIDAY, adverse)
        self.assertEqual(first.account, -9899)
        self.assertFalse(first.funding_feasible)
        negative.start_bar(MONDAY, MONDAY, adverse)
        second = close_bar(negative, MONDAY, adverse)
        borrow = 10000 * 0.004 * 3 / 365
        self.assertAlmostEqual(second.free_cash, 1 - borrow)
        self.assertAlmostEqual(second.unfunded_cash_shortfall, max(borrow - 1, 0))
        self.assertAlmostEqual(second.account, -4899 - borrow)
        self.assertFalse(second.funding_feasible)
        self.assertFalse(second.metrics_defined)
        self.assertTrue(math.isnan(second.net_return))
        self.assertTrue(math.isnan(second["return"]))
        self.assertEqual(negative.current_position.get_stock_amount("S"), -100)


if __name__ == "__main__":
    unittest.main()
