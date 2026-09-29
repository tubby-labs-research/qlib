"""Focused execution checks for the signed daily-open exchange."""

import math
import logging
import unittest

import pandas as pd

from qlib.backtest.decision import Order
from qlib.backtest.signed_exchange import SignedExchange
from qlib.backtest.signed_position import SignedPosition


DAY = pd.Timestamp("2024-01-02")
NEXT = pd.Timestamp("2024-01-03")


class FakeQuote:
    def __init__(self):
        self.data = {"$open": 10.0, "$close": 99.0, "$factor": 1.0,
                     "limit_buy": False, "limit_sell": False}

    def get_all_stock(self):
        return {"AAA"}

    def get_data(self, stock_id, start_time, end_time, field, method="ts_data_last"):
        return self.data.get(field)


def exchange():
    result = SignedExchange.__new__(SignedExchange)
    result.quote = FakeQuote()
    result.buy_price = result.sell_price = "$open"
    result.open_cost = result.close_cost = 0.0005
    result.min_cost = result.impact_cost = 0.0
    result.logger = logging.getLogger("test_signed_exchange")
    return result


def order(amount, direction):
    return Order("AAA", amount, direction, DAY, NEXT)


class SignedExchangeTests(unittest.TestCase):
    def test_small_positive_fill_is_actually_booked(self):
        ex = exchange()
        ex.quote.data["$open"] = 1e-6
        position = SignedPosition(cash=1)
        buy = order(1, Order.BUY)
        value, cost, _ = ex.deal_order(buy, position=position)
        self.assertEqual(position.get_stock_amount("AAA"), buy.deal_amount)
        self.assertEqual(buy.deal_amount, 1)
        self.assertAlmostEqual(position.get_cash(), 1 - value - cost)

    def test_unsupported_configuration_rejected_before_data_access(self):
        for kwargs in ({"open_cost": 0.001, "close_cost": 0.002}, {"limit_threshold": 0.1},
                       {"impact_cost": 0.01}, {"min_cost": 5}, {"volume_threshold": ("cum", "$volume")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(NotImplementedError):
                SignedExchange(**kwargs)

    def test_tiny_unfunded_fill_is_clipped_not_rejected_by_ledger(self):
        ex = exchange()
        required = 10 * (1 + 0.0005)
        position = SignedPosition(cash=required - 5e-10)
        buy = order(1, Order.BUY)
        ex.deal_order(buy, position=position)
        self.assertEqual(buy.deal_amount, 0)
        self.assertEqual(position.get_stock_amount_dict(), {})

    def test_short_sale_and_cover_with_restricted_cash(self):
        ex = exchange()
        position = SignedPosition(cash=1.0)
        sell = order(10, Order.SELL)
        value, cost, price = ex.deal_order(sell, position=position)
        self.assertEqual((value, cost, price, sell.deal_amount), (100.0, 0.05, 10.0, 10.0))
        self.assertAlmostEqual(position.get_cash(), 0.95)
        self.assertEqual(position.restricted_proceeds["AAA"], 100.0)
        cover = order(10, Order.BUY)
        ex.deal_order(cover, position=position)
        self.assertEqual(cover.deal_amount, 10)
        self.assertAlmostEqual(position.get_cash(), 0.9)
        self.assertFalse(position.check_stock("AAA"))

    def test_restricted_proceeds_do_not_buy_unrelated_stock(self):
        ex = exchange()
        position = SignedPosition(cash=1.0)
        ex.deal_order(order(10, Order.SELL), position=position)
        ex.quote.data["$open"] = 1.0
        # Another stock's proceeds are still restricted when buying this one.
        ex.quote.get_all_stock = lambda: {"AAA", "BBB"}
        buy = Order("BBB", 100, Order.BUY, DAY, NEXT)
        value, cost, _ = ex.deal_order(buy, position=position)
        self.assertEqual(buy.deal_amount, 0)
        self.assertEqual(value, cost)
        self.assertEqual(buy.signed_skip_reason, "insufficient_free_cash")

    def test_profitable_cover_can_remedy_cash_deficit_then_reverse(self):
        ex = exchange()
        position = SignedPosition(cash=0.1)
        ex.deal_order(order(10, Order.SELL), position=position)
        position.position["cash"] = -5.0  # An unfunded borrow expense.
        ex.quote.data["$open"] = 5.0
        buy = order(20, Order.BUY)
        ex.deal_order(buy, position=position)
        self.assertEqual(buy.deal_amount, 18)
        self.assertEqual(buy.signed_partial_reason, "insufficient_free_cash")
        self.assertGreaterEqual(position.get_cash(), 0)
        self.assertEqual(position.get_stock_amount("AAA"), 8)

    def test_long_sale_can_remedy_cash_deficit_and_cross_zero(self):
        ex = exchange()
        position = SignedPosition(cash=100.1)
        ex.deal_order(order(10, Order.BUY), position=position)
        position.position["cash"] = -20.0
        sell = order(20, Order.SELL)
        ex.deal_order(sell, position=position)
        self.assertEqual(sell.deal_amount, 20)
        self.assertAlmostEqual(position.get_cash(), 79.9)
        self.assertEqual(position.get_stock_amount("AAA"), -10)
        self.assertEqual(position.restricted_proceeds["AAA"], 100)

    def test_missing_open_never_uses_close(self):
        ex = exchange()
        ex.quote.data["$open"] = None
        position = SignedPosition(cash=100)
        buy = order(5, Order.BUY)
        value, cost, price = ex.deal_order(buy, position=position)
        self.assertEqual((value, cost, buy.deal_amount), (0, 0, 0))
        self.assertTrue(math.isnan(price))
        self.assertEqual(buy.signed_skip_reason, "missing_or_invalid_open")

    def test_limits_and_bad_amount(self):
        ex = exchange()
        position = SignedPosition(cash=100)
        ex.quote.data["limit_buy"] = True
        buy = order(1, Order.BUY)
        ex.deal_order(buy, position=position)
        self.assertEqual(buy.signed_skip_reason, "suspended_or_limited")
        for amount in (-1, math.inf, math.nan, 1.5):
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                ex.deal_order(order(amount, Order.BUY), position=position)
        with self.assertRaises(TypeError):
            ex.deal_order(order(1, Order.BUY), position=object())

    def test_signed_target_sizing_preserves_weights_and_skips_bad_open(self):
        ex = exchange()
        ex.quote.get_all_stock = lambda: {"AAA", "BBB"}
        ex.quote.get_data = lambda stock_id, start, end, field, method="ts_data_last": (
            {"$open": 10.0 if stock_id == "AAA" else None,
             "$close": 10.0, "$factor": 1.0,
             "limit_buy": False, "limit_sell": False}[field]
        )
        self.assertEqual(ex.generate_amount_position_from_weight_position(
            {"AAA": -0.3, "BBB": 0.2}, 100, DAY, NEXT), {"AAA": -3})
        with self.assertRaises(ValueError):
            ex.generate_amount_position_from_weight_position({"AAA": -1.1}, 100, DAY, NEXT)


if __name__ == "__main__":
    unittest.main()
