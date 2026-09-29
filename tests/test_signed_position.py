"""Known-answer and safety checks for the signed position ledger."""

import copy
import json
import math
import unittest
from pathlib import Path

from qlib.backtest.decision import Order
from qlib.backtest.signed_position import SignedPosition


FIXTURES = Path(__file__).parent / "fixtures" / "signed_account"


def fill(position, symbol, delta, price, fee=0.0005):
    value = abs(delta * price)
    cost = value * fee
    order = Order(symbol, abs(delta), Order.BUY if delta > 0 else Order.SELL, None, None)
    position.update_order(order, value, cost, price)


class SignedPositionTests(unittest.TestCase):
    def test_restricted_cash_and_atomic_refusal(self):
        position = SignedPosition(10)
        fill(position, "S", -10, 10, 0)
        self.assertEqual(position.get_cash(), 10)
        self.assertEqual(position.get_total_cash(), 110)
        before = copy.deepcopy(position.__dict__)
        self.assertEqual(position.preview_fill("L", 2, 10, 0), -10)
        with self.assertRaisesRegex(ValueError, "insufficient free cash"):
            fill(position, "L", 2, 10, 0)
        self.assertEqual(position.__dict__, before)
        self.assertEqual(position.preview_fill("S", 5, 8, 0), 20)
        self.assertEqual(position.__dict__, before)

    def test_reversal_and_original_proceeds(self):
        position = SignedPosition(1000)
        fill(position, "S", -4, 10, 0)
        fill(position, "S", -6, 20, 0)
        self.assertEqual(position.restricted_proceeds["S"], 160)
        fill(position, "S", 5, 15, 0)
        self.assertEqual(position.restricted_proceeds["S"], 80)
        self.assertEqual(position.get_cash(), 1005)
        fill(position, "S", 7, 12, 0)
        self.assertEqual(position.get_stock_amount("S"), 2)
        self.assertEqual(position.get_cash(), 1001)
        self.assertEqual(position.get_total_cash(), 1001)

    def test_borrow_deficit_and_weights(self):
        position = SignedPosition(1)
        fill(position, "S", -100, 1, 0)
        position.update_stock_price("S", 100)
        self.assertEqual(position.accrue_borrow(365), 40)
        self.assertEqual(position.snapshot()["funding_shortfall"], 39)
        self.assertEqual(position.calculate_value(), -9939)
        before = copy.deepcopy(position.__dict__)
        with self.assertRaisesRegex(ValueError, "insufficient free cash"):
            fill(position, "S", -1, 1, 0)
        self.assertEqual(position.__dict__, before)
        zero = SignedPosition(100)
        fill(zero, "L", 1, 10, 0)
        fill(zero, "S", -1, 10, 0)
        with self.assertRaisesRegex(ValueError, "zero net stock exposure"):
            zero.get_stock_weight_dict(only_stock=True)
        self.assertEqual(zero.get_stock_weight_dict(), {"L": 0.1, "S": -0.1})

    def test_invalid_inputs_and_empty_equity(self):
        with self.assertRaises(ValueError):
            SignedPosition(1, {"S": {"amount": -1, "price": 1}})
        position = SignedPosition(0)
        with self.assertRaises(ValueError):
            position.preview_fill("S", -1, math.inf, 0)
        with self.assertRaises(ValueError):
            position.accrue_borrow(1.5)
        with self.assertRaises(NotImplementedError):
            position.settle_start(position.ST_CASH)
        fill(position, "S", -1, 1, 0)
        self.assertTrue(math.isnan(position.get_stock_weight_dict()["S"]))

    def test_literal_fixtures(self):
        paths = sorted(FIXTURES.glob("*.json"))
        self.assertEqual({p.name for p in paths}, {
            "pooled_short_proceeds.json", "short_partial.json", "reversals.json", "weekend_and_deficit.json"})
        event_count = 0
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            for case in data.get("cases", [data]):
                with self.subTest(path=path.name, case=case.get("initial_free_cash")):
                    position = SignedPosition(float(case["initial_free_cash"]))
                    for event in case["events"]:
                        event_count += 1
                        if "fill" in event:
                            symbol, delta, price, *fees = event["fill"]
                            fill(position, symbol, float(delta), float(price), float(fees[0]) if fees else 0.0005)
                        elif "mark" in event:
                            for symbol, price in event["mark"].items():
                                position.update_stock_price(symbol, float(price))
                        elif "borrow" in event:
                            position.accrue_borrow(int(event["borrow"]))
                        for key, expected in event["expected"].items():
                            actual = position.snapshot()[key]
                            if key == "positions":
                                self.assertEqual(set(actual), set(expected))
                                for symbol, amount in expected.items():
                                    self.assertAlmostEqual(actual[symbol], float(amount), places=8)
                            else:
                                self.assertAlmostEqual(actual, float(expected), places=8, msg=f"{path.name}: {key}")
        self.assertEqual(event_count, 26)

    def test_fractional_price_does_not_create_fractional_share_dust(self):
        position = SignedPosition(100)
        fill(position, "S", -7, 0.1, 0)
        self.assertEqual(position.get_stock_amount("S"), -7)
        fill(position, "S", 7, 0.3, 0)
        self.assertEqual(position.get_stock_amount_dict(), {})
        self.assertEqual(position.restricted_proceeds, {})


if __name__ == "__main__":
    unittest.main()
