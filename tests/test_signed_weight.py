"""Focused daily-open decisions for already-normalized signed weights."""

import unittest

import pandas as pd

from qlib.backtest.decision import Order
from qlib.backtest.signal import Signal
from qlib.backtest.signed_exchange import SignedExchange
from qlib.backtest.signed_position import SignedPosition
from qlib.contrib.strategy.signed_weight import SignedWeightStrategy


DATES = pd.date_range("2024-01-02", periods=3, freq="B")


class SessionSignal(Signal):
    def __init__(self, values):
        self.values = values
        self.queries = []

    def get_signal(self, start_time, end_time):
        self.queries.append((start_time, end_time))
        return self.values.get(start_time)


class Calendar:
    def __init__(self, step=1, start_index=0):
        self.step = step
        self.start_index = start_index

    def get_trade_step(self):
        return self.step

    def get_step_time(self, step=None, shift=0):
        if step is None:
            step = self.step
        index = self.start_index + step - shift
        if index < 0:
            raise AssertionError("negative calendar lookup")
        return DATES[index], DATES[index] + pd.Timedelta(hours=23)


def make_strategy(values, *, step=1, cash=100.0, opens=None, unavailable=()):
    signal = SessionSignal(values)
    position = SignedPosition(cash=cash)
    exchange = SignedExchange.__new__(SignedExchange)
    opens = opens or {"AAA": 10.0, "BBB": 20.0}
    exchange.is_stock_tradable = lambda symbol, start, end: symbol not in unavailable
    exchange.get_deal_price = lambda symbol, start, end, direction: opens.get(symbol)
    exchange.get_factor = lambda symbol, start, end: 1.0
    calendar = Calendar(step=step)
    strategy = SignedWeightStrategy(
        signal=signal,
        level_infra={"trade_calendar": calendar},
        common_infra={"trade_account": type("AccountStub", (), {"current_position": position})()},
        trade_exchange=exchange,
    )
    return strategy, signal, position, exchange, calendar


def order_summary(decision):
    return [(o.stock_id, o.amount, o.direction) for o in decision.get_decision()]


class SignedWeightTests(unittest.TestCase):
    def test_prior_session_open_sizing_and_rounding(self):
        weights = pd.Series({"AAA": 0.5, "BBB": -0.3})
        strategy, signal, _, _, _ = make_strategy({DATES[0]: weights}, opens={"AAA": 12.0, "BBB": 9.0})
        self.assertEqual(order_summary(strategy.generate_trade_decision()),
                         [("AAA", 4, Order.BUY), ("BBB", 3, Order.SELL)])
        self.assertEqual(signal.queries, [(DATES[0], DATES[0] + pd.Timedelta(hours=23))])
        self.assertEqual(strategy.decisions[0]["target_quantities"], {"AAA": 4, "BBB": -3})
        self.assertEqual(strategy.decisions[0]["sizing_equity"], 100.0)

    def test_first_calendar_session_does_not_wrap(self):
        strategy, signal, _, _, _ = make_strategy({DATES[2]: pd.Series({"AAA": 1.0})}, step=0)
        self.assertEqual(order_summary(strategy.generate_trade_decision()), [])
        self.assertEqual(signal.queries, [])
        self.assertEqual(strategy.decisions[0]["reasons"]["session"], "no_preceding_calendar_session")

    def test_missing_signal_is_not_zero_target(self):
        strategy, _, position, _, _ = make_strategy({})
        position._init_stock("AAA", 5, 10.0)
        self.assertEqual(order_summary(strategy.generate_trade_decision()), [])
        self.assertIsNone(strategy.decisions[0]["target_weights"])
        self.assertEqual(strategy.decisions[0]["reasons"]["session"], "missing_signal")

    def test_explicit_zero_liquidates_and_reversals_close_first(self):
        strategy, signal, position, _, calendar = make_strategy({DATES[0]: pd.Series({"AAA": -0.4, "BBB": 0.2}),
                                                                DATES[1]: pd.Series({"AAA": 0.0, "BBB": 0.0})})
        position._init_stock("AAA", 5, 10.0)
        position._init_stock("BBB", -4, 20.0)
        position.restricted_proceeds["BBB"] = 80.0
        first = order_summary(strategy.generate_trade_decision())
        self.assertEqual(first[:2], [("AAA", 5, Order.SELL), ("BBB", 4, Order.BUY)])
        self.assertEqual(first[2:], [("AAA", 6, Order.SELL), ("BBB", 1, Order.BUY)])
        calendar.step = 2
        self.assertEqual(order_summary(strategy.generate_trade_decision()),
                         [("AAA", 5, Order.SELL), ("BBB", 4, Order.BUY)])
        self.assertEqual(strategy.decisions[1]["target_weights"], {"AAA": 0.0, "BBB": 0.0})
        self.assertEqual(len(signal.queries), 2)

    def test_missing_open_and_untradable_keep_positions(self):
        strategy, _, position, _, _ = make_strategy({DATES[0]: pd.Series({"AAA": 0.5, "BBB": -0.5})},
                                                   opens={"AAA": None, "BBB": 20.0})
        position._init_stock("AAA", 2, 10.0)
        self.assertEqual(order_summary(strategy.generate_trade_decision()), [("BBB", 3, Order.SELL)])
        self.assertEqual(strategy.decisions[0]["reasons"]["AAA"], "missing_or_invalid_open")
        self.assertNotIn("AAA", strategy.decisions[0]["target_quantities"])
        strategy2, _, _, _, _ = make_strategy({DATES[0]: pd.Series({"AAA": 0.5, "BBB": -0.5})}, unavailable={"BBB"})
        self.assertEqual(order_summary(strategy2.generate_trade_decision()), [("AAA", 5, Order.BUY)])
        self.assertEqual(strategy2.decisions[0]["reasons"]["BBB"], "unavailable_for_trade")

    def test_nonpositive_equity_keeps_existing_exposure(self):
        strategy, _, position, _, _ = make_strategy({DATES[0]: pd.Series({"AAA": 0.0})}, cash=1.0)
        position._init_stock("AAA", -2, 10.0)
        position.restricted_proceeds["AAA"] = 10.0
        self.assertEqual(order_summary(strategy.generate_trade_decision()), [])
        self.assertEqual(strategy.decisions[0]["reasons"]["session"], "nonpositive_equity")

    def test_invalid_weights_and_wrong_types(self):
        for values in (pd.Series([0.1, 0.2], index=["AAA", "AAA"]),
                       pd.Series({"AAA": float("nan")}), pd.Series({"AAA": 1.1}),
                       pd.Series({"AAA": "0.2"})):
            strategy, _, _, _, _ = make_strategy({DATES[0]: values})
            with self.assertRaises(ValueError):
                strategy.generate_trade_decision()
        strategy, _, _, _, _ = make_strategy({DATES[0]: pd.Series({"AAA": 0.5})})
        strategy._trade_exchange = object()
        with self.assertRaises(TypeError):
            strategy.generate_trade_decision()


if __name__ == "__main__":
    unittest.main()
