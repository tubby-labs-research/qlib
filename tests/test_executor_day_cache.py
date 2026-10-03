"""Generic native executor day-cache checks; no provider data or strategy needed."""

from collections import defaultdict
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from qlib.backtest.decision import Order
from qlib.backtest.executor import SimulatorExecutor


def uncached_collect(executor, decision):
    """Original per-order day/reset/fill behavior as a differential oracle."""
    executor.trade_calendar.get_step_time()
    result = []
    for order in executor._get_order_iterator(decision):
        day = executor.trade_calendar.get_step_time()[0].floor(freq="D")
        if executor.deal_day is None or day > executor.deal_day:
            executor.dealt_order_amount = defaultdict(float)
            executor.deal_day = day
        value, cost, price = executor.trade_exchange.deal_order(
            order, trade_account=executor.trade_account, dealt_order_amount=executor.dealt_order_amount
        )
        result.append((order, value, cost, price))
        executor.dealt_order_amount[order.stock_id] += order.deal_amount
    return result, {"trade_info": result}


class Calendar:
    def __init__(self, times):
        self.times = times
        self.calls = 0

    def get_step_time(self):
        value = self.times[min(self.calls, len(self.times) - 1)]
        self.calls += 1
        return value, value


class CustomTime:
    """A calendar value whose floor has observable mutable behavior."""
    def __init__(self, days):
        self.days = days
        self.calls = 0

    def floor(self, freq):
        assert freq == "D"
        result = self.days[min(self.calls, len(self.days) - 1)]
        self.calls += 1
        return result


class Exchange:
    def __init__(self, after_fill=None):
        self.seen = []
        self.amount_dicts = []
        self.after_fill = after_fill

    def deal_order(self, order, *, trade_account, dealt_order_amount):
        self.seen.append((order.stock_id, dict(dealt_order_amount)))
        self.amount_dicts.append(dealt_order_amount)
        # Partial fills depend on prior same-day volume, and update the order.
        order.deal_amount = max(0, min(order.amount, 6 - dealt_order_amount[order.stock_id]))
        order.factor = 1.25
        value = order.deal_amount * 10
        cost = value * 0.001
        trade_account.cash += (-value if order.direction == Order.BUY else value) - cost
        if self.after_fill is not None:
            self.after_fill(len(self.seen))
        return value, cost, 10.0


def fixture(times, day=None, trade_type=SimulatorExecutor.TT_SERIAL, after_fill=None):
    calendar = Calendar(times)
    executor = SimulatorExecutor.__new__(SimulatorExecutor)
    executor.level_infra = {"trade_calendar": calendar}
    executor._trade_exchange = Exchange(after_fill)
    executor.trade_account = SimpleNamespace(cash=1000.0)
    executor.trade_type = trade_type
    executor.verbose = False
    executor.deal_day = day
    executor.dealt_order_amount = defaultdict(float, {"AAA": 7.0})
    return executor


def decision(count=4):
    day = pd.Timestamp("2024-01-02")
    orders = [Order("AAA" if i != 2 else "BBB", 4, Order.BUY if i % 2 else Order.SELL, day, day)
              for i in range(count)]
    return SimpleNamespace(get_decision=lambda: orders)


def snapshot(executor, result):
    fills, kwargs = result
    assert kwargs["trade_info"] is fills
    dicts = executor.trade_exchange.amount_dicts
    reset_pattern = [i == 0 or item is not dicts[i - 1] for i, item in enumerate(dicts)]
    return (
        [(o.stock_id, o.direction, o.amount, o.deal_amount, o.factor, v, c, p) for o, v, c, p in fills],
        executor.trade_exchange.seen, reset_pattern, dict(executor.dealt_order_amount),
        executor.deal_day, executor.trade_account.cash, executor.trade_calendar.calls,
    )


class ExecutorDayCacheTests(unittest.TestCase):
    def differential(self, times_factory, day=None, count=4, trade_type=SimulatorExecutor.TT_SERIAL):
        actual = fixture(times_factory(), day, trade_type)
        reference = fixture(times_factory(), day, trade_type)
        result = actual._collect_data(decision(count))
        expected = uncached_collect(reference, decision(count))
        self.assertIs(type(actual), SimulatorExecutor)
        self.assertEqual(snapshot(actual, result), snapshot(reference, expected))
        self.assertEqual(actual.trade_calendar.calls, count + 1)
        return actual, result

    def test_repeated_identity_preserves_order_and_partial_fills(self):
        day = pd.Timestamp("2024-01-02 09:30")
        executor, result = self.differential(lambda: [day] * 5)
        self.assertEqual([o.deal_amount for o, *_ in result[0]], [4, 2, 4, 0])
        self.assertEqual(dict(executor.dealt_order_amount), {"AAA": 6.0, "BBB": 4.0})
        self.assertEqual(executor.trade_exchange.seen[1], ("AAA", {"AAA": 4.0}))

    def test_recreated_equal_timestamps_use_original_floor(self):
        self.differential(lambda: [pd.Timestamp("2024-01-02 09:30") for _ in range(5)])

    def test_changed_calendar_day_mid_fill_resets_at_that_order(self):
        first, later = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")
        executor = fixture([first] * 5)
        executor.trade_exchange.after_fill = lambda _: executor.trade_calendar.times.__setitem__(
            slice(None), [later] * 5
        )
        result = executor._collect_data(decision())
        self.assertEqual([o.deal_amount for o, *_ in result[0]], [4, 4, 4, 2])
        self.assertEqual(executor.deal_day, later)
        self.assertIsNot(executor.trade_exchange.amount_dicts[0], executor.trade_exchange.amount_dicts[1])
        self.assertEqual(executor.trade_calendar.calls, 5)

    def test_intraday_and_earlier_day_never_reset_but_later_day_does(self):
        times = [pd.Timestamp(t) for t in (
            "2024-01-02 08:00", "2024-01-02 09:30", "2024-01-02 15:00",
            "2024-01-01 12:00", "2024-01-03 09:30")]
        executor, _ = self.differential(lambda: times, day=pd.Timestamp("2024-01-02"))
        self.assertEqual(executor.trade_exchange.seen[:3], [("AAA", {"AAA": 7.0})] * 2 +
                         [("BBB", {"AAA": 7.0})])
        self.assertEqual(executor.trade_exchange.seen[3], ("AAA", {}))

    def test_empty_decision_preserves_day_and_volume_identity_without_floor(self):
        custom = CustomTime([pd.Timestamp("2024-01-03")])
        executor = fixture([custom], day=pd.Timestamp("2024-01-02"))
        volumes = executor.dealt_order_amount
        self.assertEqual(executor._collect_data(decision(0)), ([], {"trade_info": []}))
        self.assertIs(executor.dealt_order_amount, volumes)
        self.assertEqual(executor.deal_day, pd.Timestamp("2024-01-02"))
        self.assertEqual(custom.calls, 0)
        self.assertEqual(executor.trade_calendar.calls, 1)

    def test_custom_mutable_floor_keeps_every_call(self):
        days = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
        executor, _ = self.differential(lambda: [CustomTime(days)] * 5)
        self.assertEqual(executor.trade_calendar.times[0].calls, 4)

    def test_native_cache_is_invalidated_by_intervening_custom_value(self):
        day = pd.Timestamp("2024-01-02 09:30")
        self.differential(lambda: [day, day, CustomTime([day.floor("D")]), day, day])

    def test_timezone_aware_fallback_and_parallel_ordering(self):
        day = pd.Timestamp("2024-03-11 09:30", tz="America/New_York")
        self.differential(lambda: [day] * 5, trade_type=SimulatorExecutor.TT_PARAL)

    def test_cache_is_local_to_each_collect_call(self):
        day = pd.Timestamp("2024-01-02")
        executor = fixture([day] * 5)
        with mock.patch.object(np, "isclose", wraps=np.isclose) as calls:
            executor._collect_data(decision())
            first = calls.call_count
            executor.trade_calendar.calls = 0
            executor._collect_data(decision())
            self.assertGreater(first, 0)
            self.assertEqual(calls.call_count, first * 2)
        self.assertEqual(executor.trade_calendar.calls, 5)
        self.assertEqual(len(executor.trade_exchange.seen), 8)

    def test_repeated_native_timestamp_reduces_floor_numpy_calls(self):
        # Pandas' floor uses np.isclose for its nanosecond conversion. Count that
        # work without replacing the immutable native Timestamp type or method.
        native_isclose = np.isclose
        for same_object, timezone in ((True, None), (False, None), (True, "UTC")):
            with self.subTest(same_object=same_object, timezone=timezone):
                day = pd.Timestamp("2024-01-02 09:30", tz=timezone)
                times = [day] * 5 if same_object else [pd.Timestamp(day.isoformat()) for _ in range(5)]
                with mock.patch.object(np, "isclose", wraps=native_isclose) as calls:
                    uncached_collect(fixture(times), decision())
                    baseline = calls.call_count
                with mock.patch.object(np, "isclose", wraps=native_isclose) as calls:
                    fixture(times)._collect_data(decision())
                    optimized = calls.call_count
                self.assertGreater(baseline, 0, "Pandas floor instrumentation must remain observable")
                self.assertEqual(optimized, baseline // 4 if same_object and timezone is None else baseline)


if __name__ == "__main__":
    unittest.main()
