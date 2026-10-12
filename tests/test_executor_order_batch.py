"""Generic opt-in dispatch checks against the installed native executor."""

from types import SimpleNamespace
import unittest
from unittest import mock

import pandas as pd

from qlib.backtest.executor import BaseExecutor, SimulatorExecutor
from test_executor_day_cache import CustomTime, decision, fixture, snapshot, uncached_collect


class OrderBatchExecutorTests(unittest.TestCase):
    def differential(self, times_factory, *, handler=None, trade_type=SimulatorExecutor.TT_SERIAL):
        actual = fixture(times_factory(), trade_type=trade_type)
        if handler is not None:
            actual._order_batch_executor = handler
        native = fixture(times_factory(), trade_type=trade_type)
        self.assertEqual(snapshot(actual, actual._collect_data(decision())),
                         snapshot(native, uncached_collect(native, decision())))
        self.assertEqual(actual.trade_calendar.calls, 5)
        return actual

    def test_old_new_and_explicit_none_instances_keep_native_loop(self):
        day = pd.Timestamp('2024-01-02 09:30')
        self.differential(lambda: [day] * 5)
        actual, native = fixture([day] * 5), fixture([day] * 5)
        actual._order_batch_executor = None
        self.assertEqual(snapshot(actual, actual._collect_data(decision())),
                         snapshot(native, uncached_collect(native, decision())))

    def test_fallback_calls_handler_once_before_calendar_and_preserves_order(self):
        seen = []
        def fallback(executor, supplied):
            self.assertEqual(executor.trade_calendar.calls, 0)
            self.assertEqual(executor.trade_exchange.seen, [])
            self.assertEqual(dict(executor.dealt_order_amount), {'AAA': 7.0})
            self.assertEqual(executor.trade_account.cash, 1000.)
            self.assertTrue(all(o.deal_amount == 0 for o in supplied.get_decision()))
            seen.append((executor, supplied))
            return NotImplemented
        day = pd.Timestamp('2024-01-02')
        handler = SimpleNamespace(try_collect=fallback)
        actual = self.differential(lambda: [day] * 5, handler=handler)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0][0], actual)
        self.assertEqual([c for c, _ in actual.trade_exchange.seen], ['AAA', 'AAA', 'BBB', 'AAA'])

    def test_fallback_keeps_custom_floor_and_parallel_sequence(self):
        handler = SimpleNamespace(try_collect=lambda executor, supplied: NotImplemented)
        days = [pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-03')]
        self.differential(lambda: [CustomTime(days)] * 5, handler=handler)
        day = pd.Timestamp('2024-03-11 09:30', tz='America/New_York')
        self.differential(lambda: [day] * 5, handler=handler,
                          trade_type=SimulatorExecutor.TT_PARAL)

    def test_success_returns_identity_without_native_double_fills(self):
        day = pd.Timestamp('2024-01-02')
        executor = fixture([day] * 5)
        supplied = decision()
        fills = [(supplied.get_decision()[0], 4., .1, 1.)]
        result = fills, {'trade_info': fills}
        callback = mock.Mock(return_value=result)
        executor._order_batch_executor = SimpleNamespace(try_collect=callback)
        self.assertIs(executor._collect_data(supplied), result)
        callback.assert_called_once_with(executor, supplied)
        self.assertIs(result[1]['trade_info'], fills)
        self.assertEqual(executor.trade_calendar.calls, 0)
        self.assertEqual(executor.trade_exchange.seen, [])
        self.assertEqual(executor.trade_account.cash, 1000.)
        self.assertEqual(dict(executor.dealt_order_amount), {'AAA': 7.0})
        self.assertIsNone(executor.deal_day)

    def test_handler_errors_propagate_without_fallback(self):
        executor = fixture([pd.Timestamp('2024-01-02')] * 5)
        error = RuntimeError('handler failed')
        executor._order_batch_executor = SimpleNamespace(try_collect=mock.Mock(side_effect=error))
        with self.assertRaises(RuntimeError) as raised:
            executor._collect_data(decision())
        self.assertIs(raised.exception, error)
        self.assertEqual(executor.trade_calendar.calls, 0)
        self.assertEqual(executor.trade_exchange.seen, [])

    def test_invalid_handler_rejected_before_base_constructor_actions(self):
        for handler in (object(), SimpleNamespace(try_collect=None), SimpleNamespace(try_collect=1)):
            with self.subTest(handler=handler), mock.patch.object(BaseExecutor, '__init__', return_value=None) as base:
                with self.assertRaisesRegex(TypeError, 'callable try_collect'):
                    SimulatorExecutor(time_per_step='day', order_batch_executor=handler)
                base.assert_not_called()

    def test_constructor_preserves_default_and_stores_valid_handler(self):
        for handler in (None, SimpleNamespace(try_collect=lambda executor, supplied: NotImplemented)):
            with self.subTest(handler=handler), mock.patch.object(BaseExecutor, '__init__', return_value=None) as base:
                executor = SimulatorExecutor(time_per_step='day', order_batch_executor=handler)
                self.assertIs(executor._order_batch_executor, handler)
                self.assertEqual(executor.trade_type, SimulatorExecutor.TT_SERIAL)
                self.assertNotIn('order_batch_executor', base.call_args.kwargs)


if __name__ == '__main__':
    unittest.main()
