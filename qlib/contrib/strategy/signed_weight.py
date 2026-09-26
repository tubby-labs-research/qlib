# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Turn prior-session signed target weights into daily-open share orders."""

from __future__ import annotations

import math
from numbers import Real

import pandas as pd

from qlib.backtest.decision import Order, TradeDecisionWO
from qlib.backtest.signed_exchange import SignedExchange
from qlib.backtest.signed_position import SignedPosition
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy


class SignedWeightStrategy(BaseSignalStrategy):
    """Consume already-normalized weights; execution and affordability belong to the exchange.

    A signal is a Series indexed by (datetime, instrument). The Qlib signal
    adapter returns the preceding session as an instrument-indexed Series.
    Each item in ``decisions`` records the intended targets and issued orders,
    even if a later exchange fill is partial or skipped.
    """

    def __init__(self, *, signal, **kwargs):
        if isinstance(signal, pd.Series):
            if not isinstance(signal.index, pd.MultiIndex) or set(signal.index.names) != {"datetime", "instrument"}:
                raise ValueError("signal must be indexed by datetime and instrument")
            if not signal.index.is_unique:
                raise ValueError("duplicate datetime/instrument signal labels")
        super().__init__(signal=signal, **kwargs)
        self.decisions = []

    @staticmethod
    def _weights(values):
        if not isinstance(values, pd.Series) or isinstance(values.index, pd.MultiIndex):
            raise ValueError("signed weights must be an instrument-indexed Series")
        if not values.index.is_unique:
            raise ValueError("duplicate instrument labels")
        weights = {}
        for symbol, value in values.items():
            if not isinstance(symbol, str) or not symbol:
                raise ValueError("instrument labels must be nonempty strings")
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError("signed weights must be finite numeric values")
            weights[symbol] = float(value)
        if math.fsum(abs(weight) for weight in weights.values()) > 1 + 1e-12:
            raise ValueError("gross target weight exceeds one")
        return weights

    def generate_trade_decision(self, execute_result=None):
        position = self.trade_position
        exchange = self.trade_exchange
        if not isinstance(position, SignedPosition) or not isinstance(exchange, SignedExchange):
            raise TypeError("SignedWeightStrategy requires SignedPosition and SignedExchange")

        calendar = self.trade_calendar
        step = calendar.get_trade_step()
        start, end = calendar.get_step_time(step)
        record = {
            "session": start,
            "signal_session": None,
            "target_weights": None,
            "target_quantities": None,
            "sizing_equity": None,
            "reasons": {},
            "issued_orders": [],
        }
        self.decisions.append(record)

        # Qlib's get_step_time uses an array index and otherwise wraps at -1.
        if calendar.start_index + step < 1:
            record["reasons"]["session"] = "no_preceding_calendar_session"
            return TradeDecisionWO([], self)
        previous_start, previous_end = calendar.get_step_time(step, shift=1)
        record["signal_session"] = previous_start
        signal = self.signal.get_signal(start_time=previous_start, end_time=previous_end)
        if signal is None:
            record["reasons"]["session"] = "missing_signal"
            return TradeDecisionWO([], self)
        weights = self._weights(signal)
        holdings = position.get_stock_amount_dict()
        target_weights = {symbol: weights.get(symbol, 0.0) for symbol in sorted(weights.keys() | holdings.keys())}
        record["target_weights"] = target_weights

        # The account's pre-open hook must already have charged elapsed borrow.
        equity = position.calculate_value()
        if not math.isfinite(equity):
            raise ValueError("sizing equity must be finite")
        record["sizing_equity"] = equity
        if equity <= 0:
            record["reasons"]["session"] = "nonpositive_equity"
            return TradeDecisionWO([], self)

        targets = {}
        for symbol, weight in target_weights.items():
            if not exchange.is_stock_tradable(symbol, start, end):
                record["reasons"][symbol] = "unavailable_for_trade"
                continue
            price = exchange.get_deal_price(symbol, start, end, Order.BUY)
            if not isinstance(price, Real) or not math.isfinite(price) or price <= 0:
                record["reasons"][symbol] = "missing_or_invalid_open"
                continue
            factor = exchange.get_factor(symbol, start, end)
            if not isinstance(factor, Real) or not math.isfinite(factor) or factor != 1:
                raise NotImplementedError("SignedWeightStrategy requires factor=1")
            targets[symbol] = (1 if weight >= 0 else -1) * math.floor(abs(weight) * equity / price)
        record["target_quantities"] = targets.copy()

        reductions = []
        increases = []

        def make_order(symbol, amount, direction):
            return Order(symbol, amount, direction, start, end)

        for symbol, target in targets.items():
            current = holdings.get(symbol, 0)
            if not isinstance(current, Real) or not math.isfinite(current) or current != math.floor(current):
                raise ValueError("signed holdings must be finite whole shares")
            current = int(current)
            if current == target:
                continue
            if current and target and (current > 0) != (target > 0):
                reductions.append(make_order(symbol, abs(current), Order.SELL if current > 0 else Order.BUY))
                increases.append(make_order(symbol, abs(target), Order.BUY if target > 0 else Order.SELL))
            elif abs(target) < abs(current):
                reductions.append(make_order(symbol, abs(target - current), Order.BUY if target > current else Order.SELL))
            else:
                increases.append(make_order(symbol, abs(target - current), Order.BUY if target > current else Order.SELL))
        # Release cash from long sales before cash-consuming short covers.
        reductions.sort(key=lambda order: (order.direction == Order.BUY, order.stock_id))
        orders = reductions + increases
        record["issued_orders"] = orders.copy()
        return TradeDecisionWO(orders, self)
