# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Daily-open exchange for the signed, restricted-proceeds research account."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .decision import Order, OrderDir
from .exchange import Exchange


class SignedExchange(Exchange):
    """Execute integer-share signed orders against a SignedPosition ledger."""

    def __init__(
        self,
        *args,
        freq="day",
        deal_price="$open",
        open_cost=0.0005,
        close_cost=0.0005,
        min_cost=0.0,
        impact_cost=0.0,
        volume_threshold=None,
        trade_unit=1,
        limit_threshold=None,
        **kwargs,
    ):
        if freq != "day" or deal_price not in ("$open", "open") or trade_unit != 1:
            raise ValueError("SignedExchange requires daily $open execution and trade_unit=1")
        if volume_threshold is not None or min_cost != 0 or impact_cost != 0:
            raise NotImplementedError("SignedExchange does not support volume caps, minimum fees, or impact")
        if limit_threshold is not None:
            raise NotImplementedError("daily-close-derived price limits are unsupported for open execution")
        if open_cost != close_cost:
            raise NotImplementedError("signed mode currently requires the same fee rate on both trade sides")
        for rate in (open_cost, close_cost):
            if not math.isfinite(rate) or rate < 0:
                raise ValueError("transaction cost rates must be finite and nonnegative")
        super().__init__(
            *args, freq=freq, deal_price="$open", open_cost=open_cost,
            close_cost=close_cost, min_cost=min_cost, impact_cost=impact_cost,
            volume_threshold=volume_threshold, trade_unit=trade_unit,
            limit_threshold=None, **kwargs,
        )

    @staticmethod
    def _require_position(position):
        from .signed_position import SignedPosition

        if not isinstance(position, SignedPosition):
            raise TypeError("SignedExchange requires SignedPosition")
        return position

    @staticmethod
    def _validate_amount(order):
        amount = order.amount
        if isinstance(amount, (bool, complex)) or not isinstance(amount, (int, float, np.integer, np.floating)):
            raise ValueError("order amount must be a finite nonnegative integer-share quantity")
        if not math.isfinite(amount) or amount < 0 or amount != math.floor(amount):
            raise ValueError("order amount must be a finite nonnegative integer-share quantity")

    def deal_order(self, order, trade_account=None, position=None, dealt_order_amount=None):
        self._validate_amount(order)
        if trade_account is not None and position is not None:
            raise ValueError("trade_account and position can only choose one")
        if trade_account is not None:
            from .signed_account import SignedAccount
            if not isinstance(trade_account, SignedAccount):
                raise TypeError("SignedExchange requires SignedAccount for account execution")
        current = self._require_position(trade_account.current_position if trade_account is not None else position)
        if dealt_order_amount is None:
            dealt_order_amount = {}
        if not self.check_order(order):
            order.deal_amount = 0.0
            return 0.0, 0.0, np.nan
        price, value, cost = self._calc_trade_info_by_order(order, current, dealt_order_amount)
        # Unlike the legacy exchange, do not report a positive tiny fill while
        # omitting its ledger update because notional happens to be <= 1e-5.
        if order.deal_amount > 0:
            if trade_account is not None:
                trade_account.update_order(order, value, cost, price)
            else:
                current.update_order(order, value, cost, price)
        return value, cost, price

    def check_order(self, order):
        tradable = super().check_order(order)
        if not tradable:
            order.signed_skip_reason = ("missing_or_invalid_open" if self.check_stock_suspended(
                order.stock_id, order.start_time, order.end_time) else "suspended_or_limited")
        return tradable

    def check_stock_suspended(self, stock_id, start_time, end_time):
        # Today's closing quote is not available when an opening order is sized.
        price = self.get_deal_price(stock_id, start_time, end_time, OrderDir.BUY)
        return price is None or not np.isscalar(price) or not math.isfinite(price) or price <= 0

    def _update_limit(self, limit_threshold):
        # Upstream also folds missing CLOSE into its limit flags, even with no
        # price limit. Do not let a future close decide an opening fill.
        if limit_threshold is not None:
            raise NotImplementedError("signed open execution does not support daily price-limit expressions")
        suspended = self.quote_df["$open"].isna() | (self.quote_df["$open"] <= 0)
        self.quote_df["limit_buy"] = suspended
        self.quote_df["limit_sell"] = suspended

    def get_deal_price(self, stock_id, start_time, end_time, direction, method="ts_data_last"):
        if direction not in (OrderDir.BUY, OrderDir.SELL):
            raise ValueError("unsupported order direction")
        if self.buy_price != "$open" or self.sell_price != "$open":
            raise ValueError("SignedExchange requires $open on both sides")
        return self.quote.get_data(stock_id, start_time, end_time, field="$open", method=method)

    def _calc_trade_info_by_order(self, order, position, dealt_order_amount):
        self._validate_amount(order)
        position = self._require_position(position)
        order.deal_amount = 0.0
        order.signed_skip_reason = None
        order.signed_partial_reason = None
        price = self.get_deal_price(order.stock_id, order.start_time, order.end_time, order.direction)
        if price is None or not np.isscalar(price) or not math.isfinite(price) or price <= 0:
            order.signed_skip_reason = "missing_or_invalid_open"
            return np.nan, 0.0, 0.0
        factor = self.get_factor(order.stock_id, order.start_time, order.end_time)
        if factor is None or not math.isfinite(factor) or factor != 1:
            raise NotImplementedError("SignedExchange requires factor=1")
        order.factor = factor
        requested = int(order.amount)
        if requested == 0:
            order.signed_skip_reason = "zero_amount"
            return float(price), 0.0, 0.0

        sign = 1 if order.direction == OrderDir.BUY else -1
        old_quantity = position.get_stock_amount(order.stock_id) if position.check_stock(order.stock_id) else 0
        if not math.isfinite(old_quantity) or old_quantity != math.floor(old_quantity):
            raise ValueError("signed position quantity must be finite integer shares")
        close_bound = min(requested, max(0, -sign * int(old_quantity)))
        rate = self.open_cost if sign == 1 else self.close_cost

        def feasible(quantity):
            value = quantity * price
            prospective = position.preview_fill(order.stock_id, sign * quantity, price, value * rate)
            if not math.isfinite(prospective):
                raise ValueError("preview_fill returned nonfinite free cash")
            return prospective >= 0.0

        # Free cash is affine on either side of the old-position close. It can
        # rise while selling a long or covering a profitable short, then fall.
        # Search the upper segment first so the result is the largest feasible fill.
        intervals = [(close_bound + 1, requested), (1, close_bound)]
        quantity = 0
        for low, high in intervals:
            if low > high:
                continue
            if feasible(high):
                quantity = high
                break
            if not feasible(low):
                continue
            while low < high:
                mid = (low + high + 1) // 2
                if feasible(mid):
                    low = mid
                else:
                    high = mid - 1
            quantity = low
            break

        if quantity == 0:
            order.signed_skip_reason = "insufficient_free_cash"
            return float(price), 0.0, 0.0
        order.deal_amount = float(quantity)
        if quantity < requested:
            order.signed_partial_reason = "insufficient_free_cash"
        value = quantity * price
        return float(price), float(value), float(value * rate)

    def generate_amount_position_from_weight_position(
        self, weight_position: dict, cash: float, start_time, end_time,
        direction: OrderDir = OrderDir.BUY,
    ) -> dict:
        """Size signed target shares from equity at this session's open."""
        if not math.isfinite(cash) or cash <= 0:
            raise ValueError("positive finite equity is required for target sizing")
        if direction not in (OrderDir.BUY, OrderDir.SELL):
            raise ValueError("unsupported sizing direction")
        if any(not isinstance(w, (int, float, np.integer, np.floating)) or not math.isfinite(w)
               for w in weight_position.values()):
            raise ValueError("weights must be finite numbers")
        if math.fsum(abs(float(w)) for w in weight_position.values()) > 1 + 1e-12:
            raise ValueError("gross target weight exceeds one")
        amounts = {}
        for stock_id, weight in weight_position.items():
            side = OrderDir.BUY if weight >= 0 else OrderDir.SELL
            if not self.is_stock_tradable(stock_id, start_time, end_time, side):
                continue
            price = self.get_deal_price(stock_id, start_time, end_time, side)
            if price is None or not np.isscalar(price) or not math.isfinite(price) or price <= 0:
                continue
            factor = self.get_factor(stock_id, start_time, end_time)
            if factor is None or not math.isfinite(factor) or factor != 1:
                raise NotImplementedError("SignedExchange requires factor=1")
            amounts[stock_id] = math.copysign(math.floor(abs(weight) * cash / price), weight) if weight else 0
        return {stock_id: int(quantity) for stock_id, quantity in amounts.items()}
