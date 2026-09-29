# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Signed-share research ledger for the pinned Qlib position interface."""

from __future__ import annotations

import math
from typing import Dict, Union

from .decision import Order
from .position import Position


def _finite(value: float, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0 or nonnegative and result < 0:
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


class SignedPosition(Position):
    """Float ledger with signed holdings and restricted original short proceeds.

    Delayed settlement and pre-seeded positions are unsupported in this mode.
    A negative free-cash balance caused by later borrow expenses is retained;
    new fills may never consume unavailable cash.
    """

    def __init__(self, cash: float = 0, position_dict: Dict[str, Union[dict, float]] | None = None) -> None:
        cash = _finite(cash, "cash", nonnegative=True)
        if position_dict:
            raise ValueError("SignedPosition requires a cash-only initial position")
        self.restricted_proceeds: dict[str, float] = {}
        self.transaction_cost = 0.0
        self.borrow_cost = 0.0
        super().__init__(cash=cash, position_dict={})

    def _fill_effect(self, stock_id: str, signed_delta: float, price: float, cost: float) -> tuple[float, float, float]:
        if not isinstance(stock_id, str) or not stock_id or stock_id in {"cash", "cash_delay", "now_account_value"}:
            raise ValueError("stock_id must be a nonempty instrument identifier")
        delta = _finite(signed_delta, "signed_delta")
        price = _finite(price, "price", positive=True)
        cost = _finite(cost, "cost", nonnegative=True)
        if delta == 0 and cost != 0:
            raise ValueError("a zero-quantity fill cannot incur cost")
        old = self.get_stock_amount(stock_id)
        basis = self.restricted_proceeds.get(stock_id, 0.0)
        if delta > 0:
            covered = min(delta, max(-old, 0.0))
            released = basis * (covered / -old) if old < 0 else 0.0
            new_basis = basis - released
            free = self.get_cash() + released - delta * price - cost
        else:
            closed_long = min(-delta, max(old, 0.0))
            new_short = -delta - closed_long
            new_basis = basis + new_short * price
            free = self.get_cash() + closed_long * price - cost
        new_amount = old + delta
        if not all(math.isfinite(x) for x in (free, new_basis, new_amount)):
            raise ValueError("fill exceeds finite ledger range")
        return free, new_basis, new_amount

    def preview_fill(self, stock_id: str, signed_delta: float, price: float, cost: float) -> float:
        """Return prospective spendable cash without changing the ledger."""
        return self._fill_effect(stock_id, signed_delta, price, cost)[0]

    def update_order(self, order: Order, trade_val: float, cost: float, trade_price: float) -> None:
        if self._settle_type != self.ST_NO:
            raise NotImplementedError("SignedPosition does not support delayed settlement")
        price = _finite(trade_price, "trade_price", positive=True)
        value = _finite(trade_val, "trade_val", nonnegative=True)
        cost = _finite(cost, "cost", nonnegative=True)
        if order.direction not in (Order.BUY, Order.SELL):
            raise ValueError("unsupported order direction")
        quantity = _finite(value / price, "filled quantity", nonnegative=True)
        nearest = round(quantity)
        if abs(quantity - nearest) > 8 * math.ulp(quantity):
            raise ValueError("signed fills require whole shares")
        requested = _finite(order.amount, "order.amount", nonnegative=True)
        if nearest > requested:
            raise ValueError("fill exceeds requested quantity")
        if value == 0:
            if cost != 0:
                raise ValueError("a zero-quantity fill cannot incur cost")
            return
        delta = nearest * (1 if order.direction == Order.BUY else -1)
        if value > 0 and delta == 0:
            raise ValueError("fill quantity underflow")
        free, basis, amount = self._fill_effect(order.stock_id, delta, price, cost)
        if not math.isfinite(self.transaction_cost + cost):
            raise ValueError("transaction cost exceeds finite ledger range")
        # Only absorb machine cancellation, never an economically meaningful loan.
        tolerance = min(1e-8, 8 * max(math.ulp(self.get_cash()), math.ulp(value), math.ulp(cost)))
        if free < -tolerance:
            raise ValueError("insufficient free cash for fill")
        if free < 0:
            free = 0.0
        if amount == 0:
            self.position.pop(order.stock_id, None)
            self.restricted_proceeds.pop(order.stock_id, None)
        else:
            if order.stock_id not in self.position:
                self._init_stock(order.stock_id, amount, price)
            else:
                self.position[order.stock_id]["amount"] = amount
                self.position[order.stock_id]["price"] = price
            if amount < 0:
                self.restricted_proceeds[order.stock_id] = basis
            else:
                self.restricted_proceeds.pop(order.stock_id, None)
        self.position["cash"] = free
        self.transaction_cost += cost

    def accrue_borrow(self, days: int, annual_rate: float = 0.004) -> float:
        if isinstance(days, bool) or not isinstance(days, int) or days < 0:
            raise ValueError("days must be a nonnegative integer")
        rate = _finite(annual_rate, "annual_rate", nonnegative=True)
        fee = self.get_short_value() * rate * days / 365
        if not all(math.isfinite(x) for x in (fee, self.get_cash() - fee, self.borrow_cost + fee)):
            raise ValueError("borrow charge exceeds finite ledger range")
        self.position["cash"] -= fee
        self.borrow_cost += fee
        return fee

    def update_stock_price(self, stock_id: str, price: float) -> None:
        checked = _finite(price, "price", positive=True)
        if stock_id not in self.position or not isinstance(self.position[stock_id], dict):
            raise KeyError(stock_id)
        if not math.isfinite(self.get_stock_amount(stock_id) * checked):
            raise ValueError("mark exceeds finite ledger range")
        super().update_stock_price(stock_id, checked)

    def get_cash(self, include_settle: bool = False) -> float:
        if include_settle and self._settle_type != self.ST_NO:
            raise NotImplementedError("SignedPosition does not support delayed settlement")
        return self.position["cash"]

    def get_total_cash(self) -> float:
        return self.get_cash() + sum(self.restricted_proceeds.values())

    def calculate_value(self) -> float:
        value = self.get_total_cash() + self.calculate_stock_value()
        if not math.isfinite(value):
            raise ValueError("account equity exceeds finite ledger range")
        return value

    def get_short_value(self) -> float:
        return sum(-self.get_stock_amount(code) * self.get_stock_price(code)
                   for code in self.get_stock_list() if self.get_stock_amount(code) < 0)

    def get_long_value(self) -> float:
        return sum(self.get_stock_amount(code) * self.get_stock_price(code)
                   for code in self.get_stock_list() if self.get_stock_amount(code) > 0)

    def get_stock_weight_dict(self, only_stock: bool = False) -> dict:
        denominator = self.calculate_stock_value() if only_stock else self.calculate_value()
        if only_stock and denominator == 0:
            raise ValueError("only_stock weights are unsupported with zero net stock exposure")
        if not only_stock and denominator <= 0:
            return {code: math.nan for code in self.get_stock_list()}
        return {code: self.get_stock_amount(code) * self.get_stock_price(code) / denominator
                for code in self.get_stock_list()}

    def settle_start(self, settle_type: str) -> None:
        if settle_type != self.ST_NO:
            raise NotImplementedError("SignedPosition does not support delayed settlement")

    def settle_commit(self) -> None:
        if self._settle_type != self.ST_NO:
            raise NotImplementedError("SignedPosition does not support delayed settlement")

    def snapshot(self) -> dict:
        free = self.get_cash()
        return {
            "free_cash": free,
            "restricted_total": sum(self.restricted_proceeds.values()),
            "equity": self.calculate_value(),
            "positions": self.get_stock_amount_dict(),
            "financing_cost": self.borrow_cost,
            "transaction_cost": self.transaction_cost,
            "funding_shortfall": max(-free, 0.0),
        }
