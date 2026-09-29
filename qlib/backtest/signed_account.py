# Copyright (c) Tubby Labs contributors.
# Licensed under the MIT License.
"""Opt-in daily signed accounting, with restricted proceeds and stock borrow."""

import math

import pandas as pd

from .account import Account
from .position import BasePosition
from .signed_position import SignedPosition


class SignedAccount(Account):
    """Cash-only initial account. Not broker margin or a cash-loan facility.

    Borrow fees use previous-close short liability and ACT/365 calendar gaps.
    Negative balances remain observable; there is no forced liquidation.
    A fresh account is required for each daily serial backtest.
    """

    def __init__(self, init_cash=1e9, annual_borrow_rate=0.004, benchmark_config=None):
        init_cash = float(init_cash)
        rate = float(annual_borrow_rate)
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("annual_borrow_rate must be finite and nonnegative")
        self.annual_borrow_rate = rate
        self._last_close_session = None
        self._active_session = None
        self.ledger_records = {}
        self._stale_marks = []
        super().__init__(
            init_cash=init_cash, position_dict={}, freq="day",
            benchmark_config={"benchmark": None} if benchmark_config is None else benchmark_config,
            pos_type="qlib.backtest.signed_position.SignedPosition",
            port_metr_enabled=True,
        )

    def validate_executor(self, executor):
        from .executor import SimulatorExecutor
        from .signed_exchange import SignedExchange
        from qlib.utils.time import Freq

        if (type(executor) is not SimulatorExecutor or Freq.parse(executor.time_per_step) != (1, "day")
                or executor.trade_type != SimulatorExecutor.TT_SERIAL
                or executor._settle_type != BasePosition.ST_NO
                or not executor.generate_portfolio_metrics
                or not isinstance(executor.trade_exchange, SignedExchange)):
            raise ValueError("signed mode requires daily serial SimulatorExecutor, SignedExchange, metrics and no settlement lag")
        if self._last_close_session is not None or self._active_session is not None:
            raise ValueError("use a fresh SignedAccount for each backtest")

    def start_bar(self, trade_start_time, trade_end_time, trade_exchange):
        from .signed_exchange import SignedExchange
        from qlib.utils.time import Freq

        if Freq.parse(self.freq) != (1, "day") or not isinstance(trade_exchange, SignedExchange):
            raise ValueError("signed mode requires daily SignedExchange")
        session = pd.Timestamp(trade_start_time).normalize()
        if pd.isna(session):
            raise ValueError("signed session date must not be missing")
        if session == self._active_session:
            return  # exact-once borrow charge if a caller repeats this hook
        if self._active_session is not None:
            raise ValueError("previous signed session has not closed")
        if self._last_close_session is not None and session <= self._last_close_session:
            raise ValueError("signed sessions must move forward")
        self._previous_transaction_cost = self.current_position.transaction_cost
        self._previous_borrow_cost = self.current_position.borrow_cost
        self._stale_marks = []
        if self._last_close_session is not None:
            days = (session - self._last_close_session).days
            fee = self.current_position.accrue_borrow(days, self.annual_borrow_rate)
            self.accum_info.add_cost(fee)
        self._active_session = session

    def update_order(self, order, trade_val, cost, trade_price):
        if self._active_session is None:
            raise ValueError("call start_bar before signed account fills")
        if pd.Timestamp(order.start_time).normalize() != self._active_session:
            raise ValueError("order date differs from active signed session")
        position = self.current_position
        before = position.calculate_value()
        position.update_order(order, trade_val, cost, trade_price)
        self.accum_info.add_cost(cost)
        self.accum_info.add_turnover(trade_val)
        self.accum_info.add_return_value(position.calculate_value() - before + cost)

    def update_current_position(self, trade_start_time, trade_end_time, trade_exchange):
        position = self.current_position
        for code in position.get_stock_list():
            close = trade_exchange.get_close(code, trade_start_time, trade_end_time)
            if close is None or not math.isfinite(float(close)) or float(close) <= 0:
                self._stale_marks.append(code)
                continue
            position.update_stock_price(code, float(close))
        position.add_count_all(bar=self.freq)

    def update_portfolio_metrics(self, trade_start_time, trade_end_time):
        position = self.current_position
        pm = self.portfolio_metrics
        previous = self.init_cash if pm.is_empty() else pm.get_latest_account_value()
        previous_cost = 0.0 if pm.is_empty() else pm.get_latest_total_cost()
        previous_turnover = 0.0 if pm.is_empty() else pm.get_latest_total_turnover()
        equity = position.calculate_value()
        cost = self.accum_info.get_cost - previous_cost
        turnover = self.accum_info.get_turnover - previous_turnover
        valid = previous > 0 and equity > 0
        ratio = lambda value: value / previous if valid else float("nan")
        pm.update_portfolio_metrics_record(
            trade_start_time=trade_start_time, trade_end_time=trade_end_time,
            account_value=equity, cash=position.get_cash(),
            return_rate=ratio(equity - previous + cost),
            total_turnover=self.accum_info.get_turnover, turnover_rate=ratio(turnover),
            total_cost=self.accum_info.get_cost, cost_rate=ratio(cost),
            stock_value=position.calculate_stock_value(),
        )
        self.ledger_records[trade_start_time] = {
            "net_return": ratio(equity - previous),
            "transaction_cost_amount": position.transaction_cost - self._previous_transaction_cost,
            "borrow_cost_amount": position.borrow_cost - self._previous_borrow_cost,
            "total_cost_amount": cost,
            "free_cash": position.get_cash(), "cash_total": position.get_total_cash(),
            "restricted_cash": position.get_total_cash() - position.get_cash(),
            "long_value": position.get_long_value(), "short_liability": position.get_short_value(),
            "unfunded_cash_shortfall": max(-position.get_cash(), 0.0),
            "funding_feasible": position.get_cash() >= 0 and equity > 0,
            "metrics_defined": valid, "stale_marks": tuple(sorted(self._stale_marks)),
        }

    def update_bar_end(self, trade_start_time, trade_end_time, trade_exchange, atomic, **kwargs):
        if not atomic:
            raise ValueError("nested signed execution is not supported")
        session = pd.Timestamp(trade_start_time).normalize()
        if self._active_session != session:
            raise ValueError("signed bar close requires matching start_bar")
        super().update_bar_end(trade_start_time, trade_end_time, trade_exchange, atomic, **kwargs)
        self._last_close_session = session
        self._active_session = None

    def get_portfolio_metrics(self):
        report, positions = super().get_portfolio_metrics()
        extra = pd.DataFrame.from_dict(self.ledger_records, orient="index")
        return report.join(extra), positions
