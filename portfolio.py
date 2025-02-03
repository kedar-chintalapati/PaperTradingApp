from typing import Dict, List, DefaultDict
from collections import defaultdict
import numpy as np

class Portfolio:
    def __init__(self, initial_cash: float = 100000.0, margin_interest: float = 0.08, 
                 max_leverage: float = 3.0, transaction_cost: float = 0.0005):
        self.cash = initial_cash
        self.positions: DefaultDict[str, float] = defaultdict(float)
        self.options: DefaultDict[str, List[Dict]] = defaultdict(list)
        self.margin_interest = margin_interest / 252
        self.max_leverage = max_leverage
        self.transaction_cost = transaction_cost
        self.history: List[float] = []
        self.current_prices: Dict[str, float] = {}
        self.trade_history: List[Dict] = []

    def update_prices(self, prices: Dict[str, float]):
        self.current_prices = prices

    def equity(self) -> float:
        total = self.cash
        # Stock positions
        for symbol, qty in self.positions.items():
            if not symbol.startswith('OPTION'):
                total += qty * self.current_prices.get(symbol, 0)
        
        # Options valuation
        for option in self.options.values():
            for contract in option:
                underlying_price = self.current_prices.get(contract['underlying'], 0)
                time_left = (contract['expiry'] - self.current_day) / 252
                iv = contract['iv']
                bs_price = self._black_scholes(
                    underlying_price, contract['strike'], time_left,
                    self.risk_free_rate, iv, contract['type']
                )
                total += bs_price * contract['qty'] * 100
        
        return total

    def _black_scholes(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        if T <= 0:
            return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        else:
            return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def margin_available(self) -> float:
        equity = self.equity()
        return (self.max_leverage * equity) - (equity - self.cash)

    def place_order(self, symbol: str, qty: float, option_contract: Dict = None):
        # Implementation handles stocks and options
        pass

    def liquidate(self, symbol: str):
        # Full implementation
        pass

    def daily_update(self, current_day: int):
        # Handle margin interest, option expiration
        pass
