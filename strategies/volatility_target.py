import numpy as np
from .base import TradingStrategy

class VolatilityTargetStrategy(TradingStrategy):
    def __init__(self, target_vol: float = 0.15, lookback: int = 21):
        self.target_vol = target_vol
        self.lookback = lookback

    def step(self, market: StockMarket, portfolio: Portfolio):
        prices = market.get_prices()
        portfolio.update_prices(prices)
        
        for symbol in market.stocks:
            history = market.get_history(symbol)[-self.lookback:]
            if len(history) < 2:
                continue
                
            returns = np.diff(np.log(history))
            vol = np.std(returns) * np.sqrt(252)
            
            current_value = portfolio.positions.get(symbol, 0) * prices[symbol]
            target_value = portfolio.equity() * (self.target_vol / vol) / len(market.stocks)
            
            delta = (target_value - current_value) / prices[symbol]
            portfolio.place_order(symbol, delta)
