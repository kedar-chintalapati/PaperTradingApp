from abc import ABC, abstractmethod
from market import StockMarket
from portfolio import Portfolio

class TradingStrategy(ABC):
    @abstractmethod
    def step(self, market: StockMarket, portfolio: Portfolio):
        pass
