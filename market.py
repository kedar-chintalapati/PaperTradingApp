import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm
from typing import Dict, List

class StockMarket:
    def __init__(self, n_stocks: int = 50, sectors: List[str] = None, risk_free_rate: float = 0.02,
                 initial_price: float = 100.0, regime_params: Dict = None):
        self.n_stocks = n_stocks
        self.sectors = sectors or ['Tech', 'Energy', 'Finance', 'Healthcare', 'Consumer']
        self.risk_free_rate = risk_free_rate
        self.current_day = 0
        self.stocks = self._initialize_stocks(initial_price, regime_params)
        self._setup_correlation_matrix()
        self.regime_state = 'normal'
        self.regime_params = regime_params or {
            'transition_matrix': [[0.95, 0.05], [0.15, 0.85]],  # [normal, crash]
            'regime_returns': {'normal': 0.08/252, 'crash': -0.20/252}
        }

    def _initialize_stocks(self, initial_price: float, regime_params: Dict) -> Dict:
        stocks = {}
        sector_weights = np.random.dirichlet(np.ones(len(self.sectors)))
        
        for i in range(self.n_stocks):
            sector = np.random.choice(self.sectors, p=sector_weights)
            volatility = 0.2 + 0.05 * (i % 5)
            stocks[f"STK{i}"] = {
                'sector': sector,
                'price': [initial_price],
                'volatility': volatility,
                'garch': {'omega': 0.05, 'alpha': 0.1, 'beta': 0.85},
                'jump_params': {'prob': 0.02, 'mean': 0.0, 'std': 0.1},
                'drift': 0.08/252,
                'correlated_returns': None
            }
        return stocks

    def _setup_correlation_matrix(self):
        n = self.n_stocks
        base_corr = 0.7
        self.corr_matrix = np.full((n, n), base_corr)
        np.fill_diagonal(self.corr_matrix, 1.0)
        
        # Increase intra-sector correlation
        sector_map = {stk: info['sector'] for stk, info in self.stocks.items()}
        for i in range(n):
            for j in range(n):
                if sector_map[f"STK{i}"] == sector_map[f"STK{j}"]:
                    self.corr_matrix[i,j] = min(self.corr_matrix[i,j] + 0.2, 0.95)

    def _update_regime(self):
        if self.regime_state == 'normal':
            if np.random.rand() < self.regime_params['transition_matrix'][0][1]:
                self.regime_state = 'crash'
        else:
            if np.random.rand() < self.regime_params['transition_matrix'][1][0]:
                self.regime_state = 'normal'

    def next_day(self) -> Dict[str, float]:
        self.current_day += 1
        self._update_regime()
        cholesky = np.linalg.cholesky(self.corr_matrix)
        innovations = np.dot(cholesky, np.random.randn(self.n_stocks))
        
        regime_return = self.regime_params['regime_returns'][self.regime_state]
        
        for i, (symbol, stock) in enumerate(self.stocks.items()):
            # Update GARCH volatility
            last_return = np.log(stock['price'][-1]/stock['price'][-2]) if len(stock['price'])>1 else 0
            stock['volatility'] = np.sqrt(
                stock['garch']['omega'] + 
                stock['garch']['alpha'] * last_return**2 + 
                stock['garch']['beta'] * stock['volatility']**2
            )
            
            # Generate returns with regime effect
            drift = regime_return + stock['drift']
            diffusion = stock['volatility'] * innovations[i] * np.sqrt(1/252)
            
            # Add jumps
            if np.random.rand() < stock['jump_params']['prob']:
                jump = np.random.normal(stock['jump_params']['mean'], stock['jump_params']['std'])
            else:
                jump = 0
                
            new_price = stock['price'][-1] * np.exp(drift + diffusion + jump)
            stock['price'].append(new_price)
        
        return self.get_prices()

    def get_prices(self) -> Dict[str, float]:
        return {symbol: data['price'][-1] for symbol, data in self.stocks.items()}

    def get_history(self, symbol: str) -> List[float]:
        return self.stocks[symbol]['price']
    
    def reset(self):
        self.current_day = 0
        for symbol in self.stocks:
            self.stocks[symbol]['price'] = [self.stocks[symbol]['price'][0]]
