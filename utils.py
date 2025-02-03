import numpy as np
from scipy.stats import gmean

def calculate_metrics(returns: pd.Series) -> Dict:
    return {
        'CAGR': gmean(returns + 1)**252 - 1,
        'Volatility': returns.std() * np.sqrt(252),
        'Sharpe': returns.mean() / returns.std() * np.sqrt(252),
        'Max Drawdown': (returns.cummax() - returns).max()
    }

def format_pct(x: float) -> str:
    return f"{x:.2%}"
