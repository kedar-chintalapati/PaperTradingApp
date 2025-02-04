import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm, gmean
from arch import arch_model
from statsmodels.tsa.stattools import coint
from collections import defaultdict
from typing import Dict, List, DefaultDict, Tuple
import itertools
import math
import time
import datetime

# For historical data download
import yfinance as yf

# --------------------------------------------------
# Helper functions for Black-Scholes pricing
# --------------------------------------------------

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate call option premium using Black-Scholes formula."""
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate put option premium using Black-Scholes formula."""
    if T <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# --------------------------------------------------
# SECTION 1: MARKET SIMULATION CLASSES
# --------------------------------------------------

class StockMarket:
    """Simulated market with regime switching, GARCH volatility, sector correlations, and jump processes."""
    def __init__(self, n_stocks: int = 50, sectors: List[str] = None, risk_free_rate: float = 0.02, 
                 initial_price: float = 100.0, regime_params: Dict = None):
        self.n_stocks = n_stocks
        self.sectors = sectors or ['Tech', 'Energy', 'Finance', 'Healthcare']
        self.risk_free_rate = risk_free_rate
        self.current_day = 0
        self.stocks = self._initialize_stocks(initial_price)
        self._setup_correlation_matrix()
        self.regime_state = 'normal'
        self.regime_params = regime_params or {
            'transition_matrix': [[0.95, 0.05], [0.15, 0.85]],
            'regime_returns': {'normal': 0.08 / 252, 'crash': -0.20 / 252}
        }

    def _initialize_stocks(self, initial_price: float) -> Dict[str, Dict]:
        stocks = {}
        # Create a random distribution over sectors
        sector_dist = np.random.dirichlet(np.ones(len(self.sectors)))
        for i in range(self.n_stocks):
            sector = np.random.choice(self.sectors, p=sector_dist)
            stocks[f"STK{i}"] = {
                'sector': sector,
                'price': [initial_price],
                'volatility': 0.2 + 0.05 * (i % 4),
                'garch': {'omega': 0.05, 'alpha': 0.1, 'beta': 0.85},
                'jump_params': {'prob': 0.02, 'mean': 0.0, 'std': 0.1},
                'drift': 0.08 / 252,
                'correlated_returns': None
            }
        return stocks

    def _setup_correlation_matrix(self):
        n = self.n_stocks
        self.corr_matrix = np.full((n, n), 0.6)
        np.fill_diagonal(self.corr_matrix, 1.0)
        sector_map = {stk: info['sector'] for stk, info in self.stocks.items()}
        for i in range(n):
            for j in range(n):
                if sector_map[f"STK{i}"] == sector_map[f"STK{j}"]:
                    self.corr_matrix[i, j] = 0.8

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
        # Add a tiny jitter to the diagonal to ensure the matrix is positive definite
        jitter = np.eye(self.n_stocks) * 1e-6
        cholesky = np.linalg.cholesky(self.corr_matrix + jitter)
        innovations = np.dot(cholesky, np.random.randn(self.n_stocks))
        regime_return = self.regime_params['regime_returns'][self.regime_state]

        for i, (symbol, stock) in enumerate(self.stocks.items()):
            # GARCH volatility update
            if len(stock['price']) > 1:
                last_return = np.log(stock['price'][-1] / stock['price'][-2])
            else:
                last_return = 0
            stock['volatility'] = np.sqrt(
                stock['garch']['omega'] +
                stock['garch']['alpha'] * last_return**2 +
                stock['garch']['beta'] * stock['volatility']**2
            )

            # Price calculation with drift, diffusion, and jump process
            drift = regime_return + stock['drift']
            diffusion = stock['volatility'] * innovations[i] * np.sqrt(1 / 252)

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

class HistoricalMarket:
    """
    Historical market that loads adjusted-close prices via yfinance and “plays back” the data one day at a time.
    Mimics the same interface as StockMarket (get_prices, get_history, next_day).
    """
    def __init__(self, tickers: List[str], start_date: str, end_date: str, risk_free_rate: float = 0.02):
        self.tickers = tickers
        self.risk_free_rate = risk_free_rate
        self.data = {}
        self.dates = None
        for ticker in tickers:
            #df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            # Use adjusted close prices
            #df = df[['Adj Close']].rename(columns={'Adj Close': 'price'})
            #df = df.reset_index()
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                st.warning(f"No data found for ticker {ticker}. Skipping.")
                continue
            if 'Adj Close' in df.columns:
                df = df[['Adj Close']].rename(columns={'Adj Close': 'price'})
            elif 'Close' in df.columns:
                df = df[['Close']].rename(columns={'Close': 'price'})
            else:
                st.warning(f"Ticker {ticker} does not have a recognized price column. Skipping.")
                continue
            df = df.reset_index()

            if self.dates is None:
                self.dates = df['Date']
            self.data[ticker] = df['price'].tolist()
        self.current_index = 0
        self.total_days = len(self.dates)
        # Build histories dictionary to mimic StockMarket interface
        self.histories = {ticker: [self.data[ticker][0]] for ticker in tickers}

    def next_day(self) -> Dict[str, float]:
        self.current_index += 1
        if self.current_index >= self.total_days:
            self.current_index = self.total_days - 1
        prices = {}
        for ticker in self.tickers:
            price = self.data[ticker][self.current_index]
            self.histories[ticker].append(price)
            prices[ticker] = price
        return prices

    def get_prices(self) -> Dict[str, float]:
        prices = {}
        for ticker in self.tickers:
            prices[ticker] = self.data[ticker][self.current_index]
        return prices

    def get_history(self, symbol: str) -> List[float]:
        return self.histories[symbol]

# --------------------------------------------------
# SECTION 2: PORTFOLIO MANAGEMENT (unchanged)
# --------------------------------------------------

class Portfolio:
    def __init__(self, initial_cash: float = 100000.0, margin_interest: float = 0.08, 
                 max_leverage: float = 3.0, transaction_cost: float = 0.0005):
        self.cash = initial_cash
        self.positions = defaultdict(float)
        self.options = defaultdict(list)
        self.margin_interest = margin_interest / 252
        self.max_leverage = max_leverage
        self.transaction_cost = transaction_cost
        self.history = []
        self.current_prices = {}
        self.trade_history = []
        self.current_day = 0

    def update_prices(self, prices: Dict[str, float]):
        self.current_prices = prices

    def equity(self) -> float:
        total = self.cash
        # Value of stock positions
        for symbol, qty in self.positions.items():
            if symbol.startswith('OPTION'):
                continue
            total += qty * self.current_prices.get(symbol, 0)

        # Options valuation
        for contracts in self.options.values():
            for contract in contracts:
                S = self.current_prices.get(contract['underlying'], 0)
                T = (contract['expiry'] - self.current_day) / 252
                price = self._black_scholes(
                    S, contract['strike'], T,
                    contract['risk_free_rate'], contract['iv'], contract['type']
                )
                total += price * contract['qty'] * 100
        return total

    def _black_scholes(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        if T <= 0:
            return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def margin_available(self) -> float:
        equity = self.equity()
        return (self.max_leverage * equity) - (equity - self.cash)

    def place_order(self, symbol: str, qty: float, option_contract: Dict = None):
        if option_contract:
            self.options[symbol].append(option_contract)
            cost = option_contract['premium'] * 100 * abs(qty)
        else:
            price = self.current_prices[symbol]
            cost = price * qty * (1 + self.transaction_cost)
            self.positions[symbol] += qty

        if cost > self.margin_available() + self.cash:
            st.error(f"Insufficient funds for {symbol} order")
            return

        self.cash -= cost
        self.trade_history.append({
            'day': self.current_day,
            'symbol': symbol,
            'qty': qty,
            'type': 'option' if option_contract else 'stock'
        })

    def daily_update(self, current_day: int):
        self.current_day = current_day
        if self.cash < 0:
            self.cash *= (1 + self.margin_interest)
        for symbol in list(self.options.keys()):
            self.options[symbol] = [c for c in self.options[symbol] if c['expiry'] > current_day]
        self.history.append(self.equity())

# --------------------------------------------------
# SECTION 3: TRADING STRATEGIES (unchanged)
# --------------------------------------------------

class VolatilityTargetStrategy:
    def __init__(self, target_vol: float = 0.15, lookback: int = 21):
        self.target_vol = target_vol
        self.lookback = lookback

    def execute(self, market, portfolio: Portfolio):
        prices = market.get_prices()
        portfolio.update_prices(prices)
        for symbol in market.tickers if hasattr(market, 'tickers') else market.stocks.keys():
            history = market.get_history(symbol)[-self.lookback:]
            if len(history) < 2:
                continue
            returns = np.diff(np.log(history))
            vol = np.std(returns) * np.sqrt(252)
            leverage = min(self.target_vol / vol, portfolio.max_leverage) if vol > 0 else 1
            current_value = portfolio.positions.get(symbol, 0) * prices[symbol]
            target_value = portfolio.equity() * leverage / (len(market.tickers) if hasattr(market, 'tickers') else len(market.stocks))
            delta = (target_value - current_value) / prices[symbol]
            portfolio.place_order(symbol, delta)

class SectorMomentumStrategy:
    def __init__(self, lookback: int = 63, top_n: int = 2):
        self.lookback = lookback
        self.top_n = top_n

    def execute(self, market, portfolio: Portfolio):
        if market.current_day < self.lookback:
            return
        sector_returns = defaultdict(list)
        # For simulated market we have a 'sector' attribute; for historical, we assign "N/A"
        if hasattr(market, 'stocks'):
            for symbol, data in market.stocks.items():
                if len(data['price']) < self.lookback:
                    continue
                ret = (data['price'][-1] - data['price'][-self.lookback]) / data['price'][-self.lookback]
                sector = data['sector']
                sector_returns[sector].append(ret)
        else:
            # For historical market, treat all tickers as one sector
            for symbol in market.tickers:
                history = market.get_history(symbol)
                if len(history) < self.lookback:
                    continue
                ret = (history[-1] - history[-self.lookback]) / history[-self.lookback]
                sector_returns["Historical"].append(ret)
        sector_perf = {sector: np.mean(returns) for sector, returns in sector_returns.items()}
        sorted_sectors = sorted(sector_perf.items(), key=lambda x: x[1], reverse=True)
        long_sectors = [s for s, _ in sorted_sectors[:self.top_n]]
        short_sectors = [s for s, _ in sorted_sectors[-self.top_n:]]
        long_stocks = []
        short_stocks = []
        if hasattr(market, 'stocks'):
            for symbol, data in market.stocks.items():
                if data['sector'] in long_sectors:
                    long_stocks.append(symbol)
                elif data['sector'] in short_sectors:
                    short_stocks.append(symbol)
        else:
            # In historical mode, treat all tickers equally.
            long_stocks = market.tickers
            short_stocks = []
        equity = portfolio.equity()
        target_long = equity / len(long_stocks) if long_stocks else 0
        target_short = -equity / len(short_stocks) if short_stocks else 0
        prices = market.get_prices()
        for symbol in long_stocks:
            price = prices[symbol]
            current_value = portfolio.positions.get(symbol, 0) * price
            target_qty = target_long / price
            delta_qty = target_qty - portfolio.positions.get(symbol, 0)
            portfolio.place_order(symbol, delta_qty)
        for symbol in short_stocks:
            price = prices[symbol]
            current_value = portfolio.positions.get(symbol, 0) * price
            target_qty = target_short / price
            delta_qty = target_qty - portfolio.positions.get(symbol, 0)
            portfolio.place_order(symbol, delta_qty)

class PairsTradingStrategy:
    def __init__(self, lookback: int = 63, entry_threshold: float = 2.0, exit_threshold: float = 0.5):
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.pairs = None
        self.positions = {}

    def execute(self, market, portfolio: Portfolio):
        if market.current_day < self.lookback:
            return
        if self.pairs is None:
            symbols = market.tickers if hasattr(market, 'tickers') else list(market.stocks.keys())
            self.pairs = []
            for sym1, sym2 in itertools.combinations(symbols, 2):
                hist1 = market.get_history(sym1)[-self.lookback:]
                hist2 = market.get_history(sym2)[-self.lookback:]
                try:
                    score, pvalue, _ = coint(hist1, hist2)
                except Exception:
                    continue
                if pvalue < 0.05:
                    self.pairs.append((sym1, sym2))
                    self.positions[(sym1, sym2)] = 0
        prices = market.get_prices()
        for sym1, sym2 in self.pairs:
            price1 = prices[sym1]
            price2 = prices[sym2]
            spread = np.log(price1) - np.log(price2)
            hist1 = np.array(market.get_history(sym1)[-self.lookback:])
            hist2 = np.array(market.get_history(sym2)[-self.lookback:])
            spread_history = np.log(hist1) - np.log(hist2)
            spread_mean = np.mean(spread_history)
            spread_std = np.std(spread_history)
            zscore = (spread - spread_mean) / spread_std if spread_std > 0 else 0
            position = self.positions.get((sym1, sym2), 0)
            if position == 0:
                if zscore > self.entry_threshold:
                    self.positions[(sym1, sym2)] = -1
                    allocation = portfolio.equity() * 0.01
                    qty1 = - allocation / price1
                    qty2 = allocation / price2
                    portfolio.place_order(sym1, qty1)
                    portfolio.place_order(sym2, qty2)
                elif zscore < -self.entry_threshold:
                    self.positions[(sym1, sym2)] = 1
                    allocation = portfolio.equity() * 0.01
                    qty1 = allocation / price1
                    qty2 = - allocation / price2
                    portfolio.place_order(sym1, qty1)
                    portfolio.place_order(sym2, qty2)
            else:
                if abs(zscore) < self.exit_threshold:
                    current_qty1 = portfolio.positions.get(sym1, 0)
                    current_qty2 = portfolio.positions.get(sym2, 0)
                    portfolio.place_order(sym1, -current_qty1)
                    portfolio.place_order(sym2, -current_qty2)
                    self.positions[(sym1, sym2)] = 0

class TrendFollowingStrategy:
    def __init__(self, short_window: int = 21, long_window: int = 63):
        self.short_window = short_window
        self.long_window = long_window

    def execute(self, market, portfolio: Portfolio):
        if market.current_day < self.long_window:
            return
        prices = market.get_prices()
        symbols = market.tickers if hasattr(market, 'tickers') else market.stocks.keys()
        for symbol in symbols:
            history = (market.get_history(symbol) if hasattr(market, 'tickers') 
                       else market.stocks[symbol]['price'])[-self.long_window:]
            if len(history) < self.long_window:
                continue
            short_ma = np.mean(history[-self.short_window:])
            long_ma = np.mean(history)
            price = prices[symbol]
            current_qty = portfolio.positions.get(symbol, 0)
            if short_ma > long_ma:
                target_value = portfolio.equity() / (len(symbols))
            elif short_ma < long_ma:
                target_value = - portfolio.equity() / (len(symbols))
            else:
                target_value = 0
            target_qty = target_value / price
            delta_qty = target_qty - current_qty
            portfolio.place_order(symbol, delta_qty)

class OptionsStrategy:
    def __init__(self, min_shares: int = 100, premium_threshold: float = 0.02):
        self.min_shares = min_shares
        self.premium_threshold = premium_threshold

    def execute(self, market, portfolio: Portfolio):
        prices = market.get_prices()
        symbols = market.tickers if hasattr(market, 'tickers') else market.stocks.keys()
        for symbol in symbols:
            if symbol.startswith('OPTION'):
                continue
            if portfolio.positions.get(symbol, 0) < self.min_shares:
                continue
            option_key = "OPTION_" + symbol
            existing_contracts = sum([abs(contract.get('qty', 0)) for contract in portfolio.options.get(option_key, [])])
            total_contracts_needed = int(portfolio.positions.get(symbol, 0) // 100)
            contracts_to_sell = total_contracts_needed - existing_contracts
            if contracts_to_sell <= 0:
                continue
            S = prices[symbol]
            K = S * 1.05
            T = 30 / 252
            r = market.risk_free_rate
            sigma = (market.stocks[symbol]['volatility'] if hasattr(market, 'stocks') 
                     else 0.2)
            premium = black_scholes_call(S, K, T, r, sigma)
            if premium / S < self.premium_threshold:
                continue
            option_contract = {
                'underlying': symbol,
                'expiry': portfolio.current_day + 30,
                'risk_free_rate': r,
                'iv': sigma,
                'type': 'call',
                'strike': K,
                'premium': premium,
                'qty': -contracts_to_sell
            }
            portfolio.place_order(option_key, -contracts_to_sell, option_contract=option_contract)

# --------------------------------------------------
# SECTION 4: METRICS & FORMATTING
# --------------------------------------------------

def compute_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    for strategy in results_df['strategy'].unique():
        strat_data = results_df[results_df['strategy'] == strategy]
        returns = strat_data['value'].pct_change().dropna()
        if returns.std() != 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = np.nan
        metrics.append({
            "Strategy": strategy,
            "CAGR": gmean(returns + 1) ** 252 - 1,
            "Volatility": returns.std() * np.sqrt(252),
            "Sharpe": sharpe,
            "Max Drawdown": (strat_data['value'].cummax() - strat_data['value']).max()
        })
    return pd.DataFrame(metrics)

def format_metrics_df(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()
    df["CAGR"] = df["CAGR"].apply(lambda x: f"{x:.2%}")
    df["Volatility"] = df["Volatility"].apply(lambda x: f"{x:.2%}")
    df["Sharpe"] = df["Sharpe"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
    df["Max Drawdown"] = df["Max Drawdown"].apply(lambda x: f"${x:,.2f}")
    return df

# --------------------------------------------------
# SECTION 5: STREAMLIT UI
# --------------------------------------------------

def main():
    st.set_page_config(layout="wide", page_title="Algo Trading Simulator")

    # Allow the user to choose between simulated and historical backtesting.
    market_mode = st.sidebar.radio("Select Market Type", options=["Simulated Market", "Historical Data"])

    if market_mode == "Simulated Market":
        initial_capital = st.sidebar.number_input("Initial Capital ($)", 10000, 1000000, 100000)
        years = st.sidebar.slider("Simulation Years", 1, 20, 5)
        n_stocks = st.sidebar.slider("Number of Stocks", 10, 100, 50)
    else:
        initial_capital = st.sidebar.number_input("Initial Capital ($)", 10000, 1000000, 100000)
        """
        tickers = st.sidebar.multiselect("Select Ticker Symbols", 
                                         options=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "FB", "NFLX"],
                                         default=["AAPL", "MSFT", "GOOGL", "AMZN"])
                                         """

        tickers_input = st.sidebar.text_input("Enter Ticker Symbols (comma separated)", 
                                                value="AAPL, MSFT, GOOGL, AMZN, TSLA")
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

        
        start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
        end_date = st.sidebar.date_input("End Date", datetime.date(2023, 1, 1))
        if start_date >= end_date:
            st.sidebar.error("Error: End date must fall after start date.")
            st.stop()

    st.sidebar.subheader("Strategies")
    strategies = {
        "Buy & Hold": None,
        "Volatility Targeting": VolatilityTargetStrategy(),
        "Sector Momentum": SectorMomentumStrategy(),
        "Pairs Trading": PairsTradingStrategy(),
        "Trend Following": TrendFollowingStrategy(),
        "Covered Call": OptionsStrategy()
    }
    selected_strategies = st.sidebar.multiselect("Select Strategies", list(strategies.keys()), 
                                                   default=["Buy & Hold", "Volatility Targeting"])

    if st.sidebar.button("Run Backtest"):
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        # Initialize market instance based on mode
        if market_mode == "Simulated Market":
            market = StockMarket(n_stocks=n_stocks)
            total_days = years * 252
        else:
            market = HistoricalMarket(tickers=tickers, start_date=str(start_date), end_date=str(end_date))
            total_days = market.total_days

        # Create separate portfolio for each strategy
        portfolios = {name: Portfolio(initial_capital) for name in selected_strategies}
        strategy_instances = {name: strategies[name] for name in selected_strategies if strategies[name] is not None}

        results = []
        for day in range(total_days):
            prices = market.next_day()
            for name in selected_strategies:
                portfolio = portfolios[name]
                portfolio.update_prices(prices)
                portfolio.current_day = day
                # For Buy & Hold, invest equally on day 0
                if name == "Buy & Hold" and day == 0:
                    if market_mode == "Simulated Market":
                        symbols = list(market.stocks.keys())
                    else:
                        symbols = market.tickers
                    for symbol in symbols:
                        price = prices[symbol]
                        target_value = portfolio.equity() / len(symbols)
                        qty = target_value / price
                        portfolio.place_order(symbol, qty)
                elif name in strategy_instances:
                    strategy_instances[name].execute(market, portfolio)
                portfolio.daily_update(day)
                results.append({"day": day, "strategy": name, "value": portfolio.equity()})
            progress_bar.progress((day + 1) / total_days)
            status_text.text(f"Day {day + 1} of {total_days}")
        st.session_state.results = pd.DataFrame(results)
        raw_metrics_df = compute_metrics(st.session_state.results)
        st.session_state.metrics_df = raw_metrics_df
        st.session_state.formatted_metrics_df = format_metrics_df(raw_metrics_df)
        st.success("Backtest complete!")

    st.header("Performance Analysis")
    if "results" in st.session_state and not st.session_state.results.empty:
        fig = px.line(st.session_state.results, x="day", y="value", color="strategy",
                      title="Strategy Comparison", labels={"value": "Portfolio Value ($)", "day": "Trading Days"})
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Performance Metrics")
        if "formatted_metrics_df" in st.session_state and st.session_state.formatted_metrics_df is not None:
            st.table(st.session_state.formatted_metrics_df)

if __name__ == "__main__":
    main()
