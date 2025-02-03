import streamlit as st
import pandas as pd
import plotly.express as px
from market import StockMarket
from portfolio import Portfolio
from strategies import VolatilityTargetStrategy, SectorMomentumStrategy

def main():
    st.set_page_config(layout="wide", page_title="Algo Trading Simulator")
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Sidebar controls
    with st.sidebar:
        st.header("Simulation Configuration")
        initial_capital = st.number_input("Initial Capital ($)", 10000, 1000000, 100000)
        years = st.slider("Simulation Years", 1, 20, 5)
        n_stocks = st.slider("Number of Stocks", 10, 100, 50)
        
        st.subheader("Strategies")
        strategy_options = {
            "Buy & Hold": None,
            "Volatility Targeting": VolatilityTargetStrategy(),
            "Sector Momentum": SectorMomentumStrategy(),
        }
        selected_strategies = st.multiselect(
            "Select Strategies", 
            list(strategy_options.keys()),
            default=["Buy & Hold", "Volatility Targeting"]
        )
        
        st.subheader("Market Parameters")
        enable_jumps = st.checkbox("Enable Market Jumps", True)
        enable_regimes = st.checkbox("Enable Market Regimes", True)
        
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Market Simulation")
        
        if st.button("Run Simulation"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize market and portfolios
            market = StockMarket(n_stocks=n_stocks)
            portfolios = {
                strategy: Portfolio(initial_capital)
                for strategy in selected_strategies
            }
            
            # Initialize strategies
            strategy_instances = {
                name: strategy_options[name] 
                for name in selected_strategies 
                if strategy_options[name] is not None
            }
            
            # Run simulation
            results = []
            total_days = years * 252
            
            for day in range(total_days):
                prices = market.next_day()
                
                for strategy_name in selected_strategies:
                    portfolio = portfolios[strategy_name]
                    portfolio.update_prices(prices)
                    
                    if strategy_name in strategy_instances:
                        strategy_instances[strategy_name].step(market, portfolio)
                    
                    portfolio.daily_update(day)
                    results.append({
                        "day": day,
                        "strategy": strategy_name,
                        "value": portfolio.equity()
                    })
                
                # Update progress
                progress_bar.progress((day + 1) / total_days)
                status_text.text(f"Processing Day {day+1}/{total_days}")
            
            st.session_state.results = pd.DataFrame(results)
            st.success("Simulation Complete!")
            
    # Results display
    if st.session_state.results is not None:
        with col2:
            st.header("Performance Metrics")
            selected_strategy = st.selectbox(
                "Select Strategy", 
                selected_strategies
            )
            
            strat_df = st.session_state.results[
                st.session_state.results['strategy'] == selected_strategy
            ]
            
            latest_value = strat_df['value'].iloc[-1]
            initial_value = strat_df['value'].iloc[0]
            cagr = (latest_value / initial_value) ** (1/years) - 1
            
            st.metric("Final Portfolio Value", f"${latest_value:,.2f}")
            st.metric("CAGR", f"{cagr:.2%}")
            
        st.header("Performance Visualization")
        fig = px.line(
            st.session_state.results, 
            x="day", y="value", color="strategy",
            title="Strategy Performance Comparison",
            labels={"value": "Portfolio Value ($)", "day": "Trading Days"}
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
