import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Page Configuration
st.set_page_config(page_title="Stock Analysis Pro", layout="wide")

# 2. Centralized Data Processing Functions
@st.cache_data
def load_and_process_full_data(filepath):
    if not os.path.exists(filepath):
        st.error(f"File not found: {filepath}")
        return None, None
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=lambda x: x.capitalize())
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Ticker', 'Date'])
    
    # Calculate daily returns for all tickers
    df['Daily_return'] = df.groupby('Ticker')['Close'].pct_change()
    
    vol_df = df.groupby('Ticker')['Daily_return'].std().reset_index()
    vol_df.columns = ['Ticker', 'Volatility']
    vol_df = vol_df.sort_values(by='Volatility', ascending=False).dropna()
    return df, vol_df

@st.cache_data
def fetch_sectors(ticker_list):
    sector_map = {}
    for symbol in ticker_list:
        try:
            full_symbol = f"{symbol}.NS" if ".NS" not in symbol else symbol
            ticker_obj = yf.Ticker(full_symbol)
            sector_map[symbol] = ticker_obj.info.get('sector', 'Other')
        except:
            sector_map[symbol] = 'Other'
    return sector_map

# --- DATA PATHS ---
combined_data_path = r'C:\Users\Haritha\OneDrive\Desktop\guvi\stock\combined_stock_data.csv'
folder_path = r"C:\Users\Haritha\OneDrive\Desktop\guvi\stock\tickers_split"

# Ensure the save folder exists
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# --- MAIN APPLICATION LOGIC ---
try:
    df, volatility_df = load_and_process_full_data(combined_data_path)
    
    if df is not None:
        # Pre-calculate Sector and Performance Data
        unique_tickers = df['Ticker'].unique().tolist()
        
        with st.spinner('Fetching sector information...'):
            mapping = fetch_sectors(unique_tickers)
        
        sector_data = pd.DataFrame(list(mapping.items()), columns=['Ticker', 'Sector'])
        
        # Calculate performance
        perf = df.groupby('Ticker')['Close'].agg(['first', 'last'])
        ticker_performance = (perf['last'] - perf['first']) / perf['first']
        
        sector_perf_df = ticker_performance.to_frame(name='Return').reset_index()
        sector_perf_df = pd.merge(sector_perf_df, sector_data, on='Ticker', how='left').fillna('Other')

        st.title("📈 Data-Driven Stock Analysis Dashboard")
        
        # 3. UI LAYOUT - Define 5 Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎢 Market Volatility", 
            "⭐ Top Performance", 
            "🏢 Sector Analysis", 
            "🔗 Correlation", 
            "📅 Monthly Returns"
        ])

        # --- TAB 1: VOLATILITY ---
        with tab1:
            st.header("🎢 Market Volatility (Top 10)")
            top_10_vol = volatility_df.head(10)
            
            # Save CSV
            save_path_vol = os.path.join(folder_path, "market_volatility_top_10.csv")
            top_10_vol.to_csv(save_path_vol, index=False)
            st.success(f"✅ Volatility CSV saved to: {save_path_vol}")

            fig_vol = px.bar(top_10_vol, x='Ticker', y='Volatility', color='Volatility', color_continuous_scale='Oranges')
            st.plotly_chart(fig_vol, use_container_width=True)

        # --- TAB 2: TOP PERFORMANCE ---
        with tab2:
            st.header("⭐ Top 5 Cumulative Returns")
            start_prices = df.groupby('Ticker')['Close'].first()
            top_5_tickers = ticker_performance.sort_values(ascending=False).head(5).index
            
            df_top5 = df[df['Ticker'].isin(top_5_tickers)].copy()
            df_top5['Baseline'] = df_top5['Ticker'].map(start_prices)
            df_top5['Cum_Return'] = (df_top5['Close'] - df_top5['Baseline']) / df_top5['Baseline']
            
            # Save CSV
            save_path_perf = os.path.join(folder_path, "top_5_performance.csv")
            df_top5.to_csv(save_path_perf, index=False)
            st.success(f"✅ Performance CSV saved")

            chart_pivot = df_top5.pivot(index='Date', columns='Ticker', values='Cum_Return')
            st.line_chart(chart_pivot)

        # --- TAB 3: SECTORS ---
        with tab3:
            st.header("🏢 Sector-Wise Average Performance")
            sector_plot_df = sector_perf_df.groupby('Sector').agg(
                Average_Return=('Return', 'mean'),
                Stocks=('Ticker', lambda x: ', '.join(x))
            ).reset_index().sort_values(by='Average_Return', ascending=False)

            # Save CSV
            save_path_sector = os.path.join(folder_path, "sector_analysis.csv")
            sector_plot_df.to_csv(save_path_sector, index=False)
            st.success(f"✅ Sector CSV saved")

            fig_sector = px.bar(sector_plot_df, x='Sector', y='Average_Return', color='Average_Return',
                                color_continuous_scale='RdYlGn', hover_data={'Stocks': True, 'Average_Return': ':.2%'})
            st.plotly_chart(fig_sector, use_container_width=True)

        # --- TAB 4: CORRELATION ---
        with tab4:
            st.header("🔗 Stock Price Correlation")
            selected_tickers = st.multiselect("Select Tickers", options=unique_tickers, 
                                              default=unique_tickers[:5] if len(unique_tickers) >=5 else unique_tickers)

            if len(selected_tickers) >= 2:
                corr_data = df[df['Ticker'].isin(selected_tickers)].pivot(index='Date', columns='Ticker', values='Close')
                corr_matrix = corr_data.pct_change().corr()

                # Save CSV
                save_path_corr = os.path.join(folder_path, "correlation_matrix.csv")
                corr_matrix.to_csv(save_path_corr) 
                st.success(f"✅ Correlation CSV saved")

                fig_corr, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", ax=ax)
                st.pyplot(fig_corr)
            else:
                st.warning("Please select at least 2 tickers.")

        # --- TAB 5: MONTHLY GAINERS & LOSERS ---
        with tab5:
            st.header("📅 Monthly Performance Grid")
            df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
            
            monthly_returns = df.groupby(['YearMonth', 'Ticker'])['Close'].agg(['first', 'last'])
            monthly_returns['Return'] = ((monthly_returns['last'] - monthly_returns['first']) / monthly_returns['first']) * 100
            monthly_df = monthly_returns.reset_index()
            
            # Save CSV
            save_path_monthly = os.path.join(folder_path, "monthly_performance_full.csv")
            monthly_df.to_csv(save_path_monthly, index=False)
            st.success(f"✅ Monthly CSV saved")

            months = sorted(monthly_df['YearMonth'].unique())
            rows = (len(months) // 3) + (1 if len(months) % 3 != 0 else 0)
            fig_m = make_subplots(rows=rows, cols=3, subplot_titles=months, vertical_spacing=0.05)
            
            for i, m in enumerate(months):
                m_data = monthly_df[monthly_df['YearMonth'] == m].sort_values(by='Return', ascending=False)
                top_bottom = pd.concat([m_data.head(5), m_data.tail(5)])
                r, c = (i // 3) + 1, (i % 3) + 1
                fig_m.add_trace(go.Bar(x=top_bottom['Ticker'], y=top_bottom['Return'], 
                                       marker_color=['green']*5 + ['red']*5, showlegend=False), row=r, col=c)
            
            fig_m.update_layout(height=300 * rows, title_text="Monthly Top 5 Gainers vs Losers")
            st.plotly_chart(fig_m, use_container_width=True)

except Exception as e:
    st.error(f"Critical System Error: {e}")