import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import datetime
import plotly.graph_objects as go
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="Financial Market Dashboard", layout="wide")

# --- Title ---
st.markdown("<h1 style='text-align: center; color: #007bff;'>📈 ML-Based Financial Market Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar Category Selection ---
st.sidebar.header("⚙️ Settings")
section = st.sidebar.radio("📁 Select Asset Class", [
    "Forex", "Indices", "Stocks", "Crypto", "Futures",
    "ETFs", "REITs", "Economy", "Gov Bonds", "Corp Bonds"
])

# --- Extended Symbols Dictionary ---
symbols_dict = {
    "Forex": [
        "EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X",
        "USDCHF=X", "NZDUSD=X", "EURJPY=X", "GBPJPY=X", "EURGBP=X"
    ],
    "Indices": [
        "^GSPC", "^DJI", "^IXIC", "^FTSE", "^N225",
        "^NSEI", "^BSESN", "^CNXMIDCAP", "^CNXSMALLCAP", "^BSE500",
        "^BSE100", "^BSE200", "^BSESMEIPO", "^BSEIPO", "^BSEGREENEX", "^BSECARBONEX"
    ],
    "Stocks": [
        "AAPL", "GOOGL", "AMZN", "MSFT", "TSLA",
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "SBIN.NS", "KOTAKBANK.NS", "LT.NS", "ITC.NS", "HINDUNILVR.NS",
        "AXISBANK.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "MARUTI.NS", "SUNPHARMA.NS",
        "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "ADANIENT.NS", "ADANIGREEN.NS"
    ],
    "Crypto": [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
        "ADA-USD", "DOGE-USD", "DOT-USD", "LTC-USD", "TRX-USD"
    ],
    "Futures": [
        "CL=F", "GC=F", "SI=F", "NG=F", "ZC=F",
        "ZS=F", "ZW=F", "LE=F", "HE=F", "KC=F"
    ],
    "ETFs": [
        "SPY", "QQQ", "VTI", "DIA", "ARKK",
        "IVV", "VOO", "IWM", "EFA", "EEM",
        "VNQ", "LQD", "HYG", "BND", "TIP"
    ],
    "REITs": [
        "VNQ", "SCHH", "XLRE", "IYR", "RWR",
        "ICF", "REM", "RWX", "FREL", "USRT"
    ],
    "Economy": [
        "TIP", "SHY", "IEF", "TLT", "BIL",
        "AGG", "MUB", "LQD", "HYG", "BND"
    ],
    "Gov Bonds": [
        "^TNX", "^IRX", "^FVX", "^TYX", "^UST10Y"
    ],
    "Corp Bonds": [
        "LQD", "HYG", "VCSH", "VCIT", "IGIB",
        "JNK", "BKLN", "SJNK", "SHYG", "ANGL"
    ]
}

symbol = st.sidebar.selectbox(f"Select {section} Asset", symbols_dict[section], key=section)

# --- Date and Interval ---
interval = st.sidebar.selectbox("Select Interval", ("1m", "5m", "15m", "1h", "1d"), index=2)
today = datetime.date.today()
interval_limits = {"1m": 7, "5m": 30, "15m": 60, "1h": 180, "1d": 730}
max_days = interval_limits.get(interval, 60)
min_start_date = today - datetime.timedelta(days=max_days)

start = st.sidebar.date_input("Start Date", min_start_date)
end = st.sidebar.date_input("End Date", today)

if start < min_start_date:
    st.sidebar.error(f"⚠️ Data for '{interval}' only available after {min_start_date}.")
    st.stop()

# --- Load Data ---
@st.cache_data
def load_data(symbol, start, end, interval):
    df = yf.download(symbol, start=start, end=end, interval=interval)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")
    df['Return'] = df['Close'].pct_change()
    df['Direction'] = np.where(df['Return'] > 0, 1, 0)
    df.dropna(inplace=True)
    return df

data = load_data(symbol, start, end, interval)

if data.empty:
    st.warning("⚠️ No data found.")
    st.stop()

# --- ML Prediction Section ---
st.subheader("🔍 ML-Based Price Direction Prediction")

features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[features]
y = data['Direction']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled[:-1], y[:-1])

latest_data = X_scaled[-1].reshape(1, -1)
prediction = model.predict(latest_data)[0]

last_close_price = data['Close'].iloc[-1]
price_change = 0.002

if prediction == 1:
    predicted_price = last_close_price * (1 + price_change)
    st.success(f"📢 **Bullish Signal** 🐂: Price will go UP — **BUY Signal** 🔼")
else:
    predicted_price = last_close_price * (1 - price_change)
    st.error(f"📢 **Bearish Signal** 🐻: Price will go DOWN — **SELL Signal** 🔽")

st.markdown(f"**Predicted Price:** {predicted_price:.4f} (Current Price: {last_close_price:.4f})")

# --- Raw Data ---
with st.expander("📋 Show Raw Price Data (last 10 rows)"):
    st.dataframe(data.tail(10), use_container_width=True)

# --- Charts Section ---
st.markdown("---")
st.subheader("📊 Market Visualization")

data['Candle_Direction'] = np.where(data['Close'] > data['Open'], 'Bullish', 'Bearish')
data['Candle_Body'] = abs(data['Close'] - data['Open'])

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🕯️ Candlestick Chart")
    fig_candle = go.Figure(data=[go.Candlestick(
        x=data['Datetime'] if 'Datetime' in data.columns else data['Date'],
        open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
        increasing_line_color='green', decreasing_line_color='red'
    )])
    fig_candle.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig_candle, use_container_width=True)

with col2:
    st.markdown("#### 📊 Bullish vs Bearish Bar Chart")
    fig_candle_bars = px.bar(
        data,
        x='Datetime' if 'Datetime' in data.columns else 'Date',
        y='Candle_Body',
        color='Candle_Direction',
        color_discrete_map={'Bullish': 'green', 'Bearish': 'red'},
        labels={'Candle_Body': 'Candle Body Size'},
        template="plotly_white"
    )
    st.plotly_chart(fig_candle_bars, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Built with ❤️ using Streamlit & yFinance</p>", unsafe_allow_html=True)
