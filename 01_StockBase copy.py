import streamlit as st
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from binance.client import Client
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="DK Stock Portfolio",
    page_icon="🤖",
)

st.markdown(
    """
# Automatic Trading Bot
            
Welcome to my DK Stock Portfolio!
            
Here are the apps I made:

"""
)


api_key = st.secrets["Binan_API_KEY"]
api_secret = st.secrets["Binan_SECRET_KEY"]
client = Client(api_key, api_secret)
# 데이터 가져오기 함수


# 데이터 가져오기 함수에 스트림릿 캐시 적용
@st.cache_data(show_spinner="Embedding file...")
def get_data(symbol, interval, start, end=None):
    candles = client.get_historical_klines(symbol, interval, start, end)
    dates = [x[0] for x in candles]
    data = [x[1:5] for x in candles]  # Open, High, Low, Close
    df = pd.DataFrame(
        data,
        index=pd.to_datetime(dates, unit="ms"),
        columns=["Open", "High", "Low", "Close"],
    ).astype(float)
    # 이동 평균선 계산
    for ma in [20, 30, 60, 120, 200]:
        df[f"MA{ma}"] = df["Close"].rolling(window=ma).mean()

    # 이동 평균 및 볼린저 밴드 계산
    period = 18
    multiplier = 2.0
    df["MA"] = df["Close"].rolling(window=period).mean()
    df["STD"] = df["Close"].rolling(window=period).std()
    df["Upper"] = df["MA"] + (df["STD"] * multiplier)
    df["Lower"] = df["MA"] - (df["STD"] * multiplier)

    return df


# RSI 계산 함수
def compute_rsi(data, window=14):
    diff = data.diff(1).dropna()
    gain = (diff.where(diff > 0, 0)).rolling(window=window).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# MACD 계산 함수
def compute_macd(data, slow=26, fast=12, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


# 데이터에 RSI 및 MACD 추가
data_15m = get_data("BTCUSDT", Client.KLINE_INTERVAL_15MINUTE, "1 day ago UTC")
data_15m["RSI"] = compute_rsi(data_15m["Close"])
data_15m["MACD"], data_15m["Signal"] = compute_macd(data_15m["Close"])


fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=("Candlestick",),
)
# 캔들스틱 차트 추가
fig.add_trace(
    go.Candlestick(
        x=data_15m.index,
        open=data_15m["Open"],
        high=data_15m["High"],
        low=data_15m["Low"],
        close=data_15m["Close"],
        name="Candlestick",
    ),
    row=1,
    col=1,
)

# 이동 평균선 추가
colors = ["blue", "green", "red", "orange", "purple"]
for i, ma in enumerate([20, 30, 60, 120, 200]):
    fig.add_trace(
        go.Scatter(
            x=data_15m.index,
            y=data_15m[f"MA{ma}"],
            mode="lines",
            name=f"MA{ma}",
            line=dict(color=colors[i]),
        ),
        row=1,
        col=1,
    )

# 볼린저 밴드 추가
fig.add_trace(
    go.Scatter(
        x=data_15m.index,
        y=data_15m["Upper"],
        mode="lines",
        name="Upper Band",
        line=dict(color="red"),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=data_15m.index,
        y=data_15m["Lower"],
        mode="lines",
        name="Lower Band",
        line=dict(color="red"),
    ),
    row=1,
    col=1,
)
# # RSI 추가
fig.add_trace(
    go.Scatter(
        x=data_15m.index,
        y=data_15m["RSI"],  # RSI 그래프를 아래로 이동
        mode="lines",
        name="RSI",
        line=dict(color="purple"),
    ),
    row=3,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=data_15m.index,
        y=data_15m["MACD"],
        mode="lines",
        name="MACD",
        line=dict(color="purple"),
    ),
    row=4,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=data_15m.index,
        y=data_15m["Signal"],
        mode="lines",
        name="Signal",
        line=dict(color="yellow"),
    ),
    row=4,
    col=1,
)


# 레이아웃 설정
fig.update_layout(height=1200, title="15 Minute BTCUSDT Chart")

# Streamlit에 차트 표시
st.plotly_chart(fig)
