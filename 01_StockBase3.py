import streamlit as st
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from binance.client import Client
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# 현재 날짜와 시간을 기준으로 일정 범위를 설정 (예: 최근 2일간의 데이터)

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
    for ma in [18, 56, 112, 224]:
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
data_15m = get_data("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "30 day ago UTC")
data_15m["RSI_14"] = compute_rsi(data_15m["Close"], window=14)
data_15m["RSI_9"] = compute_rsi(data_15m["Close"], window=9)
data_15m["MACD"], data_15m["Signal"] = compute_macd(data_15m["Close"])

# 데이터의 최신 부분을 기준으로 초기 확대 범위를 설정합니다.
# 예를 들어, 마지막 100개 데이터 포인트를 기준으로:
if len(data_15m) > 100:  # 데이터 포인트가 충분한 경우
    initial_start = data_15m.index[-100]
else:  # 데이터 포인트가 100개 미만인 경우
    initial_start = data_15m.index[0]
initial_end = data_15m.index[-1]

latest_date = data_15m.index.max()  # Latest date in your data
one_week_ago = latest_date - pd.Timedelta(days=2)  # Date one week be


fig = make_subplots(
    rows=5,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=("Candlestick", "Moving Averages", "RSI", "MACD"),  # 서브플롯 제목 추가
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
colors = ["blue", "green", "purple", "black"]
for i, ma in enumerate([18, 56, 112, 224]):
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
# Plotly 차트 생성 및 RSI 트레이스 추가
fig.add_trace(
    go.Scatter(
        x=data_15m.index,
        y=data_15m["RSI_14"],
        mode="lines",
        name="RSI 14",
        line=dict(color="blue"),  # 14일 RSI는 파란색으로 표시
    ),
    row=3,  # RSI 그래프를 3번째 행에 추가
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=data_15m.index,
        y=data_15m["RSI_9"],
        mode="lines",
        name="RSI 9",
        line=dict(color="orange"),  # 9일 RSI는 오렌지색으로 표시
    ),
    row=3,  # RSI 그래프를 3번째 행에 추가
    col=1,
)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)  # 과매수 구간
fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)  # 과매도 구간


# Plotly 차트에 MACD 및 신호선 트레이스 추가
fig.add_trace(
    go.Scatter(
        x=data_15m.index,
        y=data_15m["MACD"],
        mode="lines",
        name="MACD",
        line=dict(color="red"),  # MACD는 파란색으로 표시
    ),
    row=5,  # MACD 그래프를 4번째 행에 추가
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=data_15m.index,
        y=data_15m["Signal"],
        mode="lines",
        name="Signal Line",
        line=dict(color="pink"),  # 신호선은 오렌지색으로 표시
    ),
    row=5,  # 신호선 그래프를 4번째 행에 추가
    col=1,
)


fig.update_layout(
    height=1200,
    title="15 Minute BTCUSDT Chart",
    xaxis=dict(
        rangeslider=dict(visible=True),
        type="date",
        range=[one_week_ago, latest_date],  # Set the initial range to the last week
    ),
    xaxis2=dict(
        rangeslider=dict(visible=True), type="date", range=[one_week_ago, latest_date]
    ),
    xaxis3=dict(
        rangeslider=dict(visible=True), type="date", range=[one_week_ago, latest_date]
    ),
    xaxis5=dict(
        rangeslider=dict(visible=True), type="date", range=[one_week_ago, latest_date]
    ),
    dragmode="zoom",  # 드래그 모드를 'zoom'으로 설정하여 차트 확대 가능
)
# Streamlit에 차트 표시
st.plotly_chart(fig, use_container_width=True)
