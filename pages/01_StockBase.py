import streamlit as st
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from binance.client import Client
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from pandas.tseries.offsets import DateOffset
import math
import numpy as np

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


# 데이터에 RSI 및 MACD 추가 ETH
data_15m = get_data("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "30 day ago UTC")
data_15m["RSI_14"] = compute_rsi(data_15m["Close"], window=14)
data_15m["RSI_9"] = compute_rsi(data_15m["Close"], window=9)
data_15m["MACD"], data_15m["Signal"] = compute_macd(data_15m["Close"])

# MACD와 Signal 라인 간의 차이를 계산합니다.
data_15m["MACD_diff"] = data_15m["MACD"] - data_15m["Signal"]

print(data_15m["MACD_diff"])
# Calculate the difference in MACD values between the current and the previous row
data_15m["MACD_slope"] = data_15m["MACD"].diff()

# Alternatively, if you want to calculate the slope over a different period (e.g., over 2 periods), you can use:
# data_15m["MACD_slope"] = data_15m["MACD"].diff(2)

# Check the sign of the MACD_slope to determine if it's going up or down
data_15m["MACD_direction"] = data_15m["MACD_slope"].apply(
    lambda x: "up" if x > 0 else ("down" if x < 0 else "flat")
)
# print(data_15m["MACD_direction"] == "up")

data_15m["MACD_bounce_up_with_positive_slope"] = (
    (data_15m["MACD"].shift(2) > data_15m["Signal"].shift(2))
    & (data_15m["MACD"].shift(1) < data_15m["Signal"].shift(1))
    & (data_15m["MACD"] > data_15m["Signal"])  # 바로 이전에는 MACD가 Signal 아래였음
    & (  # 현재는 MACD가 Signal 위에 있음
        data_15m["MACD_direction"] == "up"
    )  # MACD의 기울기가 양수, 즉 상승세
)


# # MACD 지지 및 저항 판단
# data_15m["MACD_support"] = (
#     data_15m["MACD_within_tolerance"] | (data_15m["MACD"] < data_15m["Signal"])
# ) & (data_15m["MACD_slope"] > 0)
# data_15m["MACD_resistance"] = (
#     data_15m["MACD_within_tolerance"] | (data_15m["MACD"] > data_15m["Signal"])
# ) & (data_15m["MACD_slope"] < 0)

data_15m["MACD_diff_by0"] = np.round(data_15m["MACD_diff"].abs()) == 0
data_15m["MACD_diff_above1"] = np.round(data_15m["MACD_diff"].abs(), 1) >= 0.5
# print(np.round(data_15m["MACD_diff"].abs(), 2))
# MACD_diff가 특정 양의 값(여기서는 0.5) 이상인 경우를 찾습니다.

data_15m["MACD_diff_above2"] = (
    data_15m["MACD"].shift(2) - data_15m["Signal"].shift(1)
).abs() >= 0.5


data_15m["MACD_diff_above3"] = (
    data_15m["MACD"].shift(1) - data_15m["Signal"].shift(1)
).abs() <= 0.3
# 16.36 - 15.146


# data_15m["MACD_cross_up"] = (
#     (data_15m["MACD"] > data_15m["Signal"])
#     & (data_15m["MACD"].shift(1) < data_15m["Signal"].shift(1))
#     & (data_15m["MACD"].shift(2) < data_15m["Signal"].shift(2))
#     & (data_15m["MACD"].shift(3) < data_15m["Signal"].shift(3))
# ) | ((data_15m["MACD"] < data_15m["Signal"]) & data_15m["MACD_diff_above3"])

data_15m["MACD_cross_up"] = (
    (data_15m["MACD"] > data_15m["Signal"])
    & (data_15m["MACD"].shift(1) < data_15m["Signal"].shift(1))
    & (data_15m["MACD"].shift(2) < data_15m["Signal"].shift(2))
    & (data_15m["MACD"].shift(3) < data_15m["Signal"].shift(3))
) | data_15m["MACD_bounce_up_with_positive_slope"]

data_15m["MACD_cross_up2"] = (
    (data_15m["MACD"].shift(1) < data_15m["Signal"].shift(1))
    & (data_15m["MACD"].shift(2) < data_15m["Signal"].shift(2))
    & (data_15m["MACD"].shift(3) < data_15m["Signal"].shift(3))
) & data_15m["MACD_diff_above3"]


# RSI 9가 40 미만이었던 최근 2개 캔들 확인
data_15m["RSI_recently_below_40"] = (
    (data_15m["RSI_14"] < 45)
    | (data_15m["RSI_14"].shift(1) < 45)
    | (data_15m["RSI_14"].shift(2) < 45)
) | (
    (data_15m["RSI_14"].shift(6) < 40)
    | (data_15m["RSI_14"].shift(5) < 40)
    | (data_15m["RSI_14"].shift(7) < 40)
    | (data_15m["RSI_14"].shift(8) < 40)
)


data_15m["RSI_recently_below_60"] = (
    (data_15m["RSI_14"] < 65)
    | (data_15m["RSI_14"].shift(1) < 65)
    | (data_15m["RSI_14"].shift(2) < 65)
)
data_15m["RSI_recently_above_40"] = data_15m["RSI_14"] < 40

# 11/20 17:00 macd  16.36  signal  15.146

# Check if RSI is above 50 and in an upward trend
data_15m["RSI_above_50_and_rising"] = data_15m["RSI_recently_below_60"] & (
    data_15m["RSI_14"] > data_15m["RSI_14"].shift(1)
)


# 11 27 rsi 55 macd 겨우 겹침 통과할랑말랑
# 12 2일 11시엔 keep 이나 buy 였어야함 근데 why?
# 12 3일엔 샀어야함 근데 없음 why ?
# 12 3 21:00 매도가뜸 시그널이 더큼 같은시간에
# 12 3 13:00  ~ 19:00 사야됨
# 12 5 16:00
# 11 20  17:00

# Define buy condition: MACD bullish crossover and (RSI below 40 or RSI above 50 and rising)
data_15m["buy_condition"] = (
    (data_15m["MACD_cross_up"] & (data_15m["RSI_recently_below_40"]))
    | (data_15m["MACD_cross_up"] & data_15m["RSI_above_50_and_rising"])
    | (data_15m["MACD_cross_up2"] & data_15m["RSI_recently_above_40"])
)

data_15m["MACD_cross_down2"] = (
    (data_15m["MACD"].shift(1) > data_15m["Signal"].shift(1))
    & (data_15m["MACD"].shift(2) < data_15m["Signal"].shift(2))
    & (data_15m["MACD"].shift(3) > data_15m["Signal"].shift(3))
) & (data_15m["MACD"] < data_15m["Signal"])

## 11 29 참고
data_15m["MACD_cross_down"] = (
    (data_15m["MACD"] < data_15m["Signal"])
    & (data_15m["MACD"].shift(1) > data_15m["Signal"].shift(1))
    & (data_15m["MACD"].shift(2) > data_15m["Signal"].shift(2))
)


data_15m["RSI_recently_above_60"] = (
    (data_15m["RSI_14"] > 60)
    | (data_15m["RSI_14"].shift(1) > 60)
    | (data_15m["RSI_14"].shift(2) > 60)
)


data_15m["RSI_recently_above_45"] = (
    (data_15m["RSI_14"] > 45)
    | (data_15m["RSI_14"].shift(1) > 45)
    | (data_15m["RSI_14"].shift(2) > 45)
)


data_15m["RSI_above_50_and_falling"] = data_15m["RSI_recently_above_45"] & (
    data_15m["RSI_14"] < data_15m["RSI_14"].shift(1)
)


data_15m["sell_condition"] = (
    (data_15m["MACD_cross_down"] & (data_15m["RSI_recently_above_60"]))
    | (data_15m["MACD_cross_down"] & data_15m["RSI_above_50_and_falling"])
    | data_15m["MACD_cross_down2"]
)


# Extract buy and sell signals
buy_signals = data_15m[data_15m["buy_condition"]]
sell_signals = data_15m[data_15m["sell_condition"]]


data_15m["MA18"] = data_15m["Close"].rolling(window=18).mean()
data_15m["Bullish"] = data_15m["Close"] > data_15m["Open"]  # Bullish candlestick

# Define keep condition
data_15m["Keep"] = (data_15m["Close"] >= data_15m["MA18"]) | (
    data_15m["Close"] >= data_15m["Upper"]
)
# & data_15m["Bullish"]


data_15m["Keep_Signal"] = False


# Loop through the DataFrame using the index
for i in data_15m.index:
    if data_15m.loc[i, "buy_condition"]:
        j = i
        one_hour = DateOffset(hours=1)

        while j in data_15m.index and not data_15m.loc[j, "sell_condition"]:
            if data_15m.loc[j, "Keep"]:
                data_15m.loc[j, "Keep_Signal"] = True
            j += one_hour  # Assuming 'j' is a Timestamp here, not an integer

# Adjust the sell condition based on the keep signal
data_15m["Adjusted_Sell_Condition"] = data_15m["sell_condition"] & (
    ~data_15m["Keep_Signal"]
)


# Filter for adjusted sell signals
adjusted_sell_signals = data_15m[data_15m["Adjusted_Sell_Condition"]]


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

fig.add_trace(
    go.Scatter(
        x=buy_signals.index,
        y=buy_signals["Open"],
        mode="markers",
        marker=dict(color="red", size=15, symbol="triangle-up"),
        name="Buy Signal",
    ),
    row=1,
    col=1,
)

keep_signals = data_15m[data_15m["Keep_Signal"]]

# Add traces for Keep signals
fig.add_trace(
    go.Scatter(
        x=keep_signals.index,
        y=keep_signals["Close"],
        mode="markers",
        marker=dict(color="orange", size=10, symbol="star"),
        name="Keep Signal",
    ),
    row=1,
    col=1,
)


fig.add_trace(
    go.Scatter(
        x=adjusted_sell_signals.index,
        y=adjusted_sell_signals["Open"],
        mode="markers",
        marker=dict(color="blue", size=15, symbol="triangle-down"),
        name="Adjusted Sell Signal",
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
fig.add_hline(y=60, line_dash="dash", line_color="blue", row=3, col=1)  # 과매수 구간
fig.add_hline(y=40, line_dash="dash", line_color="blue", row=3, col=1)  # 과매수 구간


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
# Set the initial zoom to the most recent 100 data points
initial_zoom_points = 100
if len(data_15m) > initial_zoom_points:
    end_date = data_15m.index[-1]
    start_date = data_15m.index[-initial_zoom_points]

    # fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    # Add your traces here

    # Update the layout to set the x-axis range
    fig.update_layout(
        height=4500,
        title="15 Minute BTCUSDT Chart",
        xaxis=dict(
            rangeslider=dict(visible=True), type="date", range=[start_date, end_date]
        ),
        # Repeat for all x-axis if you have multiple subplots
        xaxis2=dict(
            rangeslider=dict(visible=True), type="date", range=[start_date, end_date]
        ),
        xaxis3=dict(
            rangeslider=dict(visible=True), type="date", range=[start_date, end_date]
        ),
        xaxis5=dict(
            rangeslider=dict(visible=True), type="date", range=[start_date, end_date]
        ),
        # Set the drag mode to zoom for better user experience
        dragmode="zoom",
    )

    # Render the figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)


# fig.update_layout(
#     height=4500,
#     title="15 Minute BTCUSDT Chart",
#     xaxis=dict(
#         rangeslider=dict(visible=True),
#         type="date",
#         range=[one_week_ago, latest_date],  # Set the initial range to the last week
#     ),
#     xaxis2=dict(
#         rangeslider=dict(visible=True), type="date", range=[one_week_ago, latest_date]
#     ),
#     xaxis3=dict(
#         rangeslider=dict(visible=True), type="date", range=[one_week_ago, latest_date]
#     ),
#     xaxis5=dict(
#         rangeslider=dict(visible=True), type="date", range=[one_week_ago, latest_date]
#     ),
#     dragmode="zoom",  # 드래그 모드를 'zoom'으로 설정하여 차트 확대 가능
# )
# # Streamlit에 차트 표시
# st.plotly_chart(fig, use_container_width=True)


# 스토캐스틱 슬로우
# 지지저항

# Stocastic ->  MACD -> RSI -> 볼링저 지지저항
