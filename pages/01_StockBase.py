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
# Class for MACD Analysis
class MACDAnalyzer:
    def __init__(self, df):
        self.df = df
        print(self.df["Close"])
        self.df["MACD"], self.df["Signal"] = self.calculate_macd(self.df["Close"])
        self.compute_macd()

    def calculate_macd(self, data, slow=26, fast=12, signal=9):
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    # MACD 계산 함수
    def compute_macd(self, slow=26, fast=12, signal=9):
        self.df["MACD_diff"] = self.df["MACD"] - self.df["Signal"]
        # Calculate the difference in MACD values between the current and the previous row
        self.df["MACD_slope"] = self.df["MACD"].diff()
        # Check the sign of the MACD_slope to determine if it's going up or down
        self.df["MACD_direction"] = self.df["MACD_slope"].apply(
            lambda x: "up" if x > 0 else ("down" if x < 0 else "flat")
        )
        self.df["MACD_bounce_up_with_positive_slope"] = (
            (self.df["MACD"].shift(2) > self.df["Signal"].shift(2))
            & (
                (self.df["MACD"].shift(1) < self.df["Signal"].shift(1))
                | (self.df["MACD"].shift(1) > self.df["Signal"].shift(1))
            )
            & (self.df["MACD"] > self.df["Signal"])  # 바로 이전에는 MACD가 Signal 아래였음
            & (  # 현재는 MACD가 Signal 위에 있음
                self.df["MACD_direction"] == "up"
            )  # MACD의 기울기가 양수, 즉 상승세
        )

        self.df["MACD_diff_above1"] = np.round(self.df["MACD_diff"].abs(), 1) >= 0.5
        # MACD_diff가 특정 양의 값(여기서는 0.5) 이상인 경우를 찾습니다.
        self.df["MACD_diff_error"] = (
            self.df["MACD"].shift(1) - self.df["Signal"].shift(1)
        ).abs() <= 0.3

    # def check_macd_condition(self, shifts, condition):
    #     conditions = [
    #         self.df["MACD"].shift(i) - self.df["Signal"].shift(i)
    #         for i in range(1, shifts + 1)
    #     ]
    #     return np.logical_and.reduce([condition(c) for c in conditions])

    def check_macd_below_signal(self, shifts):
        """
        Check if MACD was below the Signal line for the specified number of previous points.
        """
        conditions = [
            self.df["MACD"].shift(i) < self.df["Signal"].shift(i)
            for i in range(1, shifts + 1)
        ]
        return np.logical_and.reduce(conditions)

    def check_macd_above_signal(self, shifts):
        """
        Check if MACD was above the Signal line for the specified number of previous points.
        """
        conditions = [
            self.df["MACD"].shift(i) > self.df["Signal"].shift(i)
            for i in range(1, shifts + 1)
        ]
        return np.logical_and.reduce(conditions)

    def macd_cross_up(self):
        return (
            (self.df["MACD"] > self.df["Signal"]) & self.check_macd_below_signal(3)
        ) | self.df["MACD_bounce_up_with_positive_slope"]

    def macd_cross_JustBefore_Reverse(self):
        return (
            (self.check_macd_below_signal(3))
            & self.df["MACD_diff_error"]
            & (self.df["MACD"] > self.df["Signal"])
        )

    def macd_bounce_down_with_negative_slope(self):
        return (
            (self.df["MACD"].shift(2) < self.df["Signal"].shift(2))
            & (
                (self.df["MACD"].shift(1) > self.df["Signal"].shift(1))
                | (self.df["MACD"].shift(1) < self.df["Signal"].shift(1))
            )
            & (self.df["MACD"] < self.df["Signal"])  # MACD is currently below Signal
            & (self.df["MACD_direction"] == "down")  # MACD direction is negative
        )

    def macd_cross_down(self):
        return (
            (self.df["MACD"] < self.df["Signal"])
            & (self.check_macd_above_signal(3) | self.check_macd_above_signal(2))
        ) | self.macd_bounce_down_with_negative_slope()

    def macd_cross_down2(self):
        return (
            (self.df["MACD"].shift(1) > self.df["Signal"].shift(1))
            & (self.df["MACD"].shift(2) < self.df["Signal"].shift(2))
            & (self.df["MACD"].shift(3) > self.df["Signal"].shift(3))
        ) & (self.df["MACD"] < self.df["Signal"])

    def macd_cross_JustBefore_Reversal_below(self):
        return (
            self.check_macd_above_signal(3)
            & self.df["MACD_diff_error"]
            & (self.df["MACD"] < self.df["Signal"])
        )


class RSIAnalyzer:
    def __init__(self, df, window=14):
        self.df = df
        self.window = window
        self.compute_rsi()

    def compute_rsi(self):
        diff = self.df["Close"].diff(1)
        gain = diff.where(diff > 0, 0).rolling(window=self.window).mean()
        loss = -diff.where(diff < 0, 0).rolling(window=self.window).mean()
        rs = gain / loss
        self.df["RSI"] = 100 - (100 / (1 + rs))

    def check_rsi_above_threshold_over_multiple_ranges(
        self, df, column, thresholds, period_ranges
    ):
        """
        # data_15m["RSI_14"] < 40  (data_15m, "RSI_14", [40], [(0, 0)])
        Check if the RSI values fall below specified thresholds over multiple ranges of periods.

        :param df: DataFrame containing the data
        :param column: The column name to check (e.g., "RSI_14")
        :param thresholds: List of thresholds to check against
        :param period_ranges: List of tuples, where each tuple contains the start and end of a period range
        :return: A boolean Series indicating whether the condition is met in any of the specified ranges
        """
        condition = pd.Series([False] * len(df), index=df.index)
        for threshold, (start, end) in zip(thresholds, period_ranges):
            for i in range(start, end + 1):
                condition |= df[column].shift(i) > threshold  # Direct comparison
        return condition

    def check_rsi_below_threshold_over_multiple_ranges(
        self, df, column, thresholds, period_ranges
    ):
        condition = pd.Series([False] * len(df), index=df.index)
        for threshold, (start, end) in zip(thresholds, period_ranges):
            for i in range(start, end + 1):
                condition |= df[column].shift(i) < threshold  # Direct comparison
        return condition

    def rsi_below_falling(self):
        return self.rsi_recently_above_45() & (self.df["RSI"] < self.df["RSI"].shift(1))


def extract_signals(df, condition_col):
    return df[df[condition_col]]


def calculate_moving_average(df, column, window):
    df[f"MA{window}"] = df[column].rolling(window=window).mean()


def calculate_bullish_indicator(df):
    df["Bullish"] = df["Close"] > df["Open"]


def define_keep_condition(df, ma_column, upper_band_column):
    df["Keep"] = (df["Close"] >= df[ma_column]) | (df["Close"] >= df[upper_band_column])


def compute_keep_signal(df, buy_condition_col, sell_condition_col):
    df["Keep_Signal"] = False
    for i in df.index:
        if df.loc[i, buy_condition_col]:
            j = i
            one_hour = DateOffset(hours=1)
            while j in df.index and not df.loc[j, sell_condition_col]:
                if df.loc[j, "Keep"]:
                    df.loc[j, "Keep_Signal"] = True
                j += one_hour


def set_initial_zoom_range(df, points=100):
    if len(df) > points:
        initial_start = df.index[-points]
    else:
        initial_start = df.index[0]
    initial_end = df.index[-1]
    return initial_start, initial_end


@st.cache_data(show_spinner="Embedding file...")
def get_data(_client, symbol, interval, start_str):
    candles = _client.get_historical_klines(symbol, interval, start_str)
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


def main():
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

    # Data fetching
    symbol = "ETHUSDT"
    interval = Client.KLINE_INTERVAL_1HOUR
    start_str = "30 day ago UTC"
    data_15m = get_data(client, symbol, interval, start_str)
    # print(data_15m)
    macd_analyzer = MACDAnalyzer(data_15m)

    data_15m["MACD_cross_up"] = macd_analyzer.macd_cross_up()
    data_15m[
        "MACD_cross_JustBefore_Reversal"
    ] = macd_analyzer.macd_cross_JustBefore_Reverse()
    ########
    rsi_analyzer_14 = RSIAnalyzer(data_15m, 14)
    data_15m["RSI_14"] = rsi_analyzer_14.df["RSI"]  # Assign the calculated RSI values
    data_15m["RSI_9"] = RSIAnalyzer(data_15m, 9)

    data_15m[
        "RSI_recently_below_40"
    ] = rsi_analyzer_14.check_rsi_below_threshold_over_multiple_ranges(
        data_15m, "RSI_14", [40], [(0, 0)]
    )
    data_15m[
        "RSI_recently_below_65"
    ] = rsi_analyzer_14.check_rsi_below_threshold_over_multiple_ranges(
        data_15m, "RSI_14", [65], [(0, 2)]
    )
    data_15m[
        "RSI_recently_above_40"
    ] = rsi_analyzer_14.check_rsi_below_threshold_over_multiple_ranges(
        data_15m, "RSI_14", [40], [(0, 0)]
    )

    data_15m["RSI_above_rising"] = data_15m["RSI_recently_below_65"] & (
        data_15m["RSI_14"] > data_15m["RSI_14"].shift(1)
    )
    # Check if RSI is above 50 and in an upward trend
    # data_15m["RSI_above_rising"] = data_15m["RSI_recently_below_65"] & (
    #     data_15m["RSI_14"] > data_15m["RSI_14"].shift(1)
    # )

    data_15m["buy_condition"] = (
        (data_15m["MACD_cross_up"] & (data_15m["RSI_recently_below_40"]))
        | (data_15m["MACD_cross_up"] & data_15m["RSI_above_rising"])
        | (
            data_15m["MACD_cross_JustBefore_Reversal"]
            & data_15m["RSI_recently_above_40"]
        )
    )
    data_15m[
        "MACD_bounce_down_with_negative_slope"
    ] = macd_analyzer.macd_bounce_down_with_negative_slope()
    data_15m["MACD_cross_down"] = macd_analyzer.macd_cross_down()
    # data_15m["MACD_cross_down2"] = macd_analyzer.macd_cross_down2()
    data_15m[
        "MACD_cross_JustBefore_Reversal_below"
    ] = macd_analyzer.macd_cross_JustBefore_Reversal_below()
    ################

    data_15m[
        "RSI_recently_above_60"
    ] = rsi_analyzer_14.check_rsi_above_threshold_over_multiple_ranges(
        data_15m, "RSI_14", [60], [(0, 2)]
    )
    data_15m[
        "RSI_recently_above_45"
    ] = rsi_analyzer_14.check_rsi_above_threshold_over_multiple_ranges(
        data_15m, "RSI_14", [45], [(0, 2)]
    )
    data_15m["RSI_below_falling"] = data_15m["RSI_recently_above_45"] & (
        data_15m["RSI_14"] < data_15m["RSI_14"].shift(1)
    )

    # Define sell condition with the updated RSI checks
    data_15m["sell_condition"] = (
        (data_15m["MACD_cross_down"] & data_15m["RSI_recently_above_60"])
        | (data_15m["MACD_cross_down"] & data_15m["RSI_below_falling"])
        | (
            data_15m["MACD_cross_JustBefore_Reversal_below"]
            & data_15m["RSI_recently_above_60"]
        )
    )

    buy_signals = extract_signals(data_15m, "buy_condition")
    sell_signals = extract_signals(data_15m, "sell_condition")

    calculate_moving_average(data_15m, "Close", 18)
    calculate_bullish_indicator(data_15m)
    define_keep_condition(data_15m, "MA18", "Upper")
    compute_keep_signal(data_15m, "buy_condition", "sell_condition")

    initial_start, initial_end = set_initial_zoom_range(data_15m, 100)
    latest_date = data_15m.index.max()

    chart_plotter = ChartPlotter(data_15m)
    chart_plotter.add_buy_sell_signals(buy_signals, sell_signals)
    chart_plotter.add_keep_signals_to_chart()
    chart_plotter.add_rsi()
    chart_plotter.add_macd()

    st.plotly_chart(
        chart_plotter.render_chart(initial_start, initial_end), use_container_width=True
    )


class ChartPlotter:
    def __init__(self, df):
        self.df = df
        self.fig = make_subplots(
            rows=5,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Candlestick", "Moving Averages", "RSI", "MACD"),
        )
        self.setup_chart()

    def add_candlestick(self):
        self.fig.add_trace(
            go.Candlestick(
                x=self.df.index,
                open=self.df["Open"],
                high=self.df["High"],
                low=self.df["Low"],
                close=self.df["Close"],
                name="Candlestick",
            ),
            row=1,
            col=1,
        )

    def add_moving_averages(self, ma_list):
        colors = ["blue", "green", "purple", "black"]
        for i, ma in enumerate(ma_list):
            self.fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df[f"MA{ma}"],
                    mode="lines",
                    name=f"MA{ma}",
                    line=dict(color=colors[i]),
                ),
                row=1,
                col=1,
            )

    def add_bollinger_bands(self):
        self.fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df["Upper"],
                mode="lines",
                name="Upper Band",
                line=dict(color="red"),
            ),
            row=1,
            col=1,
        )
        self.fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df["Lower"],
                mode="lines",
                name="Lower Band",
                line=dict(color="red"),
            ),
            row=1,
            col=1,
        )

    # keep_signals = data_15m[data_15m["Keep_Signal"]]

    # # Add traces for Keep signals
    # fig.add_trace(
    #     go.Scatter(
    #         x=keep_signals.index,
    #         y=keep_signals["Close"],
    #         mode="markers",
    #         marker=dict(color="orange", size=10, symbol="star"),
    #         name="Keep Signal",
    #     ),
    #     row=1,
    #     col=1,
    # )

    def add_keep_signals_to_chart(
        self,
    ):
        keep_signals = self.df[self.df["Keep_Signal"]]

        self.fig.add_trace(
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

    def add_buy_sell_signals(self, buy_signals, sell_signals):
        self.fig.add_trace(
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
        self.fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals["Open"],
                mode="markers",
                marker=dict(color="blue", size=15, symbol="triangle-down"),
                name="Sell Signal",
            ),
            row=1,
            col=1,
        )

    def add_rsi(self):
        self.fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df["RSI_14"],
                mode="lines",
                name="RSI 14",
                line=dict(color="blue"),
            ),
            row=3,
            col=1,
        )
        # Additional RSI traces can be added here if needed

    def add_macd(self):
        self.fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df["MACD"],
                mode="lines",
                name="MACD",
                line=dict(color="red"),
            ),
            row=5,
            col=1,
        )
        self.fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df["Signal"],
                mode="lines",
                name="Signal Line",
                line=dict(color="pink"),
            ),
            row=5,
            col=1,
        )

    def setup_chart(self):
        self.add_candlestick()
        self.add_moving_averages([18, 56, 112, 224])
        self.add_bollinger_bands()
        # Add other chart elements like buy/sell signals, RSI, MACD here
        # You can also add these elements outside this function if they depend on dynamic data

    def render_chart(self, start_date, end_date):
        # Set initial zoom and update layout
        # initial_start, initial_end = set_initial_zoom_range(self.df, 100)
        self.fig.update_layout(
            height=4500,
            title="15 Minute BTCUSDT Chart",
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date",
                range=[start_date, end_date],
            ),
            xaxis2=dict(
                rangeslider=dict(visible=True),
                type="date",
                range=[start_date, end_date],
            ),
            xaxis3=dict(
                rangeslider=dict(visible=True),
                type="date",
                range=[start_date, end_date],
            ),
            xaxis5=dict(
                rangeslider=dict(visible=True),
                type="date",
                range=[start_date, end_date],
            ),
            dragmode="zoom",
        )
        return self.fig


if __name__ == "__main__":
    main()


# 스토캐스틱 슬로우
# 지지저항

# Stocastic ->  MACD -> RSI -> 볼링저 지지저항
