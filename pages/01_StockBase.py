import streamlit as st
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from binance.client import Client
from pandas.tseries.offsets import DateOffset
import math
import numpy as np
from indicator.macd import MACDAnalyzer
from indicator.rsi import RSIAnalyzer
from indicator.chartplotter import ChartPlotter


class CoinDataFetcher:
    def __init__(self, client):
        self.client = client

    def get_all_coin_data(self, interval):
        exchange_info = self.client.get_exchange_info()
        symbols = [
            item["symbol"]
            for item in exchange_info["symbols"]
            if item["status"] == "TRADING"
        ]

        coin_data = {}
        for symbol in symbols:
            try:
                klines = self.get_data(self.client, symbol, interval, "1 day ago UTC")
                coin_data[symbol] = pd.DataFrame(
                    klines, columns=["Open", "High", "Low", "Close"]
                )
                self.process_data(coin_data[symbol])
            except Exception as e:
                print(f"Error retrieving data for {symbol}: {e}")

        return coin_data


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

    fetcher = CoinDataFetcher(client)
    print(fetcher)
    # Data fetching
    symbol = "ETHUSDT"
    interval = Client.KLINE_INTERVAL_5MINUTE
    # KLINE_INTERVAL_5MINUTE
    # KLINE_INTERVAL_1DAY
    # "30 day ago UTC"
    start_str = "10 day ago UTC"
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


if __name__ == "__main__":
    main()


# Stocastic ->  MACD -> RSI -> 볼링저 지지저항
