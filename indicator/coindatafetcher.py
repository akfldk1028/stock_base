from indicator.macd import MACDAnalyzer
from indicator.rsi import RSIAnalyzer
from indicator.chartplotter import ChartPlotter
import streamlit as st
import pandas as pd
from pandas.tseries.offsets import DateOffset

class CoinDataFetcher:
    def __init__(self, client, whitelisted_symbols):
        self.client = client
        self.whitelisted_symbols = whitelisted_symbols

        self.coin_data = {}  # coin_data 속성 초기화
        self.exist_coin_data = {}

    def get_all_coin_data(self, interval):
        exchange_info = self.client.get_exchange_info()
        symbols = [
            item["symbol"]
            for item in exchange_info["symbols"]
            if item["status"] == "TRADING"
            and item["symbol"].endswith("USDT")  # USDT로 끝나는 심볼만 선택
            and item["symbol"]
            in self.whitelisted_symbols  # Check if symbol is in whitelist
        ]
        valid_coin_data = {}

        # coin_data = {}
        for symbol in symbols:
            try:
                klines = CoinDataFetcher.get_data(
                    self.client, symbol, interval, "1 day ago UTC"
                )
                df = pd.DataFrame(klines, columns=["Open", "High", "Low", "Close"])
                self.add_bollinger_bands(df)
                self.process_data(df)
                self.coin_data[symbol] = df

                # Check the absolute value of the last row's MACD and Signal
                if (abs(df["MACD"].iloc[-1]) >= 0.001) and (
                    abs(df["Signal"].iloc[-1]) >= 0.001
                ):
                    valid_coin_data[symbol] = df
            except Exception as e:
                print(f"Error retrieving data for {symbol}: {e}")

        self.coin_data = valid_coin_data

        return self.coin_data

    @staticmethod
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

        return df

    def add_bollinger_bands(self, df, period=18, multiplier=2.0):
        df["MA"] = df["Close"].rolling(window=period).mean()
        df["STD"] = df["Close"].rolling(window=period).std()
        df["Upper"] = df["MA"] + (df["STD"] * multiplier)
        df["Lower"] = df["MA"] - (df["STD"] * multiplier)

    def process_data(self, df):
        macd_analyzer = MACDAnalyzer(df)

        df["MACD_cross_up"] = macd_analyzer.macd_cross_up()
        df[
            "MACD_cross_JustBefore_Reversal"
        ] = macd_analyzer.macd_cross_JustBefore_Reverse()
        ########
        rsi_analyzer_14 = RSIAnalyzer(df, 14)
        df["RSI_14"] = rsi_analyzer_14.df["RSI"]  # Assign the calculated RSI values
        df["RSI_9"] = RSIAnalyzer(df, 9)

        df[
            "RSI_recently_below_40"
        ] = rsi_analyzer_14.check_rsi_below_threshold_over_multiple_ranges(
            df, "RSI_14", [40], [(0, 0)]
        )
        df[
            "RSI_recently_below_65"
        ] = rsi_analyzer_14.check_rsi_below_threshold_over_multiple_ranges(
            df, "RSI_14", [65], [(0, 2)]
        )
        df[
            "RSI_recently_above_40"
        ] = rsi_analyzer_14.check_rsi_below_threshold_over_multiple_ranges(
            df, "RSI_14", [40], [(0, 0)]
        )

        df["RSI_above_rising"] = df["RSI_recently_below_65"] & (
            df["RSI_14"] > df["RSI_14"].shift(1)
        )

        df["buy_condition"] = (
            (df["MACD_cross_up"] & (df["RSI_recently_below_40"]))
            | (df["MACD_cross_up"] & df["RSI_above_rising"])
            | (df["MACD_cross_JustBefore_Reversal"] & df["RSI_recently_above_40"])
        )
        df[
            "MACD_bounce_down_with_negative_slope"
        ] = macd_analyzer.macd_bounce_down_with_negative_slope()
        df["MACD_cross_down"] = macd_analyzer.macd_cross_down()
        # data_15m["MACD_cross_down2"] = macd_analyzer.macd_cross_down2()
        df[
            "MACD_cross_JustBefore_Reversal_below"
        ] = macd_analyzer.macd_cross_JustBefore_Reversal_below()
        ################

        df[
            "RSI_recently_above_60"
        ] = rsi_analyzer_14.check_rsi_above_threshold_over_multiple_ranges(
            df, "RSI_14", [60], [(0, 2)]
        )
        df[
            "RSI_recently_above_45"
        ] = rsi_analyzer_14.check_rsi_above_threshold_over_multiple_ranges(
            df, "RSI_14", [45], [(0, 2)]
        )
        df["RSI_below_falling"] = df["RSI_recently_above_45"] & (
            df["RSI_14"] < df["RSI_14"].shift(1)
        )

        # Define sell condition with the updated RSI checks
        df["sell_condition"] = (
            (df["MACD_cross_down"] & df["RSI_recently_above_60"])
            | (df["MACD_cross_down"] & df["RSI_below_falling"])
            | (df["MACD_cross_JustBefore_Reversal_below"] & df["RSI_recently_above_60"])
        )

        # buy_signals = extract_signals(data_15m, "buy_condition")
        # sell_signals = extract_signals(data_15m, "sell_condition")

        self.calculate_moving_average(df, "Close", 18)
        self.calculate_bullish_indicator(df)
        self.define_keep_condition(df, "MA18", "Upper")
        self.compute_keep_signal(df, "buy_condition", "sell_condition")

        return df

    def update_data_for_holdings(self, holdings):
        # 보유 자산에 대한 데이터 업데이트
        for asset in holdings:
            symbol = f"{asset}USDT"
            try:
                # 새로운 데이터 수집
                klines = self.get_data(self.client, symbol, "1h", "1 day ago UTC")
                df = pd.DataFrame(klines, columns=["Open", "High", "Low", "Close"])
                self.add_bollinger_bands(df)
                self.process_data(df)

                # 기존에 같은 심볼이 존재한다면 삭제 후 새로운 데이터로 대체
                self.exist_coin_data = df
                if symbol in self.coin_data:
                    del self.coin_data[symbol]
                self.coin_data[symbol] = df
            except Exception as e:
                print(f"Error retrieving data for {symbol}: {e}")

    def extract_signals(df, condition_col):
        return df[df[condition_col]]

    def calculate_moving_average(self, df, column, window):
        df[f"MA{window}"] = df[column].rolling(window=window).mean()

    def calculate_bullish_indicator(self, df):
        df["Bullish"] = df["Close"] > df["Open"]

    def define_keep_condition(self, df, ma_column, upper_band_column):
        df["Keep"] = (df["Close"] >= df[ma_column]) | (
            df["Close"] >= df[upper_band_column]
        )

    def compute_keep_signal(self, df, buy_condition_col, sell_condition_col):
        df["Keep_Signal"] = False
        for i in df.index:
            if df.loc[i, buy_condition_col]:
                j = i
                one_hour = DateOffset(hours=1)
                while j in df.index and not df.loc[j, sell_condition_col]:
                    if df.loc[j, "Keep"]:
                        df.loc[j, "Keep_Signal"] = True
                    j += one_hour

    def __str__(self):
        output = "Coin Data Summary:\n"
        for symbol, df in self.coin_data.items():
            output += f"\nSymbol: {symbol}\n"
            output += f"Data Points: {len(df)}\n"
            output += f"Latest Close: {df['Close'].iloc[-1]}\n"
            output += f"Latest MACD: {df['MACD'].iloc[-1]:.8f}\n"
            output += f"Latest Signal: {df['Signal'].iloc[-1]:.8f}\n"
            output += f"Latest RSI: {df['RSI_14'].iloc[-1]:.2f}\n"
            output += f"Latest Buy Condition: {'Yes' if df['buy_condition'].iloc[-1] else 'No'}\n"
            output += f"Latest Sell Condition: {'Yes' if df['sell_condition'].iloc[-1] else 'No'}\n"
            output += f"Latest Keep Condition: {'Yes' if df['Keep_Signal'].iloc[-1] else 'No'}\n"  # Keep 상태 추가

        return output

    def get_summary_dataframe(self):
        summary_data = []
        for symbol, df in self.coin_data.items():
            summary = {
                "Symbol": symbol,
                "Data Points": len(df),
                "Latest Close": df["Close"].iloc[-1],
                "Latest MACD": df["MACD"].iloc[-1],
                "Latest Signal": df["Signal"].iloc[-1],
                "Latest RSI": df["RSI_14"].iloc[-1],
                "Latest Buy Condition": "Yes" if df["buy_condition"].iloc[-1] else "No",
                "Latest Sell Condition": "Yes"
                if df["sell_condition"].iloc[-1]
                else "No",
                "Latest Keep Condition": "Yes" if df["Keep_Signal"].iloc[-1] else "No",
            }
            summary_data.append(summary)

        return pd.DataFrame(summary_data)
