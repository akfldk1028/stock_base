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
from datetime import datetime
import pprint
from indicator.whitelist import WHITELISTED_SYMBOLS


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


class Trade:
    def __init__(self, summary_df):
        self.summary_df = summary_df
        self.active_trades = set()  # Stores active trade symbols
        self.tokens = 0  # Current number of tokens
        self.max_tokens = 3  # Maximum number of tokens

    def revoke_tokens(self):
        return self.tokens

    def reset_active_trades(self):
        self.active_trades = ()
        return self.active_trades

    def evaluate_trades(self, account, fetcher):
        current_holdings = account.get_holdings_above_threshold()
        fetcher.update_data_for_holdings(current_holdings.keys())

        for asset in current_holdings:
            symbol = asset + "USDT"
            print(symbol)
            if symbol in fetcher.coin_data:
                sell_condition = (
                    fetcher.coin_data[symbol]["sell_condition"].iloc[-1] == "Yes"
                )
                if sell_condition:
                    amount_current_holdings_by_symbol = account.get_quantity_of_symbol(
                        symbol
                    )
                    self.execute_trade(
                        account, symbol, "SELL", amount_current_holdings_by_symbol
                    )

                elif (
                    not sell_condition
                    and symbol not in self.active_trades
                    and self.tokens < self.max_tokens
                ):
                    self.active_trades.add(symbol)

        self.tokens = len(self.active_trades)

        if self.tokens < self.max_tokens:
            buy_candidates = self.summary_df[
                self.summary_df["Latest Buy Condition"] == "Yes"
            ]
            sorted_candidates = buy_candidates.sort_values("Latest RSI").head(
                self.max_tokens - self.tokens
            )

            # Execute trades based on available tokens
            for symbol in sorted_candidates["Symbol"]:
                if symbol not in self.active_trades:
                    coin_amount = account.calculate_quantity_to_buy(
                        symbol, self.max_tokens
                    )
                    self.execute_trade(account, symbol, "BUY", coin_amount)
                    self.active_trades.add(symbol)
                    self.tokens = len(self.active_trades)

    def execute_trade(self, account, symbol, trade_type, coin_amount):
        current_price = account.get_current_price(symbol)
        if trade_type == "SELL":
            account.execute_trade(symbol, trade_type, coin_amount)
            account.track_trade(symbol, trade_type, coin_amount, current_price)
            #  def track_trade(self, symbol, trade_type, amount, price):
        elif trade_type == "BUY":
            account.execute_trade(symbol, trade_type, coin_amount)
            account.track_trade(symbol, trade_type, coin_amount, current_price)

    # def execute_trades(self, account, fetcher):
    #     usdt_balance = account.get_tradeable_usdt_balance()
    #     amount_per_trade = (usdt_balance / 2) / self.max_tokens

    #     print(f"{self.active_trades} + phase2")

    #     for symbol in self.active_trades:
    #         print(f"{fetcher.coin_data[symbol]} + phase2")
    #         if symbol == fetcher.coin_data[symbol]:
    #             return
    #         else:
    #             summary_row = self.summary_df[self.summary_df["Symbol"] == symbol].iloc[
    #                 0
    #             ]

    #             current_price = account.get_current_price(symbol)
    #             # print(f"{current_price} + phase2")
    #             if summary_row["Latest Buy Condition"] == "Yes":
    #                 if usdt_balance >= amount_per_trade:
    #                     # Execute buy order
    #                     account.execute_trade(symbol, "BUY", amount_per_trade)
    #                     print(f"Bought {symbol} for {amount_per_trade} USDT")

    #                     account.track_trade(
    #                         symbol, "BUY", amount_per_trade, current_price
    #                     )
    #             elif summary_row["Latest Sell Condition"] == "Yes":
    #                 # Execute sell order
    #                 account.execute_trade(symbol, "SELL", amount_per_trade)
    #                 print(f"Sold {symbol} for {amount_per_trade} USDT")

    #                 self.active_trades.remove(symbol)
    #                 self.tokens -= 1
    #                 account.track_trade(symbol, "SELL", amount_per_trade, current_price)

    def show_active_trades(self):
        return self.active_trades, self.tokens

    def check_tocken_count(self):
        return self.tokens >= self.max_tokens


class BinanceAccount:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)
        self.trade_history = []  # Initialize trade history
        self.initial_balance = (
            self.get_tradeable_usdt_balance()
        )  # Record initial balance

    def get_holdings_above_threshold(self, threshold=3.0):
        holdings = {}
        account_info = self.client.get_account()
        balances = account_info["balances"]

        # 유효한 심볼 목록 얻기
        exchange_info = self.client.get_exchange_info()
        valid_symbols = {
            item["symbol"]
            for item in exchange_info["symbols"]
            if item["status"] == "TRADING"
        }

        for balance in balances:
            asset = balance["asset"]
            free_balance = float(balance["free"])
            if asset != "USDT":
                symbol = f"{asset}USDT"
                # 유효한 심볼만 처리
                if symbol in valid_symbols:
                    current_price = self.get_current_price(symbol)
                    if current_price:
                        usdt_value = free_balance * current_price
                        if usdt_value >= threshold:
                            holdings[asset] = usdt_value
        return holdings

    def get_server_time(self):
        # Get server time in a readable format
        server_time = self.client.get_server_time()
        readable_time = datetime.fromtimestamp(server_time["serverTime"] / 1000)
        return readable_time

    def get_roi(self):
        # Calculate ROI
        current_balance = self.get_tradeable_usdt_balance()
        if self.initial_balance == 0:
            return 0  # Prevent division by zero
        roi = ((current_balance - self.initial_balance) / self.initial_balance) * 100
        return roi

    def get_quantity_of_symbol(self, symbol):
        """특정 심볼에 해당하는 코인의 수량을 반환합니다."""
        account_info = self.client.get_account()
        balances = account_info["balances"]

        for balance in balances:
            if balance["asset"] == symbol:
                return float(balance["free"])

        return 0.0  # 해당 심볼이 없는 경우 0 반환

    def calculate_quantity_to_buy(self, symbol, max_tokens):
        """구매할 코인의 수량을 계산합니다."""
        usdt_balance = self.get_tradeable_usdt_balance()  # 현재 USDT 잔액
        amount_per_token = (usdt_balance / 2) / max_tokens  # 토큰당 할당 금액

        current_price = self.get_current_price(symbol)  # 현재 코인 가격
        if current_price == 0:
            return 0
        else:
            symbol_info = self.client.get_symbol_info(symbol)
            step_size = 0.0
            min_notional = 0.0
            for filter in symbol_info["filters"]:
                if filter["filterType"] == "LOT_SIZE":
                    step_size = float(filter["stepSize"])
                if filter["filterType"] == "MIN_NOTIONAL":
                    min_notional = float(filter["minNotional"])

            # 최신 가격 가져오기
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker["price"])

            # 수량 계산 및 반올림
            quantity = amount_per_token / current_price
            precision = int(round(-math.log(step_size, 10), 0))
            quantity = round(quantity, precision)

        return quantity

    def get_account_balance(self):
        # Get account balance
        account_info = self.client.get_account()
        balances = account_info["balances"]
        return pd.DataFrame(balances)

    def get_trade_history(self, symbol):
        # Get recent trades for a given symbol
        trades = self.client.get_my_trades(symbol=symbol)
        return pd.DataFrame(trades)

    def get_specific_balance(self, assets):
        # Get account balance for specific assets (e.g., 'USDT' and 'BTC')
        account_info = self.client.get_account()
        balances = account_info["balances"]
        balance_df = pd.DataFrame(balances)
        return balance_df[balance_df["asset"].isin(assets)]

    def get_tradeable_usdt_balance(self):
        # Fetches the USDT balance that is available for trading
        balance_info = self.get_specific_balance(["USDT"])
        if not balance_info.empty:
            return float(balance_info["free"].iloc[0])
        else:
            return 0.0

    def get_current_price(self, symbol):
        """Fetch the current price of a symbol."""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception as e:
            print(f"Error fetching current price for {symbol}: {e}")
            return None

    def execute_trade(self, symbol, trade_type, amount):
        try:
            min_notional = 0.0

            ticker = self.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker["price"])
            quantity = amount
            # 최소 거래 금액 확인
            if quantity * current_price < min_notional:
                print(f"Trade amount for {symbol} is below the minimum notional value.")
                return None
            # Execute market order
            if trade_type == "BUY":
                order = self.client.create_order(
                    symbol=symbol,
                    side=Client.SIDE_BUY,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity,
                )
                print(order)
            elif trade_type == "SELL":
                order = self.client.create_order(
                    symbol=symbol,
                    side=Client.SIDE_SELL,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity,
                )
                print(order)
            else:
                raise ValueError("Invalid trade type. Must be 'BUY' or 'SELL'.")

            self.track_trade(
                symbol, trade_type, amount, current_price
            )  # Track the trade
            return order
        except Exception as e:
            print(f"An error occurred during trade execution: {e}")
            return None

    def track_trade(self, symbol, trade_type, amount, price):
        # Track each trade
        roi = self.get_roi()  # Calculate ROI at the time of the trade
        trade = {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": symbol,
            "Type": trade_type,
            "Amount": amount,
            "Price": price,
            "Total": amount * price,
            "ROI": roi,  # Include ROI in the trade record
        }
        self.trade_history.append(trade)

    def get_trade_history(self):
        # Return the trade history
        return pd.DataFrame(self.trade_history)


def main():
    st.set_page_config(
        page_title="DK Stock Portfolio",
        page_icon="🤖",
    )

    st.markdown(
        """
    # Automatic Minutes Trading Bot
                
    """
    )

    api_key = st.secrets["Binan_API_KEY"]
    api_secret = st.secrets["Binan_SECRET_KEY"]
    account = BinanceAccount(api_key, api_secret)

    usdt_balance = account.get_tradeable_usdt_balance()
    print(f"Tradeable USDT balance: {usdt_balance}")
    client = Client(api_key, api_secret)

    fetcher = CoinDataFetcher(client, WHITELISTED_SYMBOLS)
    fetcher.get_all_coin_data(Client.KLINE_INTERVAL_1HOUR)
    # Filter for each signal condition
    summary_df = fetcher.get_summary_dataframe()

    trade_manager = Trade(summary_df)
    trade_manager.evaluate_trades(account, fetcher)

    # Display trade history with ROI
    st.markdown("## Trade History with ROI")
    trade_history_df = pd.DataFrame(account.trade_history)
    st.dataframe(trade_history_df)

    active_trades, tokens = trade_manager.show_active_trades()

    # Display active trades and token count in Streamlit
    st.markdown("## Active Trades")
    for symbol in active_trades:
        st.write(f"Symbol: {symbol}")
    st.write(f"Tokens: {tokens}")
    # active_trades = trade_manager.show_active_trades()
    # Display in Streamlit
    st.markdown("## Al Trades")
    st.dataframe(summary_df)
    st.markdown("## Buy Signals")
    buy_signals_df = summary_df[summary_df["Latest Buy Condition"] == "Yes"]
    st.dataframe(buy_signals_df)
    st.markdown("## Sell Signals")
    sell_signals_df = summary_df[summary_df["Latest Sell Condition"] == "Yes"]
    st.dataframe(sell_signals_df)
    # keep_signals_df = summary_df[summary_df["Latest Keep Condition"] == "Yes"]


if __name__ == "__main__":
    main()
# [TODO]
# WEBSOCKET 으로 UST(timezone)시간으로 실시간데이터 및 시간분석
