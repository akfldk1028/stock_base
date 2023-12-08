from binance.client import Client
from pandas.tseries.offsets import DateOffset
from datetime import datetime
import math
import pandas as pd


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
