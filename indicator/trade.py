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
        elif trade_type == "BUY":
            account.execute_trade(symbol, trade_type, coin_amount)
            account.track_trade(symbol, trade_type, coin_amount, current_price)

    def show_active_trades(self):
        return self.active_trades, self.tokens

    def check_tocken_count(self):
        return self.tokens >= self.max_tokens
