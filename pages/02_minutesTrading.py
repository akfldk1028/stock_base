import streamlit as st

import pandas as pd
from binance.client import Client
from pandas.tseries.offsets import DateOffset

from indicator.chartplotter import ChartPlotter
from indicator.whitelist import WHITELISTED_SYMBOLS
from indicator.coindatafetcher import CoinDataFetcher
from indicator.trade import Trade
from indicator.binanceAccount import BinanceAccount
import schedule
import time
import threading

import streamlit as st
import pandas as pd
from binance.client import Client
import schedule
import time
import threading

# 기존 클래스들 (CoinDataFetcher, Trade, BinanceAccount 등)을 여기에 포함시킵니다.


class TimeManager:
    def __init__(self):
        self.scheduler = schedule.Scheduler()
        self.trade_history = None
        self.active_trades = None
        self.tokens = None
        self.summary_df = None
        self.buy_signals_df = None
        self.sell_signals_df = None

        self.job()

    def job(self):
        # 여기에 실행할 코드를 넣습니다.
        print("Running scheduled job...")

        # API 키와 비밀번호를 설정합니다.
        api_key = st.secrets["Binan_API_KEY"]
        api_secret = st.secrets["Binan_SECRET_KEY"]

        # BinanceAccount, CoinDataFetcher, Trade 객체를 생성합니다.
        account = BinanceAccount(api_key, api_secret)
        client = Client(api_key, api_secret)

        fetcher = CoinDataFetcher(client, WHITELISTED_SYMBOLS)
        fetcher.get_all_coin_data(Client.KLINE_INTERVAL_1HOUR)

        self.summary_df = fetcher.get_summary_dataframe()

        trade_manager = Trade(fetcher.get_summary_dataframe())
        # 거래 매니저를 사용하여 거래를 평가합니다.
        trade_manager.evaluate_trades(account, fetcher)
        # 거래 정보를 출력합니다.
        print("Trade History:", account.trade_history)
        self.trade_history = pd.DataFrame(account.trade_history)
        self.active_trades, self.tokens = trade_manager.show_active_trades()

        print("Active Trades:", trade_manager.show_active_trades())
        self.buy_signals_df = self.summary_df[
            self.summary_df["Latest Buy Condition"] == "Yes"
        ]
        self.sell_signals_df = self.summary_df[
            self.summary_df["Latest Sell Condition"] == "Yes"
        ]

    def start_scheduler(self):
        self.scheduler.every(1).hours.do(self.job)

        while True:
            self.scheduler.run_pending()
            time.sleep(1)

    def run_in_thread(self):
        thread = threading.Thread(target=self.start_scheduler)
        thread.start()


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

    # api_key = st.secrets["Binan_API_KEY"]
    # api_secret = st.secrets["Binan_SECRET_KEY"]
    # account = BinanceAccount(api_key, api_secret)

    # usdt_balance = account.get_tradeable_usdt_balance()
    # print(f"Tradeable USDT balance: {usdt_balance}")
    # client = Client(api_key, api_secret)

    # fetcher = CoinDataFetcher(client, WHITELISTED_SYMBOLS)
    # fetcher.get_all_coin_data(Client.KLINE_INTERVAL_1HOUR)
    # # Filter for each signal condition
    # summary_df = fetcher.get_summary_dataframe()

    # trade_manager = Trade(summary_df)
    # trade_manager.evaluate_trades(account, fetcher)

    # # Display trade history with ROI
    # st.markdown("## Trade History with ROI")
    # trade_history_df = pd.DataFrame(account.trade_history)
    # st.dataframe(trade_history_df)

    # active_trades, tokens = trade_manager.show_active_trades()

    # Display active trades and token count in Streamlit
    # st.markdown("## Active Trades")
    # for symbol in active_trades:
    #     st.write(f"Symbol: {symbol}")
    # st.write(f"Tokens: {tokens}")
    # active_trades = trade_manager.show_active_trades()
    # Display in Streamlit
    # st.markdown("## Al Trades")
    # st.dataframe(summary_df)
    # st.markdown("## Buy Signals")
    # buy_signals_df = summary_df[summary_df["Latest Buy Condition"] == "Yes"]
    # st.dataframe(buy_signals_df)
    # st.markdown("## Sell Signals")
    # sell_signals_df = summary_df[summary_df["Latest Sell Condition"] == "Yes"]
    # st.dataframe(sell_signals_df)
    # keep_signals_df = summary_df[summary_df["Latest Keep Condition"] == "Yes"]

    # TimeManager 인스턴스 생성 및 스레드 시작
    time_manager = TimeManager()

    # TimeManager의 거래 이력 표시
    st.markdown("## Trade History with ROI")
    if time_manager.trade_history is not None:
        st.dataframe(time_manager.trade_history)
    else:
        st.write("No trade history available yet.")

    st.markdown("## Active Trades")
    if time_manager.active_trades is not None:
        for symbol in time_manager.active_trades:
            st.write(f"Symbol: {symbol}")
    else:
        st.write("No active_trade available yet.")

    if time_manager.tokens is not None:
        st.write(f"Tokens: {time_manager.tokens}")
    else:
        st.write("No tokens available yet.")

    st.write("TimeManager is running in the background.")

    st.markdown("## Al Trades")
    if time_manager.summary_df is not None:
        st.dataframe(time_manager.summary_df)

    st.markdown("## Buy Signals")
    if time_manager.buy_signals_df is not None:
        st.dataframe(time_manager.buy_signals_df)

    st.markdown("## Sell Signals")
    if time_manager.sell_signals_df is not None:
        st.dataframe(time_manager.sell_signals_df)

    time_manager.run_in_thread()
    st.write("TimeManager is running in the background.")


if __name__ == "__main__":
    main()


# [TODO]
# WEBSOCKET 으로 UST(timezone)시간으로 실시간데이터 및 시간분석
