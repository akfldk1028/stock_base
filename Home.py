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
# Hello!
            
Welcome to my DK Stock Portfolio!
            
Here are the apps I made:
            
- [x] [📃 StockBase](/StockBase)
- [x] [📃 MinutesTrading](/minutesTrading)
- [x] [📃 FuturesTrading](/futuresTrading)
"""
)
