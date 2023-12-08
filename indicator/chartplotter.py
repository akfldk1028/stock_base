import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
