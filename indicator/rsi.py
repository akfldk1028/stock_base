import pandas as pd


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
