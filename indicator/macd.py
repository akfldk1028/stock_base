import numpy as np


# 현재 날짜와 시간을 기준으로 일정 범위를 설정 (예: 최근 2일간의 데이터)
# Class for MACD Analysis
class MACDAnalyzer:
    def __init__(self, df):
        self.df = df
        # print(self.df["Close"])
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
