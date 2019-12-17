import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2


def relative_strength_index(df, n):
    """Calculate Relative Strength Index(RSI) for given data.

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        UpMove = df.loc[i + 1, 'high'] - df.loc[i, 'high']
        DoMove = df.loc[i, 'low'] - df.loc[i + 1, 'low']
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
    RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))
    df = df.join(RSI)
    return df


import pandas as pd

df = pd.read_csv("Data/1.csv", names=['ind', 'open', 'high', 'low', 'close', 'volume'])
df = relative_strength_index(df, 14)

df["close"] = (df["close"] - df["close"].min()) / (df["close"].max() - df["close"].min())
M20 = df["close"].rolling(20).mean()
M10 = df["close"].rolling(10).mean()

# print(RSI)
plt.plot(df["ind"][-100:], df["RSI_14"][-100:])
plt.show()
