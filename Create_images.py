import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

from tqdm import tqdm
import pandas as pd


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


ar = []
kk = 1

for i in tqdm(range(1, 50)):
    df = pd.read_csv("Data/" + str(i) + ".csv", names=['ind', 'open', 'high', 'low', 'close', 'volume'])
    df = relative_strength_index(df, 14)

    df["close"] = (df["close"] - df["close"].min()) / (df["close"].max() - df["close"].min())
    M20 = df["close"].rolling(20).mean()
    M10 = df["close"].rolling(10).mean()

    for i in range(30, len(M20) - 3):

        p1 = M20[i] - M10[i]
        p2 = M20[i - 1] - M10[i - 1]
        p3 = M20[i - 2] - M10[i - 2]

        if p1 < 0 < p3 and p2 >= 0:
            cp = df["close"][i]

            c11 = df["close"][i + 1]
            c21 = df["close"][i + 2]
            c31 = df["close"][i + 3]

            c1 = (c11 - cp) / cp
            c2 = (c21 - cp) / cp
            c3 = (c31 - cp) / cp
            pc = (c1 * c11 + c2 * c21 + c3 * c31) / (abs(c1) + abs(c2) + abs(c3))
            pc = pc * 100

            plt.rcParams['figure.figsize'] = [4, 4]
            plt.xlim([0, 9])
            # plt.ylim([0, 1])
            plt.plot(range(10), M20[i - 9:i + 1], label="M20", linewidth=5, color="g")
            plt.plot(range(10), M10[i - 9:i + 1], label="M20", linewidth=5, color="b")
            plt.axis('off')
            plt.savefig("img/MA.png", bbox_inches='tight', pad_inches=0)
            plt.clf()

            plt.rcParams['figure.figsize'] = [4, 2]
            # plt.xlim([0, 9])
            plt.axis('off')
            plt.bar(range(10), df["volume"][i - 9:i + 1], width=0.4, color='r')
            plt.savefig("img/VL.png", bbox_inches='tight', pad_inches=0)
            plt.clf()

            plt.axis('off')
            plt.xlim([0, 9])

            plt.plot(range(10), df["RSI_14"][i - 9:i + 1], linewidth=2, color='m')
            plt.savefig("img/RSI.png", bbox_inches='tight', pad_inches=0)
            plt.clf()

            img1 = cv2.imread('img/MA.png')
            img2 = cv2.imread('img/VL.png')
            img3 = cv2.imread('img/RSI.png')
            img2 = cv2.resize(img2, (310, 150))
            img3 = cv2.resize(img3, (310, 200))
            dst = np.concatenate((img1, img3, img2), axis=0)
            dst = cv2.resize(dst, (256, 256))

            name = ""

            if pc < -25:
                name = 'img/y1/' + str(kk) + '.png'

            elif -25 <= pc < 0:
                name = 'img/y2/' + str(kk) + '.png'

            elif 0 <= pc < 25:
                name = 'img/y3/' + str(kk) + '.png'

            else:
                name = 'img/y4/' + str(kk) + '.png'

            nw = cv2.imwrite(name, dst)
            kk += 1

    #         break
    # break

# 17 y1
# 53 y4
