import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

ar = []

for i in range(1, 50):
    df = pd.read_csv("Data/" + str(i) + ".csv", names=['ind', 'open', 'high', 'low', 'close', 'volume'])

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
            # print(c1, c2, c3)
            pc = (c1 * c11 + c2 * c21 + c3 * c31) / (abs(c1) + abs(c2) + abs(c3))

            # print(df["close"][i - 9:i + 4])
            # print(cp, pc * 100)
            ar.append(pc * 100)

print(ar)
# plt.xlim([0, 9])
# plt.ylim([0, 1])
# plt.plot(range(13), M20[i - 9:i + 4], label="M20", linewidth=4, color="g")
# plt.plot(range(13), M10[i - 9:i + 4], label="M20", linewidth=4, color="b")
# plt.plot(range(13), df["close"][i - 9:i + 4], label="M20", linewidth=4, color="r")
# plt.show()

# plt.plot(df["ind"], M20, label="M20")
# plt.plot(df["ind"], M10, label="M20")
# plt.show()

#
#
# plt.rcParams['figure.figsize'] = [4, 2]
# #
# #
# # M20 = df["close"].rolling(20).mean()
# # M10 = df["close"].rolling(10).mean()
# #
# # VOl = df["volume"]
# #
# plt.bar(df["ind"][-10:], df["volume"][-10:], width=0.4, color='g')
# #
# # # # plt.plot(df["ind"][-10:], df["close"][-100:], label="close")
# # # plt.plot(df["ind"][-10:], M20[-10:], label="M20")
# # # plt.plot(df["ind"][-10:], M10[-10:], label="M10")
# # #
# # # # hello
# plt.axis('off')
# plt.savefig("vol.png", bbox_inches='tight', pad_inches=0)
# # plt.clf()
# # # plt.legend()
# # plt.show()
#
#
# import cv2
#
# img1 = cv2.imread('abc.png')
# img2 = cv2.imread('vol.png')
#
# print(np.shape(img1))
# print(np.shape(img2))
# # img2 = cv2.resize(img2, (310, 155))
# dst = np.concatenate((img1, img2), axis=0)
# dst = cv2.resize(dst, (500, 500))
# # cv2.imshow('image', dst)
# cv2.imwrite('nw.png', dst)
# # # cv2.waitKey(0)
