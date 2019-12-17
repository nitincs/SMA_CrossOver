import matplotlib.pyplot as plt

import pandas as pd


df = pd.read_csv("Data/1.csv", names=['ind', 'open', 'high', 'low', 'close', 'volume'])

M20 = df["close"].rolling(20).mean()
M10 = df["close"].rolling(10).mean()

plt.plot(df["ind"][-100:], df["close"][-100:], label="close")
plt.plot(df["ind"][-100:], M20[-100:], label="M20")
plt.plot(df["ind"][-100:], M10[-100:], label="M10")

# hello

plt.legend()
plt.show()

# op1

