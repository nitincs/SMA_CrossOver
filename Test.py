import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv("Data/1.csv", names=['ind', 'open', 'high', 'low', 'close', 'volume'])

plt.plot(df["ind"], df["close"])
plt.legend()
plt.show()
