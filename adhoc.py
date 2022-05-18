# %%
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import SpectralEmbedding
from tsmoothie.smoother import *


# %%

data = pd.read_csv("./data.txt", header=0, sep="\t", usecols=["Elapsed Time (s)", "Current (A)"])[
    10000:15000
]  # names=["Current (A)", "Elapsed Time (s)"])
data.shape
# %%
# smoother = ExponentialSmoother(window_len=50, alpha=0.1)
# smoother = PolynomialSmoother(degree=10)
# smoother = SpectralSmoother(smooth_fraction=0.1, pad_len=20)
smoother = ConvolutionSmoother(window_len=10, window_type="hanning")
smoother.smooth(data["Current (A)"])
# data = data[20:]

data["Current (A) smoothed"] = smoother.smooth_data[0]

# %%

px.line(data, x="Elapsed Time (s)", y=["Current (A)", "Current (A) smoothed"])
#%%
poly1 = np.polyfit(x=data["Elapsed Time (s)"], y=data["Current (A)"], deg=2)
y1 = np.polyval(poly1, data["Elapsed Time (s)"])
data["y1"] = y1

# %%
poly2 = np.polyfit(x=data["Elapsed Time (s)"][:300], y=data["Current (A)"][:300], deg=1)
y2 = np.polyval(poly2, data["Elapsed Time (s)"])
data["y2"] = y2

# %%
px.line(data, x="Elapsed Time (s)", y=["Current (A)", "y1", "y2"])
