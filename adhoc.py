# %%
import pandas as pd
import plotly.express as px

# %%

data = pd.read_csv(
    "./data.txt", header=0, sep="\t", usecols=["Elapsed Time (s)", "Current (A)"]
)  # names=["Current (A)", "Elapsed Time (s)"])
px.line(data, x="Elapsed Time (s)", y="Current (A)")
