#!/usr/bin/env python
# encoding: utf-8

import streamlit as st
import pandas as pd
import plotly.express as px
from tsmoothie.smoother import ExponentialSmoother, SpectralSmoother, ConvolutionSmoother
from io import StringIO

st.set_page_config(
    page_title="Currencell", page_icon="random", layout="wide", initial_sidebar_state="auto", menu_items=None
)

st.title("Currencell")
st.markdown("What kind of wonders happen when electricity passes through a cell?")

st.header("Smoother Setup")
col1, col2, col3 = st.columns(3)

with col1:
    smoother_name = st.selectbox(
        label="Smoother", options=["Exponential Smoother", "Spectral Smoother", "Convolution Smoother"], index=1
    )


def parameter_panel(smoother_name):
    params = []
    if smoother_name == "Exponential Smoother":
        with col2:
            window_len = st.number_input(label="Window Length", min_value=1, max_value=100, value=2)
        with col3:
            alpha = st.number_input(label="Alpha", min_value=0.01, max_value=1.0, value=0.1)
        params.extend([window_len, alpha])

        smoother = ExponentialSmoother(window_len=window_len, alpha=alpha)

    elif smoother_name == "Spectral Smoother":
        with col2:
            pad_len = st.number_input(label="Pad Length", min_value=1, max_value=100, value=10)
        with col3:
            smooth_fraction = st.number_input(label="Smooth Fraction", min_value=0.01, max_value=1.0, value=0.1)
        smoother = SpectralSmoother(smooth_fraction=smooth_fraction, pad_len=pad_len)

    elif smoother_name == "Convolution Smoother":
        with col2:
            window_len = st.number_input(label="Window Length", min_value=1, max_value=100, vlaue=20)
        with col3:
            window_type = st.selectbox(
                label="Window Type", options=["ones", "hanning", "hamming", "bartlett", "blackman"], value="hanning"
            )
        smoother = ConvolutionSmoother(window_len=10, window_type=window_type)

    return {"smoother_name": smoother_name, "smoother": smoother}


smoother_meta = parameter_panel(smoother_name)


def smooth_process(data, smoother_meta):
    smoother = smoother_meta["smoother"]
    smoother.smooth(data["Current (A)"])
    smoothed_value = smoother.smooth_data[0]
    original_value = smoother.data[0]
    smoothed_df = pd.DataFrame(
        {
            "Elapsed Time (s)": data["Elapsed Time (s)"][-len(smoothed_value) :],
            "Smoothed Current (A)": smoothed_value,
            "Current (A)": original_value,
        }
    )
    return smoothed_df


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    default_file = uploaded_file
else:
    default_file = "./data.txt"

data = pd.read_csv(default_file, header=0, sep="\t", usecols=["Elapsed Time (s)", "Current (A)"])[6000:]
smoothed_data = smooth_process(data, smoother_meta)

fig = px.line(smoothed_data, x="Elapsed Time (s)", y=["Current (A)", "Smoothed Current (A)"])
fig.update_layout(showlegend=False)

st.plotly_chart(fig, use_container_width=True)

# st.table(data.head(20).style.format({"Current (A)": "{:.8f}", "Elapsed Time (s)": "{:.2f}"}))
