#!/usr/bin/env python
# encoding: utf-8

from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from tsmoothie.smoother import ConvolutionSmoother, ExponentialSmoother, PolynomialSmoother, SpectralSmoother

st.set_page_config(
    page_title="Currencell", page_icon="random", layout="wide", initial_sidebar_state="auto", menu_items=None
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
            window_len = st.number_input(label="Window Length", min_value=1, max_value=100, value=20)
        with col3:
            window_type = st.selectbox(
                label="Window Type", options=["ones", "hanning", "hamming", "bartlett", "blackman"], value="hanning"
            )
        smoother = ConvolutionSmoother(window_len=10, window_type=window_type)

    return {"smoother_name": smoother_name, "smoother": smoother}


def smooth_process(data, smoother_meta):
    smoother = smoother_meta["smoother"]
    smoother.smooth(data["Current (A)"])
    smoothed_value = smoother.smooth_data[0]
    original_value = smoother.data[0]
    smoothed_df = pd.DataFrame(
        {
            "Elapsed Time (s)": data["Elapsed Time (s)"][-len(smoothed_value) :],
            "Current (A)": original_value,
            "Smoothed Current (A)": smoothed_value,
        }
    )
    return smoothed_df


st.title("Currencell")
st.markdown(
    "After the cell is stretched and hydrogen peroxide is released, how can the released bioelectricity be measured?"
)

# Upload file
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    default_file = uploaded_file
else:
    default_file = "./data.txt"


st.header("Smoother Setup")
col1, col2, col3 = st.columns(3)
with col1:
    smoother_name = st.selectbox(
        label="Smoother", options=["Exponential Smoother", "Spectral Smoother", "Convolution Smoother"], index=1
    )
smoother_meta = parameter_panel(smoother_name)  # col2 and col3


st.header("Intervals & Baseline Setup")
with st.form(key="baseline panel"):
    ccc0, ccc1 = st.columns(2)
    with ccc0:
        intervals_string = st.text_input(
            label="Enter the intervals",
            placeholder="Please input the intervals. (100, 150), (500, 600)",
            value="(1200,1400), (3000,3100)",
        )
    with ccc1:
        baseline_model_name = st.selectbox(label="Baseline Model", options=["Polynomial"], index=0)

    baseline_setup_submitted = st.form_submit_button(label="Submit")


data = pd.read_csv(default_file, header=0, sep="\t", usecols=["Elapsed Time (s)", "Current (A)"])[6000:-5000]
st.header("Visualization")
fig = px.line(data, x="Elapsed Time (s)", y=[_ for _ in data.columns if _ != "Elapsed Time (s)"])
fig.update_layout(showlegend=True)


def interval_process(raw_string):
    intervals_string = raw_string.replace("(", " ").replace(")", " ").replace(",", " ")
    intervals = [float(_) for _ in intervals_string.split()]
    paired_intervals = []
    for index in range(len(intervals) // 2):
        paired_intervals.append((intervals[index * 2], intervals[index * 2 + 1]))

    return paired_intervals


smoothed_data = smooth_process(data, smoother_meta)
fig = px.line(smoothed_data, x="Elapsed Time (s)", y=[_ for _ in smoothed_data.columns if _ != "Elapsed Time (s)"])


if baseline_setup_submitted:
    intervals = []
    intervals = interval_process(intervals_string)

    for pair in intervals:
        baseline_input_data = smoothed_data.drop(smoothed_data.index[int(pair[0]) : int(pair[1])])

    poly = np.polyfit(x=baseline_input_data["Elapsed Time (s)"], y=baseline_input_data["Smoothed Current (A)"], deg=3)
    baseline_value = np.polyval(poly, data["Elapsed Time (s)"])

    smoothed_data["Baseline"] = baseline_value

    fig = px.line(smoothed_data, x="Elapsed Time (s)", y=["Current (A)", "Smoothed Current (A)", "Baseline"])
    for pair in intervals:
        fig.add_vrect(x0=pair[0], x1=pair[1], line_width=0, fillcolor="red", opacity=0.2)

    clipped_dfs = []

    for pair in intervals:
        clipped_dfs.append(
            smoothed_data[
                (smoothed_data["Elapsed Time (s)"] > (pair[0] - 50))
                & (smoothed_data["Elapsed Time (s)"] < (pair[1] + 50))
            ]
        )

    col1, col2 = st.columns(2)
    with col1:
        clip_fig_1 = px.line(
            clipped_dfs[0],
            x="Elapsed Time (s)",
            y=[_ for _ in smoothed_data.columns if _ != "Elapsed Time (s)"],
        )
        clip_fig_1.update_layout(showlegend=False)
        st.plotly_chart(clip_fig_1, use_container_width=True)
    with col2:
        clip_fig_2 = px.line(
            clipped_dfs[1], x="Elapsed Time (s)", y=[_ for _ in smoothed_data.columns if _ != "Elapsed Time (s)"]
        )
        clip_fig_2.update_layout(showlegend=False)
        st.plotly_chart(clip_fig_2, use_container_width=True)


st.plotly_chart(fig, use_container_width=True)
