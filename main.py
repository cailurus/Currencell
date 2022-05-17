#!/usr/bin/env python
# encoding: utf-8

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Currencell", page_icon="random", layout="wide", initial_sidebar_state="auto", menu_items=None
)

st.title("Currencell")

data = pd.read_csv("./data.txt", header=0, sep="\t", usecols=["Elapsed Time (s)", "Current (A)"])

fig = px.line(data[100:500], x="Elapsed Time (s)", y="Current (A)")
st.plotly_chart(fig, use_container_width=True)

st.table(data.head(20).style.format({"Current (A)": "{:.8f}", "Elapsed Time (s)": "{:.2f}"}))
