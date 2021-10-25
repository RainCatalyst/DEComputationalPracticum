import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

x = np.arange(100)
df = pd.DataFrame({
  'x1': x,
  'x2': np.sin(x / 5)
}, index=x)

df = df.reset_index().melt('index')
print(df)
print(df.rename(columns={'index': 'x', 'value': 'y', 'variable': 'method'}))

#chart = alt.layer(*[alt.Chart(df).mark_line(point=True).encode(x='index', y=col, color=f'{col}:N') for col in df.columns if col != 'index'])
chart = alt.Chart(df).mark_line(point=True).encode(x='index', y='value', color='variable').configure_point(size=20)
st.altair_chart(chart, use_container_width=True)
