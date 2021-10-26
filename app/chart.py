import altair as alt
import pandas as pd


def df_to_chart(df: pd.DataFrame, x_label: str='x', y_label:str ='y', font_size: int=18) -> alt.Chart:
    df = df.reset_index().melt('index').rename(columns={'index': 'x', 'value': 'y', 'variable': 'method'})
    chart = alt.Chart(df).mark_line(size=3).encode(
        x=alt.X('x', title=x_label),
        y=alt.Y('y', title=y_label),
        color='method',
        strokeDash='method',
    ).configure_axis(
        labelFontSize=font_size, titleFontSize=font_size
    ).configure_header(
        titleFontSize=font_size, labelFontSize=font_size
    ).configure_legend(
        titleFontSize=font_size, labelFontSize=font_size
    ).properties(
        width=600, height=400
    ).interactive()
    return chart