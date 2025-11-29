# %%
# Importing necessary packages
import pandas as pd
import numpy as np
import plotly.express as px

from dash import Dash, html, dcc, callback, Output, Input
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import os

print("File exists?", os.path.exists("CDC-2019-2021-2023-DATA.csv"))
# %%
# Importing and cleaning the data

df = pd.read_csv('CDC-2019-2021-2023-DATA.csv',low_memory=False)
df = df.query("IYEAR != 2024")


# %%
app = Dash()

app.layout = [
    html.H1(children='Predicting Mental Health with Behavioral Risk Factor Variables'),
    html.Div(children='Final Dataset after dropping irrelevant columns'),
    dag.AgGrid(
        rowData=df.to_dict('records'),
        columnDefs=[{"field": i} for i in df.columns[1:]]
    )
]

if __name__ == '__main__':
    app.run(debug=True)
# %%
