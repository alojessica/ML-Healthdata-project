#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import packages
from dash import Dash, html
import dash_ag_grid as dag
import pandas as pd


# ### Loading in cleaned data

# In[4]:


data = pd.read_csv('CDC-2019-2021-2023-DATA.csv')


# In[5]:


data = data.drop(['Unnamed: 0'], axis=1) # need to drop this 


# In[6]:


# dropping all the nan data
data = data.dropna()
print(data.shape)


# In[7]:


data = data.dropna(subset=['ADDEPEV3'])


# In[8]:


y = data['ADDEPEV3']
X = data[['BIRTHSEX', 'MENTHLTH', 'POORHLTH',
         'DECIDE', 'DIFFALON', 'IYEAR', 
        'ACEDEPRS', 'ACEDRINK', 'ACEDRUGS','ACEPRISN', 
        'ACEDIVRC', 'ACEPUNCH', 'ACEHURT1', 'ACESWEAR',
        'ACETOUCH','ACETTHEM', 'ACEHVSEX']]


# In[9]:


nums = ['POORHLTH', 'MENTHLTH']
cats = ['IYEAR', 'BIRTHSEX', 'ACEDEPRS', 
        'DECIDE', 'DIFFALON', 'ACEDRINK', 
        'ACEDRUGS','ACEPRISN', 'ACEDIVRC', 
        'ACEPUNCH', 'ACEHURT1', 'ACESWEAR',
        'ACETOUCH','ACETTHEM', 'ACEHVSEX']


# In[10]:


data.head()


# ### Adding KNN Model

# In[11]:


#import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import plotly.express as px


# In[12]:


preprocess = ColumnTransformer(transformers=[('encoder',OneHotEncoder(drop='first'),cats),
                                             ('numeric','passthrough',nums)])


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42,stratify=y)


# In[15]:


pipe=Pipeline([ ("preprocess", preprocess),
                ("scaler",StandardScaler()),
                ("knn",KNeighborsClassifier(weights="distance"))
])


# In[16]:


from pandas.core.groupby.indexing import GroupByIndexingMixin
# going to try find the k next; before you fit the model you need to define k
param_grid = {"knn__n_neighbors": range(1, 41, 2)}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="balanced_accuracy", n_jobs=-1)
grid.fit(X_train, y_train)


# In[19]:


results_df = pd.DataFrame(grid.cv_results_)

results_df["k"] = results_df["param_knn__n_neighbors"]
results_df["mean_score"] = results_df["mean_test_score"]

best_k = grid.best_params_["knn__n_neighbors"]
best_score = grid.best_score_

fig = px.line(
    results_df,
    x="k",
    y="mean_score",
    title=f"Cross-Validated Balanced Accuracy vs. K (best k = {best_k})",
    markers=True,
    labels={"k": "Number of Neighbors (k)", "mean_score": "Mean CV Balanced Accuracy"}
)


fig.add_scatter(
    x=[best_k],
    y=[best_score],
    mode="markers+text",
    text=[f"Best k = {best_k}"],
    textposition="top center",
    name="Best k"
)

fig.update_layout(hovermode="x unified")
fig.show()


# In[22]:


from dash import dcc


# In[ ]:


def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

app = Dash()

colors = {
    'background': '#7FDBFF',
    'text': '#111111'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='KNN Classifier Dashboard',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Model for KNN Classifier', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    dcc.Graph(
        id='knn_model',
        figure=fig
    ),

    html.H4(children='CDC Data'),
    generate_table(data),
])

if __name__ == '__main__':
    app.run(debug=True)

