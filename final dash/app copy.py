# Mental Health Dashboard

# Importing necessary packages
import pandas as pd
import numpy as np
import plotly.express as px

from dash import Dash, html, dcc, callback, Output, Input
import dash_ag_grid as dag
import dash_bootstrap_components as dbc

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score,
                             log_loss,
                             confusion_matrix,
                             roc_curve,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

# KNN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from pandas.core.groupby.indexing import GroupByIndexingMixin

# Hierarchical Clustering



# Importing and cleaning the data

df = pd.read_csv('CDC-2019-2021-2023-DATA.csv',low_memory=False)
df = df.query("IYEAR != 2024").dropna().drop('Unnamed: 0', axis=1)
df.ADDEPEV3 = df['ADDEPEV3'].replace({'Yes':1,'No':0}).astype(float)
df.head()

# X and y for the Logistc Regression and KNN
logit_knn_X = df[["BIRTHSEX","MENTHLTH","POORHLTH","DECIDE","DIFFALON","IYEAR",
                  "ACEDEPRS","ACEDRINK","ACEDRUGS","ACEPRISN","ACEDIVRC","ACEPUNCH",
                  "ACEHURT1","ACESWEAR","ACETOUCH","ACETTHEM","ACEHVSEX"]]
logit_knn_y = df["ADDEPEV3"] 

# Multiple Linear regression

import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.preprocessing import SplineTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

lr_data = pd.read_csv('CDC-2019-2023-DATA_nums.csv', low_memory=False)
lr_data = lr_data.drop(['Unnamed: 0'], axis=1)

ace_vars = ["ACEDEPRS", "ACESWEAR", "ACETTHEM"]

num_cols_base = ['AVEDRNK3', 'EXEROFT1', 'STRENGTH', 'PHYSHLTH', 'POORHLTH']
other_cat_cols = ['IYEAR', 'EMPLOY1']

def compute_ace_models(lr_data, ace_vars, num_cols_base, other_cat_cols):
    ace_metrics = {}
    ace_predictions = {}
    ace_categories_map = {}

    def adjusted_r2_score(y_true, y_pred, n_features):
        r2 = r2_score(y_true, y_pred)
        n = len(y_true)
        return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    for ace in ace_vars:
        cols_needed = ['MENTHLTH'] + num_cols_base + other_cat_cols + [ace]
        df_ace = lr_data.dropna(subset=cols_needed)

        X = df_ace[num_cols_base + other_cat_cols + [ace]]
        y = df_ace['MENTHLTH']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        preprocess = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first'), other_cat_cols + [ace]),
                ('num', 'passthrough', num_cols_base)
            ]
        )

        model = Pipeline([
            ("preprocess", preprocess),
            ("spline", SplineTransformer(degree=3, n_knots=8, include_bias=False)),
            ("linreg", LinearRegression())
        ])

        model.fit(X_train, y_train)

        yhat_train = model.predict(X_train)
        yhat_test = model.predict(X_test)

        n_features = model.named_steps['preprocess'].transform(X_train).shape[1]

        ace_metrics[ace] = {
            "Train RMSE": mean_squared_error(y_train, yhat_train)**0.5,
            "Test RMSE": mean_squared_error(y_test, yhat_test)**0.5,
            "Train Adj R2": adjusted_r2_score(y_train, yhat_train, n_features),
            "Test Adj R2": adjusted_r2_score(y_test, yhat_test, n_features)
        }

        categories = sorted(df_ace[ace].unique())
        ace_categories_map[ace] = categories

        base = {col: df_ace[col].mean() for col in num_cols_base}
        for col in other_cat_cols:
            base[col] = df_ace[col].mode()[0]

        plot_df = pd.DataFrame([base.copy() for _ in categories])
        plot_df[ace] = categories
        plot_df = plot_df[num_cols_base + other_cat_cols + [ace]]

        ace_predictions[ace] = model.predict(plot_df)

    return ace_metrics, ace_predictions, ace_categories_map

# Run ACE model computation ONCE at startup
ace_metrics, ace_predictions, ace_categories_map = compute_ace_models(
    lr_data, ace_vars, num_cols_base, other_cat_cols
)


bar_color = "#404075"

ace_labels_pretty = {
    "ACEDEPRS": "ACEDEPRS",
    "ACESWEAR": "ACESWEAR",
    "ACETTHEM": "ACETTHEM"
}

fig_ace = go.Figure()
buttons = []

for i, ace in enumerate(ace_vars):

    x_vals = ace_categories_map[ace]
    y_vals = ace_predictions[ace]
    adj_r2 = ace_metrics[ace]["Test Adj R2"]

    fig_ace.add_trace(
        go.Bar(
            x=x_vals,
            y=y_vals,
            name=ace_labels_pretty[ace],
            visible=(i == 0),
            marker_color=bar_color
        )
    )

    mask = [False] * len(ace_vars)
    mask[i] = True

    buttons.append(
        dict(
            label=ace_labels_pretty[ace],
            method="update",
            args=[
                {"visible": mask},
                {
                    "title": (
                        f"Cubic Spline Prediction of Bad Mental Health Days<br>"
                        f"<sup>{ace_labels_pretty[ace]} — "
                        f"Test Adjusted R² = {adj_r2:.3f}</sup>"
                    )
                }
            ]
        )
    )

first_ace = ace_vars[0]
first_adj_r2 = ace_metrics[first_ace]["Test Adj R2"]

fig_ace.update_layout(
    title=(
        f"Cubic Spline Prediction of Bad Mental Health Days<br>"
        f"<sup>{ace_labels_pretty[first_ace]} — "
        f"Test Adjusted R² = {first_adj_r2:.3f}</sup>"
    ),
    xaxis_title="ACE Category",
    yaxis_title="Predicted Bad Mental Health Days",
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            x=1.05,
            xanchor="left",
            y=1.0,
            yanchor="top",
            showactive=True
        )
    ]
)



def do_logit(X, y, test_size, threshold):
    """
    Fit logistic regression with given test_size and threshold.
    Returns metrics, confusion matrix, ROC info, and coefficient table.
    """
    # Target and predictors

    nums = ["POORHLTH", "MENTHLTH"]
    cats = ["IYEAR","BIRTHSEX","ACEDEPRS","DECIDE","DIFFALON","ACEDRINK",
            "ACEDRUGS","ACEPRISN","ACEDIVRC","ACEPUNCH","ACEHURT1",
            "ACESWEAR","ACETOUCH","ACETTHEM","ACEHVSEX"]

    preprocess = ColumnTransformer(
        transformers=[
            ("encoder", OneHotEncoder(drop="first"), cats),
            ("numeric", "passthrough", nums),
        ]
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0, stratify=y
    )

    # fit pipeline
    pipe.fit(X_train, y_train)

    # predicted probabilities + labels on test set
    p_test = pipe.predict_proba(X_test)[:, 1]
    y_hat_test = (p_test >= threshold).astype(int)

    # metrics
    acc = accuracy_score(y_test, y_hat_test)
    ll = log_loss(y_test, p_test)
    cm = confusion_matrix(y_test, y_hat_test)

    # ROC + AUC
    fpr, tpr, _ = roc_curve(y_test, p_test)
    auc = roc_auc_score(y_test, p_test)

    # coefficient importance
    logit = pipe.named_steps["model"]
    preprocess_step = pipe.named_steps["preprocess"]

    try:
        feature_names = preprocess_step.get_feature_names_out()
    except AttributeError:
        # Fallback
        feature_names = [f"feature_{i}" for i in range(logit.coef_.shape[1])]

    coefs = logit.coef_.ravel()

    coef_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefs,
                "abs_coeff": np.abs(coefs),
            }
        )
        .sort_values("abs_coeff", ascending=False)
        .head(15)
    )

    return acc, ll, cm, fpr, tpr, auc, coef_df

def do_logit(X, y, test_size, threshold):
    """
    Fit logistic regression with given test_size and threshold.
    Returns metrics, confusion matrix, ROC info, and coefficient table.
    """
    # Target and predictors

    nums = ["POORHLTH", "MENTHLTH"]
    cats = ["IYEAR","BIRTHSEX","ACEDEPRS","DECIDE","DIFFALON","ACEDRINK",
            "ACEDRUGS","ACEPRISN","ACEDIVRC","ACEPUNCH","ACEHURT1",
            "ACESWEAR","ACETOUCH","ACETTHEM","ACEHVSEX"]

    preprocess = ColumnTransformer(
        transformers=[
            ("encoder", OneHotEncoder(drop="first"), cats),
            ("numeric", "passthrough", nums),
        ]
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0, stratify=y
    )

    # fit pipeline
    pipe.fit(X_train, y_train)

    # predicted probabilities + labels on test set
    p_test = pipe.predict_proba(X_test)[:, 1]
    y_hat_test = (p_test >= threshold).astype(int)

    # metrics
    acc = accuracy_score(y_test, y_hat_test)
    ll = log_loss(y_test, p_test)
    cm = confusion_matrix(y_test, y_hat_test)

    # ROC + AUC
    fpr, tpr, _ = roc_curve(y_test, p_test)
    auc = roc_auc_score(y_test, p_test)

    # coefficient importance
    logit = pipe.named_steps["model"]
    preprocess_step = pipe.named_steps["preprocess"]

    try:
        feature_names = preprocess_step.get_feature_names_out()
    except AttributeError:
        # Fallback
        feature_names = [f"feature_{i}" for i in range(logit.coef_.shape[1])]

    coefs = logit.coef_.ravel()

    coef_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefs,
                "abs_coeff": np.abs(coefs),
            }
        )
        .sort_values("abs_coeff", ascending=False)
        .head(15)
    )

    return acc, ll, cm, fpr, tpr, auc, coef_df

def do_knn(X, y):
    nums = ['POORHLTH', 'MENTHLTH']
    cats = ['IYEAR', 'BIRTHSEX', 'ACEDEPRS', 'DECIDE', 'DIFFALON', 'ACEDRINK', 'ACEDRUGS','ACEPRISN', 'ACEDIVRC', 'ACEPUNCH',
            'ACEHURT1', 'ACESWEAR','ACETOUCH','ACETTHEM', 'ACEHVSEX']
    
    preprocess = ColumnTransformer(transformers=[('encoder',OneHotEncoder(drop='first'),cats),
                                                 ('numeric','passthrough',nums)])
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42,stratify=y)

    pipe = Pipeline([("preprocess", preprocess),
                     ("scaler",StandardScaler()),
                     ("knn",KNeighborsClassifier(weights="distance"))
                    ])
    
    param_grid = {"knn__n_neighbors": range(1, 41, 2)}
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="balanced_accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

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

    fig.update_layout(hovermode="x unified", showlegend=False)
    return fig

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__,external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

#app = Dash()

app.layout = html.Div([
    html.H1(children='Behavioral Risk Mental Health Dashboard'),
                        dcc.Tabs(id='tabs', value='tab1', 
                            children=[
                                dcc.Tab(label='README: Project Overview', 
                                        value='tab1'),
                                dcc.Tab(label='Data Table', 
                                        value='tab2'),
                                dcc.Tab(label='Models', 
                                        value='tab3')
                                    ]
                                ),
                        html.Div(id='tabs-content')
                        ])


@callback(Output('tabs-content', 'children'),
          Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab1':
        return html.Div([
                html.H2('Behavioral Risk Mental Health Dashboard: Predicting Mental Health with Behavioral Risk Factor Variables'),
                html.P('''This app uses behavioral risk variables from 2019, 2021, and 2023 to predict mental health outcomes,
                          focusing primarily on variables relating to adverse childhood experiences, as well as a few other variables.'''),
                
                html.H3('About the Dataset'),
                html.P('''This dataset comes from the CDC\'s Behavioral Risk Factor Surveillance System,
                          a system of comprehensive telephone surveys conducted every year regarding health-related risk behaviors,
                          chronic health conditions, and use of preventative health services for adults in the United States. Each row
                          represents a single respondent with variables including birth sex, year survey was taken,  and 
                       ''' ),

                html.H3('Target Variables'),
                html.U('Logistic Regression and K Nearest Neighbor'),
                html.P([html.B('ADDEPEV3: '),'''Answer to survey question: (Ever told) (you had) a depressive disorder 
                                                (including depression, major depression, dysthymia, or minor depression)?''']),
                html.U('Linear Regression'),                                
                html.P([html.B('MENTHLTH: '),'''Answer to survey question: Now thinking about your mental health, which includes stress, 
                                                depression, and problems with emotions, for how many days during the past 30 days was your 
                                                mental health not good?''']),                                
                html.H3('Predictor Variables'),
                html.Ul([
                    html.Li([html.B('BIRTHSEX: '),'Assigned sex of respondent at birth']),
                    html.Li([html.B('IYEAR: '), 'Year the respondent took the survey']),
                    html.Li([html.B('POORHLTH: '),'''Answer to survey question: During the past 30 days, for about how many days did poor physical or mental health 
                                                   keep you from doing your usual activities, such as self-care, work, or recreation?''']),
                    html.Li([html.B('MENTHLTH: '), '''Answer to survey question:Now thinking about your mental health, 
                                                     which includes stress, depression, and problems with emotions, 
                                                     for how many days during the past 30 days was your mental health not good?''']),
                    html.Li([html.B('DECIDE: '), '''Answer to survey question: Because of a physical, mental, or emotional condition, 
                                                   do you have serious difficulty concentrating, remembering, or making decisions?''']),
                    html.Li([html.B('DIFFALON: '), '''Answer to survey question: Because of a physical, mental, or emotional condition, 
                                                     do you have difficulty doing errands alone such as visiting a doctor's office or shopping?''']),
                    html.Li([html.B('ACEDEPRS: '), 'Answer to survey question: (As a child) Did you live with anyone who was depressed, mentally ill, or suicidal?']),
                    html.Li([html.B('ACEDRINK: '), 'Answer to survey question: (As a child) Did you live with anyone who was a problem drinker or alcoholic?']),
                    html.Li([html.B('ACEDRUGS: '), 'Answer to survey question: (As a child) Did you live with anyone who used illegal street drugs or who abused prescription medications?']),
                    html.Li([html.B('ACEPRISN: '), 'Answer to survey question: (As a child) Did you live with anyone who served time or was sentenced to serve time in a prison, jail, or other correctional facility?']),
                    html.Li([html.B('ACEDIVRC: '), 'Answer to survey question: (As a child) Were your parents separated or divorced?']),
                    html.Li([html.B('ACEPUNCH: '), 'Answer to survey question: (As a child) How often did your parents or adults in your home ever slap, hit, kick, punch or beat each other up?']),
                    html.Li([html.B('ACEHURT1: '), 'Answer to survey question: (As a child) Not including spanking, (before age 18), how often did a parent or adult in your home ever hit, beat, kick, or physically hurt you in any way?']),
                    html.Li([html.B('ACESWEAR: '), 'Answer to survey question: (As a child) How often did a parent or adult in your home ever swear at you, insult you, or put you down']),
                    html.Li([html.B('ACETOUCH: '), 'Answer to survey question: (As a child) How often did anyone at least 5 years older than you or an adult, ever touch you sexually?']),
                    html.Li([html.B('ACETTHEM: '), 'Answer to survey question: (As a child) How often did anyone at least 5 years older than you or an adult, try to make you touch them sexually?']),
                    html.Li([html.B('ACEHVSEX: '), 'Answer to survey question: (As a child) How often did anyone at least 5 years older than you or an adult, force you to have sex?']),
                        ]),

                html.H3('Key Features of Dashboard'),
                html.Ul([
                    html.Li('View rows of the final cleaned dataset in the Data Table Tab'),
                    html.Li('Select from Logistic Regression, K Nearest Neighbor, Hierarchical Agglomerative Clustering models, or Principal Component Analysis with Lasso Regularization models'),
                    html.Li('Change hyperparameters, such as number of neighbors and train test split, to your liking to view different versions of the model')
                        ]),

                html.H3('Instructions for Use'),
                html.P('hello'),

                html.H3('Authors'),
                html.P('''Randa Ampah, Isabel Delgado, Aysha Hussen, Aniyah McWilliams, 
                          and Jessica Oseghale for the DS 6021 Final Project in the Fall 
                          25 semester of the UVA MSDS program''')

        ])
    
    if tab == 'tab2':
        return html.Div([
                dag.AgGrid(
                    rowData=df.to_dict('records'),
                    columnDefs=[{"field": i} for i in df.columns]
                          )
        ])
    
    if tab == 'tab3':
        return dbc.Container(fluid=True, children=[

            dbc.Row([

                # ---------- SIDEBAR ----------
                dbc.Col(
                    dbc.Nav(
                        [
                            dbc.NavLink("K Nearest Neighbor", href="/models/knn", active="exact"),
                            dbc.NavLink("Logistic Regression", href="/models/logit", active="exact"),
                            dbc.NavLink("Multiple Linear Regression", href="/models/linear", active="exact"),
                            dbc.NavLink("Hierarchical Clustering", href='/models/hierarchichal', active='exact'),
                            dbc.NavLink("Lasso Regularization", href='/models/lasso', active='exact')
                        ],
                        vertical=True,
                        pills=True,
                    ),
                    width=2,
                    style={"backgroundColor": "#f8f9fa", "padding": "20px", "height": "100vh"},
                ),

                # ---------- MAIN CONTENT ----------
                dbc.Col(
                    html.Div(id="sub-tabs-content"),
                    width=10,
                    style={"padding": "40px"}
                ),

            ]),

            dcc.Location(id="url")
        ])


@callback(
    Output("sub-tabs-content", "children"),
    Input("url", "pathname"),
)
def update_sidebar_content(pathname):

    if pathname == "/models/knn":
        fig = do_knn(logit_knn_X, logit_knn_y)

        return html.Div([
            html.H2("KNN Classifier Dashboard"),
            dcc.Graph(figure=fig),
        ])

    elif pathname == "/models/logit":
        return html.Div([
            html.H2("Logistic Regression Model"),

            html.Label("Test set size (%)"),
            dcc.Slider(
                id="test-size-slider",
                min=10,
                max=50,
                step=5,
                value=30,
                marks={i: f"{i}%" for i in range(10, 55, 5)},
            ),

            html.Br(),

            html.Label("Classification threshold"),
            dcc.Slider(
                id="threshold-slider",
                min=0.1,
                max=0.9,
                step=0.05,
                value=0.7,
            ),

            html.Br(),

            html.Div(id="logit-metrics"),
            dcc.Graph(id="logit-confusion"),
            dcc.Graph(id="logit-roc"),
            dcc.Graph(id="logit-coefs"),
        ])

    elif pathname == "/models/linear":
        return html.Div([
            html.H2("Multiple Linear Regression"),

            dcc.Graph(figure=fig_ace),

            html.Br(),
            html.H3("Model Performance Metrics"),

            html.Table([
                html.Thead(html.Tr([
                    html.Th("ACE Variable"),
                    html.Th("Train RMSE"),
                    html.Th("Test RMSE"),
                    html.Th("Train Adj R²"),
                    html.Th("Test Adj R²")
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(ace_labels_pretty[ace]),
                        html.Td(f"{ace_metrics[ace]['Train RMSE']:.3f}"),
                        html.Td(f"{ace_metrics[ace]['Test RMSE']:.3f}"),
                        html.Td(f"{ace_metrics[ace]['Train Adj R2']:.3f}"),
                        html.Td(f"{ace_metrics[ace]['Test Adj R2']:.3f}")
                    ])
                    for ace in ace_vars
                ])
            ], style={"width": "70%", "margin": "auto"}),

            html.Br()
        ])


    elif pathname == "/models/hierarchichal":
        return html.Div([
            html.H2("Hierarchichal Agglomerative Clustering"),
            html.P("Coming soon...")
        ])
    elif pathname == '/models/lasso':
       return html.Div([
            html.H2("Lasso Regularization"),
            html.P("Coming soon...")
        ])
    
    return html.Div("Select a model from the sidebar!")

@callback(
    Output("logit-metrics", "children"),
    Output("logit-confusion", "figure"),
    Output("logit-roc", "figure"),
    Output("logit-coefs", "figure"),
    Input("test-size-slider", "value"),
    Input("threshold-slider", "value"),
)

def update_logit_tab(test_size_pct, threshold):
    # convert slider % to proportion
    test_size = test_size_pct / 100.0

    acc, ll, cm, fpr, tpr, auc, coef_df = do_logit(logit_knn_X, logit_knn_y, test_size, threshold)

    # ----- metrics text ----- #
    metrics = html.Ul(
        [
            html.Li(f"Test size: {test_size_pct}%"),
            html.Li(f"Threshold: {threshold:.2f}"),
            html.Li(f"Accuracy: {acc:.3f}"),
            html.Li(f"Log loss: {ll:.3f}"),
            html.Li(f"AUC: {auc:.3f}"),
        ]
    )

    # ----- confusion matrix heatmap ----- #
    cm_fig = px.imshow(
        cm,
        text_auto=True,
        x=["Predicted: No depression", "Predicted: Yes depression"],
        y=["Actual: No depression", "Actual: Yes depression"],
        labels=dict(x="Predicted label", y="Actual label", color="Count"),
    )
    cm_fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))

    # ----- ROC curve ----- #
    roc_fig = px.area(
        x=fpr,
        y=tpr,
        labels=dict(x="False positive rate", y="True positive rate"),
        title=f"ROC Curve (AUC = {auc:.3f})",
    )
    roc_fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(dash="dash"),
    )
    roc_fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))

    # ----- coefficient importance bar chart ----- #
    coef_df_sorted = coef_df.sort_values("abs_coeff", ascending=True)
    coef_fig = px.bar(
        coef_df_sorted,
        x="coefficient",
        y="feature",
        orientation="h",
        title="Top Logistic Regression Coefficients (by |beta|)",
    )
    coef_fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))

    return metrics, cm_fig, roc_fig, coef_fig


if __name__ == '__main__':
    app.run(debug=True, port=8051)

