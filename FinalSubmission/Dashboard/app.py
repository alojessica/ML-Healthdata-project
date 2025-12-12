# %%
# Mental Health Dashboard

# ---------- Imports ----------
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import zipfile
import io

from dash import Dash, html, dcc, callback, Output, Input, dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc

# Logistic Regression / KNN / Linear Reg
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    balanced_accuracy_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier   # <--- added back

from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

# HCA imports
from gower import gower_matrix
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from plotly.subplots import make_subplots

# Unzip data file
zip_file_name = 'CDC-2019-2021-2023-DATA.csv.zip'
csv_file_name = 'CDC-2019-2021-2023-DATA.csv'

with zipfile.ZipFile(zip_file_name, mode='r') as archive:
    with archive.open(csv_file_name, mode='r') as csv_file:
        df = pd.read_csv(csv_file,low_memory=False)

df = df.query("IYEAR != 2024").dropna().drop("Unnamed: 0", axis=1)

# Binary target for logit / KNN
df["ADDEPEV3"] = df["ADDEPEV3"].replace({"Yes": 1, "No": 0}).astype(float)

# X and y for Logistic Regression and KNN
logit_knn_X = df[
    [
        "BIRTHSEX",
        "MENTHLTH",
        "POORHLTH",
        "DECIDE",
        "DIFFALON",
        "IYEAR",
        "ACEDEPRS",
        "ACEDRINK",
        "ACEDRUGS",
        "ACEPRISN",
        "ACEDIVRC",
        "ACEPUNCH",
        "ACEHURT1",
        "ACESWEAR",
        "ACETOUCH",
        "ACETTHEM",
        "ACEHVSEX",
    ]
]
logit_knn_y = df["ADDEPEV3"]

# ============================================================
#   MULTIPLE LINEAR REGRESSION (ACE Spline model)
# ============================================================

        
lr_zip = 'CDC-2019-2023-DATA_nums.csv.zip'
lr_csv = 'CDC-2019-2023-DATA_nums.csv'

with zipfile.ZipFile(lr_zip, mode='r') as archive1:
    with archive1.open(lr_csv, mode='r') as csv_file1:
        lr_data = pd.read_csv(csv_file1,low_memory=False)

lr_data = lr_data.drop(["Unnamed: 0"], axis=1)

ace_vars = ["ACEDEPRS", "ACESWEAR", "ACETTHEM"]
num_cols_base = ["AVEDRNK3", "EXEROFT1", "STRENGTH", "PHYSHLTH", "POORHLTH"]
other_cat_cols = ["IYEAR", "EMPLOY1"]


def compute_ace_models(lr_data, ace_vars, num_cols_base, other_cat_cols):
    ace_metrics = {}
    ace_predictions = {}
    ace_categories_map = {}

    def adjusted_r2_score(y_true, y_pred, n_features):
        r2 = r2_score(y_true, y_pred)
        n = len(y_true)
        return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    for ace in ace_vars:
        cols_needed = ["MENTHLTH"] + num_cols_base + other_cat_cols + [ace]
        df_ace = lr_data.dropna(subset=cols_needed)

        X = df_ace[num_cols_base + other_cat_cols + [ace]]
        y = df_ace["MENTHLTH"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        preprocess = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(drop="first"), other_cat_cols + [ace]),
                ("num", "passthrough", num_cols_base),
            ]
        )

        model = Pipeline(
            [
                ("preprocess", preprocess),
                ("spline", SplineTransformer(degree=3, n_knots=8, include_bias=False)),
                ("linreg", LinearRegression()),
            ]
        )

        model.fit(X_train, y_train)

        yhat_train = model.predict(X_train)
        yhat_test = model.predict(X_test)

        n_features = model.named_steps["preprocess"].transform(X_train).shape[1]

        ace_metrics[ace] = {
            "Train RMSE": mean_squared_error(y_train, yhat_train) ** 0.5,
            "Test RMSE": mean_squared_error(y_test, yhat_test) ** 0.5,
            "Train Adj R2": adjusted_r2_score(y_train, yhat_train, n_features),
            "Test Adj R2": adjusted_r2_score(y_test, yhat_test, n_features),
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


ace_metrics, ace_predictions, ace_categories_map = compute_ace_models(
    lr_data, ace_vars, num_cols_base, other_cat_cols
)

bar_color = "#a4a4e3"
ace_labels_pretty = {
    "ACEDEPRS": "Depressed household member",
    "ACESWEAR": "Verbal abuse as child",
    "ACETTHEM": "Attempted sexual assault",
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
            marker_color=bar_color,
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
                        "Cubic Spline Prediction of Bad Mental Health Days<br>"
                        f"<sup>{ace_labels_pretty[ace]} — Test Adjusted R² = {adj_r2:.3f}</sup>"
                    )
                },
            ],
        )
    )

first_ace = ace_vars[0]
first_adj_r2 = ace_metrics[first_ace]["Test Adj R2"]

fig_ace.update_layout(
    title=(
        "Cubic Spline Prediction of Bad Mental Health Days<br>"
        f"<sup>{ace_labels_pretty[first_ace]} — Test Adjusted R² = {first_adj_r2:.3f}</sup>"
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
            showactive=True,
        )
    ],
)

# ============================================================
#   LOGISTIC REGRESSION function
# ============================================================


def do_logit(X, y, test_size, threshold):
    """
    Fit logistic regression with given test_size and threshold.
    Returns metrics, confusion matrix, ROC info, and coefficient table.
    """
    nums = ["POORHLTH", "MENTHLTH"]
    cats = [
        "IYEAR",
        "BIRTHSEX",
        "ACEDEPRS",
        "DECIDE",
        "DIFFALON",
        "ACEDRINK",
        "ACEDRUGS",
        "ACEPRISN",
        "ACEDIVRC",
        "ACEPUNCH",
        "ACEHURT1",
        "ACESWEAR",
        "ACETOUCH",
        "ACETTHEM",
        "ACEHVSEX",
    ]

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0, stratify=y
    )

    pipe.fit(X_train, y_train)

    p_test = pipe.predict_proba(X_test)[:, 1]
    y_hat_test = (p_test >= threshold).astype(int)

    acc = accuracy_score(y_test, y_hat_test)
    ll = log_loss(y_test, p_test)
    cm = confusion_matrix(y_test, y_hat_test)

    fpr, tpr, _ = roc_curve(y_test, p_test)
    auc = roc_auc_score(y_test, p_test)

    logit = pipe.named_steps["model"]
    preprocess_step = pipe.named_steps["preprocess"]

    try:
        feature_names = preprocess_step.get_feature_names_out()
    except AttributeError:
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


# ============================================================
#   KNN helper
# ============================================================


def do_knn(X, y):
    nums = ["POORHLTH", "MENTHLTH"]
    cats = [
        "IYEAR",
        "BIRTHSEX",
        "ACEDEPRS",
        "DECIDE",
        "DIFFALON",
        "ACEDRINK",
        "ACEDRUGS",
        "ACEPRISN",
        "ACEDIVRC",
        "ACEPUNCH",
        "ACEHURT1",
        "ACESWEAR",
        "ACETOUCH",
        "ACETTHEM",
        "ACEHVSEX",
    ]

    preprocess = ColumnTransformer(
        transformers=[
            ("encoder", OneHotEncoder(drop="first"), cats),
            ("numeric", "passthrough", nums),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(
        [
            ("preprocess", preprocess),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(weights="distance")),
        ]
    )

    param_grid = {"knn__n_neighbors": range(1, 41, 2)}
    grid = GridSearchCV(
        pipe, param_grid, cv=5, scoring="balanced_accuracy", n_jobs=-1
    )
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
        labels={
            "k": "Number of Neighbors (k)",
            "mean_score": "Mean CV Balanced Accuracy",
        },
    )

    fig.add_scatter(
        x=[best_k],
        y=[best_score],
        mode="markers+text",
        text=[f"Best k = {best_k}"],
        textposition="top center",
        name="Best k",
    )

    fig.update_layout(hovermode="x unified", showlegend=False)

    pipe2 = Pipeline([
            ("preprocess", preprocess),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=best_k,
            weights="distance"))
        ])
    
    pipe2.fit(X_train, y_train)
    y_pred = pipe2.predict(X_test)
    prob_test = pipe2.predict_proba(X_test)[:,1]

    knn_acc = accuracy_score(y_test, y_pred)
    knn_bal_acc = balanced_accuracy_score(y_test, y_pred)

    knn_fpr, knn_tpr, _ = roc_curve(y_test, prob_test)
    knn_auc = roc_auc_score(y_test,prob_test)
    knn_roc_fig = px.area(
        x=knn_fpr,
        y=knn_tpr,
        labels=dict(x="False positive rate", y="True positive rate"),
        title=f"ROC Curve (AUC = {knn_auc:.3f})",
    )
    knn_roc_fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(dash="dash"),
    )
    knn_roc_fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))

    knn_scatter = px.scatter(
                    X_test.iloc[:10,:], x='POORHLTH', y='MENTHLTH',
                    color=prob_test[:10], color_continuous_scale='Magenta',
                    symbol=y_test[:10], symbol_map={'0': 'square-dot', '1': 'circle-dot'},
                    labels={'symbol': 'label', 'color': 'probability of <br>first class <br>(Being Depressed)'},
                    title="KNN Depression Classification Displayed Across Days Depressed and Days Unmotivated"
                )

    knn_scatter.update_traces(marker_size=12, marker_line_width=1.5)
    knn_scatter.update_layout(legend_orientation='h')

    cm = confusion_matrix(y_test, y_pred)


    knn_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale='RdPu',
        x=["Predicted: No depression", "Predicted: Yes depression"],
        y=["Actual: No depression", "Actual: Yes depression"],
        labels=dict(x="Predicted label", y="Actual label", color="Count", title="Confusion Matrix of Depressive Disorder Predictions"),
    )
    knn_cm.update_layout(margin=dict(l=40, r=40, t=40, b=40))

    knn_met = html.Ul(
        [
            html.Li(f"Accuracy: {knn_acc:.3f}"),
            html.Li(f"Balanced Accuracy: {knn_bal_acc:.3f}"),
            html.Li(f"AUC: {knn_auc:.3f}"),
        ]
    )

    return fig, knn_roc_fig, knn_met, knn_scatter, knn_cm


# ============================================================
#   HIERARCHICAL CLUSTERING: precompute clusters & figures
# ============================================================

# ============================================================
#   HIERARCHICAL CLUSTERING: precompute clusters & figures
# ============================================================

# 1. Define Columns
ace_YN = ["ACEDEPRS", "ACEDRINK", "ACEDRUGS", "ACEPRISN", "ACEDIVRC"]
ace_NOM = ["ACEPUNCH", "ACEHURT1", "ACESWEAR", "ACETOUCH", "ACETTHEM", "ACEHVSEX"]
cat_cols = ace_YN + ace_NOM
cat_mh_cols = ["ADDEPEV3", "DECIDE", "DIFFALON"]
num_mh_cols = ["MENTHLTH", "POORHLTH"]

df_hac = df.dropna(subset=cat_cols + cat_mh_cols + num_mh_cols).copy()

# 2. Sample Data (5000 rows for performance)
df_sample = (
    df_hac.sample(n=5000, random_state=42)
    if len(df_hac) > 5000
    else df_hac.copy()
)

# 3. Calculate Distances & Linkage
distance_matrix = gower_matrix(df_sample[cat_cols])
Z = linkage(squareform(distance_matrix, checks=False), method="average")

# 4. Silhouette Score Graph
sil_results = []
for k in range(2, 5):
    model = AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="average"
    )
    labels_k = model.fit_predict(distance_matrix)
    sil = silhouette_score(distance_matrix, labels_k, metric="precomputed")
    sil_results.append({"k": k, "silhouette": sil})

sil_df = pd.DataFrame(sil_results)
best_k = int(sil_df.loc[sil_df["silhouette"].idxmax(), "k"])

hac_sil_fig = px.line(
    sil_df,
    x="k",
    y="silhouette",
    markers=True,
    title="Silhouette Scores for Different K (Gower Distance)",
)

# 5. Final Clustering (K=2)
final_model = AgglomerativeClustering(
    n_clusters=best_k, metric="precomputed", linkage="average"
)
final_labels = final_model.fit_predict(distance_matrix)

df_sample = df_sample.reset_index(drop=True)
df_sample["cluster"] = final_labels

# 6. DYNAMIC RISK DETECTION (The Fix)
# Map ADDEPEV3 back to Yes/No for plots
df_sample["ADDEPEV3"] = df_sample["ADDEPEV3"].replace(
    {0.0: "No", 1.0: "Yes", 0: "No", 1: "Yes"}
)

# Calculate means to find the "High Risk" group
cluster_means = df_sample.groupby("cluster")["MENTHLTH"].mean()

if cluster_means[0] > cluster_means[1]:
    # Cluster 0 is worse
    c0_name = "High Risk / High ACE"
    c1_name = "Low Risk / Low ACE"
    c0_color = "#6b8cce"  # Blue
    c1_color = "#ff69b4"  # Pink
    high_risk_id = 0
    low_risk_id = 1
else:
    # Cluster 1 is worse
    c0_name = "Low Risk / Low ACE"
    c1_name = "High Risk / High ACE"
    c0_color = "#ff69b4"  # Pink
    c1_color = "#6b8cce"  # Blue
    high_risk_id = 1
    low_risk_id = 0

# 7. Generate Heatmap (Dynamic)
heatmap_matrix = []
heatmap_labels = []

for col in cat_cols:
    if col in df_sample.columns:
        crosstab = (
            pd.crosstab(df_sample["cluster"], df_sample[col], normalize="index") * 100
        )
        for category in crosstab.columns:
            heatmap_matrix.append(
                [crosstab.loc[high_risk_id, category], crosstab.loc[low_risk_id, category]]
            )
            heatmap_labels.append(f"{col}: {category}")

hac_heatmap_fig = go.Figure(
    data=go.Heatmap(
        z=heatmap_matrix,
        y=heatmap_labels,
        x=["High Risk / High ACE", "Low Risk / Low ACE"],
        colorscale=[
            [0.0, "#d1eeff"],
            [0.33, "#6b8cce"],
            [0.66, "#9b59b6"],
            [1.0, "#ff69b4"],
        ],
        text=np.round(heatmap_matrix, 1),
        texttemplate="%{text}%",
        textfont={"size": 11},
        colorbar=dict(title="Percentage (%)"),
        xgap=2, ygap=2,
    )
)
hac_heatmap_fig.update_layout(
    title="All ACE Variables: Proportion Heatmap by Cluster",
    height=max(600, len(heatmap_labels) * 35),
    yaxis=dict(dtick=1, automargin=True),
)

# 8. Generate Dropdown Bar Chart (Dynamic)
hac_cat_fig = go.Figure()
buttons = []

for i, col in enumerate(cat_cols):
    crosstab = pd.crosstab(df_sample["cluster"], df_sample[col], normalize="index") * 100
    categories = crosstab.columns.tolist()
    is_visible = (i == 0)

    hac_cat_fig.add_trace(go.Bar(
        x=categories, y=crosstab.loc[0], name=c0_name, marker_color=c0_color, visible=is_visible
    ))
    hac_cat_fig.add_trace(go.Bar(
        x=categories, y=crosstab.loc[1], name=c1_name, marker_color=c1_color, visible=is_visible
    ))

    visible_settings = [False] * (len(cat_cols) * 2)
    visible_settings[2 * i] = True
    visible_settings[2 * i + 1] = True

    buttons.append(dict(
        label=col, method="update",
        args=[{"visible": visible_settings}, {"title": f"ACE Category by Cluster: {col}"}]
    ))

hac_cat_fig.update_layout(
    title=f"ACE Category by Cluster: {cat_cols[0]}",
    yaxis_title="Percentage (%)",
    barmode="group",
    updatemenus=[dict(buttons=buttons, direction="down", x=1.05, y=0.85, showactive=True)]
)

# 9. Generate Mental Health Bar Subplots (Dynamic)
fig_mh_bar = make_subplots(
    rows=3, cols=1,
    subplot_titles=["Depressive Disorder", "Diff. Concentrating", "Diff. Errands"]
)

for i, col in enumerate(cat_mh_cols):
    ct = pd.crosstab(df_sample["cluster"], df_sample[col], normalize="index") * 100
    labels = ct.columns.astype(str)
    
    fig_mh_bar.add_trace(go.Bar(
        x=labels, y=ct.loc[0], name=c0_name, marker_color=c0_color, showlegend=(i==0)
    ), row=i+1, col=1)
    
    fig_mh_bar.add_trace(go.Bar(
        x=labels, y=ct.loc[1], name=c1_name, marker_color=c1_color, showlegend=(i==0)
    ), row=i+1, col=1)

fig_mh_bar.update_layout(title="Mental Health Outcomes by Cluster", barmode="group", height=800)

# 10. Generate Violin Plots (Dynamic)
fig_violin = go.Figure()
metrics = ["MENTHLTH", "POORHLTH"]
x_labels = ["Mental health (days)", "Physical health (days)"]

for i, col in enumerate(metrics):
    fig_violin.add_trace(go.Violin(
        y=df_sample[df_sample["cluster"] == 0][col],
        x=[x_labels[i]] * len(df_sample[df_sample["cluster"] == 0]),
        legendgroup=c0_name, scalegroup="Cluster0", name=c0_name,
        side="negative", line_color=c0_color, opacity=0.6, meanline_visible=True,
        showlegend=(i==0), spanmode="hard"
    ))
    fig_violin.add_trace(go.Violin(
        y=df_sample[df_sample["cluster"] == 1][col],
        x=[x_labels[i]] * len(df_sample[df_sample["cluster"] == 1]),
        legendgroup=c1_name, scalegroup="Cluster1", name=c1_name,
        side="positive", line_color=c1_color, opacity=0.6, meanline_visible=True,
        showlegend=(i==0), spanmode="hard"
    ))

fig_violin.update_traces(width=0.5, points=False)
fig_violin.update_layout(title="Distribution of Unhealthy Days", violinmode="overlay", height=600)

# ============================================================
#   DASH APP LAYOUT
# ============================================================

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(
    __name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True
)
server = app.server

app.layout = html.Div(
    [
        html.H1(children="Behavioral Risk Mental Health Dashboard"),
        dcc.Tabs(
            id="tabs",
            value="tab1",
            children=[
                dcc.Tab(label="README: Project Overview", value="tab1"),
                dcc.Tab(label="Data Table", value="tab2"),
                dcc.Tab(label="Models", value="tab3"),
            ],
        ),
        html.Div(id="tabs-content"),
    ]
)


@callback(Output("tabs-content", "children"), Input("tabs", "value"))
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
                          represents a single respondent with variables including birth sex, year survey was taken, and over 100 behavioral risk
                          related variables. More information can be gathered at this link : https://www.cdc.gov/brfss/about/brfss_faq.htm
                       ''' ),

                html.H3('Target Variables'),
                html.U('Logistic Regression and K Nearest Neighbor'),
                html.P([html.B('ADDEPEV3: '),'''Answer to survey question: (Ever told) (you had) a depressive disorder 
                                                (including depression, major depression, dysthymia, or minor depression)?''']),
                html.U('Linear Regression and Lasso Regularization'),                                
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
                    html.Li([html.B('EMPLOY1: '), '''Answer to survey question: Are you currently... (Employed for wages, Self-employed, Out of work for 1 year or more, 
                                                    Out of work for less than 1 year, A homemaker, A student, Retired, or Unable to work)?''']),
                    html.Li([html.B('AVEDRNK3: '), 'Answer to survey question: During the past 30 days, on the days when you drank, about how many drinks did you drink on the average?']),
                    html.Li([html.B('EXEROFT1: '), 'Answer to survey question: How many times per week or per month did you take part in this (physical) activity during the past month?']),
                    html.Li([html.B('STRENGTH: '), 'Answer to survey question: During the past month, how many times per week or per month did you do physical activities or exercises to STRENGTHEN your muscles?']),
                    html.Li([html.B('PHYSHLTH: '), '''Answer to survey question: Now thinking about your physical health, which includes physical illness and injury, 
                                                for how many days during the past 30 days was your physical health not good?'''])
                        ]),

                html.H3('Key Features of Dashboard'),
                html.Ul([
                    html.Li('View rows of the final cleaned dataset in the Data Table tab'),
                    html.Li('See results and interact with variables of the different models in the Models tab')
                        ]),

                html.H3('Instructions for Use'),
                html.Ul([
                    html.Li('Select from K Nearest Neighbor, Logistic Regression, Logistic Regression, Hierarchical Agglomerative Clustering models on the sidebar within the Models tab'),
                    html.Li('Change hyperparameters, such as number of neighbors and train test split, to your liking to view different versions of the Logistic Regression model'),
                    html.Li('Most visualizations are interactive, so hovering over different parts will show more detailed information.'),
                    html.Li('Dropdown menus in visualizations will allow you to visualize different variables'),
                    html.Li('Click on variables in the legend of the violin plots and bar graphs of the Hierarchical Clustering tab to isolate variables to visualize'),
                    html.Li('Note: the KNN tab takes a bit of time to render, so please be patient!')
                        ]),

                html.H3('Authors'),
                html.P('''Randa Ampah, Isabel Delgado, Aysha Hussen, Aniyah McWilliams, 
                          and Jessica Oseghale for the DS 6021 Final Project in the Fall 
                          25 semester of the UVA MSDS program''')

        ])

    if tab == "tab2":
        return html.Div(
            [
                dag.AgGrid(
                    rowData=df.to_dict("records"),
                    columnDefs=[{"field": i} for i in df.columns],
                )
            ]
        )

    if tab == "tab3":
        return dbc.Container(
            fluid=True,
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Nav(
                                [
                                    dbc.NavLink(
                                        "K Nearest Neighbor",
                                        href="/models/knn",
                                        active="exact",
                                    ),
                                    dbc.NavLink(
                                        "Logistic Regression",
                                        href="/models/logit",
                                        active="exact",
                                    ),
                                    dbc.NavLink(
                                        "Multiple Linear Regression w/ Lasso",
                                        href="/models/linear",
                                        active="exact",
                                    ),
                                    dbc.NavLink(
                                        "Hierarchical Clustering",
                                        href="/models/hierarchichal",
                                        active="exact",
                                    )
                                ],
                                vertical=True,
                                pills=True,
                            ),
                            width=2,
                            style={
                                "backgroundColor": "#f8f9fa",
                                "padding": "20px",
                                "height": "100vh",
                            },
                        ),
                        dbc.Col(
                            html.Div(id="sub-tabs-content"),
                            width=10,
                            style={"padding": "40px"},
                        ),
                    ]
                ),
                dcc.Location(id="url"),
            ],
        )


@callback(Output("sub-tabs-content", "children"), Input("url", "pathname"))
def update_sidebar_content(pathname):
    if pathname == "/models/knn":
        fig, knn_roc_fig, knn_met, knn_scatter, knn_cm = do_knn(logit_knn_X, logit_knn_y)
        return html.Div(
            [
                html.H2("K-Nearest Neighbor Classifier"),
                html.P(''),
                html.H3('Best K from Cross Validation'),
                html.P(''),
                dcc.Graph(figure=fig),
                html.H3('Model Results'),
                html.H4('Confusion Matrix of Predictions'),
                dcc.Graph(figure=knn_cm),
                html.P(''),
                html.Div([knn_met]),
                dcc.Graph(figure=knn_roc_fig),
                html.H3('Relevant Graphs'),
                html.P(''),
                dcc.Graph(figure=knn_scatter)
            ]
        )

    elif pathname == "/models/logit":
        return html.Div(
            [
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
                html.H4('Confusion Matrix of Predictions'),
                dcc.Graph(id="logit-confusion"),
                dcc.Graph(id="logit-roc"),
                dcc.Graph(id="logit-coefs"),
            ]
        )

    elif pathname == "/models/linear":
        return html.Div(
            [
                html.H2("Multiple Linear Regression (Spline with ACE Variables)"),
                html.P(''),
                html.H3("Lasso Regularization"),
                html.P('Lasso regularization was used to identify the most influential predictors to use within the linear regression model.'),
                html.Img(src='/assets/lasso_pic.png', 
                         alt='Lasso Results',
                         style={"width": "100%",
                                "height": "auto",
                                "display": "block"}),
                html.H3('MLR Results'),
                dcc.Graph(figure=fig_ace),
                html.Br(),
                html.H3("Model Performance Metrics"),
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("ACE Variable"),
                                    html.Th("Train RMSE"),
                                    html.Th("Test RMSE"),
                                    html.Th("Train Adj R²"),
                                    html.Th("Test Adj R²"),
                                ]
                            )
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(ace_labels_pretty[ace]),
                                        html.Td(
                                            f"{ace_metrics[ace]['Train RMSE']:.3f}"
                                        ),
                                        html.Td(
                                            f"{ace_metrics[ace]['Test RMSE']:.3f}"
                                        ),
                                        html.Td(
                                            f"{ace_metrics[ace]['Train Adj R2']:.3f}"
                                        ),
                                        html.Td(
                                            f"{ace_metrics[ace]['Test Adj R2']:.3f}"
                                        ),
                                    ]
                                )
                                for ace in ace_vars
                            ]
                        ),
                    ],
                    style={"width": "70%", "margin": "auto"},
                ),
            ]
        )

    elif pathname == "/models/hierarchichal":
        return html.Div([
            html.H2("Hierarchical Agglomerative Clustering"),
            html.P("Clusters based on Adverse Childhood Experiences (ACEs) using Gower Distance."),
            
            # 1. Silhouette Score
            dcc.Graph(figure=hac_sil_fig),
            html.Hr(),
            
            # 2. Heatmap
            dcc.Graph(figure=hac_heatmap_fig),
            html.Hr(),
            
            # 3. Dropdown Bar Chart
            dcc.Graph(figure=hac_cat_fig),
            html.Hr(),
            
            # 4. Mental Health Bar Charts
            dcc.Graph(figure=fig_mh_bar),
            html.Hr(),
            
            # 5. Violin Plots
            dcc.Graph(figure=fig_violin)
        ])
    return html.Div("Select a model from the sidebar.")


# ---------- LOGIT CALLBACK WITH COLORED COEFFICIENT PLOT ----------


@callback(
    Output("logit-metrics", "children"),
    Output("logit-confusion", "figure"),
    Output("logit-roc", "figure"),
    Output("logit-coefs", "figure"),
    Input("test-size-slider", "value"),
    Input("threshold-slider", "value"),
)
def update_logit_tab(test_size_pct, threshold):
    test_size = test_size_pct / 100.0

    acc, ll, cm, fpr, tpr, auc, coef_df = do_logit(
        logit_knn_X, logit_knn_y, test_size, threshold
    )

    metrics = html.Ul(
        [
            html.Li(f"Test size: {test_size_pct}%"),
            html.Li(f"Threshold: {threshold:.2f}"),
            html.Li(f"Accuracy: {acc:.3f}"),
            html.Li(f"Log loss: {ll:.3f}"),
            html.Li(f"AUC: {auc:.3f}"),
        ]
    )

    cm_fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale='RdPu',
        x=["Predicted: No depression", "Predicted: Yes depression"],
        y=["Actual: No depression", "Actual: Yes depression"],
        labels=dict(x="Predicted label", y="Actual label", color="Count"),
    )
    cm_fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))

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

    # ---- coefficient table & color-code ----
    coef_df = coef_df.copy()
    coef_df["direction"] = np.where(
        coef_df["coefficient"] > 0,
        "Risk factors (increase odds of depression)",
        "Protective factors (decrease risk)",
    )

    pretty_map = {
        "encoder__DECIDE_Yes": "Difficulty concentrating: Yes",
        "encoder__ACEDEPRS_Yes": "Lived with depressed adult: Yes",
        "encoder__BIRTHSEX_Male": "Birth sex: Male",
        "encoder__ACEHVSEX_Once": "Forced into sex: Once",
        "encoder__ACEDIVRC_Parents not married": "Parents not married",
        "encoder__ACETOUCH_Never": "Never sexually touched",
        "encoder__ACEPRISN_Yes": "Household member in prison: Yes",
        "encoder__ACEPUNCH_Once": "Witnessed violence: Once",
        "encoder__ACESWEAR_Never": "Never verbally abused",
        "encoder__ACETOUCH_Once": "Sexually touched: Once",
        "encoder__IYEAR_2021": "Survey year: 2021",
        "encoder__DIFFALON_Yes": "Difficulty errands alone: Yes",
        "encoder__ACETTHEM_Once": "Attempted sexual assault: Once",
        "encoder__ACESWEAR_Once": "Verbal abuse: Once",
        "encoder__ACEHVSEX_Never": "Never forced into sex",
    }

    coef_df["pretty_label"] = coef_df["feature"].map(pretty_map).fillna(
        coef_df["feature"]
    )

    coef_df_sorted = coef_df.sort_values("abs_coeff", ascending=True)

    coef_fig = px.bar(
        coef_df_sorted,
        x="coefficient",
        y="pretty_label",
        color="direction",
        orientation="h",
        title="Top Logistic Regression Coefficients",
        color_discrete_map={
            # darker pink = stronger risk, lighter lavender = protective
            "Risk factors (increase odds of depression)": "#d81b60",
            "Protective factors (decrease risk)": "#e1bbee",
        },
    )
    coef_fig.update_layout(
        yaxis = {'title':'Variable'},
        xaxis = {'title':'Coefficient'},
        title=dict(
            x=0.5,            # centered
            xanchor="center",
        ),
        title_font=dict(size=20),
        margin=dict(l=80, r=60, t=80, b=60),
        legend=dict(
            title="Effect direction",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
        ),
    )

    return metrics, cm_fig, roc_fig, coef_fig


if __name__ == "__main__":
    app.run(debug=True, port=8051)




# %%
