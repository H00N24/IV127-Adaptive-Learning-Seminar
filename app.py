import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP

st.title("Adaptive learning seminar - Python Code")


code_frame = pd.read_csv("data/code_frame.csv", header=0)
solutions_frame = pd.read_csv("data/solutions_frame.csv", header=0)


st.sidebar.header("Settings")
solution_row = st.sidebar.selectbox(
    "Select task",
    [row for row in solutions_frame.itertuples()],
    format_func=lambda x: f"{x[1]}. {x[2]}",
)

st.header(f"Selected: {solution_row[1]} {solution_row[2]}")

st.subheader("Solution code")
st.code(solution_row[3])
st.subheader("Cleaned code")
st.code(solution_row[4])

selected_frame = pd.DataFrame(code_frame[code_frame["item"] == solution_row[1]])


id_cols = [
    "id",
    "item",
    "correct",
    "dec_answer",
    "ast_clean",
]

data_cols = [
    x
    for x in selected_frame.columns
    if x not in id_cols and np.any(selected_frame[x] != 0)
]


def describe(df, stats) -> pd.DataFrame:
    d = df.describe()
    return d.append(df.reindex(d.columns, axis=1).agg(stats))


st.subheader("Statistics")
st.write(describe(selected_frame[data_cols], ["nunique"]).T)


clustering_method = st.sidebar.selectbox(
    "Clustering method",
    ["KMeans", "Ward linkage", "Complete linkage", "Average linkage", "Single linkage"],
)
n_clusters = st.sidebar.number_input("Number of clusters", min_value=1, value=5)

methods = {
    "KMeans": KMeans(n_clusters=n_clusters, random_state=42),
    "Ward linkage": AgglomerativeClustering(n_clusters=n_clusters, linkage="ward"),
    "Complete linkage": AgglomerativeClustering(n_clusters=n_clusters, linkage="ward"),
    "Average linkage": AgglomerativeClustering(n_clusters=n_clusters, linkage="ward"),
    "Single linkage": AgglomerativeClustering(n_clusters=n_clusters, linkage="ward"),
}

scaler = StandardScaler()

scaled = scaler.fit_transform(selected_frame[data_cols])


@st.cache
def transform_data(id: int):

    pca = PCA(n_components=2, random_state=42)
    tsne = TSNE(random_state=42)
    umap = UMAP(random_state=42)

    tmp_frame = pd.DataFrame()

    for name, i in (("pca", pca), ("tsne", tsne), ("umap", umap)):
        transformed = i.fit_transform(scaled)
        tmp_frame[f"{name}_x"] = transformed[:, 0]
        tmp_frame[f"{name}_y"] = transformed[:, 1]

    return tmp_frame


selected_frame.index = np.arange(selected_frame.shape[0])

selected_frame = pd.concat([selected_frame, transform_data(solution_row[1])], axis=1)

used_clustering = methods[clustering_method]

k = used_clustering.fit_predict(scaled)

selected_frame["clustering"] = k

isoforest = IsolationForest()
isoforest.fit(scaled)
selected_frame["anomaly_score"] = isoforest.score_samples(scaled)


to_show = [
    x
    for x in selected_frame.columns
    if x
    not in [
        "item",
        "id",
        "dec_answer",
        "ast_clean",
        "dec_len",
        "clean_len",
        "pca_x",
        "pca_y",
        "tsne_x",
        "tsne_y",
        "umap_x",
        "umap_y",
    ]
]


color_by = st.sidebar.selectbox("Coloring", to_show, index=0,)
color_type = st.sidebar.selectbox("Coloring type", ["N", "Q"])


brush = alt.selection(type="interval", resolve="global")
zoom = alt.selection_interval(bind="scales", zoom=True, translate=False)

base = (
    alt.Chart(selected_frame)
    .mark_point()
    .encode(
        opacity=alt.value(0.5),
        tooltip=["id", "correct", "clustering", "edit_distance", color_by],
    )
    .add_selection(brush, zoom)
)

if color_type == "Q":
    base = base.encode(
        color=alt.condition(
            brush,
            alt.Color(f"{color_by}:{color_type}", scale=alt.Scale(scheme="viridis"),),
            alt.ColorValue("gray"),
        ),
    )
else:
    base = base.encode(
        color=alt.condition(
            brush, alt.Color(f"{color_by}:{color_type}",), alt.ColorValue("gray"),
        ),
    )


st.header("Data visualization")
(
    base.encode(
        x=alt.X("pca_x", axis=alt.Axis(labels=False, title="")),
        y=alt.Y("pca_y", axis=alt.Axis(labels=False, title="")),
    ).properties(title="PCA")
    | base.encode(
        x=alt.X("tsne_x", axis=alt.Axis(labels=False, title="")),
        y=alt.Y("tsne_y", axis=alt.Axis(labels=False, title="")),
    ).properties(title="tSNE")
    | base.encode(
        x=alt.X("umap_x", axis=alt.Axis(labels=False, title="")),
        y=alt.Y("umap_y", axis=alt.Axis(labels=False, title="")),
    ).properties(title="UMAP")
)


st.header("Data exploration")

st.write(selected_frame[["id"] + to_show])


show_codes = st.multiselect(
    "Show codes", [x for x in selected_frame.itertuples()], format_func=lambda x: x[1]
)


for row in show_codes:
    st.subheader(f"{row[1]}")
    st.write(f"Correct: {row[3]}")
    st.code(row[4])
