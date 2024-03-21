import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def transform_data(df, threshold_pca, elbow_highlight):
    pca = PCA(n_components=threshold_pca, whiten=True)
    pca.fit(df)
    df_proj = pd.DataFrame(pca.transform(df))

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=elbow_highlight, max_iter=300, n_init=10)
    kmeans.fit(df_proj)
    labels = kmeans.labels_

    # Return the PCA-transformed DataFrame and cluster labels
    return df_proj, labels


def remove_low_variance_features(df, variance_threshold=0.1):
    """Remove features with low variance."""
    return df.loc[:, df.var() > variance_threshold]

def remove_high_correlation_features(df, correlation_threshold=0.9):
    corr_matrix = df.corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

    df_reduced = df.drop(columns=to_drop)
    return df_reduced


def label_dataframe(df, labels, new_label_column_name="label"):
    # Ensure labels is a pandas Series
    if not isinstance(labels, pd.Series):
        labels = pd.Series(labels)

    # Concatenate the DataFrame and the labels Series
    df_labelled = pd.concat([df, labels], axis=1).rename(columns={0: new_label_column_name})
    return df_labelled

def fit_kmeans_and_label(df, n_clusters, max_iter=300):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter)
    kmeans.fit(df)
    labels = kmeans.labels_
    return labels
#df2 dffinal

def plot_lda(df2,column_pairs):
    fig = make_subplots(rows=1, cols=len(column_pairs),
                        subplot_titles=[f'{col1} vs {col2}' for col1, col2 in column_pairs])

    for i, (col1, col2) in enumerate(column_pairs, start=1):
        df = df2[[col1, col2]]
        labelling = fit_kmeans_and_label(df, 4)
        df_labeled = label_dataframe(df, labelling, "Cluster")

        # Plot
        for label in df_labeled['Cluster'].unique():
            cluster_df = df_labeled[df_labeled['Cluster'] == label]
            fig.add_trace(
                go.Scatter(x=cluster_df[col1], y=cluster_df[col2],
                        mode='markers', name=f'Cluster {label}',
                        marker=dict(size=12, line=dict(width=2))),
                row=1, col=i
            )

    fig.update_layout(height=600, width=600*len(column_pairs), title_text="Cluster Plots Side by Side")
    fig.show()
