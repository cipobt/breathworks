import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px

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
