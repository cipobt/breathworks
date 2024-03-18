import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px

def transform_data(df, threshold_pca, elbow_highlight):
    # Perform PCA
    pca = PCA(n_components=threshold_pca, whiten=True)
    pca.fit(df)
    df_proj = pd.DataFrame(pca.transform(df))

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=elbow_highlight, max_iter=300, n_init=10)
    kmeans.fit(df_proj)
    labels = kmeans.labels_

    # Return the PCA-transformed DataFrame and cluster labels
    return df_proj, labels

def plot_clusters(df_proj, labels):
    # Generate and show a 3D scatter plot of the clusters
    fig_scaled = px.scatter_3d(df_proj,
                               x=0,
                               y=1,
                               z=2,
                               color=labels,
                               width=500,
                               height=500)
    fig_scaled.show()


# pca = PCA()
# pca.fit(df_num)
# threhsold_pca = 4
# with plt.style.context('seaborn-deep'):
#     # figsize
#     plt.figure(figsize=(10,6))
#     # getting axes
#     ax = plt.gca()
#     # plotting
#     explained_variance_ratio_cumulated = np.cumsum(pca.explained_variance_ratio_)
#     x_axis_ticks = np.arange(1,explained_variance_ratio_cumulated.shape[0]+1)
#     ax.plot(x_axis_ticks,explained_variance_ratio_cumulated,label="cumulated variance ratio",color="purple",linestyle=":",marker="D",markersize=10)
#     # customizing
#     ax.set_xlabel('Number of Principal Components')
#     ax.set_ylabel('% cumulated explained variance')
#     ax.legend(loc="upper left")
#     ax.set_title('The Elbow Method')
#     ax.set_xticks(x_axis_ticks)
#     ax.scatter(threhsold_pca,explained_variance_ratio_cumulated[threhsold_pca-1],c='blue',s=400)
#     ax.grid(axis="x",linewidth=0.5)
#     ax.grid(axis="y",linewidth=0.5)

#     pca = PCA(n_components=threhsold_pca, whiten=True)
# pca.fit(df_num)
# df_proj = pd.DataFrame(pca.transform(df_num))
# df_proj

# fig_scaled = px.scatter_3d(df_proj, x = 0, y = 1, z = 2, opacity=0.7, width=500, height=500)
# fig_scaled.show()


# wcss = []
# for K in nb_clusters_to_try:
#     kmeans = KMeans(n_clusters = K)
#     kmeans.fit(df_proj)
#     wcss.append(kmeans.inertia_)


# kmeans = KMeans(n_clusters = elbow_highlight, max_iter = 300)

# kmeans.fit(df_proj)

# labelling = kmeans.labels_

# fig_scaled = px.scatter_3d(df_proj,
#                            x = 0,
#                            y = 1,
#                            z = 2,
#                            color=labelling,
#                            width=500,
#                            height=500)
# fig_scaled.show()
