import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def corr_plot(df):
    plt.figure(figsize=(16,16))
    sns.heatmap(df.corr(), annot=True, annot_kws={"fontsize":20}, cmap='PuRd')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_clusters(df, labels):
    # Generate and show a 3D scatter plot of the clusters
    fig_scaled = px.scatter_3d(df,
                            x=0, y=1, z=2,
                            color=labels,
                            width=500, height=500)
    fig_scaled.show()


def plot_clusters_2d(df, labels):
    fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color=labels,
                     width=600, height=600)

    fig.update_traces(marker=dict(size=12, line=dict(width=2)))

    fig.show()


def plot_clusters_3d(df, labels):
    # Generate and show a 3D scatter plot of the clusters
    fig = px.scatter_3d(df,
                        x=df.columns[0], y=df.columns[1], z=df.columns[2],
                        color=labels,
                        width=800, height=800)
    fig.show()
