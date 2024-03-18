import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def corr_plot(df):
    plt.figure(figsize=(8,8))

    sns.heatmap(df.corr(),
                annot = True,
                annot_kws = {"fontsize":4},
                cmap='PuRd');
    plt.show()

def plot_clusters(df, labels):
    # Generate and show a 3D scatter plot of the clusters
    sampled_df = df.sample(n=1000)  # Adjust n to your needs
    fig_scaled = px.scatter_3d(sampled_df,
                            x=0, y=1, z=2,
                            color=sampled_df[labels].iloc[:1000],  # Make sure to sample labels correspondingly
                            width=500, height=500)
    fig_scaled.show()
