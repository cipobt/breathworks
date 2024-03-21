import pandas as pd
from utils import get_data
from preprocessing import build_preprocessor
from cleaning import clean_data, clean_textual_columns
from LDA import splitting_into_topics
from plots import corr_plot, plot_clusters, plot_clusters_2d, plot_clusters_3d
from clustering import transform_data, label_dataframe, fit_kmeans_and_label, plot_lda
from config import drop_columns, textual_columns, categorical_columns, datetime_columns, to_drop, topics_per_column, column_pairs

def main():
    # Fetch and clean data
    dataframe = get_data()
    processed_data = clean_data(dataframe,drop_columns)
    df_transformed = clean_textual_columns(processed_data, textual_columns)

    # # Apply filters
    # df_filtered = df_transformed[(df_transformed['Gender'] == 'Male') &
    #                              (df_transformed['CourseType'].isin(['OMfH','OMfH'])) &
    #                              (df_transformed['Ethnicity'] == 'White')]

    # Apply the transformations for LDA
    df_transformed = df_transformed.drop(columns=to_drop)
    df_split = splitting_into_topics(df_transformed,topics_per_column,textual_columns)
    preprocessor = build_preprocessor(textual_columns, categorical_columns, datetime_columns)
    df_LDA = preprocessor.fit_transform(df_split)

    # final df with correct column names
    transformed_columns = preprocessor.get_feature_names_out()
    df_final = pd.DataFrame(df_LDA, columns=transformed_columns)
    df_final = df_final.apply(pd.to_numeric)

    # df_2d = df_final[[col1b,col2a]]
    # labelling = fit_kmeans_and_label(df_2d,4)
    # label_dataframe(df_2d, labelling)

    # print the clusters with their labels
    print(plot_lda(df_final,column_pairs))


if __name__ == "__main__":
    main()
