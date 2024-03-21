import pandas as pd
from utils import get_data
from preprocessing import build_preprocessor
from cleaning import clean_data, clean_textual_columns
from LDA import splitting_into_topics
from plots import corr_plot, plot_clusters, plot_clusters_2d, plot_clusters_3d
from clustering import transform_data, label_dataframe, fit_kmeans_and_label, plot_lda
from config import drop_columns, textual_columns, categorical_columns, datetime_columns, to_drop, topics_per_column, column_pairs


# drop_columns=['Daily20MPractice', 'ReferralSource','Location','DoB']
# textual_columns = ['PersonalHistory', 'Motivation']
# categorical_columns = ['CourseType', 'Gender', 'Ethnicity']
# datetime_columns = ['Days_Since_EarliestCourse', 'AgeAtCourse']
# non_textual = categorical_columns + datetime_columns
# to_drop = ['Gender', 'CourseType', 'Ethnicity']
# categorical_columns = [column for column in categorical_columns if column not in to_drop]

# topics_per_column = {
#     'PersonalHistory': 3,
#     'Motivation': 3,
# }

# col1a='remainder__PersonalHistory_Topic0'
# col2a='remainder__Motivation_Topic0'
# col1b='remainder__PersonalHistory_Topic1'
# col2b='remainder__Motivation_Topic1'
# col1c='remainder__PersonalHistory_Topic2'
# col2c='remainder__Motivation_Topic2'

# column_pairs = [
#     (col1a, col2a),
#     (col1a, col2b),
#     (col1a, col2c),
#     (col1b, col2a),
#     (col1b, col2b),
#     (col1b, col2c),
#     (col1c, col2a),
#     (col1c, col2b),
#     (col1c, col2c),
# ]

def main():
    # Fetch and clean data
    dataframe = get_data()
    processed_data = clean_data(dataframe,drop_columns)
    df_transformed = clean_textual_columns(processed_data, textual_columns)

    # # Apply filters
    # df_filtered = df_transformed[(df_transformed['Gender'] == 'Male') &
    #                              (df_transformed['CourseType'].isin(['OMfH','OMfH'])) &
    #                              (df_transformed['Ethnicity'] == 'White')]

    # Apply the transformations
    df_transformed = df_transformed.drop(columns=to_drop)
    df_split = splitting_into_topics(df_transformed,topics_per_column,textual_columns)
    preprocessor = build_preprocessor(textual_columns, categorical_columns, datetime_columns)
    df_LDA = preprocessor.fit_transform(df_split)

    transformed_columns = preprocessor.get_feature_names_out()
    df_final = pd.DataFrame(df_LDA, columns=transformed_columns)
    df_final = df_final.apply(pd.to_numeric)

    # df_2d = df_final[[col1b,col2a]]
    # labelling = fit_kmeans_and_label(df_2d,4)
    # label_dataframe(df_2d, labelling)

    print(plot_lda(df_final,column_pairs))


if __name__ == "__main__":
    main()
