import pandas as pd
from utils import get_data
from preprocessing import build_preprocessor
from cleaning import clean_data, clean_textual_columns
from LDA import splitting_into_topics
from plots import corr_plot, plot_clusters
from clustering import transform_data

drop_columns=['Daily20MPractice', 'ReferralSource','Location','DoB']
textual_columns = ['PersonalHistory', 'Motivation']
categorical_columns = ['CourseType', 'Gender', 'Ethnicity']
datetime_columns = ['Days_Since_EarliestCourse', 'AgeAtCourse']
non_textual = categorical_columns + datetime_columns


def main():
    # Fetch and clean data
    dataframe = get_data()
    processed_data = clean_data(dataframe,drop_columns)
    df_transformed = clean_textual_columns(processed_data, textual_columns)

    # Apply filters
    df_filtered = df_transformed[(df_transformed['Gender'] == 'Male') &
                                 (df_transformed['CourseType'].isin(['OMfH','OMfH'])) &
                                 (df_transformed['Ethnicity'] == 'White')]

    # Apply the transformations
    df_split = splitting_into_topics(df_filtered,4,textual_columns)
    preprocessor = build_preprocessor(textual_columns, categorical_columns, datetime_columns)
    df_transformed = preprocessor.fit_transform(df_split)
    transformed_columns = preprocessor.get_feature_names_out()
    df_final = pd.DataFrame(df_transformed, columns=transformed_columns)
    df_final = df_final.apply(pd.to_numeric)
    df_proj, labels = transform_data(df_final, 5, 2)
    corr_plot(df_final)
    plot_clusters(df_proj, labels, 100)



if __name__ == "__main__":
    main()
