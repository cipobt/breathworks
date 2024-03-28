import pandas as pd
import sys
sys.path.append('../')
from utils import get_data, go_to_data
from cleaning import clean_text, clean_data, clean_textual_columns, cleaning_advanced_column, cleaning_advanced_2
from config import drop_columns, textual_columns, topics_per_column, datetime_columns, column_pairs

def run_lda_analysis():
    # import the modules
    from preprocessing import build_preprocessor
    from LDA import splitting_into_topics, lda_visual
    from clustering import plot_lda

    # Fetch and clean data
    dataframe = get_data()
    df_rem = go_to_data('remove2.csv')
    df_combined = go_to_data('maybe_3_topic_label.csv')
    row_labels_list = df_rem['Row Labels'].tolist()
    processed_data = clean_data(dataframe, drop_columns)
    df_transformed = clean_textual_columns(processed_data, textual_columns)
    df_transformed = cleaning_advanced_2(df_transformed, textual_columns, row_labels_list)

    # filter section
    in_person = ['IMfH', 'IMfS', 'I5DMfH']
    online = ['OMfH', 'OMfS']
    to_drop = ['Gender', 'CourseType', 'Ethnicity']
    categorical_columns = [column for column in categorical_columns if column not in to_drop]

    # Apply the transformations for LDA
    df_transformed = df_transformed.drop(columns=to_drop, errors='ignore')
    df_split, lda_details = splitting_into_topics(df_transformed,topics_per_column,textual_columns)
    preprocessor = build_preprocessor(textual_columns, categorical_columns, datetime_columns)
    df_LDA = preprocessor.fit_transform(df_split)

    # visualize the lda
    lda_ph =lda_details['PersonalHistory']['lda']
    x_ph = lda_details['PersonalHistory']['X']
    vect_ph = lda_details['PersonalHistory']['vect']
    lda_visual(lda_ph, x_ph, vect_ph)
    lda_m =lda_details['Motivation']['lda']
    x_m = lda_details['Motivation']['X']
    vect_m = lda_details['Motivation']['vect']
    lda_visual(lda_m, x_m, vect_m)

    # getting final dataframe
    transformed_columns = preprocessor.get_feature_names_out()
    df_final = pd.DataFrame(df_LDA, columns=transformed_columns)
    df_final = df_final.apply(pd.to_numeric)

    # plotting the clusters
    plot_lda(df_final,column_pairs)

def run_bert_analysis():
    #import the modules
    from bert import get_topics_prob

    # Fetch and clean data
    df_rem = go_to_data('remove2.csv')
    df_combined = go_to_data('maybe_3_topic_label.csv')
    row_labels_list = df_rem['Row Labels'].tolist()
    advanced_cleaning_col= pd.Series(cleaning_advanced_column(df_combined['CustomerPurpose'], row_labels_list))
    df_combined_final = advanced_cleaning_col.apply(clean_text)

    #init BERT
    topics, probabilities, model = get_topics_prob(df_combined_final, 8)

    # visualize BERT
    model.visualize_topics()
    model.visualize_barchart(top_n_topics=8)

def main(analysis_type):
    if analysis_type == 'lda':
        run_lda_analysis()
    elif analysis_type == 'bert':
        run_bert_analysis()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analysis_type = sys.argv[1]
        main(analysis_type)
    else:
        print("Please specify the analysis type: 'lda', 'bert'")
