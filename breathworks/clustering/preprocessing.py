from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import set_config

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from breathworks.clustering.cleaning import clean_text

import numpy as np
import pandas as pd

# def build_preprocessor(textual_columns, categorical_columns, datetime_columns):
#     text_pipeline = Pipeline([
#         ('vect', CountVectorizer(stop_words='english', ngram_range=(1, 2))),
#         ('lda', LatentDirichletAllocation(n_components=2))
#     ])

#     datetime_pipeline = Pipeline([
#         ('scaler', RobustScaler())
#     ])

#     cat_pipeline = Pipeline([
#         ('scaler', OneHotEncoder())
#     ])

#     preprocessor = ColumnTransformer(
#         transformers=[
#     #        ('text', text_pipeline, textual_columns),
#             ('cat', cat_pipeline, categorical_columns),
#             ('num', datetime_pipeline, datetime_columns)
#         ],
#         remainder='passthrough'
#     )
#     return preprocessor

def build_preprocessor():#

    # df_text = df.select_dtypes(include=[object])

    text_pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('lda', LatentDirichletAllocation(n_components=5))  # Adjust the number of components as needed
    ])

    # final_pipe = ColumnTransformer(
    #     transformers=[
    #         ('text', text_pipeline, df_text)
    #     ],
    #     remainder='passthrough'
    # )
    return text_pipeline


def simple_preprocessor_with_topics(df, column_name, num_topics=5):

    # cvec = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    # lda_details = {}
    # df_text = df.select_dtypes(include=[object])

    df[column_name] = df[column_name].apply(clean_text)

    preprocessor = Pipeline([
        ('vect', TfidfVectorizer(max_df = 0.9, min_df=2, ngram_range=(1, 2), stop_words='english')),
        ('lda', LatentDirichletAllocation(n_components=num_topics))  # Adjust the number of components as needed
    ])

    vect = preprocessor.named_steps['vect']
    lda = preprocessor.named_steps['lda']

    topic_labelled_df = pd.DataFrame(preprocessor.fit_transform(df[column_name]), index = df[column_name])
    topic_labelled_df['topic_label'] = topic_labelled_df.apply(lambda row: topic_labelled_df.columns[row.argmax()], axis=1)
    topic_labelled_df

    lst = []
    for idx, topic in enumerate(lda.components_):
        lst.append([(vect.get_feature_names_out()[i], topic[i])
            for i in topic.argsort()[:-10 - 1:-1]])
    topic_only_df = pd.DataFrame(lst)

    print('topic_distribution', topic_labelled_df['topic_label'].value_counts() / len(topic_labelled_df) * 100)


    def topic_extractor(text):
        text = str(text)
        text = text.lstrip('(').split(',')[0]
        return text

    topic_df_final = pd.concat([pd.DataFrame(topic_labelled_df['topic_label'].value_counts() / len(topic_labelled_df) * 100).sort_values(by='topic_label'), pd.DataFrame(topic_only_df.map(lambda x: topic_extractor(x)).T.sum())
          ], axis=1).rename(columns={'count': 'Distribution', 0:'Keywords'})

    return topic_labelled_df, topic_df_final
