from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

def splitting_into_topics(df, topics, textual_columns):
    for text in textual_columns:
        # df.loc[:, text] = df.loc[:, text].fillna('')
        # Vectorize the text
        vect = CountVectorizer(stop_words='english')
        X = vect.fit_transform(df[text])

        lda = LatentDirichletAllocation(n_components=topics, random_state=0)
        lda.fit(X)

        topic_distributions = lda.transform(X)
        topic_columns = [f'{text}_Topic{i}' for i in range(topics)]
        topics_df = pd.DataFrame(topic_distributions, columns=[f'{text}_Topic{i}' for i in range(topics)], index=df.index)

        # Concatenate the new columns with the original DataFrame
        df = pd.concat([df, topics_df], axis=1)

    df = df.drop(columns=textual_columns)

    return df
