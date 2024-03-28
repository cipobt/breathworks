from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import pyLDAvis
import pyLDAvis.lda_model
pyLDAvis.enable_notebook()

def splitting_into_topics(df, topics_per_col, textual_columns):
    lda_details = {}
    for text in textual_columns:
        vect = TfidfVectorizer(max_df=0.95, min_df=7, ngram_range=(1,3))
        X = vect.fit_transform(df[text])

        num_topics = topics_per_col[text]

        lda = LatentDirichletAllocation(n_components=num_topics)
        lda.fit(X)

        topic_distributions = lda.transform(X)
        # topic_columns = [f'{text}_Topic{i}' for i in range(topics)]
        topics_df = pd.DataFrame(topic_distributions, columns=[f'{text}_Topic{i}' for i in range(num_topics)], index=df.index)


        def print_topics(model, vectorizer, num_words=10):
            for idx, topic in enumerate(model.components_):
                print(f'{text}')
                print("Topic %d:" % (idx))
                print([(vectorizer.get_feature_names_out()[i], topic[i])
                    for i in topic.argsort()[:-num_words - 1:-1]])


        print_topics(lda, vect)

        lda_details[text] = {'lda': lda, 'X': X, 'vect': vect}

        # Concatenate the new columns with the original DataFrame
        df = pd.concat([df, topics_df], axis=1)

    df = df.drop(columns=textual_columns)

    return df, lda_details

def lda_visual(model, data_vec, vect):
    panel = pyLDAvis.lda_model.prepare(model,data_vec, vect, mds='tsne')
    return pyLDAvis.display(panel)
