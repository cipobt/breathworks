from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from bertopic import BERTopic

pipe = make_pipeline(
    TfidfVectorizer(max_df=0.95, min_df=7, ngram_range=(1,3), stop_words='english'),
    TruncatedSVD(100)
)

def get_topics_prob(input, topic_count):
    model = BERTopic(nr_topics=topic_count, embedding_model=pipe, verbose=True)

    # Convert to list
    docs = input.to_list()
    docs = [str(i) for i in docs]

    topics, probabilities = model.fit_transform(docs)

    return topics, probabilities, model
