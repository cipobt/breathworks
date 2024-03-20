from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import set_config

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def build_preprocessor(textual_columns, categorical_columns, datetime_columns):
    text_pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('lda', LatentDirichletAllocation(n_components=2))
    ])

    datetime_pipeline = Pipeline([
        ('scaler', RobustScaler())
    ])

    cat_pipeline = Pipeline([
        ('scaler', OneHotEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
    #        ('text', text_pipeline, textual_columns),
            ('cat', cat_pipeline, categorical_columns),
            ('num', datetime_pipeline, datetime_columns)
        ],
        remainder='passthrough'
    )
    return preprocessor

