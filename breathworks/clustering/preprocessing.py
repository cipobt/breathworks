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

'-------------------------------------------'

# df_transformed = preprocessor.fit_transform(df_cleaned)
# transformed_columns = textual + [col for col in df_cleaned.columns if col not in textual]
# df_cleaned = pd.DataFrame(df_transformed, columns=transformed_columns)

# transformed = pipeline.fit_transform(df_cleaned)
# new_columns = preprocessor.get_feature_names_out()
# df_num = pd.DataFrame(transformed, columns=new_columns)
# df_num = df_num.apply(pd.to_numeric)



# pca = PCA()
# pca.fit(df_num)


# pca = PCA(n_components=threhsold_pca, whiten=True)
# pca.fit(df_num)
# df_proj = pd.DataFrame(pca.transform(df_num))
# df_proj

# kmeans = KMeans(n_clusters = elbow_highlight, max_iter = 300)

# kmeans.fit(df_proj)

# labelling = kmeans.labels_

# fig_scaled = px.scatter_3d(df_proj,
#                            x = 0,
#                            y = 1,
#                            z = 2,
#                            color=labelling,
#                            width=500,
#                            height=500)
# fig_scaled.show()
