# # Data Manipulation
# import numpy as np
import pandas as pd
# import os

# # Data Visualisation
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px

# # Pipeline and Column Transformers
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn import set_config

# # Scaling
# from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

# # Cross Validation
# from sklearn.model_selection import cross_validate
# from sklearn.model_selection import cross_val_predict

# # Unsupervised Learning
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans

# # STATISTICS
# from statsmodels.graphics.gofplots import qqplot

# Text Processing
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# # OneHotEncoder
# from sklearn.preprocessing import OneHotEncoder

# # NLTK Downloads
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# # Set pandas display option
# pd.set_option('display.max_columns', None)

# # Set sklearn display configuration
# set_config(display = "diagram")

# # Custom Transformers and Model Building
# from sklearn.base import BaseEstimator, TransformerMixin


def clean_data(df,dropping):
    df = df.drop(columns=dropping)
    df['CourseDate'] = pd.to_datetime(df['CourseDate'])
    earliest_course_date = df['CourseDate'].min()
    df['Days_Since_EarliestCourse'] = (df['CourseDate'] - earliest_course_date).dt.days
    df = df.drop(columns=['CourseDate'])
    df_cleaned = df.dropna().copy()

    return df_cleaned


def clean_text(text):
        text = str(text)
        for punctuation in string.punctuation:
            text = text.replace(punctuation, ' ')  # Remove Punctuation
        lowercased = text.lower()  # Lower Case
        tokenized = word_tokenize(lowercased)  # Tokenize
        words_only = [word for word in tokenized if word.isalpha()]  # Remove numbers

        stop_words = set(stopwords.words('english'))
        stop_words.update(['yes','none','nan'])

        without_stopwords = [word for word in words_only if not word in stop_words]  # Remove Stop Words
        lemma = WordNetLemmatizer()  # Initiate Lemmatizer
        lemmatized = [lemma.lemmatize(word) for word in without_stopwords]  # Lemmatize
        cleaned = ' '.join(lemmatized)  # Join back to a string
        return cleaned


def clean_textual_columns(df, textual_columns):
    for col in textual_columns:
        df[col] = df[col].apply(clean_text)
    return df
