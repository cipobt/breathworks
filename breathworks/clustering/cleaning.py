import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


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


        phrases_to_blank = [
        "No, currently I don't identify as having a chronic pain condition.",
        "Yes, I have experienced persistent pain that has lasted for at least the last 3 months."
        ]
        for specific_phrase in phrases_to_blank:
            text = text.replace(specific_phrase, ' ')


        for punctuation in string.punctuation:
            text = text.replace(punctuation, ' ')  # Remove Punctuation
        lowercased = text.lower()  # Lower Case
        tokenized = word_tokenize(lowercased)  # Tokenize
        words_only = [word for word in tokenized if word.isalpha()]  # Remove numbers

        stop_words = set(stopwords.words('english'))
        stop_words.update(['none','nan'])

        without_stopwords = [word for word in words_only if not word in stop_words]  # Remove Stop Words

        # lemma = WordNetLemmatizer()  # Initiate Lemmatizer
        # lemmatized = [lemma.lemmatize(word) for word in without_stopwords]  # Lemmatize
        lemmatized = lem_complete(without_stopwords)
        cleaned = ' '.join(lemmatized)  # Join back to a string
        return cleaned

def clean_textual_columns(df, textual_columns):
    for col in textual_columns:
        df[col] = df[col].apply(clean_text)
    return df

def lem_complete(sentence):
    verb_lemmatized = [
            WordNetLemmatizer().lemmatize(word, pos = "v") # v --> verbs
            for word in sentence
    ]

    # 2 - Lemmatizing the nouns
    noun_lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos = "n") # n --> nouns
        for word in verb_lemmatized
    ]

    adj_lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos = "a")
        for word in noun_lemmatized
    ]

    # adv_lemmatized=[
    #     WordNetLemmatizer().lemmatize(word, pos = "adv")
    #     for word in adj_lemmatized
    # ]

    return adj_lemmatized
