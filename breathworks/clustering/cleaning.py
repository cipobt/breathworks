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

        phrases_to_blank_neg = [
        "Yes, I have experienced persistent pain that has lasted for at least the last 3 months.",
        ]
        for specific_phrase in phrases_to_blank_neg:
            text = text.replace(specific_phrase, 'persistent pain for 3 months')

        phrases_to_blank_pos = [
        "No, currently I don't identify as having a chronic pain condition.",
        "I don't experience acute or debilitating depression or any other mental health condition.",
        ]
        for specific_phrase in phrases_to_blank_pos:
            text = text.replace(specific_phrase, 'none')


        for punctuation in string.punctuation:
            text = text.replace(punctuation, ' ')  # Remove Punctuation
        lowercased = text.lower()  # Lower Case
        tokenized = word_tokenize(lowercased)  # Tokenize
        words_only = [word for word in tokenized if word.isalpha()]  # Remove numbers

        stop_words = set(stopwords.words('english'))
        stop_words.update(['none','nan','yes','feel','get','take','since','like','would','way','also','year','want'])

        without_stopwords = [word for word in words_only if not word in stop_words]  # Remove Stop Words

        lemmatized = lem_complete(without_stopwords)
        more_than_two_letters = [word for word in lemmatized if len(word) > 2]
        cleaned = ' '.join(more_than_two_letters)
        return cleaned

def clean_textual_columns(df, textual_columns):
    for col in textual_columns:
        df[col] = df[col].apply(clean_text)
    return df

def lem_complete(sentence):
    # Lemmatize words considering their part of speech
    verb_lemmatized = [WordNetLemmatizer().lemmatize(word, pos="v") for word in sentence]  # v --> verbs
    noun_lemmatized = [WordNetLemmatizer().lemmatize(word, pos="n") for word in verb_lemmatized]  # n --> nouns
    adj_lemmatized = [WordNetLemmatizer().lemmatize(word, pos="a") for word in noun_lemmatized]  # a --> adjectives
    adv_lemmatized = [WordNetLemmatizer().lemmatize(word, pos="r") for word in adj_lemmatized]  # r --> adverbs
    return adv_lemmatized


def cleaning_advanced(column, words_to_remove):
  processed_column = []
  for sentence in column:
    new_sentence = []
    str_sentence = str(sentence)
    words_list = str_sentence.split()
    for x in words_list:
      if x not in words_to_remove:
        new_sentence.append(x)
    processed_sentence = ' '.join(new_sentence)
    processed_column.append(processed_sentence)

  return processed_column


def cleaning_advanced_2(df, columns, words_to_remove):
    # Iterate over each specified column
    for column in columns:
        processed_column = []
        # Iterate over each sentence in the column
        for sentence in df[column]:
            new_sentence = []
            str_sentence = str(sentence)
            words_list = str_sentence.split()
            # Remove specified words
            for word in words_list:
                if word not in words_to_remove:
                    new_sentence.append(word)
            processed_sentence = ' '.join(new_sentence)
            processed_column.append(processed_sentence)
        # Update the DataFrame column with cleaned data
        df[column] = processed_column
    return df

def cleaning_advanced_column(column, words_to_remove):
  processed_column = []
  for sentence in column:
    new_sentence = []
    str_sentence = str(sentence)
    words_list = str_sentence.split()
    for x in words_list:
      if x not in words_to_remove:
        new_sentence.append(x)
    processed_sentence = ' '.join(new_sentence)
    processed_column.append(processed_sentence)

  return processed_column
