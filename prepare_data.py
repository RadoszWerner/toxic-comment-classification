import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

# Load data
train_data = pd.read_csv("jigsaw-toxic-comment-classification-challenge/train.csv")
train_data["comment_text"] = train_data["comment_text"].str.lower()

# Preprocessing data
def cleaning(data):
    clean_column = re.sub('<.*?>', ' ', str(data))
    clean_column = re.sub('[^a-zA-Z0-9.]+', ' ', clean_column)
    tokenized_column = word_tokenize(clean_column)
    return tokenized_column

train_data["cleaned"] = train_data["comment_text"].apply(cleaning)

# Lemmatization
lemmatizer = WordNetLemmatizer()

def lemmatizing(data):
    lemmatized_list = [lemmatizer.lemmatize(word) for word in data]
    return lemmatized_list

train_data["lemmatized"] = train_data["cleaned"].apply(lemmatizing)
train_data["comment_text"] = train_data["lemmatized"].apply(lambda x: ' '.join(x))

X = train_data["comment_text"]
y = train_data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]