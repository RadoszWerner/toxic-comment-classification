import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

class DataPreprocessor:
    def __init__(self, path):
        self.path = path
        self.data = None
        self.X = None
        self.y = None

    def load_data(self):
        self.data = pd.read_csv(self.path)
        self.data["comment_text"] = self.data["comment_text"].str.lower()
        print(self.data)

    def clean_text(self, text):
        clean_column = re.sub('<.*?>', ' ', str(text))
        clean_column = re.sub('[^a-zA-Z0-9.]+', ' ', clean_column)
        tokenized_column = word_tokenize(clean_column)
        return tokenized_column

    def lemmatize_text(self, tokens):
        lemmatizer = WordNetLemmatizer()
        lemmatized_list = [lemmatizer.lemmatize(word) for word in tokens]
        return lemmatized_list

    def preprocess_data(self):
        self.data["cleaned"] = self.data["comment_text"].apply(self.clean_text)
        self.data["lemmatized"] = self.data["cleaned"].apply(self.lemmatize_text)
        self.data["comment_text"] = self.data["lemmatized"].apply(lambda x: ' '.join(x))

    def get_XY(self):
        self.X = self.data["comment_text"]
        self.y = self.data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
        return self.X, self.y