import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


class DataPreprocessor:
    def __init__(self, path):
        self.y_identity_hate = None
        self.y_insult = None
        self.y_threat = None
        self.y_obscene = None
        self.y_severe_toxic = None
        self.y_toxic = None
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

    def remove_stopwords(self, text):
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in text if word not in stop_words]
        return filtered_words

    def preprocess_data(self):
        self.data["cleaned"] = self.data["comment_text"].apply(self.clean_text)
        self.data["lemmatized"] = self.data["cleaned"].apply(self.lemmatize_text)
        self.data["stop_words"] = self.data["lemmatized"].apply(self.remove_stopwords)
        self.data["comment_text"] = self.data["stop_words"].apply(lambda x: ' '.join(x))

    def get_XY(self):
        self.X = self.data["comment_text"]
        self.y = self.data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
        return self.X, self.y
