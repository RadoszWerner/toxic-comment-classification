import pandas as pd
import numpy as np
import re
import nltk
from gensim.models import Word2Vec
from nltk import word_tokenize

nltk.download('punkt')


def main():
    train_data = pd.read_csv("jigsaw-toxic-comment-classification-challenge/train.csv")
    train_data["comment_text"] = train_data["comment_text"].str.lower()
    print(train_data["comment_text"])
    print("pre clean:\n\n", train_data["comment_text"])

    def cleaning(data):
        # remove the characters in the first parameter
        clean_column = re.sub('<.*?>', ' ', str(data))
        # removes non-alphanumeric characters(exclamation point, colon, etc) except periods.
        clean_column = re.sub('[^a-zA-Z0-9.]+', ' ', clean_column)
        # tokenize
        tokenized_column = word_tokenize(clean_column)
        return tokenized_column

    train_data["cleaned"] = train_data["comment_text"].apply(cleaning)
    print("post clean:\n\n", train_data["cleaned"])


if __name__ == '__main__':
    main()
