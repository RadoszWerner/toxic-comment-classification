import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize, WordNetLemmatizer

#tylko za 1 razem
nltk.download('wordnet')
nltk.download('point')

def main():
    train_data = pd.read_csv("jigsaw-toxic-comment-classification-challenge/train.csv")
    train_data["comment_text"] = train_data["comment_text"].str.lower()
    print(train_data["comment_text"])
    print("pre clean:\n\n", train_data["comment_text"])

    def cleaning(data):
        # remove the characters in the first parameter
        clean_column = re.sub('<.*?>', ' ', str(data))
        # removes non-alphanumeric characters except periods.
        clean_column = re.sub('[^a-zA-Z0-9.]+', ' ', clean_column)
        # tokenize
        tokenized_column = word_tokenize(clean_column)
        return tokenized_column

    train_data["cleaned"] = train_data["comment_text"].apply(cleaning)
    print("post clean:\n\n", train_data["cleaned"])

    # lemmatize all the words
    lemmatizer = WordNetLemmatizer()

    def lemmatizing(data):
        # input our data in function, take the cleaned column
        my_data = data["cleaned"]
        lemmatized_list = [lemmatizer.lemmatize(word) for word in my_data]
        return (lemmatized_list)

    train_data["lemmatized"] = train_data.apply(lemmatizing, axis=1)
    print("post lemmatize:\n\n", train_data["lemmatized"])

    train_data["comment_text"] = train_data["lemmatized"]
    train = train_data[["comment_text"]]
    train_labels = train_data[["toxic"]]
    print(train)

    # 2. Use train_test_split to split into train/test
    comment_train, comment_test, labels_train, labels_test = train_test_split(train, train_labels, test_size=0.2,
                                                                              random_state=42)
    # Transpose and flatten so it fits the correct dimensions
    labels_train = np.transpose(labels_train)
    labels_train = np.ravel(labels_train)
    labels_test = np.transpose(labels_test)
    labels_test = np.ravel(labels_test)

    # 3. CountVectorizer
    # Create a count matrix for each comment
    count_vect = CountVectorizer()
    comment_train_counts = count_vect.fit_transform(comment_train.comment_text.astype(str))
    print("dupa")


if __name__ == '__main__':
    main()
