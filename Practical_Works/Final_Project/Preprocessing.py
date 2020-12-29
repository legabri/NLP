import numpy as np
import pandas as pd

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def read_txt(filename):
    data = pd.read_csv(filename, delimiter="\t", header=None)
    data.columns = ["Word", "Label"]

    # Split the dataset by empty lines to group sentences together.
    sentences = list(np.split(data, data[data['Label'].isnull()].index.tolist()))

    # Remove NaN values from the original dataset.
    sentences = [sentence.dropna().to_records(index=False) for sentence in sentences if not sentence['Label'].isna().all()]
    data = data[data['Label'].notna()]

    return data, sentences

def parse_sentences(word2idx, label2idx, labels_count, sentences, max_length, categorical):
    # Get the indices of the words and pad if necessary using the PAD tag. If a word doesn't exist, use the UNK tag.
    X = [[word2idx[word[0]] if word[0] in word2idx else word2idx["UNK"] for word in sentence] for sentence in sentences]
    X = pad_sequences(X, max_length, padding="post", value=word2idx["PAD"])

    # Get the indices of the labels based on the dictionary.
    y = [[label2idx[word[1]] for word in sentence] for sentence in sentences]
    y = pad_sequences(y, max_length, padding="post", value=label2idx["O"])

    if categorical:
        # Transform the labels into a one-hot vector.
        y = np.array([to_categorical(index, labels_count) for index in y])

    return X, y