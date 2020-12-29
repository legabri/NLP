import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Embedding, Dropout, Bidirectional, LSTM, TimeDistributed, Dense
from keras.metrics import Precision, Recall
from keras.callbacks import EarlyStopping
from seqeval.metrics import f1_score, classification_report

from Preprocessing import read_txt, parse_sentences
from Postprocessing import pred2label

class BiLSTM(tf.keras.Model):
  def __init__(self, words_count, labels_count, max_length):
    super(BiLSTM, self).__init__()

    self.embedding = Embedding(words_count, 50, input_length=max_length)
    self.dropout = Dropout(0.1)
    self.lstm = Bidirectional(LSTM(100, recurrent_dropout=0.1, return_sequences=True))
    self.dense = TimeDistributed(Dense(labels_count, "softmax"))

  def call(self, inputs):
    x = self.embedding(inputs)
    x = self.dropout(x)
    x = self.lstm(x)
    x = self.dense(x)

    return x

if __name__ == "__main__":
    train, train_sentences = read_txt("data/science/train.txt")
    dev, dev_sentences = read_txt("data/science/dev.txt")
    test, test_sentences = read_txt("data/science/test.txt")

    words = list(set(train["Word"].values))
    words.extend(["PAD", "UNK"])
    words_count = len(words)

    labels = list(set(train["Label"].values))
    labels_count = len(labels)

    word2idx = {word: index for index, word in enumerate(words)}
    label2idx = {label: index for index, label in enumerate(labels)}

    # Plot length of sentences.
    #plt.clf()
    #plt.hist([len(sentence) for sentence in sentences], bins="auto")
    #plt.title("histogram")
    #plt.savefig("histogram.png")

    max_length = 50

    X_train, y_train = parse_sentences(word2idx, label2idx, labels_count, train_sentences, max_length, True)
    X_dev, y_dev = parse_sentences(word2idx, label2idx, labels_count, dev_sentences, max_length, True)
    X_test, y_test = parse_sentences(word2idx, label2idx, labels_count, test_sentences, max_length, True)

    model = BiLSTM(words_count, labels_count, max_length)

    callback = EarlyStopping(monitor="val_accuracy", patience=20, verbose=1)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy", Precision(), Recall()])
    history = model.fit(X_train, y_train, validation_data=(X_dev, y_dev), callbacks=[callback], batch_size=32, epochs=2, verbose=1)
    history = pd.DataFrame(history.history)

    plt.clf()
    plt.plot(history["accuracy"])
    plt.plot(history["val_accuracy"])
    plt.title("Accuracy")
    plt.savefig("accuracy.png")

    plt.clf()
    plt.plot(history["precision"])
    plt.plot(history["val_precision"])
    plt.title("Precision")
    plt.savefig("precision.png")

    plt.clf()
    plt.plot(history["recall"])
    plt.plot(history["val_recall"])
    plt.title("Recall")
    plt.savefig("recall.png")

    predictions = model.predict(X_test, batch_size=32, verbose=1)
    pred_labels = pred2label(predictions, label2idx)
    test_labels = pred2label(y_test, label2idx)

    print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
    print(classification_report(test_labels, pred_labels))