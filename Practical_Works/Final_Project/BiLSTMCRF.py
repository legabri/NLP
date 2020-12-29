import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense
from keras.callbacks import EarlyStopping
from seqeval.metrics import f1_score, classification_report
from tensorflow_addons.layers.crf import CRF
from tensorflow_addons.text.crf import crf_log_likelihood

from Preprocessing import read_txt, parse_sentences
from Postprocessing import pred2label

class BiLSTMCRF(tf.keras.Model):
  def __init__(self, words_count, labels_count, max_length):
    super(BiLSTMCRF, self).__init__()

    self.embedding = Embedding(words_count, 20, input_length=max_length, mask_zero=True)
    self.lstm = Bidirectional(LSTM(50, recurrent_dropout=0.1, return_sequences=True))
    self.dense = TimeDistributed(Dense(50, "relu"))
    self.crf = CRF(labels_count)

  def call(self, inputs):
    x = self.embedding(inputs)
    x = self.lstm(x)
    x = self.dense(x)
    x = self.crf(x)

    return x

  def compute_loss(self, x, y, training=False):
    y_pred = self(x, training=training)
    _, potentials, sequence_length, chain_kernel = y_pred

    crf_loss = -crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]

    return tf.reduce_mean(crf_loss), sum(self.losses)

  def train_step(self, data):
    x, y = data

    with tf.GradientTape() as tape:
        crf_loss, internal_losses = self.compute_loss(x, y, training=True)
        total_loss = crf_loss + internal_losses

    gradients = tape.gradient(total_loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    return {"crf_loss": crf_loss, "internal_losses": internal_losses}

  def test_step(self, data):
    x, y = data
    crf_loss, internal_losses = self.compute_loss(x, y)

    return {"crf_loss": crf_loss, "internal_losses": internal_losses}

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

    X_train, y_train = parse_sentences(word2idx, label2idx, labels_count, train_sentences, max_length, False)
    X_dev, y_dev = parse_sentences(word2idx, label2idx, labels_count, dev_sentences, max_length, False)
    X_test, y_test = parse_sentences(word2idx, label2idx, labels_count, test_sentences, max_length, False)

    model = BiLSTMCRF(words_count, labels_count, max_length)

    callback = EarlyStopping(monitor="val_crf_loss", patience=20, verbose=1)
    model.compile(optimizer="rmsprop")
    history = model.fit(X_train, y_train, validation_data=(X_dev, y_dev), callbacks=[callback], batch_size=32, epochs=500, verbose=1)
    history = pd.DataFrame(history.history)

    plt.clf()
    plt.plot(history["crf_loss"])
    plt.plot(history["val_crf_loss"])
    plt.title("Loss")
    plt.savefig("loss.png")

    predictions = model.predict(X_test, batch_size=32, verbose=1)[1]
    pred_labels = pred2label(predictions, label2idx)
    test_labels = pred2label(y_test, label2idx)

    print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
    print(classification_report(test_labels, pred_labels))