import numpy as np

def pred2label(predictions, label2idx):
    idx2label = {index: word for word, index in label2idx.items()}
    labels = [[idx2label[np.argmax(output)].replace("PAD", "O") for output in prediction] for prediction in predictions]
    return labels