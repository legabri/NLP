import json
import spacy
import numpy as np

from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

from nltk.corpus import sentiwordnet

from negation_conversion import set_custom_boundaries, convert_negated_words

reviews_dataset = {
    'train_pos_fn' : "./data/train_positive.txt",
    'train_neg_fn' : "./data/train_negative.txt",
    'test_pos_fn' : "./data/test_positive.txt",
    'test_neg_fn' : "./data/test_negative.txt"
}

def tokenize(review):
    document = nlp(review)
    tokens = [token.lemma_.lower().strip() if token.lemma_ != '-PRON-' else token.lower_ for token in document if not token.is_stop and not token.is_punct]
    return tokens

def analyse_sentiment_jurafsky(features, word, pos):
    synset = list(sentiwordnet.senti_synsets(word, pos))
    if synset:
        argscore = np.argmax(np.asarray([synset[0].pos_score(), synset[0].neg_score(), synset[0].obj_score()]))
        if argscore == 0:
            features[0] += 1
        elif argscore == 1:
            features[1] += 1

def analyse_sentiment_ohana(features, word, pos):
    synset = list(sentiwordnet.senti_synsets(word, pos))
    if synset:
        features[9] += synset[0].pos_score()
        features[10] += synset[0].neg_score()

def jurafsky(review):
    features = np.zeros(6)
    pronouns = ['i', 'me', 'my', 'mine', 'myself','you', 'your', 'yours', 'yourself', 'we', 'us', 'our', 'ours', 'ourselves', 'yourselves']
    document = nlp(review)
    for token in document:
        if token.is_punct:
            if token.text == '!':
                features[4] += 1
        else:
            if 'NN' in token.tag_:
                analyse_sentiment_jurafsky(features, token.text, 'n')
            elif 'VB' in token.tag_:
                analyse_sentiment_jurafsky(features, token.text, 'v')
            elif 'JJ' in token.tag_:
                analyse_sentiment_jurafsky(features, token.text, 'a')
            elif 'RB' in token.tag_:
                analyse_sentiment_jurafsky(features, token.text, 'r')
            if token.dep_ == 'neg':
                features[2] = 1
            if token.lemma_ == '-PRON-' and token.text in pronouns:
                features[3] += 1
            features[5] += 1
    features[5] = np.log(features[5])
    return features

def ohana(review):
    features = np.zeros(12)
    document = nlp(review)
    for token in document:
        if not token.is_punct:
            if token.pos_ == 'NOUN':
                features[0] += 1
            elif token.pos_ == 'PROPN':
                features[1] += 1
            elif token.pos_ == 'ADJ':
                features[2] += 1
            elif token.pos_ == 'VERB':
                features[3] += 1
            elif token.pos_ == 'ADV':
                features[4] += 1
            elif token.pos_ == 'INTJ':
                features[5] += 1
            if token.pos_ == 'INTJ' or token.pos_ == 'DET' or token.pos_ == 'ADP' or token.pos_ == 'CCONJ' or token.pos_ == 'PRON' or token.tag_ == 'MD':
                features[8] += 1
            if 'NN' in token.tag_:
                analyse_sentiment_ohana(features, token.text, 'n')
            elif 'VB' in token.tag_:
                analyse_sentiment_ohana(features, token.text, 'v')
            elif 'JJ' in token.tag_:
                analyse_sentiment_ohana(features, token.text, 'a')
            elif 'RB' in token.tag_:
                analyse_sentiment_ohana(features, token.text, 'r')
    for sent in document.sents:
        features[6] += 1
        features[7] += len([token for token in nlp(sent.text) if not token.is_punct])
    features[7] /= features[6]
    if features[9] != 0.0 and features[10] != 0.0:
        features[11] = features[9] / features[10]
    return features

def train_and_test_classifier(dataset, model='NB', features='words'):
    """
    :param dataset: les 4 fichiers utilisées pour entraîner et tester les classificateurs. Voir reviews_dataset.
    :param model: le type de classificateur. NB = Naive Bayes, LG = Régression logistique, NN = réseau de neurones
    :param features: le type d'attributs (features) que votre programme doit construire
                 - 'jurafsky': les 6 attributs proposés dans le livre de Jurafsky et Martin.
                 - 'ohana': les 12 attributs représentant le style de rédaction (Ohana et al.)
                 - 'combined': tous les attributs 'jurafsky' et 'ohaha'
                 - 'words': des vecteurs de mots
                 - 'negated_words': des vecteur de mots avec conversion des mots dans la portée d'une négation
    :return: un dictionnaire contenant 3 valeurs:
                 - l'accuracy à l'entraînement (validation croisée)
                 - l'accuracy sur le jeu de test
                 - la matrice de confusion obtenu de scikit-learn
    """
    if model == 'NB':
        classifier = BernoulliNB(alpha=1.0)
    elif model == 'LG':
        classifier = LogisticRegression(penalty='l2', C=100, solver='saga')
    elif model == 'NN':
        classifier = MLPClassifier(hidden_layer_sizes=(100, 1), activation='logistic', solver='adam', alpha=0.5, max_iter=300)
    else:
        raise ValueError("Classificateur invalide.")

    global nlp
    nlp = spacy.load('en_core_web_sm')

    train_pos = load_reviews(dataset['train_pos_fn'])
    train_neg = load_reviews(dataset['train_neg_fn'])
    test_pos = load_reviews(dataset['test_pos_fn'])
    test_neg = load_reviews(dataset['test_neg_fn'])

    if features == 'jurafsky':
        X_train = [jurafsky(review) for review in train_pos + train_neg]
        X_test = [jurafsky(review) for review in test_pos + test_neg]
    elif features == 'ohana':
        X_train = [ohana(review) for review in train_pos + train_neg]
        X_test = [ohana(review) for review in test_pos + test_neg]
    elif features == 'combined':
        X_train = [np.concatenate((jurafsky(review), ohana(review))) for review in train_pos + train_neg]
        X_test = [np.concatenate((jurafsky(review), ohana(review))) for review in test_pos + test_neg]
    elif features == 'words':
        X_train = train_pos + train_neg
        X_test = test_pos + test_neg
    elif features == 'negated_words':
        nlp.add_pipe(set_custom_boundaries, name='set_custom_boundaries', before='parser')
        X_train = [convert_negated_words(review) for review in train_pos + train_neg]
        X_test = [convert_negated_words(review) for review in test_pos + test_neg]
        nlp.remove_pipe('set_custom_boundaries')
    else:
        raise ValueError("Type d'attributs invalide.")

    y_train = ['positive' for i in range(len(train_pos))] + ['negative' for i in range(len(train_neg))]
    y_test = ['positive' for i in range(len(test_pos))] + ['negative' for i in range(len(test_neg))]

    if features == 'words' or features == 'negated_words':
        pipeline = Pipeline([('vectorizer', CountVectorizer(tokenizer = tokenize)), ('classifier', classifier)]).fit(X_train, y_train)
    else:
        pipeline = Pipeline([('classifier', classifier)]).fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    results = {}
    results['accuracy_train'] = np.mean(cross_val_score(pipeline, X_train, y_train, cv=5))
    results['accuracy_test'] = accuracy_score(y_test, y_pred)
    results['confusion_matrix'] = confusion_matrix(y_test, y_pred)

    return results

def load_reviews(filename):
    with open(filename, 'r') as fp:
        reviews_list = json.load(fp)
    return reviews_list

if __name__ == '__main__':
    results = train_and_test_classifier(reviews_dataset, model='NB', features='negated_words')
    print("Accuracy - entraînement: ", results['accuracy_train'])
    print("Accuracy - test: ", results['accuracy_test'])
    print("Matrice de confusion: \n", results['confusion_matrix'])
