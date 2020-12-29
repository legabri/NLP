from __future__ import unicode_literals
import glob
import os
import string
import unicodedata
import json

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from nltk import word_tokenize
from nltk.lm.preprocessing import pad_both_ends
from nltk.util import ngrams

datafiles = "./data/names/*.txt"
test_filename = './data/test-names-t3.txt'

names_by_origin = {}
all_origins = []

def load_names():
    """Lecture des noms et langues d'origine d'un fichier. Par la suite,
       sauvegarde des noms pour chaque origine dans le dictionnaire names_by_origin."""
    for filename in find_files(datafiles):
        origin = get_origin_from_filename(filename)
        all_origins.append(origin)
        names = read_names(filename)
        names_by_origin[origin] = names

def find_files(path):
    """Retourne le nom des fichiers contenus dans un répertoire.
       glob fait le matching du nom de fichier avec un pattern - par ex. *.txt"""
    return glob.glob(path)

def get_origin_from_filename(filename):
    """Passe-passe qui retourne la langue d'origine d'un nom de fichier.
       Par ex. cette fonction retourne Arabic pour "./data/names/Arabic.txt". """
    return os.path.splitext(os.path.basename(filename))[0]

def read_names(filename):
    """Retourne une liste de tous les noms contenus dans un fichier."""
    with open(filename, encoding='utf-8') as f:
        names = f.read().strip().split('\n')
    return [unicode_to_ascii(name) for name in names]

def unicode_to_ascii(s):
    """Convertion des caractères spéciaux en ascii. Par exemple, Hélène devient Helene.
       Tiré d'un exemple de Pytorch. """
    all_letters = string.ascii_letters + " .,;'"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def load_test_names(filename):
    """Retourne un dictionnaire contenant les données à utiliser pour évaluer vos modèles.
       Le dictionnaire contient une liste de noms (valeurs) et leur origine (clé)."""
    with open(filename, 'r') as fp:
        test_data = json.load(fp)
    return test_data

def train_classifiers():
    load_names()
    X = [unicode_to_ascii(name.lower().strip()) for names in list(names_by_origin.values()) for name in names]
    y = [label for labels in [[origin] * len(names_by_origin[origin]) for origin in all_origins] for label in labels]

    global models
    models = {
        'naive_bayes-1-tf': Pipeline([('tf', CountVectorizer(ngram_range=(1, 1), analyzer='char')), ('naive_bayes', MultinomialNB())]).fit(X, y),
        'naive_bayes-1-tfidf': Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 1), analyzer='char')), ('naive_bayes', MultinomialNB())]).fit(X, y),
        'naive_bayes-2-tf': Pipeline([('tf', CountVectorizer(ngram_range=(2, 2), analyzer='char')), ('naive_bayes', MultinomialNB())]).fit(X, y),
        'naive_bayes-2-tfidf': Pipeline([('tfidf', TfidfVectorizer(ngram_range=(2, 2), analyzer='char')), ('naive_bayes', MultinomialNB())]).fit(X, y),
        'naive_bayes-3-tf': Pipeline([('tf', CountVectorizer(ngram_range=(3, 3), analyzer='char')), ('naive_bayes', MultinomialNB())]).fit(X, y),
        'naive_bayes-3-tfidf': Pipeline([('tfidf', TfidfVectorizer(ngram_range=(3, 3), analyzer='char')), ('naive_bayes', MultinomialNB())]).fit(X, y),
        'naive_bayes-multi-tf': Pipeline([('tf', CountVectorizer(ngram_range=(1, 3), analyzer='char')), ('naive_bayes', MultinomialNB())]).fit(X, y),
        'naive_bayes-multi-tfidf': Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 3), analyzer='char')), ('naive_bayes', MultinomialNB())]).fit(X, y),
        'logistic_regression-1-tf': Pipeline([('tf', CountVectorizer(ngram_range=(1, 1), analyzer='char')), ('logistic_regression', LogisticRegression(max_iter=400))]).fit(X, y),
        'logistic_regression-1-tfidf': Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 1), analyzer='char')), ('logistic_regression', LogisticRegression(max_iter=400))]).fit(X, y),
        'logistic_regression-2-tf': Pipeline([('tf', CountVectorizer(ngram_range=(2, 2), analyzer='char')), ('logistic_regression', LogisticRegression(max_iter=400))]).fit(X, y),
        'logistic_regression-2-tfidf': Pipeline([('tfidf', TfidfVectorizer(ngram_range=(2, 2), analyzer='char')), ('logistic_regression', LogisticRegression(max_iter=400))]).fit(X, y),
        'logistic_regression-3-tf': Pipeline([('tf', CountVectorizer(ngram_range=(3, 3), analyzer='char')), ('logistic_regression', LogisticRegression(max_iter=400))]).fit(X, y),
        'logistic_regression-3-tfidf': Pipeline([('tfidf', TfidfVectorizer(ngram_range=(3, 3), analyzer='char')), ('logistic_regression', LogisticRegression(max_iter=400))]).fit(X, y),
        'logistic_regression-multi-tf': Pipeline([('tf', CountVectorizer(ngram_range=(1, 3), analyzer='char')), ('logistic_regression', LogisticRegression(max_iter=400))]).fit(X, y),
        'logistic_regression-multi-tfidf': Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 3), analyzer='char')), ('logistic_regression', LogisticRegression(max_iter=400))]).fit(X, y)
        }

def get_classifier(type, n=3, weight='tf'):
    """Retourne le classifieur selon les paramètres:
       - type = 'naive_bayes', 'logistic_regression'
       - n = 1, 2, 3, 'multi'
       - weight = 'tf', 'tfidf'"""
    classifier = models[type + '-' + str(n) + '-' + weight]
    return classifier

def origin(name, type, n=3, weight='tf'):
    name_origin = get_classifier(type, n, weight).predict([unicode_to_ascii(name.lower().strip())])
    return name_origin

def test_classifier(test_fn, type, n=3, weight='tf'):
    test_data = load_test_names(test_fn)
    scores = [1 if origin(name, type, n, weight) == label else 0 for label in all_origins for name in test_data[label]]
    test_accuracy = sum(scores) / len(scores)
    return test_accuracy

def evaluate_classifiers(filename):
    for key in models:
        parameters = key.split("-")
        scores = [1 if origin(name, parameters[0], parameters[1], parameters[2]) == label else 0 for label in all_origins for name in names_by_origin[label]]
        test_accuracy = sum(scores) / len(scores)
        print("Précision: {0} Modèle: {1}".format(test_accuracy, key))

if __name__ == '__main__':
    train_classifiers()
    print("Les {} langues d'origine sont: \n{}".format(len(all_origins), all_origins))

    print("\nRésultats en test:")
    for key in models:
        parameters = key.split("-")
        test_accuracy = test_classifier(test_filename, type=parameters[0], n=parameters[1], weight=parameters[2])
        print("Précision: {0} Modèle: {1}".format(test_accuracy, key))

    print("\nRésultats en entraînement:")
    evaluate_classifiers(datafiles)
