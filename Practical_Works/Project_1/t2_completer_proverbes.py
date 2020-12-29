import json
from collections import Counter, defaultdict
from math import log
from statistics import mean
from nltk import word_tokenize
from nltk.lm.models import Laplace
from nltk.lm.preprocessing import pad_both_ends, flatten
from nltk.util import ngrams

proverbs_fn = "./data/proverbes.txt"
test1_fn = "./data/test_proverbes1.txt"

test_solutions = ["a beau mentir qui vient de loin", "a beau mentir qui vient de loin",
                  "l’occasion fait le larron", "aide-toi, le ciel t’aidera",
                  "année de gelée, année de blé", "après la pluie, le beau temps",
                  "aux échecs, les fous sont les plus près des rois", "ce que femme veut, dieu le veut",
                  "bien mal acquis ne profite jamais", "bon ouvrier ne querelle pas ses outils",
                  "ce n’est pas tous les jours pâques", "pour le fou, c’est tous les jours fête",
                  "dire et faire, sont deux", "mieux vaut tard que jamais",
                  "d’un sac on ne peut tirer deux moutures", "à qui dieu aide, nul ne peut nuire",
                  "il n’y a pas de rose de cent jours", "il faut le voir pour le croire",
                  "on ne vend pas le poisson qui est encore dans la mer",
                  "la langue d’un muet vaut mieux que celle d’un menteur"]

class Bigrams():
    def __init__(self):
        self.vocabulary = Counter()
        self.counts = defaultdict(lambda: defaultdict(int))

    def fit(self, text, vocabulary):
        self.vocabulary.update(vocabulary)
        for sentence in text:
            for bigram in sentence:
                parsed_bigram = [word if word in self.vocabulary else 'UNK' for word in bigram]
                self.counts[parsed_bigram[0]][parsed_bigram[1]] += 1

    def logscore(self, word, context):
        word = word if word in self.vocabulary else 'UNK'
        context = context if context in self.vocabulary else 'UNK'
        counts = self.counts[context]
        return log((counts[word] + 1) / (sum(counts.values()) + len(self.vocabulary)))

    def perplexity(self, bigrams):
        return (mean([self.logscore(bigram[1], bigram[0]) for bigram in bigrams])) ** 2

def load_proverbs(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    return [x.strip() for x in raw_lines]

def load_tests(filename):
    with open(filename, 'r', encoding='utf-8') as fp:
        test_data = json.load(fp)
    return test_data

def train_models(filename):
    """ Vous ajoutez à partir d'ici tout le code dont vous avez besoin
        pour construire les différents modèles N-grammes.
        Voir les consignes de l'énoncé du travail pratique concernant les modèles à entraîner.

        Vous pouvez ajouter au fichier toutes les fonctions, classes, méthodes et variables que vous jugerez nécessaire.
        Merci de ne pas modifier les signatures (noms de fonctions et arguments) déjà présentes dans le fichier.
    """
    proverbs = load_proverbs(filename)
    print("\nNombre de proverbes : ", len(proverbs))

    tokens = [word_tokenize(sentence) for sentence in proverbs]

    global models
    models = {1: Laplace(1), 2: Laplace(2), 3: Laplace(3), 20: Bigrams()}

    models[1].fit([list(ngrams(pad_both_ends(sentence, 1), 1)) for sentence in tokens], set(flatten(pad_both_ends(sentence, 1) for sentence in tokens)))
    models[2].fit([list(ngrams(pad_both_ends(sentence, 2), 2)) for sentence in tokens], set(flatten(pad_both_ends(sentence, 2) for sentence in tokens)))
    models[3].fit([list(ngrams(pad_both_ends(sentence, 3), 3)) for sentence in tokens], set(flatten(pad_both_ends(sentence, 3) for sentence in tokens)))
    models[20].fit([list(ngrams(pad_both_ends(sentence, 2), 2)) for sentence in tokens], set(flatten(pad_both_ends(sentence, 2) for sentence in tokens)))

def cloze_test(incomplete_proverb, choices, n=3):
    """ Fonction qui complète un texte à trous en ajoutant le(s) bon(s) mot(s).
        En anglais, on nomme ce type de tâche un cloze test.

        La paramètre n désigne le modèle utilisé.
        1 - unigramme NLTK, 2 - bigramme NLTK, 3 - trigramme NLTK, 20 - votre modèle bigramme
    """
    results = []

    for choice in choices:
        split = incomplete_proverb.split()
        index = [character for character, token in enumerate(split) if '***' in token][0]
        proverb = incomplete_proverb.replace('***', choice)

        tokens = word_tokenize(proverb)
        order = 2 if n == 20 else n
        multigrams = list(ngrams(pad_both_ends(tokens, order), order))

        if n == 1:
            logscore = models[n].logscore(choice)
            choice_perplexity = models[n].perplexity(multigrams)
        elif n == 2:
            logscore = models[n].logscore(choice, [split[index - 1]])
            choice_perplexity = models[n].perplexity(multigrams)
        elif n == 3:
            logscore = models[n].logscore(choice, [split[index - 2], split[index - 1]])
            choice_perplexity = models[n].perplexity(multigrams)
        elif n == 20:
            logscore = models[n].logscore(choice, split[index - 1])
            choice_perplexity = models[n].perplexity(multigrams)

        results.append((proverb, multigrams, logscore)) # choice_perplexity

    results.sort(key=lambda x: x[2], reverse=True) # False
    result = results[0][0]
    perplexity = models[n].perplexity(results[0][1])

    return result, perplexity

if __name__ == '__main__':
    test_proverbs = load_tests(test1_fn)
    print("\nNombre de tests du fichier {}: {}\n".format(test1_fn, len(test_proverbs)))

    train_models(proverbs_fn)
    print("Les résultats des tests sont:")
    score = 0

    for i, item in enumerate(test_proverbs.items()):
        partial_proverb, options = item
        solution, perplexity = cloze_test(partial_proverb, options, n=20)
        score += 1 if solution == test_solutions[i] else 0
        print("\n\tProverbe incomplet: {} , Options: {}".format(partial_proverb, options))
        print("\tSolution = {} , Perplexité = {}".format(solution, perplexity))

    print("Score: {0}".format(score / len(test_solutions)))
