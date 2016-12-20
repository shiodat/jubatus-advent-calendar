
import copy
import sys
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from jubatus.common import Datum
from embedded_jubatus import Classifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# Default Config
CONFIG = {
    'method': 'perceptron',
    'converter': {
        'num_filter_types': {},
        'num_filter_values': [],
        'string_filter_types': {},
        'string_filter_values': [],
        'num_types': {},
        'num_rules': [
            {'key': '*', 'type': 'num'}
        ],
        'string_types': {},
        'string_rules': [
            {'key': '*', 'type': 'space', 'sample_weight': 'log_tf', 'global_weight': 'bm25'}
        ]
    },
    'parameter': {}
}

def linear_classifier(method='AROW', regularization_weight=1.0):
    """ 線形分類器を起動する """
    cfg = copy.deepcopy(CONFIG)
    cfg['method'] = method
    if method not in ('perceptron', 'PA'):  # perceptron, PA 以外はパラメータが必要
        cfg['parameter']['regularization_weight'] = regularization_weight
    return Classifier(cfg)


def sklearn_linear_classifier(method='Perceptron(sk)'):
    sgd_params = {'penalty': 'l2', 'n_iter':1, 'shuffle': False, 'random_state': 42}
    pa_params = {'C': 1.0, 'n_iter': 1, 'shuffle': False, 'random_state': 42}
    if method == 'Perceptron(sk)':
        return Perceptron(n_iter=1, shuffle=False, random_state=42)
    elif method == 'LSVM(sk)':
        return SGDClassifier(loss='hinge', **sgd_params)
    elif method == 'LR(sk)':
        return SGDClassifier(loss='log', **sgd_params)
    elif method == 'PA1(sk)':
        return PassiveAggressiveClassifier(loss='hinge', **pa_params)
    elif method == 'PA2(sk)':
        return PassiveAggressiveClassifier(loss='squared_hinge', **pa_params)
    else:
        raise NotImprementedError()


def evaluate(X_train, X_test, y_train, y_test, n_trials=4):
    jubatus_methods = ['perceptron', 'PA', 'PA1', 'PA2', 'CW', 'AROW', 'NHERD']
    sklearn_methods = ['Perceptron(sk)', 'PA1(sk)', 'PA2(sk)', 'LSVM(sk)', 'LR(sk)']
    results = dict.fromkeys(jubatus_methods + sklearn_methods, 0)
    vectorizer = TfidfVectorizer()
    for i in range(n_trials):
        X_train, y_train = shuffle(X_train, y_train, random_state=42) 
        for method in jubatus_methods:
            clf = linear_classifier(method=method)
            clf.fit(X_train, y_train)
            test_score = accuracy_score(y_test, clf.predict(X_test))
            print('{0:.3f}\t{1}'.format(test_score, method))
            results[method] += test_score
        for method in sklearn_methods:
            clf = sklearn_linear_classifier(method=method)
            clf.fit(X_train, y_train)
            test_score = accuracy_score(y_test, clf.predict(X_test))
            print('{0:.3f}\t{1}'.format(test_score, method))
            results[method] += test_score
    results = {k: v / n_trials for k, v in results.items()}
    return results


def load_dataset(key='mnist'):
    if key == 'mnist':
        from sklearn.datasets import fetch_mldata
        dataset = fetch_mldata('MNIST original')
    elif key == 'cancer':
        from sklearn.datasets import load_breast_cancer
        dataset = load_breast_cancer()
    elif key == 'digits':
        from sklearn.datasets import load_digits
        dataset = load_digits()
    else:
        from sklearn.datasets import load_iris
        dataset = load_iris()
    X = csr_matrix(dataset.data, dtype=np.float64)
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    key = sys.argv[1]
    X_train, X_test, y_train, y_test = load_dataset(key)
    results = evaluate(X_train, X_test, y_train, y_test)
    print('{}\n{}\n{}'.format('-'*60, 'Experimental Result', '-'*60))
    for method, test_score in results.items():
        print('{0:.3f}\t{1}'.format(test_score, method))
    
