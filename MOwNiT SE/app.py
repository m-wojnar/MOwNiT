# Maksymilian Wojnar

import flask
from flask import Flask, render_template

import numpy as np
import shelve

from scipy.sparse import csr_matrix, lil_matrix

from nltk.stem import SnowballStemmer
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords

from sklearn.preprocessing import normalize

RESULTS_LENGTH = 10

stoplist = set(stopwords.words('english'))
ss = SnowballStemmer('english', ignore_stopwords=True)

persistent_file_name = './shelve.out'
persistent_dict = shelve.open(persistent_file_name)

for key in persistent_dict:
    globals()[key] = persistent_dict[key]

persistent_dict.close()


def stem_text(text):
    return ' '.join(
        ss.stem(word) for word in regexp_tokenize(text, '[0-9a-zA-Z\']+') if word in all_words and word not in stoplist
    )


def get_cosines(stemmed_text, func):
    q = lil_matrix((1, m))

    for word in stemmed_text.split():
        q[0, all_words[word]] = 1

    q = normalize(csr_matrix(q), axis=1)

    cosines = np.zeros(n)
    for i in range(n):
        row = normalize(func(i), axis=0)
        cosines[i] = np.abs(q @ row)

    results = cosines.argsort()[::-1]
    return results, cosines[results]


def vsr(stemmed_text):
    get_row_func = lambda i: td.getcol(i).toarray()
    return get_cosines(stemmed_text, get_row_func)


def vsr_idf(stemmed_text):
    get_row_func = lambda i: td_ifd.getcol(i).toarray()
    return get_cosines(stemmed_text, get_row_func)


def lsi(stemmed_text):
    get_row_func = lambda i: (U_lsi @ VT_lsi[:, i]).reshape(-1, 1)
    return get_cosines(stemmed_text, get_row_func)


def edlsi(stemmed_text, x=0.2):
    get_row_func = lambda i: (U_edlsi @ VT_edlsi[:, i]).reshape(-1, 1) * x + td_ifd.getcol(i) * (1 - x)
    return get_cosines(stemmed_text, get_row_func)


def get_results(method, text):
    stemmed_text = stem_text(text)

    if stemmed_text in cache[method]:
        return cache[method][stemmed_text]

    if method == 'vsr':
        indexes, cosines = vsr(stemmed_text)
    elif method == 'vsr-idf':
        indexes, cosines = vsr_idf(stemmed_text)
    elif method == 'lsi':
        indexes, cosines = lsi(stemmed_text)
    else:
        indexes, cosines = edlsi(stemmed_text)

    results = [(all_links[indexes[i]], all_titles[indexes[i]], f'{cosines[i]:6f}') for i in range(RESULTS_LENGTH)]
    cache[method][stemmed_text] = results

    return results


app = Flask(__name__, static_folder='templates/static')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search/<method>/<text>')
def result(method, text):
    if method not in ['vsr', 'vsr-idf', 'lsi', 'edlsi']:
        method = 'edlsi'

    results = get_results(method, text)
    return render_template('result.html', results=results, text=text)


@app.route('/lucky/<method>/<text>')
def lucky(method, text):
    if method not in ['vsr', 'vsr-idf', 'lsi', 'edlsi']:
        method = 'edlsi'

    results = get_results(method, text)
    link = results[np.random.choice(len(results))][0].strip()
    return flask.redirect(link)


@app.route('/save')
def save():
    persistent_dict = shelve.open(persistent_file_name, 'n')

    for key in ['all_links', 'all_titles', 'all_words', 'td', 'td_ifd', 'U_lsi', 'VT_lsi', 'U_edlsi', 'VT_edlsi', 'm',
                'n', 'cache']:
        persistent_dict[key] = globals()[key]

    persistent_dict.close()
    return render_template('index.html')
