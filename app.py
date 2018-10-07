import os
import time

import requests
from flask import Flask, render_template, request, url_for
from flask_cors import CORS, cross_origin
from flask_jsonpify import jsonify, jsonpify

import pandas as pd
import numpy as np
import math as math
import nltk
from nltk.tag import tnt
import pickle

app = Flask(__name__)
CORS(app)


# tnt_pos_tagger = tnt.TnT()

@app.route('/', methods=['POST'])
def index():
    param = request.get_json()
    sentence = param.get('inputText')
    # nltk.sys.setrecursionlimit(3000)
    errors = []
    results = {}
    start = ''
    end = ''
    if not os.path.isfile('tnt_pos_tagger.pickle'):
        tnt_pos_tagger = train_pos_tagger(tnt.TnT())
    else:
        tnt_pos_tagger = pd.read_pickle('tnt_pos_tagger.pickle')

    nepali_words = pd.read_excel('nepali word.xlsx')
    nepali_words = nepali_words.dropna()
    suffixes = nepali_words.SUFFIX

    if request.method == "POST":
        start = time.ctime(time.time())

        tokenized_words = tokenize(sentence)
        tokenized_words = remove_attached_symbols(tokenized_words)
        tokenized_words = remove_stop_word(tokenized_words)
        tagged_words = tag_words(tokenized_words, tnt_pos_tagger)
        unk_tagged_data = get_unk_data(tagged_words)

        if unk_tagged_data is 0:
            return jsonpify(tagged_words.to_json())

        stemmed_data = stem_words(unk_tagged_data, suffixes)
        final_data = get_final_result(tagged_words, stemmed_data)

        end = time.ctime(time.time())

        results.update({'result': final_data})
        results = results['result'].to_html(classes="mat-elevation-z8 striped result-table")

        final_result = {"pyResult": results, "startTime": start, "endTime": end}
    return jsonify(final_result)
    # return render_template('index.html', errors=errors, results=results, input_sentence=sentence, start_time=start, end_time=end)


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def remove_attached_symbols(tokenized_words):
    symbols = ['।', '?', ',', '\'', '"', '<', '>', ']', '[', ')', '(', '‘', '’', '!', '@', '\“', '-']

    for index, token_word in enumerate(tokenized_words):
        if token_word in symbols:
            continue
        for symbol in symbols:
            if symbol == token_word[len(token_word) - 1]:
                s = token_word.replace(symbol, '')
                tokenized_words.remove(token_word)
                tokenized_words.insert(index, s)

    return tokenized_words


def remove_stop_word(tokenized_words):
    stop_words = pd.read_csv('nepali_stopwords.txt', header=None, names=['Stop Word'])

    for tokenized_word in tokenized_words:
        for stop_word in stop_words.iterrows():
            if stop_word[1][0] == tokenized_word:
                tokenized_words.remove(tokenized_word)
                break

    return tokenized_words


def tag_words(tokenized_words, tnt_pos_tagger):
    results = tnt_pos_tagger.tag(tokenized_words)
    df = pd.DataFrame(results)

    tagged_words = []
    for i, result in df.iterrows():
        tagged_words.append((result[0], result[1]))

    tagged_words = pd.DataFrame(tagged_words, columns=['Word', 'POS_Tag'])
    return tagged_words


def get_unk_data(tagged_words):
    unk_data = tagged_words[tagged_words['POS_Tag'].isin(['Unk'])]

    if len(unk_data) is not 0:
        unk_data['Word'].values[0]
        unk_data['Root'] = math.nan
        unk_data['Suffix'] = math.nan

        return unk_data
    return 0


def stem_words(unk_tagged_data, suffixes):
    for word in unk_tagged_data.iterrows():
        index = word[0]
        word = word[1]['Word']
        word_length = len(word)
        suffixCount = 0
        suff = ""
        root = ""

        for key in range(word_length - 1, -1, -1):

            for suffix in suffixes:
                if suffix == word[key:]:
                    suff = suffix
                    root = word[:key]
                    suffixCount += 1
                    break;
        if suffixCount > 0:
            unk_tagged_data.loc[index, 'Root'] = root
            unk_tagged_data.loc[index, 'Suffix'] = suff
        if suffixCount is 0:
            unk_tagged_data.loc[index, 'Root'] = word

    return unk_tagged_data


def get_final_result(tagged_words, stemmed_data):
    result = tagged_words.append(stemmed_data)
    result = result.reset_index()

    result = result.drop_duplicates(subset=['index'], keep='last')
    result = result.sort_values('index')
    result.set_index('index', inplace=True)
    del result.index.name

    return result


def train_pos_tagger(tnt_pos_tagger):
    data = pd.read_csv('NCorpus.txt', sep=" ", header=None, names=['Word', 'POS _ag'])

    x = []
    temp = []
    for i, datum in data.iterrows():
        temp.append((datum[0], datum[1]))
        if i % 5 == 0:
            x.append(temp)
            temp = []

    test_data = x[:10000]
    train_data = x[10000:]
    tnt_pos_tagger.train(train_data)
    tnt_pos_tagger.evaluate(test_data)

    f = open('tnt_pos_tagger.pickle', 'wb')
    pickle.dump(tnt_pos_tagger, f)

    return tnt_pos_tagger


if __name__ == '__main__':
    app.run(port=5001)
