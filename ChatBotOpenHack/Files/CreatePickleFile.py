import pandas as pd
import numpy as np
import re
import pickle

utter_file = "utter.csv"
intent_file = 'intent.csv'
vocab_file = 'vocab.csv'
stopwords_file = 'stopwords.csv'
prerepdict_file = 'prepredict.csv'
postrepdict_file = 'postpredict.csv'
regex_file = 'regex.csv'

def create_array(utter_dict):
    keytrain_list = []
    Xtrain_list = []

    for key, value in utter_dict.items():
        keytrain_list.append(key)
        Xtrain_list.append(utter_dict[key])

        utter_array = np.array(Xtrain_list).astype(int)
        utterid_array = np.array(keytrain_list)

    return utter_array,utterid_array

def replace_words(text, replace_dict):
    for word, replacement in replace_dict.items():
        text = text.replace(word + ' ',replacement + ' ')
        if text.endswith(word):
            text = text.replace(word,replacement + ' ') # words at the end of the string need not be followed by space
    return text


def text_transform(text):
    text = text.lower()
    text = re.sub('\s+',' ', text)
    return text


def remove_num(text):
    text = re.sub('[^A-Za-z]+', ' ', text)
    return text


def rmv_stopwords(text, stop_words):
    words = text.split(' ')
    wordnew = [ word for word in words if word not in stop_words]
    return ' '.join(wordnew)


def clean_data(text, prerep_dict, postrep_dict, stop_words):
    text = remove_num(text)
    text = replace_words(text, prerep_dict)
    text = text_transform(text)
    text = replace_words(text, postrep_dict)
    text = rmv_stopwords(text, stop_words)
    return text

def create_utter_vectors(utterances, vocab_size, vocab_to_int, clean_data, prerep_dict, postrep_dict, stop_words):
    utter_dict = {}
    for idx, row in utterances.iterrows():
        layer_0 = [0] * vocab_size
        sent = row['utterance']
        sent = clean_data(sent, prerep_dict, postrep_dict, stop_words)
        # print(sent)
        for word in sent.split():
            if (word in vocab_to_int.keys()):
                layer_0[vocab_to_int[word]] = 1
        utter_dict[row['utter_id']] = layer_0

    return utter_dict

def vectorizer(utterances, vocab_size, vocab_to_int,  prerep_dict, postrep_dict, stop_words):
    utter_dict = create_utter_vectors(utterances, vocab_size, vocab_to_int, clean_data, prerep_dict,
                                      postrep_dict, stop_words)
    utter_array, utterid_array = create_array(utter_dict)
    print(utter_array)
    return utter_array, utterid_array


def replacedict(repdict_file):
    rep = pd.read_csv(repdict_file)
    rep_dict = {row['word']: row['replacement'] for idx, row in rep.iterrows()}
    return rep_dict


def stopwords(stopwords_file):
    stop_words = pd.read_csv(stopwords_file)
    stop_words = list(stop_words['word'])
    #print(stop_words)
    return stop_words


def vocab_lookups(vocab):
    vocab_to_int = {row['word']: row['vocab_id'] for idx, row in vocab.iterrows()}
    # int_to_vocab = {value:key for key,value in vocab_to_int.items()}
    vocab_size = len(vocab_to_int)
    #print(vocab_to_int,": ", vocab_size)
    return vocab_to_int, vocab_size

def load_files(utter_file, intent_file, vocab_file, regex_file):

    utterances = pd.read_csv(utter_file)
    intents = pd.read_csv(intent_file)
    vocab = pd.read_csv(vocab_file)
    regexs = pd.read_csv(regex_file)
    return utterances, intents, vocab, regexs




def loadingData(utter_file, intent_file, vocab_file, stopwords_file, prerepdict_file, postrepdict_file, regex_file):
    utterances, intents, vocab, regexs=load_files(utter_file, intent_file, vocab_file, regex_file)
    vocab_to_int, vocab_size = vocab_lookups(vocab)
    stop_words = stopwords(stopwords_file)
    prerep_dict = replacedict(prerepdict_file)
    postrep_dict = replacedict(postrepdict_file)
    return utterances, intents, vocab, vocab_to_int, vocab_size, stop_words, prerep_dict, postrep_dict, regexs


utterances,intents, vocab, vocab_to_int, vocab_size,stop_words,prerep_dict,postrep_dict, regexs=loadingData(utter_file, intent_file, vocab_file, stopwords_file, prerepdict_file, postrepdict_file, regex_file)
utter_array , utterid_array = vectorizer(utterances, vocab_size, vocab_to_int, prerep_dict, postrep_dict,
                                         stop_words)