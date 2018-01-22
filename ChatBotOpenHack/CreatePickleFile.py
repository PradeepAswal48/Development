import pandas as pd
import numpy as np
import re
import pickle

utter_file = '/Files/utter.csv'
intent_file = '/Files/intents.csv'
vocab_file = '/Files/vocab.csv'
stopwords_file = '/Files/stop_words.csv'
prerepdict_file = '/Files/prerepdict.csv'
postrepdict_file = '/Files/postrepdict.csv'
regex_file = '/Files/regexs.csv'

def load_files(utter_file, intent_file, vocab_file, regex_file):

    utterances = pd.read_csv(utter_file)
    intents = pd.read_csv(intent_file)
    vocab = pd.read_csv(vocab_file)
    regexs = pd.read_csv(regex_file)
    return utterances, intents, vocab, regexs

print(load_files(utter_file, intent_file, vocab_file, regex_file))