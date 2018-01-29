import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import pickle
import json
# import spellchecker as spcor


with open('Files/chat_pkl.pkl', 'rb') as f:
              (utter_array,utterid_array,utterances,intents,vocab,vocab_to_int,vocab_size,
                  stop_words, prerep_dict, postrep_dict, regexs) = pickle.load(f)

def vectorizer(test_utter, vocab_size, vocab_to_int, prerep_dict, postrep_dict, stop_words, regexs, context_id):
    print("Vectorizer logic starts")

def pred(input_text, vocab_size, vocab_to_int, utter_array, intents, context_id, from_orc, input_code, regexs, user_resp):
    print("Prediction logic Starts")
    utter_array_test = vectorizer(test_utter, vocab_size, vocab_to_int, prerep_dict, postrep_dict, stop_words, regexs,
                                  context_id)

def chat(input_text, context_id):
    response, confidence, output_code, intent_id_best, context_id, to_orc = pred(input_text, vocab_size, vocab_to_int,
                                                                                 utter_array, intents, context_id,
                                                                                 from_orc,
                                                                                 input_code, regexs, user_resp)
