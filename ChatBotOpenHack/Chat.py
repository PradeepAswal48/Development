import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import pickle
import json

threshold = 60  # Confidence threshold
unknown_intent = 404
with open('Files/chat_pkl.pkl', 'rb') as f:
    (utter_array, utterid_array, utterances, intents, vocab, vocab_to_int, vocab_size,
     stop_words, prerep_dict, postrep_dict, regexs) = pickle.load(f)


def replace_words(text, replace_dict):
    for word, replacement in replace_dict.items():
        text = text.replace(word + ' ',replacement + ' ')
    return text


def remove_numchar(text, regexs, context_id):
    if context_id != 0:
        text = replace_entity(text, regexs, context_id)

    text = re.sub('[^A-Za-z]+', ' ', text)
    return text

def text_transform(text):
    text = text.lower()
    text = re.sub('\s+',' ', text)
    return text


def clean_data(text, prerep_dict, postrep_dict, stop_words, regexs, context_id):
    text = replace_words(text, prerep_dict)
    # text = remove_numchar(text, regexs, context_id)
    text = text_transform(text)
    text = replace_words(text, postrep_dict)
    text = rmv_stopwords(text, stop_words)
    return text

def rmv_stopwords(text, stop_words):
    words = text.split(' ')
    wordnew = [ word for word in words if word not in stop_words]
    return ' '.join(wordnew)

def create_test_utter_vectors(input_text, vocab_size, vocab_to_int, prerep_dict, postrep_dict, stop_words, regexs, context_id):
    utter_test_id = 0
    unk_count = 0
    utter_dict_test = {}
    layer_0 = [0] * vocab_size
    # sent = test_utter
    # print(test_utter)
    sent = clean_data(input_text, prerep_dict, postrep_dict, stop_words, regexs, context_id)
    # sent = (input_text)
    # print("sent is",sent)
    for word in sent.split():
        # print("for word: ",word)
        if (word in vocab_to_int.keys()):
            layer_0[vocab_to_int[word]] = 1
            # print(layer_0)
    utter_dict_test[utter_test_id] = layer_0
    # print(utter_dict_test)
    return utter_dict_test


def create_array(utter_dict):
    keytrain_list = []
    Xtrain_list = []

    for key, value in utter_dict.items():
        keytrain_list.append(key)
        Xtrain_list.append(utter_dict[key])
        utter_array_test = np.array(Xtrain_list).astype(int)
        utter_array_test_id = np.array(keytrain_list)
    print(utter_array_test,utter_array_test_id)
    return utter_array_test,utter_array_test_id

def find_unknowns(text, vocab_to_int, prerep_dict, postrep_dict, stop_words, regexs, context_id):

    unk_count=0
    #text = clean_data(text, prerep_dict, postrep_dict, stop_words, regexs, context_id)
    for word in text.split():
        if (word not in vocab_to_int.keys()):
            unk_count += 1
    print("unknows num are: ",unk_count)
    return unk_count

def create_cosine_similarity_table(utter_array_test, utter_array):
    # print("am here")
    # print("utter array is : ", utter_array)
    # print("utter_array_test is", utter_array_test)
    # for utrep in utter_array:
    #     print(utrep.reshape(1, -1))[0][0])
    sim = [cosine_similarity(utter_array_test.reshape(1,-1), utrep.reshape(1,-1))[0][0] for utrep in utter_array]
    print(sim)
    return sim

def vectorizer(input_text, vocab_size, vocab_to_int, prerep_dict, postrep_dict, stop_words, regexs, context_id):
    print("in vectorizer the inout is: ", input_text)
    utter_dict_test = create_test_utter_vectors(input_text, vocab_size, vocab_to_int, prerep_dict, postrep_dict,
                                                stop_words,
                                                regexs, context_id)
    utter_array_test, _ = create_array(utter_dict_test)
    # print(utter_array_test)
    return utter_array_test

def intents_best_match(sim, utterances, intents, unk_count, context_id):
    utterances['similarity'] = np.asarray(sim) * 100
    print(utterances)
    print(context_id)
    intents_subset = intents[intents['parent_id'] == context_id]
    print(intents_subset)
    intents_subset = list(intents_subset['intent_id'])
    print(intents_subset)
    utterances_subset = utterances[utterances['intent_id'].isin(intents_subset)]
    print(utterances_subset)
    utt_maxbyintent = utterances_subset.groupby('intent_id')['similarity'].max()
    for key, row in intents.iterrows():
        intents.loc[key, 'similarity'] = utt_maxbyintent.get(row['intent_id'], 0)
        print(intents.loc[key,'similarity'])

    intent_max = intents['similarity'].max()
    print("intent_max is : ",intent_max)
    # print(unk_count)
    confidence = intent_max * (0.5 ** unk_count)
    intent_id_best = intents['intent_id'][intents['similarity'].idxmax()] if confidence > threshold else unknown_intent
    response = intents['Response'][intents['intent_id'] == intent_id_best].values[0]
    # print(intent_id_best, confidence, context_id)
    return intent_id_best, confidence, context_id, response

def pred(input_text, vocab_size, vocab_to_int,utter_array, intents, context_id, from_orc, input_code, regexs):
    utter_array_test = vectorizer(input_text, vocab_size, vocab_to_int, prerep_dict, postrep_dict, stop_words, regexs,
                                  context_id)
    similarity=create_cosine_similarity_table(utter_array_test, utter_array)
    unk_count = find_unknowns(input_text, vocab_to_int, prerep_dict, postrep_dict, stop_words, regexs, context_id)
    intent_id_best, confidence, context_id,response = intents_best_match(similarity, utterances, intents, unk_count, context_id)
    return response, confidence, intent_id_best, context_id
    # response, output_code, intent_id_best, context_id, to_orc = response_best_match(intent_id_best, intents, context_id,
    #                                                                                 from_orc, input_code, input_text)

def chat(input_text, context_id, input_code, from_orc):
    # print("utter array", utter_array)
    # print("utter_array_id", utterid_array)
    # print("utterance", utterances)
    # print("intents", intents)
    # print("vocab", vocab)
    # print("vocab_to_int", vocab_to_int)
    # print("vocab size", vocab_size)
    # print("stopwords", stop_words)
    # print("prerep_dict", prerep_dict)
    # print("postrep_dict", postrep_dict)
    # print("regexs", regexs)
    # print("input text is ",input_text)
    response, confidence, intent_id_best, context_id,  = pred(input_text, vocab_size, vocab_to_int,utter_array, intents, context_id, from_orc,input_code, regexs)
    print(response)

chat("how are you ?",1,"O",0)