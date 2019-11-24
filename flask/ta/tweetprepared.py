import pandas as pd
import numpy as np
import re
import nltk
import pymorphy2
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

nltk.download('stopwords')

def tw_full_preprocess(text):
    text = text.lower()
    reg_rn = re.compile(r'\r\n')
    text = reg_rn.sub(' ', text)
    tokenizer = TweetTokenizer()
    morph = pymorphy2.MorphAnalyzer()
    stop_words = set(stopwords.words('russian'))
    text_list = []
    # text_ = str()
    for word in text.split():
        text_list.append(tokenizer.tokenize(word))

    string_dirty = ' '.join(str(v) for v in text_list)
    stringnew = string_dirty.replace('[', ' ')
    stringnew = stringnew.replace(']', ' ')
    stringnew = stringnew.replace('*', ' ')
    stringnew = stringnew.replace(':', ' ')
    stringnew = stringnew.replace(';', ' ')
    stringnew = stringnew.replace('!', ' ')
    stringnew = stringnew.replace('-', ' ')
    stringnew = stringnew.replace('.', ' ')
    stringnew = stringnew.replace('"', ' ')
    stringnew = stringnew.replace('?', ' ')
    stringnew = stringnew.replace(',', ' ')

    reg2 = re.compile(r'@\w+')
    stringnew_2 = reg2.sub('', stringnew)
    stringnew_2 = stringnew_2.replace('rt', ' ')

    reg_num = re.compile(r'[0-9]')
    stringnew_num = reg_num.sub(' ', stringnew_2)

    reg4 = re.compile(r'\squot')
    stringnew_4 = reg4.sub('', stringnew_num)

    reg5 = re.compile(r'\shttp?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),](?:%[0-9a-fA-F][0-9a-fA-F]))+')
    stringnew_5 = reg5.sub('', stringnew_4)

    reg6 = re.compile(r'[a-zA-Z]')
    stringnew_6 = reg6.sub(' ', stringnew_5)

    reg7 = re.compile(r'(ха|ах)+')
    stringnew_7 = reg7.sub(' ', stringnew_6)

    stringnew_7 = stringnew_7.replace("'", ' ')

    reg8 = re.compile('\s+')
    stringnew_8 = reg8.sub(' ', stringnew_7)
    for i in range(2):
        stringnew_8 = reg8.sub(' ', stringnew_8)

    finish_string = str()
    for word in stringnew_8.split():
        if word not in stop_words:
            if len(word) >= 3:
                finish_string = finish_string + ' ' + morph.parse(word)[0].normal_form

    reg9 = re.compile('\s+')
    return_string = reg9.sub(' ', finish_string)

    return str(return_string)