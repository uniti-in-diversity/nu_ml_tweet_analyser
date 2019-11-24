import flask
from flask import render_template
import pandas as pd
import numpy as np
import re
import nltk
import pymorphy2
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])

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

model_mnb = pickle.load(open('/app/model_mnb_r2_1.pkl','rb'))
vocabulary = pickle.load(open('/app/vocabulary_r2_1.pkl','rb'))


def main():
    if flask.request.method == 'GET':
        return render_template('main.html' )
        
    if flask.request.method == 'POST':
        #answer = ''
        exp_s = flask.request.form['tweet']
        exp = tw_full_preprocess(exp_s)
        count_vect = CountVectorizer(analyzer='word', encoding='cp1251', vocabulary=vocabulary)
        count_vect._validate_vocabulary()
        exp_vect_2 = count_vect.transform([exp])
        tfidf_transformer = TfidfTransformer(use_idf=False)
        exp_vect_2_tfidf = tfidf_transformer.transform(exp_vect_2)
        predict = model_mnb.predict(exp_vect_2_tfidf)
        if predict[0] == 0:
            answer = 'Негативная тональность'
        else:
            answer = 'Позитивная тональность'
        #temp = predict[0]
        return render_template('main.html', result=answer)

if __name__ == '__main__':
    app.run()