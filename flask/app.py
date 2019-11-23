import flask
from flask import render_template
import pickle
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html' )
        
    if flask.request.method == 'POST':
        temp = ''
        with open('model_mnb_r2.pkl', 'rb') as fh:
            loaded_model = pickle.load(fh)
        exp = flask.request.form['tweet']
        count_vect = CountVectorizer(analyzer='word', encoding='cp1251')
        tfidf_transformer = TfidfTransformer()
        exp_counts = count_vect.transform(exp)
        exp_tfidf = tfidf_transformer.transform(exp_counts)
        temp = loaded_model.predict(exp_tfidf)
        return render_template('main.html', result = temp)

if __name__ == '__main__':
    app.run()