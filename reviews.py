from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import re
import string
import pandas as pd
import os

from flask_cors import CORS


# Load and preprocess the data
dir_Path = 'archive'
Reviewdata = pd.read_csv(os.path.join(dir_Path, 'train.csv'))
Reviewdata.drop(columns=['User_ID', 'Browser_Used',
                'Device_Used'], inplace=True)


def text_clean_1(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def cleaned1(x): return text_clean_1(x)


Reviewdata['cleaned_description'] = pd.DataFrame(
    Reviewdata.Description.apply(cleaned1))


def text_clean_2(text):
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text


def cleaned2(x): return text_clean_2(x)


Reviewdata['cleaned_description_new'] = pd.DataFrame(
    Reviewdata['cleaned_description'].apply(cleaned2))

# Train the model
Independent_var = Reviewdata.cleaned_description_new
Dependent_var = Reviewdata.Is_Response

tvec = TfidfVectorizer()
clf2 = LogisticRegression(solver="lbfgs")
clf2 = LogisticRegression(solver="lbfgs", max_iter=1000)
model = Pipeline([('vectorizer', tvec), ('classifier', clf2)])
model.fit(Independent_var, Dependent_var)


