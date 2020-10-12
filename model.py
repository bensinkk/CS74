import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import re
from statistics import mean

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.svm import LinearSVC
from sklearn import metrics

def read_data(root_dir='data', ext='json', test=False):
	if ext == 'csv':
		return pd.read_csv(root_dir+'/Sports_and_Outdoors_Ratings_training.csv')
	elif test:
		return pd.read_json(root_dir+'/Sports_and_Outdoors_Reviews_test.json',
							lines=True)
	else:
		return pd.read_json(root_dir+'/Sports_and_Outdoors_Reviews_training.json',
							lines=True)

def process_text(text):
    text = ' '.join(str(text).split())
    return text

def rate_product(overall):
    if overall > 4.5:
        result = 'awesome'
    else:
        result = 'not'
        
    return result

print("Setting up DataFrame, this may take a while...")
raw_df = read_data(root_dir='data')
grouped_df = raw_df.groupby('asin')
grouped_lists = grouped_df['summary'].apply(process_text).reset_index()
mean_df = grouped_df['overall'].mean()
mean_df = mean_df.reset_index()
final_df = pd.merge(grouped_lists, mean_df, on='asin')
final_df['class'] = final_df.apply(lambda row: rate_product(row['overall']), axis=1)

print("Setting up NLTK processing...")
# some NLTK things
nltk.download('stopwords')
stemmer = SnowballStemmer('english', ignore_stopwords=True)
stop_words = stopwords.words('english')
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(preprocessor=stemmer.stem, stop_words=stop_words, ngram_range = (1,3), tokenizer=token.tokenize)
text_counts = cv.fit_transform(final_df['summary'])

print("Running SVC Classifier...")
SVC_classifier = LinearSVC()
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=5)
count = 1
X = text_counts
y = final_df['class']
f1_scores, auc_scores = list(), list()
for train_idx, test_idx in sss.split(X, y):
    print ("Group" , count)
    count += 1
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    SVC_classifier.fit(X_train, y_train)
    y_score = SVC_classifier.decision_function(X_test)
    y_pred_SVC = SVC_classifier.predict(X_test)
    # checking accuracy
    f1_score_SVC = metrics.f1_score(y_test, y_pred_SVC, average='weighted')
    precision_score_SVC = metrics.average_precision_score(y_test, y_score, average='weighted', pos_label='not')
    
    f1_scores.append(f1_score_SVC)
    auc_scores.append(precision_score_SVC)
    
    print("SVC F1: " + str('{:04.2f}'.format(f1_score_SVC*100)) + "%")
    print("SVC Precision-Recall AUC: " + str('{:04.2f}'.format(precision_score_SVC*100)) + "%\n")

print("F1 average: " + str('{:04.2f}'.format(mean(f1_scores)*100)) + "%")
print("Precision-Recall AUC average: " + str('{:04.2f}'.format(mean(auc_scores)*100)) + "%")


