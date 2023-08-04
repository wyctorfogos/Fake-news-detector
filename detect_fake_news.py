import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    SGDClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix


##Define classifiers
classifiers = [
    ("SGD", SGDClassifier(max_iter=110)),
    ("ASGD", SGDClassifier(max_iter=110, average=True)),
    ("Perceptron", Perceptron(max_iter=110)),
    (
        "Passive-Aggressive I",
        PassiveAggressiveClassifier(max_iter=110, loss="hinge", C=1.0, tol=1e-4),
    ),
    (
        "Passive-Aggressive II",
        PassiveAggressiveClassifier(
            max_iter=110, loss="squared_hinge", C=1.0, tol=1e-4
        ),
    ),
    (
        "SAG",
        LogisticRegression(max_iter=110, solver="sag", tol=1e-1, C=1.0),
    ),
]


##Define the wnted models and how to train it
def train_and_evaluate(features,labels):
	#Split data
	X_train, X_test, y_train, y_test= train_test_split(features, labels, test_size=0.3, shuffle=True)

	#Train model
	
	for name, clf in classifiers:
		print("training %s" % name)
		clf.fit(X_train, y_train)
		##Make prediction
		
		acc=accuracy_score(y_test, clf.predict(X_test))*100
		print("Using the content titles as the input to be anaysed")
		print("Accuracy: {} %".format(acc))
		print("=======")


data= pd.read_csv("./DATASET/news.csv")

#When using the title as the input features
##PREPROCESS DATA
transf_vector=TfidfVectorizer(stop_words='english', max_df=0.75)
features=transf_vector.fit_transform(data['title'])

labels=data['label']
train_and_evaluate(features,labels)

###Now, using the text as the input
#When using the title as the input features
##PREPROCESS DATA
print("Using the 'text' content!!!!")
features=transf_vector.fit_transform(data['text'])
train_and_evaluate(features,labels)

    
