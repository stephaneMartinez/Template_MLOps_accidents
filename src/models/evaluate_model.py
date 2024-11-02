
import sklearn
import pandas as pd 
from sklearn import ensemble
from sklearn.metrics import accuracy_score
import joblib, json
import numpy as np

print(joblib.__version__)

X_train = pd.read_csv('data/preprocessed/X_train.csv')
X_test = pd.read_csv('data/preprocessed/X_test.csv')
y_train = pd.read_csv('data/preprocessed/y_train.csv')
y_test = pd.read_csv('data/preprocessed/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

#--charge le modèle
model_filename = './models/trained_model.joblib'
rf_classifier = joblib.load(model_filename)

#--Evaluate the model
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
data_to_write = {"accuracy": accuracy}

# Écrire le résultat dans un fichier JSON
with open("./metrics/accuracy.json", "w") as f:
    json.dump(data_to_write, f)

print("Model evaluated  successfully.")
