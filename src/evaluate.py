import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

with open("../models/log_model.pkl","rb") as f:
    model = pickle.load(f)
with open("../models/tf_idfVectorizer.pkl","rb") as f:
    vectorizer = pickle.load(f)

data = pd.read_csv("../dataset/preprocessed/cleaned_data.csv")
train = data["facts"]
label = data["label"]

train_test_vec = vectorizer.transform(train)
output_pred = model.predict(train_test_vec)

print(classification_report(label, output_pred))
print(confusion_matrix(label, output_pred))
