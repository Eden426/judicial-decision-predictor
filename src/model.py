
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("../dataset/preprocessed/cleaned_data.csv")
train = data["facts"]
output = data["label"]

facts_train, facts_test, label_train, label_test = (
    train_test_split(train,output, test_size= 0.2, random_state=42, stratify=output  ))
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words="english"
)
facts_train_idf = vectorizer.fit_transform(facts_train)
facts_test_idf = vectorizer.transform(facts_test)

model = LogisticRegression(max_iter=500, class_weight="balanced")
model.fit(facts_train_idf, label_train)




print("Accuracy:",model.score(facts_test_idf,label_test))

with open("../models/log_model.pkl","wb") as f:
    pickle.dump(model, f)
print("Saved model!")

with open("../models/tf_idfVectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Saved vectorizer!")

