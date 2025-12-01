import pandas as pd
import re
import os

data = pd.read_csv('../dataset/firstDS/legalCase.csv', index_col=["ID"])

data["disposition"] = data["disposition"].astype(str).str.lower().str.strip()

data["label"] = data["disposition"].apply(
    lambda x: "affirmed" if x == "affirmed" else "reversed"
)
data = data[["facts","label"]]


data["facts"] = data["facts"].astype(str)
data = data[data["facts"].astype(str).str.strip().astype(bool)]
def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

data["facts"] = data["facts"].apply(clean_text)

data = data.drop_duplicates()
print(data.duplicated().sum())



print(data["label"].value_counts())
print(data)
data.to_csv("../dataset/preprocessed/cleaned_data.csv", index=False)