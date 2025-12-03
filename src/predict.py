import pickle

with open("../models/log_model.pkl","rb") as f:
    model = pickle.load(f)
with open("../models/tf_idfVectorizer.pkl" ,"rb") as f:
    vectorizer= pickle.load(f)

def predict_labels(text:str):
    if not text.strip():
        return {"error": "Input text is empty"}
    try:
        facts = vectorizer.transform([text])
        prediction = model.predict(facts)[0]
        confidence = round(float(model.predict_proba(facts).max())* 100,2)
        return{
            "prediction": prediction,
            "confidence_percent": confidence
        }
    except Exception as e:
      return {"error": f"Prediction failed: {e}"}


