from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schema import PredictFacts,PredictOutcome
from .model_loader import model,vectorizer

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def home():
    return {"text": "court outcome prediction is alive"}



@app.post("/predict", response_model = PredictOutcome)
def predict(data: PredictFacts):
    text = data.facts.strip()

    if not text:
        raise HTTPException(400, "Input text can not be empty!!")

    try:
        facts_vector = vectorizer.transform([text])
        prediction = model.predict(facts_vector)[0]
        confidence = float(model.predict_proba(facts_vector).max() * 100)

        return PredictOutcome(
            prediction=prediction,
            confidence=round(confidence, 2)
        )

    except Exception as error:
        raise HTTPException(500, f"Prediction failed: {error}")
