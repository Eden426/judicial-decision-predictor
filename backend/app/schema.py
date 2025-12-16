from pydantic import BaseModel


class PredictFacts(BaseModel):
    facts: str

class PredictOutcome(BaseModel):
    prediction : str
    confidence : float
