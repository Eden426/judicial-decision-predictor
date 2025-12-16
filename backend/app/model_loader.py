from pathlib import Path
import pickle

# Go up two levels: app/ → backend/ → project root
BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "log_model.pkl"
VEC_PATH = BASE_DIR / "models" / "tf_idfVectorizer.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VEC_PATH, "rb") as f:
    vectorizer = pickle.load(f)
