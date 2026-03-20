import os
import json
import logging
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

logger = logging.getLogger("sentiment-api")
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE_DIR, "artefacts")

MODEL_PATH = os.path.join(ART_DIR, "mod.keras")
TOKENIZER_PATH = os.path.join(ART_DIR, "tokenizer.json")
CONFIG_PATH = os.path.join(ART_DIR, "config.json")

print("BASE_DIR:", BASE_DIR)
print("MODEL_PATH:", MODEL_PATH)
print("Files:", os.listdir(ART_DIR) if os.path.exists(ART_DIR) else "No dir")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing {MODEL_PATH}")

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Missing {TOKENIZER_PATH}")

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Missing {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = json.load(f)

MAX_LEN = int(CFG["max_len"])
THRESHOLD = float(CFG.get("threshold", 0.5))
LABEL_MAP = CFG.get("label_map", {"0": "negative", "1": "positive"})

with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    TOKENIZER = tokenizer_from_json(f.read())


MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        logger.info("loading_model")
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        logger.info("model_loaded")
    return MODEL


app = FastAPI(title="P7 Sentiment API", version="1.0")

class PredictIn(BaseModel):
    text: str = Field(..., min_length=1)


class PredictOut(BaseModel):
    label: int
    label_name: str
    proba: float

def predict_one(text: str):
    model = get_model()
    seq = TOKENIZER.texts_to_sequences([text])
    x = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    proba = float(model.predict(x, verbose=0).ravel()[0])
    label = int(proba >= THRESHOLD)
    label_name = LABEL_MAP.get(str(label), str(label))
    return label, label_name, proba

@app.get("/")
def root():
    return {"message": "P7 Sentiment API is running"}


@app.get("/health")
def health():
    logger.info(
        "health_check",
        extra={
            "event_name": "health_check",
            "max_len": MAX_LEN,
        },
    )
    return {"status": "ok", "max_len": MAX_LEN}


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    label, label_name, proba = predict_one(payload.text)

    logger.info(
        "prediction_made",
        extra={
            "event_name": "prediction_made",
            "label": label,
            "label_name": label_name,
            "predicted_sentiment": label_name,
            "proba": proba,
            "text_len": len(payload.text),
            "threshold": THRESHOLD,
        },
    )

    return PredictOut(label=label, label_name=label_name, proba=proba)
