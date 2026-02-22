import os
import logging
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import DominoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Domino AI Keras Server")

# Store multiple model instances keyed by model_id
models: dict[str, DominoModel] = {}

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def get_or_create_model(model_id: str, hidden_dims: Optional[list[int]] = None) -> DominoModel:
    if model_id not in models:
        dims = hidden_dims or [150]
        logger.info(f"Creating model '{model_id}' with hidden_dims={dims}")
        models[model_id] = DominoModel(hidden_dims=dims)
    return models[model_id]


# --- Request/Response schemas ---

class PredictRequest(BaseModel):
    model_id: str
    features: list[float]       # 126 floats
    valid_mask: list[float]     # 56 floats
    hidden_dims: Optional[list[int]] = None

class PredictResponse(BaseModel):
    card: int
    side: str
    confidence: float

class TrainSample(BaseModel):
    features: list[float]       # 126 floats
    target: list[float]         # 56 floats
    action_mask: list[float]    # 56 floats

class TrainRequest(BaseModel):
    model_id: str
    samples: list[TrainSample]
    learning_rate: float = 0.001
    hidden_dims: Optional[list[int]] = None

class TrainResponse(BaseModel):
    avg_loss: float
    weights: Optional[list[list[float]]] = None  # flattened weight matrices returned after training

class ModelIDRequest(BaseModel):
    model_id: str
    hidden_dims: Optional[list[int]] = None

class SaveLoadRequest(BaseModel):
    model_id: str
    path: Optional[str] = None
    hidden_dims: Optional[list[int]] = None

class OkResponse(BaseModel):
    ok: bool


# --- Endpoints ---

@app.get("/health")
def health():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
        "gpu_available": len(gpus) > 0,
        "gpu_devices": [g.name for g in gpus],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.features) != 126:
        raise HTTPException(400, f"Expected 126 features, got {len(req.features)}")
    if len(req.valid_mask) != 56:
        raise HTTPException(400, f"Expected 56 valid_mask, got {len(req.valid_mask)}")

    model = get_or_create_model(req.model_id, req.hidden_dims)
    features = np.array(req.features, dtype=np.float32)
    valid_mask = np.array(req.valid_mask, dtype=np.float32)

    card, side, confidence = model.predict(features, valid_mask)
    return PredictResponse(card=card, side=side, confidence=confidence)


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    if len(req.samples) == 0:
        return TrainResponse(avg_loss=0.0)

    model = get_or_create_model(req.model_id, req.hidden_dims)

    features_batch = np.array([s.features for s in req.samples], dtype=np.float32)
    target_batch = np.array([s.target for s in req.samples], dtype=np.float32)
    mask_batch = np.array([s.action_mask for s in req.samples], dtype=np.float32)

    if features_batch.shape[1] != 126:
        raise HTTPException(400, f"Expected 126 features per sample, got {features_batch.shape[1]}")
    if target_batch.shape[1] != 56:
        raise HTTPException(400, f"Expected 56 target values per sample, got {target_batch.shape[1]}")
    if mask_batch.shape[1] != 56:
        raise HTTPException(400, f"Expected 56 mask values per sample, got {mask_batch.shape[1]}")

    avg_loss = model.train_batch(features_batch, target_batch, mask_batch, req.learning_rate)

    # Return updated weights so Go can do fast local inference
    weights = []
    for w in model.model.get_weights():
        weights.append(w.flatten().tolist())
    return TrainResponse(avg_loss=avg_loss, weights=weights)


@app.post("/save", response_model=OkResponse)
def save(req: SaveLoadRequest):
    model = get_or_create_model(req.model_id, req.hidden_dims)
    path = req.path or os.path.join(WEIGHTS_DIR, f"{req.model_id}.weights.h5")
    model.save_weights(path)
    logger.info(f"Saved model '{req.model_id}' to {path}")
    return OkResponse(ok=True)


@app.post("/load", response_model=OkResponse)
def load(req: SaveLoadRequest):
    model = get_or_create_model(req.model_id, req.hidden_dims)
    path = req.path or os.path.join(WEIGHTS_DIR, f"{req.model_id}.weights.h5")
    if not os.path.exists(path):
        raise HTTPException(404, f"Weight file not found: {path}")
    model.load_weights(path)
    logger.info(f"Loaded model '{req.model_id}' from {path}")
    return OkResponse(ok=True)


@app.post("/get_weights")
def get_weights(req: ModelIDRequest):
    model = get_or_create_model(req.model_id, req.hidden_dims)
    weights = []
    shapes = []
    for w in model.model.get_weights():
        shapes.append(list(w.shape))
        weights.append(w.flatten().tolist())
    return {"weights": weights, "shapes": shapes}


@app.post("/reset", response_model=OkResponse)
def reset(req: ModelIDRequest):
    """Re-initialize a model with fresh random weights."""
    dims = None
    if req.model_id in models:
        dims = models[req.model_id].hidden_dims
        del models[req.model_id]
    get_or_create_model(req.model_id, req.hidden_dims or dims)
    logger.info(f"Reset model '{req.model_id}'")
    return OkResponse(ok=True)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8777"))
    uvicorn.run(app, host="0.0.0.0", port=port)
