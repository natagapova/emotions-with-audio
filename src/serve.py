"""FastAPI serving endpoint for emotion detection."""

from __future__ import annotations

import io
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from src.dataset import EMOTION_LABELS, IMAGE_SIZE
from src.model import EmotionCNN

CHECKPOINT_PATH = Path("models/best_model.pt")
MAX_RECENT_PREDICTIONS = 100

app = FastAPI(title="Emotion Detection API", version="1.0.0")

_request_count = 0
_recent_confidences: deque[float] = deque(maxlen=MAX_RECENT_PREDICTIONS)
_model: EmotionCNN | None = None
_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def get_model() -> EmotionCNN:
    global _model
    if _model is None:
        model = EmotionCNN()
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _model = model
    return _model


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    tensor = _transform(image).unsqueeze(0)
    return tensor


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    global _request_count
    _request_count += 1

    contents = await file.read()
    tensor = preprocess_image(contents)

    model = get_model()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()

    confidence = float(np.max(probs))
    _recent_confidences.append(confidence)

    predictions = {
        label: float(probs[i])
        for i, label in enumerate(EMOTION_LABELS)
    }
    predicted_label = EMOTION_LABELS[int(np.argmax(probs))]

    return JSONResponse({
        "emotion": predicted_label,
        "confidence": confidence,
        "probabilities": predictions,
    })


@app.get("/metrics")
async def metrics() -> JSONResponse:
    avg_confidence = (
        sum(_recent_confidences) / len(_recent_confidences)
        if _recent_confidences
        else 0.0
    )
    return JSONResponse({
        "total_requests": _request_count,
        "recent_predictions_count": len(_recent_confidences),
        "avg_confidence_last_n": round(avg_confidence, 4),
        "monitoring_note": "avg_confidence drop may indicate data drift",
    })


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})
