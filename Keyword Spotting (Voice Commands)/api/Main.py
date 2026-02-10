from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import tempfile
import os
import json

import torch
import torch.nn as nn
import torchaudio
import numpy as np

app = FastAPI(title="Project 03 - Keyword Spotting API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "kws_model.pth"
LABELS_PATH = MODELS_DIR / "labels.json"

TARGET_SR = 16000
CLIP_SECONDS = 1.0
CLIP_SAMPLES = int(TARGET_SR * CLIP_SECONDS)


def pad_or_trim(waveform: torch.Tensor) -> torch.Tensor:
    T = waveform.shape[-1]
    if T > CLIP_SAMPLES:
        return waveform[..., :CLIP_SAMPLES]
    if T < CLIP_SAMPLES:
        return torch.nn.functional.pad(waveform, (0, CLIP_SAMPLES - T))
    return waveform


def load_audio(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    m = wav.abs().max()
    if m > 0:
        wav = wav / m
    wav = pad_or_trim(wav)
    return wav  # [1, 16000]


class MFCCFeaturizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=TARGET_SR,
            n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64},
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        x = self.mfcc(wav)          # [B, n_mfcc, frames]
        return x.unsqueeze(1)       # [B, 1, n_mfcc, frames]


class SmallKWSNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def load_model():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        return None, None

    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    ckpt = torch.load(str(MODEL_PATH), map_location="cpu")

    model = SmallKWSNet(num_classes=len(labels))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, labels


featurizer = MFCCFeaturizer()
model, labels = load_model()


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    top3: list[list[str | float]]


@app.get("/health")
def health():
    ready = model is not None and labels is not None
    return {"status": "ok", "model_ready": ready, "expected_sr": TARGET_SR}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    global model, labels

    if model is None or labels is None:
        raise HTTPException(
            status_code=400,
            detail="Model not found. Train first: python scripts/train.py (this creates models/kws_model.pth).",
        )

    suffix = os.path.splitext(file.filename)[1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        wav = load_audio(tmp_path)               # [1, 16000]
        wav_b = wav.unsqueeze(0)                 # [B=1, 1, 16000]
        feats = featurizer(wav_b)                # [1, 1, 40, frames]

        with torch.no_grad():
            logits = model(feats)                # [1, C]
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_idx = probs.argsort()[::-1][:3]
        top3 = [[labels[i], float(probs[i])] for i in top_idx]

        pred_idx = int(top_idx[0])
        return PredictResponse(
            prediction=labels[pred_idx],
            confidence=float(probs[pred_idx]),
            top3=top3,
        )

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
