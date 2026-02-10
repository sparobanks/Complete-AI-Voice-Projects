from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import tempfile
import os
import json
from typing import List

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from transformers import AutoProcessor, AutoModel

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

LABELS_PATH = MODELS_DIR / "labels.json"
CFG_PATH = MODELS_DIR / "config.json"
HEAD_PATH = MODELS_DIR / "accent_head.pth"

TARGET_SR = 16000
MAX_SECONDS = 8.0
DEVICE = "cpu"

app = FastAPI(title="Project 05 - Accent Classification API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    top3: List[List[float | str]]

class Head(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def load_audio(path: str):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    m = wav.abs().max()
    if m > 0:
        wav = wav / m
    max_len = int(TARGET_SR * MAX_SECONDS)
    if wav.shape[-1] > max_len:
        wav = wav[..., :max_len]
    return wav.squeeze(0)

def load_assets():
    if not (LABELS_PATH.exists() and CFG_PATH.exists() and HEAD_PATH.exists()):
        return None

    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    model_name = cfg.get("model_name", "facebook/wav2vec2-base")

    processor = AutoProcessor.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name).to(DEVICE)
    backbone.eval()

    # infer embedding dim
    with torch.no_grad():
        dummy = processor([torch.zeros(TARGET_SR)], sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        out = backbone(**{k: v.to(DEVICE) for k, v in dummy.items()}).last_hidden_state
        emb_dim = out.shape[-1]

    head = Head(emb_dim, len(labels)).to(DEVICE)
    ckpt = torch.load(HEAD_PATH, map_location="cpu")
    head.load_state_dict(ckpt["head_state"])
    head.eval()

    return {"labels": labels, "processor": processor, "backbone": backbone, "head": head}

assets = load_assets()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": assets is not None,
        "expected_sr": TARGET_SR,
        "max_seconds": MAX_SECONDS,
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    global assets
    if assets is None:
        raise HTTPException(
            status_code=400,
            detail="Model not ready. Train first: python scripts/train.py (creates models/accent_head.pth etc.)",
        )

    suffix = os.path.splitext(file.filename or "")[1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        audio = load_audio(tmp_path)
        processor = assets["processor"]
        backbone = assets["backbone"]
        head = assets["head"]
        labels = assets["labels"]

        inputs = processor([audio], sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            hs = backbone(**inputs).last_hidden_state  # [1, T, D]
            emb = hs.mean(dim=1)                       # [1, D]
            logits = head(emb).squeeze(0)
            probs = torch.softmax(logits, dim=0).cpu().numpy()

        top_idx = probs.argsort()[::-1][:3]
        top3 = [[labels[i], float(probs[i])] for i in top_idx]

        pred_i = int(top_idx[0])
        return PredictResponse(
            prediction=labels[pred_i],
            confidence=float(probs[pred_i]),
            top3=top3,
        )

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
