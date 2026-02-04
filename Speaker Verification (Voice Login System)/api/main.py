from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import tempfile
import os
import json
import numpy as np
import torch
import torchaudio

# SpeechBrain speaker embedding model
from speechbrain.inference.speaker import SpeakerRecognition

app = FastAPI(title="Speaker Verification API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
SPEAKERS_DIR = BASE_DIR / "data" / "speakers"
MODELS_DIR = BASE_DIR / "models"
SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load pretrained ECAPA-TDNN verification model (downloads once)
verifier = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(MODELS_DIR / "spkrec-ecapa-voxceleb"),
)

TARGET_SR = 16000


class EnrollResponse(BaseModel):
    username: str
    message: str


class VerifyResponse(BaseModel):
    username: str
    similarity: float
    threshold: float
    verified: bool


class UsersResponse(BaseModel):
    users: list[str]


def _safe_username(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("Username cannot be empty.")
    # Keep it filesystem-safe
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    cleaned = "".join(ch for ch in name if ch in allowed)
    if not cleaned:
        raise ValueError("Username contains no valid characters.")
    return cleaned


def _load_audio_as_mono_16k(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)  # shape: [channels, time]
    if wav.ndim != 2:
        raise ValueError("Invalid audio tensor shape.")
    # Convert to mono
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    # Resample
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    # Normalize (avoid division by zero)
    max_val = torch.max(torch.abs(wav))
    if max_val > 0:
        wav = wav / max_val
    return wav  # shape: [1, time]


def _embed_from_wav_tensor(wav: torch.Tensor) -> np.ndarray:
    # SpeechBrain expects [batch, time] or [batch, channels, time] depending
    # encode_batch works with [batch, time] or [batch, channels, time]
    if wav.ndim == 2:
        # [1, time] -> add batch dimension? already batch-like. We'll treat as [batch, time]
        batch = wav
    elif wav.ndim == 3:
        batch = wav.squeeze(1)
    else:
        raise ValueError("Unexpected wav dimensions.")

    with torch.no_grad():
        emb = verifier.encode_batch(batch)  # [batch, emb_dim] or [batch, 1, emb_dim]
        emb = emb.squeeze()
        emb = emb.detach().cpu().numpy().astype(np.float32)

    # L2 normalize embedding for cosine similarity stability
    norm = np.linalg.norm(emb) + 1e-12
    emb = emb / norm
    return emb


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def _user_paths(username: str):
    emb_path = SPEAKERS_DIR / f"{username}.npy"
    meta_path = SPEAKERS_DIR / f"{username}.json"
    return emb_path, meta_path


@app.get("/health")
def health():
    return {"status": "ok", "model": "speechbrain/spkrec-ecapa-voxceleb", "sr": TARGET_SR}


@app.get("/users", response_model=UsersResponse)
def list_users():
    users = sorted([p.stem for p in SPEAKERS_DIR.glob("*.npy")])
    return UsersResponse(users=users)


@app.delete("/users/{username}", response_model=EnrollResponse)
def delete_user(username: str):
    try:
        username = _safe_username(username)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    emb_path, meta_path = _user_paths(username)
    deleted_any = False
    if emb_path.exists():
        emb_path.unlink()
        deleted_any = True
    if meta_path.exists():
        meta_path.unlink()
        deleted_any = True

    if not deleted_any:
        raise HTTPException(status_code=404, detail="User not found.")

    return EnrollResponse(username=username, message="Deleted user profile.")


@app.post("/enroll/{username}", response_model=EnrollResponse)
async def enroll(username: str, file: UploadFile = File(...)):
    try:
        username = _safe_username(username)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    suffix = os.path.splitext(file.filename)[1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        wav = _load_audio_as_mono_16k(tmp_path)
        emb = _embed_from_wav_tensor(wav)

        emb_path, meta_path = _user_paths(username)

        # If user already exists, update by averaging embeddings (running mean)
        if emb_path.exists() and meta_path.exists():
            old_emb = np.load(emb_path).astype(np.float32)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            n = int(meta.get("enroll_count", 1))

            new_emb = (old_emb * n + emb) / (n + 1)
            new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-12)

            np.save(emb_path, new_emb)
            meta["enroll_count"] = n + 1
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            return EnrollResponse(username=username, message=f"Updated voice profile. Enroll count = {n+1}")

        # New user
        np.save(emb_path, emb)
        meta = {"username": username, "enroll_count": 1}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return EnrollResponse(username=username, message="Created voice profile. Enroll count = 1")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Enroll failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.post("/verify/{username}", response_model=VerifyResponse)
async def verify(username: str, file: UploadFile = File(...), threshold: float = 0.75):
    try:
        username = _safe_username(username)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    emb_path, _ = _user_paths(username)
    if not emb_path.exists():
        raise HTTPException(status_code=404, detail="User not enrolled yet.")

    suffix = os.path.splitext(file.filename)[1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        stored = np.load(emb_path).astype(np.float32)
        wav = _load_audio_as_mono_16k(tmp_path)
        test_emb = _embed_from_wav_tensor(wav)

        sim = _cosine_similarity(stored, test_emb)
        verified = sim >= float(threshold)

        return VerifyResponse(
            username=username,
            similarity=sim,
            threshold=float(threshold),
            verified=bool(verified),
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Verify failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
