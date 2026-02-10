from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import tempfile
import os
import json
from typing import List, Optional

import pandas as pd
from pydub import AudioSegment

# pyannote
from pyannote.audio import Pipeline


app = FastAPI(title="Project 04 - Speaker Diarization API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Config ----
DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"  # common diarization pipeline
TARGET_SR = 16000


class Segment(BaseModel):
    speaker: str
    start: float
    end: float
    duration: float


class DiarizeResponse(BaseModel):
    model: str
    num_speakers: int
    segments: List[Segment]


_pipeline: Optional[Pipeline] = None
_loaded_model_name: Optional[str] = None


def get_token() -> Optional[str]:
    # User sets this before starting API: export HF_TOKEN=...
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")


def load_pipeline(model_name: str = DEFAULT_MODEL) -> Pipeline:
    global _pipeline, _loaded_model_name

    token = get_token()
    if not token:
        raise HTTPException(
            status_code=400,
            detail=(
                "Missing Hugging Face token. Set it before running the API:\n"
                "export HF_TOKEN='YOUR_TOKEN'\n"
                "Then restart uvicorn."
            ),
        )

    if _pipeline is None or _loaded_model_name != model_name:
        try:
            _pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
            _loaded_model_name = model_name
        except TypeError:
            # Some versions use token= instead of use_auth_token=
            _pipeline = Pipeline.from_pretrained(model_name, token=token)
            _loaded_model_name = model_name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load diarization pipeline: {e}")

    return _pipeline


def convert_to_wav_16k_mono(src_path: str) -> str:
    """
    Convert any audio to WAV 16k mono using pydub (requires ffmpeg installed).
    Returns new temp wav path.
    """
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_channels(1).set_frame_rate(TARGET_SR)

    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)
    audio.export(out_path, format="wav")
    return out_path


@app.get("/health")
def health():
    return {
        "status": "ok",
        "token_set": bool(get_token()),
        "default_model": DEFAULT_MODEL,
        "expected_sr": TARGET_SR,
    }


@app.post("/diarize", response_model=DiarizeResponse)
async def diarize(
    file: UploadFile = File(...),
    model_name: str = DEFAULT_MODEL,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
):
    """
    Upload audio -> diarization segments.
    You can optionally pass min_speakers/max_speakers to guide diarization.
    """
    suffix = os.path.splitext(file.filename or "")[1].lower() or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    wav_path = None
    try:
        # Convert to standard format (more reliable across mp3/m4a/etc)
        wav_path = convert_to_wav_16k_mono(tmp_path)

        pipeline = load_pipeline(model_name)

        diarization_kwargs = {}
        if min_speakers is not None:
            diarization_kwargs["min_speakers"] = int(min_speakers)
        if max_speakers is not None:
            diarization_kwargs["max_speakers"] = int(max_speakers)

        # Run diarization
        diar = pipeline(wav_path, **diarization_kwargs)

        # Convert output into segments
        rows = []
        for turn, _, speaker in diar.itertracks(yield_label=True):
            start = float(turn.start)
            end = float(turn.end)
            rows.append(
                {
                    "speaker": str(speaker),
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(end - start, 3),
                }
            )

        if not rows:
            return DiarizeResponse(model=model_name, num_speakers=0, segments=[])

        df = pd.DataFrame(rows).sort_values(["start", "end"]).reset_index(drop=True)

        # Count unique speakers
        num_speakers = int(df["speaker"].nunique())

        segments = [
            Segment(
                speaker=row["speaker"],
                start=float(row["start"]),
                end=float(row["end"]),
                duration=float(row["duration"]),
            )
            for _, row in df.iterrows()
        ]

        return DiarizeResponse(model=model_name, num_speakers=num_speakers, segments=segments)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Diarization failed: {e}")
    finally:
        for p in [tmp_path, wav_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
