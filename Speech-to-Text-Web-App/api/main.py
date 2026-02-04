from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI(title="Voice Lab API", version="1.0")

# CORS so Streamlit can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model config (set in terminal: export WHISPER_MODEL=base)
MODEL_NAME = os.getenv("WHISPER_MODEL", "base")

# CPU-friendly defaults:
# compute_type="int8" is fast and light on CPU
model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")


class TranscribeResponse(BaseModel):
    text: str
    language: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    # Save upload to a temporary file
    suffix = os.path.splitext(file.filename)[1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(tmp_path)

        # Join segments into one text
        text = " ".join(seg.text for seg in segments).strip()

        return TranscribeResponse(text=text, language=getattr(info, "language", None))

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
