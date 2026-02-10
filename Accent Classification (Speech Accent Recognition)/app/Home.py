import streamlit as st
import requests
import tempfile
import os
from audiorecorder import audiorecorder

API_URL = "http://127.0.0.1:8006"

st.set_page_config(page_title="Project 05 — Accent Classification", page_icon="🌍", layout="centered")
st.title("🌍 Project 05 — Accent Classification")
st.caption("Record or upload a short voice clip. The model predicts the accent label with confidence and top-3 scores.")

with st.expander("API status"):
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        st.json(r.json())
    except Exception as e:
        st.error(f"API not reachable: {e}")

def get_audio_input(key_prefix: str):
    mode = st.radio(
        "Input method:",
        ["Record from mic", "Upload file"],
        horizontal=True,
        key=f"{key_prefix}_mode",
    )

    audio_bytes = None
    filename = None

    if mode == "Upload file":
        up = st.file_uploader(
            "Upload audio (3–8 seconds recommended)",
            type=["wav", "mp3", "m4a", "aac", "ogg", "flac", "webm"],
            key=f"{key_prefix}_upload",
        )
        if up is not None:
            audio_bytes = up.read()
            filename = up.name
            st.audio(audio_bytes)
    else:
        st.write("Record 3–8 seconds (speak naturally). Then stop.")
        audio = audiorecorder("⏺️ Record", "⏹️ Stop", key=f"{key_prefix}_rec")
        if len(audio) > 0:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name
            audio.export(tmp_path, format="wav")
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            os.remove(tmp_path)
            filename = "mic_clip.wav"
            st.audio(audio_bytes)

    return filename, audio_bytes

st.info("If the API says model_ready=false, train first: python scripts/train.py")

filename, audio_bytes = get_audio_input("accent")

if st.button("🔎 Predict Accent", type="primary", disabled=(audio_bytes is None)):
    try:
        files = {"file": (filename, audio_bytes)}
        resp = requests.post(f"{API_URL}/predict", files=files, timeout=120)
        if resp.status_code != 200:
            st.error(resp.text)
        else:
            data = resp.json()
            st.success("Done!")
            st.metric("Accent Prediction", data["prediction"])
            st.metric("Confidence", f"{data['confidence']:.3f}")

            st.subheader("Top 3 predictions")
            for label, prob in data["top3"]:
                st.write(f"- {label}: {float(prob):.3f}")

    except Exception as e:
        st.error(f"Failed: {e}")
