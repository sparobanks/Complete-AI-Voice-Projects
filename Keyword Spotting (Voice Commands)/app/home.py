import streamlit as st
import requests
import tempfile
import os
from audiorecorder import audiorecorder

API_URL = "http://127.0.0.1:8004"  # We'll run API on 8004 to avoid conflicts

st.set_page_config(page_title="Project 03 — Keyword Spotting", page_icon="🎛️", layout="centered")
st.title("🎛️ Project 03 — Keyword Spotting (Voice Commands)")
st.caption("Record or upload a short clip. The model predicts the spoken command (e.g., yes/no/up/down/left/right/stop/go).")

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
            "Upload audio (1 second is ideal)",
            type=["wav", "mp3", "m4a", "aac", "ogg", "flac", "webm"],
            key=f"{key_prefix}_upload",
        )
        if up is not None:
            audio_bytes = up.read()
            filename = up.name
            st.audio(audio_bytes)

    else:
        st.write("Record ~1 second command (say one word). Then stop.")
        audio = audiorecorder("⏺️ Record", "⏹️ Stop", key=f"{key_prefix}_rec")
        if len(audio) > 0:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name
            audio.export(tmp_path, format="wav")
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            os.remove(tmp_path)
            filename = "mic_command.wav"
            st.audio(audio_bytes)

    return filename, audio_bytes


filename, audio_bytes = get_audio_input("kws")

st.divider()
st.info("If the API says model_ready=false, run:  python scripts/train.py  (it creates the model weights).")

if st.button("🔎 Predict Command", type="primary", disabled=(audio_bytes is None)):
    with st.spinner("Predicting..."):
        try:
            files = {"file": (filename, audio_bytes)}
            resp = requests.post(f"{API_URL}/predict", files=files, timeout=120)

            if resp.status_code != 200:
                st.error(resp.text)
            else:
                data = resp.json()
                st.success("Done!")
                st.metric("Prediction", data["prediction"])
                st.metric("Confidence", f"{data['confidence']:.3f}")

                st.subheader("Top 3")
                for label, prob in data["top3"]:
                    st.write(f"- {label}: {float(prob):.3f}")

        except Exception as e:
            st.error(f"Failed: {e}")
