import streamlit as st
import requests
import tempfile
import os

# Mic recorder component
from audiorecorder import audiorecorder

API_URL = "http://127.0.0.1:8002"

st.set_page_config(page_title="Voice Lab (ASR)", page_icon="🎙️", layout="centered")
st.title("🎙️ Voice Lab — Speech to Text")
st.caption("Upload an audio file or record from your mic, then get a transcript using Faster-Whisper.")

# API status check
with st.expander("API status"):
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        st.json(r.json())
    except Exception as e:
        st.error(f"API not reachable: {e}")

mode = st.radio("Choose input method:", ["Upload audio file", "Record from mic"], horizontal=True)

audio_bytes = None
filename = None

if mode == "Upload audio file":
    up = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "aac", "ogg", "flac", "webm"])
    if up is not None:
        audio_bytes = up.read()
        filename = up.name
        st.audio(audio_bytes)

else:
    st.write("Click record, speak, then stop.")
    audio = audiorecorder("⏺️ Record", "⏹️ Stop")

    if len(audio) > 0:
        # Export mic recording to WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name
        audio.export(tmp_path, format="wav")

        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()

        os.remove(tmp_path)

        filename = "mic_recording.wav"
        st.audio(audio_bytes)

st.divider()

st.info("Tip: Start with **base** model. If you want more accuracy (slower), use **small**.")

if st.button("📝 Transcribe", type="primary", disabled=(audio_bytes is None)):
    with st.spinner("Transcribing..."):
        try:
            files = {"file": (filename, audio_bytes)}
            resp = requests.post(f"{API_URL}/transcribe", files=files, timeout=600)

            if resp.status_code != 200:
                st.error(f"API error: {resp.status_code} — {resp.text}")
            else:
                data = resp.json()
                st.success("Done!")
                st.write(f"**Detected language:** {data.get('language')}")

                text = data.get("text", "")

                st.subheader("Transcript")
                st.text_area("", value=text, height=220)

                st.download_button(
                    "⬇️ Download transcript (.txt)",
                    data=text.encode("utf-8"),
                    file_name="transcript.txt",
                    mime="text/plain"
                )

        except Exception as e:
            st.error(f"Failed: {e}")

st.caption("Made with Streamlit + FastAPI + Faster-Whisper")
