import streamlit as st
import requests
import pandas as pd
import json

API_URL = "http://127.0.0.1:8005"

st.set_page_config(page_title="Project 04 — Speaker Diarization", page_icon="🎧", layout="centered")
st.title("🎧 Project 04 — Speaker Diarization (Who Spoke When)")
st.caption("Upload an audio file and get time-stamped speaker segments. Output includes speaker labels and time ranges.")

with st.expander("API status"):
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        st.json(r.json())
        if not r.json().get("token_set", False):
            st.warning(
                "HF_TOKEN is not set on the API machine. Set it and restart the API:\n"
                "export HF_TOKEN='YOUR_TOKEN'"
            )
    except Exception as e:
        st.error(f"API not reachable: {e}")

st.subheader("Upload audio")
up = st.file_uploader("Upload audio (mp3, wav, m4a, etc.)", type=["wav", "mp3", "m4a", "aac", "ogg", "flac", "webm"])

st.subheader("Optional diarization controls")
col1, col2 = st.columns(2)
with col1:
    min_speakers = st.number_input("min_speakers (optional)", min_value=0, value=0, step=1)
with col2:
    max_speakers = st.number_input("max_speakers (optional)", min_value=0, value=0, step=1)

model_name = st.text_input(
    "Model name",
    value="pyannote/speaker-diarization-3.1",
    help="Default works for most cases. Keep it unless you know what you’re doing.",
)

if up is not None:
    audio_bytes = up.read()
    st.audio(audio_bytes)

    if st.button("🎯 Run diarization", type="primary"):
        params = {"model_name": model_name}
        if min_speakers > 0:
            params["min_speakers"] = int(min_speakers)
        if max_speakers > 0:
            params["max_speakers"] = int(max_speakers)

        try:
            with st.spinner("Running diarization... (may take time for long audio)"):
                files = {"file": (up.name, audio_bytes)}
                resp = requests.post(f"{API_URL}/diarize", files=files, params=params, timeout=600)

            if resp.status_code != 200:
                st.error(resp.text)
            else:
                data = resp.json()
                st.success(f"Done! Speakers detected: {data['num_speakers']}")
                segments = data.get("segments", [])

                if not segments:
                    st.warning("No segments found. Try a clearer audio file or adjust min/max speakers.")
                else:
                    df = pd.DataFrame(segments)
                    df = df.sort_values(["start", "end"]).reset_index(drop=True)

                    st.subheader("Segments")
                    st.dataframe(df, use_container_width=True)

                    st.subheader("Download outputs")

                    # TXT output
                    lines = []
                    for _, row in df.iterrows():
                        lines.append(f"[{row['start']:0.3f} - {row['end']:0.3f}] {row['speaker']}")
                    txt_out = "\n".join(lines)
                    st.download_button("⬇️ Download TXT", txt_out, file_name="diarization.txt")

                    # JSON output
                    json_out = json.dumps(data, indent=2)
                    st.download_button("⬇️ Download JSON", json_out, file_name="diarization.json")

        except Exception as e:
            st.error(f"Failed: {e}")
