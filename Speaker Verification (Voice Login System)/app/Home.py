import streamlit as st
import requests
import tempfile
import os
from audiorecorder import audiorecorder

API_URL = "http://127.0.0.1:8003"  # Make sure your FastAPI runs on this port

st.set_page_config(page_title="Speaker Verification", page_icon="🗣️", layout="centered")
st.title("🗣️ Speaker Verification (Voice Login)")
st.caption("Enroll a voice profile, then verify later using cosine similarity on ECAPA-TDNN embeddings (SpeechBrain).")

# ----------------------------
# API status
# ----------------------------
with st.expander("API status"):
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        st.json(r.json())
    except Exception as e:
        st.error(f"API not reachable: {e}")


# ----------------------------
# Helper: record or upload (with unique Streamlit keys)
# ----------------------------
def record_or_upload(key_prefix: str):
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
            "Upload audio",
            type=["wav", "mp3", "m4a", "aac", "ogg", "flac", "webm"],
            key=f"{key_prefix}_upload",
        )
        if up is not None:
            audio_bytes = up.read()
            filename = up.name
            st.audio(audio_bytes)

    else:
        st.write("Click record, speak for ~5–10 seconds, then stop.")
        audio = audiorecorder(
            "⏺️ Record",
            "⏹️ Stop",
            key=f"{key_prefix}_recorder",
        )

        if len(audio) > 0:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name

            audio.export(tmp_path, format="wav")

            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()

            try:
                os.remove(tmp_path)
            except Exception:
                pass

            filename = "mic_recording.wav"
            st.audio(audio_bytes)

    return filename, audio_bytes


# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["Enroll", "Verify", "Users"])


# ----------------------------
# Enroll Tab
# ----------------------------
with tab1:
    st.subheader("Enroll voice profile")
    username = st.text_input("Username (letters/numbers/_/- only)", placeholder="e.g., jasper", key="enroll_username")

    filename, audio_bytes = record_or_upload("enroll")

    st.caption("Tip: Enroll 2–3 times for the same username to make the voice profile more stable.")

    if st.button("✅ Enroll", type="primary", disabled=not (username and audio_bytes), key="enroll_btn"):
        try:
            files = {"file": (filename, audio_bytes)}
            resp = requests.post(f"{API_URL}/enroll/{username}", files=files, timeout=300)

            if resp.status_code != 200:
                st.error(resp.text)
            else:
                data = resp.json()
                st.success(data.get("message", "Enrolled successfully."))
        except Exception as e:
            st.error(f"Enroll failed: {e}")


# ----------------------------
# Verify Tab
# ----------------------------
with tab2:
    st.subheader("Verify voice (login check)")

    users = []
    try:
        users = requests.get(f"{API_URL}/users", timeout=10).json().get("users", [])
    except Exception:
        users = []

    if users:
        username2 = st.selectbox("Select enrolled user", users, key="verify_user_select")
    else:
        username2 = st.text_input("Username (must already be enrolled)", key="verify_username_text")

    threshold = st.slider("Verification threshold", 0.50, 0.95, 0.75, 0.01, key="verify_threshold")
    st.caption("Higher threshold = stricter verification. Start around 0.75.")

    filename2, audio_bytes2 = record_or_upload("verify")

    if st.button("🔐 Verify", type="primary", disabled=not (username2 and audio_bytes2), key="verify_btn"):
        try:
            files = {"file": (filename2, audio_bytes2)}
            resp = requests.post(
                f"{API_URL}/verify/{username2}",
                files=files,
                params={"threshold": threshold},
                timeout=300
            )

            if resp.status_code != 200:
                st.error(resp.text)
            else:
                data = resp.json()
                sim = float(data.get("similarity", 0.0))
                ok = bool(data.get("verified", False))

                st.metric("Similarity Score", f"{sim:.3f}")
                st.write(f"Threshold: **{float(data.get('threshold', threshold)):.2f}**")

                # Visual gauge
                st.progress(min(max(sim, 0.0), 1.0))

                if ok:
                    st.success("✅ VERIFIED (MATCH)")
                else:
                    st.error("❌ NOT VERIFIED (NO MATCH)")

        except Exception as e:
            st.error(f"Verify failed: {e}")


# ----------------------------
# Users Tab
# ----------------------------
with tab3:
    st.subheader("Manage enrolled users")

    try:
        users_data = requests.get(f"{API_URL}/users", timeout=10).json()
        users_list = users_data.get("users", [])
    except Exception as e:
        st.error(f"Could not load users: {e}")
        users_list = []

    if users_list:
        st.write("Enrolled users:")
        st.write(users_list)

        user_to_delete = st.selectbox("Select user to delete", users_list, key="delete_user_select")

        if st.button("🗑️ Delete user", type="secondary", key="delete_btn"):
            try:
                resp = requests.delete(f"{API_URL}/users/{user_to_delete}", timeout=30)
                if resp.status_code != 200:
                    st.error(resp.text)
                else:
                    st.success("Deleted. Refresh the page to update the user list.")
            except Exception as e:
                st.error(f"Delete failed: {e}")
    else:
        st.info("No users enrolled yet. Enroll a voice profile first.")
