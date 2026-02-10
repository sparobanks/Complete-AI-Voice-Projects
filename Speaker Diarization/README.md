# Speaker Diarization (Who Spoke When)
Speaker Segmentation System using FastAPI, Streamlit & pyannote.audio

Project 04 is a **Speaker Diarization system** that answers the question:

“Who spoke, and when?”

Given a long audio recording (meeting, interview, call, podcast), the system automatically:
- Detects speaker changes
- Assigns speaker labels
- Outputs precise time ranges for each speaker segment

## FEATURES

- Upload long-form audio (wav, mp3, m4a, etc.)
- Automatic speaker diarization
- Time-stamped speaker segments
- Speaker count estimation
- Optional min/max speaker constraints
- Download results as TXT or JSON
- Clean API + Web UI separation
- CPU-friendly inference (GPU optional)

## PROBLEM

In many real-world scenarios (meetings, interviews, call centers), we need to know:
- Who spoke
- When they spoke
- For how long

Traditional speech-to-text systems do not provide speaker identity.  
Speaker diarization solves this gap but is rarely implemented as a **complete, usable system**.

## SOLUTION

This project implements an **end-to-end diarization pipeline** using a pretrained, industry-standard model.

It includes:
- A FastAPI backend for diarization inference
- A Streamlit frontend for user interaction
- Automatic audio preprocessing
- Structured, exportable results

The result is a **production-style diarization service**, not just a research demo.

## ARCHITECTURE

User (Browser)  
→ Streamlit Frontend  
→ FastAPI Backend  
→ Audio Preprocessing (16kHz mono)  
→ pyannote.audio Diarization Pipeline  
→ Speaker Segments + Timeline  
→ Downloadable Outputs

## MODEL CHOICE

pyannote.audio — Speaker Diarization Pipeline

Why pyannote.audio:
- Industry-standard diarization toolkit
- State-of-the-art pretrained models
- Robust on real conversations
- Used in research and production systems

Model used by default:
pyannote/speaker-diarization-3.1

## OUTPUT FORMAT

Each diarization result includes:

- Speaker label (Speaker_0, Speaker_1, etc.)
- Start time (seconds)
- End time (seconds)
- Segment duration

Example:

[0.000 – 12.430] Speaker_0  
[12.430 – 25.180] Speaker_1  
[25.180 – 40.900] Speaker_0  

## TECH STACK

Language: Python 3.10  
Frontend: Streamlit  
Backend: FastAPI  
Diarization: pyannote.audio  
Audio: pydub, ffmpeg, soundfile  
Data Handling: pandas  
Runtime: CPU (GPU optional)

## REQUIREMENTS

This project requires a **Hugging Face access token** because the diarization model is gated.

You must:
1. Create a Hugging Face account
2. Generate an access token
3. Export it before running the API

## SETUP & RUN

**Create environment**
```text
conda create -n voice_lab python=3.10 -y  
conda activate voice_lab  
```
**Install dependencies**
```text
pip install -r requirements.txt  
```
**Set Hugging Face token (required)**
```text
export HF_TOKEN="YOUR_HF_TOKEN_HERE"  
```
**Start FastAPI backend (port 8005)**
```text
export KMP_DUPLICATE_LIB_OK=TRUE  
python -m uvicorn api.main:app --host 127.0.0.1 --port 8005  
```
**API endpoints**

http://127.0.0.1:8005/health  
http://127.0.0.1:8005/docs  

**Start Streamlit frontend**
```text
streamlit run app/Home.py  
```
**Web interface**

http://localhost:8501

## RESULTS

- Accurate detection of speaker turns
- Clear speaker segmentation on meetings and interviews
- Structured outputs suitable for downstream tasks
- Stable execution on macOS (Intel & Apple Silicon)


## CURRENT LIMITATIONS

- No speaker name assignment (generic labels only)
- No speaker-to-identity mapping
- No automatic transcription
- No real-time streaming diarization

These are intentional to keep the project focused and stable.


## PLANNED IMPROVEMENTS

- Combine diarization with speech-to-text
- Speaker-labeled transcripts
- Speaker renaming in UI
- Meeting analytics (talk time per speaker)
- Cloud deployment

## WHY THIS PROJECT MATTERS

This project demonstrates:
- Practical speaker diarization
- Audio segmentation and clustering
- Use of industry-grade pretrained models
- Turning research pipelines into usable applications

## AUTHOR

Jasper Chinedu (LordSparo)  
MSc Computer Science  
Voice AI • Machine Learning • Applied AI Systems
