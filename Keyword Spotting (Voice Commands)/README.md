# Keyword Spotting (Voice Commands)
Voice Command Recognition using FastAPI, Streamlit & PyTorch

Project 03 is a **Keyword Spotting (KWS) system** that detects short spoken commands such as  
**yes, no, up, down, left, right, stop, go** from audio input.

Users can **record audio from a microphone** or **upload an audio file**, and the system predicts the spoken command with a confidence score.

This project is **Project #3** in my **Voice AI Portfolio**, following:
- Project 01 — Speech-to-Text  
- Project 02 — Speaker Verification  

----------------------------------------------------------------

FEATURES

- Short-command detection (keyword spotting)
- Microphone recording or audio file upload
- Real-time prediction
- Confidence score output
- Top-3 predictions
- REST API + Web UI separation
- Lightweight CNN model
- CPU-friendly (no GPU required)

----------------------------------------------------------------

PROBLEM

Voice assistants and smart devices rely on fast and accurate detection of short spoken commands.  
However, many examples online:
- Focus only on training notebooks
- Do not expose usable APIs
- Are not deployable as real applications

Keyword spotting is a **core building block** of voice-controlled systems, but is often poorly demonstrated.

----------------------------------------------------------------

SOLUTION

This project implements a **complete keyword spotting pipeline**:

- A training script to build a custom KWS model
- A FastAPI backend for inference
- A Streamlit frontend for real user interaction
- A compact CNN trained on MFCC features

The result is a **deployable voice-command recognition system**.

----------------------------------------------------------------

ARCHITECTURE

User (Browser)  
→ Streamlit Frontend  
→ FastAPI Backend  
→ MFCC Feature Extraction  
→ CNN Keyword Spotting Model  
→ Predicted Command + Confidence

----------------------------------------------------------------

DATASET

Google Speech Commands Dataset (v0.02)

- Publicly available
- Widely used for keyword spotting research
- Contains thousands of short spoken words
- Automatically downloaded during training

Keywords used in this project:
yes, no, up, down, left, right, stop, go  
Additional classes:
unknown, silence

----------------------------------------------------------------

MODEL DESIGN

- Input: 1-second audio clip
- Features: MFCC (Mel-Frequency Cepstral Coefficients)
- Model: Small Convolutional Neural Network (CNN)
- Output: Softmax probabilities over command classes

This design balances **accuracy, speed, and simplicity**.

----------------------------------------------------------------

TECH STACK

Language: Python 3.10  
Frontend: Streamlit  
Backend: FastAPI  
Model: PyTorch CNN  
Audio: Torchaudio, SoundFile  
Features: MFCC  
Dataset: Google Speech Commands  
Runtime: CPU

----------------------------------------------------------------

TRAIN THE MODEL

Create environment

conda create -n voice_lab python=3.10 -y  
conda activate voice_lab  

Install dependencies

pip install -r requirements.txt  

Create data directory (important)

mkdir -p data  

Train the model

python scripts/train.py  

After training, the following files are created:

models/kws_model.pth  
models/labels.json  

----------------------------------------------------------------

RUN THE APPLICATION

Start FastAPI backend (port 8004)

export KMP_DUPLICATE_LIB_OK=TRUE  
python -m uvicorn api.main:app --host 127.0.0.1 --port 8004  

API endpoints

http://127.0.0.1:8004/health  
http://127.0.0.1:8004/docs  

Start Streamlit frontend

streamlit run app/Home.py  

Web interface

http://localhost:8501

----------------------------------------------------------------

RESULTS

- Accurate detection of short spoken commands
- Fast inference suitable for real-time use
- Clear confidence scores for predictions
- Stable execution on macOS (Intel & Apple Silicon)

----------------------------------------------------------------

CURRENT LIMITATIONS

- Fixed command vocabulary
- Single-language (English)
- No streaming inference
- No background noise augmentation
- No deployment to edge devices

----------------------------------------------------------------

PLANNED IMPROVEMENTS

- Add more keywords
- Background noise robustness
- Streaming keyword detection
- On-device / edge deployment
- Quantized models for low-power devices

----------------------------------------------------------------

AUTHOR

Jasper Chinedu Nwaangere
MSc Computer Science  
Voice AI • Machine Learning • Applied AI Systems

----------------------------------------------------------------

WHY THIS PROJECT MATTERS

This project demonstrates:
- End-to-end voice command recognition
- Feature extraction for speech
- Lightweight deep-learning model design
- Turning ML training into a real web application

