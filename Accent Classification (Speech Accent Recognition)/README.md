# Accent Classification (Speech Accent Recognition)
Accent Detection System using FastAPI, Streamlit & wav2vec2


Given a short audio clip (3–8 seconds), the system outputs:
- Predicted accent
- Confidence score
- Top-3 accent probabilities

## FEATURES

- Accent prediction from short speech clips
- Microphone recording or audio file upload
- Top-3 predictions with confidence scores
- Transfer learning using wav2vec2 embeddings
- Lightweight classifier head
- REST API + Web UI separation
- CPU-friendly inference
- Easily extensible to new accents

## PROBLEM

Accent variation significantly impacts speech systems in:
- Call centers
- Hiring platforms
- Speech personalization
- ASR accuracy

Many systems fail to account for accent diversity, leading to bias and reduced performance.

Accent classification provides valuable insights while enabling fairer and more adaptive speech technologies.

## SOLUTION

This project implements a **modern accent recognition pipeline** using transfer learning:

- Pretrained wav2vec2 model for speech representation
- A custom classification head trained on accent labels
- A FastAPI backend for inference
- A Streamlit frontend for real user interaction

The system balances **performance, interpretability, and simplicity**.


## ARCHITECTURE

User (Browser)  
→ Streamlit Frontend  
→ FastAPI Backend  
→ Audio Preprocessing (16kHz mono)  
→ wav2vec2 Feature Extractor  
→ Accent Classifier  
→ Accent Label + Confidence Scores

## MODEL DESIGN

Base model:
- facebook/wav2vec2-base (pretrained)

Approach:
- Freeze wav2vec2 backbone
- Mean-pool hidden states
- Train a small neural network classifier on top

This approach:
- Requires little data
- Trains quickly
- Produces strong results on limited datasets

## DATASET STRUCTURE

Audio files are organized by accent label:

```text
data/raw/
├── british/
├── american/
├── nigerian/
├── indian/
└── australian/
```

Each folder contains short audio clips (3–8 seconds).

The training manifest is generated automatically.

## TECH STACK

Language: Python 3.10  
Frontend: Streamlit  
Backend: FastAPI  
Speech Model: wav2vec2  
Deep Learning: PyTorch  
Audio: Torchaudio, SoundFile, pydub  
ML Tools: scikit-learn, pandas  
Runtime: CPU

## TRAIN THE MODEL

### Create environment
```text
conda create -n voice_lab python=3.10 -y  
conda activate voice_lab  
```
### Install dependencies

```text
pip install -r requirements.txt  
```
### Prepare dataset
```text
python scripts/prepare_data.py  
```
### Train accent classifier
```text
python scripts/train.py  
```
**Generated files:**

models/accent_head.pth  
models/labels.json  
models/config.json  

## RUN THE APPLICATION

Start FastAPI backend (port 8006)
```text
export KMP_DUPLICATE_LIB_OK=TRUE  
python -m uvicorn api.main:app --host 127.0.0.1 --port 8006  
```
**API endpoints**

http://127.0.0.1:8006/health  
http://127.0.0.1:8006/docs  

**Start Streamlit frontend**
```text
streamlit run app/Home.py  
```
**Web interface**

http://localhost:8501

## OUTPUT EXAMPLE

Prediction: British  
Confidence: 0.83  

Top-3 predictions:
- British (0.83)
- American (0.11)
- Nigerian (0.06)

## CURRENT LIMITATIONS

- Small training dataset
- Limited accent categories
- No language detection
- No bias evaluation metrics

## PLANNED IMPROVEMENTS

- Larger multilingual datasets
- Bias and fairness evaluation
- Accent-aware ASR integration
- Streaming inference
- Edge deployment optimization


## AUTHOR

Jasper Chinedu (LordSparo)  
MSc Computer Science  
Voice AI • Machine Learning • Applied AI Systems


## WHY THIS PROJECT MATTERS

This project demonstrates:
- Transfer learning for speech
- Accent-aware modeling
- Ethical considerations in voice AI
- End-to-end ML system design

This is a **production-style accent classification system**, not a notebook experiment.
