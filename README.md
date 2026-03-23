# parkinson_detection
🧠 Deep learning system for Parkinson's Disease detection &amp; stage classification using CNN (spiral handwriting analysis) + RNN (voice biomarkers). Upload your spiral drawing and voice recording to get instant results via an interactive Streamlit web app.


# 🧠 Parkinson's Disease Detection & Stage Classification

A multimodal machine learning application that detects Parkinson's Disease and classifies its stage of progression using twoinputs — spiral handwriting images and voice recordings. Built with CNN, RNN, and deployed as an interactive Streamlit web app.

📌 Table of Contents

- Overview
- How It Works
- Datasets
- Model Architecture
- Stages of Parkinson's
- Project Structure
- Results
- Installation
- Usage
- Streamlit App
- Contributing
- License

🔍 Overview

Parkinson's Disease (PD) is a progressive neurological disorder that affects motor 
control and vocal ability. Two of its earliest and most measurable symptoms are:

- Handwriting degradation — especially in spiral drawing tasks (micrographia)
-  tremors — measurable changes in vocal frequency and stability

This project uses two specialized machine learning models one per modality — 
and fuses their predictions to detect PD and classify the patient's current stage 
(1 through 5 on the Hoehn & Yahr scale).

⚙️ How It Works

User Input
    │
    ├── 🖊️ Spiral Handwriting Image
    │         └──► CNN Model ──────────────┐
    │                                      ▼
    └── 🎙️ Voice Recording               Fusion Layer
              └──► RNN Model ─────────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │  Parkinson's Detected?  │
                              │  Yes / No               │
                              │  Stage: 0–5 (H&Y Scale) │
                              └────────────────────────┘

1. User uploads a spiral drawing (image) via the Streamlit interface
2. User uploads or records a voice sample (audio)
3. CNN processes the image for motor tremor patterns
4. RNN processes voice features (MFCC, jitter, shimmer, etc.) for vocal biomarkers
5. A fusion layer combines both predictions
6. The app displays detection result + Parkinson's stage with confidence score


📊 Datasets

🖊️ Handwriting — Spiral Drawing
  Source:[UCSB HandPD Dataset](http://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/) 
  and [Kaggle Parkinson's Spiral Dataset](https://www.kaggle.com/datasets/kmader/parkinsons-drawings)
  Classes:Healthy vs Parkinson's (spiral and wave drawings)
  Input:Grayscale spiral images resized to 128×128

🎙️ Voice Parameters
- Source:[UCI Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons) + 
  [mPower Dataset](https://www.synapse.org/#!Synapse:syn4993293/wiki/)
- Features extracted:
  - MFCC (Mel-Frequency Cepstral Coefficients)
  - Jitter & Shimmer (vocal perturbation)
  - HNR — Harmonics-to-Noise Ratio
  - RPDE, DFA, PPE — nonlinear dynamical features
    
🏗️ Model Architecture

 🖼️ CNN — Spiral Handwriting Analysis

Input Image (128×128×1)
    → Conv2D(32) + ReLU + MaxPooling
    → Conv2D(64) + ReLU + MaxPooling
    → Conv2D(128) + ReLU + MaxPooling
    → Flatten
    → Dense(256) + Dropout(0.5)
    → Dense(128)
    → Output (Stage Probabilities)

🔊 RNN — Voice Biomarker Analysis

Input Voice Features (timesteps × features)
    → LSTM(128) → Dropout(0.3)
    → LSTM(64)  → Dropout(0.3)
    → Dense(64)
    → Output (Stage Probabilities)


🔗 Fusion Layer

CNN Output + RNN Output
    → Concatenate
    → Dense(64) + ReLU
    → Dense(32)
    → Softmax → Final Stage Classification (0–5)

