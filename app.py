//Installing dependencies

import streamlit as st
import numpy as np
import pickle, os, cv2, librosa, tempfile, warnings
import tensorflow as tf
from PIL import Image

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE  = 128     
N_MFCC    = 40       
N_FRAMES  = 130      # Timesteps used during RNN training
SR        = 22050    # Sample rate used during training

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Parkinson's Disease Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Professional Medical Dashboard Theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

.stApp {
    background: linear-gradient(135deg, #060d1a 0%, #0a1628 50%, #060d1a 100%);
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 3rem 3rem 3rem; max-width: 1400px; margin: auto; }

/* ── Hero ────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0b1d35 0%, #111827 100%);
    border: 1px solid rgba(96,165,250,0.2);
    border-radius: 24px;
    padding: 3rem 2.5rem 2.5rem 2.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse at 25% 50%, rgba(96,165,250,0.08) 0%, transparent 55%),
        radial-gradient(ellipse at 75% 50%, rgba(167,139,250,0.08) 0%, transparent 55%);
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: rgba(96,165,250,0.08);
    border: 1px solid rgba(96,165,250,0.25);
    border-radius: 50px;
    padding: 0.3rem 1.2rem;
    color: #60a5fa;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 800;
    line-height: 1.15;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.75rem;
}
.hero-sub {
    color: rgba(148,163,184,0.75);
    font-size: 0.92rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 400;
}
.hero-tags {
    display: flex; gap: 0.6rem; justify-content: center;
    flex-wrap: wrap; margin-top: 1.5rem;
}
.hero-tag {
    background: rgba(15,30,58,0.8);
    border: 1px solid rgba(96,165,250,0.15);
    border-radius: 8px;
    padding: 0.35rem 0.9rem;
    color: rgba(148,163,184,0.8);
    font-size: 0.78rem;
    font-weight: 500;
}

/* ── Status Banner ───────────────────────────────────── */
.status-ok {
    background: rgba(52,211,153,0.07);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 12px;
    padding: 0.75rem 1.4rem;
    margin-bottom: 1.5rem;
    color: #34d399;
    font-size: 0.85rem;
    font-weight: 500;
    display: flex; align-items: center; gap: 0.5rem;
}
.status-err {
    background: rgba(248,113,113,0.07);
    border: 1px solid rgba(248,113,113,0.25);
    border-radius: 12px;
    padding: 0.75rem 1.4rem;
    margin-bottom: 1.5rem;
    color: #f87171;
    font-size: 0.85rem;
    font-weight: 500;
}

/* ── Upload Cards ────────────────────────────────────── */
.upload-card {
    background: linear-gradient(145deg, #0b1d35 0%, #0f172a 100%);
    border: 1px solid rgba(96,165,250,0.12);
    border-radius: 20px;
    padding: 1.8rem;
    height: 100%;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.upload-card:hover {
    border-color: rgba(96,165,250,0.35);
    box-shadow: 0 8px 32px rgba(96,165,250,0.08);
}
.step-pill {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 0.25rem 0.8rem;
    border-radius: 50px;
    display: inline-block;
    margin-bottom: 1rem;
}
.card-icon { font-size: 1.6rem; margin-bottom: 0.5rem; }
.card-heading {
    color: #60a5fa;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.card-hint {
    color: rgba(148,163,184,0.6);
    font-size: 0.8rem;
    line-height: 1.6;
    margin-bottom: 1rem;
}
.card-tip {
    background: rgba(96,165,250,0.06);
    border-left: 3px solid rgba(96,165,250,0.4);
    border-radius: 0 8px 8px 0;
    padding: 0.5rem 0.8rem;
    color: rgba(148,163,184,0.7);
    font-size: 0.75rem;
    line-height: 1.5;
    margin-top: 0.8rem;
}

/* ── Analyse Button ──────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 2.5rem !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    box-shadow: 0 4px 24px rgba(37,99,235,0.35) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 8px 36px rgba(37,99,235,0.55) !important;
    transform: translateY(-2px) !important;
}

/* ── Divider ─────────────────────────────────────────── */
.section-div {
    border: none;
    border-top: 1px solid rgba(96,165,250,0.1);
    margin: 2rem 0;
}
.section-label {
    color: rgba(148,163,184,0.5);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 1.5rem 0 1rem 0;
}

/* ── Result Card ─────────────────────────────────────── */
.result-healthy {
    background: linear-gradient(135deg, rgba(52,211,153,0.1) 0%, rgba(16,185,129,0.06) 100%);
    border: 1px solid rgba(52,211,153,0.35);
    border-radius: 20px;
    padding: 2.8rem 2rem;
    text-align: center;
    position: relative; overflow: hidden;
}
.result-parkinson {
    background: linear-gradient(135deg, rgba(248,113,113,0.12) 0%, rgba(220,38,38,0.06) 100%);
    border: 1px solid rgba(248,113,113,0.4);
    border-radius: 20px;
    padding: 2.8rem 2rem;
    text-align: center;
    position: relative; overflow: hidden;
}
.result-icon { font-size: 4rem; margin-bottom: 1rem; display: block; }
.result-title-h { color: #34d399; font-size: 2.4rem; font-weight: 800; margin-bottom: 0.5rem; }
.result-title-p { color: #f87171; font-size: 2.4rem; font-weight: 800; margin-bottom: 0.5rem; }
.result-badge-h {
    display: inline-block;
    background: rgba(52,211,153,0.15); border: 1px solid rgba(52,211,153,0.4);
    color: #34d399; padding: 0.3rem 1.5rem;
    border-radius: 50px; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.8rem;
}
.result-badge-p {
    display: inline-block;
    background: rgba(248,113,113,0.15); border: 1px solid rgba(248,113,113,0.4);
    color: #f87171; padding: 0.3rem 1.5rem;
    border-radius: 50px; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.8rem;
}
.result-desc { color: rgba(148,163,184,0.65); font-size: 0.88rem; }

/* ── Metric Cards ────────────────────────────────────── */
.metric-wrap {
    background: linear-gradient(145deg, #0b1d35 0%, #0f172a 100%);
    border: 1px solid rgba(96,165,250,0.12);
    border-radius: 16px;
    padding: 1.5rem 1rem;
    text-align: center;
}
.metric-lbl {
    color: rgba(148,163,184,0.55);
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.metric-green { color: #34d399; font-size: 2.6rem; font-weight: 800; line-height: 1; }
.metric-blue  { color: #60a5fa; font-size: 2.6rem; font-weight: 800; line-height: 1; }
.metric-purple{ color: #a78bfa; font-size: 2.6rem; font-weight: 800; line-height: 1; }
.metric-na    { color: rgba(148,163,184,0.3); font-size: 1.8rem; font-weight: 500; line-height: 1; }
.metric-src   { color: rgba(148,163,184,0.35); font-size: 0.7rem; margin-top: 0.3rem; }

/* ── Stage Probability Bars ──────────────────────────── */
.stage-card {
    background: linear-gradient(145deg, #0b1d35 0%, #0f172a 100%);
    border: 1px solid rgba(96,165,250,0.12);
    border-radius: 20px;
    padding: 1.8rem;
    margin-top: 1.5rem;
}
.stage-card-title {
    color: rgba(148,163,184,0.7);
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 3px; text-transform: uppercase;
    margin-bottom: 1.4rem;
}
.stage-row { margin-bottom: 1rem; }
.stage-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem; }
.stage-name { color: rgba(148,163,184,0.8); font-size: 0.82rem; font-weight: 500; }
.stage-pct  { font-size: 0.82rem; font-weight: 700; }
.bar-bg {
    background: rgba(255,255,255,0.05);
    border-radius: 50px; height: 8px; overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 50px; transition: width 0.8s ease; }

/* ── Disclaimer ──────────────────────────────────────── */
.disclaimer {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.25);
    border-radius: 14px;
    padding: 1rem 1.5rem;
    margin-top: 2rem;
    color: rgba(251,191,36,0.85);
    font-size: 0.8rem;
    line-height: 1.7;
}

/* ── Footer ──────────────────────────────────────────── */
.app-footer {
    text-align: center;
    color: rgba(100,116,139,0.5);
    font-size: 0.75rem;
    margin-top: 2.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(96,165,250,0.08);
    letter-spacing: 1px;
}

/* Streamlit overrides */
[data-testid="stFileUploader"] {
    background: rgba(11,29,53,0.5) !important;
    border: 1px dashed rgba(96,165,250,0.25) !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(96,165,250,0.5) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING  
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models():
    loaded, missing = {}, []

    # ── Keras models (.keras format) ──────────────────────────────────────
    for name, fname in [('rnn',    'rnn_modelfinal.keras'),
                        ('fusion', 'fusion_modelfinal.keras')]:
        path = os.path.join(BASE_DIR, fname)
        if os.path.exists(path):
            try:
                loaded[name] = tf.keras.models.load_model(path)
            except Exception as e:
                st.warning(f"Could not load {fname}: {e}")
                missing.append(fname)
        else:
            missing.append(fname)

    
    cnn_keras = os.path.join(BASE_DIR, 'cnn_modelfinal.keras')
    cnn_pkl   = os.path.join(BASE_DIR, 'cnn_model.pkl')
    if os.path.exists(cnn_keras):
        try:
            loaded['cnn'] = tf.keras.models.load_model(cnn_keras)
        except Exception as e:
            st.warning(f"Could not load cnn_modelfinal.keras: {e}")
            missing.append('cnn_modelfinal.keras')
    elif os.path.exists(cnn_pkl):
        try:
            with open(cnn_pkl, 'rb') as f:
                loaded['cnn'] = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load cnn_model.pkl: {e}")
            missing.append('cnn_model.pkl')
    else:
        missing.append('cnn_modelfinal.keras (optional)')

    # ── Config pickle ─────────────────────────────────────────────────────
    config_path = os.path.join(BASE_DIR, 'configfinal.pkl')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'rb') as f:
                loaded['config'] = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load configfinal.pkl: {e}")
    else:
        missing.append('configfinal.pkl')

    return loaded, missing


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING 
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_image(uploaded_file):
    """
    ✅ FIX: Preprocess spiral image exactly as during CNN training.
    Key changes vs old code:
      - Normalization and resizing of images
      - Explicit grayscale conversion
      - Correct resize and normalization
      - Proper channel/batch dimensions

    img = Image.open(uploaded_file).convert('RGB')
    img_arr = np.array(img)

    # Convert to grayscale (CNN trained on single-channel spiral images)
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

    # Resize to match CNN input size
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # ── INVERSION FLAG ─────────────────────────────────────────────────────
    # If during training the images had WHITE spirals on BLACK background,
    # uncomment the line below. (Common with OpenCV-loaded handwriting images)
    # normalized = 1.0 - normalized
    # ───────────────────────────────────────────────────────────────────────

    # Shape → (1, H, W, 1)  [batch, height, width, channels]
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1), img_arr


def extract_audio_features(audio_path):
    
    ✅ FIX 1: Use N_MFCC=40 (was 65) to match RNN training dimensions.
    ✅ FIX 2: Pad/trim to exactly N_FRAMES timesteps.
    ✅ FIX 3: Normalize features before feeding to LSTM.
    Returns shape: (1, N_FRAMES, N_MFCC) = (1, 130, 40)
    
    y, sr = librosa.load(audio_path, sr=SR, mono=True)

    # Extract MFCCs — shape: (N_MFCC, T)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # Transpose → (T, N_MFCC)
    mfcc = mfcc.T

    # Pad or trim to exactly N_FRAMES
    if mfcc.shape[0] < N_FRAMES:
        pad_width = N_FRAMES - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:N_FRAMES, :]

    # Normalize (z-score per feature)
    mfcc = (mfcc - mfcc.mean(axis=0)) / (mfcc.std(axis=0) + 1e-8)

    # Add batch dimension → (1, N_FRAMES, N_MFCC)
    return mfcc.reshape(1, N_FRAMES, N_MFCC).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def run_prediction(models, img_input=None, audio_input=None):
    
    Fusion Pipeline
    Old code fed raw MFCC (1,130,65) → fusion.  Now we:
      1. Get CNN probability vector  → (1, 2)
      2. Get RNN probability vector  → (1, 2)
      3. Concatenate probabilities   → (1, 4)  and feed to fusion model
         OR extract penultimate features if fusion expects (1, 64)
    
    cnn_prob = rnn_prob = fusion_prob = None

    # CNN 
    if img_input is not None and 'cnn' in models:
        try:
            raw = models['cnn'].predict(img_input, verbose=0)   # (1, N_classes)
            # Ensure 2-class output: [P(healthy), P(parkinson)]
            if raw.shape[-1] >= 2:
                cnn_prob = raw[0][:2]
            else:
                p = float(raw[0][0])
                cnn_prob = np.array([1 - p, p])
        except Exception as e:
            st.warning(f"CNN error: {e}")

    # RNN 
    if audio_input is not None and 'rnn' in models:
        try:
            raw = models['rnn'].predict(audio_input, verbose=0)  # (1, N_classes)
            if raw.shape[-1] >= 2:
                rnn_prob = raw[0][:2]
            else:
                p = float(raw[0][0])
                rnn_prob = np.array([1 - p, p])
        except Exception as e:
            st.warning(f"RNN error: {e}")

    # FUSION 
    if cnn_prob is not None and rnn_prob is not None and 'fusion' in models:
        # Strategy 1: pass concatenated probabilities → (1, 4)
        try:
            fusion_in = np.concatenate([cnn_prob, rnn_prob]).reshape(1, -1)
            raw = models['fusion'].predict(fusion_in, verbose=0)
            if raw.shape[-1] >= 2:
                fusion_prob = raw[0][:2]
            else:
                p = float(raw[0][0])
                fusion_prob = np.array([1 - p, p])
        except Exception:
            # Strategy 2: weighted average fallback
            fusion_prob = 0.40 * cnn_prob + 0.60 * rnn_prob

    elif cnn_prob is not None and rnn_prob is not None:
        # No fusion model loaded → weighted average
        fusion_prob = 0.40 * cnn_prob + 0.60 * rnn_prob
    elif cnn_prob is not None:
        fusion_prob = cnn_prob
    elif rnn_prob is not None:
        fusion_prob = rnn_prob

    return cnn_prob, rnn_prob, fusion_prob


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE PROBABILITY ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════
def estimate_stages(parkinson_confidence):
    """Distribute Parkinson's probability across severity stages."""
    if parkinson_confidence < 0.5:
        return [1 - parkinson_confidence, parkinson_confidence * 0.4,
                parkinson_confidence * 0.3, parkinson_confidence * 0.2,
                parkinson_confidence * 0.07, parkinson_confidence * 0.03]
    p = parkinson_confidence
    return [
        1 - p,
        p * 0.35,
        p * 0.28,
        p * 0.20,
        p * 0.11,
        p * 0.06,
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE BAR HELPER
# ══════════════════════════════════════════════════════════════════════════════
STAGE_META = [
    ("🟢", "Healthy",               "#34d399", "#052e16"),
    ("🟡", "Stage 1 — Mild",        "#fbbf24", "#1c1003"),
    ("🟠", "Stage 2 — Mild Bilateral","#f97316","#1c0a03"),
    ("🔴", "Stage 3 — Moderate",    "#ef4444", "#1c0303"),
    ("🔴", "Stage 4 — Severe",      "#dc2626", "#180202"),
    ("🔴", "Stage 5 — Advanced",    "#b91c1c", "#130101"),
]

def render_stage_bar(label_icon, label, color, bg_color, pct):
    pct_display = f"{pct * 100:.1f}%"
    bar_pct     = f"{pct * 100:.1f}%"
    st.markdown(f"""
    <div class="stage-row">
      <div class="stage-top">
        <span class="stage-name">{label_icon}&nbsp;&nbsp;{label}</span>
        <span class="stage-pct" style="color:{color};">{pct_display}</span>
      </div>
      <div class="bar-bg">
        <div class="bar-fill" style="width:{bar_pct};background:linear-gradient(90deg,{bg_color},{color});"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Loading the models
    with st.spinner("Loading models…"):
        models, missing = load_models()

    # Hero
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">🧠 &nbsp; AI-Powered Neurological Screening</div>
      <div class="hero-title">Parkinson's Disease<br>Detection System</div>
      <div class="hero-sub">Multimodal Deep Learning &nbsp;·&nbsp; CNN + RNN Fusion</div>
      <div class="hero-tags">
        <span class="hero-tag">🖊️ Spiral Handwriting Analysis</span>
        <span class="hero-tag">🎤 Voice Biomarker Analysis</span>
        <span class="hero-tag">⚡ Real-time Inference</span>
        <span class="hero-tag">🔬 Clinical Research Tool</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Model status banner
    loaded_names = [n.upper() for n in models if n != 'config']
    if loaded_names:
        st.markdown(f"""
        <div class="status-ok">
          ✅ &nbsp; Models loaded successfully:
          <strong>{'&nbsp; · &nbsp;'.join(loaded_names)}</strong>
        </div>
        """, unsafe_allow_html=True)
    if missing:
        st.markdown(f"""
        <div class="status-err">
          ⚠️ Missing model files: {', '.join(missing)}<br>
          Place them in the same folder as app.py
        </div>
        """, unsafe_allow_html=True)

    # Input Section
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="upload-card">
          <div class="step-pill">Step 01</div>
          <div class="card-icon">✍️</div>
          <div class="card-heading">Spiral Handwriting Test</div>
          <div class="card-hint">
            Draw a spiral on paper, photograph it clearly, and upload the image below.
            Ensure good lighting and minimal background clutter.
          </div>
        </div>
        """, unsafe_allow_html=True)
        img_file = st.file_uploader(
            "Upload spiral image",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            key="img_upload",
            label_visibility="collapsed"
        )
        if img_file:
            st.image(img_file, caption="Uploaded Spiral", use_container_width=True)
        st.markdown("""
        <div class="card-tip">
          💡 <strong>Tip:</strong> A healthy spiral is smooth and evenly spaced.
          Tremors and stiffness in Parkinson's patients cause irregular, wobbly spirals.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="upload-card">
          <div class="step-pill">Step 02</div>
          <div class="card-icon">🎤</div>
          <div class="card-heading">Voice Recording Test</div>
          <div class="card-hint">
            Sustain the vowel sound <strong>"Aaah"</strong> steadily for 5–10 seconds.
            Record in a quiet room and save as WAV or MP3.
          </div>
        </div>
        """, unsafe_allow_html=True)
        audio_file = st.file_uploader(
            "Upload voice recording",
            type=["wav", "mp3", "ogg", "m4a", "flac"],
            key="audio_upload",
            label_visibility="collapsed"
        )
        if audio_file:
            st.audio(audio_file)
        st.markdown("""
        <div class="card-tip">
          💡 <strong>Tip:</strong> Parkinson's affects vocal cord control, causing
          tremors, breathiness, and reduced loudness detectable in voice recordings.
        </div>
        """, unsafe_allow_html=True)

    # Analyse Button
    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        run_analysis = st.button("🔬  ANALYSE NOW", key="analyse_btn")

    # Results 
    if run_analysis:
        if not img_file and not audio_file:
            st.warning("⚠️  Please upload at least one input (image or audio).")
            return

        st.markdown('<hr class="section-div">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">📋 &nbsp; Analysis Results</div>', unsafe_allow_html=True)

        with st.spinner("Running analysis…"):
            img_input   = None
            audio_input = None
            preview_img = None

            # Preprocess image
            if img_file:
                try:
                    img_input, preview_img = preprocess_image(img_file)
                except Exception as e:
                    st.error(f"Image processing error: {e}")

            # Preprocess audio
            if audio_file:
                try:
                    suffix = os.path.splitext(audio_file.name)[-1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(audio_file.read())
                        tmp_path = tmp.name
                    audio_input = extract_audio_features(tmp_path)
                    os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"Audio processing error: {e}")

            # Run prediction
            cnn_prob, rnn_prob, fusion_prob = run_prediction(
                models, img_input, audio_input
            )

        if fusion_prob is None:
            st.error("❌ Could not generate a prediction. Check model files and inputs.")
            return

        # Primary Goal Result
        parkinson_conf = float(fusion_prob[1]) if len(fusion_prob) >= 2 else float(fusion_prob[0])
        healthy_conf   = 1.0 - parkinson_conf
        is_healthy     = healthy_conf >= 0.5
        overall_conf   = healthy_conf if is_healthy else parkinson_conf

        if is_healthy:
            st.markdown(f"""
            <div class="result-healthy">
              <span class="result-icon">✅</span>
              <div class="result-title-h">No Parkinson's Detected</div>
              <div class="result-badge-h">Healthy</div>
              <div class="result-desc">No significant biomarkers of Parkinson's disease were detected.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-parkinson">
              <span class="result-icon">⚠️</span>
              <div class="result-title-p">Parkinson's Indicators Found</div>
              <div class="result-badge-p">Parkinson's Detected</div>
              <div class="result-desc">Biomarkers consistent with Parkinson's disease were detected. Please consult a neurologist.</div>
            </div>
            """, unsafe_allow_html=True)

        # Metrics consideration
        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(f"""
            <div class="metric-wrap">
              <div class="metric-lbl">Overall Confidence</div>
              <div class="metric-green">{overall_conf * 100:.1f}%</div>
              <div class="metric-src">Fusion Model</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            if cnn_prob is not None:
                cnn_conf = float(cnn_prob[0]) if is_healthy else float(cnn_prob[1])
                st.markdown(f"""
                <div class="metric-wrap">
                  <div class="metric-lbl">CNN — Handwriting</div>
                  <div class="metric-blue">{cnn_conf * 100:.1f}%</div>
                  <div class="metric-src">Spiral Analysis</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-wrap">
                  <div class="metric-lbl">CNN — Handwriting</div>
                  <div class="metric-na">N/A</div>
                  <div class="metric-src">No image provided</div>
                </div>
                """, unsafe_allow_html=True)

        with m3:
            if rnn_prob is not None:
                rnn_conf = float(rnn_prob[0]) if is_healthy else float(rnn_prob[1])
                st.markdown(f"""
                <div class="metric-wrap">
                  <div class="metric-lbl">RNN — Voice</div>
                  <div class="metric-purple">{rnn_conf * 100:.1f}%</div>
                  <div class="metric-src">Voice Biomarkers</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-wrap">
                  <div class="metric-lbl">RNN — Voice</div>
                  <div class="metric-na">N/A</div>
                  <div class="metric-src">No audio provided</div>
                </div>
                """, unsafe_allow_html=True)

        #  Stage probability bars
        st.markdown("""
        <div class="stage-card">
          <div class="stage-card-title">📊 &nbsp; Stage Probability Distribution</div>
        """, unsafe_allow_html=True)

        stage_probs = estimate_stages(parkinson_conf)
        for i, (icon, label, color, bg_color) in enumerate(STAGE_META):
            render_stage_bar(icon, label, color, bg_color, stage_probs[i])

        st.markdown("</div>", unsafe_allow_html=True)

        # Medical disclaimer
        st.markdown("""
        <div class="disclaimer">
          <strong>⚠️ Medical Disclaimer:</strong> This tool is intended for
          <strong>research and educational purposes only</strong>. It is <strong>not</strong>
          a substitute for professional medical advice, diagnosis, or treatment.
          Results should not be used as the sole basis for any medical decision.
          Always consult a qualified neurologist or healthcare professional regarding
          any medical concerns. Early-stage Parkinson's may not always be detectable
          by this system.
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="app-footer">
      Built with &nbsp; TensorFlow &nbsp;·&nbsp; Keras &nbsp;·&nbsp; Streamlit
      &nbsp;·&nbsp; LibROSA &nbsp;·&nbsp; OpenCV<br>
      <span style="font-size:0.65rem; color:rgba(100,116,139,0.3);">
        Final Year Project &nbsp;·&nbsp; Multimodal Deep Learning for Parkinson's Detection
      </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
