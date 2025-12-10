# streamlit_app.py (fixed) 
import streamlit as st
import joblib, numpy as np, time, re
from pathlib import Path
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer

# ensure vader is available
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Sentiment Demo", layout="wide")
st.title("Twitter Sentiment Analysis")

MODEL_DIR = Path("models")
EMB_PATH = MODEL_DIR/"embedder_quick.joblib"
LGB_PATH = MODEL_DIR/"lgbm_model_quick.joblib"

label_map = {0:"Negative", 1:"Neutral", 2:"Positive"}

def clean_text(s: str) -> str:
    s = str(s)
    s = re.sub(r'http\S+|www\.\S+',' ', s)
    s = re.sub(r'@\w+',' ', s)
    s = re.sub(r'#','', s)
    s = re.sub(r'\s+',' ', s).strip()
    return s

def vader_score(s: str) -> float:
    return float(sia.polarity_scores(str(s))["compound"])

# Load models once and cache in session_state
if "models_loaded" not in st.session_state:
    st.session_state["models_loaded"] = False

if st.button("Load models"):
    try:
        emb = joblib.load(EMB_PATH)
        lgb = joblib.load(LGB_PATH)
        st.session_state["emb"] = emb
        st.session_state["lgb"] = lgb
        st.session_state["models_loaded"] = True
        st.success("Loaded SBERT + LightGBM (quick)")
    except Exception as e:
        st.error("Model load failed: " + str(e))

text = st.text_area("Enter tweet text:", height=120, placeholder="Type or paste tweet here...")

col1, col2 = st.columns(2)
with col1:
    if st.button("Analyze") and text.strip():
        if not st.session_state.get("models_loaded", False):
            st.warning("Press Load models first")
        else:
            emb = st.session_state["emb"]
            lgb = st.session_state["lgb"]
            cleaned = clean_text(text)
            v_score = vader_score(cleaned)
            # get embedding (1D)
            emb_vec = emb.encode([cleaned])[0]  # 1D array length 384
            # append vader as additional dimension -> 385
            vec = np.hstack([emb_vec, np.array([v_score])]).reshape(1, -1)
            # predict
            probs = lgb.predict(vec)
            if probs.ndim == 2:
                idx = int(np.argmax(probs, axis=1)[0]); conf = float(np.max(probs))
            else:
                idx = int(probs[0] > 0.5); conf = float(probs[0])
            st.subheader(f"Prediction: {label_map.get(idx,'Unknown')}")
            st.write(f"Confidence: {conf:.3f}")
            st.write(f"VADER score: {v_score:.3f}")
            st.write(f"Time: {time.time():.3f}")

with col2:
    st.markdown("### Quick samples")
    if st.button("Positive sample"):
        st.session_state['sample']= "I absolutely love this new phone!"
    if st.button("Neutral sample"):
        st.session_state['sample']= "I am going to Pune."
    if st.button("Negative sample"):
        st.session_state['sample']= "This is terrible service and I am angry."

if 'sample' in st.session_state and st.session_state['sample']:
    st.text_area("Sample text", value=st.session_state['sample'], height=80)

if st.checkbox("Show model files (models/)"):
    st.write(list(MODEL_DIR.glob("*")))
