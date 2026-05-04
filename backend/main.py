"""
Multimodal Emotion Recognition - FastAPI Backend
Models: face_emotion_fer2013.keras | speech_emotion_ravdess_cnn.keras | text_emotion_trained.keras
Text model: 7-class GoEmotions CNN-1D with tokenizer.pkl
Models are downloaded from Google Drive at startup if not present locally.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import io
import os
import logging

# ─────────────────────────────────────────────
# DOWNLOAD MODELS FROM GOOGLE DRIVE AT STARTUP
# ─────────────────────────────────────────────

DRIVE_FILES = {
    "models/face_emotion_fer2013.keras":       "1ZjsgpX7Exd-TyAYFNx5cO9ypKPfcWrN_",
    "models/speech_emotion_ravdess_cnn.keras": "11lV_sYfUtlPo9_hwcJi5Uk42xlrzSPXP",
    "models/text_emotion_trained.keras":       "1Zs_HNfK_SUjCwlN-8qIPCF9Qg1Y4YpYF",
    "models/text_tokenizer.pkl":               "1Pf6TwN4__lRu5Vcu-mSaPG8oBeI9qy4K",
}

def download_models():
    try:
        import gdown
    except ImportError:
        raise RuntimeError("gdown not installed. Add 'gdown==5.2.0' to requirements.txt")

    os.makedirs("models", exist_ok=True)

    for path, file_id in DRIVE_FILES.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✓ Already exists, skipping: {path} ({size_mb:.1f} MB)")
            continue
        print(f"⬇  Downloading {path} ...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False, fuzzy=True)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✓  Saved: {path} ({size_mb:.1f} MB)")
        else:
            raise RuntimeError(
                f"Download failed for {path}. "
                "Make sure the file is shared as 'Anyone with the link → Viewer' on Google Drive."
            )

    print("\n✅ All models ready.\n")

# Download at startup before anything else loads
download_models()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multimodal Emotion Recognition API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# SHARED EMOTION CLASSES
# Order must match FER2013 and retrained text model exactly
# ─────────────────────────────────────────────
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# RAVDESS 8-class → 7-class mapping
RAVDESS_MAP = {
    0: "Neutral",   # neutral
    1: "Neutral",   # calm → Neutral
    2: "Happy",
    3: "Sad",
    4: "Angry",
    5: "Fear",
    6: "Disgust",
    7: "Surprise",
}

# Fusion weights based on model accuracy
MODEL_WEIGHTS = {
    "face":   0.60,
    "speech": 0.65,
    "text":   0.75,
}

# ─────────────────────────────────────────────
# LAZY MODEL LOADING
# ─────────────────────────────────────────────
_models = {}
_tokenizer = None

def get_model(name: str):
    if name not in _models:
        try:
            import tensorflow as tf
            paths = {
                "face":   "models/face_emotion_fer2013.keras",
                "speech": "models/speech_emotion_ravdess_cnn.keras",
                "text":   "models/text_emotion_trained.keras",
            }
            path = paths[name]
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            logger.info(f"Loading model: {path}")
            _models[name] = tf.keras.models.load_model(path)
            logger.info(f"Model '{name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load '{name}' model: {e}")
            raise HTTPException(status_code=500, detail=f"Model '{name}' could not be loaded: {str(e)}")
    return _models[name]


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        import pickle
        path = "models/text_tokenizer.pkl"
        if not os.path.exists(path):
            raise FileNotFoundError(
                "text_tokenizer.pkl not found in models/. "
                "Make sure you copied it from your training output."
            )
        logger.info("Loading text tokenizer...")
        with open(path, "rb") as f:
            _tokenizer = pickle.load(f)
        logger.info("Tokenizer loaded successfully.")
    return _tokenizer


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_face(image_bytes: bytes) -> np.ndarray:
    """Image bytes → (1, 48, 48, 1) normalized grayscale array."""
    import cv2
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image. Make sure it is a valid PNG/JPG.")
    img = cv2.resize(img, (48, 48))
    img = img.astype("float32") / 255.0
    return img.reshape(1, 48, 48, 1)


def preprocess_speech(audio_bytes: bytes) -> np.ndarray:
    import librosa
    import tempfile

    # Save incoming webm audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        # Load using librosa
        y, sr = librosa.load(tmp.name, sr=22050, duration=3.0)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    target_len = 130
    if mfcc.shape[1] < target_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, target_len - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :target_len]

    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

    return mfcc.reshape(1, 40, target_len, 1)


def preprocess_text(text: str) -> np.ndarray:
    """
    Text string → padded token sequence (1, 64).
    Uses the saved tokenizer from training (text_tokenizer.pkl).
    max_len=64 matches the retrained 7-class model's input shape.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tokenizer = get_tokenizer()
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=64, padding="post", truncating="post")
    return padded


# ─────────────────────────────────────────────
# OUTPUT NORMALIZATION
# ─────────────────────────────────────────────

def normalize_ravdess_output(probs: np.ndarray) -> dict:
    """Map 8-class RAVDESS probabilities → 7 normalized emotion scores."""
    scores = {e: 0.0 for e in EMOTIONS}
    for ravdess_idx, emotion in RAVDESS_MAP.items():
        if ravdess_idx < len(probs):
            scores[emotion] += float(probs[ravdess_idx])
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}
    return scores


# ─────────────────────────────────────────────
# FUSION LOGIC
# ─────────────────────────────────────────────

def majority_vote(predictions: list) -> str:
    """Return the emotion with the most first-place votes."""
    from collections import Counter
    votes = [max(p, key=p.get) for p in predictions]
    return Counter(votes).most_common(1)[0][0]


def weighted_average(predictions: list, weights: list) -> dict:
    """Weighted average of emotion probability dicts."""
    total_weight = sum(weights)
    fused = {e: 0.0 for e in EMOTIONS}
    for pred, w in zip(predictions, weights):
        for emotion in EMOTIONS:
            fused[emotion] += pred.get(emotion, 0.0) * (w / total_weight)
    return fused


def run_fusion(results: dict, method: str = "weighted") -> dict:
    """
    results : { "face": {emotion: score}, "speech": ..., "text": ... }
    method  : "weighted" | "majority"
    """
    present = {k: v for k, v in results.items() if v is not None}
    if not present:
        raise ValueError("No modality results available to fuse.")

    preds      = list(present.values())
    modalities = list(present.keys())

    if method == "majority":
        final_emotion = majority_vote(preds)
        scores = {e: (1.0 if e == final_emotion else 0.0) for e in EMOTIONS}
    else:
        weights = [MODEL_WEIGHTS[m] for m in modalities]
        scores  = weighted_average(preds, weights)
        final_emotion = max(scores, key=scores.get)

    return {
        "emotion":         final_emotion,
        "scores":          scores,
        "method":          method,
        "modalities_used": modalities,
    }


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "emotions": EMOTIONS}


@app.post("/predict/face")
async def predict_face(file: UploadFile = File(...)):
    """Accept a face image → return 7-class emotion probabilities."""
    try:
        image_bytes = await file.read()
        model  = get_model("face")
        x      = preprocess_face(image_bytes)
        probs  = model.predict(x, verbose=0)[0]
        scores = {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}
        return {
            "emotion":    EMOTIONS[int(np.argmax(probs))],
            "confidence": float(np.max(probs)),
            "scores":     scores,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/speech")
async def predict_speech(file: UploadFile = File(...)):
    """Accept a WAV/WebM audio file → return 7-class emotion probabilities."""
    try:
        audio_bytes = await file.read()
        model  = get_model("speech")
        x      = preprocess_speech(audio_bytes)
        probs  = model.predict(x, verbose=0)[0]
        scores = normalize_ravdess_output(probs)
        return {
            "emotion":    max(scores, key=scores.get),
            "confidence": float(max(scores.values())),
            "scores":     scores,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class TextRequest(BaseModel):
    text: str

@app.post("/predict/text")
async def predict_text(req: TextRequest):
    """
    Accept text → return 7-class emotion probabilities.
    Uses saved tokenizer.pkl + retrained 7-class GoEmotions CNN-1D model.
    Input shape: (1, 64) | Output shape: (1, 7)
    Class order: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
    """
    try:
        model  = get_model("text")
        x      = preprocess_text(req.text)
        probs  = model.predict(x, verbose=0)[0]   # shape (7,)
        scores = {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}
        return {
            "emotion":    max(scores, key=scores.get),
            "confidence": float(max(probs)),
            "scores":     scores,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class FuseRequest(BaseModel):
    face:   dict | None = None
    speech: dict | None = None
    text:   dict | None = None
    method: str = "weighted"   # "weighted" | "majority"

@app.post("/predict/fuse")
async def predict_fuse(req: FuseRequest):
    """Accept pre-computed scores from all modalities → return fused result."""
    try:
        results = {
            k: v for k, v in {
                "face":   req.face,
                "speech": req.speech,
                "text":   req.text,
            }.items() if v is not None
        }
        if not results:
            raise HTTPException(status_code=400, detail="At least one modality result is required.")
        return run_fusion(results, method=req.method)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
