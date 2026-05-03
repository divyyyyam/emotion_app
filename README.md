# Multimodal Emotion Recognition — Setup Guide

## Project Structure

```
emotion_app/
├── backend/
│   ├── main.py               ← FastAPI server
│   ├── requirements.txt      ← Python dependencies
│   └── models/               ← ⚠️ PUT YOUR .keras FILES HERE
│       ├── face_emotion_fer2013.keras
│       ├── speech_emotion_ravdess_cnn.keras
│       └── text_emotion_trained.keras
└── frontend/
    └── index.html            ← Open this in browser
```

---

## Step 1 — Place your model files

Copy your three `.keras` files into `backend/models/`:

```
backend/models/face_emotion_fer2013.keras
backend/models/speech_emotion_ravdess_cnn.keras
backend/models/text_emotion_trained.keras
```

If your text model used a Keras tokenizer, also place it here:
```
backend/models/text_tokenizer.pkl
```

---

## Step 2 — Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

> Note: TensorFlow installation may take a few minutes.
> If you're on Apple Silicon (M1/M2), use `tensorflow-macos` instead.

---

## Step 3 — Start the backend

```bash
cd backend
python main.py
```

The API runs at: **http://localhost:8000**

Test it's working:
```bash
curl http://localhost:8000/health
```

You should see:
```json
{"status":"ok","emotions":["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]}
```

---

## Step 4 — Open the frontend

Simply open `frontend/index.html` in your browser.

> No web server needed — it's plain HTML/JS.

---

## API Endpoints

| Method | Endpoint | Input | Returns |
|--------|----------|-------|---------|
| GET | `/health` | — | Status + emotion classes |
| POST | `/predict/face` | image file (multipart) | emotion + scores |
| POST | `/predict/speech` | audio file (multipart) | emotion + scores |
| POST | `/predict/text` | `{"text": "..."}` JSON | emotion + scores |
| POST | `/predict/fuse` | scores dict + method | fused emotion |

### Example: Fusion request
```json
POST /predict/fuse
{
  "face":   {"Happy": 0.72, "Sad": 0.05, ...},
  "speech": {"Happy": 0.60, "Sad": 0.15, ...},
  "text":   {"Happy": 0.80, "Sad": 0.08, ...},
  "method": "weighted"
}
```

---

## Fusion Methods

### Weighted Average (recommended)
Weights are set based on model accuracy:
- Face model:   0.60 (60% accuracy on FER2013)
- Speech model: 0.65 (65% accuracy on RAVDESS)
- Text model:   0.75 (higher accuracy assumed)

### Majority Vote
Each modality votes for its top emotion; the most common wins.

---

## Preprocessing Details

| Modality | Processing |
|----------|-----------|
| Face | Resize to 48×48, grayscale, normalize to [0,1], shape: (1,48,48,1) |
| Speech | librosa MFCC (40 coefficients, 130 frames), z-score normalized, shape: (1,40,130,1) |
| Text | Tokenized via saved tokenizer.pkl → padded sequences, maxlen=100 |

---

## Class Mapping

RAVDESS has 8 classes; FER2013 and text models use 7.
"Calm" from RAVDESS is mapped to "Neutral" automatically.

Final 7 classes: **Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**

---

## Troubleshooting

**"Model could not be loaded"**
→ Check the `.keras` file is in `backend/models/` with the exact filename.

**"Camera not starting"**
→ Browser requires HTTPS or localhost for camera access. Use Chrome/Firefox.

**"Audio error: no MFCC"**
→ `librosa` may need `soundfile` or `ffmpeg`. Run: `pip install soundfile`

**CORS error in browser console**
→ Backend already has CORS enabled for all origins. Make sure backend is running.

---

## Deployment (Optional)

### HuggingFace Spaces
1. Create a Space with `gradio` or `fastapi` runtime
2. Upload `main.py`, `requirements.txt`, and your model files
3. Update `API` constant in `index.html` to your Space URL

### Local network (show to others)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
Then access via your local IP: `http://192.168.x.x:8000`
