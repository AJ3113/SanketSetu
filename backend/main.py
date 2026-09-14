from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sqlite3
import cv2
import numpy as np

from backend.ml.config import ROOT_MODEL_PATH, KERAS_MODEL_PATH
from backend.ml.inference import get_classifier

app = FastAPI(title='SanketSetu Detection API')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:3000'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
DB_PATH = Path(__file__).with_name('sanketsetu.db')

@app.on_event('startup')
def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.execute('CREATE TABLE IF NOT EXISTS detections (id INTEGER PRIMARY KEY, label TEXT, confidence REAL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)')

@app.get('/health')
def health():
    model_ready = ROOT_MODEL_PATH.exists() or KERAS_MODEL_PATH.exists()
    return {'status': 'ok', 'model_ready': model_ready, 'voice_ready': False}

@app.post('/detect')
async def detect(frame: UploadFile = File(...)):
    if not (ROOT_MODEL_PATH.exists() or KERAS_MODEL_PATH.exists()):
        return {'status': 'model_unconfigured', 'message': 'Trained model.keras not found.'}

    contents = await frame.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {'status': 'error', 'message': 'Failed to decode image frame.'}

    classifier = get_classifier()
    res = classifier.predict_frame(img)

    if res['label'] != 'UNKNOWN':
        try:
            with sqlite3.connect(DB_PATH) as db:
                db.execute('INSERT INTO detections (label, confidence) VALUES (?, ?)', (res['label'], res['confidence']))
        except Exception:
            pass

    return {
        'status': 'ok',
        'label': res['label'],
        'confidence': res['confidence'],
        'hand_detected': res['hand_detected'],
        'probabilities': res['probabilities'],
    }

