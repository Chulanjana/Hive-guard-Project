import os
import time
import datetime
import requests
import librosa
import numpy as np
from dotenv import load_dotenv

from device.sensors import read_temp_hum_inside, read_temp_hum_outside
from utils.inference import load_models, run_inference
from device.mic import record_audio_and_extract_mfcc

# Load .env variables
load_dotenv()

# Load ML models
model_qp, model_qa, model_anomaly = load_models()

# Firebase Realtime Database URL
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")


# Function to extract MFCCs from audio
def extract_audio_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean


# Function to push data to Firebase
def push_to_firebase(payload):
    try:
        response = requests.patch(FIREBASE_DB_URL, json=payload)
        if response.status_code == 200:
            print("✅ Data pushed to Firebase")
        else:
            print("❌ Firebase error:", response.text)
    except Exception as e:
        print("🔥 Error pushing to Firebase:", e)


# Main loop
while True:
    # 1. Read temperature and humidity
    outside_temp, outside_hum = read_temp_hum_outside()
    inside_temp, inside_hum = read_temp_hum_inside()

    # 2. Get MFCC from mic
    # mfccs = record_audio_and_extract_mfcc()

    # 2. Extract MFCCs from audio file
    mfccs = extract_audio_features("test_trimmed.wav")

    # 3. Run inference
    results = run_inference(
        model_qp,
        model_qa,
        model_anomaly,
        outside_temp,
        outside_hum,
        inside_temp,
        inside_hum,
        mfccs,
    )

    # 5. Create payload
    payload = {
        "outside_temp": outside_temp,
        "outside_humidity": outside_hum,
        "inside_temp": inside_temp,
        "inside_humidity": inside_hum,
        "queen_presence": results["queen_presence"],
        "queen_acceptance":results["queen_acceptance"],
        "anomaly": results["anomaly"],
        "last_updated": datetime.datetime.utcnow().isoformat() + "Z",
    }

    print("Payload:", payload)

    # 6. Send to Firebase
    push_to_firebase(payload)

    # 7. Wait before next run
    time.sleep(10)
