import joblib
import numpy as np
import pandas as pd


def load_models():
    model_qp = joblib.load("models/queen_presence_model.pkl")
    model_qa = joblib.load("models/queen_acceptance_model.pkl")
    model_anomaly = joblib.load("models/bee_sound_anomaly_model.pkl")

    return model_qp, model_qa, model_anomaly


def run_inference(
    model_qp, model_qa, model_anomaly, w_temp, w_hum, h_temp, h_hum, mfccs
):
    # Combined Feature Vector
    all_features = np.concatenate(([h_temp, h_hum, w_temp, w_hum], mfccs))
    all_feature_names = [
        "hive temp",
        "hive humidity",
        "weather temp",
        "weather humidity",
    ] + [f"mfcc_{i+1}" for i in range(len(mfccs))]
    features_df = pd.DataFrame([all_features], columns=all_feature_names)

    queen_presence = model_qp.predict(features_df)[0]
    anomaly = model_anomaly.predict(features_df)[0]  # -1 is anomaly

    if queen_presence == 1:
        queen_acceptance = model_qa.predict(features_df)[0]

    return {
        "queen_presence": "Yes" if queen_presence == 1 else "No",
        "queen_acceptance": (
            ("Accepted" if queen_acceptance == 2 else "Not Accepted")
            if queen_presence == 1
            else "Unavailable"
        ),
        "anomaly": "Anomaly" if anomaly == -1 else "Normal",
    }
