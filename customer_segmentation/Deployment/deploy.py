import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from typing import List

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR,"kmeans_model.pkl")
scale_path = os.path.join(BASE_DIR, "scale.pkl")

# ----------------------------
# Load model and scaler
# ----------------------------
model = joblib.load(model_path)
scale = joblib.load(scale_path)

# ----------------------------
# Columns info from training
# ----------------------------
trained_columns = [
    'minutes_watched', 'clv_wins',
    'region_USA/Canada/As', 'region_West_EU',
    'channel_Friend', 'channel_Google', 'channel_Instagram',
    'channel_LinkedIn', 'channel_Other', 'channel_Twitter',
    'channel_YouTube'
]

region_cols = ['region_USA/Canada/As', 'region_West_EU']
channel_cols = [
    'channel_Friend', 'channel_Google', 'channel_Instagram',
    'channel_LinkedIn', 'channel_Other', 'channel_Twitter',
    'channel_YouTube'
]

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()

@app.post("/predict")
def predict(data: List[List]):
    """
    data: list of lists
    Each inner list: [minutes_watched, clv_wins, region, channel]
    """
    df = pd.DataFrame(data, columns=["minutes_watched","clv_wins","region","channel"])
    
    # ------------------------
    # Scaling numeric features
    # ------------------------
    numeric_cols = ["minutes_watched", "clv_wins"]
    df[numeric_cols] = scale.transform(df[numeric_cols])
    
    # ------------------------
    # Encoding categorical features
    # ------------------------
    # Region
    df_region = pd.get_dummies(df["region"]).reindex(columns=region_cols, fill_value=0)
    
    # Channel
    df_channel = pd.get_dummies(df["channel"]).reindex(columns=channel_cols, fill_value=0)
    
    # Combine all features
    df_final = pd.concat([df.drop(columns=["region","channel"]), df_region, df_channel], axis=1)
    
    # Ensure same column order as training
    df_final = df_final[trained_columns]
    
    # ------------------------
    # Prediction
    # ------------------------
    prediction = model.predict(df_final)
    
    return {"prediction": prediction.tolist()}

# # Example test prediction
# sample_data = [
#     [12.5, 300, "USA/Canada/As", "Google"]
# ]

# import pandas as pd
# df_test = pd.DataFrame(sample_data, columns=["minutes_watched","clv_wins","region","channel"])

# # Scaling + encoding (مثل ما في endpoint)
# cols_to_scale = ["minutes_watched","clv_wins"]
# df_test[cols_to_scale] = scale.transform(df_test[cols_to_scale])

# region_dummies = pd.get_dummies(df_test["region"]).reindex(columns=region_cols, fill_value=0)
# channel_dummies = pd.get_dummies(df_test["channel"]).reindex(columns=channel_cols, fill_value=0)
# df_test_final = pd.concat([df_test.drop(columns=["region","channel"]), region_dummies, channel_dummies], axis=1)
# df_test_final = df_test_final[trained_columns]

# pred = model.predict(df_test_final)
# print("Test prediction:", pred)
