import numpy as np
import os
import joblib
import pandas as pd

import warnings
warnings.filterwarnings("ignore") 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "kmeans_model.pkl")
scale_path = os.path.join(BASE_DIR , "scale.pkl")

model = joblib.load(model_path)
scale = joblib.load(scale_path)
# print("Model loaded successfully!")

# Example data (ensure same preprocessing as training!)
sample_data = pd.DataFrame([[12.5, 300, 0, 1, 0, 0, 1, 0, 0, 0, 0]],
                           columns=['minutes_watched', 'clv_wins',
                                    'region_USA/Canada/As', 'region_West_EU',
                                    'channel_Friend', 'channel_Google',
                                    'channel_Instagram', 'channel_LinkedIn',
                                    'channel_Other', 'channel_Twitter',
                                    'channel_YouTube'])

sample_data.loc[:,["minutes_watched" , "clv_wins"]] = scale.transform(sample_data.loc[:,["minutes_watched" , "clv_wins"]])

# Predict
prediction = model.predict(sample_data)
print("Prediction:", prediction)
