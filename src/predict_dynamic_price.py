from joblib import load
import pandas as pd

# Load the saved pipeline model
best_model = load('../models/best_model.joblib')

def predict_single(data, pipeline):
    """
    Function to make single predictions using the loaded pipeline model.

    Parameters:
        data (dict or DataFrame): Input data for prediction.
        pipeline (Pipeline): Loaded pipeline model.

    Returns:
        float: Predicted value.
    """
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    return round(pipeline.predict(data)[0], 2)