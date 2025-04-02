import tensorflow as tf
import numpy as np

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict_single_record(model, record, threshold):
    # Preprocess single record
    record = record.drop('date', axis=1)
    data = record.values.astype('float32')
    
    # Reshape data to match model's expected input shape (None, 12)
    data = data.reshape(-1, 12)  # Added this line
    
    # Get prediction
    prediction = model.predict(data, verbose=0)
    
    # Calculate MSE
    mse = np.mean(np.power(data - prediction, 2))
    
    # Check if anomaly
    is_anomaly = mse > threshold
    
    return {
        'mse': float(mse),
        'threshold': threshold,
        'is_anomaly': bool(is_anomaly)
    }