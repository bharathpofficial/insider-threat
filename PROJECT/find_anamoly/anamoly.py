import pandas as pd
from model_utils import load_model, predict_single_record

class AnomalyDetector:
    def __init__(self, model_path, threshold):
        # Load model once during initialization
        self.model = load_model(model_path)
        self.threshold = threshold
        self.detection_results = []

    def test_records(self, csv_path):
        # Read test records
        df = pd.read_csv(csv_path)
        
        # Test each record
        for idx, row in df.iterrows():
            # print(f"\nTesting Record {idx + 1}")
            
            # print("-" * 50)
            # print(f"Input Record:\n{row}")
            
            # Create single record DataFrame
            single_record = pd.DataFrame([row])
            
            # Get prediction using class model
            result = predict_single_record(self.model, single_record, self.threshold)
            
            # if result['is_anomaly']:
                
            #     is_threat = "THREAT DETECTED!"
                
            #     print(f"\nResults:")
            #     print(f"MSE: {result['mse']:.2f}")
            #     print(f"Threshold: {result['threshold']}")
            #     print(f"Status: {is_threat}")
            #     if result['is_anomaly']:
            #         print("⚠️ Warning: Suspicious activity detected!")
            #         print(f"Error value {result['mse']:.2f} exceeds threshold {result['threshold']}")
            #     print("-" * 50)
            # Store result with original data
            self.detection_results.append({
                "timestamp": pd.Timestamp.now().isoformat(),
                "is_anomaly": result['is_anomaly'],
                "mse": result['mse'],
                "original_data": row.to_dict()  # Include original row data
            })


